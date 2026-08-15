# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Iterator, List, Union

import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
)


Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )


    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)


    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def set_current_microbatch(model, microbatch_id):
    """Set the current microbatch."""
    decoder_exists = True
    decoder = None
    try:
        decoder = get_attr_wrapped_model(model, "decoder")
    except RuntimeError:
        decoder_exists = False
    if decoder_exists and decoder is not None:
        for layer in decoder.layers:
            layer.current_microbatch = microbatch_id


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)
    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor *= parallel_state.get_context_parallel_world_size()
                    output_tensor /= num_microbatches
            else:

                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor *= parallel_state.get_context_parallel_world_size()
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()




    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:

        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )

        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)


    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:

        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )

        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)



    model_type = get_model_type(model)
    if (
        model_type == ModelType.encoder_and_decoder
        and encoder_decoder_xattn
        and parallel_state.is_inside_decoder()
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""





    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()


    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]


    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])






    if output_tensor[0].requires_grad:
        if config.deallocate_pipeline_outputs:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])


    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)



    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and model_type == ModelType.encoder_and_decoder
        and len(output_tensor_grad) > 1
    ):
        if output_tensor_grad[1] is not None:
            assert input_tensor_grad[-1] is not None
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    """Check if it is the first validation step."""
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
            )
            total_num_tokens += num_tokens
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)



    output_tensor, num_tokens = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
        current_microbatch=num_microbatches - 1,
    )
    total_num_tokens += num_tokens

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    if config.finalize_model_grads_func is not None and not forward_only:


        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store


def clear_embedding_activation_buffer(config, model):
    """Clear embedding activation buffer."""

    if (
        parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        and config.defer_embedding_wgrad_compute
    ):
        if isinstance(model, list):
            embedding_module = get_attr_wrapped_model(
                model[-1], 'post_process', return_model_obj=True
            )
        else:
            embedding_module = get_attr_wrapped_model(model, 'post_process', return_model_obj=True)


        embedding_module.embedding_activation_buffer.clear()

        return embedding_module
    else:
        return None


def finish_embedding_wgrad_compute(config, embedding_module):
    """Finish embedding wgrad compute."""
    if (
        parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        and config.defer_embedding_wgrad_compute
    ):
        embedding_activation_buffer = embedding_module.embedding_activation_buffer
        grad_output_buffer = embedding_module.grad_output_buffer
        weight = (
            embedding_module.output_layer.weight
            if embedding_module.share_embeddings_and_output_weights
            else embedding_module.shared_embedding_or_output_weight()
        )

        drain_embedding_wgrad_compute(
            config, embedding_activation_buffer, grad_output_buffer, weight
        )


def get_pp_rank_microbatches(
    num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage, forward_only=False
):
    """Get the number of total, warmup, and remaining microbatches in PP scheduling."""
    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    virtual_pipeline_parallel_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    total_num_microbatches = num_microbatches * num_model_chunks
    are_all_microbatches_in_warmup = False

    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    elif pipeline_parallel_size > 1:
        if virtual_pipeline_parallel_size is None:

            num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
        else:





            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
    else:

        num_warmup_microbatches = 1

    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches
        are_all_microbatches_in_warmup = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    return (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    )


def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """Get the schedule table for PP scheduling."""
    schedule_table = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:

            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:

            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
                    )
                ]
            )
    return schedule_table


def convert_schedule_table_to_order(num_warmup_microbatches, num_model_chunks, schedule_table):
    """Convert a tunable schedule lookup table to the te.make_graphed_callables() accepted
    order format. For example, the tunable schedule table for PP2 N3M5 with VP2 is as below:
    virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    model_chunk_id        | 0 0 0 1 1 1 0 0 1 1

    Then the forward backward separated order is:
    forward               | 1 1 1 2 2 2 1 1 2 2
    backward              | -2 -2 -2 -1 -1 -1 -2 -2 -1 -1

    If num_warmup_microbatches is 5, the output order is:
    1 1 1 2 2 2 -2 1 -2 1 -2 2 -1 2 -1 -1 -2 -2 -1 -1
    """
    _, model_chunk_id_table = zip(*schedule_table)
    forward_order = [chunk_id + 1 for chunk_id in model_chunk_id_table]
    backward_order = [chunk_id - num_model_chunks for chunk_id in model_chunk_id_table]
    order = forward_order[:num_warmup_microbatches]
    for i in range(num_warmup_microbatches, len(forward_order)):
        order.append(forward_order[i])
        order.append(backward_order[i - num_warmup_microbatches])
    if num_warmup_microbatches > 0:
        order.extend(backward_order[-num_warmup_microbatches:])
    return order


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""










    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")


    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)


    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]



    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()


    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    forward_data_store = []
    output_tensor_grads = None
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
    else:
        output_tensor_grads = None

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if (
        config.microbatch_group_size_per_vp_stage > num_microbatches
        or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
    ):
        msg = (
            'The number of contiguous micro-batches in a virtual pipeline stage'
            f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
        )
        raise ValueError(msg)



    final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
    if 0 < final_microbatch_group_size < pipeline_parallel_size:
        msg = 'The remainder of M (the total micro-batches) divided by N (number of '
        msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
        msg += 'or larger than or equal to the pipeline-parallel size, but it is '
        msg += f'{final_microbatch_group_size}. '
        msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
        msg += 'and reduces throughput.'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])

    if model_type == ModelType.encoder_and_decoder:
        xattn_needed = get_model_xattn(model)
        assert (
            not xattn_needed
        ), "Interleaving is not supported when xattn is required between encoder and decoder"
        tensor_shape = get_tensor_shapes(
            rank=parallel_state.get_pipeline_model_parallel_rank(),
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
            encoder_decoder_xattn=xattn_needed,
        )
        tensor_shape = list(tensor_shape[0])
    else:
        tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
        if config.sequence_parallel:
            tensor_shape[0] = (
                tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
            )


    num_model_chunks = len(model)
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches, num_model_chunks, config.microbatch_group_size_per_vp_stage, forward_only
    )









    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1


    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())








    schedule_table = get_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
    )









    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:

            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True

        is_leading_pipeline_stage = (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            if forward
            else parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:







            if virtual_microbatch_id < (pipeline_parallel_size - 1):

                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            else:



                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )

            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:

                next_model_chunk_id += 1
            else:

                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id

    def forward_step_helper(
        virtual_microbatch_id, microbatch_id, checkpoint_activations_microbatch
    ):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)






        if config.param_sync_func is not None:
            param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
            if (
                param_sync_virtual_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
            ):
                param_sync_chunk_id = (
                    get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                )
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )


        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)






        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
        )

        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens


        if forward_only:

            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(virtual_microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)


        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            virtual_microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)


        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )






        if config.grad_sync_func is not None:
            grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
            if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_virtual_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(
                    grad_sync_virtual_microbatch_id, forward=False
                )
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad


    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        fwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
        bwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(cur_model_chunk_id)

        if config.overlap_p2p_comm_warmup_flush:
            if not parallel_state.is_pipeline_first_stage() and k != 0:
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()


        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)


        if k == (total_num_microbatches - 1):
            recv_prev = False


        if config.overlap_p2p_comm_warmup_flush and not parallel_state.is_pipeline_first_stage(
            ignore_virtual=True
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communication.send_forward_recv_forward(
                    output_tensor=None,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))


        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        microbatch_id = get_microbatch_id_in_model_chunk(k, forward=True)
        output_tensor = forward_step_helper(k, microbatch_id, checkpoint_activations_microbatch)


        if parallel_state.is_pipeline_last_stage():
            output_tensor = None



        if not config.overlap_p2p_comm_warmup_flush:
            if (
                k == (num_warmup_microbatches - 1)
                and not config.overlap_p2p_comm
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):

                _, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=False,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            else:
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communication.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    output_tensor_grads[num_model_chunks - 1].append(bwd_recv_buffer[-1])


    for k in range(num_microbatches_remaining):

        forward_k = k + num_warmup_microbatches


        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(cur_model_chunk_id)
        microbatch_id = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        if config.overlap_p2p_comm:
            if not parallel_state.is_pipeline_first_stage():
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_prev_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                        'should have registered recv handle'
                    )
                    recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                    recv_prev_wait_handle.wait()
                else:
                    if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            output_tensor = forward_step_helper(
                forward_k, microbatch_id, checkpoint_activations_microbatch
            )



            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)


            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )



            if k == (num_microbatches_remaining - 1):
                recv_prev = False



            fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                p2p_communication.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))



            backward_k = k
            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if not parallel_state.is_pipeline_last_stage():
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            input_tensor_grad = backward_step_helper(backward_k)


            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )
            )
            if send_prev_wait_handle is not None:
                send_prev_wait_handle.wait()
            if bwd_wait_handles is not None:
                send_prev_wait_handle = (
                    bwd_wait_handles.pop("send_prev") if "send_prev" in bwd_wait_handles else None
                )
                if "recv_next" in bwd_wait_handles:
                    recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))



            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(
                    bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                )
                bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
        else:
            output_tensor = forward_step_helper(
                forward_k, microbatch_id, checkpoint_activations_microbatch
            )


            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)






            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )



            if k == (num_microbatches_remaining - 1):
                recv_prev = False


            (input_tensor, output_tensor_grad) = (
                p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)



            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)


    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if are_all_microbatches_in_warmup:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(tensor_shape, config=config)
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(cur_model_chunk_id)
            if not parallel_state.is_pipeline_last_stage() and k != 0:
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False


            if config.overlap_p2p_comm_warmup_flush and not parallel_state.is_pipeline_last_stage(
                ignore_virtual=True
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad=None,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            input_tensor_grad = backward_step_helper(k)


            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            if config.overlap_p2p_comm_warmup_flush:
                if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    _, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communication.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            config=config,
                            overlap_p2p_comm=True,
                        )
                    )

                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config
                )

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()


        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)

    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

    if config.finalize_model_grads_func is not None and not forward_only:



        finish_embedding_wgrad_compute(config, embedding_module)




        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )


    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store


def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    encoder_decoder_xattn: bool,
):
    """
    Determine right tensor sizes (based on position of rank with respect to split rank) and
    model size.
    Send two tensors if model decoder requires the encoder's output (via cross-attention) and
    rank is in decoder stage.
    First tensor is decoder. Second tensor is encoder.
    If model has an encoder & decoder and rank is at the boundary, send one tensor.
    Otherwise, send one tensor.
    """
    tensor_shapes = []

    seq_length = seq_length // parallel_state.get_context_parallel_world_size()
    if model_type == ModelType.encoder_and_decoder:
        decoder_seq_length = decoder_seq_length // parallel_state.get_context_parallel_world_size()

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_inside_encoder(rank) and not parallel_state.is_inside_decoder(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        elif encoder_decoder_xattn:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes, config):
    """Wrapper for p2p_communication.recv_forward used with non-interleaving schedule."""
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))
    return input_tensors


def recv_backward(tensor_shapes, config):
    """Wrapper for p2p_communication.recv_backward used with non-interleaving schedule."""
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, config))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, config):
    """Wrapper for p2p_communication.send_forward used with non-interleaving schedule."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, config)


def send_backward(input_tensor_grads, tensor_shapes, config):
    """Wrapper for p2p_communication.send_backward used with non-interleaving schedule."""
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, config)


def send_forward_recv_backward(output_tensors, tensor_shapes, config):
    """Wrapper for p2p_communication.send_forward_recv_backward used
    with non-interleaving schedule."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, config
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config):
    """Wrapper for p2p_communication.send_backward_recv_forward used
    with non-interleaving schedule."""
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, config
        )
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )


    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)


    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()


    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches









    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )


    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []


    for i in range(num_warmup_microbatches):

        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        send_forward(output_tensor, send_tensor_shapes, config)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)




    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config)


    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)


        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        total_num_tokens += num_tokens

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )


            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)



            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)



            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )


    if not forward_only:
        for i in range(num_warmup_microbatches):






            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config)


        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:



        finish_embedding_wgrad_compute(config, embedding_module)




        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store
