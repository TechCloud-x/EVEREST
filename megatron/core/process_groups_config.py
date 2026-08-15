# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Dataclasses for organizing model parallelism and gradient communication process groups."""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class ModelCommProcessGroups:
    """Process groups for transformer model parallelism.

    Fields use init=False and must be set after instance creation.

    Args:
        tp: Tensor parallel process group
        pp: Pipeline parallel process group
        mp: Model parallel group (tensor + pipeline)
        embd: Embedding process group
        pos_embd: Position embedding process group
        cp: Context parallel process group
        tp_cp: Tensor and context parallel group
        hcp: Hierarchical context parallel groups
        ep: Expert model parallel group
        expt_tp: Expert tensor parallel group
        tp_ep: Tensor and expert parallel group
        tp_ep_pp: Tensor, expert, and pipeline parallel group

    Example:
        # Create instance and set needed process groups
        model_pgs = ModelCommProcessGroups()
        model_pgs.tp = tp_group
        model_pgs.pp = pp_group

        # Pass to model components
        model = TransformerModel(..., process_groups=model_pgs)
    """


    tp: torch.distributed.ProcessGroup = field(init=False)


    pp: torch.distributed.ProcessGroup = field(init=False)


    mp: torch.distributed.ProcessGroup = field(init=False)


    embd: torch.distributed.ProcessGroup = field(init=False)


    pos_embd: torch.distributed.ProcessGroup = field(init=False)


    cp: torch.distributed.ProcessGroup = field(init=False)


    tp_cp: torch.distributed.ProcessGroup = field(init=False)


    hcp: List[torch.distributed.ProcessGroup] = field(init=False)


    ep: torch.distributed.ProcessGroup = field(init=False)


    expt_tp: torch.distributed.ProcessGroup = field(init=False)


    tp_ep: torch.distributed.ProcessGroup = field(init=False)


    tp_ep_pp: torch.distributed.ProcessGroup = field(init=False)


@dataclass
class GradCommProcessGroups:
    """Process groups for gradient communication in distributed training.

    Fields use init=False and must be set after instance creation.

    Args:
        dp: Data parallel process group
        dp_cp: Data and context parallel group
        expt_dp: Expert data parallel group
        intra_dp_cp: Intra partial data parallel group
        inter_dp_cp: Inter partial data parallel group

    Example:
        # Create instance and set needed process groups
        grad_pgs = GradCommProcessGroups()
        grad_pgs.dp = dp_group

        # Pass to distributed data parallel wrapper
        ddp_model = DistributedDataParallel(..., process_groups=grad_pgs)
    """


    dp: torch.distributed.ProcessGroup = field(init=False)


    dp_cp: torch.distributed.ProcessGroup = field(init=False)


    expt_dp: torch.distributed.ProcessGroup = field(init=False)


    intra_dp_cp: torch.distributed.ProcessGroup = field(init=False)


    inter_dp_cp: torch.distributed.ProcessGroup = field(init=False)
