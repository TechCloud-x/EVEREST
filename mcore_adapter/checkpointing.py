import os

import torch
from megatron.core import dist_checkpointing, mpu
from transformers.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    get_checkpoint_shard_files,
    load_state_dict,
)

from .constants import TRACKER_FILENAME
from .utils import get_logger


logger = get_logger(__name__)

"""
The following is modified based on Megatron-LM training/checkpointing.py
"""


def get_checkpoint_tracker_filename(checkpoints_path):
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, TRACKER_FILENAME)


def ensure_directory_exists(filename, check_parent=True):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename) if check_parent else filename
    os.makedirs(dirname, exist_ok=True)


def read_metadata(tracker_filename):


    iteration = 0
    release = False
    with open(tracker_filename, "r") as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == "release"
            if not release:
                raise ValueError("ERROR: Invalid metadata file {}. Exiting".format(tracker_filename))
    assert iteration > 0 or release, "error parsing metadata file {}".format(tracker_filename)






















    max_iter = iteration
    return max_iter, release


def get_checkpoint_dir(
    checkpoints_path,
    iteration=1,
    release=False,
    pipeline_parallel=None,
    tensor_rank=None,
    pipeline_rank=None,
    expert_parallel=None,
    expert_rank=None,
    return_base_dir=False,
):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    if return_base_dir:
        common_path = os.path.join(checkpoints_path, directory)
        return common_path


    if pipeline_parallel is None:
        pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    if expert_parallel is None:
        expert_parallel = mpu.get_expert_model_parallel_world_size() > 1
    if expert_rank is None:
        expert_rank = mpu.get_expert_model_parallel_rank()




    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, directory, f"mp_rank_{tensor_rank:02d}")
    else:
        common_path = os.path.join(checkpoints_path, directory, f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}")

    if expert_parallel:
        common_path = common_path + f"_{expert_rank:03d}"
    return common_path


def get_checkpoint_name(
    checkpoints_path,
    iteration=1,
    release=False,
    pipeline_parallel=None,
    tensor_rank=None,
    pipeline_rank=None,
    expert_parallel=None,
    expert_rank=None,
    return_base_dir=False,
):
    common_path = get_checkpoint_dir(
        checkpoints_path,
        iteration,
        release,
        pipeline_parallel,
        tensor_rank,
        pipeline_rank,
        expert_parallel,
        expert_rank,
        return_base_dir,
    )
    return os.path.join(common_path, "model_optim_rng.pt")


def find_checkpoint_rank_0(checkpoints_path, iteration, release=False):
    """Finds the checkpoint for rank 0 without knowing if we are using
    pipeline parallelism/expert parallelism or not.

    Since the checkpoint naming scheme changes if pipeline or expert
    parallelism is present, we need to look for both naming schemes if
    we don't know if the checkpoint has pipeline or expert parallelism.
    """


    filename = get_checkpoint_name(
        checkpoints_path,
        iteration,
        release,
        pipeline_parallel=False,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=False,
        expert_rank=0,
    )
    if os.path.isfile(filename):
        return filename


    filename = get_checkpoint_name(
        checkpoints_path,
        iteration,
        release,
        pipeline_parallel=False,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=True,
        expert_rank=0,
    )
    if os.path.isfile(filename):
        return filename


    filename = get_checkpoint_name(
        checkpoints_path,
        iteration,
        release,
        pipeline_parallel=True,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=False,
        expert_rank=0,
    )
    if os.path.isfile(filename):
        return filename


    filename = get_checkpoint_name(
        checkpoints_path,
        iteration,
        release,
        pipeline_parallel=True,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=True,
        expert_rank=0,
    )
    if os.path.isfile(filename):
        return filename


    filename = get_checkpoint_name(checkpoints_path, iteration, release, pipeline_parallel=True, return_base_dir=True)
    if dist_checkpointing.check_is_distributed_checkpoint(filename):
        return filename

    return None


def _load_base_checkpoint(
    load_dir, rank0=False, sharded_state_dict=None, exit_on_missing_checkpoint=True, checkpoint_step=None
):
    """Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """


    tracker_filename = get_checkpoint_tracker_filename(load_dir)


    if not os.path.isfile(tracker_filename):
        logger.warning(f"could not find the metadata file {tracker_filename}")


        if exit_on_missing_checkpoint:
            logger.error(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            raise RuntimeError("could not find the metadata file {tracker_filename}")

        return None, "", False



    if checkpoint_step is not None:
        iteration = checkpoint_step
        release = False
    else:
        iteration, release = read_metadata(tracker_filename)


    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        is_dist_ckpt = checkpoint_name is not None and dist_checkpointing.check_is_distributed_checkpoint(
            checkpoint_name
        )
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
        is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
        if not is_dist_ckpt:
            checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=False)
        dist_infix = "distributed " if is_dist_ckpt else ""
        if release:
            logger.info(f" loading release {dist_infix}checkpoint from {load_dir}")
        else:
            logger.info(f" loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}")

    state_dict = torch.load(checkpoint_name, map_location="cpu")
    return state_dict, checkpoint_name, release


def load_state_dict_from_checkpoint(checkpoint_dir):

    return _load_base_checkpoint(checkpoint_dir, exit_on_missing_checkpoint=False)[0]
