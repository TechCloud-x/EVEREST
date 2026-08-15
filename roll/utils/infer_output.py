import os
import re
from typing import Dict


_TRAIN_TIME_COMPONENT = re.compile(r"(?:^|[\\/])(\d{2}_\d{2}_\d{2}_\d{2})(?=[\\/]|$)")


def extract_train_time(pretrain: str) -> str:
    """Derive a stable run identifier from a checkpoint path or model identifier."""
    match = _TRAIN_TIME_COMPONENT.search(str(pretrain))
    if match is not None:
        return match.group(1)
    candidate = os.path.basename(os.path.normpath(str(pretrain))) or "checkpoint"
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-._")
    return candidate[:64] or "checkpoint"


def build_infer_output_paths(pretrain: str, base_output_dir: str) -> Dict[str, str]:
    """Build every inference artifact directory under a checkpoint-specific run."""
    train_time = extract_train_time(pretrain)
    run_name = f"infer_{train_time}"
    normalized_base = os.path.normpath(base_output_dir)

    if os.path.basename(normalized_base) == run_name:
        run_dir = normalized_base
    else:
        run_dir = os.path.join(normalized_base, run_name)

    return {
        "train_time": train_time,
        "output_dir": run_dir,
        "logging_dir": os.path.join(run_dir, "logs"),
        "checkpoint_dir": os.path.join(run_dir, "checkpoint"),
        "tensorboard_dir": os.path.join(run_dir, "tensorboard"),
        "profiler_dir": os.path.join(run_dir, "profiler"),
    }
