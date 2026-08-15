# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Container dataclass for GPT chunk datasets (train, valid, and test)."""

from dataclasses import dataclass


@dataclass
class RetroGPTChunkDatasets:
    """Container dataclass for GPT chunk datasets."""


    train: dict = None
    valid: dict = None
    test: dict = None
