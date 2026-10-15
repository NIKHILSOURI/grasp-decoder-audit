"""Seeding helpers: same config and seed give identical numbers."""

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def rng(seed: int, *salt: int) -> np.random.Generator:
    """Independent, reproducible generator for a (seed, salt...) combination."""
    return np.random.default_rng([seed, *salt])
