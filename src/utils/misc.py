"""Miscellaneous utilities (seed setting, etc.).
Kept minimal — checkpoint and metrics functions moved to their own modules.
"""
from __future__ import annotations
import os
import random
import numpy as np
import torch
def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.
    Enables cuDNN benchmark for faster conv algorithms on fixed input sizes.
    For fully deterministic results, set ``torch.backends.cudnn.deterministic = True``
    and ``torch.backends.cudnn.benchmark = False`` — but this reduces performance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
