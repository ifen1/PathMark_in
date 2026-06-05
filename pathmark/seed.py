"""Reproducibility helpers — set every PRNG we touch."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Pin Python / NumPy / PyTorch RNGs to `seed`.

    Also disables non-deterministic cuDNN kernel selection. Calling this
    once at the start of `train.py` is enough to get bit-identical
    forward passes on the same hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
