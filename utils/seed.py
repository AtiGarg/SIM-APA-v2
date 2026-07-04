"""
Random seed utilities for SIM-APA v2.0.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducible experiments.

    Parameters
    ----------
    seed : int
        Seed value used for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
