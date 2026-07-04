"""
Visualization utilities for the Semantic Importance Module.
"""

from __future__ import annotations

import torch
import matplotlib.pyplot as plt


def plot_importance_map(
    importance_map: torch.Tensor,
    title: str = "Semantic Importance Map",
) -> None:
    """
    Plot a semantic importance map.

    Parameters
    ----------
    importance_map : torch.Tensor
        Tensor of shape [1, H, W] or [H, W].
    title : str
        Plot title.
    """
    if importance_map.ndim == 3:
        importance_map = importance_map.squeeze(0)

    plt.figure(figsize=(5, 5))
    plt.imshow(importance_map.detach().cpu(), cmap="hot")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.show()
