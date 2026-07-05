"""
Evaluation metrics for SIM-APA v2.0.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def compute_mse(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(reconstructed, target)


def compute_cosine_similarity(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reconstructed_flat = reconstructed.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)

    return F.cosine_similarity(
        reconstructed_flat,
        target_flat,
        dim=1,
    ).mean()


def compute_psnr(mse: torch.Tensor, max_value: float = 1.0) -> float:
    mse_value = mse.item()

    if mse_value == 0:
        return float("inf")

    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse_value)


def compute_power(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.pow(2).mean()


def compute_compression_ratio(
    input_channels: int,
    latent_channels: int,
) -> float:
    return input_channels / latent_channels
