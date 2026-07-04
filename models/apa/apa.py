"""
Adaptive Power Allocation for SIM-APA v2.0.

This module predicts a spatial power map and applies it to fused semantic
features before DeepJSCC encoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Lightweight residual convolution block.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class AdaptivePowerAllocation(nn.Module):
    """
    Learnable spatial Adaptive Power Allocation module.

    Input
    -----
    fused_features : torch.Tensor
        Tensor of shape [B, 256, 20, 20].

    Output
    ------
    dict
        power_map : torch.Tensor
            Normalized spatial power map [B, 1, 20, 20].
        weighted_features : torch.Tensor
            Power-weighted features [B, 256, 20, 20].
        average_power : torch.Tensor
            Average power value.
        max_power : torch.Tensor
            Maximum power value.
    """

    def __init__(
        self,
        input_channels: int = 256,
        hidden_channels_1: int = 128,
        hidden_channels_2: int = 64,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.eps = eps

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels_1, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels_1),
            nn.ReLU(inplace=True),

            ResidualBlock(hidden_channels_1),

            nn.Conv2d(hidden_channels_1, hidden_channels_2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels_2),
            nn.ReLU(inplace=True),

            ResidualBlock(hidden_channels_2),

            nn.Conv2d(hidden_channels_2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def normalize_power(self, power_map: torch.Tensor) -> torch.Tensor:
        """
        Normalize power map to unit average power per sample.
        """
        mean_power = power_map.mean(dim=(1, 2, 3), keepdim=True)
        normalized_power = power_map / (mean_power + self.eps)

        return normalized_power

    def forward(self, fused_features: torch.Tensor) -> dict:
        """
        Predict and apply spatial power allocation.

        Parameters
        ----------
        fused_features : torch.Tensor
            Input tensor [B, 256, 20, 20].

        Returns
        -------
        dict
            APA outputs.
        """
        assert fused_features.ndim == 4, "fused_features must be [B, C, H, W]"

        raw_power_map = self.net(fused_features)
        power_map = self.normalize_power(raw_power_map)

        weighted_features = fused_features * power_map

        return {
            "power_map": power_map,
            "weighted_features": weighted_features,
            "average_power": power_map.mean(),
            "max_power": power_map.max(),
        }
