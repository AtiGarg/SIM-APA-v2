"""
Feature-aware fusion for SIM-APA v2.0.

This module fuses projected YOLO features with the semantic importance map
using a lightweight residual attention gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAwareFusion(nn.Module):
    """
    Fuse visual features and semantic importance maps.

    Parameters
    ----------
    feature_channels : int
        Number of channels in the input feature map.
    sim_channels : int
        Number of channels in the semantic importance map.
    hidden_channels : int
        Hidden channel width used for fusion.

    Inputs
    ------
    features : torch.Tensor
        Feature tensor of shape [B, 256, 20, 20].
    sim_map : torch.Tensor
        Semantic importance map of shape [B, 1, 20, 20] or [1, 20, 20].

    Returns
    -------
    torch.Tensor
        Fused feature tensor of shape [B, 256, 20, 20].
    """

    def __init__(
        self,
        feature_channels: int = 256,
        sim_channels: int = 1,
        hidden_channels: int = 256,
    ) -> None:
        super().__init__()

        self.feature_channels = feature_channels
        self.sim_channels = sim_channels
        self.hidden_channels = hidden_channels

        self.feature_embed = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.sim_embed = nn.Sequential(
            nn.Conv2d(sim_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(hidden_channels, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,
        sim_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply residual attention fusion.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor [B, C, H, W].
        sim_map : torch.Tensor
            Semantic map [B, 1, H, W] or [1, H, W].

        Returns
        -------
        torch.Tensor
            Fused feature tensor [B, C, H, W].
        """
        if sim_map.ndim == 3:
            sim_map = sim_map.unsqueeze(0)

        if sim_map.shape[0] == 1 and features.shape[0] > 1:
            sim_map = sim_map.repeat(features.shape[0], 1, 1, 1)

        assert features.ndim == 4, "features must have shape [B, C, H, W]"
        assert sim_map.ndim == 4, "sim_map must have shape [B, 1, H, W]"
        assert features.shape[0] == sim_map.shape[0], "batch sizes must match"
        assert features.shape[2:] == sim_map.shape[2:], "spatial sizes must match"

        feature_embedding = self.feature_embed(features)
        sim_embedding = self.sim_embed(sim_map)

        attention_mask = self.attention(feature_embedding + sim_embedding)

        fused_features = features * (1.0 + attention_mask)

        return fused_features
