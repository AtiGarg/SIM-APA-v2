"""
Feature refinement head for SIM-APA.

Refines reprojected YOLO layer-10 features before feeding them
to the frozen YOLO neck/head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualRefinementBlock(nn.Module):
    def __init__(self, channels: int = 512, groups: int = 32) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class FeatureRefinementHead(nn.Module):
    def __init__(
        self,
        channels: int = 512,
        hidden_channels: int = 512,
        groups: int = 32,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
        ]

        for _ in range(num_blocks):
            layers.append(
                ResidualRefinementBlock(
                    channels=hidden_channels,
                    groups=groups,
                )
            )

        layers.extend(
            [
                nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(groups, channels),
                nn.SiLU(inplace=True),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
