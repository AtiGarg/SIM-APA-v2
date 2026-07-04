"""
DeepJSCC encoder and decoder for SIM-APA v2.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepJSCCEncoder(nn.Module):
    """
    DeepJSCC encoder.

    Compresses feature map:
        256 -> 128 -> 64 -> 32
    """

    def __init__(self, input_channels: int = 256, latent_channels: int = 32) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DeepJSCCDecoder(nn.Module):
    """
    DeepJSCC decoder.

    Reconstructs feature map:
        32 -> 64 -> 128 -> 256
    """

    def __init__(self, latent_channels: int = 32, output_channels: int = 256) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
