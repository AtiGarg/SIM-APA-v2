"""
Residual DeepJSCC encoder and decoder for SIM-APA v2.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    """
    Residual convolution block for stable feature transformation.
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


class ChannelProjectionBlock(nn.Module):
    """
    Channel projection block with residual refinement.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.residual = ResidualConvBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.residual(x)
        return x


class DeepJSCCEncoder(nn.Module):
    """
    Residual DeepJSCC encoder.

    Compresses:
        256 -> 128 -> 64 -> 32
    """

    def __init__(self, input_channels: int = 256, latent_channels: int = 32) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            ChannelProjectionBlock(input_channels, 128),
            ChannelProjectionBlock(128, 64),
            ChannelProjectionBlock(64, latent_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DeepJSCCDecoder(nn.Module):
    """
    Residual DeepJSCC decoder.

    Reconstructs:
        32 -> 64 -> 128 -> 256
    """

    def __init__(self, latent_channels: int = 32, output_channels: int = 256) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            ChannelProjectionBlock(latent_channels, 64),
            ChannelProjectionBlock(64, 128),
            ChannelProjectionBlock(128, output_channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
