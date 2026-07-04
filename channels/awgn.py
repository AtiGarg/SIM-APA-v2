"""
AWGN channel model.
"""

from __future__ import annotations

import torch

from channels.base_channel import BaseChannel


class AWGNChannel(BaseChannel):
    """
    Additive White Gaussian Noise channel.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise_std = self.compute_noise_std(x)
        noise = noise_std * torch.randn_like(x)
        return x + noise
