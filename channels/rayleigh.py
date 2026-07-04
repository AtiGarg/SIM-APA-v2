"""
Rayleigh fading channel model.
"""

from __future__ import annotations

import torch

from channels.base_channel import BaseChannel


class RayleighChannel(BaseChannel):
    """
    Rayleigh fading channel with AWGN.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_real = torch.randn_like(x) / torch.sqrt(torch.tensor(2.0, device=x.device))
        h_imag = torch.randn_like(x) / torch.sqrt(torch.tensor(2.0, device=x.device))

        h = torch.sqrt(h_real ** 2 + h_imag ** 2)

        faded = h * x

        noise_std = self.compute_noise_std(faded)
        noise = noise_std * torch.randn_like(faded)

        return faded + noise
