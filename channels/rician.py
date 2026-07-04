"""
Rician fading channel model.
"""

from __future__ import annotations

import torch

from channels.base_channel import BaseChannel


class RicianChannel(BaseChannel):
    """
    Rician fading channel with AWGN.
    """

    def __init__(self, snr_db: float = 10.0, k_factor: float = 5.0) -> None:
        super().__init__(snr_db=snr_db)
        self.k_factor = k_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.tensor(self.k_factor, device=x.device)

        los_component = torch.sqrt(k / (k + 1.0))
        scatter_std = torch.sqrt(1.0 / (2.0 * (k + 1.0)))

        h_real = los_component + scatter_std * torch.randn_like(x)
        h_imag = scatter_std * torch.randn_like(x)

        h = torch.sqrt(h_real ** 2 + h_imag ** 2)

        faded = h * x

        noise_std = self.compute_noise_std(faded)
        noise = noise_std * torch.randn_like(faded)

        return faded + noise
