"""
Base wireless channel interface for SIM-APA v2.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaseChannel(nn.Module):
    """
    Abstract base class for wireless channel models.
    """

    def __init__(self, snr_db: float = 10.0) -> None:
        super().__init__()
        self.snr_db = snr_db

    def compute_noise_std(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute AWGN noise standard deviation from SNR.
        """
        signal_power = x.pow(2).mean()
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        return torch.sqrt(noise_power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
