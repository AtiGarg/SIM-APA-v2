"""
Channel factory for SIM-APA v2.0.
"""

from __future__ import annotations

from channels.awgn import AWGNChannel
from channels.rayleigh import RayleighChannel
from channels.rician import RicianChannel


def build_channel(channel_type: str, snr_db: float = 10.0, k_factor: float = 5.0):
    """
    Build a wireless channel module.
    """
    channel_type = channel_type.lower()

    if channel_type == "awgn":
        return AWGNChannel(snr_db=snr_db)

    if channel_type == "rayleigh":
        return RayleighChannel(snr_db=snr_db)

    if channel_type == "rician":
        return RicianChannel(snr_db=snr_db, k_factor=k_factor)

    raise ValueError(f"Unsupported channel type: {channel_type}")
