"""
Checkpoint utilities for SIM-APA v2.0.
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: str,
    filename: str = "latest.pth",
) -> Path:
    """
    Save model and optimizer state.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / filename

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    map_location: str = "cpu",
) -> dict:
    """
    Load model and optionally optimizer state.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
