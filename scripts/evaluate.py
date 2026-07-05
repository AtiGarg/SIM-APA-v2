"""
Evaluation script for SIM-APA v2.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config
from utils.checkpoint import load_checkpoint
from datasets.kitti_dataset import KITTIDataset
from models.simapa_model import SIMAPAModel
from evaluation.evaluator import Evaluator


def main() -> None:
    cfg = Config(PROJECT_ROOT / "configs")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = KITTIDataset(
        root_dir=cfg.dataset.root_dir,
        image_folder=cfg.dataset.image_folder,
        image_size=cfg.dataset.image_size,
    )

    test_indices = list(range(7000, 7481))
    test_dataset = Subset(dataset, test_indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    model = SIMAPAModel(cfg).to(device)

    checkpoint_path = PROJECT_ROOT / "checkpoints" / "awgn" / "best.pth"

    load_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=model,
        optimizer=None,
        map_location=device,
    )

    evaluator = Evaluator(
        model=model,
        device=device,
    )

    metrics = evaluator.evaluate(test_loader)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
