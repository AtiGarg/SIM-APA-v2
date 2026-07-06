"""
Training script for SIM-APA v2.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from datasets.kitti_dataset import KITTIDataset
from models.simapa_model import SIMAPAModel
from losses.reconstruction_loss import FeatureReconstructionLoss
from losses.reprojection_loss import ReprojectionLoss
from training.trainer import Trainer


def main() -> None:
    cfg = Config(PROJECT_ROOT / "configs")
    set_seed(cfg.project.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = KITTIDataset(
        root_dir=cfg.dataset.root_dir,
        image_folder=cfg.dataset.image_folder,
        image_size=cfg.dataset.image_size,
    )

    train_indices = list(range(0, 6000))
    val_indices = list(range(6000, 7000))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    model = SIMAPAModel(cfg).to(device)

    criterion = FeatureReconstructionLoss(
        mse_weight=1.0,
        cosine_weight=0.1,
    )

    reprojection_criterion = ReprojectionLoss(
        mse_weight=1.0,
        cosine_weight=0.1,
    )

    optimizer = torch.optim.AdamW(
        list(model.sim.parameters())
        + list(model.fusion.parameters())
        + list(model.apa.parameters())
        + list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.reprojection.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        reprojection_criterion=reprojection_criterion,
        reprojection_weight=0.1,
    )

    best_val_loss = float("inf")

    for epoch in range(1, 51):
        train_metrics = trainer.train_one_epoch(
            dataloader=train_loader,
            epoch=epoch,
        )

        val_metrics = trainer.validate(
            dataloader=val_loader,
            epoch=epoch,
        )

        metrics = {
            **train_metrics,
            **val_metrics,
        }

        print(f"Epoch {epoch} metrics:", metrics)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            checkpoint_dir=cfg.paths.checkpoints,
            filename="last.pth",
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]

            best_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                checkpoint_dir=cfg.paths.checkpoints,
                filename="best.pth",
            )

            print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
