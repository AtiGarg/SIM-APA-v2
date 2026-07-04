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
from datasets.kitti_dataset import KITTIDataset
from models.simapa_model import SIMAPAModel
from losses.reconstruction_loss import FeatureReconstructionLoss
from training.trainer import Trainer
from utils.checkpoint import save_checkpoint


def main() -> None:
    cfg = Config(PROJECT_ROOT / "configs")
    set_seed(cfg.project.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = KITTIDataset(
        root_dir=cfg.dataset.root_dir,
        image_folder=cfg.dataset.image_folder,
        image_size=cfg.dataset.image_size,
    )

    # Small subset for debugging.
    # Later we will remove this and use the full dataset.
    train_subset = Subset(dataset, list(range(64)))

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    model = SIMAPAModel(cfg).to(device)

    criterion = FeatureReconstructionLoss(
        mse_weight=1.0,
        cosine_weight=0.1,
    )

    optimizer = torch.optim.AdamW(
        list(model.sim.parameters())
        + list(model.fusion.parameters())
        + list(model.apa.parameters())
        + list(model.encoder.parameters())
        + list(model.decoder.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    for epoch in range(1, 3):
        metrics = trainer.train_one_epoch(
            dataloader=train_loader,
            epoch=epoch,
        )

        print(f"Epoch {epoch} metrics:", metrics)

        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            checkpoint_dir=cfg.paths.checkpoints,
            filename=f"epoch_{epoch}.pth",
        )

        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
