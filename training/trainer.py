"""
Trainer for SIM-APA v2.0.
"""

from __future__ import annotations

import torch
from tqdm import tqdm


class Trainer:
    """
    Training and validation loop for SIM-APA.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader, epoch: int = 1) -> dict:
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_cosine = 0.0
        total_similarity = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Train Epoch {epoch}")

        for batch in progress:
            images = batch["image"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss_out = self.criterion(
                reconstructed=outputs["reconstructed"],
                target=outputs["features"].detach(),
            )

            loss_out["loss"].backward()
            self.optimizer.step()

            total_loss += loss_out["loss"].item()
            total_mse += loss_out["mse_loss"].item()
            total_cosine += loss_out["cosine_loss"].item()
            total_similarity += loss_out["cosine_similarity"].item()
            num_batches += 1

            progress.set_postfix(
                loss=loss_out["loss"].item(),
                cos=loss_out["cosine_similarity"].item(),
            )

        return {
            "loss": total_loss / num_batches,
            "mse_loss": total_mse / num_batches,
            "cosine_loss": total_cosine / num_batches,
            "cosine_similarity": total_similarity / num_batches,
        }

   
    def validate(self, dataloader, epoch: int = 1) -> dict:
        self.model.eval()
        torch.manual_seed(42)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        total_loss = 0.0
        total_mse = 0.0
        total_cosine = 0.0
        total_similarity = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        with torch.no_grad():
            for batch in progress:
                images = batch["image"].to(self.device)

                outputs = self.model(images)

                loss_out = self.criterion(
                reconstructed=outputs["reconstructed"],
                target=outputs["features"].detach(),
                )

                total_loss += loss_out["loss"].item()
                total_mse += loss_out["mse_loss"].item()
                total_cosine += loss_out["cosine_loss"].item()
                total_similarity += loss_out["cosine_similarity"].item()
                num_batches += 1

                progress.set_postfix(
                    val_loss=loss_out["loss"].item(),
                    val_cos=loss_out["cosine_similarity"].item(),
                )

        return {
            "val_loss": total_loss / num_batches,
            "val_mse_loss": total_mse / num_batches,
            "val_cosine_loss": total_cosine / num_batches,
            "val_cosine_similarity": total_similarity / num_batches,
        }
