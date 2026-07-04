"""
Trainer for SIM-APA v2.0.
"""

from __future__ import annotations

import torch
from tqdm import tqdm


class Trainer:
    """
    Training loop for SIM-APA.

    Parameters
    ----------
    model : torch.nn.Module
        Integrated SIM-APA model.
    criterion : torch.nn.Module
        Reconstruction loss.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : str
        Training device.
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
        """
        Train for one epoch.
        """
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_cosine = 0.0
        total_similarity = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}")

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
