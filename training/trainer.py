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
        reprojection_criterion=None,
        reprojection_weight: float = 0.1,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.reprojection_criterion = reprojection_criterion
        self.reprojection_weight = reprojection_weight

    def train_one_epoch(self, dataloader, epoch: int = 1) -> dict:
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_cosine = 0.0
        total_similarity = 0.0
        total_reprojection = 0.0
        total_reprojection_similarity = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Train Epoch {epoch}")

        for batch in progress:
            images = batch["image"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            recon_out = self.criterion(
                reconstructed=outputs["reconstructed"],
                target=outputs["features"].detach(),
            )

            loss = recon_out["loss"]

            reproj_loss_value = torch.tensor(0.0, device=self.device)
            reproj_similarity_value = torch.tensor(0.0, device=self.device)

            if self.reprojection_criterion is not None:
                reproj_out = self.reprojection_criterion(
                    reprojected=outputs["reprojected"],
                    raw_feature=outputs["raw_feature"].detach(),
                )

                reproj_loss_value = reproj_out["reprojection_loss"]
                reproj_similarity_value = reproj_out["reprojection_cosine_similarity"]

                loss = loss + self.reprojection_weight * reproj_loss_value

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_mse += recon_out["mse_loss"].item()
            total_cosine += recon_out["cosine_loss"].item()
            total_similarity += recon_out["cosine_similarity"].item()
            total_reprojection += reproj_loss_value.item()
            total_reprojection_similarity += reproj_similarity_value.item()
            num_batches += 1

            progress.set_postfix(
                loss=loss.item(),
                cos=recon_out["cosine_similarity"].item(),
                reproj=reproj_similarity_value.item(),
            )

        return {
            "loss": total_loss / num_batches,
            "mse_loss": total_mse / num_batches,
            "cosine_loss": total_cosine / num_batches,
            "cosine_similarity": total_similarity / num_batches,
            "reprojection_loss": total_reprojection / num_batches,
            "reprojection_cosine_similarity": total_reprojection_similarity / num_batches,
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
        total_reprojection = 0.0
        total_reprojection_similarity = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Val Epoch {epoch}")

        with torch.no_grad():
            for batch in progress:
                images = batch["image"].to(self.device)

                outputs = self.model(images)

                recon_out = self.criterion(
                    reconstructed=outputs["reconstructed"],
                    target=outputs["features"].detach(),
                )

                loss = recon_out["loss"]

                reproj_loss_value = torch.tensor(0.0, device=self.device)
                reproj_similarity_value = torch.tensor(0.0, device=self.device)

                if self.reprojection_criterion is not None:
                    reproj_out = self.reprojection_criterion(
                        reprojected=outputs["reprojected"],
                        raw_feature=outputs["raw_feature"].detach(),
                    )

                    reproj_loss_value = reproj_out["reprojection_loss"]
                    reproj_similarity_value = reproj_out["reprojection_cosine_similarity"]

                    loss = loss + self.reprojection_weight * reproj_loss_value

                total_loss += loss.item()
                total_mse += recon_out["mse_loss"].item()
                total_cosine += recon_out["cosine_loss"].item()
                total_similarity += recon_out["cosine_similarity"].item()
                total_reprojection += reproj_loss_value.item()
                total_reprojection_similarity += reproj_similarity_value.item()
                num_batches += 1

                progress.set_postfix(
                    val_loss=loss.item(),
                    val_cos=recon_out["cosine_similarity"].item(),
                    val_reproj=reproj_similarity_value.item(),
                )

        return {
            "val_loss": total_loss / num_batches,
            "val_mse_loss": total_mse / num_batches,
            "val_cosine_loss": total_cosine / num_batches,
            "val_cosine_similarity": total_similarity / num_batches,
            "val_reprojection_loss": total_reprojection / num_batches,
            "val_reprojection_cosine_similarity": total_reprojection_similarity / num_batches,
        }
