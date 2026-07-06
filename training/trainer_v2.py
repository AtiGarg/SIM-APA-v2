"""
Trainer V2 for SIM-APA.

Adds:
1. Reconstruction loss
2. Reprojection loss
3. Detection distillation loss
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from evaluation.yolo_head_forward import forward_from_layer10


class TrainerV2:
    def __init__(
        self,
        model,
        reconstruction_criterion,
        reprojection_criterion,
        detection_criterion,
        optimizer,
        device: str = "cuda",
        reconstruction_weight: float = 1.0,
        reprojection_weight: float = 0.1,
        detection_weight: float = 0.01,
    ) -> None:
        self.model = model
        self.reconstruction_criterion = reconstruction_criterion
        self.reprojection_criterion = reprojection_criterion
        self.detection_criterion = detection_criterion
        self.optimizer = optimizer
        self.device = device

        self.reconstruction_weight = reconstruction_weight
        self.reprojection_weight = reprojection_weight
        self.detection_weight = detection_weight

    def train_one_epoch(self, dataloader, epoch: int = 1) -> dict:
        self.model.train()
        self.model.backbone.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_reproj = 0.0
        total_detect = 0.0
        total_cos = 0.0
        total_reproj_cos = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"TrainV2 Epoch {epoch}")

        for batch in progress:
            images = batch["image"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            recon_out = self.reconstruction_criterion(
                reconstructed=outputs["reconstructed"],
                target=outputs["features"].detach(),
            )

            reproj_out = self.reprojection_criterion(
                reprojected=outputs["reprojected"],
                raw_feature=outputs["raw_feature"].detach(),
            )

            with torch.no_grad():
                original_detections = self.model.backbone.model(images)

            reconstructed_detections = forward_from_layer10(
                yolo_model=self.model.backbone.model,
                images=images,
                layer10_feature=outputs["reprojected"],
                enable_grad=True,
            )

            detection_out = self.detection_criterion(
                reconstructed_detections=reconstructed_detections,
                original_detections=original_detections,
            )

            loss = (
                self.reconstruction_weight * recon_out["loss"]
                + self.reprojection_weight * reproj_out["reprojection_loss"]
                + self.detection_weight * detection_out["detection_loss"]
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_out["loss"].item()
            total_reproj += reproj_out["reprojection_loss"].item()
            total_detect += detection_out["detection_loss"].item()
            total_cos += recon_out["cosine_similarity"].item()
            total_reproj_cos += reproj_out["reprojection_cosine_similarity"].item()
            num_batches += 1

            progress.set_postfix(
                loss=loss.item(),
                cos=recon_out["cosine_similarity"].item(),
                reproj=reproj_out["reprojection_cosine_similarity"].item(),
                det=detection_out["detection_loss"].item(),
            )

        return {
            "loss": total_loss / num_batches,
            "reconstruction_loss": total_recon / num_batches,
            "reprojection_loss": total_reproj / num_batches,
            "detection_loss": total_detect / num_batches,
            "cosine_similarity": total_cos / num_batches,
            "reprojection_cosine_similarity": total_reproj_cos / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader, epoch: int = 1) -> dict:
        self.model.eval()

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        total_loss = 0.0
        total_recon = 0.0
        total_reproj = 0.0
        total_detect = 0.0
        total_cos = 0.0
        total_reproj_cos = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"ValV2 Epoch {epoch}")

        for batch in progress:
            images = batch["image"].to(self.device)

            outputs = self.model(images)

            recon_out = self.reconstruction_criterion(
                reconstructed=outputs["reconstructed"],
                target=outputs["features"].detach(),
            )

            reproj_out = self.reprojection_criterion(
                reprojected=outputs["reprojected"],
                raw_feature=outputs["raw_feature"].detach(),
            )

            original_detections = self.model.backbone.model(images)

            reconstructed_detections = forward_from_layer10(
                yolo_model=self.model.backbone.model,
                images=images,
                layer10_feature=outputs["reprojected"],
                enable_grad=False,
            )

            detection_out = self.detection_criterion(
                reconstructed_detections=reconstructed_detections,
                original_detections=original_detections,
            )

            loss = (
                self.reconstruction_weight * recon_out["loss"]
                + self.reprojection_weight * reproj_out["reprojection_loss"]
                + self.detection_weight * detection_out["detection_loss"]
            )

            total_loss += loss.item()
            total_recon += recon_out["loss"].item()
            total_reproj += reproj_out["reprojection_loss"].item()
            total_detect += detection_out["detection_loss"].item()
            total_cos += recon_out["cosine_similarity"].item()
            total_reproj_cos += reproj_out["reprojection_cosine_similarity"].item()
            num_batches += 1

            progress.set_postfix(
                val_loss=loss.item(),
                val_reproj=reproj_out["reprojection_cosine_similarity"].item(),
                val_det=detection_out["detection_loss"].item(),
            )

        return {
            "val_loss": total_loss / num_batches,
            "val_reconstruction_loss": total_recon / num_batches,
            "val_reprojection_loss": total_reproj / num_batches,
            "val_detection_loss": total_detect / num_batches,
            "val_cosine_similarity": total_cos / num_batches,
            "val_reprojection_cosine_similarity": total_reproj_cos / num_batches,
        }
