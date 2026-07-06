"""
Detection distillation loss for SIM-APA v2.0.

This encourages detections from reconstructed features to match detections
from original YOLO features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionDistillationLoss(nn.Module):
    def __init__(
        self,
        box_weight: float = 1.0,
        conf_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.box_weight = box_weight
        self.conf_weight = conf_weight

    def forward(
        self,
        reconstructed_detections,
        original_detections,
    ) -> dict:
        if isinstance(reconstructed_detections, tuple):
            reconstructed = reconstructed_detections[0]
        else:
            reconstructed = reconstructed_detections

        if isinstance(original_detections, tuple):
            original = original_detections[0]
        else:
            original = original_detections

        original = original.detach()

        box_loss = F.mse_loss(
            reconstructed[..., :4],
            original[..., :4],
        )

        conf_loss = F.mse_loss(
            reconstructed[..., 4],
            original[..., 4],
        )

        loss = self.box_weight * box_loss + self.conf_weight * conf_loss

        return {
            "detection_loss": loss,
            "detection_box_loss": box_loss,
            "detection_conf_loss": conf_loss,
        }
