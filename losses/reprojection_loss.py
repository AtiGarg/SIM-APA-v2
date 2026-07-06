"""
Reprojection loss for SIM-APA v2.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReprojectionLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.eps = eps

    def forward(
        self,
        reprojected: torch.Tensor,
        raw_feature: torch.Tensor,
    ) -> dict:
        mse_loss = F.mse_loss(reprojected, raw_feature)

        reprojected_flat = reprojected.flatten(start_dim=1)
        raw_flat = raw_feature.flatten(start_dim=1)

        cosine_similarity = F.cosine_similarity(
            reprojected_flat,
            raw_flat,
            dim=1,
            eps=self.eps,
        ).mean()

        cosine_loss = 1.0 - cosine_similarity

        loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss

        return {
            "reprojection_loss": loss,
            "reprojection_mse": mse_loss,
            "reprojection_cosine_loss": cosine_loss,
            "reprojection_cosine_similarity": cosine_similarity,
        }
