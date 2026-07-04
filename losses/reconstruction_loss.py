"""
Loss functions for SIM-APA v2.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureReconstructionLoss(nn.Module):
    """
    Reconstruction loss between transmitted and recovered feature maps.

    Combines:
        1. MSE loss
        2. Cosine feature similarity loss
    """

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
        reconstructed: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """
        Compute reconstruction loss.

        Parameters
        ----------
        reconstructed : torch.Tensor
            Recovered feature tensor [B, C, H, W].
        target : torch.Tensor
            Target feature tensor [B, C, H, W].

        Returns
        -------
        dict
            loss, mse_loss, cosine_loss, cosine_similarity
        """
        mse_loss = F.mse_loss(reconstructed, target)

        reconstructed_flat = reconstructed.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)

        cosine_similarity = F.cosine_similarity(
            reconstructed_flat,
            target_flat,
            dim=1,
            eps=self.eps,
        ).mean()

        cosine_loss = 1.0 - cosine_similarity

        loss = self.mse_weight * mse_loss + self.cosine_weight * cosine_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "cosine_loss": cosine_loss,
            "cosine_similarity": cosine_similarity,
        }
