"""
Semantic Importance Module for SIM-APA v2.0.

This module combines:
1. Adaptive importance scoring
2. Gaussian semantic projection

It produces the final 20x20 semantic importance map used by later modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.sim.importance import AdaptiveImportanceScorer
from models.sim.projection import GaussianProjection


class SemanticImportanceModule(nn.Module):
    """
    End-to-end Semantic Importance Module.

    Parameters
    ----------
    alpha : float
        Contribution of semantic weight.
    beta : float
        Contribution of detection confidence.
    lambda_refine : float
        Strength of adaptive refinement.
    image_size : int
        Input image size.
    feature_size : int
        Output semantic map size.

    Returns
    -------
    tuple
        importance_map : torch.Tensor
            Semantic importance map of shape [1, 20, 20].
        importance_info : dict
            Object-level importance details.
    """

    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 0.2,
        lambda_refine: float = 0.5,
        image_size: int = 640,
        feature_size: int = 20,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()

        self.scorer = AdaptiveImportanceScorer(
            alpha=alpha,
            beta=beta,
            lambda_refine=lambda_refine,
            image_size=image_size,
            hidden_dim=hidden_dim,
        )

        self.projector = GaussianProjection(
            image_size=image_size,
            feature_size=feature_size,
        )

    def forward(self, detections: dict) -> tuple[torch.Tensor, dict]:
        """
        Generate semantic importance map from detections.

        Parameters
        ----------
        detections : dict
            Dictionary containing boxes, classes, and confidences.

        Returns
        -------
        tuple
            importance_map, importance_info
        """
        importance_info = self.scorer(detections)
        importance_map = self.projector(importance_info)

        return importance_map, importance_info
