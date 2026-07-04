"""
Adaptive importance score computation for SIM-APA v2.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.sim.hierarchy import get_class_weight
from models.sim.adaptive_weight import AdaptiveWeightNetwork


class AdaptiveImportanceScorer(nn.Module):
    """
    Computes adaptive semantic importance scores for detected objects.

    Each object receives an importance score using:

        adaptive_weight = base_weight * (1 + lambda_refine * adaptive_factor)

        importance = alpha * adaptive_weight + beta * confidence

    Parameters
    ----------
    alpha : float
        Contribution of semantic weight.
    beta : float
        Contribution of detection confidence.
    lambda_refine : float
        Strength of adaptive refinement.
    image_size : int
        Input image size used for normalized area computation.
    """

    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 0.2,
        lambda_refine: float = 0.5,
        image_size: int = 640,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.lambda_refine = lambda_refine
        self.image_size = image_size

        self.adaptive_network = AdaptiveWeightNetwork(
            input_dim=4,
            hidden_dim=hidden_dim,
            output_dim=1,
        )

    def _build_object_features(
        self,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        confidences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build object-level feature vectors.

        Features:
            1. base semantic weight
            2. detection confidence
            3. normalized object area
            4. aspect ratio
        """
        device = boxes.device

        base_weights = torch.tensor(
            [get_class_weight(int(cls.item())) for cls in classes],
            dtype=torch.float32,
            device=device,
        )

        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)

        areas = widths * heights
        normalized_areas = areas / float(self.image_size * self.image_size)

        aspect_ratios = widths / heights

        object_features = torch.stack(
            [
                base_weights,
                confidences.float(),
                normalized_areas.float(),
                aspect_ratios.float(),
            ],
            dim=1,
        )

        return object_features

    def forward(self, detections: dict) -> dict:
        """
        Compute adaptive importance scores.

        Parameters
        ----------
        detections : dict
            Dictionary containing:
                boxes: Tensor[N, 4]
                classes: Tensor[N]
                confidences: Tensor[N]

        Returns
        -------
        dict
            Dictionary containing boxes, classes, confidences, base_weights,
            adaptive_factors, adaptive_weights, and importance scores.
        """
        boxes = detections["boxes"]
        classes = detections["classes"]
        confidences = detections["confidences"]

        if boxes.numel() == 0:
            return {
                "boxes": boxes,
                "classes": classes,
                "confidences": confidences,
                "base_weights": torch.empty(0, device=boxes.device),
                "adaptive_factors": torch.empty(0, device=boxes.device),
                "adaptive_weights": torch.empty(0, device=boxes.device),
                "scores": torch.empty(0, device=boxes.device),
            }

        object_features = self._build_object_features(
            boxes=boxes,
            classes=classes,
            confidences=confidences,
        )

        base_weights = object_features[:, 0]

        adaptive_factors = self.adaptive_network(object_features)

        adaptive_weights = base_weights * (
            1.0 + self.lambda_refine * adaptive_factors
        )

        scores = self.alpha * adaptive_weights + self.beta * confidences.float()

        return {
            "boxes": boxes,
            "classes": classes,
            "confidences": confidences,
            "base_weights": base_weights,
            "adaptive_factors": adaptive_factors,
            "adaptive_weights": adaptive_weights,
            "scores": scores,
        }
