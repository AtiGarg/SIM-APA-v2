"""
Gaussian semantic projection for SIM-APA v2.0.

This module converts object-level importance scores into a spatial
semantic importance map aligned with the YOLO P5 feature map resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianProjection(nn.Module):
    """
    Projects object importance scores onto a 2D feature grid using Gaussians.

    Parameters
    ----------
    image_size : int
        Size of the resized input image.
    feature_size : int
        Spatial size of the feature map.
    min_sigma : float
        Minimum Gaussian spread for numerical stability.
    """

    def __init__(
        self,
        image_size: int = 640,
        feature_size: int = 20,
        min_sigma: float = 1.0,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.feature_size = feature_size
        self.min_sigma = min_sigma

    def forward(self, importance: dict) -> torch.Tensor:
        """
        Generate a semantic importance map.

        Parameters
        ----------
        importance : dict
            Dictionary containing boxes and scores.

        Returns
        -------
        torch.Tensor
            Importance map of shape [1, feature_size, feature_size].
        """
        boxes = importance["boxes"]
        scores = importance["scores"]

        device = boxes.device
        importance_map = torch.zeros(
            1,
            self.feature_size,
            self.feature_size,
            device=device,
        )

        if boxes.numel() == 0:
            return importance_map

        scale = self.feature_size / self.image_size

        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.feature_size, device=device),
            torch.arange(self.feature_size, device=device),
            indexing="ij",
        )

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box

            cx = ((x1 + x2) / 2.0) * scale
            cy = ((y1 + y2) / 2.0) * scale

            width = (x2 - x1).clamp(min=1.0) * scale
            height = (y2 - y1).clamp(min=1.0) * scale

            sigma_x = torch.clamp(width / 3.0, min=self.min_sigma)
            sigma_y = torch.clamp(height / 3.0, min=self.min_sigma)

            gaussian = torch.exp(
                -(
                    ((grid_x - cx) ** 2) / (2.0 * sigma_x ** 2)
                    + ((grid_y - cy) ** 2) / (2.0 * sigma_y ** 2)
                )
            )

            importance_map[0] += score * gaussian

        max_value = importance_map.max()

        if max_value > 0:
            importance_map = importance_map / max_value

        return importance_map
