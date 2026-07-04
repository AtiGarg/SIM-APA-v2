"""
Adaptive semantic weighting network for SIM-APA v2.0.

This module refines fixed semantic hierarchy weights using object-level
scene information such as confidence and bounding-box geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveWeightNetwork(nn.Module):
    """
    Small MLP for adaptive object importance refinement.

    Input features per object:
        1. base semantic weight
        2. detection confidence
        3. normalized object area
        4. aspect ratio

    Output:
        adaptive scale factor in [0, 1]
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 16,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive weighting factors.

        Parameters
        ----------
        object_features : torch.Tensor
            Tensor of shape [N, 4], where N is the number of detected objects.

        Returns
        -------
        torch.Tensor
            Adaptive factors of shape [N].
        """
        adaptive_factors = self.network(object_features)

        return adaptive_factors.squeeze(-1)
