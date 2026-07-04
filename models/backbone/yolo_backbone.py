"""
YOLOv10 backbone wrapper for SIM-APA v2.0.

Uses Ultralytics only to load pretrained weights, then works directly with
the underlying PyTorch DetectionModel.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOBackbone(nn.Module):
    """
    Frozen YOLOv10 feature extractor.

    Outputs:
        projected_features : [B, 256, 20, 20]
        detections         : raw model detection output
    """

    def __init__(
        self,
        model_name: str = "yolov10s.pt",
        hook_layer_index: int = 10,
        input_channels: int = 512,
        output_channels: int = 256,
    ) -> None:
        super().__init__()

        yolo = YOLO(model_name)

        self.model = yolo.model
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.hook_layer_index = hook_layer_index
        self._captured_feature: torch.Tensor | None = None

        self.projection = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
        )

        self._register_hook()

    def _register_hook(self) -> None:
        layers = self.model.model

        if self.hook_layer_index >= len(layers):
            raise ValueError("Invalid hook layer index.")

        def hook_fn(
            module: nn.Module,
            inputs: Any,
            output: torch.Tensor,
        ) -> None:
            self._captured_feature = output

        layers[self.hook_layer_index].register_forward_hook(hook_fn)

    def train(self, mode: bool = True):
        """
        Keep YOLO model frozen even when parent model enters train mode.
        """
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x: torch.Tensor):
        """
        Extract projected P5 features and raw detection output.
        """
        self._captured_feature = None

        with torch.no_grad():
            detections = self.model(x)

        if self._captured_feature is None:
            raise RuntimeError("Feature hook failed.")

        projected = self.projection(self._captured_feature)

        return projected, detections
