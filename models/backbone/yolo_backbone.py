"""
YOLOv10 backbone wrapper for SIM-APA v2.0.

This module provides two outputs required by the SIM-APA pipeline:

1. Projected P5 feature map of shape [B, 256, 20, 20]
2. YOLO object detections required by the Semantic Importance Module

The raw YOLOv10s P5 feature map is 512 channels, so a learnable 1x1
projection layer maps it to 256 channels.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO


class YOLOBackbone(nn.Module):
    """
    YOLOv10 feature extractor and detector for SIM-APA.

    Parameters
    ----------
    model_name : str
        YOLO weight file name, e.g. "yolov10s.pt".
    hook_layer_index : int
        Layer index whose activation is captured as the P5 feature map.
    input_channels : int
        Number of channels in the captured P5 feature map.
    output_channels : int
        Number of channels after 1x1 projection.
    conf_threshold : float
        Minimum detection confidence retained for SIM.

    Returns
    -------
    tuple
        projected_features : torch.Tensor
            Tensor of shape [B, 256, 20, 20].
        detections : list
            List of Ultralytics detection results.
    """

    def __init__(
        self,
        model_name: str = "yolov10s.pt",
        hook_layer_index: int = 10,
        input_channels: int = 512,
        output_channels: int = 256,
        conf_threshold: float = 0.25,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.hook_layer_index = hook_layer_index
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conf_threshold = conf_threshold

        self.yolo = YOLO(model_name)
        self.yolo_model = self.yolo.model

        self._captured_feature: torch.Tensor | None = None

        self.projection = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._register_feature_hook()

    def _register_feature_hook(self) -> None:
        """
        Register a forward hook on the selected YOLO layer.

        The hook stores the intermediate feature tensor produced during
        the normal YOLO forward pass.
        """
        layers = self.yolo_model.model

        if self.hook_layer_index >= len(layers):
            raise ValueError(
                f"Invalid hook_layer_index={self.hook_layer_index}. "
                f"The YOLO model has only {len(layers)} layers."
            )

        target_layer = layers[self.hook_layer_index]

        def hook_fn(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
            self._captured_feature = output

        target_layer.register_forward_hook(hook_fn)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract projected P5 features from an input image tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 3, 640, 640].

        Returns
        -------
        torch.Tensor
            Projected feature tensor of shape [B, 256, 20, 20].
        """
        self._captured_feature = None

        _ = self.yolo_model(x)

        if self._captured_feature is None:
            raise RuntimeError(
                "Feature hook did not capture any tensor. "
                "Check hook_layer_index."
            )

        projected_features = self.projection(self._captured_feature)

        return projected_features

    def detect(self, x: torch.Tensor):
        """
        Run YOLO object detection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 3, 640, 640].

        Returns
        -------
        list
            Ultralytics detection results.
        """
        results = self.yolo.predict(
            source=x,
            conf=self.conf_threshold,
            verbose=False,
        )

        return results

    def forward(self, x: torch.Tensor):
        """
        Return both projected features and detections.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 3, 640, 640].

        Returns
        -------
        tuple
            projected_features, detections
        """
        projected_features = self.extract_features(x)
        detections = self.detect(x)

        return projected_features, detections
