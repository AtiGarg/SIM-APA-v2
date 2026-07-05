"""
Forward YOLOv10 detection head from a reconstructed layer-10 feature.

This utility:
1. Runs YOLO up to layer 10 to collect skip features.
2. Replaces layer-10 output with reconstructed/reprojected feature.
3. Runs layers 11–23 to obtain detections.
"""

from __future__ import annotations

import torch


def forward_from_layer10(
    yolo_model,
    images: torch.Tensor,
    layer10_feature: torch.Tensor,
):
    """
    Forward YOLOv10 layers 11–23 using an externally supplied layer-10 feature.

    Parameters
    ----------
    yolo_model:
        Ultralytics DetectionModel.
    images:
        Input image tensor [B, 3, 640, 640].
    layer10_feature:
        Reconstructed YOLO layer-10 feature [B, 512, 20, 20].
    """
    layers = yolo_model.model

    outputs = []
    x = images

    with torch.no_grad():
        # Run layers 0–10 normally to collect required skip features.
        for i in range(0, 11):
            layer = layers[i]

            if layer.f == -1:
                x_in = x
            elif isinstance(layer.f, int):
                x_in = outputs[layer.f]
            else:
                x_in = [x if j == -1 else outputs[j] for j in layer.f]

            x = layer(x_in)

            if i == 10:
                x = layer10_feature

            outputs.append(x)

        # Continue from layers 11–23.
        for i in range(11, len(layers)):
            layer = layers[i]

            if layer.f == -1:
                x_in = x
            elif isinstance(layer.f, int):
                x_in = outputs[layer.f]
            else:
                x_in = [x if j == -1 else outputs[j] for j in layer.f]

            x = layer(x_in)
            outputs.append(x)

    return x
