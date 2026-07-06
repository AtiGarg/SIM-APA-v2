"""
Forward YOLOv10 detection head from a reconstructed layer-10 feature.

This utility:
1. Runs YOLO up to layer 10 to collect skip features.
2. Replaces layer-10 output with reconstructed/reprojected feature.
3. Runs layers 11–23 to obtain detections.

If enable_grad=True, gradients flow through layers 11–23 back to layer10_feature.
"""

from __future__ import annotations

import torch


def forward_from_layer10(
    yolo_model,
    images: torch.Tensor,
    layer10_feature: torch.Tensor,
    enable_grad: bool = False,
):
    layers = yolo_model.model

    outputs = []
    x = images

    # Collect skip features using frozen YOLO layers.
    with torch.no_grad():
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
                x = layer10_feature if enable_grad else layer10_feature.detach()

            outputs.append(x)

    # Important: layer 10 must remain connected to graph when enable_grad=True.
    outputs[10] = layer10_feature if enable_grad else layer10_feature.detach()
    x = outputs[10]

    context = torch.enable_grad() if enable_grad else torch.no_grad()

    with context:
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
