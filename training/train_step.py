"""
Single training step for SIM-APA v2.0.
"""

from __future__ import annotations

import torch


def train_step(
    image_tensor: torch.Tensor,
    backbone,
    sim,
    fusion,
    apa,
    encoder,
    channel,
    decoder,
    criterion,
    optimizer,
) -> dict:
    """
    Run one optimization step.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Input image tensor [B, 3, 640, 640].

    Returns
    -------
    dict
        Loss values and tensor shapes.
    """
    optimizer.zero_grad()

    with torch.no_grad():
        features, detections = backbone(image_tensor)

        boxes = detections[0].boxes

        parsed_detections = {
            "boxes": boxes.xyxy.to(image_tensor.device),
            "classes": boxes.cls.long().to(image_tensor.device),
            "confidences": boxes.conf.to(image_tensor.device),
        }

    sim_map, sim_info = sim(parsed_detections)
    fused_features = fusion(features, sim_map.to(image_tensor.device))
    apa_out = apa(fused_features)

    latent = encoder(apa_out["weighted_features"])
    received = channel(latent)
    reconstructed = decoder(received)

    loss_out = criterion(
        reconstructed=reconstructed,
        target=features.detach(),
    )

    loss_out["loss"].backward()
    optimizer.step()

    return {
        "loss": loss_out["loss"].item(),
        "mse_loss": loss_out["mse_loss"].item(),
        "cosine_loss": loss_out["cosine_loss"].item(),
        "cosine_similarity": loss_out["cosine_similarity"].item(),
        "latent_shape": tuple(latent.shape),
        "reconstructed_shape": tuple(reconstructed.shape),
    }
