"""
Evaluator for SIM-APA v2.0.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from evaluation.metrics import (
    compute_mse,
    compute_cosine_similarity,
    compute_psnr,
    compute_power,
    compute_compression_ratio,
)


class Evaluator:
    def __init__(self, model, device: str = "cuda") -> None:
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        self.model.eval()

        total_mse = 0.0
        total_cosine = 0.0
        total_psnr = 0.0
        total_latent_power = 0.0
        total_received_power = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc="Evaluation")

        for batch in progress:
            images = batch["image"].to(self.device)

            outputs = self.model(images)

            mse = compute_mse(
                reconstructed=outputs["reconstructed"],
                target=outputs["features"],
            )

            cosine = compute_cosine_similarity(
                reconstructed=outputs["reconstructed"],
                target=outputs["features"],
            )

            psnr = compute_psnr(mse)

            latent_power = compute_power(outputs["latent"])
            received_power = compute_power(outputs["received"])

            total_mse += mse.item()
            total_cosine += cosine.item()
            total_psnr += psnr
            total_latent_power += latent_power.item()
            total_received_power += received_power.item()
            num_batches += 1

            progress.set_postfix(
                mse=mse.item(),
                cos=cosine.item(),
            )

        return {
            "mse": total_mse / num_batches,
            "cosine_similarity": total_cosine / num_batches,
            "psnr": total_psnr / num_batches,
            "latent_power": total_latent_power / num_batches,
            "received_power": total_received_power / num_batches,
            "compression_ratio": compute_compression_ratio(
                input_channels=256,
                latent_channels=32,
            ),
        }
