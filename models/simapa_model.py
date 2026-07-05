"""
Integrated SIM-APA model for end-to-end semantic communication.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.backbone.yolo_backbone import YOLOBackbone
from models.sim.semantic_importance import SemanticImportanceModule
from models.fusion.fusion import FeatureAwareFusion
from models.apa.apa import AdaptivePowerAllocation
from models.deepjscc.autoencoder import DeepJSCCEncoder, DeepJSCCDecoder
from channels.factory import build_channel


class SIMAPAModel(nn.Module):
    """
    End-to-end SIM-APA model.

    This version uses YOLOv10 only as a frozen feature extractor.
    During training, detections are set to empty because raw YOLO outputs from
    the PyTorch DetectionModel are not Ultralytics Results objects.
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        self.backbone = YOLOBackbone(
            model_name="yolov10s.pt",
            hook_layer_index=10,
            input_channels=cfg.model.raw_feature_channels,
            output_channels=cfg.model.feature_channels,
        )

        self.backbone.eval()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.sim = SemanticImportanceModule(
            alpha=cfg.sim.alpha,
            beta=cfg.sim.beta,
            image_size=cfg.dataset.image_size,
            feature_size=cfg.model.feature_height,
            hidden_dim=cfg.adaptive_sim.hidden_dim,
        )

        self.fusion = FeatureAwareFusion(
            feature_channels=cfg.model.feature_channels,
            sim_channels=1,
            hidden_channels=cfg.model.feature_channels,
        )

        self.apa = AdaptivePowerAllocation(
            input_channels=cfg.model.feature_channels,
            hidden_channels_1=128,
            hidden_channels_2=64,
        )

        self.encoder = DeepJSCCEncoder(
            input_channels=cfg.deepjscc.input_channels,
            latent_channels=cfg.deepjscc.latent_channels,
        )

        self.channel = build_channel(
            channel_type=cfg.channel.type,
            snr_db=cfg.channel.snr_db,
            k_factor=cfg.rician.k_factor,
        )

        self.decoder = DeepJSCCDecoder(
            latent_channels=cfg.deepjscc.latent_channels,
            output_channels=cfg.deepjscc.input_channels,
        )

        # Reproject reconstructed semantic feature back to YOLO layer-10 space.
        # Decoder output: [B, 256, 20, 20]
        # YOLO layer 10:  [B, 512, 20, 20]
        self.reprojection = nn.Conv2d(
            cfg.deepjscc.input_channels,
            cfg.model.raw_feature_channels,
            kernel_size=1,
        )

    def train(self, mode: bool = True):
        """
        Set train/eval mode while keeping YOLO backbone frozen.
        """
        super().train(mode)

        self.backbone.eval()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        return self

    def _empty_detections(self, device: torch.device) -> dict:
        """
        Create an empty detection dictionary.

        This keeps SIM stable during training when using frozen YOLO features.
        """
        return {
            "boxes": torch.empty((0, 4), device=device),
            "classes": torch.empty((0,), dtype=torch.long, device=device),
            "confidences": torch.empty((0,), device=device),
        }

    def forward(self, images: torch.Tensor) -> dict:
        """
        Run full forward pass.

        Parameters
        ----------
        images : torch.Tensor
            Input image tensor [B, 3, 640, 640].

        Returns
        -------
        dict
            Model outputs.
        """
        with torch.no_grad():
            backbone_out = self.backbone(images)

        raw_feature = backbone_out["raw_feature"]
        features = backbone_out["projected_feature"]

        detections = self._empty_detections(images.device)

        sim_map, sim_info = self.sim(detections)

        fused_features = self.fusion(
            features,
            sim_map.to(images.device),
        )

        apa_out = self.apa(fused_features)

        latent = self.encoder(apa_out["weighted_features"])

        received = self.channel(latent)

        reconstructed = self.decoder(received)

        reprojected = self.reprojection(reconstructed)

        return {
            "raw_feature": raw_feature,
            "features": features,
            "detections": detections,
            "sim_map": sim_map,
            "sim_info": sim_info,
            "fused_features": fused_features,
            "power_map": apa_out["power_map"],
            "weighted_features": apa_out["weighted_features"],
            "latent": latent,
            "received": received,
            "reconstructed": reconstructed,
            "reprojected": reprojected,
        }
