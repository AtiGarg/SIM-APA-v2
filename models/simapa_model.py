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

    Pipeline:
        Image
        -> YOLO Backbone
        -> Semantic Importance Module
        -> Feature-aware Fusion
        -> APA
        -> DeepJSCC Encoder
        -> Wireless Channel
        -> DeepJSCC Decoder
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        self.backbone = YOLOBackbone(
            model_name=f"{cfg.model.backbone_variant}.pt",
            hook_layer_index=10,
            input_channels=cfg.model.raw_feature_channels,
            output_channels=cfg.model.feature_channels,
            conf_threshold=0.25,
        )

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

    def _parse_detections(self, detections, device: torch.device) -> dict:
        """
        Convert Ultralytics detections into clean tensors.
        """
        boxes = detections[0].boxes

        return {
            "boxes": boxes.xyxy.to(device),
            "classes": boxes.cls.long().to(device),
            "confidences": boxes.conf.to(device),
        }

    def forward(self, images: torch.Tensor) -> dict:
        """
        Run full SIM-APA forward pass.

        Parameters
        ----------
        images : torch.Tensor
            Input images [B, 3, 640, 640].

        Returns
        -------
        dict
            Complete model outputs.
        """
        features, detections = self.backbone(images)

        parsed_detections = self._parse_detections(
            detections=detections,
            device=images.device,
        )

        sim_map, sim_info = self.sim(parsed_detections)

        fused_features = self.fusion(features, sim_map.to(images.device))

        apa_out = self.apa(fused_features)

        latent = self.encoder(apa_out["weighted_features"])

        received = self.channel(latent)

        reconstructed = self.decoder(received)

        return {
            "features": features,
            "detections": parsed_detections,
            "sim_map": sim_map,
            "sim_info": sim_info,
            "fused_features": fused_features,
            "power_map": apa_out["power_map"],
            "weighted_features": apa_out["weighted_features"],
            "latent": latent,
            "received": received,
            "reconstructed": reconstructed,
        }
