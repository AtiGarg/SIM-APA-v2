"""
KITTI dataset loader for SIM-APA v2.0.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class KITTIDataset(Dataset):
    """
    KITTI object detection image dataset.
    """

    def __init__(
        self,
        root_dir: str,
        image_folder: str,
        image_size: int = 640,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_folder
        self.image_size = image_size

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths = sorted(self.image_dir.glob("*.png"))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No PNG images found in: {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.image_size, self.image_size))

        image_tensor = torch.from_numpy(image_resized).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0

        return {
            "image": image_tensor,
            "path": str(image_path),
        }
