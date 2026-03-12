from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


@dataclass
class CnnResult:
    sleepy_prob: float
    awake_prob: float


class TorchEyeCnn:
    """
    Loads your trained ResNet18 (grayscale, 2-class) and returns sleepy probability.

    Expected class order: ['awake', 'sleepy'] -> index 0 = awake, index 1 = sleepy
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        input_size: int = 224,
        grayscale: bool = True,
        mean: Tuple[float, ...] = (0.5,),
        std: Tuple[float, ...] = (0.5,),
        device: str = "cpu",
    ):
        # If no path given, pull from config so callers don't hardcode it
        if model_path is None:
            try:
                import config
                model_path = config.CNN_MODEL_PATH
            except ImportError:
                raise RuntimeError(
                    "No model_path provided and config.py not found. "
                    "Pass model_path explicitly or create config.py in project root."
                )

        self.model_path = Path(model_path)
        self.input_size = int(input_size)
        self.grayscale  = bool(grayscale)
        self.mean       = mean
        self.std        = std
        self.device     = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.model_path}")

        self.model = self._build_model()
        self._load_weights()
        self.model.eval()

        self.tfms = self._build_transform()

    def _build_model(self) -> torch.nn.Module:
        m = models.resnet18(weights=None)
        if self.grayscale:
            m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m.to(self.device)

    def _load_weights(self):
        ckpt = torch.load(self.model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"], strict=True)
        elif isinstance(ckpt, dict):
            self.model.load_state_dict(ckpt, strict=True)
        else:
            raise RuntimeError("Unsupported checkpoint format (expected dict).")

    def _build_transform(self):
        t = [
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
        ]
        if self.grayscale:
            t.append(transforms.Grayscale(num_output_channels=1))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
        return transforms.Compose(t)

    @torch.no_grad()
    def predict_roi_bgr(self, roi_bgr: np.ndarray) -> Optional[CnnResult]:
        """
        roi_bgr: OpenCV BGR image (H,W,3). Returns None if roi is invalid.
        """
        if roi_bgr is None:
            return None
        if not isinstance(roi_bgr, np.ndarray):
            return None
        if roi_bgr.size == 0:
            return None

        # Convert BGR -> RGB for PIL consistency
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        x = self.tfms(roi_rgb).unsqueeze(0).to(self.device)  # [1,C,H,W]
        logits = self.model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        awake  = float(probs[0])
        sleepy = float(probs[1])
        return CnnResult(sleepy_prob=sleepy, awake_prob=awake)