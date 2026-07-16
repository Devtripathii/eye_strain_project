"""
src/ml/anomaly.py
-----------------
Personal anomaly detector using IsolationForest.

Trains on YOUR first N readings of normal eye behaviour.
Flags when your current signals deviate from YOUR personal baseline —
not from a population average.

This is the key differentiator vs reminder apps:
  "Your laptop's reminder doesn't know what YOUR normal eyes look like."

Usage:
    detector = PersonalAnomalyDetector()
    detector.add_sample(features)          # call each frame during RUN
    if detector.is_trained:
        anomaly, score = detector.predict(features)
"""

from __future__ import annotations

import json
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Feature vector order (must be consistent across add_sample / predict)
FEATURE_KEYS = [
    "ear",          # eye aspect ratio
    "perclos",      # % eye closure
    "blink_rate",   # blinks/min
    "mar",          # mouth aspect ratio
    "fhp_angle",    # forward head posture degrees
    "ear_asym",     # bilateral EAR asymmetry
]

MIN_SAMPLES_TO_TRAIN = 60    # ~3 seconds at 20fps — enough for a personal baseline
CONTAMINATION        = 0.05  # expect 5% anomalous readings


class PersonalAnomalyDetector:
    """
    Trains an IsolationForest on your personal eye signal baseline,
    then flags deviations from YOUR normal — not population averages.
    """

    def __init__(
        self,
        min_samples: int = MIN_SAMPLES_TO_TRAIN,
        contamination: float = CONTAMINATION,
        buffer_size: int = 500,
    ):
        self.min_samples   = int(min_samples)
        self.contamination = float(contamination)

        self._buffer: deque = deque(maxlen=buffer_size)
        self._model:  Optional[IsolationForest] = None
        self._scaler: StandardScaler = StandardScaler()
        self.is_trained: bool = False
        self.samples_seen: int = 0

    def _to_vec(self, features: dict) -> Optional[np.ndarray]:
        """Convert feature dict to numpy vector. Returns None if any key missing."""
        try:
            return np.array([float(features[k]) for k in FEATURE_KEYS],
                            dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None

    def add_sample(self, features: dict) -> bool:
        """
        Add one frame's worth of signals to the training buffer.
        Automatically trains once min_samples is reached.
        Returns True if training triggered this call.
        """
        vec = self._to_vec(features)
        if vec is None:
            return False

        self._buffer.append(vec)
        self.samples_seen += 1

        # Auto-train once we have enough data
        if not self.is_trained and len(self._buffer) >= self.min_samples:
            self._train()
            return True
        return False

    def _train(self) -> None:
        X = np.array(list(self._buffer), dtype=np.float32)
        try:
            X_scaled = self._scaler.fit_transform(X)
            self._model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=42,
                n_jobs=1,
            )
            self._model.fit(X_scaled)
            self.is_trained = True
        except Exception as e:
            print(f"[AnomalyDetector] Training failed: {e}")

    def predict(self, features: dict) -> Tuple[bool, float]:
        """
        Returns (is_anomaly, anomaly_score).
        anomaly_score: higher = more anomalous (range roughly -0.5 to 0.5)
        Returns (False, 0.0) if not yet trained.
        """
        if not self.is_trained or self._model is None:
            return False, 0.0

        vec = self._to_vec(features)
        if vec is None:
            return False, 0.0

        try:
            X = vec.reshape(1, -1)
            X_scaled = self._scaler.transform(X)
            pred  = self._model.predict(X_scaled)[0]        # 1=normal, -1=anomaly
            score = float(-self._model.score_samples(X_scaled)[0])  # higher = more anomalous
            return pred == -1, round(score, 4)
        except Exception:
            return False, 0.0

    def retrain_from_buffer(self) -> bool:
        """Force retrain on current buffer (call after a long session to refresh baseline)."""
        if len(self._buffer) < self.min_samples:
            return False
        self._train()
        return self.is_trained

    def reset(self) -> None:
        self._buffer.clear()
        self._model       = None
        self._scaler      = StandardScaler()
        self.is_trained   = False
        self.samples_seen = 0

    @property
    def training_progress(self) -> float:
        """0.0 → 1.0 progress toward min_samples needed for training."""
        return min(1.0, len(self._buffer) / max(1, self.min_samples))

    def save(self, path: Path) -> None:
        """Persist scaler mean/scale so baseline survives sessions."""
        if not self.is_trained:
            return
        data = {
            "scaler_mean":  self._scaler.mean_.tolist(),
            "scaler_scale": self._scaler.scale_.tolist(),
            "samples_seen": self.samples_seen,
            "feature_keys": FEATURE_KEYS,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> bool:
        """Restore scaler from saved file (model is retrained on next session)."""
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._scaler.mean_  = np.array(data["scaler_mean"])
            self._scaler.scale_ = np.array(data["scaler_scale"])
            self._scaler.var_   = self._scaler.scale_ ** 2
            self._scaler.n_features_in_ = len(FEATURE_KEYS)
            self.samples_seen   = int(data.get("samples_seen", 0))
            return True
        except Exception:
            return False