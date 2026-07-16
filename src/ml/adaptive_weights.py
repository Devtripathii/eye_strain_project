"""
src/ml/adaptive_weights.py
--------------------------
Online learning: adapts signal weights based on user feedback.

When the user says "I was NOT tired" → model over-estimated → reduce weights.
When the user says "I WAS tired" → model under-estimated → increase weights.

This is what separates EyeGuard from every reminder app:
  "The model literally changes its parameters based on your corrections."

Weight vector covers the signals used in fuse_risk.py, extended with
the new signals (MAR, FHP, ear_asym, IBR).

Usage:
    weights = AdaptiveWeights.load(path)   # load or create default
    weights.correct(predicted_high=True, was_tired=False)  # user says not tired
    weights.save(path)
    
    # Get current weights for display / logging
    w = weights.as_dict()
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


# Default weights — should sum to 1.0
# Aligned with fuse_risk.py contribution levels
DEFAULT_WEIGHTS: Dict[str, float] = {
    "perclos":    0.28,   # from fuse_risk: 0.50 max contribution → 28% of full score
    "blink_rate": 0.20,   # from fuse_risk: 0.25 max
    "cnn_prob":   0.22,   # from fuse_risk: 0.35 max (highest single signal)
    "mar":        0.10,   # new: yawn / MAR signal
    "fhp_angle":  0.08,   # new: forward head posture
    "ear_asym":   0.07,   # new: bilateral asymmetry
    "ibr":        0.05,   # new: incomplete blink ratio
}

LEARN_RATE      = 0.04   # per correction (4% weight shift)
WEIGHT_MIN      = 0.02   # no signal drops below 2%
WEIGHT_MAX      = 0.50   # no single signal dominates above 50%


@dataclass
class AdaptiveWeights:
    weights: Dict[str, float]
    corrections: int = 0
    false_positives: int = 0   # model said tired, user wasn't
    false_negatives: int = 0   # model missed real tiredness

    def correct(self, predicted_high: bool, was_tired: bool) -> Dict[str, float]:
        """
        Apply one user correction and update weights.

        predicted_high: True if model said MED or HIGH risk
        was_tired:      True if user confirms they were actually tired

        Returns the updated weight dict.
        """
        self.corrections += 1

        if predicted_high and not was_tired:
            # False positive → reduce all weights slightly
            self.false_positives += 1
            self._scale_all(1.0 - LEARN_RATE)

        elif not predicted_high and was_tired:
            # False negative → increase all weights slightly
            self.false_negatives += 1
            self._scale_all(1.0 + LEARN_RATE)

        # Always renormalise to sum = 1.0
        self._normalise()
        return dict(self.weights)

    def _scale_all(self, factor: float) -> None:
        for k in self.weights:
            self.weights[k] = max(WEIGHT_MIN,
                                  min(WEIGHT_MAX, self.weights[k] * factor))

    def _normalise(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] = round(self.weights[k] / total, 4)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.weights)

    def accuracy_summary(self) -> Dict[str, int]:
        return {
            "total_corrections": self.corrections,
            "false_positives":   self.false_positives,
            "false_negatives":   self.false_negatives,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights":        self.weights,
            "corrections":    self.corrections,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "AdaptiveWeights":
        """Load from file, or create defaults if file doesn't exist."""
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return AdaptiveWeights(
                    weights        = data.get("weights", dict(DEFAULT_WEIGHTS)),
                    corrections    = int(data.get("corrections", 0)),
                    false_positives= int(data.get("false_positives", 0)),
                    false_negatives= int(data.get("false_negatives", 0)),
                )
            except Exception:
                pass
        return AdaptiveWeights(weights=dict(DEFAULT_WEIGHTS))

    @staticmethod
    def default() -> "AdaptiveWeights":
        return AdaptiveWeights(weights=dict(DEFAULT_WEIGHTS))