# src/core/state.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Signals:
    # Raw/instant signals
    ear: float = 0.0
    ear_drop_ratio: float = 0.0
    perclos: float = 0.0
    blink_rate_bpm: float = 0.0

    # CNN outputs
    cnn_prob: float = 0.0
    cnn_ewma: float = 0.0
    cnn_trend: float = 0.0  # slope

    # Fusion
    risk_score: float = 0.0
    risk_level: str = "LOW"
    reasons: List[str] = field(default_factory=list)

    # Timing / quality
    fps: float = 0.0
    dt: float = 0.0
    quality: float = 1.0  # placeholder for Phase 3 confidence


@dataclass
class SessionState:
    user_name: str = "User"

    # Baselines
    ear_baseline: float = 0.0
    blink_baseline_bpm: float = 0.0

    # Counters
    frame_index: int = 0
    elapsed_sec: float = 0.0

    # Any extra per-app values
    meta: Dict[str, object] = field(default_factory=dict)

    def step(self, dt: float) -> None:
        self.frame_index += 1
        self.elapsed_sec += float(dt)