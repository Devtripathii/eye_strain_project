from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class UserBaseline:
    """
    Simple, robust per-user baseline using EWMA (exponentially weighted moving average).
    This is intentionally lightweight and works without heavy dependencies.

    We track baseline for:
    - blink_rate (blinks/min)
    - perclos (0..1)
    - ear_mean
    """
    user: str
    ema_blink_rate: float = 15.0
    ema_perclos: float = 0.10
    ema_ear_mean: float = 0.28

    # How fast the baseline adapts (0..1). Lower = slower drift.
    alpha: float = 0.15

    # Quality gates (avoid learning from bad sessions)
    min_duration_sec: int = 20
    min_frames_ok: int = 100

    def update_from_session(
        self,
        duration_sec: float,
        frames_ok: int,
        blink_rate_final: float,
        perclos_final: float,
        ear_mean: Optional[float],
    ) -> Dict[str, Any]:
        """
        Updates EWMA baselines from a completed session.
        Returns diagnostics dict.
        """
        diag: Dict[str, Any] = {"updated": False, "reason": ""}

        if duration_sec < self.min_duration_sec:
            diag["reason"] = f"duration too short ({duration_sec:.1f}s)"
            return diag
        if frames_ok < self.min_frames_ok:
            diag["reason"] = f"too few frames_ok ({frames_ok})"
            return diag

        # sanitize
        br = float(max(0.0, min(60.0, blink_rate_final)))
        pc = float(max(0.0, min(1.0, perclos_final)))

        # update
        a = float(self.alpha)
        self.ema_blink_rate = (1 - a) * self.ema_blink_rate + a * br
        self.ema_perclos = (1 - a) * self.ema_perclos + a * pc

        if ear_mean is not None and math.isfinite(float(ear_mean)):
            em = float(max(0.05, min(0.60, float(ear_mean))))
            self.ema_ear_mean = (1 - a) * self.ema_ear_mean + a * em

        diag["updated"] = True
        diag["reason"] = "ok"
        return diag

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "UserBaseline":
        return UserBaseline(
            user=str(d.get("user", "User")),
            ema_blink_rate=float(d.get("ema_blink_rate", 15.0)),
            ema_perclos=float(d.get("ema_perclos", 0.10)),
            ema_ear_mean=float(d.get("ema_ear_mean", 0.28)),
            alpha=float(d.get("alpha", 0.15)),
            min_duration_sec=int(d.get("min_duration_sec", 20)),
            min_frames_ok=int(d.get("min_frames_ok", 100)),
        )


def load_user_baseline(baseline_path: Path, user: str) -> UserBaseline:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if baseline_path.exists():
        try:
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
            ub = UserBaseline.from_dict(data)
            ub.user = user
            return ub
        except Exception:
            pass
    return UserBaseline(user=user)


def save_user_baseline(baseline_path: Path, baseline: UserBaseline) -> None:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline.to_dict(), indent=2), encoding="utf-8")