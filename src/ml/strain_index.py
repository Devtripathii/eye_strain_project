from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class StrainIndex:
    """
    CSI (Cumulative Strain Index): session-level accumulated strain load.

    - risk_score in [0..1]
    - dt_sec time step in seconds
    - nonlinear exponent makes HIGH risk contribute more than MED
    - session_bonus increases load as session gets longer without breaks
    """
    csi: float = 0.0
    exponent: float = 1.3
    session_bonus_strength: float = 0.25  # 0..1
    max_csi: float = 500.0  # clamp to avoid runaway

    def update(self, risk_score: float, dt_sec: float, seconds_since_break: float) -> float:
        r = float(max(0.0, min(1.0, risk_score)))
        dt = float(max(0.0, dt_sec))

        # session bonus: grows slowly after 20 minutes without break
        # 0 before 20 min, approaches ~session_bonus_strength
        s = max(0.0, float(seconds_since_break) - 20.0 * 60.0)
        bonus = self.session_bonus_strength * (1.0 - math.exp(-s / (15.0 * 60.0)))  # 15-min time constant

        inc = (r ** self.exponent) * dt * (1.0 + bonus)
        self.csi = float(min(self.max_csi, self.csi + inc))
        return self.csi

    def recovery(self, dt_sec: float, recovery_tau_sec: float = 240.0) -> float:
        """
        Exponential decay during breaks.
        recovery_tau_sec: higher => slower recovery
        """
        dt = float(max(0.0, dt_sec))
        tau = float(max(1.0, recovery_tau_sec))
        self.csi = float(self.csi * math.exp(-dt / tau))
        return self.csi

    def status(self) -> Dict[str, Any]:
        """
        Converts CSI into a simple category (for UI and scheduling).
        """
        # these thresholds are heuristic and will become personalized later
        if self.csi >= 220:
            return {"level": "HIGH", "label": "Overload", "csi": self.csi}
        if self.csi >= 120:
            return {"level": "MED", "label": "Accumulating", "csi": self.csi}
        return {"level": "LOW", "label": "Normal", "csi": self.csi}