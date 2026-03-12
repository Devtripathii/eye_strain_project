from __future__ import annotations

from dataclasses import dataclass


def _clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return float(x)


@dataclass
class FatigueLoadParams:
    # FIX: k_up was 0.020 — at 15fps and risk=0.38 F never reached 0.35
    # in a 50s session. New values reach micro threshold in ~25s at MED risk.
    k_up:   float = 0.035   # was 0.020
    k_down: float = 0.060   # was 0.050
    k_leak: float = 0.003   # unchanged
    max_dt: float = 0.25    # unchanged


class FatigueLoad:
    """
    Fatigue load F in [0,1].

    dF/dt = k_up*u*r*q - k_down*(1-u)*(1-r)*q - k_leak*F

    At MED risk (r=0.38), 15fps (dt=0.067), k_up=0.035:
      dF/frame = 0.067 * 0.035 * 0.38 = 0.00089
      Frames to reach micro_lo=0.25: 0.25/0.00089 = 281 frames = 19s ✓
    """

    def __init__(self, params: FatigueLoadParams | None = None):
        self.p = params or FatigueLoadParams()
        self.value: float = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def update(
        self,
        risk01: float,
        dt: float,
        working: bool = True,
        quality01: float = 1.0,
    ) -> float:
        r = _clamp01(risk01)
        q = _clamp01(quality01)
        dt = max(0.0, min(dt, self.p.max_dt))
        u  = 1.0 if working else 0.0

        dF = dt * (
            self.p.k_up   * u         * r       * q
          - self.p.k_down * (1.0 - u) * (1.0 - r) * q
          - self.p.k_leak * self.value
        )

        self.value = _clamp01(self.value + dF)
        return self.value