# src/scheduler/fatigue_load.py
from __future__ import annotations

from dataclasses import dataclass


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass
class FatigueLoadParams:
    # per-second rates (tune later from logs)
    k_up: float = 0.020     # accumulation rate while working
    k_down: float = 0.050   # recovery rate while on break and risk is low
    k_leak: float = 0.003   # slow decay always present
    max_dt: float = 0.25    # safety clamp (should match Timebase dt_max)


class FatigueLoad:
    """
    Fatigue load F in [0,1], updated each tick using real dt.

    dF/dt = k_up*u*r*q - k_down*(1-u)*(1-r)*q - k_leak*F

    where:
      r = instantaneous risk score in [0,1]
      u = 1 if working, 0 if on break
      q = signal quality/confidence in [0,1] (use 1.0 for now)
    """

    def __init__(self, params: FatigueLoadParams | None = None):
        self.p = params or FatigueLoadParams()
        self.value: float = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def update(self, risk01: float, dt: float, working: bool = True, quality01: float = 1.0) -> float:
        r = _clamp01(risk01)
        q = _clamp01(quality01)

        if dt < 0.0:
            dt = 0.0
        if dt > self.p.max_dt:
            dt = self.p.max_dt

        u = 1.0 if working else 0.0

        dF = dt * (
            self.p.k_up * u * r * q
            - self.p.k_down * (1.0 - u) * (1.0 - r) * q
            - self.p.k_leak * self.value
        )

        self.value = _clamp01(self.value + dF)
        return self.value