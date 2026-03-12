from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque


def _clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return float(x)


@dataclass
class BreakThresholds:
    # FIX: micro_lo lowered 0.35→0.25 so micro breaks trigger in 60s sessions.
    # FIX: sustain_risk_hi lowered 0.55→0.40, sustain_seconds 30→20
    # so sustained MED risk triggers a short break within the session window.
    micro_lo:  float = 0.25   # was 0.35
    micro_hi:  float = 0.45   # was 0.55
    short_hi:  float = 0.55   # was 0.55 (unchanged)
    long_hi:   float = 0.75   # unchanged

    sustain_risk_hi:    float = 0.40   # was 0.55
    sustain_seconds:    float = 20.0   # was 30.0

    microsleep_duration_sec:    float = 0.50
    microsleep_window_sec:      float = 120.0
    microsleep_short_break_count: int = 1
    microsleep_long_break_count:  int = 2

    min_gap_between_breaks_sec: float = 60.0


@dataclass
class BreakDurations:
    micro_sec: int = 20
    short_sec: int = 180
    long_sec:  int = 600


@dataclass
class BreakState:
    mode:             str   = "WORK"
    remaining_sec:    float = 0.0
    last_break_end_t: float = -1e9
    breaks_taken:     int   = 0
    hi_risk_accum_sec: float = 0.0
    microsleep_times: Deque[float] = field(default_factory=lambda: deque(maxlen=200))


class BreakScheduler:
    """
    Converts fatigue_load + risk + microsleep events into break modes.

    Priority:
      1) Microsleep trigger (SHORT / LONG)
      2) Fatigue-load LONG
      3) Sustained high-risk SHORT  (now triggers at 0.40 for 20s)
      4) Fatigue-load SHORT
      5) Fatigue-load MICRO         (now triggers at F=0.25)
    """

    def __init__(
        self,
        thresholds: BreakThresholds | None = None,
        durations:  BreakDurations  | None = None,
    ):
        self.th    = thresholds or BreakThresholds()
        self.du    = durations  or BreakDurations()
        self.state = BreakState()

    def reset(self) -> None:
        self.state = BreakState()

    def is_on_break(self) -> bool:
        return self.state.mode in ("MICRO", "SHORT", "LONG")

    def microsleep_count(self, t_sec: float) -> int:
        while (self.state.microsleep_times
               and (t_sec - self.state.microsleep_times[0]) > float(self.th.microsleep_window_sec)):
            self.state.microsleep_times.popleft()
        return len(self.state.microsleep_times)

    def update(
        self,
        t_sec: float,
        dt: float,
        risk01: float,
        fatigue_load01: float,
        microsleep_event: bool = False,
    ) -> BreakState:
        r    = _clamp01(risk01)
        f    = _clamp01(fatigue_load01)
        t_sec = float(t_sec)
        dt    = float(dt)

        if microsleep_event and not self.is_on_break():
            self.state.microsleep_times.append(t_sec)

        # Sustained risk accumulator
        if not self.is_on_break() and r >= self.th.sustain_risk_hi:
            self.state.hi_risk_accum_sec += dt
        else:
            self.state.hi_risk_accum_sec = max(
                0.0, self.state.hi_risk_accum_sec - 2.0 * dt)

        # Countdown active break
        if self.is_on_break():
            self.state.remaining_sec = max(0.0, self.state.remaining_sec - dt)
            if self.state.remaining_sec <= 0.0:
                self.state.last_break_end_t = t_sec
                self.state.mode = "WORK"
            self.microsleep_count(t_sec)
            return self.state

        # Cooldown gate
        if (t_sec - self.state.last_break_end_t) < self.th.min_gap_between_breaks_sec:
            self.microsleep_count(t_sec)
            return self.state

        # Priority 1: microsleep
        ms_count = self.microsleep_count(t_sec)
        if ms_count >= int(self.th.microsleep_long_break_count):
            return self._start_break("LONG",  self.du.long_sec)
        if ms_count >= int(self.th.microsleep_short_break_count):
            return self._start_break("SHORT", self.du.short_sec)

        # Priority 2: fatigue load LONG
        if f >= self.th.long_hi:
            return self._start_break("LONG", self.du.long_sec)

        # Priority 3: sustained high risk
        if self.state.hi_risk_accum_sec >= self.th.sustain_seconds:
            self.state.hi_risk_accum_sec = 0.0
            return self._start_break("SHORT", self.du.short_sec)

        # Priority 4: fatigue load SHORT
        if f >= self.th.short_hi:
            return self._start_break("SHORT", self.du.short_sec)

        # Priority 5: fatigue load MICRO
        if self.th.micro_lo <= f < self.th.micro_hi:
            return self._start_break("MICRO", self.du.micro_sec)

        return self.state

    def _start_break(self, mode: str, seconds: int) -> BreakState:
        self.state.mode          = mode
        self.state.remaining_sec = float(seconds)
        self.state.breaks_taken += 1
        return self.state