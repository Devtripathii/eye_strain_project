# src/scheduler/break_scheduler.py
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass
class BreakThresholds:
    # Fatigue-load bands (Phase 2)
    micro_lo: float = 0.35
    micro_hi: float = 0.55
    short_hi: float = 0.55
    long_hi: float = 0.75

    # Instant risk trigger (Phase 2)
    sustain_risk_hi: float = 0.55
    sustain_seconds: float = 30.0

    # Microsleep trigger (Phase 3)
    microsleep_duration_sec: float = 0.50
    microsleep_window_sec: float = 120.0
    microsleep_short_break_count: int = 1  # 1 microsleep -> SHORT
    microsleep_long_break_count: int = 2   # 2 microsleeps in window -> LONG

    # Cooldown between starting breaks
    min_gap_between_breaks_sec: float = 60.0


@dataclass
class BreakDurations:
    micro_sec: int = 20
    short_sec: int = 180
    long_sec: int = 600


@dataclass
class BreakState:
    mode: str = "WORK"          # WORK / MICRO / SHORT / LONG
    remaining_sec: float = 0.0  # countdown when on break
    last_break_end_t: float = -1e9
    breaks_taken: int = 0

    # For sustained risk detection
    hi_risk_accum_sec: float = 0.0

    # Microsleep tracking: timestamps of microsleep events
    microsleep_times: Deque[float] = field(default_factory=lambda: deque(maxlen=200))


class BreakScheduler:
    """
    Converts fatigue_load + instantaneous risk (+ microsleep events) into break modes.

    Priority:
      1) Microsleep trigger (SHORT/LONG)
      2) Fatigue-load LONG
      3) Sustained high-risk SHORT
      4) Fatigue-load SHORT
      5) Fatigue-load MICRO
    """

    def __init__(self, thresholds: BreakThresholds | None = None, durations: BreakDurations | None = None):
        self.th = thresholds or BreakThresholds()
        self.du = durations or BreakDurations()
        self.state = BreakState()

    def reset(self) -> None:
        self.state = BreakState()

    def is_on_break(self) -> bool:
        return self.state.mode in ("MICRO", "SHORT", "LONG")

    def microsleep_count(self, t_sec: float) -> int:
        # prune microsleep times by window
        while self.state.microsleep_times and (t_sec - self.state.microsleep_times[0]) > float(self.th.microsleep_window_sec):
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
        r = _clamp01(risk01)
        f = _clamp01(fatigue_load01)
        t_sec = float(t_sec)
        dt = float(dt)

        # Register microsleep event timestamp (if any)
        if microsleep_event and not self.is_on_break():
            self.state.microsleep_times.append(t_sec)

        # Update sustained high-risk accumulator (only while working)
        if not self.is_on_break() and r >= self.th.sustain_risk_hi:
            self.state.hi_risk_accum_sec += dt
        else:
            # decay quickly so brief spikes don't count
            self.state.hi_risk_accum_sec = max(0.0, self.state.hi_risk_accum_sec - 2.0 * dt)

        # If currently on break, countdown and finish when done
        if self.is_on_break():
            self.state.remaining_sec = max(0.0, self.state.remaining_sec - dt)
            if self.state.remaining_sec <= 0.0:
                self.state.last_break_end_t = t_sec
                self.state.mode = "WORK"
            # still prune microsleep window
            self.microsleep_count(t_sec)
            return self.state

        # Cooldown gate
        since_last_break = t_sec - self.state.last_break_end_t
        if since_last_break < self.th.min_gap_between_breaks_sec:
            self.microsleep_count(t_sec)
            return self.state

        # ---- Priority 1: Microsleep triggers ----
        ms_count = self.microsleep_count(t_sec)
        if ms_count >= int(self.th.microsleep_long_break_count):
            self._start_break("LONG", self.du.long_sec)
            return self.state
        if ms_count >= int(self.th.microsleep_short_break_count):
            self._start_break("SHORT", self.du.short_sec)
            return self.state

        # ---- Priority 2: Long break by fatigue load ----
        if f >= self.th.long_hi:
            self._start_break("LONG", self.du.long_sec)
            return self.state

        # ---- Priority 3: Sustained high risk -> short break ----
        if self.state.hi_risk_accum_sec >= self.th.sustain_seconds:
            self._start_break("SHORT", self.du.short_sec)
            self.state.hi_risk_accum_sec = 0.0
            return self.state

        # ---- Priority 4: Short break by fatigue load ----
        if f >= self.th.short_hi:
            self._start_break("SHORT", self.du.short_sec)
            return self.state

        # ---- Priority 5: Micro break band ----
        if self.th.micro_lo <= f < self.th.micro_hi:
            self._start_break("MICRO", self.du.micro_sec)
            return self.state

        return self.state

    def _start_break(self, mode: str, seconds: int) -> None:
        self.state.mode = mode
        self.state.remaining_sec = float(seconds)
        self.state.breaks_taken += 1