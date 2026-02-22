# src/signals/blink_duration.py
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Optional, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return float(x)


@dataclass
class BlinkEvent:
    start_t: float
    end_t: float
    duration_sec: float
    is_microsleep: bool


@dataclass
class BlinkDurationTracker:
    """
    Tracks eye-closure segments using the boolean `closed` (from EAR threshold logic).
    Emits a BlinkEvent when the eye re-opens after being closed.

    This is robust and model-agnostic:
      - Uses dt from Timebase (real loop timing)
      - Detects microsleeps when closure duration >= microsleep_sec
      - Maintains rolling stats + microsleep count in a time window
    """

    microsleep_sec: float = 0.50
    microsleep_window_sec: float = 120.0
    max_dt: float = 0.25

    # internal state
    _in_closed: bool = False
    _closed_accum_sec: float = 0.0
    _closed_start_t: Optional[float] = None

    # rolling storage
    recent_blinks_sec: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    microsleep_times: Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    last_blink_duration_sec: Optional[float] = None
    last_microsleep: bool = False

    def reset(self) -> None:
        self._in_closed = False
        self._closed_accum_sec = 0.0
        self._closed_start_t = None
        self.recent_blinks_sec.clear()
        self.microsleep_times.clear()
        self.last_blink_duration_sec = None
        self.last_microsleep = False

    def update(self, closed: bool, dt: float, t_sec: float) -> Tuple[Optional[BlinkEvent], int]:
        """
        Returns:
          (event_or_none, microsleep_count_in_window)
        """
        dt = _clamp(float(dt), 0.0, self.max_dt)
        t_sec = float(t_sec)

        event: Optional[BlinkEvent] = None

        if closed:
            if not self._in_closed:
                self._in_closed = True
                self._closed_start_t = t_sec
                self._closed_accum_sec = 0.0
            self._closed_accum_sec += dt
        else:
            if self._in_closed:
                # closing ended -> emit blink/closure event
                start_t = float(self._closed_start_t) if self._closed_start_t is not None else max(0.0, t_sec - self._closed_accum_sec)
                end_t = t_sec
                dur = float(self._closed_accum_sec)

                is_micro = dur >= float(self.microsleep_sec)
                event = BlinkEvent(
                    start_t=start_t,
                    end_t=end_t,
                    duration_sec=dur,
                    is_microsleep=is_micro,
                )

                self.last_blink_duration_sec = dur
                self.last_microsleep = is_micro
                self.recent_blinks_sec.append(dur)

                if is_micro:
                    self.microsleep_times.append(end_t)

                # reset closure tracking
                self._in_closed = False
                self._closed_accum_sec = 0.0
                self._closed_start_t = None

        # prune microsleep times to window
        while self.microsleep_times and (t_sec - self.microsleep_times[0]) > float(self.microsleep_window_sec):
            self.microsleep_times.popleft()

        return event, len(self.microsleep_times)

    def last_blink_ms(self) -> Optional[float]:
        if self.last_blink_duration_sec is None:
            return None
        return 1000.0 * float(self.last_blink_duration_sec)

    def max_blink_ms_recent(self) -> Optional[float]:
        if not self.recent_blinks_sec:
            return None
        return 1000.0 * float(max(self.recent_blinks_sec))