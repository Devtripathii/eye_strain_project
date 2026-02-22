# src/core/timebase.py
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class TimebaseTick:
    dt: float
    fps: float
    timestamp: float


class Timebase:
    """
    Tracks real dt and fps from the processing loop.

    - dt is clamped to avoid huge spikes when Streamlit pauses / tab switches.
    - fps is computed as a rolling average.
    """

    def __init__(
        self,
        fps_window: int = 30,
        dt_min: float = 1e-4,
        dt_max: float = 0.25,
    ):
        self.fps_window = int(max(5, fps_window))
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        self._last_t: Optional[float] = None
        self._dt_hist: Deque[float] = deque(maxlen=self.fps_window)

    def reset(self) -> None:
        self._last_t = None
        self._dt_hist.clear()

    def tick(self) -> TimebaseTick:
        now = time.perf_counter()

        if self._last_t is None:
            self._last_t = now
            # First tick: assume a neutral dt, avoid division issues.
            dt = 1.0 / 20.0
        else:
            dt = now - self._last_t
            self._last_t = now

        # Clamp dt to avoid huge pauses skewing FPS and accumulators.
        if dt < self.dt_min:
            dt = self.dt_min
        elif dt > self.dt_max:
            dt = self.dt_max

        self._dt_hist.append(dt)

        avg_dt = sum(self._dt_hist) / max(1, len(self._dt_hist))
        fps = (1.0 / avg_dt) if avg_dt > 0 else 0.0

        return TimebaseTick(dt=dt, fps=fps, timestamp=time.time())