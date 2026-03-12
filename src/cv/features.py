from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional

# ── Eye landmark indices (valid for both 468 and 478 point meshes) ────────────
LEFT_EYE  = [33, 160, 158, 157, 133, 153, 145, 144]
RIGHT_EYE = [362, 385, 387, 388, 263, 373, 374, 380]

_EAR_ALPHA = 0.25


def _pt(lms, idx: int, w: int, h: int) -> np.ndarray:
    p = lms[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def eye_aspect_ratio(lms, w: int, h: int, eye_idx: list) -> float:
    """8-point EAR — 3 vertical distances / horizontal distance."""
    p1 = _pt(lms, eye_idx[0], w, h)
    p2 = _pt(lms, eye_idx[1], w, h)
    p3 = _pt(lms, eye_idx[2], w, h)
    p4 = _pt(lms, eye_idx[3], w, h)
    p5 = _pt(lms, eye_idx[4], w, h)
    p6 = _pt(lms, eye_idx[5], w, h)
    p7 = _pt(lms, eye_idx[6], w, h)
    p8 = _pt(lms, eye_idx[7], w, h)

    v1    = np.linalg.norm(p2 - p8)
    v2    = np.linalg.norm(p3 - p7)
    v3    = np.linalg.norm(p4 - p6)
    hdist = np.linalg.norm(p1 - p5) + 1e-6
    return float((v1 + v2 + v3) / (3.0 * hdist))


def ear_both_eyes(lms, w: int, h: int) -> float:
    ear_l = eye_aspect_ratio(lms, w, h, LEFT_EYE)
    ear_r = eye_aspect_ratio(lms, w, h, RIGHT_EYE)
    return float((ear_l + ear_r) / 2.0)


# ── EARSmoother ───────────────────────────────────────────────────────────────

class EARSmoother:
    """
    EWMA smoothing on raw EAR values.
    USE FOR: PERCLOS and display only.
    DO NOT USE FOR: BlinkCounter — smoothing hides fast blinks.
    """

    def __init__(self, alpha: float = _EAR_ALPHA):
        self.alpha   = float(alpha)
        self._value: Optional[float] = None

    def update(self, ear: float) -> float:
        if self._value is None:
            self._value = float(ear)
        else:
            self._value = self.alpha * float(ear) + (1.0 - self.alpha) * self._value
        return float(self._value)

    def reset(self):
        self._value = None


# ── BlinkCounter ──────────────────────────────────────────────────────────────

class BlinkCounter:
    """
    Blink counter with hysteresis. REQUIRES RAW (unsmoothed) EAR.

    At 15fps a fast blink = 1 frame (67ms). Smoothed EAR at that frame:
      0.25 * 0.115 + 0.75 * 0.265 = 0.228  →  never crosses threshold ~0.180
    Raw EAR drops to 0.10-0.13 and crosses cleanly. Hysteresis=0.03 on
    raw EAR prevents double-counting without missing any real blinks.
    """

    def __init__(self, ear_threshold: float, hysteresis: float = 0.03):
        self.threshold_close = float(ear_threshold)
        self.threshold_open  = float(ear_threshold) + float(hysteresis)
        self.last_eye_closed = False
        self.blink_count     = 0

    def update(self, ear_raw: float):
        """Pass raw (unsmoothed) EAR only."""
        ear = float(ear_raw)

        if self.last_eye_closed:
            eye_closed = ear < self.threshold_open
        else:
            eye_closed = ear < self.threshold_close

        blink_occurred = False
        if self.last_eye_closed and not eye_closed:
            self.blink_count += 1
            blink_occurred    = True

        self.last_eye_closed = eye_closed
        return self.blink_count, eye_closed, blink_occurred


# ── FatigueMetrics ────────────────────────────────────────────────────────────

class FatigueMetrics:
    """
    PERCLOS + blink rate per minute over sliding windows.
    Blink rate uses 30s rolling window so it populates within the 50s assessment.
    """

    def __init__(self, fps: int = 20, window_seconds: int = 30, **kwargs):
        if "frame_rate" in kwargs and fps == 20:
            fps = int(kwargs["frame_rate"])

        self.fps             = int(fps)
        self.window_seconds  = int(window_seconds)
        self.window_frames   = max(1, self.fps * self.window_seconds)
        self.closed_hist: deque = deque(maxlen=self.window_frames)
        self.blink_times: deque = deque(maxlen=600)
        self.BLINK_WINDOW_SEC   = 30.0

    def update(self, eye_closed: bool, t_sec: float, blink_occurred: bool):
        self.closed_hist.append(1 if eye_closed else 0)
        if blink_occurred:
            self.blink_times.append(float(t_sec))

        perclos        = sum(self.closed_hist) / max(1, len(self.closed_hist))
        window_start   = max(0.0, t_sec - self.BLINK_WINDOW_SEC)
        recent         = [t for t in self.blink_times if t >= window_start]
        # FIX: use actual window duration not total elapsed as denominator.
        # At t=45s with 30s window: old used 45/60=0.75min → 33% undercount.
        # New: min(45,30)/60=0.5min → correct bpm once window is full.
        window_elapsed = min(t_sec, self.BLINK_WINDOW_SEC)
        elapsed_min    = max(window_elapsed, 1.0) / 60.0
        blink_rate_bpm = float(len(recent)) / elapsed_min

        return float(perclos), float(blink_rate_bpm)