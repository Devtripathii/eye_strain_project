import numpy as np
from collections import deque

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _pt(lms, idx, w, h):
    p = lms[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def eye_aspect_ratio(lms, w, h, eye_idx):
    p1 = _pt(lms, eye_idx[0], w, h)
    p2 = _pt(lms, eye_idx[1], w, h)
    p3 = _pt(lms, eye_idx[2], w, h)
    p4 = _pt(lms, eye_idx[3], w, h)
    p5 = _pt(lms, eye_idx[4], w, h)
    p6 = _pt(lms, eye_idx[5], w, h)

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    hdist = np.linalg.norm(p1 - p4) + 1e-6
    return (v1 + v2) / (2.0 * hdist)


def ear_both_eyes(lms, w, h):
    ear_l = eye_aspect_ratio(lms, w, h, LEFT_EYE)
    ear_r = eye_aspect_ratio(lms, w, h, RIGHT_EYE)
    return float((ear_l + ear_r) / 2.0)


class BlinkCounter:
    def __init__(self, ear_threshold: float):
        self.ear_threshold = float(ear_threshold)
        self.last_eye_closed = False
        self.blink_count = 0

    def update(self, ear: float):
        eye_closed = float(ear) < self.ear_threshold

        blink_occurred = False
        if self.last_eye_closed and not eye_closed:
            self.blink_count += 1
            blink_occurred = True

        self.last_eye_closed = eye_closed
        return self.blink_count, eye_closed, blink_occurred


class FatigueMetrics:
    """
    Computes:
    - PERCLOS over a sliding window (fraction of frames eyes are closed)
    - Blink rate (blinks per minute) over last 60 seconds

    Backward compatible init:
      FatigueMetrics(fps, window_seconds=30)
      FatigueMetrics(fps=20, window_seconds=30)
      FatigueMetrics(window_seconds=30)  -> defaults fps=20
    """

    def __init__(self, fps: int = 20, window_seconds: int = 30, **kwargs):
        # Accept older code paths that passed different keyword names
        # e.g. FatigueMetrics(frame_rate=20) or FatigueMetrics(fps=20)
        if "frame_rate" in kwargs and fps == 20:
            fps = int(kwargs["frame_rate"])

        self.fps = int(fps)
        self.window_seconds = int(window_seconds)

        self.window_frames = max(1, self.fps * self.window_seconds)
        self.closed_hist = deque(maxlen=self.window_frames)
        self.blink_times = deque(maxlen=600)

    def update(self, eye_closed: bool, t_sec: float, blink_occurred: bool):
        self.closed_hist.append(1 if eye_closed else 0)

        if blink_occurred:
            self.blink_times.append(float(t_sec))

        perclos = sum(self.closed_hist) / max(1, len(self.closed_hist))
        recent_blinks = [t for t in self.blink_times if (t_sec - t) <= 60.0]
        blink_rate = float(len(recent_blinks))  # blinks/min

        return float(perclos), float(blink_rate)