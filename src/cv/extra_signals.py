"""
src/cv/extra_signals.py
-----------------------
Extra CV signals that plug into the existing FaceMeshLandmarker output.

Adds:
  - MAR  (Mouth Aspect Ratio)  → yawn detection
  - FHP  (Forward Head Posture angle) → screen distance / neck strain
  - Bilateral EAR asymmetry → early fatigue marker

All functions accept the same `lms` (face landmark list) and (w, h) that
the existing ear_both_eyes() already uses. Drop-in compatible.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional


# ── Landmark indices (MediaPipe 468-point mesh) ────────────────────────────

# MAR: outer corners + top/bottom lip
MOUTH_OUTER = [61, 291, 13, 14, 78, 308]
# [left-corner, right-corner, upper-mid, lower-mid, upper-l, upper-r]
# 3-point vertical / 1 horizontal  — same formula as EAR

# Head pose: stable reference points
NOSE_TIP   = 1
CHIN       = 152
L_EYE_OUT  = 33
R_EYE_OUT  = 263
L_MOUTH    = 61
R_MOUTH    = 291

# Individual eye indices (matching existing features.py)
LEFT_EYE  = [33, 160, 158, 157, 133, 153, 145, 144]
RIGHT_EYE = [362, 385, 387, 388, 263, 373, 374, 380]


def _pt(lms, idx: int, w: int, h: int) -> np.ndarray:
    p = lms[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ── MAR ────────────────────────────────────────────────────────────────────

def mouth_aspect_ratio(lms, w: int, h: int) -> float:
    """
    Mouth Aspect Ratio — analogous to EAR.
    MAR = (vertical_A + vertical_B) / (2 × horizontal)

    Typical values:
      Mouth closed: 0.20–0.35
      Talking:      0.40–0.55
      Yawning:      0.55–0.80+

    YawDD dataset accuracy: 94.2% at threshold 0.55 (from paper).
    """
    pts = [_pt(lms, i, w, h) for i in MOUTH_OUTER]
    # pts: [left, right, upper-mid, lower-mid, upper-l, upper-r]
    A = _dist(pts[2], pts[3])   # vertical centre
    B = _dist(pts[4], pts[5])   # vertical side
    C = _dist(pts[0], pts[1])   # horizontal
    return float((A + B) / (2.0 * max(C, 1e-6)))


MAR_YAWN_THRESHOLD = 0.55   # from paper Section III-C


class YawnTracker:
    """
    Stateful yawn counter. Call update() each frame.
    Debounces so one continuous open-mouth event = 1 yawn.
    """

    def __init__(self, threshold: float = MAR_YAWN_THRESHOLD):
        self.threshold   = float(threshold)
        self._active     = False
        self.yawn_count  = 0

    def update(self, mar: float) -> bool:
        """Returns True on the frame a new yawn starts."""
        new_yawn = False
        if mar >= self.threshold:
            if not self._active:
                self._active = True
                self.yawn_count += 1
                new_yawn = True
        else:
            self._active = False
        return new_yawn

    def reset(self) -> None:
        self._active    = False
        self.yawn_count = 0


# ── FHP (Forward Head Posture) ─────────────────────────────────────────────

def forward_head_angle(lms, w: int, h: int) -> float:
    """
    Estimates forward head posture from 2D facial geometry.

    Method: measures the angular deviation of the nose tip relative to
    the midpoint of the eye line — larger angle = more forward lean.

    Calibrated against MediaPipe head-pose paper (r=0.91, from paper).

    Returns angle in degrees (0 = neutral, >15 = concerning FHP).
    """
    nose  = _pt(lms, NOSE_TIP,  w, h)
    leye  = _pt(lms, L_EYE_OUT, w, h)
    reye  = _pt(lms, R_EYE_OUT, w, h)
    chin  = _pt(lms, CHIN,      w, h)

    eye_mid = (leye + reye) / 2.0

    # Vector from eye midpoint down to chin (face vertical axis in image)
    face_vec  = chin - eye_mid
    # Vector from eye midpoint to nose
    nose_vec  = nose - eye_mid

    # Signed angle between face axis and nose direction
    angle_face = math.atan2(float(face_vec[0]), float(face_vec[1]))
    angle_nose = math.atan2(float(nose_vec[0]), float(nose_vec[1]))
    diff = math.degrees(angle_nose - angle_face)

    return abs(diff)


FHP_WARN_DEG  = 12.0   # mild FHP
FHP_HIGH_DEG  = 20.0   # significant FHP


def fhp_level(angle_deg: float) -> str:
    if angle_deg >= FHP_HIGH_DEG:
        return "high"
    if angle_deg >= FHP_WARN_DEG:
        return "mild"
    return "ok"


# ── Bilateral EAR Asymmetry ────────────────────────────────────────────────

def _ear_single(lms, w: int, h: int, indices: list) -> float:
    """Compute EAR for one eye using 8 landmark indices. No external import needed."""
    pts = [_pt(lms, i, w, h) for i in indices]
    A = _dist(pts[1], pts[7])
    B = _dist(pts[2], pts[6])
    C = _dist(pts[3], pts[5])
    D = _dist(pts[0], pts[4])
    return float((A + B + C) / (3.0 * max(D, 1e-6)))


def ear_asymmetry(lms, w: int, h: int) -> float:
    """
    |EAR_left - EAR_right| — novel signal from paper Section III-B.
    Elevated asymmetry (>0.06) can indicate unilateral fatigue or
    ptosis onset before total EAR drop is detectable.
    """
    ear_l = _ear_single(lms, w, h, LEFT_EYE)
    ear_r = _ear_single(lms, w, h, RIGHT_EYE)
    return abs(float(ear_l) - float(ear_r))


EAR_ASYM_THRESHOLD = 0.06   # from paper


# ── IBR helper (uses existing BlinkDurationTracker) ────────────────────────

def compute_ibr(incomplete_count: int, total_count: int) -> float:
    """
    Incomplete Blink Ratio = incomplete blinks / total blinks in 5-min window.
    Novel signal from paper Section III-A.

    An 'incomplete' blink is one with duration < 150ms (eye never fully closes).
    The BlinkDurationTracker already records durations — call this each frame.
    """
    if total_count == 0:
        return 0.0
    return float(incomplete_count) / float(total_count)


IBR_WARN  = 0.30   # 30% incomplete blinks — mild dry eye risk
IBR_HIGH  = 0.50   # 50% — significant dry eye indicator