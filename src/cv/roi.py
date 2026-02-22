from __future__ import annotations

import numpy as np

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _lm_xy(lms, idx, w, h):
    p = lms[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def crop_eye_roi(frame_bgr, lms, side: str = "left", pad: float = 0.55):
    if frame_bgr is None or lms is None:
        return None

    h, w = frame_bgr.shape[:2]
    eye_idx = LEFT_EYE if side.lower().startswith("l") else RIGHT_EYE

    pts = np.array([_lm_xy(lms, i, w, h) for i in eye_idx], dtype=np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    px = bw * float(pad)
    py = bh * float(pad)

    X1 = int(max(0, x1 - px))
    Y1 = int(max(0, y1 - py))
    X2 = int(min(w - 1, x2 + px))
    Y2 = int(min(h - 1, y2 + py))

    if X2 <= X1 or Y2 <= Y1:
        return None

    return frame_bgr[Y1:Y2, X1:X2].copy()


def crop_both_eyes(frame_bgr, lms, pad: float = 0.55, min_size: int = 24):
    left = crop_eye_roi(frame_bgr, lms, "left", pad=pad)
    right = crop_eye_roi(frame_bgr, lms, "right", pad=pad)

    if left is None and right is None:
        return None

    if left is None:
        roi = right
    elif right is None:
        roi = left
    else:
        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]
        H = min(lh, rh)
        if H <= 0:
            return None
        roi = np.concatenate([left[:H], right[:H]], axis=1)

    if roi is None:
        return None
    if roi.shape[0] < int(min_size) or roi.shape[1] < int(min_size):
        return None

    return roi