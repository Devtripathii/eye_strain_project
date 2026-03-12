from __future__ import annotations

"""
fuse_risk  —  multi-signal fatigue risk fusion
"""

_score_ewma: float | None = None
_EWMA_ALPHA: float = 0.30      # FIX: was 0.20 — too slow for 60s sessions


def reset_fusion_state() -> None:
    global _score_ewma
    _score_ewma = None


def clamp(x, lo: float = 0.0, hi: float = 1.0) -> float | None:
    if x is None:
        return None
    return max(lo, min(hi, float(x)))


def fuse_risk(
    perclos: float,
    blink_rate: float,
    cnn_prob: float | None,
    lstm_prob: float | None,
) -> tuple[str, float, list[str]]:
    """
    Returns (level, smoothed_score, reasons).

    Weights (no CNN):  PERCLOS 0.50 + blink 0.25 = 0.75 max
    Weights (with CNN): adds up to 0.35 → total 1.10 → clamped to 1.0

    PERCLOS thresholds (screen use):
      ≥ 0.20 → high   ≥ 0.12 → moderate   ≥ 0.07 → slight

    Blink rate thresholds:
      < 6/min → very low    < 12/min → low    > 35/min → irritation

    Risk levels:
      HIGH ≥ 0.55    MED ≥ 0.28    LOW < 0.28
    """
    global _score_ewma

    reasons:   list[str] = []
    raw_score: float     = 0.0

    # ── 1. PERCLOS ────────────────────────────────────────────────────────────
    p = float(clamp(perclos) or 0.0)
    if p >= 0.20:
        contrib = 0.50
        reasons.append(f"High eye closure (PERCLOS={p:.2f}).")
    elif p >= 0.12:
        contrib = 0.28
        reasons.append(f"Moderate eye closure (PERCLOS={p:.2f}).")
    elif p >= 0.07:
        contrib = 0.12
        reasons.append(f"Slight eye closure (PERCLOS={p:.2f}).")
    else:
        contrib = 0.0
    raw_score += contrib

    # ── 2. Blink rate ─────────────────────────────────────────────────────────
    br = float(blink_rate)
    if br < 6:
        raw_score += 0.25
        reasons.append(f"Very low blink rate ({br:.0f}/min) — staring detected.")
    elif br < 12:
        raw_score += 0.12
        reasons.append(f"Low blink rate ({br:.0f}/min).")
    elif br > 35:
        raw_score += 0.08
        reasons.append(f"High blink rate ({br:.0f}/min) — possible irritation.")

    # ── 3. CNN ────────────────────────────────────────────────────────────────
    if cnn_prob is not None:
        cp = float(clamp(cnn_prob) or 0.0)
        if cp >= 0.65:
            raw_score += 0.35
            reasons.append(f"CNN: strong eye-closure signal ({cp:.2f}).")
        elif cp >= 0.45:
            raw_score += 0.20
            reasons.append(f"CNN: moderate eye-closure ({cp:.2f}).")
        elif cp >= 0.30:
            raw_score += 0.08
            reasons.append(f"CNN: mild eye-closure ({cp:.2f}).")

    # ── 4. LSTM (optional) ────────────────────────────────────────────────────
    if lstm_prob is not None:
        lp = float(clamp(lstm_prob) or 0.0)
        if lp >= 0.50:
            raw_score += 0.15
            reasons.append(f"Temporal fatigue signal ({lp:.2f}).")

    # ── 5. Clamp + EWMA smoothing ─────────────────────────────────────────────
    raw_score = max(0.0, min(1.0, raw_score))

    if _score_ewma is None:
        _score_ewma = raw_score
    else:
        _score_ewma = _EWMA_ALPHA * raw_score + (1.0 - _EWMA_ALPHA) * _score_ewma

    smoothed = float(_score_ewma)

    # ── 6. Level ──────────────────────────────────────────────────────────────
    if smoothed >= 0.55:
        level = "HIGH"
    elif smoothed >= 0.28:
        level = "MED"
    else:
        level = "LOW"

    return level, smoothed, reasons