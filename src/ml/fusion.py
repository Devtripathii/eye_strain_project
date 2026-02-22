def clamp(x, lo=0.0, hi=1.0):
    if x is None:
        return None
    return max(lo, min(hi, float(x)))


def fuse_risk(perclos: float, blink_rate: float, cnn_prob: float | None, lstm_prob: float | None):
    """
    cnn_prob: probability of sleepy/closed (0..1)
    lstm_prob: optional fatigue trend probability (0..1)
    """
    reasons = []
    score = 0.0

    # 1) CNN (your strongest signal now)
    if cnn_prob is not None:
        cp = clamp(cnn_prob)
        score += 0.55 * cp
        reasons.append(f"CNN sleepy probability={cp:.2f}")

    # 2) LSTM / temporal (optional)
    if lstm_prob is not None:
        lp = clamp(lstm_prob)
        score += 0.20 * lp
        reasons.append(f"Temporal fatigue probability={lp:.2f}")

    # 3) PERCLOS (research-grade indicator)
    if perclos >= 0.25:
        score += 0.15
        reasons.append("High PERCLOS.")
    elif perclos >= 0.15:
        score += 0.08
        reasons.append("Moderate PERCLOS.")

    # 4) Blink rate (low blink = dry eye risk)
    if blink_rate < 8:
        score += 0.10
        reasons.append("Low blink rate.")
    elif blink_rate > 30:
        score += 0.05
        reasons.append("High blink rate.")

    score = max(0.0, min(1.0, score))

    if score >= 0.70:
        level = "HIGH"
    elif score >= 0.40:
        level = "MED"
    else:
        level = "LOW"

    return level, score, reasons