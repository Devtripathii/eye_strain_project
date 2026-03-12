from __future__ import annotations

"""
recommendations_dynamic  —  metric-driven advice engine
────────────────────────────────────────────────────────
Changes from original:
  1. Break level thresholds aligned with new fuse_risk score range
  2. PERCLOS thresholds match fusion.py (screen-use values)
  3. Blink rate advice widened for short assessment windows
  4. Cleaner bullet composition — no duplicate tips
  5. Reasons appended only when genuinely informative
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Advice:
    title: str
    bullets: List[str]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def recommendations_dynamic(
    level: str,
    score01: float,
    perclos: float,
    blink_rate: float,
    cnn_sleepy: float | None,
    baseline_blink_rate: float | None = None,
    ear_drop_ratio: float | None = None,
    trend_cnn: float | None = None,
) -> Advice:
    """
    Generates contextual recommendations based on current fatigue signals.

    Parameters
    ----------
    level              : "LOW" | "MED" | "HIGH"
    score01            : smoothed risk score 0..1
    perclos            : fraction of time eyes closed 0..1
    blink_rate         : blinks per minute
    cnn_sleepy         : CNN sleepiness probability 0..1 (optional)
    baseline_blink_rate: user's personal baseline (optional)
    ear_drop_ratio     : current EAR / baseline EAR (optional)
    trend_cnn          : CNN trend slope (optional)
    """
    level   = (level or "LOW").upper().strip()
    score01 = _clamp01(score01)

    reasons: list[str] = []
    bullets: list[str] = []

    # ── Collect informative reasons ───────────────────────────────────────────
    if cnn_sleepy is not None:
        cs = _clamp01(cnn_sleepy)
        if cs >= 0.65:
            reasons.append(f"eye-closure model shows strong fatigue ({cs:.0%})")
        elif cs >= 0.40:
            reasons.append(f"eye-closure model shows mild fatigue ({cs:.0%})")

    if perclos >= 0.20:
        reasons.append(f"eyes were closed {perclos:.0%} of the time (high)")
    elif perclos >= 0.12:
        reasons.append(f"eyes were closed {perclos:.0%} of the time (moderate)")

    if baseline_blink_rate is not None and baseline_blink_rate > 0:
        ratio = blink_rate / baseline_blink_rate
        if ratio < 0.60:
            reasons.append(
                f"blink rate dropped to {blink_rate:.0f}/min "
                f"(your baseline is {baseline_blink_rate:.0f}/min)"
            )
        elif ratio > 1.60:
            reasons.append(
                f"blink rate rose to {blink_rate:.0f}/min "
                f"(baseline {baseline_blink_rate:.0f}/min) — possible irritation"
            )
    else:
        if blink_rate < 6:
            reasons.append(f"blink rate is very low ({blink_rate:.0f}/min) — staring detected")
        elif blink_rate < 12:
            reasons.append(f"blink rate is low ({blink_rate:.0f}/min)")
        elif blink_rate > 35:
            reasons.append(f"blink rate is elevated ({blink_rate:.0f}/min) — possible dryness")

    if ear_drop_ratio is not None and ear_drop_ratio < 0.82:
        reasons.append("eye opening noticeably reduced from your baseline")

    if trend_cnn is not None and trend_cnn > 0.03:
        reasons.append("fatigue trend is increasing over this session")

    # ── Determine break level from smoothed score + perclos ───────────────────
    # Aligned with new fuse_risk thresholds
    if score01 >= 0.55 or perclos >= 0.20:
        break_level = 3   # long break
    elif score01 >= 0.35 or perclos >= 0.14:
        break_level = 2   # short break
    elif score01 >= 0.20 or blink_rate < 10:
        break_level = 1   # micro break
    else:
        break_level = 0   # all good

    # ── Build advice ──────────────────────────────────────────────────────────
    if break_level == 3:
        title = "🚨 Take a proper break now (5–10 min)"
        bullets = [
            "Stop screen work — close your laptop or turn away completely.",
            "20-20-20: look at something 20 feet away for 20 seconds, repeat 3×.",
            "Do 10 slow deliberate blinks: close gently, hold 1s, open slowly.",
            "Stand up, stretch neck and shoulders, drink some water.",
        ]
        if perclos >= 0.20:
            bullets.append(
                f"Your eyes were closed {perclos:.0%} of frames — "
                "a strong sign of fatigue building up."
            )

    elif break_level == 2:
        title = "⏳ Short break recommended (2–3 min)"
        bullets = [
            "Look away from the screen for 2 minutes — out a window if possible.",
            "Do 6–8 slow blinks to refresh the tear film.",
            "Check your posture: screen slightly below eye level, arm's-length away.",
            "Reduce glare: adjust blinds or screen angle.",
        ]
        if blink_rate < 12:
            bullets.append(
                f"Blink rate is {blink_rate:.0f}/min — try blinking more "
                "deliberately while reading."
            )

    elif break_level == 1:
        title = "✅ Micro-break suggested (20–30 sec)"
        bullets = [
            "Look away for 20 seconds — relax your focus to the distance.",
            "Blink softly 5–6 times to rewet the eye surface.",
            "Drop your shoulders and unclench your jaw.",
        ]

    else:
        title = "🟢 Eyes looking good — keep it up"
        bullets = [
            "Good blink rate and eye openness — stay relaxed.",
            "Remember 20-20-20 every 20 minutes as a habit.",
            "Keep screen slightly below eye level to reduce lid strain.",
        ]

    # ── Targeted extra tips (max 2, no duplicates) ────────────────────────────
    extras: list[str] = []

    if cnn_sleepy is not None and cnn_sleepy >= 0.65 and break_level < 3:
        extras.append(
            "The eye-closure model is flagging fatigue — "
            "consider a longer break than suggested."
        )

    if ear_drop_ratio is not None and ear_drop_ratio < 0.82 and not any("reduced" in b for b in bullets):
        extras.append(
            "Your eye opening has reduced since calibration — "
            "this often means squinting from glare or font size."
        )

    if trend_cnn is not None and trend_cnn > 0.03 and break_level < 2:
        extras.append("Fatigue is trending upward — plan a break in the next few minutes.")

    bullets.extend(extras[:2])

    # ── Append compact reason summary ─────────────────────────────────────────
    if reasons and break_level >= 1:
        summary = "Signals: " + "; ".join(reasons[:3]) + "."
        bullets.append(summary)

    return Advice(title=title, bullets=bullets)