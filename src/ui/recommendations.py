from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


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
    Metric-driven recommendations (more variety, more precision).

    Parameters:
      score01: 0..1 risk score
      perclos: 0..1
      blink_rate: blinks/min
      cnn_sleepy: 0..1 (optional)
      baseline_blink_rate: user baseline (optional)
      ear_drop_ratio: ear / baseline_ear (optional), lower => more closure
      trend_cnn: slope-like (-1..+1) or small float indicating increasing fatigue (optional)
    """
    level = (level or "").upper().strip()
    score01 = _clamp01(score01)

    bullets: List[str] = []
    reasons: List[str] = []

    # --- Interpret signals ---
    if cnn_sleepy is not None:
        cs = _clamp01(cnn_sleepy)
        if cs >= 0.70:
            reasons.append(f"CNN indicates frequent eye-closure (sleepy={cs:.2f}).")
        elif cs >= 0.45:
            reasons.append(f"CNN indicates mild eye-closure (sleepy={cs:.2f}).")

    if perclos >= 0.25:
        reasons.append(f"PERCLOS is high ({perclos:.2f}).")
    elif perclos >= 0.15:
        reasons.append(f"PERCLOS is moderate ({perclos:.2f}).")

    # Blink baseline comparison
    if baseline_blink_rate is not None and baseline_blink_rate > 0:
        ratio = blink_rate / baseline_blink_rate
        if ratio < 0.65:
            reasons.append(f"Blink rate dropped vs your baseline ({blink_rate:.0f}/min vs {baseline_blink_rate:.0f}/min).")
        elif ratio > 1.50:
            reasons.append(f"Blink rate is elevated vs baseline ({blink_rate:.0f}/min). Possible irritation.")
    else:
        if blink_rate < 8:
            reasons.append(f"Blink rate is low ({blink_rate:.0f}/min).")
        elif blink_rate > 30:
            reasons.append(f"Blink rate is high ({blink_rate:.0f}/min). Possible dryness/irritation.")

    if ear_drop_ratio is not None:
        if ear_drop_ratio < 0.85:
            reasons.append("Your EAR dropped noticeably from baseline (more squint/closure).")

    if trend_cnn is not None:
        if trend_cnn > 0.03:
            reasons.append("Fatigue trend is increasing (CNN trend rising).")

    # --- Decide break type ---
    # 0 = none, 1 = micro, 2 = short, 3 = long
    break_level = 0
    if score01 >= 0.75 or perclos >= 0.30:
        break_level = 3
    elif score01 >= 0.55 or perclos >= 0.22:
        break_level = 2
    elif score01 >= 0.35 or blink_rate < 10:
        break_level = 1

    # --- Compose advice bullets ---
    if break_level == 3:
        title = "🚨 Break recommended NOW (5–10 minutes)"
        bullets += [
            "Look 20 feet away for 20–30 seconds, repeat 3–5 times.",
            "Do 10 slow blinks: close gently, pause 1 sec, open.",
            "Stand up + stretch neck/shoulders. Hydrate if possible.",
            "Lower brightness or enable night light / warm color filter.",
        ]
    elif break_level == 2:
        title = "⏳ Take a short break soon (2–3 minutes)"
        bullets += [
            "2 minutes off-screen: look far away and relax focus.",
            "Do 6–8 slow blinks to re-wet eyes.",
            "Increase font size slightly to reduce squinting.",
            "Check glare: rotate screen or adjust room lighting.",
        ]
    elif break_level == 1:
        title = "✅ Micro-break suggested (20–30 seconds)"
        bullets += [
            "Look away from screen for 20 seconds.",
            "Blink normally for 10 seconds (don’t force it).",
            "Reposition: screen slightly below eye level, at arm’s length.",
        ]
    else:
        title = "🟢 Keep going (good signs)"
        bullets += [
            "Maintain posture and comfortable screen distance.",
            "Use 20-20-20 every 20 minutes.",
            "Avoid staring; blink naturally.",
        ]

    # Add targeted tip
    if blink_rate < 10:
        bullets.append("Tip: low blink rate often happens during reading—add intentional soft blinks occasionally.")
    if perclos >= 0.25:
        bullets.append("Tip: frequent closure can indicate fatigue—reduce continuous screen time.")
    if cnn_sleepy is not None and cnn_sleepy >= 0.70:
        bullets.append("Tip: your CNN eye-closure signal is strong—treat this like real fatigue.")

    # Add “why” at the end (short)
    if reasons:
        bullets.append("Why: " + " ".join(reasons[:3]))

    return Advice(title=title, bullets=bullets)