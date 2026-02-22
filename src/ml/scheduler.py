from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BreakDecision:
    action: str  # "NONE" | "MICROBREAK" | "BREAK_NOW"
    minutes: int
    reason: str


def decide_break(
    risk_level: str,
    risk_score: float,
    csi: float,
    seconds_since_break: float,
    blink_rate: float,
    perclos: float,
) -> BreakDecision:
    """
    Deterministic, explainable break scheduling policy.

    - Uses current risk + accumulated strain (CSI) + time since break.
    - Returns an action + recommended minutes.
    """
    rl = (risk_level or "").upper().strip()
    rs = float(max(0.0, min(1.0, risk_score)))
    ssb = float(max(0.0, seconds_since_break))

    # 1) Hard triggers
    if rl == "HIGH" or rs >= 0.72:
        return BreakDecision("BREAK_NOW", 8, "High instantaneous risk.")

    # CSI overload
    if csi >= 220:
        return BreakDecision("BREAK_NOW", 10, "High cumulative strain (CSI overload).")

    # 2) Medium triggers
    if rl == "MED" or (0.40 <= rs < 0.72):
        if csi >= 120 or ssb >= 30 * 60:
            return BreakDecision("BREAK_NOW", 5, "Moderate risk with accumulated strain / long session.")
        return BreakDecision("MICROBREAK", 2, "Moderate risk: take a short reset.")

    # 3) Preventive schedule (LOW risk)
    # Recommend preventive break if working too long, or blink-rate looks low + perclos elevated
    if ssb >= 45 * 60:
        return BreakDecision("MICROBREAK", 2, "Long session without a break.")
    if blink_rate < 8 and perclos >= 0.15:
        return BreakDecision("MICROBREAK", 2, "Low blink rate + moderate PERCLOS.")
    if ssb >= 20 * 60:
        return BreakDecision("MICROBREAK", 1, "Preventive microbreak (20-minute rule).")

    return BreakDecision("NONE", 0, "No break needed now.")


def decision_to_ui(dec: BreakDecision) -> Dict[str, Any]:
    if dec.action == "BREAK_NOW":
        return {"badge": "🟥 BREAK NOW", "text": f"Take {dec.minutes} min break. {dec.reason}"}
    if dec.action == "MICROBREAK":
        return {"badge": "🟧 MICROBREAK", "text": f"Take {dec.minutes} min microbreak. {dec.reason}"}
    return {"badge": "🟩 OK", "text": dec.reason}