from __future__ import annotations

"""
tests/test_features.py
----------------------
Regression tests for the three bugs fixed in session 7.
Run with:  pytest tests/test_features.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv.features import BlinkCounter, EARSmoother, FatigueMetrics


# ── Test 1: BlinkCounter detects a 1-frame blink (Bug 1 regression) ──────────

def test_blink_counter_detects_one_frame_blink():
    """
    At 15fps a natural blink lasts ~1 frame (67ms).
    Raw EAR: 0.265 → 0.110 → 0.265
    BlinkCounter must count exactly 1 blink.
    """
    bc = BlinkCounter(ear_threshold=0.18, hysteresis=0.01)
    frames = [0.265] * 5 + [0.110] + [0.265] * 5
    for ear in frames:
        bc.update(ear)
    assert bc.blink_count == 1, f"Expected 1 blink, got {bc.blink_count}"


def test_blink_counter_detects_multiple_blinks():
    """Three separated blinks must all be counted."""
    bc = BlinkCounter(ear_threshold=0.18, hysteresis=0.01)
    blink = [0.265, 0.265, 0.110, 0.265, 0.265]
    frames = blink * 3
    for ear in frames:
        bc.update(ear)
    assert bc.blink_count == 3, f"Expected 3 blinks, got {bc.blink_count}"


def test_blink_counter_ignores_noise_above_threshold():
    """Small fluctuations above threshold must not trigger false blinks."""
    bc = BlinkCounter(ear_threshold=0.18, hysteresis=0.01)
    # EAR jitters between 0.25 and 0.22 — never crosses 0.18
    frames = [0.25, 0.22, 0.24, 0.23, 0.25, 0.22, 0.24]
    for ear in frames:
        bc.update(ear)
    assert bc.blink_count == 0, f"Expected 0 blinks, got {bc.blink_count}"


# ── Test 2: Smoothed EAR misses the same blink (proves why fix was needed) ───

def test_smoothed_ear_misses_one_frame_blink():
    """
    EWMA-smoothed EAR (alpha=0.25) never crosses threshold=0.18 during
    a 1-frame blink. This proves the root cause of the session 7 bug.
    If this test ever FAILS it means the smoother somehow caught it —
    which would contradict the mathematical proof.
    """
    alpha = 0.25
    smooth = 0.265
    raw_frames = [0.265] * 5 + [0.110] + [0.265] * 5
    smoothed = []
    for r in raw_frames:
        smooth = alpha * r + (1 - alpha) * smooth
        smoothed.append(smooth)

    bc = BlinkCounter(ear_threshold=0.18, hysteresis=0.01)
    for s in smoothed:
        bc.update(s)

    # Smoothed EAR at blink frame: 0.25*0.110 + 0.75*0.265 = 0.2263 > 0.18
    assert bc.blink_count == 0, (
        f"Smoothed EAR should NOT detect the blink, but got {bc.blink_count}. "
        "Check that this test is using smoothed values."
    )


# ── Test 3: FatigueMetrics blink rate uses window time not session time ───────

def test_blink_rate_uses_window_not_session_time():
    """
    Bug: elapsed_min used total session time (t_sec / 60).
    Fix: elapsed_min uses min(t_sec, BLINK_WINDOW_SEC) / 60.

    Setup: 5 blinks recorded in last 30s, queried at t=50s.
    - Buggy:  5 / (50/60) = 6.0 bpm  (wrong — deflated)
    - Fixed:  5 / (30/60) = 10.0 bpm (correct)
    """
    fm = FatigueMetrics(fps=20, window_seconds=30)

    # Record 5 blinks spread across t=20..40s (all within the 30s window at t=50)
    blink_times = [20.0, 25.0, 30.0, 35.0, 40.0]
    t = 0.0
    perclos, rate = 0.0, 0.0
    while t <= 50.0:
        blink_now = any(abs(t - bt) < 0.1 for bt in blink_times)
        perclos, rate = fm.update(eye_closed=False, t_sec=t, blink_occurred=blink_now)
        t += 0.5

    # At t=50s, 5 blinks in last 30s = 10/min
    assert rate > 8.0, (
        f"Blink rate should be ~10 bpm (5 blinks / 0.5 min), got {rate:.2f}. "
        "Likely still using full session time instead of window time."
    )
    assert rate < 12.0, f"Blink rate {rate:.2f} seems too high — check window logic."


# ── Test 4: EARSmoother reset clears state ────────────────────────────────────

def test_ear_smoother_reset():
    """After reset(), next value is taken as-is (no memory of prior state)."""
    s = EARSmoother(alpha=0.25)
    s.update(0.30)
    s.update(0.29)
    s.reset()
    result = s.update(0.20)
    assert result == 0.20, f"After reset, first update should return raw value, got {result}"


# ── Test 5: BlinkCounter hysteresis prevents re-trigger on slow rise ──────────

def test_hysteresis_prevents_double_count():
    """
    Eye closes (drops below threshold_close=0.18),
    then rises slowly to 0.185 (below threshold_open=0.19),
    then rises to 0.20 (above threshold_open).
    Must count exactly 1 blink, not 2.
    """
    bc = BlinkCounter(ear_threshold=0.18, hysteresis=0.01)
    # threshold_close=0.18, threshold_open=0.19
    frames = [
        0.265,  # open
        0.110,  # closed (below 0.18) — eye closing
        0.185,  # still closed (below threshold_open 0.19)
        0.200,  # open again (above 0.19) — blink counted here
        0.265,  # open
    ]
    for ear in frames:
        bc.update(ear)
    assert bc.blink_count == 1, f"Expected 1 blink, got {bc.blink_count}"