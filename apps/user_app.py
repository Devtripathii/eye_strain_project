
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Any

import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

import config

from src.core.timebase import Timebase
from src.cv.landmarks import FaceMeshLandmarker
from src.cv.features import ear_both_eyes, BlinkCounter, FatigueMetrics, EARSmoother
from src.cv.roi import crop_both_eyes
from src.ml.fusion import fuse_risk, reset_fusion_state
from src.ml.baseline import load_user_baseline, save_user_baseline
from src.ml.torch_cnn import TorchEyeCnn
from src.ui.recommendations import recommendations_dynamic
from src.scheduler.fatigue_load import FatigueLoad, FatigueLoadParams
from src.scheduler.break_scheduler import BreakScheduler, BreakThresholds, BreakDurations
from src.signals.blink_duration import BlinkDurationTracker


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EyeGuard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = config.APP_VERSION

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:      #0a0f1e;
    --charcoal:  #111827;
    --surface:   #161f35;
    --surface2:  #1e2d4a;
    --border:    #1e3a5f;
    --cyan:      #00d4ff;
    --cyan-dim:  #0099bb;
    --cyan-glow: rgba(0, 212, 255, 0.15);
    --amber:     #f59e0b;
    --red:       #ef4444;
    --green:     #10b981;
    --text:      #e2e8f0;
    --text-dim:  #94a3b8;
    --text-mute: #475569;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--navy) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--charcoal) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.eyeguard-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 6px 0 20px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.eyeguard-logo {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: -1px;
    text-shadow: 0 0 20px var(--cyan-glow);
}
.eyeguard-sub {
    font-size: 11px;
    color: var(--text-mute);
    text-transform: uppercase;
    letter-spacing: 2px;
}
.version-badge {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-mute);
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: auto;
}

.section-heading {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--cyan);
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 12px 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--cyan);
    opacity: 0.6;
}
.metric-card:hover { border-color: var(--cyan-dim); }
.metric-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-mute);
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.metric-value.good   { color: var(--green); }
.metric-value.warn   { color: var(--amber); }
.metric-value.danger { color: var(--red);   }
.metric-unit { font-size: 12px; color: var(--text-mute); margin-top: 4px; }

.comfort-ring-wrap { text-align: center; padding: 16px 0; }
.comfort-score-big {
    font-family: 'Space Mono', monospace;
    font-size: 56px;
    font-weight: 700;
    line-height: 1;
}
.comfort-score-big.good   { color: var(--green); text-shadow: 0 0 30px rgba(16,185,129,0.4); }
.comfort-score-big.warn   { color: var(--amber); text-shadow: 0 0 30px rgba(245,158,11,0.4); }
.comfort-score-big.danger { color: var(--red);   text-shadow: 0 0 30px rgba(239,68,68,0.4); }
.comfort-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-mute);
    margin-top: 6px;
}

.progress-wrap { margin: 16px 0; }
.progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 8px;
}
.progress-track {
    background: var(--surface2);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--cyan-dim), var(--cyan));
    box-shadow: 0 0 12px var(--cyan-glow);
    transition: width 0.4s ease;
}
.progress-fill.calib { background: linear-gradient(90deg, #1e3a5f, var(--cyan-dim)); }

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.status-pill.idle  { background: var(--surface2); color: var(--text-mute); border: 1px solid var(--border); }
.status-pill.calib { background: rgba(0,212,255,0.1); color: var(--cyan); border: 1px solid var(--cyan-dim); }
.status-pill.run   { background: rgba(16,185,129,0.1); color: var(--green); border: 1px solid var(--green); }
.status-pill.done  { background: rgba(16,185,129,0.15); color: var(--green); border: 1px solid var(--green); }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.status-dot.pulse { animation: pulse 1.4s ease-in-out infinite; }
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.7); }
}

.cnn-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    margin: 8px 0;
    width: 100%;
}
.cnn-badge.good   { border-color: var(--green); color: var(--green); }
.cnn-badge.warn   { border-color: var(--amber); color: var(--amber); }
.cnn-badge.danger { border-color: var(--red);   color: var(--red);   }
.cnn-badge.muted  { color: var(--text-mute); }

.break-alert {
    border-radius: 10px;
    padding: 14px 16px;
    margin: 12px 0;
    border-left: 3px solid;
}
.break-alert.work  { background: rgba(16,185,129,0.08); border-color: var(--green); color: var(--green); }
.break-alert.micro { background: rgba(0,212,255,0.08);  border-color: var(--cyan);  color: var(--cyan); }
.break-alert.short { background: rgba(245,158,11,0.08); border-color: var(--amber); color: var(--amber); }
.break-alert.long  { background: rgba(239,68,68,0.08);  border-color: var(--red);   color: var(--red); }
.break-title { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.break-body  { font-size: 12px; opacity: 0.85; }

.focus-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 12px;
}
.focus-card-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--cyan);
    margin-bottom: 10px;
    font-weight: 600;
}
.focus-card-body   { font-size: 15px; color: var(--text); line-height: 1.6; }
.focus-card-prompt { font-size: 12px; color: var(--text-mute); margin-top: 8px; font-style: italic; }

.ear-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin: 12px 0;
}
.ear-cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.ear-cell-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-mute);
    margin-bottom: 4px;
}
.ear-cell-value { font-family: 'Space Mono', monospace; font-size: 20px; color: var(--cyan); }

.fps-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--text-mute);
}

.rec-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin: 12px 0;
}
.rec-title  { font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 10px; }
.rec-bullet {
    display: flex;
    gap: 8px;
    font-size: 13px;
    color: var(--text-dim);
    margin-bottom: 6px;
    line-height: 1.5;
}
.rec-bullet::before { content: '›'; color: var(--cyan); font-weight: 700; flex-shrink: 0; }

.report-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
.report-badge {
    background: rgba(16,185,129,0.15);
    border: 1px solid var(--green);
    color: var(--green);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
}

.history-row {
    display: grid;
    grid-template-columns: 160px 1fr 1fr 1fr 1fr auto;
    gap: 12px;
    align-items: center;
    padding: 12px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 13px;
}
.history-row:hover { border-color: var(--cyan-dim); }
.history-time { color: var(--text-mute); font-family: 'Space Mono', monospace; font-size: 11px; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.pill.low  { background: rgba(16,185,129,0.15); color: var(--green); }
.pill.med  { background: rgba(245,158,11,0.15);  color: var(--amber); }
.pill.high { background: rgba(239,68,68,0.15);   color: var(--red);  }

div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-mute) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
    font-size: 24px !important;
}

button[data-testid="baseButton-primary"] {
    background: var(--cyan) !important;
    color: var(--navy) !important;
    border: none !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    border-radius: 8px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px var(--cyan-glow) !important;
}
button[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 0 30px rgba(0,212,255,0.4) !important;
    transform: translateY(-1px) !important;
}
button[data-testid="baseButton-secondary"] {
    background: var(--surface) !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
button[data-testid="baseButton-secondary"]:hover {
    border-color: var(--cyan-dim) !important;
    color: var(--cyan) !important;
}

div[data-testid="stProgress"] > div {
    background: var(--surface2) !important;
    border-radius: 999px !important;
    height: 6px !important;
}
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--cyan-dim), var(--cyan)) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 10px var(--cyan-glow) !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

hr { border-color: var(--border) !important; opacity: 1 !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_SECONDS  = config.TOTAL_SECONDS
CALIB_SECONDS  = config.CALIB_SECONDS
RUN_SECONDS    = config.RUN_SECONDS
WEBRTC_KEY     = config.WEBRTC_KEY
MIN_FPS_GATE   = 8.0
LOOP_SLEEP_SEC = config.LOOP_SLEEP_SEC

OUT_DIR     = config.OUTPUTS_DIR
SESS_DIR    = config.SESSIONS_DIR
PROFILE_DIR = config.PROFILES_DIR
CNN_PATH    = config.CNN_MODEL_PATH

FOCUS_TEXTS = [
    "Read naturally. Don't force blinking.",
    "Relax your face and shoulders.",
    "Keep the screen slightly below eye level.",
    "Soften your gaze and blink normally.",
]
PROMPTS = [
    "Keep reading naturally.",
    "Relax your jaw and shoulders.",
    "Avoid squinting — adjust brightness if needed.",
    "Blink normally (don't force it).",
]


def focus_text(t: float) -> str:
    return FOCUS_TEXTS[int(t // 12) % len(FOCUS_TEXTS)]


def micro_prompt(t: float) -> str:
    return PROMPTS[int(t // 8) % len(PROMPTS)]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def comfort_score_from_risk(risk01: float) -> int:
    return max(0, min(100, int(round(100.0 * (1.0 - _clamp01(risk01))))))


def comfort_band(score100: int) -> tuple[str, str]:
    if score100 >= 70:
        return ("Good", "good")
    if score100 >= 40:
        return ("Warning", "warn")
    return ("Danger", "danger")


def _break_css_class(mode: str) -> str:
    return {"WORK": "work", "MICRO": "micro", "SHORT": "short", "LONG": "long"}.get(
        (mode or "WORK").upper(), "work")


def _break_label(mode: str) -> str:
    return {
        "WORK":  "✅ Continue working",
        "MICRO": "👁️ Quick micro-break",
        "SHORT": "☕ Short break",
        "LONG":  "🚶 Long break",
    }.get((mode or "WORK").upper(), "✅ Continue")


def _break_body(mode: str, remaining: float) -> str:
    if mode == "WORK":
        return "Eyes looking comfortable. Keep going."
    return f"Time remaining: {remaining:.0f}s"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RuntimeCfg:
    user_name: str
    enable_cnn: bool
    cnn_every_n_frames: int
    ewma_alpha: float
    enable_scheduler: bool
    microsleep_sec: float
    microsleep_window_sec: float
    microsleep_short_count: int
    microsleep_long_count: int
    micro_sec: int
    short_sec: int
    long_sec: int
    min_gap: float
    k_up: float
    k_down: float
    k_leak: float


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class EyeProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock  = threading.Lock()
        self.phase = "IDLE"
        self.cfg: Optional[RuntimeCfg] = None

        self.last: dict[str, Any] = {
            "ok": False, "msg": "Waiting for camera...",
            "event": None, "event_payload": None,
            "t": 0.0, "fps": 0.0, "dt": 0.0,
            "ear": None, "ear_raw": None, "ear_drop_ratio": None,
            "blink_rate": None, "perclos": None,
            "risk_level": None, "risk_score": None,
            "comfort": None, "comfort_band": None, "comfort_css": None,
            "fatigue_load": None,
            "break_mode": "WORK", "break_remaining_sec": 0.0,
            "microsleeps_window": 0,
            "cnn_ewma": None, "cnn_trend": None,
            "baseline_ear": None, "ear_threshold": None,
        }

        self.timebase     = Timebase(fps_window=30, dt_max=0.25)
        self.fm           = FaceMeshLandmarker(min_det=0.50, min_track=0.50, refine=False)
        self.ear_smoother = EARSmoother(alpha=0.25)
        self._cnn_skip_counter = 0

        self._calib_start_ts: Optional[float] = None
        self._run_start_ts:   Optional[float] = None
        self._calib_ears: list[float] = []
        self._rows:       list[dict]  = []

        self.blink        = None
        self.fatigue      = None
        self.fatigue_load = None
        self.scheduler    = None
        self.blink_dur    = None
        self.cnn          = None
        self.cnn_ewma: Optional[float] = None
        self.cnn_hist   = deque(maxlen=25)
        self.last_roi   = None

    def set_cfg(self, cfg: RuntimeCfg):
        with self.lock:
            self.cfg = cfg

    def set_phase(self, phase: str):
        phase = (phase or "IDLE").upper()
        if phase not in ("IDLE", "CALIB", "RUN", "DONE"):
            phase = "IDLE"
        with self.lock:
            if phase == self.phase:
                return
            if phase in ("CALIB", "IDLE"):
                self._calib_start_ts = None
                self._run_start_ts   = None
                self.ear_smoother.reset()
                reset_fusion_state()
            elif phase == "RUN":
                self._run_start_ts = None
            self.phase = phase

    def get_phase(self) -> str:
        with self.lock:
            return self.phase

    def get_last(self) -> dict:
        with self.lock:
            return dict(self.last)

    def pop_event(self) -> tuple[Optional[str], Optional[dict]]:
        with self.lock:
            ev      = self.last.get("event")
            payload = self.last.get("event_payload")
            self.last["event"]         = None
            self.last["event_payload"] = None
            return ev, payload

    def pop_rows(self) -> list[dict]:
        with self.lock:
            rows       = list(self._rows)
            self._rows = []
            return rows

    def _update_last(self, **kwargs):
        with self.lock:
            self.last.update(kwargs)

    def _ensure_cnn(self, enable: bool):
        if not enable or self.cnn is not None:
            return
        if CNN_PATH.exists():
            self.cnn = TorchEyeCnn(CNN_PATH, grayscale=True, device="cpu")

    def _init_runtime(self, ear_threshold: float, cfg: RuntimeCfg):
        hysteresis = config.BLINK_HYSTERESIS
        try:
            self.blink = BlinkCounter(float(ear_threshold), hysteresis=hysteresis)
        except TypeError:
            self.blink = BlinkCounter(float(ear_threshold))

        self.fatigue      = FatigueMetrics(fps=20, window_seconds=30)
        self.fatigue_load = FatigueLoad(FatigueLoadParams(
            k_up=float(cfg.k_up), k_down=float(cfg.k_down),
            k_leak=float(cfg.k_leak), max_dt=0.25,
        ))
        self.scheduler = BreakScheduler(
            thresholds=BreakThresholds(
                micro_lo=0.25, micro_hi=0.45,
                short_hi=0.55, long_hi=0.75,
                sustain_risk_hi=0.40, sustain_seconds=20.0,
                microsleep_window_sec=float(cfg.microsleep_window_sec),
                microsleep_short_break_count=int(cfg.microsleep_short_count),
                microsleep_long_break_count=int(cfg.microsleep_long_count),
                min_gap_between_breaks_sec=float(cfg.min_gap),
            ),
            durations=BreakDurations(
                micro_sec=int(cfg.micro_sec),
                short_sec=int(cfg.short_sec),
                long_sec=int(cfg.long_sec),
            ),
        )
        self.blink_dur = BlinkDurationTracker(
            microsleep_sec=float(cfg.microsleep_sec),
            microsleep_window_sec=float(cfg.microsleep_window_sec),
            max_dt=0.25,
        )

    def recv(self, frame):
        import cv2
        tick    = self.timebase.tick()
        dt      = float(tick.dt)
        fps     = float(tick.fps)
        img_bgr = frame.to_ndarray(format="bgr24")

        with self.lock:
            cfg   = self.cfg
            phase = self.phase

        if cfg is None:
            self._update_last(ok=False, msg="Loading config...", dt=dt, fps=fps)
            return frame

        h_orig, w_orig = img_bgr.shape[:2]
        MAX_H = 480
        if h_orig > MAX_H:
            scale     = MAX_H / h_orig
            w_small   = int(w_orig * scale)
            img_small = cv2.resize(img_bgr, (w_small, MAX_H), interpolation=cv2.INTER_LINEAR)
        else:
            img_small = img_bgr

        lms = self.fm.get_landmarks(img_small)

        if lms is None:
            self._update_last(ok=False, msg="Face not detected. Center your face.",
                              dt=dt, fps=fps)
            return frame

        h, w    = img_small.shape[:2]
        ear_raw = float(ear_both_eyes(lms, w, h))
        ear     = self.ear_smoother.update(ear_raw)

        if phase == "IDLE":
            self._update_last(ok=True, msg="Ready", t=0.0, dt=dt, fps=fps,
                              ear=ear, ear_raw=ear_raw)
            return frame

        if phase == "CALIB":
            if self._calib_start_ts is None:
                self._calib_start_ts = time.time()
                self._calib_ears     = []
                self._rows           = []
                self.cnn_ewma        = None
                self.cnn_hist.clear()
                self._update_last(event=None, event_payload=None,
                                  baseline_ear=None, ear_threshold=None)

            t_cal = time.time() - self._calib_start_ts
            self._calib_ears.append(ear_raw)
            p = min(1.0, t_cal / max(1e-6, CALIB_SECONDS))
            self._update_last(ok=True, msg=f"Calibrating… {int(p*100)}%",
                              t=t_cal, dt=dt, fps=fps, ear=ear, ear_raw=ear_raw)

            if t_cal >= CALIB_SECONDS and len(self._calib_ears) >= 20:
                ears    = sorted(self._calib_ears)
                n       = len(ears)
                trim    = max(1, int(n * config.CALIB_TRIM_PCT))
                trimmed = ears[trim: n - trim]

                baseline  = float(trimmed[len(trimmed) // 2])
                threshold = round(baseline * config.CALIB_THRESHOLD_MULT, 4)

                # FIX 11: headroom check — threshold must sit at least
                # CALIB_HEADROOM_MIN below baseline so partial blinks are caught
                headroom_min = getattr(config, "CALIB_HEADROOM_MIN", 0.06)
                threshold    = min(threshold, baseline - headroom_min)
                threshold    = max(
                    config.CALIB_THRESHOLD_MIN,
                    min(threshold, baseline * config.CALIB_THRESHOLD_MAX_MULT),
                )

                self._init_runtime(threshold, cfg)
                self._update_last(
                    baseline_ear=baseline, ear_threshold=threshold,
                    event="CALIB_DONE",
                    event_payload={"baseline_ear": baseline, "ear_threshold": threshold},
                )
                with self.lock:
                    self.phase           = "RUN"
                    self._run_start_ts   = None
                    self._calib_start_ts = None
            return frame

        if phase == "RUN":
            if self._run_start_ts is None:
                self._run_start_ts = time.time()

            t_run         = time.time() - self._run_start_ts
            baseline_ear  = self.get_last().get("baseline_ear")
            ear_threshold = self.get_last().get("ear_threshold")

            if ear_threshold is None:
                self._update_last(ok=False, msg="Missing calibration.",
                                  t=t_run, dt=dt, fps=fps, ear=ear)
                return frame

            if self.blink is None:
                self._init_runtime(float(ear_threshold), cfg)

            ear_drop_ratio = None
            if baseline_ear and float(baseline_ear) > 0:
                ear_drop_ratio = ear / max(1e-6, float(baseline_ear))

            blink_count, closed, blink_occurred = self.blink.update(ear_raw)

            if t_run < 3.0:
                closed         = False
                blink_occurred = False

            perclos, blink_rate = self.fatigue.update(
                eye_closed=closed, t_sec=t_run, blink_occurred=blink_occurred)

            blink_event, ms_count = self.blink_dur.update(
                closed=bool(closed), dt=dt, t_sec=t_run)
            microsleep_event = bool(blink_event.is_microsleep) if blink_event else False

            self._ensure_cnn(bool(cfg.enable_cnn))
            cnn_prob = None
            if cfg.enable_cnn and self.cnn is not None:
                roi = crop_both_eyes(img_small, lms, pad=0.55)
                if roi is not None:
                    self.last_roi = roi
                self._cnn_skip_counter += 1
                if self.last_roi is not None and (
                        self._cnn_skip_counter % config.CNN_EVERY_N_FRAMES == 0):
                    res = self.cnn.predict_roi_bgr(self.last_roi)
                    if res is not None:
                        cnn_prob = float(res.sleepy_prob)
                        self.cnn_hist.append(cnn_prob)
                        a = float(cfg.ewma_alpha)
                        self.cnn_ewma = (cnn_prob if self.cnn_ewma is None
                                         else a * cnn_prob + (1 - a) * self.cnn_ewma)

            trend_cnn = None
            if len(self.cnn_hist) >= 8:
                trend_cnn = ((self.cnn_hist[-1] - self.cnn_hist[0])
                             / max(1.0, len(self.cnn_hist)))

            level_live, score_live, _ = fuse_risk(
                perclos=perclos, blink_rate=blink_rate,
                cnn_prob=self.cnn_ewma, lstm_prob=None,
            )

            if cfg.enable_scheduler:
                on_break    = self.scheduler.is_on_break()
                F           = self.fatigue_load.update(
                    risk01=score_live, dt=dt, working=not on_break, quality01=1.0)
                sched_state = self.scheduler.update(
                    t_sec=t_run, dt=dt, risk01=score_live,
                    fatigue_load01=F, microsleep_event=microsleep_event)
            else:
                F           = self.fatigue_load.update(
                    risk01=score_live, dt=dt, working=True, quality01=1.0)
                sched_state = self.scheduler.state

            comfort   = comfort_score_from_risk(score_live)
            band, css = comfort_band(comfort)

            with self.lock:
                self._rows.append({
                    "t": round(t_run, 2), "fps": fps,
                    "ear": round(ear, 3), "ear_raw": round(ear_raw, 3),
                    "ear_drop_ratio": ear_drop_ratio,
                    "blink_rate_bpm": blink_rate, "perclos": perclos,
                    "microsleep_event": microsleep_event,
                    "microsleep_count_window": ms_count,
                    "cnn_sleepy_prob": cnn_prob,
                    "cnn_sleepy_ewma": self.cnn_ewma,
                    "cnn_trend": trend_cnn,
                    "risk_level_live": level_live,
                    "risk_score_live": score_live,
                    "comfort_score_100": comfort,
                    "fatigue_load": F,
                    "break_mode": sched_state.mode,
                    "break_remaining_sec": sched_state.remaining_sec,
                })

            self._update_last(
                ok=True, msg="Assessing",
                t=t_run, dt=dt, fps=fps,
                ear=ear, ear_raw=ear_raw,
                ear_drop_ratio=ear_drop_ratio,
                blink_rate=blink_rate, perclos=perclos,
                risk_level=level_live, risk_score=score_live,
                comfort=comfort, comfort_band=band, comfort_css=css,
                fatigue_load=F,
                break_mode=sched_state.mode,
                break_remaining_sec=sched_state.remaining_sec,
                microsleeps_window=ms_count,
                cnn_ewma=self.cnn_ewma, cnn_trend=trend_cnn,
            )

            if t_run >= RUN_SECONDS - 0.5:
                self._update_last(event="DONE",
                                  event_payload={"reason": "duration_reached"})
                with self.lock:
                    self.phase = "DONE"
            return frame

        self._update_last(ok=True, msg="Done", dt=dt, fps=fps, ear=ear)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in [
    ("ui_phase",                  "IDLE"),
    ("page",                      "Assessment"),
    ("pending_start",             False),
    ("run_cfg",                   None),
    ("report_done",               False),
    ("baseline_ear",              None),
    ("ear_threshold",             None),
    ("rows_cache",                []),
    ("report_data",               None),
    ("user_baseline_blink_rate",  None),
    ("user_baseline_perclos",     None),
    ("profile", {"name": "User", "enable_cnn": True, "enable_scheduler": True}),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="eyeguard-header">
        <div>
            <div class="eyeguard-logo">👁 EyeGuard</div>
            <div class="eyeguard-sub">Eye Strain Monitor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = {"Assessment": "📊", "Profile": "👤", "History": "🗂️", "Settings": "⚙️"}
    for page_name, icon in pages.items():
        is_active = st.session_state.page == page_name
        if st.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = page_name

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="version-badge">v {APP_VERSION}</div>',
                unsafe_allow_html=True)
    st.caption("Not a medical device.")


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def section(label: str):
    st.markdown(f'<div class="section-heading">{label}</div>',
                unsafe_allow_html=True)


def metric_card(label: str, value: str, unit: str = "", css_class: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>"""


def status_pill(phase: str) -> str:
    labels = {"IDLE": "Idle", "CALIB": "Calibrating", "RUN": "Live", "DONE": "Complete"}
    pulse  = ' pulse' if phase in ("CALIB", "RUN") else ""
    return (f'<span class="status-pill {phase.lower()}">'
            f'<span class="status-dot{pulse}"></span>'
            f'{labels.get(phase, phase)}</span>')


def progress_bar(pct: float, phase: str, label_left: str, label_right: str) -> str:
    fill_class = "calib" if phase == "CALIB" else ""
    return f"""
    <div class="progress-wrap">
        <div class="progress-label">
            <span>{label_left}</span><span>{label_right}</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill {fill_class}" style="width:{pct*100:.1f}%"></div>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PROFILE
# ─────────────────────────────────────────────────────────────────────────────
def page_profile():
    st.markdown(
        '<div class="eyeguard-logo" style="font-size:20px;margin-bottom:4px">👤 Profile</div>',
        unsafe_allow_html=True)
    st.caption("Your personal settings are used for every assessment.")
    st.markdown("---")

    section("Personal Info")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Display Name",
                             value=st.session_state.profile.get("name", "User"))
    with col2:
        st.text_input("Role / Occupation (optional)",
                      placeholder="e.g. Software Engineer")

    section("Assessment Preferences")
    c1, c2 = st.columns(2)
    with c1:
        enable_cnn = st.checkbox(
            "Use CNN model",
            value=st.session_state.profile.get("enable_cnn", True),
            help="Uses your trained eye model for better accuracy",
        )
        enable_scheduler = st.checkbox(
            "Smart break reminders",
            value=st.session_state.profile.get("enable_scheduler", True),
        )
    with c2:
        cnn_status = "✅ Loaded" if CNN_PATH.exists() else "❌ Not found"
        st.markdown(f"""
        <div class="metric-card" style="margin-top:8px">
            <div class="metric-label">CNN Model Status</div>
            <div style="font-family:'Space Mono',monospace;font-size:14px;
                        color:var(--cyan)">{cnn_status}</div>
            <div class="metric-unit">eye_model_best.pth</div>
        </div>""", unsafe_allow_html=True)

    if st.button("💾  Save Profile", type="primary"):
        st.session_state.profile = {
            "name": name.strip() or "User",
            "enable_cnn": enable_cnn,
            "enable_scheduler": enable_scheduler,
        }
        profile_path = PROFILE_DIR / f"{name.strip() or 'User'}.json"
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as f:
            json.dump(st.session_state.profile, f, indent=2)
        st.success("Profile saved ✅")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def page_history():
    st.markdown(
        '<div class="eyeguard-logo" style="font-size:20px;margin-bottom:4px">🗂️ History</div>',
        unsafe_allow_html=True)
    st.caption("Past assessment sessions saved to disk.")
    st.markdown("---")

    sessions = sorted(SESS_DIR.glob("*.json"), reverse=True) if SESS_DIR.exists() else []

    if not sessions:
        st.info("No sessions recorded yet. Run an assessment to see history here.")
        return

    section(f"Sessions ({len(sessions)} found)")
    st.markdown("""
    <div style="display:grid;grid-template-columns:160px 80px 90px 90px 90px 1fr;
                gap:12px;padding:6px 16px;font-size:10px;text-transform:uppercase;
                letter-spacing:1.5px;color:var(--text-mute);">
        <span>Timestamp</span><span>Risk</span>
        <span>Comfort</span><span>Blink</span><span>PERCLOS</span><span>Advice</span>
    </div>""", unsafe_allow_html=True)

    for sess_path in sessions[:20]:
        try:
            with open(sess_path) as f:
                d = json.load(f)
            risk     = d.get("final_risk_level", "?")
            pill_cls = {"LOW": "low", "MED": "med", "HIGH": "high"}.get(risk, "low")
            ts       = d.get("timestamp", sess_path.stem)[-19:].replace("_", " ")
            comfort  = d.get("final_comfort_score_100", "—")
            blink    = d.get("final_blink_rate", 0)
            perclos  = d.get("final_perclos", 0)
            advice   = d.get("advice_title", "")[:40]
            st.markdown(f"""
            <div class="history-row">
                <span class="history-time">{ts}</span>
                <span><span class="pill {pill_cls}">{risk}</span></span>
                <span style="font-family:'Space Mono',monospace">{comfort}/100</span>
                <span style="color:var(--text-dim)">{blink:.0f}/min</span>
                <span style="color:var(--text-dim)">{perclos*100:.0f}%</span>
                <span style="color:var(--text-mute);font-size:12px">{advice}</span>
            </div>""", unsafe_allow_html=True)
        except Exception:
            continue


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
def page_settings():
    st.markdown(
        '<div class="eyeguard-logo" style="font-size:20px;margin-bottom:4px">⚙️ Settings</div>',
        unsafe_allow_html=True)
    st.caption("Advanced assessment parameters.")
    st.markdown("---")

    section("Break Scheduler Durations")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.number_input("Micro break (sec)",  value=config.BREAK_MICRO_SEC,
                        min_value=5,   max_value=60)
    with s2:
        st.number_input("Short break (sec)",  value=config.BREAK_SHORT_SEC,
                        min_value=30,  max_value=600)
    with s3:
        st.number_input("Long break (sec)",   value=config.BREAK_LONG_SEC,
                        min_value=120, max_value=1800)

    section("Risk Thresholds")
    t1, t2 = st.columns(2)
    with t1:
        st.number_input("Min gap between breaks (sec)",
                        value=int(config.BREAK_MIN_GAP_SEC),
                        min_value=10, max_value=300)
    with t2:
        st.number_input("Microsleep window (sec)",
                        value=int(config.MICROSLEEP_WINDOW_SEC),
                        min_value=30, max_value=300)

    section("CNN Inference")
    st.number_input("Run CNN every N frames", value=config.CNN_EVERY_N_FRAMES,
                    min_value=1, max_value=30)
    st.slider("CNN EWMA smoothing (alpha)", 0.05, 0.50, config.CNN_EWMA_ALPHA, 0.01)

    section("Data")
    if st.button("🗑️  Clear all session history", type="secondary"):
        if SESS_DIR.exists():
            for f in SESS_DIR.glob("*.json"):
                f.unlink()
        st.success("Session history cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _render(phase: str, snap: dict,
            ph_focus, ph_progress, ph_ear, ph_comfort, ph_status, ph_break, ph_recs):
    t = float(snap.get("t", 0.0))

    # ── Focus task ────────────────────────────────────────────────────────────
    with ph_focus.container():
        section("Focus Task")
        if phase == "IDLE":
            st.markdown("""
            <div class="focus-card">
                <div class="focus-card-title">Ready</div>
                <div class="focus-card-body">Start the camera, then click
                <strong>Start Assessment</strong>.</div>
            </div>""", unsafe_allow_html=True)
        elif phase == "CALIB":
            calib_t   = float(snap.get("t", 0.0))
            calib_pct = int(min(100, calib_t / CALIB_SECONDS * 100))
            st.markdown(f"""
            <div class="focus-card">
                <div class="focus-card-title">Calibrating — {calib_pct}%</div>
                <div class="focus-card-body">Look naturally at the screen.
                Don't force blinking.</div>
                <div class="focus-card-prompt">Hold still for 10 seconds while
                we measure your baseline EAR.</div>
            </div>""", unsafe_allow_html=True)
            ear_live = snap.get("ear")
            if ear_live is not None:
                st.markdown(
                    f'<div style="font-size:12px;color:var(--text-mute);margin-top:4px">'
                    f'Live EAR: <span style="color:var(--cyan);'
                    f'font-family:Space Mono,monospace">'
                    f'{float(ear_live):.4f}</span>'
                    f' — should be 0.20–0.35 with eyes open</div>',
                    unsafe_allow_html=True)
        elif phase == "RUN":
            st.markdown(f"""
            <div class="focus-card">
                <div class="focus-card-title">Assessment Running</div>
                <div class="focus-card-body">{focus_text(t)}</div>
                <div class="focus-card-prompt">{micro_prompt(t)}</div>
            </div>""", unsafe_allow_html=True)
        else:  # DONE
            st.markdown("""
            <div class="focus-card">
                <div class="focus-card-title">Complete</div>
                <div class="focus-card-body">Assessment finished.
                See your report below.</div>
            </div>""", unsafe_allow_html=True)

    # ── Progress ──────────────────────────────────────────────────────────────
    with ph_progress.container():
        section("Progress")
        if phase == "CALIB":
            p    = min(1.0, t / CALIB_SECONDS)
            html = progress_bar(p, "CALIB",
                                f"Calibrating… {int(p*100)}%",
                                f"{CALIB_SECONDS - t:.0f}s left")
        elif phase == "RUN":
            p      = min(1.0, t / RUN_SECONDS)
            remain = max(0.0, RUN_SECONDS - t)
            html   = progress_bar(p, "RUN",
                                  f"Assessing… {int(p*100)}%",
                                  f"~{remain:.0f}s left")
        elif phase == "DONE":
            html = progress_bar(1.0, "DONE", "Complete ✅", "")
        else:
            html = progress_bar(0.0, "IDLE", "Idle", "")
        st.markdown(html, unsafe_allow_html=True)

    # ── EAR row — FIX 12: clear on DONE so stale value doesn't persist ───────
    with ph_ear.container():
        if phase == "DONE":
            st.empty()
        elif phase == "RUN":
            ev = snap.get("ear")
            er = snap.get("ear_raw")
            et = snap.get("ear_threshold")
            if ev is not None:
                st.markdown(f"""
                <div class="ear-row">
                    <div class="ear-cell">
                        <div class="ear-cell-label">EAR Smooth</div>
                        <div class="ear-cell-value">{float(ev):.3f}</div>
                    </div>
                    <div class="ear-cell">
                        <div class="ear-cell-label">EAR Raw</div>
                        <div class="ear-cell-value">
                            {f"{float(er):.3f}" if er is not None else "—"}
                        </div>
                    </div>
                    <div class="ear-cell">
                        <div class="ear-cell-label">Threshold</div>
                        <div class="ear-cell-value">
                            {f"{float(et):.3f}" if et is not None else "—"}
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Comfort score ─────────────────────────────────────────────────────────
    with ph_comfort.container():
        section("Live Status")
        c   = snap.get("comfort")
        css = snap.get("comfort_css", "")
        lbl = snap.get("comfort_band", "")
        if c is None:
            st.markdown("""
            <div class="comfort-ring-wrap">
                <div class="comfort-score-big" style="color:var(--text-mute)">—</div>
                <div class="comfort-label">Awaiting data</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="comfort-ring-wrap">
                <div class="comfort-score-big {css}">{int(c)}</div>
                <div class="comfort-label">/ 100 · {lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Status / FPS / metrics ────────────────────────────────────────────────
    with ph_status.container():
        fps_val   = float(snap.get("fps", 0.0))
        fps_color = ("var(--green)" if fps_val >= 18
                     else "var(--amber)" if fps_val >= 12
                     else "var(--red)")
        msg = snap.get("msg", "")
        st.markdown(
            f'<div style="margin:8px 0 4px 0;font-size:13px;color:var(--text-dim)">'
            f'{status_pill(phase)} &nbsp; '
            f'<span class="fps-chip" style="color:{fps_color}">'
            f'⚡ {fps_val:.0f} fps</span></div>',
            unsafe_allow_html=True,
        )
        if msg and phase not in ("IDLE", "DONE"):
            st.markdown(
                f'<div style="font-size:12px;color:var(--text-mute);margin:4px 0">'
                f'{msg}</div>',
                unsafe_allow_html=True)
        if phase == "RUN":
            be = snap.get("baseline_ear")
            et = snap.get("ear_threshold")
            if be and et:
                st.markdown(
                    f'<div style="font-size:11px;color:var(--text-mute);margin:2px 0">'
                    f'Baseline: <code>{float(be):.4f}</code> · '
                    f'Threshold: <code>{float(et):.4f}</code></div>',
                    unsafe_allow_html=True)

        if phase == "RUN":
            cnn_val = snap.get("cnn_ewma")
            if cnn_val is None:
                st.markdown('<div class="cnn-badge muted">⚪ CNN — warming up</div>',
                            unsafe_allow_html=True)
            else:
                pct = int(cnn_val * 100)
                if pct >= 65:
                    cls, icon = "danger", "🔴"
                elif pct >= 40:
                    cls, icon = "warn", "🟡"
                else:
                    cls, icon = "good", "🟢"
                st.markdown(
                    f'<div class="cnn-badge {cls}">'
                    f'{icon} CNN Model · {pct}% sleepiness</div>',
                    unsafe_allow_html=True)

        if phase == "RUN":
            blink   = snap.get("blink_rate")
            perclos = snap.get("perclos")
            if blink is not None:
                b_css = ("danger" if blink < 6
                         else "warn" if blink < 12 else "good")
                p_css = ("danger" if (perclos or 0) > 0.20
                         else "warn" if (perclos or 0) > 0.12 else "good")
                st.markdown(f"""
                <div class="metric-grid">
                    {metric_card("Blink Rate",  f"{blink:.0f}",
                                 "/min", b_css)}
                    {metric_card("Eye Closure", f"{(perclos or 0)*100:.0f}",
                                 "%",    p_css)}
                </div>""", unsafe_allow_html=True)

    # ── Break alert ───────────────────────────────────────────────────────────
    with ph_break.container():
        if phase == "RUN":
            bm      = snap.get("break_mode", "WORK")
            br      = float(snap.get("break_remaining_sec", 0.0))
            css_cls = _break_css_class(bm)
            lbl     = _break_label(bm)
            body    = _break_body(bm, br)
            st.markdown(f"""
            <div class="break-alert {css_cls}">
                <div class="break-title">{lbl}</div>
                <div class="break-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    # ── Recommendations — FIX 13: show in DONE using saved report advice ──────
    with ph_recs.container():
        section("Recommendations")
        if phase in ("RUN", "DONE"):
            if phase == "DONE":
                rd      = st.session_state.get("report_data")
                title   = (rd.get("advice_title", "Assessment complete.")
                           if rd else "Assessment complete.")
                bullets = rd.get("advice_bullets", []) if rd else []
            else:
                adv = recommendations_dynamic(
                    level=str(snap.get("risk_level") or "LOW"),
                    score01=float(snap.get("risk_score") or 0.0),
                    perclos=float(snap.get("perclos") or 0.0),
                    blink_rate=float(snap.get("blink_rate") or 0.0),
                    cnn_sleepy=snap.get("cnn_ewma"),
                    baseline_blink_rate=st.session_state.get(
                        "user_baseline_blink_rate"),
                    ear_drop_ratio=float(snap.get("ear_drop_ratio") or 1.0),
                    trend_cnn=snap.get("cnn_trend"),
                )
                title   = adv.title
                bullets = adv.bullets[:4]

            bullets_html = "".join(
                f'<div class="rec-bullet">{b}</div>' for b in bullets)
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">{title}</div>
                {bullets_html}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:var(--text-mute);font-size:13px">'
                'Recommendations appear during assessment.</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT — compute and save
# ─────────────────────────────────────────────────────────────────────────────
def _compute_and_save_report(proc, cfg: RuntimeCfg):
    rows = proc.pop_rows()
    if rows:
        df = pd.DataFrame(rows)
        st.session_state.rows_cache = rows
    else:
        df = pd.DataFrame(st.session_state.rows_cache)

    if df.empty:
        st.session_state.report_data = {"empty": True}
        st.session_state.report_done = True
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SESS_DIR.mkdir(parents=True, exist_ok=True)

    blink_final   = float(df["blink_rate_bpm"].tail(1).iloc[0])
    perclos_final = float(df["perclos"].tail(1).iloc[0])
    cnn_tail      = df["cnn_sleepy_ewma"].dropna().tail(1)
    cnn_final     = float(cnn_tail.iloc[0]) if len(cnn_tail) else None

    level_final, score_final, reasons_final = fuse_risk(
        perclos=perclos_final, blink_rate=blink_final,
        cnn_prob=cnn_final, lstm_prob=None,
    )

    _report_baseline_blink = None
    try:
        _bl_path = config.PROFILES_DIR / f"{cfg.user_name}_baseline.json"
        _ub      = load_user_baseline(_bl_path, cfg.user_name)
        _report_baseline_blink = _ub.ema_blink_rate
    except Exception:
        pass

    advice = recommendations_dynamic(
        level=level_final, score01=float(score_final),
        perclos=float(perclos_final), blink_rate=float(blink_final),
        cnn_sleepy=cnn_final,
        baseline_blink_rate=_report_baseline_blink,
        ear_drop_ratio=(float(df["ear_drop_ratio"].dropna().mean())
                        if len(df["ear_drop_ratio"].dropna()) else 1.0),
        trend_cnn=None,
    )

    fl  = (float(df["fatigue_load"].tail(1).iloc[0])
           if "fatigue_load" in df.columns else None)
    msw = (int(df["microsleep_count_window"].tail(1).iloc[0])
           if "microsleep_count_window" in df.columns else 0)
    fc  = (int(df["comfort_score_100"].tail(1).iloc[0])
           if "comfort_score_100" in df.columns
           else comfort_score_from_risk(score_final))
    _, fc_css = comfort_band(fc)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    with open(SESS_DIR / f"{cfg.user_name}_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({
            "user":                  cfg.user_name,
            "timestamp":             ts,
            "baseline_ear":          st.session_state.baseline_ear,
            "ear_threshold":         st.session_state.ear_threshold,
            "final_perclos":         float(perclos_final),
            "final_blink_rate":      float(blink_final),
            "final_cnn_sleepy_prob": cnn_final,
            "final_risk_level":      level_final,
            "final_risk_score":      float(score_final),
            "final_comfort_score_100": int(fc),
            "final_fatigue_load":    fl,
            "microsleeps_2min":      msw,
            "advice_title":          advice.title,
            "fusion_reasons":        reasons_final,
            "baseline_blink_rate":   _report_baseline_blink,
            "app_version":           APP_VERSION,
        }, f, indent=2)

    st.session_state.report_data = {
        "empty":          False,
        "comfort":        fc,
        "comfort_css":    fc_css,
        "blink_rate":     blink_final,
        "perclos":        perclos_final,
        "fatigue_load":   fl,
        "microsleeps":    msw,
        "risk_level":     level_final,
        "baseline_ear":   st.session_state.baseline_ear,
        "ear_threshold":  st.session_state.ear_threshold,
        "advice_title":   advice.title,
        "advice_bullets": advice.bullets,
        "csv_bytes":      df.to_csv(index=False).encode("utf-8"),
        "csv_filename":   f"assessment_{cfg.user_name}_{ts}.csv",
        "cnn_final_prob": cnn_final,
        "timestamp":      ts,
    }

    try:
        baseline_path = config.PROFILES_DIR / f"{cfg.user_name}_baseline.json"
        ub = load_user_baseline(baseline_path, cfg.user_name)
        ub.update_from_session(
            duration_sec=float(df["t"].max()),
            frames_ok=len(df),
            blink_rate_final=blink_final,
            perclos_final=perclos_final,
            ear_mean=(float(df["ear_raw"].mean())
                      if "ear_raw" in df.columns else None),
        )
        save_user_baseline(baseline_path, ub)
    except Exception:
        pass

    st.session_state.report_done = True


# ─────────────────────────────────────────────────────────────────────────────
# REPORT — display
# ─────────────────────────────────────────────────────────────────────────────
def _show_report():
    d = st.session_state.get("report_data")
    if not d:
        return

    section("Assessment Report")

    if d.get("empty"):
        st.error("No frames captured. Improve lighting and keep face centered.")
        return

    fc       = d["comfort"]
    fc_css   = d.get("comfort_css", "")
    risk     = d.get("risk_level", "LOW")
    pill_cls = {"LOW": "low", "MED": "med", "HIGH": "high"}.get(risk, "low")

    st.markdown(f"""
    <div class="report-header">
        <div class="comfort-score-big {fc_css}"
             style="font-size:40px">{fc}</div>
        <div>
            <div style="font-size:11px;color:var(--text-mute);
                        text-transform:uppercase;letter-spacing:1px">
                Comfort Score
            </div>
            <div style="margin-top:6px">
                <span class="pill {pill_cls}">{risk} RISK</span>
                <span class="report-badge" style="margin-left:8px">
                    Complete ✅
                </span>
            </div>
        </div>
        <div style="margin-left:auto;font-size:11px;color:var(--text-mute)">
            {d.get("timestamp","").replace("_"," ")}
        </div>
    </div>""", unsafe_allow_html=True)

    b_css  = ("danger" if d["blink_rate"] < 6
              else "warn" if d["blink_rate"] < 12 else "good")
    p_css  = ("danger" if d["perclos"] > 0.20
              else "warn" if d["perclos"] > 0.12 else "good")
    fl     = d["fatigue_load"]
    fl_str = "N/A" if fl is None else f"{fl:.2f}"

    cnn_val = d.get("cnn_final_prob")
    if cnn_val is not None:
        cnn_pct = int(cnn_val * 100)
        cnn_css = ("danger" if cnn_pct >= 65
                   else "warn" if cnn_pct >= 40 else "good")
        cnn_str = f"{cnn_pct}%"
    else:
        cnn_pct, cnn_css, cnn_str = None, "", "—"

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);
                gap:12px;margin:16px 0">
        {metric_card("Blink Rate",   f"{d['blink_rate']:.0f}", "/min",    b_css)}
        {metric_card("Eye Closure",  f"{d['perclos']*100:.0f}", "%",       p_css)}
        {metric_card("Fatigue Load", fl_str, "",                           "")}
        {metric_card("CNN Model",    cnn_str, "sleepiness",                cnn_css)}
    </div>""", unsafe_allow_html=True)

    ms = d["microsleeps"]
    be = d.get("baseline_ear")
    et = d.get("ear_threshold")
    stats_html = f'<span style="margin-right:20px">Microsleeps: <strong>{ms}</strong></span>'
    if be:
        stats_html += (f'Baseline EAR: <strong>{be:.3f}</strong> · '
                       f'Threshold: <strong>{et:.3f}</strong>')
    st.markdown(
        f'<div style="font-size:13px;color:var(--text-dim);margin:8px 0">'
        f'{stats_html}</div>',
        unsafe_allow_html=True)

    bullets_html = "".join(
        f'<div class="rec-bullet">{b}</div>' for b in d["advice_bullets"])
    st.markdown(f"""
    <div class="rec-card" style="margin-top:16px">
        <div class="rec-title">{d["advice_title"]}</div>
        {bullets_html}
    </div>""", unsafe_allow_html=True)

    st.download_button(
        "⬇  Download Full CSV",
        data=d["csv_bytes"],
        file_name=d["csv_filename"],
        mime="text/csv",
        key="report_csv_btn",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
def page_assessment():
    st.markdown("""
    <div class="eyeguard-header">
        <div>
            <div class="eyeguard-logo">👁 EyeGuard</div>
            <div class="eyeguard-sub">Real-time Eye Strain Monitor</div>
        </div>
    </div>""", unsafe_allow_html=True)

    profile = st.session_state.profile
    cfg_now = RuntimeCfg(
        user_name=profile.get("name", "User"),
        enable_cnn=profile.get("enable_cnn", True),
        cnn_every_n_frames=config.CNN_EVERY_N_FRAMES,
        ewma_alpha=config.CNN_EWMA_ALPHA,
        enable_scheduler=profile.get("enable_scheduler", True),
        microsleep_sec=config.MICROSLEEP_SEC,
        microsleep_window_sec=config.MICROSLEEP_WINDOW_SEC,
        microsleep_short_count=config.MICROSLEEP_SHORT_COUNT,
        microsleep_long_count=config.MICROSLEEP_LONG_COUNT,
        micro_sec=config.BREAK_MICRO_SEC,
        short_sec=config.BREAK_SHORT_SEC,
        long_sec=config.BREAK_LONG_SEC,
        min_gap=config.BREAK_MIN_GAP_SEC,
        k_up=config.FATIGUE_K_UP,
        k_down=config.FATIGUE_K_DOWN,
        k_leak=config.FATIGUE_K_LEAK,
    )
    if st.session_state.run_cfg is None:
        st.session_state.run_cfg = cfg_now

    section("Camera Feed")
    st.caption("Allow camera permission when asked. Click Start in the widget on first use.")

    webrtc_ctx = webrtc_streamer(
        key=WEBRTC_KEY,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EyeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": [config.STUN_SERVER]}]},
    )

    processor    = webrtc_ctx.video_processor
    camera_ready = bool(
        getattr(webrtc_ctx, "state", None)
        and webrtc_ctx.state.playing
        and processor is not None
    )
    if not camera_ready:
        st.markdown(
            '<div style="font-size:13px;color:var(--amber);margin-top:4px">'
            '⚠️ Camera not running — click Start above and allow permission.</div>',
            unsafe_allow_html=True)

    section("Assessment Control")
    btn1, btn2, _ = st.columns([2, 1, 3])
    with btn1:
        start_clicked = st.button("▶  Start 60s Assessment", type="primary",
                                  use_container_width=True)
    with btn2:
        stop_clicked = st.button("⏹  Stop", use_container_width=True)

    if start_clicked:
        if not camera_ready:
            st.warning("Please start the camera first.")
        else:
            st.session_state.report_done   = False
            st.session_state.rows_cache    = []
            st.session_state.baseline_ear  = None
            st.session_state.ear_threshold = None
            st.session_state.run_cfg       = cfg_now
            st.session_state.ui_phase      = "CALIB"
            st.session_state.pop("_camera_dead_since", None)

            # Load personal baseline so live recs are personalised from frame 1
            try:
                _bl_path = config.PROFILES_DIR / f"{cfg_now.user_name}_baseline.json"
                _ub      = load_user_baseline(_bl_path, cfg_now.user_name)
                st.session_state.user_baseline_blink_rate = _ub.ema_blink_rate
                st.session_state.user_baseline_perclos    = _ub.ema_perclos
            except Exception:
                st.session_state.user_baseline_blink_rate = None
                st.session_state.user_baseline_perclos    = None

            processor.set_cfg(cfg_now)
            processor.set_phase("CALIB")

    if stop_clicked:
        st.session_state.ui_phase = "IDLE"
        st.session_state.pop("_camera_dead_since", None)
        if processor is not None:
            processor.set_phase("IDLE")

    st.markdown("<br>", unsafe_allow_html=True)
    left_col, right_col = st.columns([3, 2])

    with left_col:
        ph_focus    = st.empty()
        ph_progress = st.empty()
        ph_ear      = st.empty()
        ph_recs     = st.empty()

    with right_col:
        ph_comfort = st.empty()
        ph_status  = st.empty()
        ph_break   = st.empty()

    ph_report = st.empty()

    phase = st.session_state.ui_phase
    snap  = processor.get_last() if processor is not None else {}
    _render(phase, snap, ph_focus, ph_progress, ph_ear,
            ph_comfort, ph_status, ph_break, ph_recs)

    if st.session_state.report_done:
        with ph_report.container():
            _show_report()

    if phase in ("CALIB", "RUN") and processor is not None:
        while True:
            time.sleep(LOOP_SLEEP_SEC)

            snap        = processor.get_last()
            ev, payload = processor.pop_event()
            proc_phase  = processor.get_phase()

            if proc_phase == "RUN" and st.session_state.ui_phase == "CALIB":
                st.session_state.ui_phase = "RUN"
                if ev == "CALIB_DONE" and payload:
                    st.session_state.baseline_ear  = float(payload["baseline_ear"])
                    st.session_state.ear_threshold = float(payload["ear_threshold"])
                    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
                    with open(PROFILE_DIR / f"{st.session_state.run_cfg.user_name}.json", "w") as f:
                        json.dump({
                            "user":          st.session_state.run_cfg.user_name,
                            "baseline_ear":  st.session_state.baseline_ear,
                            "ear_threshold": st.session_state.ear_threshold,
                        }, f, indent=2)
                    # FIX 14: reload baseline immediately so live recs
                    # are personalised from the first RUN frame onward
                    try:
                        _bl_path = config.PROFILES_DIR / f"{st.session_state.run_cfg.user_name}_baseline.json"
                        _ub      = load_user_baseline(_bl_path, st.session_state.run_cfg.user_name)
                        st.session_state.user_baseline_blink_rate = _ub.ema_blink_rate
                        st.session_state.user_baseline_perclos    = _ub.ema_perclos
                    except Exception:
                        pass

            if proc_phase == "DONE" or ev == "DONE":
                st.session_state.ui_phase = "DONE"
                _render("DONE", snap, ph_focus, ph_progress, ph_ear,
                        ph_comfort, ph_status, ph_break, ph_recs)
                if not st.session_state.report_done:
                    _compute_and_save_report(processor, st.session_state.run_cfg)
                with ph_report.container():
                    _show_report()
                break

            _render(st.session_state.ui_phase, snap,
                    ph_focus, ph_progress, ph_ear,
                    ph_comfort, ph_status, ph_break, ph_recs)

            # Camera grace period
            if not (getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing):
                if "_camera_dead_since" not in st.session_state:
                    st.session_state["_camera_dead_since"] = time.time()
                elif (time.time() - st.session_state["_camera_dead_since"]
                      > config.CAMERA_GRACE_SEC):
                    if not st.session_state.report_done:
                        _compute_and_save_report(processor, st.session_state.run_cfg)
                    st.session_state.ui_phase = "IDLE"
                    st.session_state.pop("_camera_dead_since", None)
                    _render("IDLE", {}, ph_focus, ph_progress, ph_ear,
                            ph_comfort, ph_status, ph_break, ph_recs)
                    break
            else:
                st.session_state.pop("_camera_dead_since", None)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "Assessment":
    page_assessment()
elif page == "Profile":
    page_profile()
elif page == "History":
    page_history()
elif page == "Settings":
    page_settings()