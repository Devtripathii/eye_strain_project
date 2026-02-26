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
from typing import Optional

import pandas as pd
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from src.core.timebase import Timebase
from src.cv.landmarks import FaceMeshLandmarker
from src.cv.features import ear_both_eyes, BlinkCounter, FatigueMetrics
from src.cv.roi import crop_both_eyes
from src.ml.fusion import fuse_risk
from src.ml.torch_cnn import TorchEyeCnn
from src.ui.recommendations import recommendations_dynamic

from src.scheduler.fatigue_load import FatigueLoad, FatigueLoadParams
from src.scheduler.break_scheduler import BreakScheduler, BreakThresholds, BreakDurations
from src.signals.blink_duration import BlinkDurationTracker


# ---------------- UI ----------------
st.set_page_config(page_title="Eye Comfort Assistant (Webcam)", layout="wide")
st.title("👁️ Eye Comfort Assistant (One-Click Assessment)")
st.caption("Browser webcam via WebRTC. Not a medical diagnosis.")

APP_VERSION = "oneclick-60s-v1"
st.caption(f"Build: `{APP_VERSION}`")

FOCUS_TEXTS = [
    "Read naturally. Don’t force blinking.",
    "Relax your face and shoulders.",
    "Keep the screen slightly below eye level.",
    "If you feel strain, soften your gaze and blink normally.",
]
PROMPTS = [
    "Keep reading naturally.",
    "Relax your jaw and shoulders.",
    "Don’t squint—adjust brightness if needed.",
    "Blink normally (don’t force it).",
]


def focus_text(t: float) -> str:
    return FOCUS_TEXTS[int(t // 12) % len(FOCUS_TEXTS)]


def micro_prompt(t: float) -> str:
    return PROMPTS[int(t // 8) % len(PROMPTS)]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def comfort_score_from_risk(risk01: float) -> int:
    r = _clamp01(float(risk01))
    score = int(round(100.0 * (1.0 - r)))
    return max(0, min(100, score))


def comfort_band(score100: int) -> tuple[str, str]:
    s = int(score100)
    if s >= 70:
        return ("Good", "🟢")
    if s >= 40:
        return ("Warning", "🟡")
    return ("Danger", "🔴")


def comfort_legend() -> str:
    return "🟢 Good (70–100)  •  🟡 Warning (40–69)  •  🔴 Danger (0–39)"


def _break_friendly(mode: str, remaining_sec: float) -> tuple[str, str]:
    mode = (mode or "WORK").upper()
    if mode == "WORK":
        return ("✅ Continue", "You can continue. Blink normally and keep a relaxed posture.")
    if mode == "MICRO":
        return ("🟡 Quick break", f"Look far away and relax your eyes.\n\nTime left: **{remaining_sec:.0f}s**")
    if mode == "SHORT":
        return ("🟠 Short break", f"Stand up, stretch, drink water.\n\nTime left: **{remaining_sec:.0f}s**")
    if mode == "LONG":
        return ("🔴 Long break", f"Step away from the screen completely.\n\nTime left: **{remaining_sec:.0f}s**")
    return ("✅ Continue", "You can continue.")


# ---------------- Paths ----------------
OUT_DIR = PROJECT_ROOT / "outputs"
SESS_DIR = OUT_DIR / "sessions"
PROFILE_DIR = PROJECT_ROOT / "data" / "user_profiles"
MODELS_DIR = PROJECT_ROOT / "models"
CNN_PATH = MODELS_DIR / "eye_model_best.pth"

# CRITICAL: constant WebRTC key (never change)
WEBRTC_KEY = "eye-comfort-stable"


def load_session_files():
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SESS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for p in files[:25]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows.append({
                "timestamp": data.get("timestamp"),
                "user": data.get("user"),
                "comfort": data.get("final_comfort_score_100"),
                "status": data.get("final_risk_level"),
                "blinks/min": data.get("final_blink_rate"),
                "eye closure %": None if data.get("final_perclos") is None else round(float(data.get("final_perclos")) * 100.0),
                "tiredness": data.get("final_fatigue_load"),
                "microsleeps": data.get("microsleeps_2min"),
                "file": p.name,
            })
        except Exception:
            continue
    return rows


# ---------------- Config (UI thread -> processor thread) ----------------
@dataclass
class RuntimeCfg:
    user_name: str
    duration_sec: float      # RUN duration
    calib_sec: float         # CALIB duration

    enable_cnn: bool
    cnn_every_n_frames: int
    ewma_alpha: float

    enable_scheduler: bool
    k_up: float
    k_down: float
    k_leak: float

    microsleep_sec: float
    microsleep_window_sec: float
    microsleep_short_count: int
    microsleep_long_count: int

    micro_sec: int
    short_sec: int
    long_sec: int
    min_gap: float


# ---------------- Video Processor (NO session_state access here) ----------------
class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()

        self.phase = "IDLE"  # IDLE / CALIB / RUN / DONE
        self.cfg: Optional[RuntimeCfg] = None

        self.last = {
            "ok": False,
            "msg": "Waiting for camera...",
            "event": None,  # None / CALIB_DONE / DONE
            "event_payload": None,

            "t": 0.0,      # time within current phase
            "dt": 0.0,
            "fps": 0.0,

            "ear": None,
            "ear_drop_ratio": None,
            "blink_rate": None,
            "perclos": None,
            "risk_level": None,
            "risk_score": None,
            "comfort": None,
            "comfort_band": None,
            "comfort_emoji": None,
            "fatigue_load": None,
            "break_mode": "WORK",
            "break_remaining_sec": 0.0,
            "microsleeps_window": 0,
            "cnn_ewma": None,
            "cnn_trend": None,

            "baseline_ear": None,
            "ear_threshold": None,
        }

        self.timebase = Timebase(fps_window=30, dt_max=0.25)
        self.fm = FaceMeshLandmarker(min_det=0.25, min_track=0.25, refine=False)

        self.cnn = None
        self.cnn_ewma = None
        self.cnn_hist = deque(maxlen=25)
        self.cnn_frames = 0
        self.last_roi = None

        self.blink = None
        self.fatigue = None
        self.fatigue_load = None
        self.scheduler = None
        self.blink_dur = None

        self._calib_ears: list[float] = []
        self._rows: list[dict] = []

        self._calib_start_ts: Optional[float] = None
        self._run_start_ts: Optional[float] = None

    # UI thread controls
    def set_cfg(self, cfg: RuntimeCfg):
        with self.lock:
            self.cfg = cfg

    def set_phase(self, phase: str):
        phase = (phase or "IDLE").upper()
        if phase not in ("IDLE", "CALIB", "RUN", "DONE"):
            phase = "IDLE"
        with self.lock:
            if phase != self.phase:
                if phase == "CALIB":
                    self._calib_start_ts = None
                    self._run_start_ts = None
                elif phase == "RUN":
                    self._run_start_ts = None
                elif phase == "IDLE":
                    self._calib_start_ts = None
                    self._run_start_ts = None
                self.phase = phase

    def _update_last(self, **kwargs):
        with self.lock:
            self.last.update(kwargs)

    def get_last(self) -> dict:
        with self.lock:
            return dict(self.last)

    def pop_event(self) -> tuple[Optional[str], Optional[dict]]:
        with self.lock:
            ev = self.last.get("event")
            payload = self.last.get("event_payload")
            self.last["event"] = None
            self.last["event_payload"] = None
            return ev, payload

    def pop_rows(self) -> list[dict]:
        with self.lock:
            rows = list(self._rows)
            self._rows = []
            return rows

    def _ensure_cnn(self, enable: bool):
        if not enable:
            return
        if self.cnn is not None:
            return
        if CNN_PATH.exists():
            self.cnn = TorchEyeCnn(CNN_PATH, grayscale=True, device="cpu")

    def _init_runtime(self, ear_threshold: float, cfg: RuntimeCfg):
        self.blink = BlinkCounter(float(ear_threshold))
        self.fatigue = FatigueMetrics(fps=20, window_seconds=30)

        self.fatigue_load = FatigueLoad(
            FatigueLoadParams(
                k_up=float(cfg.k_up),
                k_down=float(cfg.k_down),
                k_leak=float(cfg.k_leak),
                max_dt=0.25,
            )
        )

        self.scheduler = BreakScheduler(
            thresholds=BreakThresholds(
                micro_lo=0.35,
                micro_hi=0.55,
                short_hi=0.55,
                long_hi=0.75,
                sustain_risk_hi=0.55,
                sustain_seconds=30.0,
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
        tick = self.timebase.tick()
        dt = float(tick.dt)
        fps = float(tick.fps)

        img_bgr = frame.to_ndarray(format="bgr24")

        with self.lock:
            cfg = self.cfg
            phase = self.phase

        if cfg is None:
            self._update_last(ok=False, msg="Loading config...", dt=dt, fps=fps)
            return frame

        lms = self.fm.get_landmarks(img_bgr)
        if lms is None:
            self._update_last(ok=False, msg="Face not detected. Center your face.", dt=dt, fps=fps)
            return frame

        h, w = img_bgr.shape[:2]
        ear = float(ear_both_eyes(lms, w, h))

        # IDLE
        if phase == "IDLE":
            self._update_last(ok=True, msg="Ready", t=0.0, dt=dt, fps=fps, ear=float(ear))
            return frame

        # CALIB
        if phase == "CALIB":
            if self._calib_start_ts is None:
                self._calib_start_ts = time.time()
                self._calib_ears = []
                self._rows = []
                self.cnn_ewma = None
                self.cnn_hist.clear()
                self._update_last(
                    baseline_ear=None,
                    ear_threshold=None,
                    event=None,
                    event_payload=None,
                )

            t_cal = time.time() - float(self._calib_start_ts)
            self._calib_ears.append(float(ear))

            progress = min(1.0, t_cal / max(1e-6, float(cfg.calib_sec)))
            self._update_last(ok=True, msg=f"Calibrating... {progress*100:.0f}%", t=float(t_cal), dt=dt, fps=fps, ear=float(ear))

            # Better calibration: use median baseline + 25th percentile threshold
            if t_cal >= float(cfg.calib_sec) and len(self._calib_ears) >= 20:
                ears = sorted(self._calib_ears)
                mid = len(ears) // 2
                baseline = float(ears[mid])  # median
                q25 = float(ears[int(0.25 * (len(ears) - 1))])
                threshold = float(max(0.10, q25))  # conservative floor

                self._init_runtime(ear_threshold=threshold, cfg=cfg)

                self._update_last(
                    baseline_ear=baseline,
                    ear_threshold=threshold,
                    event="CALIB_DONE",
                    event_payload={"baseline_ear": baseline, "ear_threshold": threshold},
                )

                with self.lock:
                    self.phase = "RUN"
                    self._run_start_ts = None
                    self._calib_start_ts = None

            return frame

        # RUN
        if phase == "RUN":
            if self._run_start_ts is None:
                self._run_start_ts = time.time()

            t_run = time.time() - float(self._run_start_ts)

            baseline_ear = self.get_last().get("baseline_ear")
            ear_threshold = self.get_last().get("ear_threshold")
            if ear_threshold is None:
                self._update_last(ok=False, msg="Missing calibration. Start again.", t=float(t_run), dt=dt, fps=fps, ear=float(ear))
                return frame

            if self.blink is None:
                self._init_runtime(ear_threshold=float(ear_threshold), cfg=cfg)

            ear_drop_ratio = None
            if baseline_ear is not None and float(baseline_ear) > 0:
                ear_drop_ratio = float(ear / max(1e-6, float(baseline_ear)))

            blink_count, closed, blink_occurred = self.blink.update(ear)
            perclos, blink_rate = self.fatigue.update(eye_closed=closed, t_sec=float(t_run), blink_occurred=blink_occurred)

            blink_event, ms_count = self.blink_dur.update(closed=bool(closed), dt=float(dt), t_sec=float(t_run))
            microsleep_event = bool(blink_event.is_microsleep) if blink_event is not None else False

            self._ensure_cnn(bool(cfg.enable_cnn))
            cnn_prob = None
            if cfg.enable_cnn and self.cnn is not None:
                roi = crop_both_eyes(img_bgr, lms, pad=0.55)
                if roi is not None:
                    self.last_roi = roi
                self.cnn_frames += 1
                if self.last_roi is not None and (self.cnn_frames % int(cfg.cnn_every_n_frames) == 0):
                    res = self.cnn.predict_roi_bgr(self.last_roi)
                    if res is not None:
                        cnn_prob = float(res.sleepy_prob)
                        self.cnn_hist.append(cnn_prob)
                        if self.cnn_ewma is None:
                            self.cnn_ewma = cnn_prob
                        else:
                            a = float(cfg.ewma_alpha)
                            self.cnn_ewma = (a * cnn_prob) + ((1.0 - a) * self.cnn_ewma)

            trend_cnn = None
            if len(self.cnn_hist) >= 8:
                trend_cnn = float(self.cnn_hist[-1] - self.cnn_hist[0]) / max(1.0, len(self.cnn_hist))

            level_live, score_live, _ = fuse_risk(
                perclos=float(perclos),
                blink_rate=float(blink_rate),
                cnn_prob=float(self.cnn_ewma) if self.cnn_ewma is not None else None,
                lstm_prob=None,
            )

            if cfg.enable_scheduler:
                on_break = self.scheduler.is_on_break()
                F = self.fatigue_load.update(risk01=float(score_live), dt=float(dt), working=(not on_break), quality01=1.0)
                sched_state = self.scheduler.update(
                    t_sec=float(t_run),
                    dt=float(dt),
                    risk01=float(score_live),
                    fatigue_load01=float(F),
                    microsleep_event=bool(microsleep_event),
                )
            else:
                F = self.fatigue_load.update(risk01=float(score_live), dt=float(dt), working=True, quality01=1.0)
                sched_state = self.scheduler.state

            break_mode = sched_state.mode
            break_remaining = float(sched_state.remaining_sec)

            comfort = comfort_score_from_risk(float(score_live))
            band, band_emoji = comfort_band(comfort)

            row = {
                "t": round(float(t_run), 2),
                "fps": float(fps),
                "ear": round(float(ear), 3),
                "ear_drop_ratio": None if ear_drop_ratio is None else float(ear_drop_ratio),
                "blink_rate_bpm": float(blink_rate),
                "perclos": float(perclos),
                "microsleep_event": bool(microsleep_event),
                "microsleep_count_window": int(ms_count),
                "cnn_sleepy_prob": None if cnn_prob is None else float(cnn_prob),
                "cnn_sleepy_ewma": None if self.cnn_ewma is None else float(self.cnn_ewma),
                "cnn_trend": None if trend_cnn is None else float(trend_cnn),
                "risk_level_live": level_live,
                "risk_score_live": float(score_live),
                "comfort_score_100": int(comfort),
                "fatigue_load": float(F),
                "break_mode": str(break_mode),
                "break_remaining_sec": float(break_remaining),
            }
            with self.lock:
                self._rows.append(row)

            self._update_last(
                ok=True,
                msg="Assessing",
                t=float(t_run),
                dt=dt,
                fps=fps,
                ear=float(ear),
                ear_drop_ratio=ear_drop_ratio,
                blink_rate=float(blink_rate),
                perclos=float(perclos),
                risk_level=level_live,
                risk_score=float(score_live),
                comfort=int(comfort),
                comfort_band=band,
                comfort_emoji=band_emoji,
                fatigue_load=float(F),
                break_mode=str(break_mode),
                break_remaining_sec=float(break_remaining),
                microsleeps_window=int(ms_count),
                cnn_ewma=None if self.cnn_ewma is None else float(self.cnn_ewma),
                cnn_trend=None if trend_cnn is None else float(trend_cnn),
            )

            if t_run >= float(cfg.duration_sec):
                self._update_last(event="DONE", event_payload={"reason": "duration_reached"})
                with self.lock:
                    self.phase = "DONE"

            return frame

        # DONE
        self._update_last(ok=True, msg="Done", dt=dt, fps=fps, ear=float(ear))
        return frame


# ---------------- Session State (UI only) ----------------
if "ui_phase" not in st.session_state:
    st.session_state.ui_phase = "IDLE"  # IDLE / CALIB / RUN / DONE
if "rows" not in st.session_state:
    st.session_state.rows = []
if "baseline_ear" not in st.session_state:
    st.session_state.baseline_ear = None
if "ear_threshold" not in st.session_state:
    st.session_state.ear_threshold = None
if "run_cfg" not in st.session_state:
    st.session_state.run_cfg = None  # frozen cfg during assessment


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    existing_profiles = sorted([p.stem for p in PROFILE_DIR.glob("*.json")])

    running = st.session_state.ui_phase in ("CALIB", "RUN")
    st.caption("Settings lock during an assessment.")

    user_name = st.selectbox("Profile", ["(new user)"] + existing_profiles, disabled=running)
    if user_name == "(new user)":
        user_name = st.text_input("Enter your name", value="User", disabled=running).strip() or "User"

    st.markdown("---")
    st.subheader("Model + Smoothing")
    enable_cnn = st.checkbox("Use CNN eye-closure model", value=True, disabled=running)
    cnn_every_n_frames = st.slider("CNN frequency (frames)", 1, 25, 10, 1, disabled=running)
    ewma_alpha = st.slider("CNN smoothing", 0.05, 0.50, 0.18, 0.01, disabled=running)

    st.markdown("---")
    st.subheader("Smart breaks")
    enable_scheduler = st.checkbox("Enable smart breaks", value=True, disabled=running)
    k_up = st.slider("Fatigue builds speed", 0.005, 0.050, 0.020, 0.001, disabled=running)
    k_down = st.slider("Recovery speed", 0.010, 0.150, 0.050, 0.001, disabled=running)
    k_leak = st.slider("Natural recovery", 0.000, 0.020, 0.003, 0.001, disabled=running)

    microsleep_sec = st.slider("Microsleep threshold (sec)", 0.20, 1.50, 0.50, 0.05, disabled=running)
    microsleep_window_sec = st.slider("Microsleep window (sec)", 30, 300, 120, 10, disabled=running)
    microsleep_short_count = st.slider("Microsleeps → short break", 1, 5, 1, 1, disabled=running)
    microsleep_long_count = st.slider("Microsleeps → long break", 2, 10, 2, 1, disabled=running)

    micro_sec = st.slider("Quick break length (sec)", 10, 90, 20, 5, disabled=running)
    short_sec = st.slider("Short break length (sec)", 60, 600, 180, 30, disabled=running)
    long_sec = st.slider("Long break length (sec)", 120, 1200, 600, 60, disabled=running)
    min_gap = st.slider("Min time between breaks (sec)", 0, 300, 60, 10, disabled=running)

    st.markdown("---")
    with st.expander("History", expanded=False):
        hist = load_session_files()
        if hist:
            st.dataframe(pd.DataFrame(hist), width="stretch")
        else:
            st.caption("No sessions found yet.")


# ---------------- One-click assessment design ----------------
TOTAL_SECONDS = 60.0
CALIB_SECONDS = 10.0
RUN_SECONDS = TOTAL_SECONDS - CALIB_SECONDS

# Create cfg from sidebar values
current_cfg = RuntimeCfg(
    user_name=str(user_name),
    duration_sec=float(RUN_SECONDS),
    calib_sec=float(CALIB_SECONDS),
    enable_cnn=bool(enable_cnn),
    cnn_every_n_frames=int(cnn_every_n_frames),
    ewma_alpha=float(ewma_alpha),
    enable_scheduler=bool(enable_scheduler),
    k_up=float(k_up),
    k_down=float(k_down),
    k_leak=float(k_leak),
    microsleep_sec=float(microsleep_sec),
    microsleep_window_sec=float(microsleep_window_sec),
    microsleep_short_count=int(microsleep_short_count),
    microsleep_long_count=int(microsleep_long_count),
    micro_sec=int(micro_sec),
    short_sec=int(short_sec),
    long_sec=int(long_sec),
    min_gap=float(min_gap),
)

# Lock cfg snapshot when assessment starts
if st.session_state.run_cfg is None:
    st.session_state.run_cfg = current_cfg


# ---------------- Controls (SINGLE BUTTON) ----------------
bar1, bar2, bar3 = st.columns([1.6, 1.2, 2.2])

with bar1:
    if st.session_state.ui_phase in ("IDLE", "DONE"):
        if st.button("▶ Start 60s Assessment", type="primary", use_container_width=True):
            st.session_state.ui_phase = "CALIB"
            st.session_state.rows = []
            st.session_state.baseline_ear = None
            st.session_state.ear_threshold = None
            st.session_state.run_cfg = current_cfg  # freeze snapshot
    else:
        st.button("▶ Start 60s Assessment", disabled=True, use_container_width=True)

with bar2:
    if st.session_state.ui_phase in ("CALIB", "RUN"):
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.ui_phase = "IDLE"
            st.session_state.run_cfg = current_cfg
    else:
        st.button("⏹ Stop", disabled=True, use_container_width=True)

with bar3:
    st.markdown(f"**Legend:** {comfort_legend()}")

st.markdown("---")


# ---------------- Webcam (hidden in expander, auto-playing) ----------------
st.subheader("Assessment")
st.caption("Just click **Start 60s Assessment**. The camera runs in the background.")

with st.expander("Camera preview (optional)", expanded=False):
    st.caption("If your browser asks, allow camera permission.")
    webrtc_ctx = webrtc_streamer(
        key=WEBRTC_KEY,
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EyeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        desired_playing=True,  # auto-start (reduces need to click WebRTC Start)
    )

processor = webrtc_ctx.video_processor if 'webrtc_ctx' in locals() else None
if processor is None:
    st.info("Initializing webcam stream… If it doesn’t start, open the Camera preview expander once and allow permission.")
    st.stop()

# Push frozen cfg + current phase into processor
processor.set_cfg(st.session_state.run_cfg)
processor.set_phase(st.session_state.ui_phase)

# Pull last snapshot and events
last = processor.get_last()
ev, payload = processor.pop_event()

# React to calibration completion
if ev == "CALIB_DONE" and payload:
    st.session_state.baseline_ear = float(payload["baseline_ear"])
    st.session_state.ear_threshold = float(payload["ear_threshold"])

    # Save profile (UI thread only)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILE_DIR / f"{st.session_state.run_cfg.user_name}.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "user": st.session_state.run_cfg.user_name,
                "baseline_ear": st.session_state.baseline_ear,
                "ear_threshold": st.session_state.ear_threshold,
            },
            f,
            indent=2,
        )

    st.session_state.ui_phase = "RUN"
    processor.set_phase("RUN")

# React to done
if ev == "DONE":
    st.session_state.ui_phase = "DONE"
    processor.set_phase("DONE")


# ---------------- Panels ----------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Focus task")
    ph_task = st.empty()
    ph_progress_label = st.empty()
    ph_progress = st.empty()
    ph_hint = st.empty()

with right:
    st.subheader("Live status")
    ph_comfort = st.empty()
    ph_break = st.empty()

    st.subheader("Recommendations")
    ph_reco = st.empty()

    st.subheader("Quick stats")
    ph_stats = st.empty()

st.markdown("---")
st.subheader("Report")
ph_report = st.empty()


def render_idle():
    ph_task.info("Click **Start 60s Assessment**. Keep your face centered.")
    ph_progress_label.caption("No assessment running.")
    ph_progress.progress(0)
    ph_hint.caption("Tip: Good lighting + face centered improves accuracy.")
    ph_comfort.markdown("### Comfort score: **—**")
    ph_break.info("No break suggestion yet.")
    ph_reco.info("Recommendations will appear during assessment.")
    ph_stats.write("—")


def render_calib(last_dict: dict):
    ph_task.info("Calibration (10s)… Look at the screen naturally. Don’t squint.")
    t = float(last_dict.get("t", 0.0))
    p = min(1.0, t / max(1e-6, CALIB_SECONDS))
    ph_progress_label.caption(f"Calibrating… {int(p*100)}%")
    ph_progress.progress(p)
    ph_hint.caption(last_dict.get("msg", "Calibrating..."))

    ph_comfort.markdown("### Comfort score: **—**")
    ph_break.info("Hold steady; calibration improves accuracy.")
    ph_reco.info("Calibration in progress…")
    ph_stats.write(f"FPS: `{last_dict.get('fps', 0.0):.1f}` • EAR: `{last_dict.get('ear', None)}`")


def render_run(last_dict: dict):
    t = float(last_dict.get("t", 0.0))
    p = min(1.0, t / max(1e-6, RUN_SECONDS))
    ph_progress_label.caption(f"Assessing… {int(p*100)}% (about {int(max(0.0, RUN_SECONDS - t))}s left)")
    ph_progress.progress(p)

    ph_task.info(f"{focus_text(t)}\n\n*{micro_prompt(t)}*")

    comfort = last_dict.get("comfort")
    if comfort is None:
        ph_comfort.markdown("### Comfort score: **—**")
    else:
        band = str(last_dict.get("comfort_band", ""))
        emoji = str(last_dict.get("comfort_emoji", ""))
        ph_comfort.markdown(f"### **{int(comfort)}/100**  {emoji} **{band}**")

    title, body = _break_friendly(str(last_dict.get("break_mode", "WORK")), float(last_dict.get("break_remaining_sec", 0.0)))
    ph_break.info(f"**What to do now:** {title}\n\n{body}")

    # Live recommendations (fix: update continuously)
    level = last_dict.get("risk_level") or "LOW"
    score = float(last_dict.get("risk_score") or 0.0)
    perclos = float(last_dict.get("perclos") or 0.0)
    blink_rate = float(last_dict.get("blink_rate") or 0.0)
    cnn_ewma = last_dict.get("cnn_ewma")
    trend_cnn = last_dict.get("cnn_trend")

    advice_live = recommendations_dynamic(
        level=str(level),
        score01=float(score),
        perclos=float(perclos),
        blink_rate=float(blink_rate),
        cnn_sleepy=None if cnn_ewma is None else float(cnn_ewma),
        baseline_blink_rate=None,
        ear_drop_ratio=float(last_dict.get("ear_drop_ratio") or 1.0),
        trend_cnn=None if trend_cnn is None else float(trend_cnn),
    )

    ph_reco.markdown(f"**{advice_live.title}**")
    for b in advice_live.bullets[:4]:
        ph_reco.write("• " + b)

    msw = int(last_dict.get("microsleeps_window", 0))
    ph_stats.write(
        f"FPS: `{last_dict.get('fps', 0.0):.1f}`  \n"
        f"Blink/min: `{blink_rate:.0f}`  \n"
        f"Eye closure: `{perclos*100.0:.0f}%`  \n"
        f"Tiredness: `{(last_dict.get('fatigue_load') if last_dict.get('fatigue_load') is not None else '—')}`  \n"
        f"Microsleeps (window): `{msw}`"
    )

    ph_hint.caption("If accuracy feels off: increase lighting + keep your face closer and centered.")


# Render UI by phase
if st.session_state.ui_phase == "IDLE":
    render_idle()
elif st.session_state.ui_phase == "CALIB":
    render_calib(last)
elif st.session_state.ui_phase == "RUN":
    render_run(last)
elif st.session_state.ui_phase == "DONE":
    ph_task.success("Assessment complete. Generating report…")
    ph_progress_label.caption("Finalizing…")
    ph_progress.progress(1.0)


# ---------------- Report saving (UI thread only, non-sticky) ----------------
if st.session_state.ui_phase == "DONE":
    rows = processor.pop_rows()
    if rows:
        df = pd.DataFrame(rows)
        st.session_state.rows = rows
    else:
        df = pd.DataFrame(st.session_state.rows)

    if df.empty:
        ph_report.error("No frames captured. Try again with better lighting and face centered.")
        st.stop()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SESS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "assessment_results.csv"
    df.to_csv(csv_path, index=False)

    blink_final = float(df["blink_rate_bpm"].tail(1).iloc[0])
    perclos_final = float(df["perclos"].tail(1).iloc[0])

    cnn_final = df["cnn_sleepy_ewma"].dropna().tail(1)
    cnn_final_prob = float(cnn_final.iloc[0]) if len(cnn_final) else None

    level_final, score_final, reasons_final = fuse_risk(
        perclos=perclos_final,
        blink_rate=blink_final,
        cnn_prob=cnn_final_prob,
        lstm_prob=None,
    )

    advice = recommendations_dynamic(
        level=level_final,
        score01=float(score_final),
        perclos=float(perclos_final),
        blink_rate=float(blink_final),
        cnn_sleepy=cnn_final_prob,
        baseline_blink_rate=None,
        ear_drop_ratio=float(df["ear_drop_ratio"].dropna().mean()) if len(df["ear_drop_ratio"].dropna()) else 1.0,
        trend_cnn=None,
    )

    final_fatigue_load = float(df["fatigue_load"].tail(1).iloc[0]) if "fatigue_load" in df.columns else None
    microsleeps_win = int(df["microsleep_count_window"].tail(1).iloc[0]) if "microsleep_count_window" in df.columns else 0
    final_comfort = int(df["comfort_score_100"].tail(1).iloc[0]) if "comfort_score_100" in df.columns else comfort_score_from_risk(float(score_final))

    session_payload = {
        "user": st.session_state.run_cfg.user_name,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "baseline_ear": st.session_state.baseline_ear,
        "ear_threshold": st.session_state.ear_threshold,
        "final_perclos": float(perclos_final),
        "final_blink_rate": float(blink_final),
        "final_cnn_sleepy_prob": None if cnn_final_prob is None else float(cnn_final_prob),
        "final_risk_level": level_final,
        "final_risk_score": float(score_final),
        "final_comfort_score_100": int(final_comfort),
        "final_fatigue_load": None if final_fatigue_load is None else float(final_fatigue_load),
        "microsleeps_2min": int(microsleeps_win),
        "advice_title": advice.title,
        "fusion_reasons": reasons_final,
        "app_version": APP_VERSION,
    }

    session_path = SESS_DIR / f"{st.session_state.run_cfg.user_name}_{session_payload['timestamp']}.json"
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_payload, f, indent=2)

    ph_report.markdown(
        f"""
### ✅ 60s Assessment Summary
- **Comfort score:** `{final_comfort}/100`
- **Blinking (per min):** `{blink_final:.0f}`
- **Eye closure (recent):** `{(perclos_final * 100.0):.0f}%`
- **Tiredness level:** `{('N/A' if final_fatigue_load is None else f'{final_fatigue_load:.2f}')}`
- **Microsleeps (window):** `{microsleeps_win}`

### {advice.title}
"""
    )
    for b in advice.bullets:
        st.write("• " + b)

    st.download_button(
        "⬇ Download CSV Results",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="assessment_results.csv",
        mime="text/csv",
    )

    # After report is generated, go back to IDLE-ready state (no stuck UI)
    st.info("Done. Click **Start 60s Assessment** to run again.")