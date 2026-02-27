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

# Auto-refresh (prevents UI freezing at 40–50%)
from streamlit_autorefresh import st_autorefresh

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
st.set_page_config(page_title="Eye Comfort Assistant", layout="wide")
st.title("👁️ Eye Comfort Assistant")
st.caption("One-click assessment (camera permission may require one browser click). Not a medical diagnosis.")
APP_VERSION = "simple-oneclick-stable-v1"
st.caption(f"Build: `{APP_VERSION}`")

# ---------------- Constants ----------------
TOTAL_SECONDS = 60.0
CALIB_SECONDS = 10.0
RUN_SECONDS = TOTAL_SECONDS - CALIB_SECONDS

WEBRTC_KEY = "eye-comfort-stable"  # must be constant

FOCUS_TEXTS = [
    "Read naturally. Don’t force blinking.",
    "Relax your face and shoulders.",
    "Keep the screen slightly below eye level.",
    "Soften your gaze and blink normally.",
]
PROMPTS = [
    "Keep reading naturally.",
    "Relax your jaw and shoulders.",
    "Avoid squinting—adjust brightness if needed.",
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


def _break_friendly(mode: str, remaining_sec: float) -> tuple[str, str]:
    mode = (mode or "WORK").upper()
    if mode == "WORK":
        return ("✅ Continue", "You can continue. Blink normally and keep relaxed posture.")
    if mode == "MICRO":
        return ("🟡 Quick break", f"Look far away and relax.\n\nTime left: **{remaining_sec:.0f}s**")
    if mode == "SHORT":
        return ("🟠 Short break", f"Stand, stretch, drink water.\n\nTime left: **{remaining_sec:.0f}s**")
    if mode == "LONG":
        return ("🔴 Long break", f"Step away from the screen.\n\nTime left: **{remaining_sec:.0f}s**")
    return ("✅ Continue", "You can continue.")


# ---------------- Paths ----------------
OUT_DIR = PROJECT_ROOT / "outputs"
SESS_DIR = OUT_DIR / "sessions"
PROFILE_DIR = PROJECT_ROOT / "data" / "user_profiles"
MODELS_DIR = PROJECT_ROOT / "models"
CNN_PATH = MODELS_DIR / "eye_model_best.pth"


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


# ---------------- Processor (NO session_state access inside) ----------------
class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()

        self.phase = "IDLE"   # IDLE / CALIB / RUN / DONE
        self.cfg: Optional[RuntimeCfg] = None

        self.last: dict[str, Any] = {
            "ok": False,
            "msg": "Waiting for camera...",
            "event": None,  # CALIB_DONE / DONE
            "event_payload": None,

            "t": 0.0,
            "fps": 0.0,
            "dt": 0.0,

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

        self._calib_start_ts: Optional[float] = None
        self._run_start_ts: Optional[float] = None

        self._calib_ears: list[float] = []
        self._rows: list[dict] = []

        self.blink = None
        self.fatigue = None
        self.fatigue_load = None
        self.scheduler = None
        self.blink_dur = None

        self.cnn = None
        self.cnn_ewma = None
        self.cnn_hist = deque(maxlen=25)
        self.cnn_frames = 0
        self.last_roi = None

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

    def _update_last(self, **kwargs):
        with self.lock:
            self.last.update(kwargs)

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

        if phase == "IDLE":
            self._update_last(ok=True, msg="Ready", t=0.0, dt=dt, fps=fps, ear=ear)
            return frame

        if phase == "CALIB":
            if self._calib_start_ts is None:
                self._calib_start_ts = time.time()
                self._calib_ears = []
                self._rows = []
                self.cnn_ewma = None
                self.cnn_hist.clear()
                self._update_last(event=None, event_payload=None, baseline_ear=None, ear_threshold=None)

            t_cal = time.time() - float(self._calib_start_ts)
            self._calib_ears.append(float(ear))

            p = min(1.0, t_cal / max(1e-6, CALIB_SECONDS))
            self._update_last(ok=True, msg=f"Calibrating… {int(p*100)}%", t=t_cal, dt=dt, fps=fps, ear=ear)

            if t_cal >= CALIB_SECONDS and len(self._calib_ears) >= 20:
                ears = sorted(self._calib_ears)
                baseline = float(ears[len(ears) // 2])  # median
                q25 = float(ears[int(0.25 * (len(ears) - 1))])
                threshold = float(max(0.10, q25))

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

        if phase == "RUN":
            if self._run_start_ts is None:
                self._run_start_ts = time.time()

            t_run = time.time() - float(self._run_start_ts)

            baseline_ear = self.get_last().get("baseline_ear")
            ear_threshold = self.get_last().get("ear_threshold")
            if ear_threshold is None:
                self._update_last(ok=False, msg="Missing calibration.", t=t_run, dt=dt, fps=fps, ear=ear)
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

            if t_run >= RUN_SECONDS:
                self._update_last(event="DONE", event_payload={"reason": "duration_reached"})
                with self.lock:
                    self.phase = "DONE"

            return frame

        self._update_last(ok=True, msg="Done", dt=dt, fps=fps, ear=ear)
        return frame


# ---------------- Session state (UI only) ----------------
if "ui_phase" not in st.session_state:
    st.session_state.ui_phase = "IDLE"
if "pending_start" not in st.session_state:
    st.session_state.pending_start = False
if "run_cfg" not in st.session_state:
    st.session_state.run_cfg = None
if "report_done" not in st.session_state:
    st.session_state.report_done = False
if "baseline_ear" not in st.session_state:
    st.session_state.baseline_ear = None
if "ear_threshold" not in st.session_state:
    st.session_state.ear_threshold = None
if "rows_cache" not in st.session_state:
    st.session_state.rows_cache = []


# ---------------- Minimal settings (kept simple) ----------------
c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
with c1:
    user_name = st.text_input("Name", value="User")
with c2:
    enable_cnn = st.checkbox("Use CNN", value=True)
with c3:
    enable_scheduler = st.checkbox("Smart breaks", value=True)

cfg_now = RuntimeCfg(
    user_name=user_name.strip() or "User",
    enable_cnn=bool(enable_cnn),
    cnn_every_n_frames=10,
    ewma_alpha=0.18,
    enable_scheduler=bool(enable_scheduler),
    microsleep_sec=0.50,
    microsleep_window_sec=120.0,
    microsleep_short_count=1,
    microsleep_long_count=2,
    micro_sec=20,
    short_sec=180,
    long_sec=600,
    min_gap=60.0,
    k_up=0.020,
    k_down=0.050,
    k_leak=0.003,
)

if st.session_state.run_cfg is None:
    st.session_state.run_cfg = cfg_now


# ---------------- Camera (always present) ----------------
st.markdown("---")
st.subheader("Camera")

st.caption(
    "If your browser asks for permission, allow it. "
    "On the very first use you may need to press the camera Start once (browser security). "
    "After that, the assessment can be truly one-click."
)

webrtc_ctx = webrtc_streamer(
    key=WEBRTC_KEY,
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EyeProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

processor = webrtc_ctx.video_processor
camera_ready = bool(getattr(webrtc_ctx, "state", None) and webrtc_ctx.state.playing and processor is not None)

if not camera_ready:
    st.warning("Camera is not running yet. If you see a Start button in the camera widget, click it once and allow permission.")


# ---------------- One-click assessment control ----------------
st.markdown("---")
st.subheader("Assessment")

btn1, btn2 = st.columns([1.2, 1.2])
with btn1:
    if st.button("▶ Start 60s Assessment", type="primary", use_container_width=True):
        st.session_state.report_done = False
        st.session_state.rows_cache = []
        st.session_state.baseline_ear = None
        st.session_state.ear_threshold = None
        st.session_state.run_cfg = cfg_now  # freeze
        st.session_state.pending_start = True

with btn2:
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.pending_start = False
        st.session_state.ui_phase = "IDLE"
        if processor is not None:
            processor.set_phase("IDLE")

# AUTO-START logic: user clicks assessment once, then as soon as camera is ready, we start calibration automatically.
if st.session_state.pending_start and camera_ready and st.session_state.ui_phase == "IDLE":
    st.session_state.ui_phase = "CALIB"
    st.session_state.pending_start = False

# Push cfg + phase to processor (if available)
if processor is not None:
    processor.set_cfg(st.session_state.run_cfg)
    processor.set_phase(st.session_state.ui_phase)

# During CALIB/RUN, force periodic reruns so UI never freezes
if st.session_state.ui_phase in ("CALIB", "RUN"):
    st_autorefresh(interval=250, key="ui_refresh_250ms")

# Read processor snapshot + handle events
last = processor.get_last() if processor is not None else {"t": 0.0, "msg": "Camera not started"}
ev, payload = processor.pop_event() if processor is not None else (None, None)

if ev == "CALIB_DONE" and payload:
    st.session_state.baseline_ear = float(payload["baseline_ear"])
    st.session_state.ear_threshold = float(payload["ear_threshold"])

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILE_DIR / f"{st.session_state.run_cfg.user_name}.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(
            {"user": st.session_state.run_cfg.user_name, "baseline_ear": st.session_state.baseline_ear, "ear_threshold": st.session_state.ear_threshold},
            f,
            indent=2,
        )

    st.session_state.ui_phase = "RUN"
    if processor is not None:
        processor.set_phase("RUN")

if ev == "DONE":
    st.session_state.ui_phase = "DONE"
    if processor is not None:
        processor.set_phase("DONE")


# ---------------- Live UI ----------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Focus task")
    if st.session_state.ui_phase == "IDLE":
        st.info("Click **Start 60s Assessment**.")
    elif st.session_state.ui_phase == "CALIB":
        st.info("Calibration (10s)… Look naturally at the screen.")
    elif st.session_state.ui_phase == "RUN":
        t = float(last.get("t", 0.0))
        st.info(f"{focus_text(t)}\n\n*{micro_prompt(t)}*")
    else:
        st.success("Assessment complete.")

    st.subheader("Progress")
    if st.session_state.ui_phase == "CALIB":
        p = min(1.0, float(last.get("t", 0.0)) / CALIB_SECONDS)
        st.progress(p, text=f"Calibrating… {int(p*100)}%")
    elif st.session_state.ui_phase == "RUN":
        p = min(1.0, float(last.get("t", 0.0)) / RUN_SECONDS)
        remain = int(max(0.0, RUN_SECONDS - float(last.get("t", 0.0))))
        st.progress(p, text=f"Assessing… {int(p*100)}% • ~{remain}s left")
    elif st.session_state.ui_phase == "DONE":
        st.progress(1.0, text="Done")
    else:
        st.progress(0.0, text="Idle")

with right:
    st.subheader("Live status")
    comfort = last.get("comfort")
    if comfort is None:
        st.metric("Comfort", "—")
    else:
        st.metric("Comfort", f"{int(comfort)}/100")

    st.write(f"Message: **{last.get('msg', '')}**")
    st.write(f"FPS: `{float(last.get('fps', 0.0)):.1f}`")

    st.subheader("Recommendations")
    if st.session_state.ui_phase == "RUN":
        level = last.get("risk_level") or "LOW"
        score = float(last.get("risk_score") or 0.0)
        perclos = float(last.get("perclos") or 0.0)
        blink_rate = float(last.get("blink_rate") or 0.0)
        cnn_ewma = last.get("cnn_ewma")
        trend_cnn = last.get("cnn_trend")

        advice_live = recommendations_dynamic(
            level=str(level),
            score01=float(score),
            perclos=float(perclos),
            blink_rate=float(blink_rate),
            cnn_sleepy=None if cnn_ewma is None else float(cnn_ewma),
            baseline_blink_rate=None,
            ear_drop_ratio=float(last.get("ear_drop_ratio") or 1.0),
            trend_cnn=None if trend_cnn is None else float(trend_cnn),
        )
        st.write(f"**{advice_live.title}**")
        for b in advice_live.bullets[:4]:
            st.write("• " + b)
    else:
        st.info("Recommendations appear during assessment.")


# ---------------- Report (generated once) ----------------
st.markdown("---")
st.subheader("Report")

if st.session_state.ui_phase == "DONE" and processor is not None and not st.session_state.report_done:
    rows = processor.pop_rows()
    if rows:
        df = pd.DataFrame(rows)
        st.session_state.rows_cache = rows
    else:
        df = pd.DataFrame(st.session_state.rows_cache)

    if df.empty:
        st.error("No frames captured. Improve lighting and keep face centered.")
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        SESS_DIR.mkdir(parents=True, exist_ok=True)

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

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        session_payload = {
            "user": st.session_state.run_cfg.user_name,
            "timestamp": timestamp,
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

        session_path = SESS_DIR / f"{st.session_state.run_cfg.user_name}_{timestamp}.json"
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_payload, f, indent=2)

        df_csv = df.to_csv(index=False).encode("utf-8")

        st.success("Report generated ✅")
        st.write(f"**Comfort:** `{final_comfort}/100`")
        st.write(f"**Blink/min:** `{blink_final:.0f}`")
        st.write(f"**Eye closure:** `{(perclos_final * 100.0):.0f}%`")
        st.write(f"**Tiredness:** `{('N/A' if final_fatigue_load is None else f'{final_fatigue_load:.2f}')}`")
        st.write(f"**Microsleeps (window):** `{microsleeps_win}`")

        st.markdown(f"### {advice.title}")
        for b in advice.bullets:
            st.write("• " + b)

        st.download_button("⬇ Download CSV", data=df_csv, file_name="assessment_results.csv", mime="text/csv")

        st.session_state.report_done = True

elif st.session_state.ui_phase != "DONE":
    st.info("Run the 60s assessment to generate the report.")