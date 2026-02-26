from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import threading
from collections import deque

import cv2
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
st.title("👁️ Eye Comfort Assistant (Browser Webcam)")
st.caption("Runs in the browser using WebRTC webcam. Not a medical diagnosis.")

# IMPORTANT: stable version marker so you can confirm cloud updated
APP_VERSION = "webrtc-stable-v2"
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


# ---------------- Session State Defaults ----------------
if "phase" not in st.session_state:
    st.session_state.phase = "IDLE"  # IDLE / CALIB / RUN / DONE
if "rows" not in st.session_state:
    st.session_state.rows = []
if "baseline_ear" not in st.session_state:
    st.session_state.baseline_ear = None
if "ear_threshold" not in st.session_state:
    st.session_state.ear_threshold = None
if "calib_start_ts" not in st.session_state:
    st.session_state.calib_start_ts = None
if "run_start_ts" not in st.session_state:
    st.session_state.run_start_ts = None

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


# ---------------- Video Processor ----------------
class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()

        self.last = {
            "ok": False,
            "msg": "Waiting for camera...",
            "t": 0.0,
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

    def _update_last(self, **kwargs):
        with self.lock:
            self.last.update(kwargs)

    def get_last(self) -> dict:
        with self.lock:
            return dict(self.last)

    def pop_rows(self) -> list[dict]:
        with self.lock:
            return list(self._rows)

    def clear_rows(self):
        with self.lock:
            self._rows = []

    def _ensure_cnn(self, enable: bool):
        if not enable:
            return
        if self.cnn is not None:
            return
        if CNN_PATH.exists():
            self.cnn = TorchEyeCnn(CNN_PATH, grayscale=True, device="cpu")

    def _init_runtime(self, ear_threshold: float, cfg: dict):
        self.blink = BlinkCounter(float(ear_threshold))
        self.fatigue = FatigueMetrics(fps=20, window_seconds=30)

        self.fatigue_load = FatigueLoad(
            FatigueLoadParams(
                k_up=float(cfg["k_up"]),
                k_down=float(cfg["k_down"]),
                k_leak=float(cfg["k_leak"]),
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
                microsleep_window_sec=float(cfg["microsleep_window_sec"]),
                microsleep_short_break_count=int(cfg["microsleep_short_count"]),
                microsleep_long_break_count=int(cfg["microsleep_long_count"]),
                min_gap_between_breaks_sec=float(cfg["min_gap"]),
            ),
            durations=BreakDurations(
                micro_sec=int(cfg["micro_sec"]),
                short_sec=int(cfg["short_sec"]),
                long_sec=int(cfg["long_sec"]),
            ),
        )

        self.blink_dur = BlinkDurationTracker(
            microsleep_sec=float(cfg["microsleep_sec"]),
            microsleep_window_sec=float(cfg["microsleep_window_sec"]),
            max_dt=0.25,
        )

    def recv(self, frame):
        tick = self.timebase.tick()
        dt = float(tick.dt)
        fps = float(tick.fps)

        img_bgr = frame.to_ndarray(format="bgr24")

        cfg = st.session_state._cfg
        phase = st.session_state.phase

        lms = self.fm.get_landmarks(img_bgr)
        if lms is None:
            self._update_last(ok=False, msg="Face not detected. Center your face.", dt=dt, fps=fps)
            return frame

        h, w = img_bgr.shape[:2]
        ear = float(ear_both_eyes(lms, w, h))

        if phase == "CALIB":
            if st.session_state.calib_start_ts is None:
                st.session_state.calib_start_ts = time.time()
                self._calib_ears = []
                self.clear_rows()
                self.cnn_ewma = None
                self.cnn_hist.clear()

            t_cal = time.time() - float(st.session_state.calib_start_ts)
            self._calib_ears.append(float(ear))

            progress = min(1.0, t_cal / max(1e-6, float(cfg["calib_sec"])))

            self._update_last(
                ok=True,
                msg=f"Calibrating... {progress*100:.0f}%",
                t=float(t_cal),
                dt=dt,
                fps=fps,
                ear=float(ear),
            )

            if t_cal >= float(cfg["calib_sec"]) and len(self._calib_ears) >= 10:
                baseline = float(sum(self._calib_ears) / len(self._calib_ears))
                threshold = float(baseline * 0.75)

                st.session_state.baseline_ear = baseline
                st.session_state.ear_threshold = threshold

                PROFILE_DIR.mkdir(parents=True, exist_ok=True)
                profile_path = PROFILE_DIR / f"{cfg['user_name']}.json"
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump({"user": cfg["user_name"], "baseline_ear": baseline, "ear_threshold": threshold}, f, indent=2)

                self._init_runtime(ear_threshold=threshold, cfg=cfg)
                st.session_state.phase = "RUN"
                st.session_state.run_start_ts = time.time()
                st.session_state.calib_start_ts = None

            return frame

        if phase == "RUN":
            if st.session_state.run_start_ts is None:
                st.session_state.run_start_ts = time.time()

            t_run = time.time() - float(st.session_state.run_start_ts)

            ear_threshold = st.session_state.ear_threshold
            baseline_ear = st.session_state.baseline_ear
            if ear_threshold is None:
                self._update_last(ok=False, msg="Missing calibration. Press Start again.", dt=dt, fps=fps)
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

            self._ensure_cnn(bool(cfg["enable_cnn"]))
            cnn_prob = None
            if cfg["enable_cnn"] and self.cnn is not None:
                roi = crop_both_eyes(img_bgr, lms, pad=0.55)
                if roi is not None:
                    self.last_roi = roi

                self.cnn_frames += 1
                if self.last_roi is not None and (self.cnn_frames % int(cfg["cnn_every_n_frames"]) == 0):
                    res = self.cnn.predict_roi_bgr(self.last_roi)
                    if res is not None:
                        cnn_prob = float(res.sleepy_prob)
                        self.cnn_hist.append(cnn_prob)
                        if self.cnn_ewma is None:
                            self.cnn_ewma = cnn_prob
                        else:
                            a = float(cfg["ewma_alpha"])
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

            if cfg["enable_scheduler"]:
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
                msg="Running",
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

            if t_run >= float(cfg["duration_sec"]):
                st.session_state.phase = "DONE"

            return frame

        self._update_last(ok=True, msg="Idle", dt=dt, fps=fps, ear=float(ear))
        return frame


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Quick Settings")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    existing_profiles = sorted([p.stem for p in PROFILE_DIR.glob("*.json")])
    user_name = st.selectbox("Profile", ["(new user)"] + existing_profiles)
    if user_name == "(new user)":
        user_name = st.text_input("Enter your name", value="User").strip() or "User"

    duration_sec = st.slider("Test duration (sec)", 20, 180, 60, 5)
    calib_sec = st.slider("Calibration (sec)", 6, 20, 10, 1)

    st.markdown("---")
    st.subheader("Advanced (optional)")

    with st.expander("Model + Smoothing", expanded=False):
        enable_cnn = st.checkbox("Use CNN eye-closure model", value=True)
        cnn_every_n_frames = st.slider("CNN frequency (frames)", 1, 25, 10, 1)
        ewma_alpha = st.slider("CNN smoothing", 0.05, 0.50, 0.18, 0.01)

    with st.expander("Smart breaks", expanded=False):
        enable_scheduler = st.checkbox("Enable smart breaks", value=True)

        k_up = st.slider("Fatigue builds speed", 0.005, 0.050, 0.020, 0.001)
        k_down = st.slider("Recovery speed", 0.010, 0.150, 0.050, 0.001)
        k_leak = st.slider("Natural recovery", 0.000, 0.020, 0.003, 0.001)

        microsleep_sec = st.slider("Microsleep threshold (sec)", 0.20, 1.50, 0.50, 0.05)
        microsleep_window_sec = st.slider("Microsleep window (sec)", 30, 300, 120, 10)
        microsleep_short_count = st.slider("Microsleeps → short break", 1, 5, 1, 1)
        microsleep_long_count = st.slider("Microsleeps → long break", 2, 10, 2, 1)

        micro_sec = st.slider("Quick break length (sec)", 10, 90, 20, 5)
        short_sec = st.slider("Short break length (sec)", 60, 600, 180, 30)
        long_sec = st.slider("Long break length (sec)", 120, 1200, 600, 60)

        min_gap = st.slider("Min time between breaks (sec)", 0, 300, 60, 10)

    with st.expander("History", expanded=False):
        if st.checkbox("Show recent sessions table", value=True):
            hist = load_session_files()
            if hist:
                st.dataframe(pd.DataFrame(hist), width="stretch")
            else:
                st.caption("No sessions found yet.")


st.session_state._cfg = {
    "user_name": user_name,
    "duration_sec": float(duration_sec),
    "calib_sec": float(calib_sec),
    "enable_cnn": bool(enable_cnn),
    "cnn_every_n_frames": int(cnn_every_n_frames),
    "ewma_alpha": float(ewma_alpha),
    "enable_scheduler": bool(enable_scheduler),
    "k_up": float(k_up),
    "k_down": float(k_down),
    "k_leak": float(k_leak),
    "microsleep_sec": float(microsleep_sec),
    "microsleep_window_sec": float(microsleep_window_sec),
    "microsleep_short_count": int(microsleep_short_count),
    "microsleep_long_count": int(microsleep_long_count),
    "micro_sec": int(micro_sec),
    "short_sec": int(short_sec),
    "long_sec": int(long_sec),
    "min_gap": float(min_gap),
}


# ---------------- Top controls ----------------
bar1, bar2, bar3 = st.columns([1.4, 1.4, 2.2])

with bar1:
    # NOTE: no st.rerun() here (Streamlit reruns automatically on click)
    if st.session_state.phase in ("IDLE", "DONE"):
        if st.button("▶ Start assessment", type="primary", use_container_width=True):
            st.session_state.phase = "CALIB"
            st.session_state.rows = []
            st.session_state.baseline_ear = None
            st.session_state.ear_threshold = None
            st.session_state.calib_start_ts = None
            st.session_state.run_start_ts = None
    else:
        st.button("▶ Start assessment", disabled=True, use_container_width=True)

with bar2:
    if st.session_state.phase in ("CALIB", "RUN"):
        if st.button("⏸ Stop (go idle)", use_container_width=True):
            st.session_state.phase = "IDLE"
            st.session_state.calib_start_ts = None
            st.session_state.run_start_ts = None
    else:
        st.button("⏸ Stop (go idle)", disabled=True, use_container_width=True)

with bar3:
    st.markdown(f"**Legend:** {comfort_legend()}")

st.markdown("---")


# ---------------- WebRTC Stream (always on, called exactly once) ----------------
st.subheader("📷 Webcam")
st.caption("Allow camera permission. Keep webcam running; press Start assessment when ready.")

webrtc_ctx = webrtc_streamer(
    key=WEBRTC_KEY,
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EyeProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

processor = webrtc_ctx.video_processor


# ---------------- UI Panels ----------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Task")
    ph_task = st.empty()

with right:
    st.subheader("Your status")
    ph_comfort = st.empty()
    ph_comfort_bar = st.empty()
    ph_break = st.empty()

    st.subheader("Quick stats")
    ph_blink = st.empty()
    ph_closed = st.empty()
    ph_tired = st.empty()
    ph_hint = st.empty()

st.markdown("---")
st.subheader("Report")
ph_report = st.empty()


def _render_idle():
    ph_task.info("Press **Start assessment** above. Keep your face centered in the webcam.")
    ph_comfort.markdown("### Comfort score: **—**")
    ph_comfort_bar.progress(0)
    ph_break.info("No test running.")
    ph_blink.metric("Blinking (per min)", "—")
    ph_closed.metric("Eye closure (recent)", "—")
    ph_tired.metric("Tiredness level", "—")
    ph_hint.caption("Tip: Better lighting and a stable head position improves accuracy.")


def _render_live(last: dict):
    phase = st.session_state.phase

    if phase == "CALIB":
        ph_task.info("Calibration running… Keep your face centered and look at the screen naturally.")
        ph_break.info(last.get("msg", "Calibrating..."))
        t = float(last.get("t", 0.0))
        ph_comfort.markdown("### Comfort score: **—**")
        ph_comfort_bar.progress(min(1.0, t / max(1e-6, float(st.session_state._cfg["calib_sec"]))))
        ph_blink.metric("Blinking (per min)", "—")
        ph_closed.metric("Eye closure (recent)", "—")
        ph_tired.metric("Tiredness level", "—")
        ph_hint.caption("Hold steady. Avoid looking away during calibration.")
        return

    if phase == "RUN":
        t = float(last.get("t", 0.0))
        ph_task.info(f"{focus_text(t)}\n\n*{micro_prompt(t)}*")

        comfort = last.get("comfort")
        if comfort is None:
            ph_comfort.markdown("### Comfort score: **—**")
            ph_comfort_bar.progress(0)
        else:
            band = str(last.get("comfort_band", ""))
            emoji = str(last.get("comfort_emoji", ""))
            ph_comfort.markdown(f"### Comfort score: **{int(comfort)}/100**  {emoji} **{band}**")
            ph_comfort_bar.progress(int(comfort) / 100.0)

        title, body = _break_friendly(str(last.get("break_mode", "WORK")), float(last.get("break_remaining_sec", 0.0)))
        ph_break.info(f"**What to do now:** {title}\n\n{body}")

        br = last.get("blink_rate")
        pc = last.get("perclos")
        F = last.get("fatigue_load")

        ph_blink.metric("Blinking (per min)", "—" if br is None else f"{float(br):.0f}")
        ph_closed.metric("Eye closure (recent)", "—" if pc is None else f"{float(pc)*100.0:.0f}%")
        ph_tired.metric("Tiredness level", "—" if F is None else f"{float(F):.2f}")

        msw = int(last.get("microsleeps_window", 0))
        ph_hint.caption(f"Tip: Low blinking + higher eye closure usually means fatigue. Microsleeps in window: {msw}")
        return

    if phase == "DONE":
        ph_task.success("Run finished. Saving report…")
        ph_hint.caption("You can press Start assessment again for a new session.")


if processor is None:
    _render_idle()
    st.stop()

last = processor.get_last()

if st.session_state.phase == "IDLE":
    _render_idle()
else:
    _render_live(last)

if st.session_state.phase == "DONE":
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
        "user": st.session_state._cfg["user_name"],
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "baseline_ear": None if st.session_state.baseline_ear is None else float(st.session_state.baseline_ear),
        "ear_threshold": None if st.session_state.ear_threshold is None else float(st.session_state.ear_threshold),
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

    session_path = SESS_DIR / f"{st.session_state._cfg['user_name']}_{session_payload['timestamp']}.json"
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_payload, f, indent=2)

    ph_report.markdown(
        f"""
### ✅ Summary
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

    st.info("Press **Start assessment** above to run again. If webcam freezes, refresh the page.")