from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
from collections import deque

import cv2
import streamlit as st
import pandas as pd

from src.core.timebase import Timebase
from src.cv.camera import SafeCamera
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
st.caption("Simple fatigue monitoring + smart breaks. Not a medical diagnosis.")


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


def bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


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
if "cam" not in st.session_state:
    st.session_state.cam = None
if "fm" not in st.session_state:
    st.session_state.fm = None
if "cnn" not in st.session_state:
    st.session_state.cnn = None
if "timebase" not in st.session_state:
    st.session_state.timebase = Timebase(fps_window=30, dt_max=0.25)

if "baseline_ear" not in st.session_state:
    st.session_state.baseline_ear = None
if "ear_threshold" not in st.session_state:
    st.session_state.ear_threshold = None

if "rows" not in st.session_state:
    st.session_state.rows = []
if "last_session_payload" not in st.session_state:
    st.session_state.last_session_payload = None


def ensure_resources(cam_index: int, enable_cnn: bool):
    if st.session_state.cam is None:
        st.session_state.cam = SafeCamera(int(cam_index))
    if st.session_state.fm is None:
        st.session_state.fm = FaceMeshLandmarker(min_det=0.25, min_track=0.25, refine=False)
    if enable_cnn and st.session_state.cnn is None and CNN_PATH.exists():
        st.session_state.cnn = TorchEyeCnn(CNN_PATH, grayscale=True, device="cpu")


def cleanup():
    try:
        if st.session_state.cam is not None:
            st.session_state.cam.release()
    except Exception:
        pass
    try:
        if st.session_state.fm is not None:
            st.session_state.fm.close()
    except Exception:
        pass
    st.session_state.cam = None
    st.session_state.fm = None
    st.session_state.cnn = None
    try:
        if st.session_state.timebase is not None:
            st.session_state.timebase.reset()
    except Exception:
        pass


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


# ---------------- Sidebar (Simple) ----------------
with st.sidebar:
    st.header("Quick Settings")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    existing_profiles = sorted([p.stem for p in PROFILE_DIR.glob("*.json")])
    user_name = st.selectbox("Profile", ["(new user)"] + existing_profiles)

    if user_name == "(new user)":
        user_name = st.text_input("Enter your name", value="User").strip() or "User"

    cam_index = st.number_input("Camera index", 0, 5, 0, 1)
    duration_sec = st.slider("Test duration (sec)", 20, 180, 60, 5)
    calib_sec = st.slider("Calibration (sec)", 6, 20, 10, 1)
    show_preview = st.checkbox("Show camera preview", value=True)

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
        show_history = st.checkbox("Show recent sessions table", value=True)
        if show_history:
            hist = load_session_files()
            if hist:
                st.dataframe(pd.DataFrame(hist), width="stretch")
            else:
                st.caption("No sessions found yet.")

# Defaults if advanced expander never opened
if "enable_cnn" not in locals():
    enable_cnn = True
    cnn_every_n_frames = 10
    ewma_alpha = 0.18

if "enable_scheduler" not in locals():
    enable_scheduler = True
    k_up, k_down, k_leak = 0.020, 0.050, 0.003
    microsleep_sec, microsleep_window_sec = 0.50, 120
    microsleep_short_count, microsleep_long_count = 1, 2
    micro_sec, short_sec, long_sec = 20, 180, 600
    min_gap = 60


# ---------------- Top Control Bar (Always Visible) ----------------
ensure_resources(cam_index=int(cam_index), enable_cnn=bool(enable_cnn))

bar1, bar2, bar3, bar4 = st.columns([1.2, 1.2, 1.2, 2.4])

with bar1:
    if st.session_state.phase in ("IDLE", "DONE"):
        if st.button("▶ Start", type="primary", use_container_width=True):
            st.session_state.phase = "CALIB"
            st.session_state.rows = []
            st.session_state.last_session_payload = None
            try:
                st.session_state.timebase.reset()
            except Exception:
                pass
            st.rerun()
    else:
        st.button("▶ Start", disabled=True, use_container_width=True)

with bar2:
    if st.session_state.phase in ("CALIB", "RUN"):
        if st.button("🛑 Stop", use_container_width=True):
            st.session_state.phase = "IDLE"
            cleanup()
            st.rerun()
    else:
        st.button("🛑 Stop", disabled=True, use_container_width=True)

with bar3:
    if st.button("⏹ Reset", use_container_width=True):
        st.session_state.phase = "IDLE"
        st.session_state.baseline_ear = None
        st.session_state.ear_threshold = None
        st.session_state.rows = []
        st.session_state.last_session_payload = None
        cleanup()
        st.rerun()

with bar4:
    if st.session_state.rows:
        last = st.session_state.rows[-1]
        comfort = int(last.get("comfort_score_100", 0))
        band, band_emoji = comfort_band(comfort)
        mode = str(last.get("break_mode", "WORK"))
        rem = float(last.get("break_remaining_sec", 0.0))
        title, _ = _break_friendly(mode, rem)
        st.markdown(f"**Comfort:** {comfort}/100 {band_emoji} {band}  •  **Now:** {title}")
    else:
        st.markdown(f"**Legend:** {comfort_legend()}")

st.markdown("---")


# ---------------- Main UI (Friendly) ----------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Task")
    ph_task = st.empty()

    st.subheader("Camera")
    ph_cam = st.empty()

with right:
    st.subheader("Your status")
    ph_comfort = st.empty()
    ph_comfort_bar = st.empty()
    ph_comfort_help = st.empty()
    ph_break = st.empty()

    st.subheader("Quick stats")
    ph_blink = st.empty()
    ph_blink_help = st.empty()

    ph_closed = st.empty()
    ph_closed_help = st.empty()

    ph_fatigue = st.empty()
    ph_fatigue_help = st.empty()

    with st.expander("Advanced details (optional)", expanded=False):
        ph_adv = st.empty()

st.markdown("---")
st.subheader("Report")
ph_report = st.empty()


# ---------------- IDLE Screen ----------------
if st.session_state.phase == "IDLE":
    ph_task.info("Press **Start** at the top. Keep your face centered and read naturally.")
    frame = st.session_state.cam.read()
    if frame is not None and show_preview:
        ph_cam.image(bgr_to_rgb(frame), channels="RGB", caption="Preview")

    ph_comfort.markdown("### Comfort score: **—**")
    ph_comfort_bar.progress(0)
    ph_comfort_help.caption("Comfort score goes down when your eyes look more tired.")
    ph_break.info("No test running.")

    ph_blink.metric("Blinking (per min)", "—")
    ph_blink_help.caption("Lower than usual can mean intense focus or eye strain.")
    ph_closed.metric("Eye closure (recent)", "—")
    ph_closed_help.caption("More closed time can mean fatigue (or blinking more).")
    ph_fatigue.metric("Tiredness level", "—")
    ph_fatigue_help.caption("Builds up over time, drops during breaks.")

    ph_adv.info("Advanced info will appear during the test.")
    st.stop()


# ---------------- CALIBRATION ----------------
if st.session_state.phase == "CALIB":
    ph_task.info(f"Calibration for **{calib_sec} seconds**… Keep your face centered and look naturally at the screen.")
    st.session_state.baseline_ear = None
    st.session_state.ear_threshold = None

    ear_samples = []
    t0 = time.time()

    while True:
        _ = st.session_state.timebase.tick()
        t = time.time() - t0
        if t >= float(calib_sec):
            break

        frame = st.session_state.cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        lms = st.session_state.fm.get_landmarks(frame)
        if lms is not None:
            h, w = frame.shape[:2]
            ear_samples.append(float(ear_both_eyes(lms, w, h)))

        if show_preview:
            ph_cam.image(bgr_to_rgb(frame), channels="RGB", caption="Calibration")

        ph_comfort.markdown("### Comfort score: **—**")
        ph_comfort_bar.progress(min(1.0, t / float(calib_sec)))
        ph_comfort_help.caption("Calibrating your normal eye openness baseline…")
        ph_break.info("Calibrating…")
        ph_blink.metric("Blinking (per min)", "—")
        ph_closed.metric("Eye closure (recent)", "—")
        ph_fatigue.metric("Tiredness level", "—")

        time.sleep(0.02)

    if len(ear_samples) < 10:
        st.session_state.phase = "IDLE"
        cleanup()
        st.error("❌ Calibration failed. Improve lighting and keep face centered.")
        st.stop()

    baseline_ear = sum(ear_samples) / len(ear_samples)

    calib_factor = 0.75
    ear_threshold = baseline_ear * float(calib_factor)

    st.session_state.baseline_ear = float(baseline_ear)
    st.session_state.ear_threshold = float(ear_threshold)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILE_DIR / f"{user_name}.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump({"user": user_name, "baseline_ear": float(baseline_ear), "ear_threshold": float(ear_threshold)}, f, indent=2)

    try:
        st.session_state.cam.reopen()
    except Exception:
        pass
    try:
        st.session_state.timebase.reset()
    except Exception:
        pass

    st.session_state.phase = "RUN"
    st.rerun()


# ---------------- RUN ----------------
if st.session_state.phase == "RUN":
    baseline_ear = float(st.session_state.baseline_ear or 0.0)
    ear_threshold = float(st.session_state.ear_threshold or 0.0)

    if baseline_ear <= 0.0 or ear_threshold <= 0.0:
        st.session_state.phase = "IDLE"
        st.error("Internal error: missing calibration. Please press Reset and Start again.")
        st.stop()

    blink = BlinkCounter(ear_threshold)
    fatigue = FatigueMetrics(fps=20, window_seconds=30)

    fatigue_load = FatigueLoad(
        FatigueLoadParams(
            k_up=float(k_up),
            k_down=float(k_down),
            k_leak=float(k_leak),
            max_dt=0.25,
        )
    )

    scheduler = BreakScheduler(
        thresholds=BreakThresholds(
            micro_lo=0.35,
            micro_hi=0.55,
            short_hi=0.55,
            long_hi=0.75,
            sustain_risk_hi=0.55,
            sustain_seconds=30.0,
            microsleep_window_sec=float(microsleep_window_sec),
            microsleep_short_break_count=int(microsleep_short_count),
            microsleep_long_break_count=int(microsleep_long_count),
            min_gap_between_breaks_sec=float(min_gap),
        ),
        durations=BreakDurations(
            micro_sec=int(micro_sec),
            short_sec=int(short_sec),
            long_sec=int(long_sec),
        ),
    )

    blink_dur = BlinkDurationTracker(
        microsleep_sec=float(microsleep_sec),
        microsleep_window_sec=float(microsleep_window_sec),
        max_dt=0.25,
    )

    cnn_ewma = None
    cnn_hist = deque(maxlen=25)
    cnn_frames = 0
    last_roi = None

    rows = list(st.session_state.rows)

    tA = time.time()
    while True:
        tick = st.session_state.timebase.tick()
        dt = float(tick.dt)
        fps = float(tick.fps)

        t = time.time() - tA
        if t >= float(duration_sec):
            break

        frame = st.session_state.cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        ph_task.info(f"{focus_text(t)}  \n\n*{micro_prompt(t)}*")

        lms = st.session_state.fm.get_landmarks(frame)
        if lms is None:
            if show_preview:
                ph_cam.image(bgr_to_rgb(frame), channels="RGB", caption="Assessment")
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        ear = float(ear_both_eyes(lms, w, h))
        ear_drop_ratio = ear / max(1e-6, float(baseline_ear))

        blink_count, closed, blink_occurred = blink.update(ear)
        perclos, blink_rate = fatigue.update(eye_closed=closed, t_sec=t, blink_occurred=blink_occurred)

        blink_event, ms_count = blink_dur.update(closed=bool(closed), dt=float(dt), t_sec=float(t))
        microsleep_event = bool(blink_event.is_microsleep) if blink_event is not None else False

        roi = crop_both_eyes(frame, lms, pad=0.55)
        if roi is not None:
            last_roi = roi

        cnn_prob = None
        if enable_cnn and st.session_state.cnn is not None and last_roi is not None:
            cnn_frames += 1
            if cnn_frames % int(cnn_every_n_frames) == 0:
                res = st.session_state.cnn.predict_roi_bgr(last_roi)
                if res is not None:
                    cnn_prob = float(res.sleepy_prob)
                    cnn_hist.append(cnn_prob)
                    if cnn_ewma is None:
                        cnn_ewma = cnn_prob
                    else:
                        a = float(ewma_alpha)
                        cnn_ewma = (a * cnn_prob) + ((1.0 - a) * cnn_ewma)

        trend_cnn = None
        if len(cnn_hist) >= 8:
            trend_cnn = float(cnn_hist[-1] - cnn_hist[0]) / max(1.0, len(cnn_hist))

        level_live, score_live, _ = fuse_risk(
            perclos=float(perclos),
            blink_rate=float(blink_rate),
            cnn_prob=float(cnn_ewma) if cnn_ewma is not None else None,
            lstm_prob=None,
        )

        if enable_scheduler:
            on_break = scheduler.is_on_break()
            F = fatigue_load.update(risk01=float(score_live), dt=float(dt), working=(not on_break), quality01=1.0)
            sched_state = scheduler.update(
                t_sec=float(t),
                dt=float(dt),
                risk01=float(score_live),
                fatigue_load01=float(F),
                microsleep_event=bool(microsleep_event),
            )
        else:
            F = fatigue_load.update(risk01=float(score_live), dt=float(dt), working=True, quality01=1.0)
            sched_state = scheduler.state

        break_mode = sched_state.mode
        break_remaining = float(sched_state.remaining_sec)

        comfort100 = comfort_score_from_risk(float(score_live))
        band, band_emoji = comfort_band(comfort100)

        title, body = _break_friendly(break_mode, break_remaining)

        # Friendly UI
        ph_comfort.markdown(f"### Comfort score: **{comfort100}/100**  {band_emoji} **{band}**")
        ph_comfort_bar.progress(comfort100 / 100.0)
        ph_comfort_help.caption(comfort_legend())

        ph_break.info(f"**What to do now:** {title}\n\n{body}")

        ph_blink.metric("Blinking (per min)", f"{blink_rate:.0f}")
        ph_blink_help.caption("Lower than usual can mean intense focus or eye strain.")

        ph_closed.metric("Eye closure (recent)", f"{(perclos * 100.0):.0f}%")
        ph_closed_help.caption("More closed time can mean fatigue (or longer blinks).")

        ph_fatigue.metric("Tiredness level", f"{F:.2f}")
        ph_fatigue_help.caption("Builds up over time, drops during breaks.")

        # Advanced
        last_ms = blink_dur.last_blink_ms()
        max_ms = blink_dur.max_blink_ms_recent()
        ph_adv.markdown(
            f"""
**Advanced**
- FPS: `{fps:.1f}` | dt: `{dt:.3f}`
- EAR drop: `{ear_drop_ratio:.2f}`
- CNN EWMA: `{('N/A' if cnn_ewma is None else f'{cnn_ewma:.2f}')}` | Trend: `{('N/A' if trend_cnn is None else f'{trend_cnn:.4f}')}`
- Last blink: `{('N/A' if last_ms is None else f'{last_ms:.0f} ms')}` | Max closure (recent): `{('N/A' if max_ms is None else f'{max_ms:.0f} ms')}`
- Microsleeps (window): `{ms_count}`
"""
        )

        if show_preview:
            ph_cam.image(bgr_to_rgb(frame), channels="RGB", caption="Assessment")

        rows.append({
            "t": round(float(t), 2),
            "dt": float(dt),
            "fps": float(fps),
            "ear": round(float(ear), 3),
            "ear_drop_ratio": float(ear_drop_ratio),
            "blink_count": int(blink_count),
            "blink_rate_bpm": float(blink_rate),
            "perclos": float(perclos),
            "blink_duration_ms": None if blink_event is None else float(blink_event.duration_sec * 1000.0),
            "max_blink_ms_recent": None if max_ms is None else float(max_ms),
            "microsleep_event": bool(microsleep_event),
            "microsleep_count_window": int(ms_count),
            "cnn_sleepy_prob": None if cnn_prob is None else float(cnn_prob),
            "cnn_sleepy_ewma": None if cnn_ewma is None else float(cnn_ewma),
            "cnn_trend": None if trend_cnn is None else float(trend_cnn),
            "risk_level_live": level_live,
            "risk_score_live": float(score_live),
            "comfort_score_100": int(comfort100),
            "fatigue_load": float(F),
            "break_mode": str(break_mode),
            "break_remaining_sec": float(break_remaining),
        })

        st.session_state.rows = rows
        time.sleep(0.02)

    st.session_state.phase = "DONE"
    st.rerun()


# ---------------- DONE ----------------
if st.session_state.phase == "DONE":
    rows = list(st.session_state.rows)
    if not rows:
        st.session_state.phase = "IDLE"
        st.warning("No data captured. Try again.")
        st.stop()

    df = pd.DataFrame(rows)

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

    baseline_blinks = df[df["t"] <= 10.0]["blink_rate_bpm"]
    baseline_blink_rate = float(baseline_blinks.mean()) if len(baseline_blinks) else None

    cnn_hist_final = df["cnn_sleepy_ewma"].dropna().tail(25)
    trend_final = None
    if len(cnn_hist_final) >= 8:
        trend_final = float(cnn_hist_final.iloc[-1] - cnn_hist_final.iloc[0]) / max(1.0, len(cnn_hist_final))

    advice = recommendations_dynamic(
        level=level_final,
        score01=float(score_final),
        perclos=float(perclos_final),
        blink_rate=float(blink_final),
        cnn_sleepy=cnn_final_prob,
        baseline_blink_rate=baseline_blink_rate,
        ear_drop_ratio=float(df["ear_drop_ratio"].mean()),
        trend_cnn=trend_final,
    )

    avg_fps = float(df["fps"].dropna().mean()) if len(df["fps"].dropna()) else None
    final_fatigue_load = float(df["fatigue_load"].tail(1).iloc[0]) if "fatigue_load" in df.columns else None
    microsleeps_win = int(df["microsleep_count_window"].tail(1).iloc[0]) if "microsleep_count_window" in df.columns else 0
    final_comfort = int(df["comfort_score_100"].tail(1).iloc[0]) if "comfort_score_100" in df.columns else comfort_score_from_risk(float(score_final))

    session_payload = {
        "user": user_name,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "baseline_ear": float(st.session_state.baseline_ear or 0.0),
        "ear_threshold": float(st.session_state.ear_threshold or 0.0),
        "final_perclos": float(perclos_final),
        "final_blink_rate": float(blink_final),
        "final_cnn_sleepy_prob": None if cnn_final_prob is None else float(cnn_final_prob),
        "final_risk_level": level_final,
        "final_risk_score": float(score_final),
        "final_comfort_score_100": int(final_comfort),
        "final_fatigue_load": None if final_fatigue_load is None else float(final_fatigue_load),
        "microsleeps_2min": int(microsleeps_win),
        "avg_fps": None if avg_fps is None else float(avg_fps),
        "advice_title": advice.title,
        "fusion_reasons": reasons_final,
    }

    session_path = SESS_DIR / f"{user_name}_{session_payload['timestamp']}.json"
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

    cleanup()
    st.info("Press **Start** at the top to run another test, or **Reset** to clear everything.")