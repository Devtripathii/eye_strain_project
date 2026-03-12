from __future__ import annotations

import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Directories ───────────────────────────────────────────────────────────────
MODELS_DIR   = Path(os.getenv("EYEGUARD_MODELS_DIR",  str(PROJECT_ROOT / "models")))
DATA_DIR     = Path(os.getenv("EYEGUARD_DATA_DIR",    str(PROJECT_ROOT / "data")))
OUTPUTS_DIR  = Path(os.getenv("EYEGUARD_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs")))
SESSIONS_DIR = OUTPUTS_DIR / "sessions"
PROFILES_DIR = DATA_DIR / "user_profiles"

# ── Model files ───────────────────────────────────────────────────────────────
CNN_MODEL_PATH  = MODELS_DIR / os.getenv("EYEGUARD_CNN_MODEL",  "eye_model_best.pth")
FACE_MODEL_PATH = MODELS_DIR / os.getenv("EYEGUARD_FACE_MODEL", "face_landmarker.task")

# ── Session timing ────────────────────────────────────────────────────────────
TOTAL_SECONDS = float(os.getenv("EYEGUARD_SESSION_SEC", "60"))
CALIB_SECONDS = float(os.getenv("EYEGUARD_CALIB_SEC",   "10"))
RUN_SECONDS   = TOTAL_SECONDS - CALIB_SECONDS

# ── WebRTC ────────────────────────────────────────────────────────────────────
WEBRTC_KEY  = os.getenv("EYEGUARD_WEBRTC_KEY", "eye-comfort-stable")
STUN_SERVER = os.getenv("EYEGUARD_STUN",       "stun:stun.l.google.com:19302")

# ── UI loop ───────────────────────────────────────────────────────────────────
LOOP_SLEEP_SEC   = float(os.getenv("EYEGUARD_LOOP_SLEEP",   "0.25"))
CAMERA_GRACE_SEC = float(os.getenv("EYEGUARD_CAMERA_GRACE", "5.0"))

# ── EAR / calibration ─────────────────────────────────────────────────────────
EAR_SMOOTHER_ALPHA       = float(os.getenv("EYEGUARD_EAR_ALPHA",      "0.25"))
CALIB_TRIM_PCT           = float(os.getenv("EYEGUARD_CALIB_TRIM",     "0.05"))
# FIX: restored to 0.68 (was incorrectly bumped to 0.72).
# At baseline ~0.265: 0.72 gives threshold=0.191 which is ABOVE user's
# min open-eye EAR of 0.188 → false closures. 0.68 gives 0.180 → safe.
CALIB_THRESHOLD_MULT     = float(os.getenv("EYEGUARD_CALIB_MULT",     "0.68"))
CALIB_HEADROOM_MIN       = float(os.getenv("EYEGUARD_HEADROOM_MIN",   "0.06"))
CALIB_THRESHOLD_MIN      = float(os.getenv("EYEGUARD_THRESH_MIN",     "0.08"))
CALIB_THRESHOLD_MAX_MULT = float(os.getenv("EYEGUARD_THRESH_MAX",     "0.75"))
BLINK_HYSTERESIS         = float(os.getenv("EYEGUARD_HYSTERESIS",     "0.01"))

# ── CNN inference ─────────────────────────────────────────────────────────────
CNN_EVERY_N_FRAMES = int(os.getenv("EYEGUARD_CNN_FRAMES",  "8"))
CNN_EWMA_ALPHA     = float(os.getenv("EYEGUARD_CNN_ALPHA", "0.18"))

# ── Fatigue load ODE ──────────────────────────────────────────────────────────
# FIX: k_up restored to 0.020 (was bumped to 0.035).
# At 0.035 with moderate risk the load hits micro_lo=0.25 in ~7s,
# triggering micro-breaks almost immediately. 0.020 gives ~12s which
# is still responsive but avoids false alarms in a 50s assessment.
FATIGUE_K_UP   = float(os.getenv("EYEGUARD_K_UP",   "0.020"))
FATIGUE_K_DOWN = float(os.getenv("EYEGUARD_K_DOWN", "0.060"))
FATIGUE_K_LEAK = float(os.getenv("EYEGUARD_K_LEAK", "0.003"))

# ── Break scheduler ───────────────────────────────────────────────────────────
MICROSLEEP_SEC         = float(os.getenv("EYEGUARD_MICROSLEEP_SEC", "0.50"))
MICROSLEEP_WINDOW_SEC  = float(os.getenv("EYEGUARD_MS_WINDOW",      "120.0"))
MICROSLEEP_SHORT_COUNT = int(os.getenv("EYEGUARD_MS_SHORT_COUNT",   "1"))
MICROSLEEP_LONG_COUNT  = int(os.getenv("EYEGUARD_MS_LONG_COUNT",    "2"))
BREAK_MICRO_SEC        = int(os.getenv("EYEGUARD_BREAK_MICRO",      "20"))
BREAK_SHORT_SEC        = int(os.getenv("EYEGUARD_BREAK_SHORT",      "180"))
BREAK_LONG_SEC         = int(os.getenv("EYEGUARD_BREAK_LONG",       "600"))
BREAK_MIN_GAP_SEC      = float(os.getenv("EYEGUARD_BREAK_MIN_GAP",  "60.0"))

# ── UserBaseline ──────────────────────────────────────────────────────────────
BASELINE_ALPHA        = float(os.getenv("EYEGUARD_BASELINE_ALPHA",    "0.15"))
BASELINE_MIN_DURATION = int(os.getenv("EYEGUARD_BASELINE_MIN_DUR",    "20"))
BASELINE_MIN_FRAMES   = int(os.getenv("EYEGUARD_BASELINE_MIN_FRAMES", "100"))

# ── App metadata ──────────────────────────────────────────────────────────────
APP_VERSION = "phase2-v5"
APP_TITLE   = "EyeGuard"