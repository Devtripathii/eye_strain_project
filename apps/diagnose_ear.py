"""
Run this INSTEAD of user_app.py to diagnose your EAR values.
Usage: python -m streamlit run apps/diagnose_ear.py
Keep your eyes open naturally and watch the numbers for 30 seconds.
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import threading
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from src.cv.landmarks import FaceMeshLandmarker
from src.cv.features import ear_both_eyes, EARSmoother

st.set_page_config(page_title="EAR Diagnostic", layout="wide")
st.title("👁️ EAR Diagnostic Tool")
st.caption("Keep eyes OPEN naturally. Watch numbers for 20–30 seconds. Screenshot and share results.")

class DiagProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.fm   = FaceMeshLandmarker(min_det=0.5, min_track=0.5, refine=False)
        self.smoother = EARSmoother(alpha=0.25)
        self.history  = []   # last 200 EAR readings
        self.data = {"ear": None, "ear_raw": None, "fps": 0.0,
                     "face": False, "min_ear": None, "max_ear": None,
                     "mean_ear": None, "p20": None}
        self._frame_n = 0

    def recv(self, frame):
        import time, numpy as np
        img = frame.to_ndarray(format="bgr24")
        h0, w0 = img.shape[:2]
        # downscale
        if h0 > 480:
            scale = 480 / h0
            img   = cv2.resize(img, (int(w0*scale), 480))
        h, w = img.shape[:2]

        self._frame_n += 1
        lms = self.fm.get_landmarks(img)
        if lms is None:
            with self.lock:
                self.data["face"] = False
            return frame

        ear_raw = float(ear_both_eyes(lms, w, h))
        ear     = self.smoother.update(ear_raw)

        with self.lock:
            self.history.append(ear)
            if len(self.history) > 300:
                self.history = self.history[-300:]
            h_arr = sorted(self.history)
            n     = len(h_arr)
            self.data = {
                "ear":     round(ear, 4),
                "ear_raw": round(ear_raw, 4),
                "face":    True,
                "min_ear": round(h_arr[0], 4),
                "max_ear": round(h_arr[-1], 4),
                "mean_ear": round(sum(h_arr)/n, 4),
                "p20":      round(h_arr[int(0.20*(n-1))], 4),
                "p10":      round(h_arr[int(0.10*(n-1))], 4),
                "samples":  n,
            }
        return frame

    def get_data(self):
        with self.lock:
            return dict(self.data)

ctx = webrtc_streamer(
    key="diag",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DiagProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

proc = ctx.video_processor
st.markdown("---")

ph = st.empty()

import time
while True:
    time.sleep(0.3)
    if proc is None:
        break
    d = proc.get_data()
    if not d.get("face"):
        with ph.container():
            st.warning("No face detected — center your face in frame.")
        continue

    with ph.container():
        st.subheader("Live EAR Readings (eyes OPEN)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EAR (smooth)",  d.get("ear",  "—"))
        c2.metric("EAR (raw)",     d.get("ear_raw", "—"))
        c3.metric("Samples",       d.get("samples", 0))
        c4.metric("Face detected", "✅ Yes")

        st.subheader("Statistics (last 300 frames — eyes open)")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Min EAR",  d.get("min_ear",  "—"))
        s2.metric("Max EAR",  d.get("max_ear",  "—"))
        s3.metric("Mean EAR", d.get("mean_ear", "—"))
        s4.metric("P10",      d.get("p10",  "—"), help="10th percentile")
        s5.metric("P20",      d.get("p20",  "—"), help="20th percentile — current threshold")

        ear_val   = d.get("ear", 0) or 0
        p20_val   = d.get("p20", 0) or 0
        mean_val  = d.get("mean_ear", 0) or 0

        st.markdown("---")
        st.subheader("Threshold Diagnosis")
        if p20_val > 0:
            current_thresh = max(0.12, p20_val)
            if mean_val < current_thresh:
                st.error(
                    f"🚨 **BUG CONFIRMED** — Your open-eye EAR mean is `{mean_val:.4f}` "
                    f"but threshold would be set to `{current_thresh:.4f}`. "
                    f"Every frame looks 'closed'! Your EAR values are abnormally low."
                )
            else:
                st.success(
                    f"✅ Threshold looks OK — mean open EAR `{mean_val:.4f}` "
                    f"is above threshold `{current_thresh:.4f}`."
                )
            st.info(
                f"**Recommended threshold for your eyes: `{mean_val * 0.75:.4f}`** "
                f"(75% of your mean open EAR)\n\n"
                f"Min floor should be: `{min(0.12, mean_val * 0.60):.4f}` not `0.12`"
            )