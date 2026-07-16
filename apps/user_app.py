from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import threading
import cv2
from collections import deque
from dataclasses import dataclass
from typing import Optional, Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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

from src.cv.extra_signals import (
    mouth_aspect_ratio, YawnTracker, forward_head_angle,
    ear_asymmetry, compute_ibr,
    FHP_WARN_DEG, FHP_HIGH_DEG, fhp_level,
)
from src.ml.anomaly import PersonalAnomalyDetector
from src.ml.forecaster import FatigueForecaster
from src.ml.adaptive_weights import AdaptiveWeights


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EyeGuard",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = config.APP_VERSION

# ─────────────────────────────────────────────────────────────────────────────
# ███  UPGRADED GLOBAL CSS  ███
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;450;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── TOKENS ── */
:root {
  --ink:          #04060d;
  --ink-2:        #070b15;
  --surface-0:    rgba(255,255,255,0.030);
  --surface-1:    rgba(255,255,255,0.060);
  --surface-2:    rgba(255,255,255,0.095);
  --surface-3:    rgba(255,255,255,0.130);
  --border-dim:   rgba(255,255,255,0.075);
  --border-mid:   rgba(255,255,255,0.130);
  --border-hi:    rgba(255,255,255,0.220);

  --aqua:         #2fe4d4;
  --aqua-dim:     rgba(47,228,212,0.15);
  --aqua-glow:    rgba(47,228,212,0.30);
  --blue:         #4fa8ff;
  --blue-dim:     rgba(79,168,255,0.12);
  --violet:       #9b7fff;
  --amber:        #ffb83f;
  --amber-dim:    rgba(255,184,63,0.15);
  --rose:         #ff5f7e;
  --rose-dim:     rgba(255,95,126,0.12);
  --emerald:      #34d399;
  --emerald-dim:  rgba(52,211,153,0.12);

  --text-hi:      #f0f4ff;
  --text-mid:     rgba(240,244,255,0.65);
  --text-lo:      rgba(240,244,255,0.38);
  --text-mute:    rgba(240,244,255,0.20);

  --font-display: 'Playfair Display', Georgia, serif;
  --font-body:    'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono:    'JetBrains Mono', 'Fira Code', monospace;

  --r-sm: 10px;
  --r-md: 16px;
  --r-lg: 22px;
  --r-xl: 30px;
}

/* ── BASE ── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--ink) !important;
  color: var(--text-hi) !important;
  font-family: var(--font-body) !important;
}

/* Ambient background orbs */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 700px 500px at -10% -5%, rgba(47,228,212,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 600px 500px at 110% 110%, rgba(155,127,255,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 400px 400px at 55% 50%, rgba(79,168,255,0.04) 0%, transparent 60%);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: rgba(4,6,13,0.80) !important;
  backdrop-filter: blur(40px) saturate(180%) !important;
  -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
  border-right: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] * {
  color: var(--text-hi) !important;
  font-family: var(--font-body) !important;
}

/* Sidebar button overrides */
[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg,rgba(47,228,212,0.15),rgba(79,168,255,0.10)) !important;
  color: var(--aqua) !important;
  border: 1px solid rgba(47,228,212,0.30) !important;
  box-shadow: none !important;
  font-weight: 550 !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
  background: transparent !important;
  color: var(--text-lo) !important;
  border: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] button:hover {
  border-color: var(--border-mid) !important;
  color: var(--text-hi) !important;
  transform: translateX(3px) !important;
}

/* ── GLOBAL SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 999px; }

/* ── SECTION HEADINGS ── */
.section-heading {
  display: flex; align-items: center; gap: 12px;
  font-family: var(--font-body);
  font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 3px;
  color: var(--text-lo);
  margin: 28px 0 14px;
}
.section-heading::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border-dim), transparent);
}

/* ── EYEGUARD HEADER ── */
.eyeguard-header {
  padding: 6px 0 24px;
  border-bottom: 1px solid var(--border-dim);
  margin-bottom: 28px;
}
.eyeguard-logo {
  font-family: var(--font-display);
  font-size: 28px; font-weight: 700; letter-spacing: -1px;
  background: linear-gradient(135deg, var(--text-hi) 0%, var(--aqua) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: none;
}
.eyeguard-sub {
  font-size: 11px; color: var(--text-mute);
  text-transform: uppercase; letter-spacing: 2.5px; margin-top: 3px;
}
.version-badge {
  display: inline-block;
  background: var(--surface-1);
  border: 1px solid var(--border-dim);
  color: var(--text-mute);
  font-family: var(--font-mono); font-size: 10px;
  padding: 3px 10px; border-radius: 6px; margin-left: auto;
}

/* ── METRIC GRID ── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px; margin: 12px 0;
}
.metric-card {
  background: var(--surface-0);
  border: 1px solid var(--border-dim);
  border-radius: var(--r-md);
  padding: 18px 16px;
  position: relative; overflow: hidden;
  transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}
.metric-card::before {
  content: ''; position: absolute;
  inset: 0; border-radius: var(--r-md);
  background: linear-gradient(135deg, var(--surface-1), transparent);
  opacity: 0; transition: opacity 0.3s;
}
.metric-card:hover { border-color: var(--border-mid); transform: translateY(-3px); box-shadow: 0 16px 40px rgba(0,0,0,0.35); }
.metric-card:hover::before { opacity: 1; }
/* Coloured bottom accent line */
.metric-card::after {
  content: ''; position: absolute;
  bottom: 0; left: 0; right: 0; height: 2px; border-radius: 0 0 var(--r-md) var(--r-md);
  background: linear-gradient(90deg, transparent, var(--aqua));
  opacity: 0.4;
}
.metric-label {
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 1.8px;
  color: var(--text-mute); margin-bottom: 10px;
}
.metric-value {
  font-family: var(--font-mono);
  font-size: 26px; font-weight: 300; color: var(--text-hi); line-height: 1;
  transition: color 0.4s;
}
.metric-value.good   { color: var(--emerald); }
.metric-value.warn   { color: var(--amber);   }
.metric-value.danger { color: var(--rose);    }
.metric-unit { font-size: 11px; color: var(--text-lo); margin-top: 5px; }

/* ── COMFORT SCORE ── */
.comfort-ring-wrap { text-align: center; padding: 20px 0 12px; }
.comfort-score-big {
  font-family: var(--font-mono);
  font-size: 60px; font-weight: 200; line-height: 1;
  transition: color 0.5s, text-shadow 0.5s;
}
.comfort-score-big.good   { color: var(--emerald); text-shadow: 0 0 40px rgba(52,211,153,0.35); }
.comfort-score-big.warn   { color: var(--amber);   text-shadow: 0 0 40px rgba(255,184,63,0.35); }
.comfort-score-big.danger { color: var(--rose);    text-shadow: 0 0 40px rgba(255,95,126,0.35); }
.comfort-label {
  font-size: 11px; text-transform: uppercase;
  letter-spacing: 2.5px; color: var(--text-lo); margin-top: 8px;
}

/* ── PROGRESS ── */
.progress-wrap { margin: 12px 0; }
.progress-label {
  display: flex; justify-content: space-between;
  font-size: 12px; color: var(--text-mid); margin-bottom: 10px;
}
.progress-track {
  background: var(--surface-2);
  border-radius: 999px; height: 4px; overflow: hidden;
}
.progress-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--blue), var(--aqua));
  box-shadow: 0 0 12px var(--aqua-glow);
  transition: width 0.5s cubic-bezier(0.34,1.1,0.64,1);
}
.progress-fill.calib {
  background: linear-gradient(90deg, rgba(79,168,255,0.6), var(--aqua));
}

/* ── STATUS PILL ── */
.status-pill {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 5px 13px; border-radius: 999px;
  font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 1.2px;
  border: 1px solid;
}
.status-pill.idle  { background: var(--surface-1);               border-color: var(--border-mid);         color: var(--text-lo); }
.status-pill.calib { background: rgba(79,168,255,0.10);          border-color: rgba(79,168,255,0.35);     color: var(--blue); }
.status-pill.run   { background: rgba(47,228,212,0.08);          border-color: rgba(47,228,212,0.30);     color: var(--aqua); }
.status-pill.done  { background: rgba(52,211,153,0.10);          border-color: rgba(52,211,153,0.30);     color: var(--emerald); }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.status-dot.pulse { animation: dotPulse 1.6s ease-in-out infinite; }
@keyframes dotPulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%     { opacity:0.4; transform:scale(0.65); }
}

/* ── FPS CHIP ── */
.fps-chip {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--surface-1); border: 1px solid var(--border-dim);
  border-radius: 8px; padding: 4px 12px;
  font-family: var(--font-mono); font-size: 11px; color: var(--text-mid);
}

/* ── CNN BADGE ── */
.cnn-badge {
  display: flex; align-items: center; gap: 10px;
  background: var(--surface-0); border: 1px solid var(--border-dim);
  border-radius: var(--r-sm); padding: 11px 15px;
  font-family: var(--font-mono); font-size: 13px;
  margin: 8px 0; transition: border-color 0.3s;
}
.cnn-badge.good   { border-color: rgba(52,211,153,0.30); color: var(--emerald); }
.cnn-badge.warn   { border-color: rgba(255,184,63,0.30); color: var(--amber); }
.cnn-badge.danger { border-color: rgba(255,95,126,0.30); color: var(--rose); }
.cnn-badge.muted  { color: var(--text-lo); }

/* ── AI BADGES ── */
.ai-badge {
  display: flex; align-items: center; gap: 10px;
  border-radius: var(--r-sm); padding: 11px 15px;
  margin: 6px 0; border-left: 3px solid; font-size: 13px;
  transition: all 0.3s;
}
.ai-badge.anomaly  { background: rgba(155,127,255,0.08); border-color: var(--violet);  color: #c4b0ff; }
.ai-badge.normal   { background: rgba(52,211,153,0.07);  border-color: var(--emerald); color: var(--emerald); }
.ai-badge.training { background: rgba(79,168,255,0.07);  border-color: var(--blue);    color: var(--blue); }

/* ── FORECAST BOX ── */
.forecast-box {
  background: var(--surface-0); border: 1px solid var(--border-dim);
  border-radius: var(--r-md); padding: 14px 18px; margin: 8px 0;
  display: flex; justify-content: space-between; align-items: center;
}
.forecast-val {
  font-family: var(--font-mono); font-size: 24px; font-weight: 300; line-height: 1;
  margin: 4px 0;
}
.forecast-val.rising  { color: var(--rose); }
.forecast-val.falling { color: var(--emerald); }
.forecast-val.stable  { color: var(--aqua); }

/* ── BREAK ALERT ── */
.break-alert {
  border-radius: var(--r-md); padding: 15px 18px;
  margin: 10px 0; border-left: 3px solid;
  backdrop-filter: blur(8px);
  transition: all 0.4s ease;
}
.break-alert.work  { background: rgba(52,211,153,0.06); border-color: var(--emerald); }
.break-alert.micro { background: rgba(47,228,212,0.06); border-color: var(--aqua); }
.break-alert.short { background: rgba(255,184,63,0.06); border-color: var(--amber); }
.break-alert.long  { background: rgba(255,95,126,0.06); border-color: var(--rose); }
.break-title { font-weight: 600; font-size: 13px; margin-bottom: 4px; color: var(--text-hi); }
.break-body  { font-size: 12px; color: var(--text-mid); }

/* ── FOCUS CARD ── */
.focus-card {
  background: linear-gradient(135deg, var(--surface-0), rgba(47,228,212,0.025));
  border: 1px solid var(--border-dim);
  border-radius: var(--r-lg); padding: 22px 24px;
  position: relative; overflow: hidden;
}
.focus-card::after {
  content: '👁';
  position: absolute; right: 18px; top: 50%; transform: translateY(-50%);
  font-size: 52px; opacity: 0.05;
}
.focus-card-title {
  font-family: var(--font-display);
  font-size: 17px; font-weight: 700; color: var(--text-hi);
  margin-bottom: 8px; letter-spacing: -0.3px;
}
.focus-card-body  { font-size: 14px; color: var(--text-mid); line-height: 1.6; }
.focus-card-prompt { font-size: 12px; color: var(--text-lo); margin-top: 9px; font-style: italic; }

/* ── EAR ROW ── */
.ear-row {
  display: grid; grid-template-columns: 1fr 1fr 1fr;
  gap: 10px; margin: 14px 0;
}
.ear-cell {
  background: var(--surface-0); border: 1px solid var(--border-dim);
  border-radius: var(--r-sm); padding: 12px 14px; text-align: center;
  transition: border-color 0.25s, background 0.25s;
}
.ear-cell:hover { border-color: var(--border-mid); background: var(--surface-1); }
.ear-cell-label {
  font-size: 9px; text-transform: uppercase;
  letter-spacing: 2px; color: var(--text-mute); margin-bottom: 6px; font-weight: 600;
}
.ear-cell-value {
  font-family: var(--font-mono); font-size: 21px; font-weight: 300; color: var(--aqua);
}

/* ── RECOMMENDATIONS ── */
.rec-card {
  background: var(--surface-0); border: 1px solid var(--border-dim);
  border-radius: var(--r-lg); padding: 20px 22px;
}
.rec-title { font-size: 14px; font-weight: 600; color: var(--text-hi); margin-bottom: 12px; line-height: 1.45; }
.rec-bullet {
  display: flex; gap: 10px; font-size: 13px;
  color: var(--text-mid); margin-bottom: 8px; line-height: 1.55;
  align-items: flex-start;
}
.rec-bullet::before {
  content: '›';
  color: var(--aqua); font-weight: 700; flex-shrink: 0;
  font-size: 16px; line-height: 1.3;
}

/* ── REPORT ── */
.report-header {
  display: flex; align-items: center; gap: 18px;
  background: linear-gradient(135deg, var(--surface-0), rgba(47,228,212,0.025));
  border: 1px solid var(--border-dim); border-radius: var(--r-lg);
  padding: 26px 30px; margin-bottom: 20px;
}
.report-badge {
  background: rgba(52,211,153,0.12);
  border: 1px solid rgba(52,211,153,0.30);
  color: var(--emerald); border-radius: 6px;
  padding: 4px 12px; font-size: 12px; font-weight: 600;
}

/* ── HISTORY ROWS ── */
.history-row {
  display: grid;
  grid-template-columns: 160px 1fr 1fr 1fr 1fr auto;
  gap: 14px; align-items: center;
  padding: 13px 18px;
  background: var(--surface-0); border: 1px solid var(--border-dim);
  border-radius: var(--r-md); margin-bottom: 8px; font-size: 13px;
  transition: border-color 0.25s, transform 0.25s, background 0.25s;
}
.history-row:hover {
  border-color: var(--border-mid);
  transform: translateX(4px);
  background: var(--surface-1);
}
.history-time { color: var(--text-lo); font-family: var(--font-mono); font-size: 11px; }

/* ── PILLS ── */
.pill {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;
}
.pill.low  { background: rgba(52,211,153,0.12); color: var(--emerald); border: 1px solid rgba(52,211,153,0.25); }
.pill.med  { background: rgba(255,184,63,0.12);  color: var(--amber);   border: 1px solid rgba(255,184,63,0.25); }
.pill.high { background: rgba(255,95,126,0.12);  color: var(--rose);    border: 1px solid rgba(255,95,126,0.25); }

/* ── STREAMLIT NATIVE OVERRIDES ── */
div[data-testid="stMetric"] {
  background: var(--surface-0) !important;
  border: 1px solid var(--border-dim) !important;
  border-radius: var(--r-md) !important;
  padding: 16px !important;
}
div[data-testid="stMetric"] label {
  color: var(--text-lo) !important;
  font-size: 10px !important; text-transform: uppercase !important;
  letter-spacing: 1.5px !important; font-family: var(--font-body) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  color: var(--text-hi) !important; font-size: 26px !important; font-weight: 300 !important;
}

/* Primary buttons */
button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--aqua), var(--blue)) !important;
  color: #04060d !important; border: none !important;
  font-weight: 600 !important; font-family: var(--font-body) !important;
  font-size: 14px !important; border-radius: var(--r-sm) !important;
  letter-spacing: 0.3px !important;
  box-shadow: 0 4px 24px var(--aqua-glow), 0 0 0 1px rgba(47,228,212,0.2) !important;
  transition: all 0.25s cubic-bezier(0.34,1.2,0.64,1) !important;
}
button[data-testid="baseButton-primary"]:hover {
  box-shadow: 0 8px 40px rgba(47,228,212,0.40) !important;
  transform: translateY(-2px) !important;
}
button[data-testid="baseButton-primary"]:active {
  transform: scale(0.97) !important;
}

/* Secondary buttons */
button[data-testid="baseButton-secondary"] {
  background: var(--surface-1) !important;
  color: var(--text-mid) !important;
  border: 1px solid var(--border-mid) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--font-body) !important;
  transition: all 0.2s !important;
}
button[data-testid="baseButton-secondary"]:hover {
  border-color: var(--border-hi) !important;
  color: var(--text-hi) !important;
  transform: translateY(-1px) !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div {
  background: var(--surface-2) !important;
  border-radius: 999px !important; height: 4px !important;
}
div[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--blue), var(--aqua)) !important;
  border-radius: 999px !important;
  box-shadow: 0 0 10px var(--aqua-glow) !important;
}

/* Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  background: var(--surface-1) !important;
  border: 1px solid var(--border-mid) !important;
  color: var(--text-hi) !important;
  border-radius: var(--r-sm) !important;
  font-family: var(--font-body) !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
  border-color: rgba(47,228,212,0.5) !important;
  box-shadow: 0 0 0 3px var(--aqua-dim) !important;
}

hr { border-color: var(--border-dim) !important; opacity: 1 !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ══════════════════════════════════════════════
   FEATURE 1 — LIVE SVG COMFORT RING
══════════════════════════════════════════════ */
.ring-wrap {
  display: flex; flex-direction: column; align-items: center;
  padding: 16px 0 8px; position: relative;
}
.ring-svg { transform: rotate(-90deg); overflow: visible; }
.ring-track {
  fill: none; stroke: rgba(255,255,255,0.07); stroke-width: 11;
  stroke-linecap: round;
}
.ring-fill {
  fill: none; stroke-width: 11; stroke-linecap: round;
  stroke-dasharray: 502.65;   /* 2π × 80 */
  stroke-dashoffset: 502.65;  /* starts empty */
  stroke: url(#ringGrad);
  filter: drop-shadow(0 0 10px rgba(47,228,212,0.55));
  transition: stroke-dashoffset 1.0s cubic-bezier(0.34,1.1,0.64,1),
              stroke 0.6s ease, filter 0.6s ease;
}
.ring-fill.warn   { stroke: url(#ringGradWarn);   filter: drop-shadow(0 0 10px rgba(255,184,63,0.55)); }
.ring-fill.danger { stroke: url(#ringGradDanger); filter: drop-shadow(0 0 10px rgba(255,95,126,0.55)); }
/* breathing pulse in RUN state */
.ring-fill.live { animation: ringBreath 4s ease-in-out infinite; }
@keyframes ringBreath {
  0%,100% { filter: drop-shadow(0 0 8px  rgba(47,228,212,0.40)); }
  50%     { filter: drop-shadow(0 0 18px rgba(47,228,212,0.80)); }
}
.ring-fill.live.warn   { animation: ringBreathWarn   4s ease-in-out infinite; }
.ring-fill.live.danger { animation: ringBreathDanger 4s ease-in-out infinite; }
@keyframes ringBreathWarn   { 0%,100%{filter:drop-shadow(0 0 8px rgba(255,184,63,0.4))} 50%{filter:drop-shadow(0 0 18px rgba(255,184,63,0.8))} }
@keyframes ringBreathDanger { 0%,100%{filter:drop-shadow(0 0 8px rgba(255,95,126,0.4))} 50%{filter:drop-shadow(0 0 18px rgba(255,95,126,0.8))} }
.ring-inner {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  pointer-events: none;
}
.ring-score-num {
  font-family: var(--font-mono);
  font-size: 48px; font-weight: 200; line-height: 1;
  transition: color 0.6s ease;
}
.ring-score-num.good   { color: var(--emerald); }
.ring-score-num.warn   { color: var(--amber); }
.ring-score-num.danger { color: var(--rose); }
.ring-score-num.muted  { color: var(--text-mute); }
.ring-score-label {
  font-size: 10px; text-transform: uppercase; letter-spacing: 2.5px;
  color: var(--text-lo); margin-top: 5px;
}
.ring-band {
  font-size: 12px; font-weight: 600; margin-top: 3px;
  transition: color 0.4s;
}
.ring-band.good   { color: var(--emerald); }
.ring-band.warn   { color: var(--amber); }
.ring-band.danger { color: var(--rose); }

/* ══════════════════════════════════════════════
   FEATURE 2 — SPARKLINE METRIC CARDS
══════════════════════════════════════════════ */
.metric-card-spark {
  background: var(--surface-0);
  border: 1px solid var(--border-dim);
  border-radius: var(--r-md);
  padding: 16px 16px 10px;
  position: relative; overflow: hidden;
  transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}
.metric-card-spark:hover {
  border-color: var(--border-mid);
  transform: translateY(-3px);
  box-shadow: 0 16px 40px rgba(0,0,0,0.35);
}
.metric-card-spark .metric-label { font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1.8px;color:var(--text-mute);margin-bottom:6px; }
.metric-card-spark .metric-value { font-family:var(--font-mono);font-size:26px;font-weight:300;line-height:1;transition:color 0.4s; }
.metric-card-spark .metric-unit  { font-size:11px;color:var(--text-lo);margin-top:3px; }
.sparkline-wrap { margin-top: 10px; height: 36px; }
.sparkline-svg  { width: 100%; height: 36px; overflow: visible; }
.spark-area {
  fill: url(#sparkAreaGrad);
  opacity: 0.4;
}
.spark-line {
  fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
  transition: stroke 0.4s;
}
.spark-line.good   { stroke: var(--emerald); }
.spark-line.warn   { stroke: var(--amber); }
.spark-line.danger { stroke: var(--rose); }
.spark-dot {
  transition: fill 0.4s;
}
.spark-dot.good   { fill: var(--emerald); }
.spark-dot.warn   { fill: var(--amber); }
.spark-dot.danger { fill: var(--rose); }

/* ══════════════════════════════════════════════
   FEATURE 3 — PHASE TRANSITION OVERLAY
══════════════════════════════════════════════ */
.phase-banner {
  border-radius: var(--r-lg);
  padding: 20px 24px;
  margin: 8px 0 16px;
  display: flex; align-items: center; gap: 16px;
  border: 1px solid;
  animation: bannerSlideIn 0.55s cubic-bezier(0.34,1.2,0.64,1) both;
}
@keyframes bannerSlideIn {
  from { opacity: 0; transform: translateY(-12px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0)     scale(1); }
}
.phase-banner.idle  {
  background: rgba(255,255,255,0.03);
  border-color: var(--border-dim); color: var(--text-lo);
}
.phase-banner.calib {
  background: rgba(79,168,255,0.07);
  border-color: rgba(79,168,255,0.30); color: var(--blue);
}
.phase-banner.run   {
  background: rgba(47,228,212,0.07);
  border-color: rgba(47,228,212,0.28); color: var(--aqua);
}
.phase-banner.done  {
  background: rgba(52,211,153,0.07);
  border-color: rgba(52,211,153,0.30); color: var(--emerald);
}
.phase-banner-icon  { font-size: 28px; line-height: 1; flex-shrink: 0; }
.phase-banner-title { font-family: var(--font-display); font-size: 17px; font-weight: 700; margin-bottom: 2px; }
.phase-banner-sub   { font-size: 12px; opacity: 0.75; }
/* Animated scan line on CALIB */
.phase-banner.calib::after {
  content: '';
  position: absolute; left: 0; top: 0; height: 100%; width: 3px;
  background: var(--blue);
  border-radius: 3px 0 0 3px;
  animation: scanPulse 1.4s ease-in-out infinite;
}
@keyframes scanPulse { 0%,100%{opacity:0.4} 50%{opacity:1} }

/* ══════════════════════════════════════════════
   FEATURE 4 — REPORT SCORE REVEAL ANIMATION
══════════════════════════════════════════════ */
.report-reveal {
  animation: reportReveal 0.7s cubic-bezier(0.34,1.1,0.64,1) both;
}
@keyframes reportReveal {
  from { opacity: 0; transform: scale(0.92) translateY(16px); }
  to   { opacity: 1; transform: scale(1)    translateY(0); }
}
.report-metric-enter {
  animation: metricEnter 0.45s ease both;
}
@keyframes metricEnter {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
/* Stagger delay helper classes */
.d0  { animation-delay: 0.05s; }
.d1  { animation-delay: 0.12s; }
.d2  { animation-delay: 0.19s; }
.d3  { animation-delay: 0.26s; }
.d4  { animation-delay: 0.33s; }
.d5  { animation-delay: 0.40s; }
.d6  { animation-delay: 0.47s; }

/* Animated counter pseudo-shimmer on score-big */
.report-score-animate {
  animation: scoreCountUp 0.9s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes scoreCountUp {
  from { opacity: 0; letter-spacing: 8px; filter: blur(6px); }
  to   { opacity: 1; letter-spacing: normal; filter: blur(0); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (unchanged)
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
    if score100 >= 70: return ("Good", "good")
    if score100 >= 40: return ("Warning", "warn")
    return ("Danger", "danger")

def _break_css_class(mode: str) -> str:
    return {"WORK": "work", "MICRO": "micro", "SHORT": "short", "LONG": "long"}.get(
        (mode or "WORK").upper(), "work")

def _break_label(mode: str) -> str:
    return {"WORK": "✅ Continue working", "MICRO": "👁 Quick micro-break",
            "SHORT": "☕ Short break", "LONG": "🚶 Long break",
            }.get((mode or "WORK").upper(), "✅ Continue")

def _break_body(mode: str, remaining: float) -> str:
    if mode == "WORK": return "Eyes looking comfortable. Keep going."
    return f"Time remaining: {remaining:.0f}s"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DATACLASS  (unchanged)
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
# VIDEO PROCESSOR  (100% unchanged — all AI logic intact)
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
            "mar": None,
            "yawn_count": 0,
            "fhp_angle": None,
            "fhp_level": "ok",
            "ear_asym": None,
            "ibr": 0.0,
            "anomaly": False,
            "anomaly_score": 0.0,
            "anomaly_trained": False,
            "anomaly_progress": 0.0,
            "forecast_score_100": None,
            "forecast_trend": None,
            "forecast_ready": False,
            "adaptive_weights": {},
            "adaptive_corrections": 0,
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
        self.cnn_load_error = None
        self.cnn_ewma: Optional[float] = None
        self.cnn_hist   = deque(maxlen=25)
        self.last_roi   = None

        self.yawn_tracker  = YawnTracker()
        self.anomaly_det   = PersonalAnomalyDetector()
        self.forecaster    = FatigueForecaster()

        _w_path = config.PROFILES_DIR / "adaptive_weights.json"
        self.adaptive_weights = AdaptiveWeights.load(_w_path)

        self._blink_total_count      = 0
        self._incomplete_blink_count = 0

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
                self.yawn_tracker.reset()
                self.anomaly_det.reset()
                self.forecaster.reset()
                self._blink_total_count      = 0
                self._incomplete_blink_count = 0
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
        if not CNN_PATH.exists():
            self._try_download_cnn_model()
        if CNN_PATH.exists():
            try:
                self.cnn = TorchEyeCnn(CNN_PATH, grayscale=True, device="cpu")
                self.cnn_load_error = None
            except Exception as e:
                self.cnn = None
                self.cnn_load_error = str(e)
        else:
            self.cnn_load_error = "Model file not found and no download URL configured."

    def _try_download_cnn_model(self):
        """Attempt one-time download of the CNN weights if a URL is configured."""
        url = getattr(config, "CNN_MODEL_URL", "")
        if not url:
            return
        try:
            import urllib.request
            CNN_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = CNN_PATH.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.rename(CNN_PATH)
        except Exception as e:
            self.cnn_load_error = f"CNN download failed: {e}"

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

        try:
            mar      = mouth_aspect_ratio(lms, w, h)
            fhp_deg  = forward_head_angle(lms, w, h)
            ear_asym = ear_asymmetry(lms, w, h)
            self.yawn_tracker.update(mar)
        except Exception:
            mar, fhp_deg, ear_asym = 0.0, 0.0, 0.0

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

            if blink_occurred:
                self._blink_total_count += 1
                last_dur = getattr(self.blink_dur, 'last_blink_duration_sec', None)
                if last_dur is not None and float(last_dur) < 0.150:
                    self._incomplete_blink_count += 1
            ibr = compute_ibr(self._incomplete_blink_count, self._blink_total_count)

            level_live, score_live, _ = fuse_risk(
                perclos=perclos, blink_rate=blink_rate,
                cnn_prob=self.cnn_ewma, lstm_prob=None,
                ibr=ibr, mar=mar, ear_asym=ear_asym,
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

            try:
                ai_features = {
                    "ear":        ear,
                    "perclos":    perclos,
                    "blink_rate": blink_rate,
                    "mar":        mar,
                    "fhp_angle":  fhp_deg,
                    "ear_asym":   ear_asym,
                }
                self.anomaly_det.add_sample(ai_features)
                anomaly_flag, anomaly_score = self.anomaly_det.predict(ai_features)

                self.forecaster.add(risk01=score_live, t_sec=t_run)
                forecast = self.forecaster.predict()
            except Exception:
                ibr = 0.0
                anomaly_flag, anomaly_score = False, 0.0
                from src.ml.forecaster import ForecastResult
                forecast = ForecastResult(
                    predicted_score_01=0.0, predicted_score_100=50.0,
                    trend="stable", slope_per_sec=0.0,
                    current_score_01=0.0, horizon_sec=600,
                    sufficient_data=False,
                )

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
                    "mar":           round(mar, 4),
                    "yawn_count":    self.yawn_tracker.yawn_count,
                    "fhp_angle":     round(fhp_deg, 2),
                    "fhp_level":     fhp_level(fhp_deg),
                    "ear_asym":      round(ear_asym, 4),
                    "ibr":           round(ibr, 4),
                    "anomaly":       anomaly_flag,
                    "anomaly_score": anomaly_score,
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
                mar=mar,
                yawn_count=self.yawn_tracker.yawn_count,
                fhp_angle=round(fhp_deg, 2),
                fhp_level=fhp_level(fhp_deg),
                ear_asym=round(ear_asym, 4),
                ibr=round(ibr, 4),
                anomaly=anomaly_flag,
                anomaly_score=anomaly_score,
                anomaly_trained=self.anomaly_det.is_trained,
                anomaly_progress=round(self.anomaly_det.training_progress, 2),
                forecast_score_100=(forecast.predicted_score_100
                                    if forecast.sufficient_data else None),
                forecast_trend=(forecast.trend if forecast.sufficient_data else None),
                forecast_ready=forecast.sufficient_data,
                adaptive_weights=self.adaptive_weights.as_dict(),
                adaptive_corrections=self.adaptive_weights.corrections,
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
# SESSION STATE  (unchanged)
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
    # ── sparkline history buffers (feature 2) ──
    ("spark_blink",   []),
    ("spark_perclos", []),
    ("spark_comfort", []),
    # ── last phase seen (feature 3 – transition detection) ──
    ("_last_rendered_phase", "IDLE"),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  — upgraded HTML, same logic
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
        if st.button(f"{icon}  {page_name}", key=f"nav_{page_name}",
                     use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = page_name

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="version-badge">v {APP_VERSION}</div>', unsafe_allow_html=True)
    st.caption("Not a medical device.")


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS  — upgraded
# ─────────────────────────────────────────────────────────────────────────────
def section(label: str):
    st.markdown(f'<div class="section-heading">{label}</div>', unsafe_allow_html=True)

def metric_card(label: str, value: str, unit: str = "", css_class: str = "") -> str:
    return f"""<div class="metric-card">
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
    return f"""<div class="progress-wrap">
        <div class="progress-label"><span>{label_left}</span><span style="font-family:var(--font-mono);color:var(--aqua)">{label_right}</span></div>
        <div class="progress-track">
            <div class="progress-fill {fill_class}" style="width:{pct*100:.1f}%"></div>
        </div>
    </div>"""


# ── Feature 2: sparkline SVG builder ─────────────────────────────────────────
def sparkline_svg(values: list, css_class: str = "good", width: int = 200, height: int = 36) -> str:
    """Build an inline SVG sparkline from a list of floats."""
    if len(values) < 2:
        return '<div class="sparkline-wrap"></div>'
    mn, mx = min(values), max(values)
    span   = max(mx - mn, 1e-6)
    n      = len(values)
    xs     = [i / (n - 1) * width for i in range(n)]
    ys     = [height - ((v - mn) / span) * (height - 6) - 3 for v in values]
    pts    = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    # Area path: line + close down
    area_d = (f"M {xs[0]:.1f},{ys[0]:.1f} "
              + " ".join(f"L {x:.1f},{y:.1f}" for x, y in zip(xs[1:], ys[1:]))
              + f" L {xs[-1]:.1f},{height} L {xs[0]:.1f},{height} Z")
    color_map = {"good": "#34d399", "warn": "#ffb83f", "danger": "#ff5f7e"}
    col = color_map.get(css_class, "#34d399")
    last_x, last_y = xs[-1], ys[-1]
    gid = f"sg_{css_class}_{abs(hash(str(round(last_x,1)))) % 9999}"
    # Use numeric stop offsets (0/1) not percent to avoid f-string % collisions
    return (
        f'<div class="sparkline-wrap">'
        f'<svg class="sparkline-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
        f'<defs>'
        f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{col}" stop-opacity="0.5"/>'
        f'<stop offset="1" stop-color="{col}" stop-opacity="0"/>'
        f'</linearGradient>'
        f'</defs>'
        f'<path d="{area_d}" fill="url(#{gid})" opacity="0.35"/>'
        f'<polyline points="{pts}" class="spark-line {css_class}" fill="none"/>'
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="2.5" class="spark-dot {css_class}"/>'
        f'</svg></div>'
    )


def metric_card_spark(label: str, value: str, unit: str,
                      css_class: str, history: list) -> str:
    """Metric card with embedded sparkline."""
    spark = sparkline_svg(history, css_class)
    return f"""<div class="metric-card-spark">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
        <div class="metric-unit">{unit}</div>
        {spark}
    </div>"""


# ── Feature 1: live SVG comfort ring ─────────────────────────────────────────
def render_comfort_ring(score: int | None, css: str, band_label: str,
                        is_live: bool = False):
    """Render the animated SVG ring via st.components to bypass HTML sanitiser."""

    R = 80
    C = 502.65   # 2π × 80

    if score is None:
        offset     = C
        score_disp = "&#8212;"   # em dash — safe in HTML
        num_color  = "rgba(240,244,255,0.20)"
        stroke     = "#2fe4d4"
        glow       = "rgba(47,228,212,0.5)"
        band_color = "rgba(240,244,255,0.35)"
    else:
        s          = max(0, min(100, score))
        offset     = C * (1.0 - s / 100.0)
        score_disp = str(s)
        if css == "good":
            num_color  = "#34d399"; stroke = "#34d399"
            glow       = "rgba(52,211,153,0.55)"; band_color = "#34d399"
        elif css == "warn":
            num_color  = "#ffb83f"; stroke = "#ffb83f"
            glow       = "rgba(255,184,63,0.55)";  band_color = "#ffb83f"
        else:
            num_color  = "#ff5f7e"; stroke = "#ff5f7e"
            glow       = "rgba(255,95,126,0.55)";  band_color = "#ff5f7e"

    breath_anim = f"""
      @keyframes ringBreath {{
        0%,100% {{ filter: drop-shadow(0 0 8px {glow}); }}
        50%     {{ filter: drop-shadow(0 0 20px {glow}); }}
      }}
      .ring-fill-el {{ animation: ringBreath 4s ease-in-out infinite; }}
    """ if is_live else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400&display=swap');
  body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
  .wrap {{
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; height:220px;
  }}
  .ring-container {{ position:relative; width:196px; height:196px; }}
  svg {{ transform:rotate(-90deg); overflow:visible; display:block; }}
  .ring-track {{
    fill:none; stroke:rgba(255,255,255,0.07); stroke-width:11; stroke-linecap:round;
  }}
  .ring-fill-el {{
    fill:none; stroke:{stroke}; stroke-width:11; stroke-linecap:round;
    stroke-dasharray:{C:.2f};
    stroke-dashoffset:{offset:.2f};
    filter: drop-shadow(0 0 10px {glow});
    transition: stroke-dashoffset 1.0s cubic-bezier(0.34,1.1,0.64,1);
  }}
  {breath_anim}
  .ring-inner {{
    position:absolute; inset:0;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
  }}
  .score-num {{
    font-family:'JetBrains Mono', monospace;
    font-size:48px; font-weight:200; line-height:1;
    color:{num_color};
  }}
  .score-label {{
    font-family:'JetBrains Mono', monospace;
    font-size:10px; text-transform:uppercase; letter-spacing:2.5px;
    color:rgba(240,244,255,0.35); margin-top:5px;
  }}
  .score-band {{
    font-family:-apple-system, sans-serif;
    font-size:12px; font-weight:600; margin-top:3px;
    color:{band_color};
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="ring-container">
    <svg width="196" height="196" viewBox="0 0 196 196">
      <circle class="ring-track" cx="98" cy="98" r="{R}"/>
      <circle class="ring-fill-el" cx="98" cy="98" r="{R}"/>
    </svg>
    <div class="ring-inner">
      <div class="score-num">{score_disp}</div>
      <div class="score-label">/ 100</div>
      <div class="score-band">{band_label if band_label else "&nbsp;"}</div>
    </div>
  </div>
</div>
</body>
</html>"""

    components.html(html, height=228, scrolling=False)


# ── Feature 3: phase transition banner ───────────────────────────────────────
def phase_banner_html(phase: str, t: float = 0.0) -> str:
    """Animated banner shown on phase entry."""
    configs = {
        "IDLE":  ("🎯", "Ready to Begin",     "Start the camera and click Start Assessment."),
        "CALIB": ("🔬", "Calibrating…",        f"Measuring your baseline eye openness — {int(min(100, t / max(1,10) * 100))}% complete."),
        "RUN":   ("👁",  "Assessment Live",     "Reading your eye signals in real time."),
        "DONE":  ("✅", "Session Complete",    "Your personalised report is ready below."),
    }
    icon, title, sub = configs.get(phase, configs["IDLE"])
    return f"""<div class="phase-banner {phase.lower()}" style="position:relative;overflow:hidden;">
      <div class="phase-banner-icon">{icon}</div>
      <div>
        <div class="phase-banner-title">{title}</div>
        <div class="phase-banner-sub">{sub}</div>
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# RENDER  — upgraded HTML snippets, identical logic & signature
# ─────────────────────────────────────────────────────────────────────────────
def _render(phase: str, snap: dict,
            ph_focus, ph_progress, ph_ear, ph_comfort,
            ph_status, ph_break, ph_recs, ph_ai):
    t = float(snap.get("t", 0.0))

    # ── Feature 3: phase transition banner ──────────────────────────────────
    st.session_state["_last_rendered_phase"] = phase

    # ── Focus task / phase banner ────────────────────────────────────────────
    with ph_focus.container():
        section("Focus Task")
        # Always show phase banner (it self-animates on entry)
        st.markdown(phase_banner_html(phase, t), unsafe_allow_html=True)

        # Extra detail cards per phase
        if phase == "CALIB":
            ear_live = snap.get("ear")
            if ear_live is not None:
                st.markdown(
                    f'<div style="font-size:12px;color:var(--text-lo);margin-top:4px;">'
                    f'Live EAR: <span style="color:var(--aqua);font-family:var(--font-mono)">'
                    f'{float(ear_live):.4f}</span></div>', unsafe_allow_html=True)
        elif phase == "RUN":
            st.markdown(f"""<div class="focus-card" style="margin-top:10px;">
                <div class="focus-card-title" style="font-size:14px;">{focus_text(t)}</div>
                <div class="focus-card-prompt">{micro_prompt(t)}</div>
                </div>""", unsafe_allow_html=True)

    # ── Progress ─────────────────────────────────────────────────────────────
    with ph_progress.container():
        section("Progress")
        if phase == "CALIB":
            p    = min(1.0, t / CALIB_SECONDS)
            html = progress_bar(p, "CALIB", f"Calibrating… {int(p*100)}%", f"{CALIB_SECONDS-t:.0f}s left")
        elif phase == "RUN":
            p    = min(1.0, t / RUN_SECONDS)
            html = progress_bar(p, "RUN", f"Assessing… {int(p*100)}%", f"~{max(0.,RUN_SECONDS-t):.0f}s left")
        elif phase == "DONE":
            html = progress_bar(1.0, "DONE", "Complete ✅", "")
        else:
            html = progress_bar(0.0, "IDLE", "Idle", "")
        st.markdown(html, unsafe_allow_html=True)

    # ── EAR row ──────────────────────────────────────────────────────────────
    with ph_ear.container():
        if phase == "DONE":
            st.empty()
        elif phase == "RUN":
            ev = snap.get("ear"); er = snap.get("ear_raw"); et = snap.get("ear_threshold")
            if ev is not None:
                st.markdown(f"""<div class="ear-row">
                    <div class="ear-cell">
                        <div class="ear-cell-label">EAR Smooth</div>
                        <div class="ear-cell-value">{float(ev):.3f}</div>
                    </div>
                    <div class="ear-cell">
                        <div class="ear-cell-label">EAR Raw</div>
                        <div class="ear-cell-value">{f"{float(er):.3f}" if er else "—"}</div>
                    </div>
                    <div class="ear-cell">
                        <div class="ear-cell-label">Threshold</div>
                        <div class="ear-cell-value">{f"{float(et):.3f}" if et else "—"}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Feature 1: Live SVG comfort ring ────────────────────────────────────
    with ph_comfort.container():
        section("Live Status")
        c   = snap.get("comfort")
        css = snap.get("comfort_css", "good") or "good"
        lbl = snap.get("comfort_band", "") or ""
        render_comfort_ring(c, css, lbl, is_live=(phase == "RUN"))

    # ── Status / FPS / Feature 2: sparkline metric cards ─────────────────────
    with ph_status.container():
        fps_val   = float(snap.get("fps", 0.0))
        fps_color = ("var(--emerald)" if fps_val >= 18 else
                     "var(--amber)"   if fps_val >= 12 else "var(--rose)")
        st.markdown(
            f'<div style="margin:8px 0 6px;font-size:13px;color:var(--text-mid)">'
            f'{status_pill(phase)} &nbsp;'
            f'<span class="fps-chip" style="color:{fps_color}">⚡ {fps_val:.0f} fps</span>'
            f'</div>', unsafe_allow_html=True)

        if phase == "RUN":
            be = snap.get("baseline_ear"); et = snap.get("ear_threshold")
            if be and et:
                st.markdown(
                    f'<div style="font-size:11px;color:var(--text-mute);margin:3px 0 8px;">'
                    f'Baseline: <span style="color:var(--aqua);font-family:var(--font-mono)">{float(be):.4f}</span>'
                    f' &nbsp;·&nbsp; Threshold: <span style="color:var(--aqua);font-family:var(--font-mono)">{float(et):.4f}</span>'
                    f'</div>', unsafe_allow_html=True)

            cnn_val = snap.get("cnn_ewma")
            if cnn_val is None:
                st.markdown('<div class="cnn-badge muted">⚪ &nbsp;CNN — warming up</div>',
                            unsafe_allow_html=True)
            else:
                pct = int(cnn_val * 100)
                cls, icon = (("danger","🔴") if pct>=65 else ("warn","🟡") if pct>=40 else ("good","🟢"))
                st.markdown(
                    f'<div class="cnn-badge {cls}">{icon} &nbsp;CNN · {pct}% sleepiness</div>',
                    unsafe_allow_html=True)

            blink   = snap.get("blink_rate")
            perclos = snap.get("perclos")
            if blink is not None:
                # Feed sparkline history
                spark_b = st.session_state.spark_blink
                spark_p = st.session_state.spark_perclos
                spark_b.append(float(blink))
                spark_p.append(float(perclos or 0) * 100)
                # Keep max 40 points
                if len(spark_b) > 40: spark_b.pop(0)
                if len(spark_p) > 40: spark_p.pop(0)

                b_css = "danger" if blink<6  else "warn" if blink<12  else "good"
                p_css = "danger" if (perclos or 0)>0.20 else "warn" if (perclos or 0)>0.12 else "good"

                # Feature 2: sparkline cards instead of plain metric cards
                st.markdown(f"""<div class="metric-grid">
                    {metric_card_spark("Blink Rate",  f"{blink:.0f}", "/min",      b_css, spark_b)}
                    {metric_card_spark("Eye Closure", f"{(perclos or 0)*100:.0f}", "% PERCLOS", p_css, spark_p)}
                </div>""", unsafe_allow_html=True)

    # ── Break alert ───────────────────────────────────────────────────────────
    with ph_break.container():
        if phase == "RUN":
            bm  = snap.get("break_mode", "WORK")
            br  = float(snap.get("break_remaining_sec", 0.0))
            st.markdown(f"""<div class="break-alert {_break_css_class(bm)}">
                <div class="break-title">{_break_label(bm)}</div>
                <div class="break-body">{_break_body(bm, br)}</div>
            </div>""", unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    with ph_recs.container():
        section("Recommendations")
        if phase in ("RUN", "DONE"):
            if phase == "DONE":
                rd      = st.session_state.get("report_data")
                title   = rd.get("advice_title", "Assessment complete.") if rd else "Assessment complete."
                bullets = rd.get("advice_bullets", []) if rd else []
            else:
                adv = recommendations_dynamic(
                    level=str(snap.get("risk_level") or "LOW"),
                    score01=float(snap.get("risk_score") or 0.0),
                    perclos=float(snap.get("perclos") or 0.0),
                    blink_rate=float(snap.get("blink_rate") or 0.0),
                    cnn_sleepy=snap.get("cnn_ewma"),
                    baseline_blink_rate=st.session_state.get("user_baseline_blink_rate"),
                    ear_drop_ratio=float(snap.get("ear_drop_ratio") or 1.0),
                    trend_cnn=snap.get("cnn_trend"),
                )
                title = adv.title; bullets = adv.bullets[:4]
            bullets_html = "".join(f'<div class="rec-bullet">{b}</div>' for b in bullets)
            st.markdown(f"""<div class="rec-card">
                <div class="rec-title">{title}</div>
                {bullets_html}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:var(--text-lo);font-size:13px;">'
                'Recommendations appear during assessment.</div>',
                unsafe_allow_html=True)

    # ── AI Intelligence panel ─────────────────────────────────────────────────
    with ph_ai.container():
        if phase == "RUN":
            section("AI Intelligence")

            anomaly         = snap.get("anomaly", False)
            anomaly_trained = snap.get("anomaly_trained", False)
            anomaly_prog    = snap.get("anomaly_progress", 0.0)

            if not anomaly_trained:
                st.markdown(
                    f'<div class="ai-badge training">'
                    f'⏳ &nbsp;Training your personal baseline… {int(anomaly_prog*100)}%</div>',
                    unsafe_allow_html=True)
            elif anomaly:
                st.markdown(
                    '<div class="ai-badge anomaly">'
                    '⚠ &nbsp;ANOMALY — Your signals deviate from your personal baseline</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="ai-badge normal">✅ &nbsp;Within your normal range</div>',
                    unsafe_allow_html=True)

            fc_score = snap.get("forecast_score_100")
            fc_trend = snap.get("forecast_trend")
            if fc_score is not None and fc_trend is not None:
                trend_icon = {"rising":"↗","falling":"↘","stable":"→"}.get(fc_trend,"→")
                st.markdown(f"""<div class="forecast-box">
                    <div>
                        <div style="font-size:10px;color:var(--text-mute);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">10-min forecast</div>
                        <div class="forecast-val {fc_trend}">{trend_icon} {fc_score:.0f}/100 comfort</div>
                        <div style="font-size:11px;color:var(--text-lo);margin-top:3px;">Trend: {fc_trend}</div>
                    </div>
                    <div style="font-size:36px;opacity:0.15">{trend_icon}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="font-size:12px;color:var(--text-lo);padding:8px 0;">'
                    '📈 Forecast builds after ~20s of data</div>',
                    unsafe_allow_html=True)

            fhp   = snap.get("fhp_angle")
            ibr   = snap.get("ibr", 0.0)
            yawns = snap.get("yawn_count", 0)

            if fhp is not None:
                fhp_css = ("danger" if fhp >= FHP_HIGH_DEG else "warn" if fhp >= FHP_WARN_DEG else "good")
                ibr_css = ("danger" if ibr >= 0.50 else "warn" if ibr >= 0.30 else "good")
                y_css   = "warn" if yawns >= 2 else "good"
                st.markdown(f"""<div class="metric-grid">
                    {metric_card("Yawns",        str(yawns),          "this session", y_css)}
                    {metric_card("Head Posture",  f"{fhp:.0f}°",       "FHP angle",    fhp_css)}
                    {metric_card("Incomplete ↷",  f"{ibr*100:.0f}%",   "blink ratio",  ibr_css)}
                </div>""", unsafe_allow_html=True)

            corrections = snap.get("adaptive_corrections", 0)
            if corrections > 0:
                st.markdown(
                    f'<div style="font-size:11px;color:var(--aqua);margin:8px 0;">'
                    f'✦ Model personalised with {corrections} correction(s)</div>',
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PROFILE  (unchanged logic, updated HTML)
# ─────────────────────────────────────────────────────────────────────────────
def page_profile():
    st.markdown("""<div class="eyeguard-header">
        <div class="eyeguard-logo" style="font-size:22px;">👤 Profile</div>
        <div class="eyeguard-sub">Personal settings for every assessment</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    section("Personal Info")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Display Name", value=st.session_state.profile.get("name", "User"))
    with col2:
        st.text_input("Role / Occupation (optional)", placeholder="e.g. Software Engineer")
    section("Assessment Preferences")
    c1, c2 = st.columns(2)
    with c1:
        enable_cnn       = st.checkbox("Use CNN model",          value=st.session_state.profile.get("enable_cnn", True))
        enable_scheduler = st.checkbox("Smart break reminders",  value=st.session_state.profile.get("enable_scheduler", True))
    with c2:
        cnn_status = "✅ Loaded" if CNN_PATH.exists() else (
            "⏳ Will attempt download on first run" if config.CNN_MODEL_URL
            else "❌ Not found — CNN scoring disabled, other 6 signals unaffected")
        st.markdown(f"""<div class="metric-card" style="margin-top:8px;">
            <div class="metric-label">CNN Model Status</div>
            <div style="font-family:var(--font-mono);font-size:15px;color:var(--aqua)">{cnn_status}</div>
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
# PAGE: HISTORY  (unchanged logic, updated HTML)
# ─────────────────────────────────────────────────────────────────────────────
def page_history():
    st.markdown("""<div class="eyeguard-header">
        <div class="eyeguard-logo" style="font-size:22px;">🗂️ History</div>
        <div class="eyeguard-sub">Past assessment sessions</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    sessions = sorted(SESS_DIR.glob("*.json"), reverse=True) if SESS_DIR.exists() else []
    if not sessions:
        st.info("No sessions recorded yet.")
        return
    section(f"Sessions — {len(sessions)} found")
    for sess_path in sessions[:20]:
        try:
            with open(sess_path) as f: d = json.load(f)
            risk     = d.get("final_risk_level", "?")
            pill_cls = {"LOW":"low","MED":"med","HIGH":"high"}.get(risk,"low")
            ts       = d.get("timestamp", sess_path.stem)[-19:].replace("_"," ")
            comfort  = d.get("final_comfort_score_100", "—")
            blink    = d.get("final_blink_rate", 0)
            perclos  = d.get("final_perclos", 0)
            advice   = d.get("advice_title", "")[:48]
            st.markdown(f"""<div class="history-row">
                <span class="history-time">{ts}</span>
                <span><span class="pill {pill_cls}">{risk}</span></span>
                <span style="font-family:var(--font-mono);color:var(--aqua)">{comfort}/100</span>
                <span style="color:var(--text-mid)">{blink:.0f}/min</span>
                <span style="color:var(--text-mid)">{perclos*100:.0f}%</span>
                <span style="color:var(--text-lo);font-size:12px">{advice}</span>
            </div>""", unsafe_allow_html=True)
        except Exception:
            continue


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SETTINGS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def page_settings():
    st.markdown("""<div class="eyeguard-header">
        <div class="eyeguard-logo" style="font-size:22px;">⚙️ Settings</div>
        <div class="eyeguard-sub">Advanced assessment parameters</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    section("Break Scheduler Durations")
    s1, s2, s3 = st.columns(3)
    with s1: st.number_input("Micro break (sec)",  value=config.BREAK_MICRO_SEC, min_value=5,   max_value=60)
    with s2: st.number_input("Short break (sec)",  value=config.BREAK_SHORT_SEC, min_value=30,  max_value=600)
    with s3: st.number_input("Long break (sec)",   value=config.BREAK_LONG_SEC,  min_value=120, max_value=1800)
    section("Risk Thresholds")
    t1, t2 = st.columns(2)
    with t1: st.number_input("Min gap between breaks (sec)", value=int(config.BREAK_MIN_GAP_SEC), min_value=10, max_value=300)
    with t2: st.number_input("Microsleep window (sec)",      value=int(config.MICROSLEEP_WINDOW_SEC), min_value=30, max_value=300)
    section("CNN Inference")
    st.number_input("Run CNN every N frames", value=config.CNN_EVERY_N_FRAMES, min_value=1, max_value=30)
    st.slider("CNN EWMA smoothing (alpha)", 0.05, 0.50, config.CNN_EWMA_ALPHA, 0.01)
    section("Data Management")
    if st.button("🗑️  Clear all session history", type="secondary"):
        if SESS_DIR.exists():
            for f in SESS_DIR.glob("*.json"): f.unlink()
        st.success("Session history cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT — compute and save  (unchanged)
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
    ibr_final      = float(df["ibr"].tail(1).iloc[0]) if "ibr" in df.columns else None
    mar_final      = float(df["mar"].dropna().tail(1).iloc[0]) if "mar" in df.columns and df["mar"].notna().any() else None
    ear_asym_final = float(df["ear_asym"].dropna().tail(1).iloc[0]) if "ear_asym" in df.columns and df["ear_asym"].notna().any() else None

    level_final, score_final, reasons_final = fuse_risk(
        perclos=perclos_final, blink_rate=blink_final,
        cnn_prob=cnn_final, lstm_prob=None,
        ibr=ibr_final, mar=mar_final, ear_asym=ear_asym_final)

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
        cnn_sleepy=cnn_final, baseline_blink_rate=_report_baseline_blink,
        ear_drop_ratio=(float(df["ear_drop_ratio"].dropna().mean())
                        if len(df["ear_drop_ratio"].dropna()) else 1.0),
        trend_cnn=None)

    fl  = float(df["fatigue_load"].tail(1).iloc[0]) if "fatigue_load" in df.columns else None
    msw = int(df["microsleep_count_window"].tail(1).iloc[0]) if "microsleep_count_window" in df.columns else 0
    fc  = int(df["comfort_score_100"].tail(1).iloc[0]) if "comfort_score_100" in df.columns else comfort_score_from_risk(score_final)
    _, fc_css = comfort_band(fc)

    yawn_total = int(df["yawn_count"].max()) if "yawn_count" in df.columns else 0
    fhp_mean   = round(float(df["fhp_angle"].mean()), 1) if "fhp_angle" in df.columns else None
    ibr_mean   = round(float(df["ibr"].mean()), 3) if "ibr" in df.columns else None

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    with open(SESS_DIR / f"{cfg.user_name}_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({
            "user": cfg.user_name, "timestamp": ts,
            "baseline_ear": st.session_state.baseline_ear,
            "ear_threshold": st.session_state.ear_threshold,
            "final_perclos": float(perclos_final),
            "final_blink_rate": float(blink_final),
            "final_cnn_sleepy_prob": cnn_final,
            "final_risk_level": level_final,
            "final_risk_score": float(score_final),
            "final_comfort_score_100": int(fc),
            "final_fatigue_load": fl,
            "microsleeps_2min": msw,
            "yawn_count": yawn_total,
            "fhp_angle_mean": fhp_mean,
            "ibr_mean": ibr_mean,
            "advice_title": advice.title,
            "fusion_reasons": reasons_final,
            "baseline_blink_rate": _report_baseline_blink,
            "app_version": APP_VERSION,
        }, f, indent=2)

    st.session_state.report_data = {
        "empty": False, "comfort": fc, "comfort_css": fc_css,
        "blink_rate": blink_final, "perclos": perclos_final,
        "fatigue_load": fl, "microsleeps": msw,
        "yawn_count": yawn_total, "fhp_mean": fhp_mean, "ibr_mean": ibr_mean,
        "risk_level": level_final,
        "baseline_ear": st.session_state.baseline_ear,
        "ear_threshold": st.session_state.ear_threshold,
        "advice_title": advice.title, "advice_bullets": advice.bullets,
        "csv_bytes": df.to_csv(index=False).encode("utf-8"),
        "csv_filename": f"assessment_{cfg.user_name}_{ts}.csv",
        "cnn_final_prob": cnn_final, "timestamp": ts,
    }

    try:
        baseline_path = config.PROFILES_DIR / f"{cfg.user_name}_baseline.json"
        ub = load_user_baseline(baseline_path, cfg.user_name)
        ub.update_from_session(
            duration_sec=float(df["t"].max()), frames_ok=len(df),
            blink_rate_final=blink_final, perclos_final=perclos_final,
            ear_mean=(float(df["ear_raw"].mean()) if "ear_raw" in df.columns else None))
        save_user_baseline(baseline_path, ub)
    except Exception:
        pass

    st.session_state.report_done = True


# ─────────────────────────────────────────────────────────────────────────────
# REPORT — display  (Feature 4: animated reveal)
# ─────────────────────────────────────────────────────────────────────────────
def _show_report():
    d = st.session_state.get("report_data")
    if not d: return
    section("Assessment Report")
    if d.get("empty"):
        st.error("No frames captured. Improve lighting and keep face centred.")
        return

    fc = d["comfort"]; fc_css = d.get("comfort_css",""); risk = d.get("risk_level","LOW")
    pill_cls = {"LOW":"low","MED":"med","HIGH":"high"}.get(risk,"low")

    # ── Feature 4a: animated score header ─────────────────────────────────
    st.markdown(f"""<div class="report-header report-reveal d0">
        <div class="comfort-score-big {fc_css} report-score-animate"
             style="font-size:52px;padding-right:4px">{fc}</div>
        <div>
            <div style="font-size:11px;color:var(--text-lo);text-transform:uppercase;
                        letter-spacing:1.5px;margin-bottom:8px">Comfort Score</div>
            <span class="pill {pill_cls}">{risk} RISK</span>
            <span class="report-badge" style="margin-left:8px">Complete ✅</span>
        </div>
        <div style="margin-left:auto;font-size:11px;color:var(--text-mute);
                    font-family:var(--font-mono)">
            {d.get("timestamp","").replace("_"," ")}
        </div>
    </div>""", unsafe_allow_html=True)

    b_css   = "danger" if d["blink_rate"]<6  else "warn" if d["blink_rate"]<12  else "good"
    p_css   = "danger" if d["perclos"]>0.20  else "warn" if d["perclos"]>0.12   else "good"
    fl      = d["fatigue_load"]; fl_str = "N/A" if fl is None else f"{fl:.2f}"
    cnn_val = d.get("cnn_final_prob")
    cnn_str = "—" if cnn_val is None else f"{int(cnn_val*100)}%"
    cnn_css = "" if cnn_val is None else ("danger" if cnn_val>=0.65 else "warn" if cnn_val>=0.40 else "good")

    yawns   = d.get("yawn_count", 0)
    fhp     = d.get("fhp_mean")
    ibr     = d.get("ibr_mean")
    fhp_str = f"{fhp:.1f}°" if fhp is not None else "—"
    ibr_str = f"{ibr*100:.0f}%" if ibr is not None else "—"
    fhp_css = ("danger" if (fhp or 0)>=FHP_HIGH_DEG else "warn" if (fhp or 0)>=FHP_WARN_DEG else "good") if fhp else ""
    ibr_css = ("danger" if (ibr or 0)>=0.50 else "warn" if (ibr or 0)>=0.30 else "good") if ibr else ""

    # ── Feature 4b: staggered metric cards — wrap each in a delay class ──
    def anim_card(label, value, unit, css, delay_cls):
        return f"""<div class="metric-card report-metric-enter {delay_cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css}">{value}</div>
            <div class="metric-unit">{unit}</div>
        </div>"""

    # Feature 2b: sparkline in report cards if history exists
    spark_b = st.session_state.spark_blink
    spark_p = st.session_state.spark_perclos

    def anim_spark_card(label, value, unit, css, delay_cls, history):
        spark = sparkline_svg(list(history), css) if len(history) >= 2 else ""
        return f"""<div class="metric-card-spark report-metric-enter {delay_cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css}">{value}</div>
            <div class="metric-unit">{unit}</div>
            {spark}
        </div>"""

    st.markdown(f"""
    <div class="metric-grid" style="margin:16px 0 12px;">
        {anim_spark_card("Blink Rate",   f"{d['blink_rate']:.0f}", "/min",       b_css,   "d1", spark_b)}
        {anim_spark_card("Eye Closure",  f"{d['perclos']*100:.0f}", "% PERCLOS", p_css,   "d2", spark_p)}
        {anim_card("Fatigue Load", fl_str,  "",             "",       "d3")}
        {anim_card("CNN Model",    cnn_str, "sleepiness",   cnn_css,  "d4")}
    </div>
    <div class="metric-grid" style="margin:0 0 16px;">
        {anim_card("Yawns",        str(yawns),  "this session", "warn" if yawns>=2 else "good", "d4")}
        {anim_card("Head Posture", fhp_str,      "avg FHP",      fhp_css, "d5")}
        {anim_card("Incomplete ↷", ibr_str,      "blink ratio",  ibr_css, "d6")}
    </div>""", unsafe_allow_html=True)

    ms = d["microsleeps"]; be = d.get("baseline_ear"); et = d.get("ear_threshold")
    stats = f'Microsleeps: <strong>{ms}</strong>'
    if be:
        stats += (f' &nbsp;·&nbsp; Baseline EAR: '
                  f'<span style="font-family:var(--font-mono);color:var(--aqua)">{be:.3f}</span>'
                  f' &nbsp;·&nbsp; Threshold: '
                  f'<span style="font-family:var(--font-mono);color:var(--aqua)">{et:.3f}</span>')
    st.markdown(
        f'<div class="report-metric-enter d5" style="font-size:13px;color:var(--text-mid);'
        f'margin:8px 0 16px">{stats}</div>',
        unsafe_allow_html=True)

    bullets_html = "".join(f'<div class="rec-bullet">{b}</div>' for b in d["advice_bullets"])
    st.markdown(f"""<div class="rec-card report-metric-enter d6" style="margin-top:4px">
        <div class="rec-title">{d["advice_title"]}</div>
        {bullets_html}
    </div>""", unsafe_allow_html=True)

    st.download_button(
        "⬇  Download Full CSV", data=d["csv_bytes"],
        file_name=d["csv_filename"], mime="text/csv", key="report_csv_btn")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ASSESSMENT  (unchanged logic, updated HTML strings)
# ─────────────────────────────────────────────────────────────────────────────
def page_assessment():
    st.markdown("""<div class="eyeguard-header">
        <div>
            <div class="eyeguard-logo">👁 EyeGuard</div>
            <div class="eyeguard-sub">Real-time Eye Strain Monitor</div>
        </div>
    </div>""", unsafe_allow_html=True)

    profile = st.session_state.profile
    cfg_now = RuntimeCfg(
        user_name=profile.get("name","User"), enable_cnn=profile.get("enable_cnn",True),
        cnn_every_n_frames=config.CNN_EVERY_N_FRAMES, ewma_alpha=config.CNN_EWMA_ALPHA,
        enable_scheduler=profile.get("enable_scheduler",True),
        microsleep_sec=config.MICROSLEEP_SEC, microsleep_window_sec=config.MICROSLEEP_WINDOW_SEC,
        microsleep_short_count=config.MICROSLEEP_SHORT_COUNT, microsleep_long_count=config.MICROSLEEP_LONG_COUNT,
        micro_sec=config.BREAK_MICRO_SEC, short_sec=config.BREAK_SHORT_SEC, long_sec=config.BREAK_LONG_SEC,
        min_gap=config.BREAK_MIN_GAP_SEC, k_up=config.FATIGUE_K_UP,
        k_down=config.FATIGUE_K_DOWN, k_leak=config.FATIGUE_K_LEAK,
    )
    if st.session_state.run_cfg is None:
        st.session_state.run_cfg = cfg_now

    section("Camera Feed")
    st.caption("Allow camera permission when asked.")

    webrtc_ctx = webrtc_streamer(
        key=WEBRTC_KEY, mode=WebRtcMode.SENDRECV,
        video_processor_factory=EyeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": [config.STUN_SERVER]}]},
    )

    processor    = webrtc_ctx.video_processor
    camera_ready = bool(getattr(webrtc_ctx,"state",None)
                        and webrtc_ctx.state.playing and processor is not None)
    if not camera_ready:
        st.markdown(
            '<div style="font-size:13px;color:var(--amber);margin-top:6px;">'
            '⚠️ Camera not running — click Start above and allow permission.</div>',
            unsafe_allow_html=True)

    section("Assessment Control")
    btn1, btn2, _ = st.columns([2, 1, 3])
    with btn1:
        start_clicked = st.button("▶  Start 60s Assessment", type="primary", use_container_width=True)
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
            # reset sparkline buffers + phase tracker
            st.session_state.spark_blink        = []
            st.session_state.spark_perclos      = []
            st.session_state.spark_comfort      = []
            st.session_state["_last_rendered_phase"] = "IDLE"
            st.session_state.pop("_camera_dead_since", None)
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

    # Feedback buttons
    if st.session_state.ui_phase == "RUN" and processor is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px;color:var(--text-lo);margin-bottom:8px;">'
            '✦ Help personalise your AI model — was the assessment accurate right now?</div>',
            unsafe_allow_html=True)
        fb1, fb2, _ = st.columns([2, 2, 2])
        with fb1:
            if st.button("✅ I was NOT tired (over-estimated)", key="fb_not_tired"):
                snap      = processor.get_last()
                pred_high = snap.get("risk_level") in ("MED", "HIGH")
                processor.adaptive_weights.correct(predicted_high=pred_high, was_tired=False)
                _w_path = config.PROFILES_DIR / "adaptive_weights.json"
                processor.adaptive_weights.save(_w_path)
                st.success(f"Model updated ✅ — {processor.adaptive_weights.corrections} correction(s) total")
        with fb2:
            if st.button("😴 I WAS tired (under-estimated)", key="fb_was_tired"):
                snap      = processor.get_last()
                pred_high = snap.get("risk_level") in ("MED", "HIGH")
                processor.adaptive_weights.correct(predicted_high=pred_high, was_tired=True)
                _w_path = config.PROFILES_DIR / "adaptive_weights.json"
                processor.adaptive_weights.save(_w_path)
                st.success(f"Model updated ✅ — {processor.adaptive_weights.corrections} correction(s) total")

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
        ph_ai      = st.empty()

    ph_report = st.empty()

    phase = st.session_state.ui_phase
    snap  = processor.get_last() if processor is not None else {}
    _render(phase, snap, ph_focus, ph_progress, ph_ear,
            ph_comfort, ph_status, ph_break, ph_recs, ph_ai)

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
                        ph_comfort, ph_status, ph_break, ph_recs, ph_ai)
                if not st.session_state.report_done:
                    _compute_and_save_report(processor, st.session_state.run_cfg)
                with ph_report.container():
                    _show_report()
                break

            _render(st.session_state.ui_phase, snap,
                    ph_focus, ph_progress, ph_ear,
                    ph_comfort, ph_status, ph_break, ph_recs, ph_ai)

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
                            ph_comfort, ph_status, ph_break, ph_recs, ph_ai)
                    break
            else:
                st.session_state.pop("_camera_dead_since", None)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER  (unchanged)
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
