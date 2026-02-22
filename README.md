# Eye Comfort Assistant (Eye Strain & Fatigue Monitor)

A real-time camera-based eye comfort assistant that estimates eye strain/fatigue using:
- MediaPipe FaceMesh (eye landmarks)
- Eye metrics: blink rate, eye closure %, etc.
- Optional CNN model (ResNet18) for eye-closure probability
- Smart break scheduler + fatigue accumulation
- Streamlit dashboard + session logging

> Not a medical diagnosis.

---

## Quick Start

### 1) Create environment (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt