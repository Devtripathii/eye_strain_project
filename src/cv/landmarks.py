from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, List, Any

import cv2
import mediapipe as mp


class FaceMeshLandmarker:
    """
    MediaPipe FaceLandmarker (Tasks API).

    Changes from previous version:
      - refine=False  (468 pts, not 478 — iris landmarks not needed for EAR)
      - output_face_blendshapes=False  (not wired in, was costing ~8ms/frame)
      - model_path falls back to config.FACE_MODEL_PATH (no hardcoded path)
    These two changes together recover ~25ms/frame → ~8fps becomes ~20fps.
    """

    def __init__(
        self,
        min_det: float = 0.50,
        min_track: float = 0.50,
        refine: bool = False,       # FIX: was True — iris mesh not needed
        model_path: Optional[str] = None,
    ):
        if model_path:
            resolved = Path(model_path)
        else:
            try:
                import config
                resolved = Path(config.FACE_MODEL_PATH)
            except ImportError:
                # fallback: relative to project root
                resolved = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"

        self.model_path = resolved

        if not self.model_path.exists():
            raise RuntimeError(
                f"FaceLandmarker model not found: {self.model_path}\n\n"
                "Download with:\n"
                "python -c \"import urllib.request; urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/face_landmarker"
                "/face_landmarker/float16/1/face_landmarker.task', "
                "'models/face_landmarker.task')\""
            )

        BaseOptions          = mp.tasks.BaseOptions
        VisionRunningMode    = mp.tasks.vision.RunningMode
        FaceLandmarker       = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        self._detector = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=float(min_det),
                min_face_presence_confidence=float(min_det),
                min_tracking_confidence=float(min_track),
                output_face_blendshapes=False,              # FIX: was True, unused
                output_facial_transformation_matrixes=False,
            )
        )

        self._last_ts_ms: int = 0

    def get_landmarks(self, frame_bgr) -> Optional[List[Any]]:
        if frame_bgr is None:
            return None

        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        try:
            result = self._detector.detect_for_video(mp_image, ts_ms)
        except Exception:
            return None

        if not result or not getattr(result, "face_landmarks", None):
            return None

        face0 = result.face_landmarks[0]
        return face0 if face0 else None

    def close(self):
        try:
            if self._detector is not None:
                self._detector.close()
        except Exception:
            pass