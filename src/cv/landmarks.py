from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, List, Any

import cv2
import mediapipe as mp


class FaceMeshLandmarker:
    """
    MediaPipe 0.10.31+ removed `mp.solutions`.
    This implementation uses MediaPipe Tasks: FaceLandmarker.

    Returns landmarks in a compatible format: a list of points with .x .y .z
    (NormalizedLandmark objects), matching your existing EAR code.
    """

    def __init__(
        self,
        min_det: float = 0.25,
        min_track: float = 0.25,
        refine: bool = False,  # kept for API compatibility; not used by Tasks
        model_path: Optional[str] = None,
    ):
        project_root = Path(__file__).resolve().parents[2]  # .../src
        default_model = project_root / "models" / "face_landmarker.task"
        self.model_path = Path(model_path) if model_path else default_model

        if not self.model_path.exists():
            raise RuntimeError(
                f"FaceLandmarker model not found:\n{self.model_path}\n\n"
                "Fix: download the model:\n"
                'Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" '
                f'-OutFile "{self.model_path}"\n'
            )

        # Build Tasks detector
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        self._running_mode = VisionRunningMode.VIDEO  # good for webcam loop

        self._detector = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=self._running_mode,
                num_faces=1,
                min_face_detection_confidence=float(min_det),
                min_tracking_confidence=float(min_track),
                # keep outputs minimal (faster)
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )

    def get_landmarks(self, frame_bgr) -> Optional[List[Any]]:
        if frame_bgr is None:
            return None

        # Convert BGR -> RGB for MediaPipe
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Create mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp required for VIDEO mode
        timestamp_ms = int(time.time() * 1000)

        try:
            result = self._detector.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            return None

        if not result or not getattr(result, "face_landmarks", None):
            return None

        # result.face_landmarks is a list (faces) of lists (landmarks)
        # We only requested num_faces=1, so take first.
        face0 = result.face_landmarks[0]
        if not face0:
            return None

        return face0

    def close(self):
        try:
            if self._detector is not None:
                self._detector.close()
        except Exception:
            pass