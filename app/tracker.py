"""MediaPipe pose: a camera frame in, 33 body points (just numbers) out."""
import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from . import config as C


class PoseTracker:
    def __init__(self, variant: str = "full"):
        path = C.MODELS_DIR / f"pose_landmarker_{variant}.task"
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found in models/")
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def process(self, bgr: np.ndarray, timestamp_ms: int):
        """Returns a (33, 4) array of x, y, z, visibility, or None if nobody is in view.
        x and y are fractions of the image size, so they don't depend on the resolution."""
        h, w = bgr.shape[:2]
        if w > 640:  # MediaPipe shrinks the image anyway; doing it here is ~3x faster on this laptop
            bgr = cv2.resize(bgr, (640, round(640 * h / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return np.array([[p.x, p.y, p.z, p.visibility if p.visibility is not None else 1.0]
                         for p in result.pose_landmarks[0]], dtype=np.float32)

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
