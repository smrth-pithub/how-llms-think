"""Shared settings for the whole demo. Change things here, not in the other files."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # one folder per person, one .npz per recorded gesture
MODELS_DIR = ROOT / "models"        # Qwen GGUFs, MediaPipe pose model, trained gesture model
WEB_DIR = ROOT / "web"
LLAMA_DIR = ROOT / "tools" / "llama"

# the trained gesture model (+ a .json report next to it); GESTURE_MODEL env var overrides, for testing
GESTURE_MODEL_PATH = Path(os.environ.get("GESTURE_MODEL", MODELS_DIR / "gesture_model.npz"))
GESTURE_REPORT_PATH = GESTURE_MODEL_PATH.with_suffix(".json")

# The seven things the model can say. There is deliberately no "nothing" option.
GESTURES = [
    ("waving", "Waving"),
    ("thumbs_up", "Thumbs up"),
    ("hands_up", "Hands up"),
    ("clapping", "Clapping"),
    ("jumping", "Jumping"),
    ("arms_crossed", "Arms crossed"),
    ("standing_still", "Standing still"),
]
GESTURE_HINTS = {
    "waving": "Wave one hand above your shoulder. Switch hands now and then.",
    "thumbs_up": "Thumbs up at chest height: one hand, then both.",
    "hands_up": "Both hands straight up above your head.",
    "clapping": "Clap in front of your chest, at your own speed.",
    "jumping": "Jump on the spot: small jumps and big ones.",
    "arms_crossed": "Cross your arms over your chest.",
    "standing_still": "Just stand naturally. Small fidgets are fine.",
}
GESTURE_KEYS = [k for k, _ in GESTURES]
GESTURE_LABELS = [label for _, label in GESTURES]

# Sliding window: the classifier looks at the last WINDOW_SECONDS of movement.
WINDOW_SECONDS = 1.0
MIN_FRAMES = 8          # need at least this many frames with a person in the window

# MediaPipe pose landmark indices we use (upper body + hips).
# 0 nose, 7/8 ears, 11/12 shoulders, 13/14 elbows, 15/16 wrists,
# 17/18 pinkies, 19/20 index fingers, 21/22 thumbs, 23/24 hips
POINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# left/right pairs, used to mirror recordings (doubles the training data,
# and makes waving with either hand work)
MIRROR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
                (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16

# Skeleton lines drawn on the camera image
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
            (23, 25), (25, 27), (24, 26), (26, 28)]

# Next-token viewer: models in the ladder, best first.
LLM_MODELS = {
    "Qwen3-1.7B": "Qwen3-1.7B-Q8_0.gguf",
    "Qwen3-0.6B": "Qwen3-0.6B-Q8_0.gguf",
}
LLM_DEFAULT = "Qwen3-1.7B"
LLAMA_PORT = 8090
APP_PORT = 8000
