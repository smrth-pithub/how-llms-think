"""The trained gesture model, run with plain numpy.

Training uses PyTorch (train.py); for the live demo we only need the
forward pass, which is just three matrix multiplies. That keeps the demo
light: no PyTorch loaded while the camera runs.
"""
import json

import numpy as np

from . import config as C


class GestureClassifier:
    def __init__(self, path=C.GESTURE_MODEL_PATH):
        d = np.load(path, allow_pickle=False)
        n = int(d["n_layers"])
        self.layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)]
        self.mean = d["mean"]
        self.std = d["std"]
        self.keys = [str(k) for k in d["keys"]]
        self.labels = [str(k) for k in d["labels"]]
        self.report = {}
        report = path.with_suffix(".json")
        if report.exists():
            try:
                self.report = json.loads(report.read_text(encoding="utf-8"))
            except Exception:
                self.report = {}

    def logits(self, features: np.ndarray) -> np.ndarray:
        """Raw scores, one per gesture. Not percentages yet: softmax does that."""
        x = (features - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)          # ReLU
        return x


def load_or_none():
    if not C.GESTURE_MODEL_PATH.exists():
        return None
    try:
        return GestureClassifier()
    except Exception as e:  # a half-written or old model file should not crash the demo
        print(f"[gesture] could not load model: {e}")
        return None
