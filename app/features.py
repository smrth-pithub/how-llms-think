"""Landmarks -> one feature vector per sliding window.

Used by BOTH training and the live demo, so what the model sees live is
exactly what it was trained on.

A "frame" is MediaPipe's 33 pose landmarks, each (x, y, z, visibility),
with x and y as fractions of the image width and height.
"""
from collections import deque

import numpy as np

from . import config as C

_WRIST_L = C.POINTS.index(C.L_WRIST)
_WRIST_R = C.POINTS.index(C.R_WRIST)


def mirror(frames: np.ndarray) -> np.ndarray:
    """Flip a recording left-to-right (a right-hand wave becomes a left-hand wave)."""
    f = frames.copy()
    f[..., 0] = 1.0 - f[..., 0]
    for a, b in C.MIRROR_PAIRS:
        f[:, [a, b]] = f[:, [b, a]]
    return f


# arm chains per side: shoulder, elbow, wrist, then the hand points that move with the wrist
_ARMS = [(11, 13, 15, (17, 19, 21)), (12, 14, 16, (18, 20, 22))]


def augment(frames: np.ndarray, times: np.ndarray, aspect: float, rng: np.random.Generator):
    """Make a recording look like a slightly different person on a slightly different camera:
    a bit faster or slower, camera tilted a few degrees, arms a bit longer or shorter.
    Training only: it stretches a handful of people into more variety."""
    speed = rng.uniform(0.8, 1.25)
    new_times = times / speed

    f = frames.copy()
    xy = f[..., :2].astype(np.float64)
    xy[..., 0] = (xy[..., 0] - 0.5) * aspect          # same units on both axes, centred
    xy[..., 1] -= 0.5

    for shoulder, elbow, wrist, hand in _ARMS:          # arm length: upper arm and forearm separately
        k_upper, k_fore = rng.uniform(0.88, 1.12, size=2)
        old_elbow, old_wrist = xy[:, elbow].copy(), xy[:, wrist].copy()
        xy[:, elbow] = xy[:, shoulder] + (old_elbow - xy[:, shoulder]) * k_upper
        xy[:, wrist] = xy[:, elbow] + (old_wrist - old_elbow) * k_fore
        shift = xy[:, wrist] - old_wrist
        for h in hand:
            xy[:, h] += shift

    a = np.deg2rad(rng.uniform(-8, 8))                  # camera tilt
    c, s = np.cos(a), np.sin(a)
    x, y = xy[..., 0].copy(), xy[..., 1].copy()
    xy[..., 0], xy[..., 1] = c * x - s * y, s * x + c * y

    f[..., 0] = xy[..., 0] / aspect + 0.5
    f[..., 1] = xy[..., 1] + 0.5
    return f, new_times


def window_features(frames: np.ndarray, times: np.ndarray, aspect: float) -> np.ndarray:
    """frames: (T, 33, 4), times: (T,) in seconds, aspect: image width / height.

    Returns a 1-D float32 vector:
      body shape   - where each point sits relative to the shoulders (mean over the window)
      movement     - how much each point moves (spread) and how fast (speed)
      whole body   - how the body as a whole moves up/down (jumping)
      hands        - distance between the wrists and how fast it changes (clapping)
    """
    xy = frames[..., :2].astype(np.float64).copy()
    xy[..., 0] *= aspect                                   # make x and y the same unit
    ls, rs = xy[:, C.L_SHOULDER], xy[:, C.R_SHOULDER]
    center = (ls + rs) / 2                                  # (T, 2) mid-shoulder
    scale = max(float(np.median(np.linalg.norm(ls - rs, axis=1))), 1e-3)  # shoulder width

    pts = (xy[:, C.POINTS] - center[:, None, :]) / scale    # (T, P, 2) body-relative
    dt = np.clip(np.diff(times), 1e-3, 0.5)
    vel = np.diff(pts, axis=0) / dt[:, None, None]           # shoulder-widths per second

    c = center / scale
    cvel = np.diff(c, axis=0) / dt[:, None]

    wd = np.linalg.norm(pts[:, _WRIST_L] - pts[:, _WRIST_R], axis=1)
    wdv = np.diff(wd) / dt

    feats = [
        pts.mean(axis=0).ravel(),
        pts.std(axis=0).ravel(),
        np.abs(vel).mean(axis=0).ravel(),
        c.std(axis=0),
        np.abs(cvel).mean(axis=0),
        [c[:, 1].max() - c[:, 1].min()],
        [wd.mean(), wd.std(), np.abs(wdv).mean()],
    ]
    return np.concatenate([np.asarray(f, dtype=np.float64).ravel() for f in feats]).astype(np.float32)


def n_features() -> int:
    t = np.linspace(0, 1, C.MIN_FRAMES)
    f = np.random.default_rng(0).random((C.MIN_FRAMES, 33, 4))
    return int(window_features(f, t, 4 / 3).shape[0])


def window_slices(times: np.ndarray, stride: int = 2, trim: float = 0.5):
    """Yield (lo, hi) index ranges of every sliding window in one recording.
    The first `trim` seconds are skipped (people are still getting into position)."""
    first_end = times[0] + trim + C.WINDOW_SECONDS
    for end in range(0, len(times), stride):
        if times[end] < first_end:
            continue
        lo = int(np.searchsorted(times, times[end] - C.WINDOW_SECONDS, side="left"))
        if end - lo + 1 >= C.MIN_FRAMES:
            yield lo, end + 1


class WindowBuffer:
    """The live sliding window: keeps the last WINDOW_SECONDS of frames that had a person."""

    def __init__(self):
        self.frames: deque = deque()
        self.times: deque = deque()

    def add(self, landmarks: np.ndarray, t: float):
        self.frames.append(landmarks)
        self.times.append(t)
        self.prune(t)

    def prune(self, now: float):
        while self.times and now - self.times[0] > C.WINDOW_SECONDS:
            self.times.popleft()
            self.frames.popleft()

    def clear(self):
        self.frames.clear()
        self.times.clear()

    def ready(self) -> bool:
        return len(self.frames) >= C.MIN_FRAMES

    def features(self, aspect: float) -> np.ndarray:
        return window_features(np.stack(self.frames), np.asarray(self.times), aspect)
