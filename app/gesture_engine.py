"""The live gesture pipeline, running in a background thread:

webcam frame -> MediaPipe landmarks -> sliding window -> classifier -> raw scores

The browser receives the raw scores and does softmax + temperature + smoothing
itself, so the code shown on screen is the code making the percentages.
The camera only runs while a gesture or record page is open.
"""
import os
import re
import threading
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C
from .classifier import load_or_none  # noqa: E402
from .features import WindowBuffer  # noqa: E402
from .tracker import PoseTracker  # noqa: E402


def open_camera(index: int, width: int, height: int):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # lets USB webcams do 30 fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())[:32].strip("_")
    return name or "person"


def draw_skeleton(img, lm):
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in lm[:, :2]]
    vis = lm[:, 3]
    th = max(2, w // 320)
    for a, b in C.SKELETON:
        if vis[a] > 0.3 and vis[b] > 0.3:
            cv2.line(img, pts[a], pts[b], (255, 255, 255), th + 3, cv2.LINE_AA)
            cv2.line(img, pts[a], pts[b], (235, 99, 37), th, cv2.LINE_AA)
    for i in C.POINTS:
        if vis[i] > 0.3:
            cv2.circle(img, pts[i], th + 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[i], th + 1, (37, 99, 235), -1, cv2.LINE_AA)


class ReplaySource:
    """TESTING ONLY (--replay DIR): plays saved landmark recordings instead of a webcam,
    a few seconds per clip, drawn as a stick figure on a dark background."""

    def __init__(self, folder, seconds_per_clip=4.0):
        from pathlib import Path
        self.clips = []
        for f in sorted(Path(folder).glob("*/*.npz")):
            d = np.load(f, allow_pickle=False)
            self.clips.append((d["frames"], d["times"], int(d["width"]), int(d["height"])))
        if not self.clips:
            raise FileNotFoundError(f"no recordings in {folder}")
        self.order = np.random.default_rng(1).permutation(len(self.clips))
        self.per_clip = seconds_per_clip
        self.t0 = time.perf_counter()

    def read(self):
        t = time.perf_counter() - self.t0
        k = int(t // self.per_clip)
        frames, times, w, h = self.clips[self.order[k % len(self.clips)]]
        local = (t % self.per_clip) + 1.0
        i = min(int(np.searchsorted(times, local)), len(frames) - 1)
        scale = 720 / h                                   # same shape as the recording
        img = np.full((720, round(w * scale), 3), 28, np.uint8)
        return img, frames[i].copy()


class GestureEngine:
    def __init__(self, camera: int = 0, width: int = 1280, height: int = 720, pose_model: str = "full",
                 replay: str | None = None):
        self.camera, self.width, self.height, self.pose_model = camera, width, height, pose_model
        self.replay = replay
        self._lock = threading.Lock()
        self._users = 0
        self._last_user_left = 0.0
        self._thread: threading.Thread | None = None
        self.clf = load_or_none()
        self._reload = False
        self.rec = None
        self.saved_this_session: list[str] = []
        self._state = self._blank_state("stopped")
        self._jpeg = None

    # ---- lifecycle: the camera runs only while someone is watching ----
    def acquire(self):
        with self._lock:
            self._users += 1
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def release(self):
        with self._lock:
            self._users = max(0, self._users - 1)
            if self._users == 0:
                self._last_user_left = time.time()

    def reload_model(self):
        self._reload = True
        if self._thread is None or not self._thread.is_alive():
            self.clf = load_or_none()

    def model_info(self):
        clf = self.clf
        if clf is None:
            return dict(trained=False, labels=C.GESTURE_LABELS)
        return dict(trained=True, labels=clf.labels, report=clf.report)

    # ---- recording ----
    def start_recording(self, person: str, gestures: list[str], seconds: float, countdown: float):
        gestures = [g for g in gestures if g in C.GESTURE_KEYS] or list(C.GESTURE_KEYS)
        with self._lock:
            self.rec = dict(person=safe_name(person), queue=gestures, idx=0, seconds=float(seconds),
                            countdown=float(countdown), phase="countdown",
                            phase_end=time.time() + countdown, frames=[], times=[], t0=0.0,
                            missing=0, saved=[], messages=[])

    def stop_recording(self):
        with self._lock:
            self.rec = None

    def discard_last(self):
        if not self.saved_this_session:
            return None
        p = self.saved_this_session.pop()
        try:
            os.remove(p)
        except OSError:
            pass
        return os.path.basename(p)

    # ---- state for the browser ----
    def _blank_state(self, status, message=""):
        return dict(frame_id=0, status=status, message=message, has_person=False, logits=None,
                    fps=0.0, infer_ms=0.0, wrist=None, n_in_window=0, rec=None,
                    model=self.model_info(), width=0, height=0)

    def snapshot(self):
        with self._lock:
            return dict(self._state), self._jpeg

    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    def _rec_public(self):
        r = self.rec
        if r is None:
            return None
        g = r["queue"][r["idx"]] if r["idx"] < len(r["queue"]) else None
        return dict(person=r["person"], phase=r["phase"], index=r["idx"], total=len(r["queue"]),
                    gesture=g, label=C.GESTURE_LABELS[C.GESTURE_KEYS.index(g)] if g else None,
                    remaining=max(0.0, r["phase_end"] - time.time()), saved=list(r["saved"]),
                    messages=list(r["messages"][-3:]), frames=len(r["frames"]))

    def _rec_step(self, lm, t, w, h):
        r = self.rec
        if r is None or r["phase"] == "done":
            return
        now = time.time()
        if r["phase"] == "countdown":
            if now >= r["phase_end"]:
                r.update(phase="recording", phase_end=now + r["seconds"], frames=[], times=[], t0=t, missing=0)
            return
        # recording
        if lm is not None:
            r["frames"].append(lm)
            r["times"].append(t - r["t0"])
        else:
            r["missing"] += 1
        if now < r["phase_end"]:
            return
        g = r["queue"][r["idx"]]
        n = len(r["frames"])
        if n >= 20:
            folder = C.DATA_DIR / r["person"]
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{g}_{time.strftime('%Y%m%d-%H%M%S')}.npz"
            np.savez(path, frames=np.stack(r["frames"]).astype(np.float32),
                     times=np.asarray(r["times"], np.float64), width=w, height=h,
                     gesture=g, person=r["person"])
            self.saved_this_session.append(str(path))
            r["saved"].append(g)
            seen = n / max(1, n + r["missing"])
            if seen < 0.8:
                r["messages"].append(f"{C.GESTURE_LABELS[C.GESTURE_KEYS.index(g)]}: person only visible "
                                     f"{100 * seen:.0f}% of the time")
        else:
            r["messages"].append(f"{C.GESTURE_LABELS[C.GESTURE_KEYS.index(g)]}: nobody seen, not saved")
        r["idx"] += 1
        if r["idx"] >= len(r["queue"]):
            r["phase"] = "done"
        else:
            r.update(phase="countdown", phase_end=now + r["countdown"])

    # ---- the loop ----
    def _run(self):
        self._set(status="starting", message="Opening camera...")
        replay = cap = tracker = None
        if self.replay:
            replay = ReplaySource(self.replay)
        else:
            cap = open_camera(self.camera, self.width, self.height)
            if cap is None:
                self._set(status="error", message=f"No camera at index {self.camera}. "
                                                  f"Plug in the webcam, or restart with --camera 1")
                self._idle_until_unused()
                return
            try:
                tracker = PoseTracker(self.pose_model)
            except Exception as e:
                cap.release()
                self._set(status="error", message=f"MediaPipe failed to start: {e}")
                self._idle_until_unused()
                return

        buf = WindowBuffer()
        t_start = time.perf_counter()
        last_ts = -1
        fps = 0.0
        prev = None
        fails = 0
        dark_since = None
        frame_id = self._state.get("frame_id", 0)
        self._set(status="running", message="")
        try:
            while True:
                with self._lock:
                    if self._users == 0 and time.time() - self._last_user_left > 2.0:
                        break
                if self._reload:
                    self._reload = False
                    self.clf = load_or_none()
                    buf.clear()
                replay_lm = None
                if replay is not None:
                    time.sleep(0.04)                       # ~25 fps, like a webcam
                    ok, (frame, replay_lm) = True, replay.read()
                else:
                    ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails > 30:
                        self._set(status="error", message="Camera stopped sending frames")
                    time.sleep(0.02)
                    continue
                fails = 0
                t = time.perf_counter() - t_start
                ts = max(int(t * 1000), last_ts + 1)
                last_ts = ts
                h, w = frame.shape[:2]

                t0 = time.perf_counter()
                lm = replay_lm if replay is not None else tracker.process(frame, ts)
                if lm is not None:
                    buf.add(lm, t)
                else:
                    buf.prune(t)
                logits = None
                clf = self.clf
                if clf is not None and buf.ready():
                    logits = clf.logits(buf.features(w / h)).tolist()
                infer_ms = (time.perf_counter() - t0) * 1000

                with self._lock:
                    self._rec_step(lm, t, w, h)
                    rec = self._rec_public()

                if lm is not None:
                    draw_skeleton(frame, lm)
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])

                now = time.perf_counter()
                if prev is not None:
                    inst = 1.0 / max(now - prev, 1e-3)
                    fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                prev = now
                frame_id += 1
                wrist = None
                if lm is not None:
                    wrist = [round(float(lm[C.R_WRIST, 0]), 3), round(float(lm[C.R_WRIST, 1]), 3)]
                message = ""
                if replay is None and frame_id % 15 == 0:
                    dark = float(frame[::16, ::16].mean()) < 5
                    dark_since = (dark_since or time.time()) if dark else None
                if dark_since and time.time() - dark_since > 2:
                    message = (f"Camera {self.camera} sends a black picture. Wrong camera, closed shutter, "
                               f"or another app is using it? Run check_camera.bat")
                with self._lock:
                    self._jpeg = jpg.tobytes() if ok else None
                    self._state.update(frame_id=frame_id, status="running", message=message,
                                       has_person=lm is not None, logits=logits, fps=round(fps, 1),
                                       infer_ms=round(infer_ms, 1), wrist=wrist, n_in_window=len(buf.frames),
                                       rec=rec, model=self.model_info(), width=w, height=h)
        finally:
            if cap is not None:
                cap.release()
            if tracker is not None:
                tracker.close()
            with self._lock:
                self._jpeg = None
                if self.rec is not None and self.rec["phase"] != "done":
                    self.rec = None            # page closed mid-recording: drop the unfinished clip
                self._state.update(status="stopped", message="", logits=None, has_person=False, rec=None)

    def _idle_until_unused(self):
        while True:
            with self._lock:
                if self._users == 0:
                    return
            time.sleep(0.5)
