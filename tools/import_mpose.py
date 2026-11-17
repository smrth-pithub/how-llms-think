"""Adds the public MPOSE2021 dataset as extra training data (about 90 MB download, once).

    .venv\\Scripts\\python -m tools.import_mpose

MPOSE2021: 15,429 one-second clips of body points from 100 people, CC BY 4.0.
Mazzia, Angarano, Salvetti, Angelini, Chiaberge (2021), https://zenodo.org/records/5507363

Five of our seven gestures are in it (waving, clapping, jumping, arms crossed, standing still);
thumbs up and hands up are not, so your own recordings are still needed. Its "scratch-head"
clips are kept apart, never trained on, and used to test the head-scratch moment.

Writes data_extra/mpose/<person>/<gesture>_<clip>.npz (same format as the app's recordings)
and data_extra/headscratch/<person>/scratch_head_<clip>.npz.
"""
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config as C  # noqa: E402

URL = "https://zenodo.org/records/5507363/files/"
FILES = {"1.zip": 88475554, "1.txt": 560222}
DOWNLOADS = ROOT / "downloads" / "mpose2021"
OUT = C.EXTRA_DIR / "mpose"
OUT_SCRATCH = C.EXTRA_DIR / "headscratch"
FPS = 25.0                      # the source videos are (mostly) 25 frames a second
CLIPS_PER_GESTURE = 150         # enough variety without drowning out the club's own recordings

# MPOSE2021 label number -> our gesture key
LABELS = {0: "standing_still", 2: "arms_crossed", 8: "waving",
          14: "clapping", 17: "jumping", 18: "jumping"}
SCRATCH_HEAD = 3

# MPOSE2021 uses PoseNet's 17 COCO points; this is where each one goes in MediaPipe's 33.
COCO_TO_MP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
              11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
NEEDED = [0, 5, 6, 7, 8, 9, 10, 11, 12]   # nose, shoulders, elbows, wrists, hips must be seen


def download():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        dest = DOWNLOADS / name
        if dest.exists() and dest.stat().st_size == size:
            continue
        print(f"  downloading MPOSE2021 {name} ({size / 1e6:.0f} MB)...", flush=True)
        req = urllib.request.Request(URL + name + "?download=1", headers={"User-Agent": "how-llms-think-setup"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest.with_suffix(".part"), "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        dest.with_suffix(".part").replace(dest)
        if dest.stat().st_size != size:
            raise RuntimeError(f"{name}: incomplete download, run again")
    with zipfile.ZipFile(DOWNLOADS / "1.zip") as z:
        for n in ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"):
            if not (DOWNLOADS / n).exists():
                z.extract(n, DOWNLOADS)


def to_frames(clip):
    """(30, 17, 3) pixel x, y, confidence -> (T, 33, 4) like the app records, plus frame times.
    Frames where the needed points weren't seen are dropped. Finger points (which PoseNet
    doesn't have) are estimated from the forearm direction."""
    frames, times = [], []
    S = 1000.0                                 # same scale on both axes; features don't care which
    for i, f in enumerate(clip):
        if (f[NEEDED, 2] < 0.05).any() or (f[NEEDED, :2] == 0).any():
            continue
        out = np.zeros((33, 4), np.float32)
        for c, m in COCO_TO_MP.items():
            out[m, :2] = f[c, :2] / S
            out[m, 3] = f[c, 2]
        sw = np.linalg.norm(out[11, :2] - out[12, :2]) + 1e-6
        for ear, side in ((7, 1), (8, -1)):        # ears often missed: put them beside the nose
            if f[3 if ear == 7 else 4, 2] < 0.05:
                out[ear, :2] = out[0, :2] + np.array([0.2 * sw * side, 0.0])
        for wrist, elbow, pinky, index, thumb in ((15, 13, 17, 19, 21), (16, 14, 18, 20, 22)):
            fwd = out[wrist, :2] - out[elbow, :2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
            side = np.array([-fwd[1], fwd[0]])
            out[pinky, :2] = out[wrist, :2] + 0.18 * sw * fwd - 0.06 * sw * side
            out[index, :2] = out[wrist, :2] + 0.20 * sw * fwd + 0.06 * sw * side
            out[thumb, :2] = out[wrist, :2] + 0.12 * sw * fwd + 0.12 * sw * side
            out[[pinky, index, thumb], 3] = out[wrist, 3]
        frames.append(out)
        times.append(i / FPS)
    return frames, times


def main():
    if OUT.exists() and any(OUT.glob("*/*.npz")):
        print(f"MPOSE2021 already imported ({sum(1 for _ in OUT.glob('*/*.npz'))} clips in {OUT.relative_to(ROOT)})")
        return 0
    download()
    rows = [l.split("\t") for l in (DOWNLOADS / "1.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    clips = []                                  # (name, person, label, data)
    for split in ("train", "test"):
        names = [r[0] for r in rows if r[1].strip() == split]
        X = np.load(DOWNLOADS / f"X_{split}.npy", allow_pickle=False)
        y = np.load(DOWNLOADS / f"y_{split}.npy", allow_pickle=False)
        assert len(names) == len(X) == len(y)
        for n, x, lab in zip(names, X, y):
            if int(lab) in LABELS or int(lab) == SCRATCH_HEAD:
                clips.append((n, "_".join(n.split("_")[:2]), int(lab), x))

    # spread each gesture's quota across as many different people as possible
    rng = np.random.default_rng(0)
    by_gesture = defaultdict(lambda: defaultdict(list))
    for c in clips:
        key = "scratch_head" if c[2] == SCRATCH_HEAD else LABELS[c[2]]
        by_gesture[key][c[1]].append(c)
    chosen = []
    for key, people in by_gesture.items():
        pools = [rng.permutation(len(v)).tolist() for v in people.values()]
        lists = list(people.values())
        picked = []
        quota = 10 ** 9 if key == "scratch_head" else CLIPS_PER_GESTURE
        while len(picked) < quota and any(pools):
            for pool, lst in zip(pools, lists):
                if pool and len(picked) < quota:
                    picked.append(lst[pool.pop()])
        chosen += [(key, c) for c in picked]

    for d in (OUT, OUT_SCRATCH):
        if d.exists():
            shutil.rmtree(d)
    saved = defaultdict(int)
    people = set()
    for key, (name, person, _, x) in chosen:
        frames, times = to_frames(x)
        if len(frames) < 20:
            continue
        folder = (OUT_SCRATCH if key == "scratch_head" else OUT) / person
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"{key}_{name}.npz", frames=np.stack(frames), times=np.asarray(times),
                            width=1000, height=1000, gesture=key, person=person, source="MPOSE2021")
        saved[key] += 1
        if key != "scratch_head":
            people.add(person)
    (C.EXTRA_DIR / "ATTRIBUTION.txt").write_text(
        "data_extra/ is derived from MPOSE2021 (CC BY 4.0):\n"
        "V. Mazzia, S. Angarano, F. Salvetti, F. Angelini, M. Chiaberge, \"Action Transformer: A Self-Attention\n"
        "Model for Short-Time Pose-Based Human Action Recognition\", Pattern Recognition, 2021.\n"
        "Dataset: https://zenodo.org/records/5507363\n"
        "Changes: PoseNet points remapped to MediaPipe's layout, finger points estimated, frames without a\n"
        "visible upper body dropped, a subset of 5 actions kept (+ scratch-head for testing).\n", encoding="utf-8")
    print("MPOSE2021 imported: " + ", ".join(f"{k} {v}" for k, v in sorted(saved.items()))
          + f" clips, from {len(people)} people")
    return 0


if __name__ == "__main__":
    sys.exit(main())
