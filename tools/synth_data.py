"""FOR TESTING THE CODE ONLY: fake stick-figure recordings.

Writes data_synth/<person>/<gesture>.npz in the same format as real
recordings, so train.py and the live page can be exercised without a webcam.
Never mix this into data/ - a model trained on stick figures is useless on people.

    python -m tools.synth_data --people 4
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config as C  # noqa: E402

W, H = 640, 480


def arm(shoulder, side, a_deg, b_deg, sw, ua=0.75, fa=0.68):
    """side +1 = person's left (image right). Angles from straight down, + = outward."""
    def d(deg):
        r = math.radians(deg)
        return np.array([side * math.sin(r), math.cos(r)])
    elbow = shoulder + ua * sw * d(a_deg)
    wrist = elbow + fa * sw * d(b_deg)
    return elbow, wrist


def hand(wrist, elbow, sw, thumb_up=False):
    fwd = wrist - elbow
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    side = np.array([-fwd[1], fwd[0]])
    if thumb_up:
        return wrist + 0.08 * sw * fwd + 0.05 * sw * side, wrist + 0.1 * sw * fwd, wrist + np.array([0, -0.28 * sw])
    pinky = wrist + 0.18 * sw * fwd - 0.06 * sw * side
    index = wrist + 0.2 * sw * fwd + 0.06 * sw * side
    thumb = wrist + 0.12 * sw * fwd + 0.12 * sw * side
    return pinky, index, thumb


def pose(g, t, p, rng):
    sw, cx, sy = p["sw"], p["cx"], p["sy"]
    f = p["speed"]
    amp = p["amp"]
    dy = 0.0
    if g == "jumping":
        dy = -0.35 * amp * sw * abs(math.sin(math.pi * 1.8 * f * t))
    sway = 0.03 * sw * math.sin(2 * math.pi * 0.3 * t)
    cx = cx + sway
    sy = sy + dy
    L = np.array([cx + sw / 2, sy])
    R = np.array([cx - sw / 2, sy])
    pts = np.zeros((33, 2))
    nose = np.array([cx, sy - 0.62 * sw])
    pts[0] = nose
    for i, ox in zip(range(1, 7), [-0.05, -0.08, -0.11, 0.05, 0.08, 0.11]):
        pts[i] = nose + np.array([-ox * sw, -0.08 * sw])
    pts[7] = nose + np.array([0.2 * sw, -0.02 * sw])
    pts[8] = nose + np.array([-0.2 * sw, -0.02 * sw])
    pts[9] = nose + np.array([0.06 * sw, 0.1 * sw])
    pts[10] = nose + np.array([-0.06 * sw, 0.1 * sw])
    pts[11], pts[12] = L, R

    thumbs = [False, False]
    if g == "standing_still":
        eL, wL = arm(L, +1, 8, 4, sw)
        eR, wR = arm(R, -1, 8, 4, sw)
    elif g == "hands_up":
        eL, wL = arm(L, +1, 165, 172, sw)
        eR, wR = arm(R, -1, 165, 172, sw)
    elif g == "waving":
        osc = 30 * amp * math.sin(2 * math.pi * 2.5 * f * t)
        if p["left_hand"]:
            eL, wL = arm(L, +1, 95, 160 + osc, sw)
            eR, wR = arm(R, -1, 8, 4, sw)
        else:
            eR, wR = arm(R, -1, 95, 160 + osc, sw)
            eL, wL = arm(L, +1, 8, 4, sw)
    elif g == "clapping":
        s = (1 + math.sin(2 * math.pi * 2.2 * f * t)) / 2
        gap = (0.08 + 0.3 * amp * s) * sw
        eL = np.array([cx + 0.75 * sw, sy + 0.85 * sw])
        eR = np.array([cx - 0.75 * sw, sy + 0.85 * sw])
        wL = np.array([cx + gap, sy + 0.75 * sw])
        wR = np.array([cx - gap, sy + 0.75 * sw])
    elif g == "jumping":
        k = 15 * abs(math.sin(math.pi * 1.8 * f * t))
        eL, wL = arm(L, +1, 15 + k, 10 + k, sw)
        eR, wR = arm(R, -1, 15 + k, 10 + k, sw)
    elif g == "arms_crossed":
        eL = np.array([cx + 0.62 * sw, sy + 0.9 * sw])
        eR = np.array([cx - 0.62 * sw, sy + 0.9 * sw])
        wL = np.array([cx - 0.45 * sw, sy + 0.72 * sw])
        wR = np.array([cx + 0.45 * sw, sy + 0.72 * sw])
    elif g == "thumbs_up":
        eL, wL = arm(L, +1, 8, 4, sw)
        eR = np.array([cx - 0.65 * sw, sy + 0.95 * sw])
        wR = np.array([cx - 0.35 * sw, sy + 0.55 * sw])
        thumbs[1] = True
        if p["both_thumbs"]:
            eL = np.array([cx + 0.65 * sw, sy + 0.95 * sw])
            wL = np.array([cx + 0.35 * sw, sy + 0.55 * sw])
            thumbs[0] = True
    else:
        raise ValueError(g)

    pts[13], pts[14], pts[15], pts[16] = eL, eR, wL, wR
    pts[17], pts[19], pts[21] = hand(wL, eL, sw, thumbs[0])
    pts[18], pts[20], pts[22] = hand(wR, eR, sw, thumbs[1])
    hip_y = sy + 1.3 * sw
    pts[23] = np.array([cx + 0.35 * sw, hip_y])
    pts[24] = np.array([cx - 0.35 * sw, hip_y])
    for k, (a, b) in enumerate([(23, 25), (24, 26), (25, 27), (26, 28)]):
        pts[b] = pts[a] + np.array([0, 0.9 * sw])
    pts[29], pts[30] = pts[27] + [0, 0.05 * sw], pts[28] + [0, 0.05 * sw]
    pts[31], pts[32] = pts[27] + [0.1 * sw, 0.08 * sw], pts[28] + [-0.1 * sw, 0.08 * sw]

    pts += rng.normal(0, 0.025 * sw, pts.shape)
    out = np.zeros((33, 4), np.float32)
    out[:, 0] = pts[:, 0] / W
    out[:, 1] = pts[:, 1] / H
    out[:, 3] = 0.95
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--out", default=str(ROOT / "data_synth"))
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    out = Path(args.out)
    for i in range(args.people):
        person = f"synth{i + 1}"
        p = dict(sw=rng.uniform(70, 110), cx=rng.uniform(250, 390), sy=rng.uniform(140, 200),
                 speed=rng.uniform(0.75, 1.25), amp=rng.uniform(0.75, 1.25),
                 left_hand=bool(rng.random() < 0.3), both_thumbs=bool(rng.random() < 0.5))
        fps = rng.uniform(18, 30)
        (out / person).mkdir(parents=True, exist_ok=True)
        for g in C.GESTURE_KEYS:
            n = int(args.seconds * fps)
            times = np.cumsum(rng.uniform(0.8, 1.2, n) / fps)
            frames = np.stack([pose(g, t, p, rng) for t in times])
            np.savez(out / person / f"{g}.npz", frames=frames, times=times, width=W, height=H,
                     gesture=g, person=person)
        print(f"wrote {person} ({fps:.0f} fps)")


if __name__ == "__main__":
    main()
