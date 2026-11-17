"""Train the gesture classifier on everything in data/.

    python -m app.train                 # train + test on people it never saw
    python -m app.train --data data_synth   (test data, for checking the code only)

"Done when" from the plan: the model tells the gestures apart on someone it
wasn't trained on. So with 2+ people we hold out one person at a time,
train on the rest, and score the held-out person.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C
from .features import augment, mirror, window_features, window_slices


def log(*a):
    print(*a, flush=True)


EXTRA_PREFIX = "mpose:"     # people from the public MPOSE2021 dataset (always in training, never tested)


def load_recordings(data_dir: Path, prefix="", trim=0.5, keys=None):
    """trim = seconds skipped at the start of each clip (people getting into position).
    MPOSE2021 clips are already cut to the action, so they use 0."""
    keys = set(keys or C.GESTURE_KEYS)
    recs = []
    for f in sorted(data_dir.glob("*/*.npz")):
        try:
            d = np.load(f, allow_pickle=False)
            g = str(d["gesture"])
            if g not in keys:
                continue
            recs.append(dict(person=prefix + f.parent.name, gesture=g, frames=d["frames"], times=d["times"],
                             aspect=float(d["width"]) / float(d["height"]), file=f.name, trim=trim))
        except Exception as e:
            log(f"  skipping {f}: {e}")
    return recs


def build(recs, use_mirror=True, n_augment=1, stride=2, seed=0):
    """Every recording is used as-is, mirrored (left/right swapped), and n_augment more times
    as a slightly different 'person' (see features.augment). Only the as-is version is ever
    used for testing; the extra versions are marked with ~ in their group name."""
    rng = np.random.default_rng(seed)
    X, y, groups, tfrac = [], [], [], []
    for r in recs:
        label = C.GESTURE_KEYS.index(r["gesture"])
        t0 = r["times"]
        versions = [(r["frames"], t0, "")]
        if use_mirror:
            versions.append((mirror(r["frames"]), t0, "~m"))
        for i in range(n_augment):
            src = mirror(r["frames"]) if (use_mirror and rng.random() < 0.5) else r["frames"]
            f, t = augment(src, t0, r["aspect"], rng)
            versions.append((f, t, f"~a{i}"))
        for frames, t, tag in versions:
            span = max(t[-1] - t[0], 1e-6)
            for lo, hi in window_slices(t, stride=stride, trim=r.get("trim", 0.5)):
                X.append(window_features(frames[lo:hi], t[lo:hi], r["aspect"]))
                y.append(label)
                groups.append(r["person"] + tag)
                tfrac.append((t[hi - 1] - t[0]) / span)
    return (np.asarray(X, np.float32), np.asarray(y, np.int64),
            np.asarray(groups), np.asarray(tfrac, np.float32))


def fit(X, y, epochs=60, seed=0):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Xt = torch.tensor((X - mean) / std)
    yt = torch.tensor(y)
    n_cls = len(C.GESTURE_KEYS)
    model = nn.Sequential(
        nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, n_cls),                       # one raw score (logit) per gesture
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    # label smoothing keeps the model from being 100% sure of everything,
    # so the temperature slider has something to show at T = 1
    # class weights: gestures with fewer examples (thumbs up and hands up have no MPOSE2021 clips)
    # count for more, so the model doesn't just learn to avoid them
    counts = np.bincount(y, minlength=n_cls).astype(np.float64)
    cw = np.where(counts > 0, len(y) / (n_cls * np.maximum(counts, 1)), 0.0).clip(0, 5)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32), label_smoothing=0.1)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            xb = Xt[idx] + 0.1 * torch.randn(len(idx), Xt.shape[1], generator=gen)  # jitter
            loss = loss_fn(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    linears = [m for m in model if isinstance(m, nn.Linear)]
    weights = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in linears]
    return weights, mean, std


def predict(weights, mean, std, X):
    x = (X - mean) / std
    for i, (W, b) in enumerate(weights):
        x = x @ W + b
        if i < len(weights) - 1:
            x = np.maximum(x, 0)
    return x.argmax(1)


def headscratch_check(weights, mean, std):
    """Segment 1's moment, tested: MPOSE2021 clips of people scratching their head (never trained on).
    The model has no 'nothing' option, so what does it say?"""
    recs = load_recordings(C.EXTRA_DIR / "headscratch", prefix=EXTRA_PREFIX, trim=0.0, keys=["scratch_head"])
    if not recs:
        return None
    feats = [window_features(r["frames"][lo:hi], r["times"][lo:hi], r["aspect"])
             for r in recs for lo, hi in window_slices(r["times"], stride=2, trim=0.0)]
    if not feats:
        return None
    x = (np.asarray(feats, np.float32) - mean) / std
    for i, (W, b) in enumerate(weights):
        x = x @ W + b
        if i < len(weights) - 1:
            x = np.maximum(x, 0)
    p = np.exp(x - x.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    top = p.argmax(1)
    share = {C.GESTURE_LABELS[k]: round(100 * float((top == k).mean()), 1) for k in range(len(C.GESTURE_KEYS))}
    share = dict(sorted(((k, v) for k, v in share.items() if v > 0), key=lambda kv: -kv[1]))
    log(f"\nHead-scratch check ({len(recs)} clips of people scratching their head, never trained on):")
    log("  the model's answer: " + ", ".join(f"{k} {v:.0f}%" for k, v in share.items()))
    log(f"  how sure it is of its answer, on average (temperature 1.0): {100 * float(p.max(1).mean()):.0f}%")
    log("  (these clips are full-body, filmed in labs; try a real head-scratch on your own webcam too)")
    return dict(answers=share, mean_confidence=round(100 * float(p.max(1).mean()), 1), clips=len(recs))


def confusion(y_true, y_pred):
    n = len(C.GESTURE_KEYS)
    m = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def print_confusion(m):
    short = [lbl[:8] for lbl in C.GESTURE_LABELS]
    log("            true \\ predicted")
    log("  " + " " * 15 + "".join(f"{s:>9}" for s in short))
    for i, row in enumerate(m):
        tot = max(row.sum(), 1)
        log(f"  {C.GESTURE_LABELS[i]:<15}" + "".join(f"{v:>9}" for v in row) + f"   {100 * row[i] / tot:5.1f}%")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_DIR))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-eval", action="store_true", help="skip the held-out-person test")
    ap.add_argument("--with-mpose", action="store_true",
                    help="also train on the public MPOSE2021 clips (tools/import_mpose.py). Off by default: "
                         "with 4 club members it scored 82.6%% vs 84.1%% on unseen people")
    ap.add_argument("--augment", type=int, default=1,
                    help="extra varied copies of each recording (speed, camera tilt, arm length); 0 = off")
    ap.add_argument("--out", default=str(C.GESTURE_MODEL_PATH))
    args = ap.parse_args(argv)

    t0 = time.time()
    data_dir = Path(args.data)
    recs = load_recordings(data_dir)
    extra = load_recordings(C.EXTRA_DIR / "mpose", prefix=EXTRA_PREFIX, trim=0.0) if args.with_mpose else []
    if not recs and not extra:
        log(f"No recordings found in {data_dir}. Record some on the Record page first.")
        return 1
    people = sorted({r["person"] for r in recs})
    if recs:
        log(f"Loaded {len(recs)} recordings from {len(people)} people: {', '.join(people)}")
    else:
        log("No club recordings yet: training on the public MPOSE2021 clips only.")
    extra_people = sorted({r["person"] for r in extra})
    if extra:
        log(f"Plus {len(extra)} MPOSE2021 clips from {len(extra_people)} more people "
            f"(CC BY 4.0, Mazzia et al. 2021): always used for training, never for the test below")
    elif args.with_mpose:
        log("(No MPOSE2021 extra data found; run: .venv\\Scripts\\python -m tools.import_mpose)")
    have = {g: sorted({r['person'] for r in recs + extra if r['gesture'] == g}) for g in C.GESTURE_KEYS}
    missing = [C.GESTURE_LABELS[i] for i, g in enumerate(C.GESTURE_KEYS) if not have[g]]
    if missing:
        log(f"WARNING: no recordings at all for: {', '.join(missing)}. The model can never predict these well.")

    X, y, groups, tfrac = build(recs + extra, n_augment=args.augment)
    base = np.array([g.split("~")[0] for g in groups])
    is_extra = np.array([g.startswith(EXTRA_PREFIX) for g in groups])
    unmirrored = np.array(["~" not in g for g in groups])
    log(f"Each recording used as-is, mirrored, and {args.augment} more time(s) with a different speed, "
        f"camera tilt and arm length")
    log(f"{len(X)} training windows ({X.shape[1]} numbers each), "
        + ", ".join(f"{C.GESTURE_LABELS[i]} {int((y == i).sum())}" for i in range(len(C.GESTURE_KEYS))))

    report = dict(people=people, n_windows=int(len(X)), n_features=int(X.shape[1]), augment=args.augment,
                  extra=dict(source="MPOSE2021", clips=len(extra), people=len(extra_people)) if extra else None,
                  trained_at=time.strftime("%Y-%m-%d %H:%M"), heldout=None)

    if not args.no_eval and not people:
        # no club recordings: test on MPOSE2021 people held out from training (5 gestures only)
        rng = np.random.default_rng(0)
        held = set(rng.choice(extra_people, size=max(1, len(extra_people) // 5), replace=False).tolist())
        te = np.isin(base, list(held)) & unmirrored
        tr = ~np.isin(base, list(held))
        log(f"\nTest on {len(held)} MPOSE2021 people the model never saw (only the 5 gestures they did):")
        w, mu, sd = fit(X[tr], y[tr], epochs=args.epochs)
        pred = predict(w, mu, sd, X[te])
        acc = float((pred == y[te]).mean())
        log(f"  accuracy {100 * acc:.1f}%  ({te.sum()} windows)\n")
        print_confusion(confusion(y[te], pred))
        report["heldout"] = dict(kind="unseen MPOSE2021 people", overall=round(100 * acc, 1))
    elif not args.no_eval:
        if len(people) >= 2:
            log("\nTest on people the model never saw (train on everyone else, test on them):")
            all_t, all_p, per = [], [], {}
            for p in people:
                tr = base != p
                te = groups == p                    # held-out person, un-mirrored only
                if tr.sum() == 0 or te.sum() == 0:
                    continue
                w, mu, sd = fit(X[tr], y[tr], epochs=args.epochs)
                pred = predict(w, mu, sd, X[te])
                acc = float((pred == y[te]).mean())
                per[p] = round(100 * acc, 1)
                log(f"  {p:<16} {100 * acc:5.1f}%  ({te.sum()} windows)")
                all_t.append(y[te]); all_p.append(pred)
            yt, yp = np.concatenate(all_t), np.concatenate(all_p)
            overall = float((yt == yp).mean())
            log(f"  {'overall':<16} {100 * overall:5.1f}%\n")
            print_confusion(confusion(yt, yp))
            report["heldout"] = dict(kind="unseen people", overall=round(100 * overall, 1), per_person=per)
        else:
            log("\nOnly one person recorded, so we can't test on an unseen person yet.")
            log("Testing on the last 30% of each recording instead (easier than a new person):")
            tr = (tfrac < 0.7) | is_extra
            te = (tfrac >= 0.7) & unmirrored & ~is_extra
            w, mu, sd = fit(X[tr], y[tr], epochs=args.epochs)
            pred = predict(w, mu, sd, X[te])
            acc = float((pred == y[te]).mean())
            log(f"  accuracy {100 * acc:.1f}%\n")
            print_confusion(confusion(y[te], pred))
            report["heldout"] = dict(kind="same person, later in the clip", overall=round(100 * acc, 1))

    log("\nTraining the final model on everyone...")
    w, mu, sd = fit(X, y, epochs=args.epochs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"n_layers": np.array(len(w)), "mean": mu, "std": sd,
              "keys": np.array(C.GESTURE_KEYS), "labels": np.array(C.GESTURE_LABELS)}
    for i, (W, b) in enumerate(w):
        arrays[f"W{i}"] = W.astype(np.float32)
        arrays[f"b{i}"] = b.astype(np.float32)
    report["headscratch"] = headscratch_check(w, mu, sd)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    train_acc = float((predict(w, mu, sd, X) == y).mean())
    log(f"Saved {out.name}  (accuracy on its own training data: {100 * train_acc:.1f}%)")
    log(f"Done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
