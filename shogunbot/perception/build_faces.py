"""Build perception/assets/faces.npz: tile-face icon templates.

    python -m shogunbot.perception.build_faces ROOT    # ROOT holds runs/ and collect/

Sources:
  PEDESTALS  tile-choice frames with both offers named by hand
  HAND_RUNS  bot runs with the default kit: every hand face is labelled
             by its cooldown marks (3 = Swirl, 5 = Arrow)
  MIXED      frames whose hand also holds other tiles: faces with an
             ambiguous mark count are labelled by the templates built so far
When a new tile shows up in a run, add its frame to PEDESTALS (or MIXED)
and rebuild.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import reward_screen, tiles as T

PEDESTALS = {
    # frame: [left offer, right offer]   [bot/collect frames, labels read on screen]
    "runs/20261027-180145/frames/f0030_CLICK.png": ["arrow", "katana"],
    "runs/20261027-182237/frames/f0026_CLICK.png": ["back_charge", "shuriken"],
    "runs/20261028-044407/frames/f0034_CLICK.png": ["shuriken", "charge"],
    "runs/20261028-053637/frames/f0032_CLICK.png": ["charge", "swirl"],
    "runs/20261028-061420/frames/f0034_CLICK.png": ["back_charge", "swirl"],
    "collect/20261027-164727/f0042_CLICK.png": ["shadow_dash", "spear"],
}
HAND_RUNS = ["runs/20261027-180145/frames", "runs/20261027-182237/frames", "runs/20261028-044407/frames",
             "runs/20261028-053637/frames", "runs/20261028-061420/frames"]
DEFAULT_BY_MARKS = {3: "swirl", 5: "arrow"}
MIXED = ("collect/20261027-164727", {5: ("arrow", "spear"), 3: ("swirl",)})
PER_BUCKET = 12   # hand samples kept per (tile, ready, charged) bucket


def _load(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def build(root: Path) -> dict:
    V, K, S = [], [], []

    def add(rgb, box, label, src):
        V.append(T.icon_vec(T.grid(rgb, box)))
        K.append(label)
        S.append(src)

    for f, labels in PEDESTALS.items():
        rgb = _load(root / f)
        boxes = reward_screen.pedestal_faces(rgb)
        if len(boxes) != len(labels):
            raise SystemExit(f"{f}: {len(boxes)} pedestal faces, expected {len(labels)}")
        for b, lab in zip(boxes, labels):
            add(rgb, b, lab, f)

    buckets = defaultdict(int)
    for d in HAND_RUNS:
        for p in sorted((root / d).glob("f*.png")):
            rgb = _load(p)
            for b in T.find_faces(rgb, 600, 1320, *T.HAND_ROW):
                g = T.grid(rgb, b)
                cd, ch = T.read_marks(g)
                lab = DEFAULT_BY_MARKS.get(cd)
                if lab is None:
                    continue
                key = (lab, T.is_ready(g), ch)
                if buckets[key] >= PER_BUCKET:
                    continue
                buckets[key] += 1
                add(rgb, b, lab, str(p.relative_to(root)))

    # mixed hands: label ambiguous faces with the templates built so far
    T._T = (np.array(V, np.float32), np.array(K))
    d, by_cd = MIXED
    for p in sorted((root / d).glob("f*.png")):
        rgb = _load(p)
        for b in T.find_faces(rgb, 600, 1320, *T.HAND_ROW):
            g = T.grid(rgb, b)
            cd, ch = T.read_marks(g)
            cands = by_cd.get(cd)
            if not cands:
                continue
            lab, dist, conf = T.classify(g, cands)
            if lab is None or (len(cands) > 1 and conf < 0.3):
                continue
            key = ("mixed", lab, T.is_ready(g), ch)
            if buckets[key] >= PER_BUCKET:
                continue
            buckets[key] += 1
            add(rgb, b, lab, str(p.relative_to(root)))

    out = Path(T.ASSETS) / "faces.npz"
    np.savez_compressed(out, V=np.array(V, np.float32), K=np.array(K), S=np.array(S))
    T._T = None
    counts = defaultdict(int)
    for k in K:
        counts[k] += 1
    return {"file": str(out), "templates": len(K), "by_tile": dict(counts)}


if __name__ == "__main__":
    print(json.dumps(build(Path(sys.argv[1] if len(sys.argv) > 1 else ".")), indent=1))
