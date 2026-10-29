"""Bubble icon templates for tiles first seen on Rei the Ruthless (0.27) and Iwao (0.28).

    python -m shogunbot.perception.build_icons ROOT    # ROOT holds runs/ and collect/

`perception/assets/icons.npz` holds the enemy bubble icons that `perceive.classify_icon` matches
(an icon interior shrunk to 25 x 24 px). The Stage 1 icons (Sword, Swirl, Arrow, Charge, Spear,
Summon) came with 0.10 and stay as they are; this builder owns the labels in SAMPLES and replaces
them on every run, so it can be run again when samples are added.

Rei's icons [bot: runs 20261029-072323 ... -103242, every bubble over her on the boss frames]:
Dragon Punch is a white fist trailing a flame, damage 1 (0.24 read it as Charge 48 times and as
Sword 9 times); Back Charge is Charge's teal arrow pointing down (read as Charge or Sword); the red
bell is her double-strike Summon, which brings two Twin Tachi (read as Arrow, Charge or Summon);
her orange bell is the ordinary Summon (read right: the orange rule). Samples: every Back Charge
and red bell seen, every fourth Dragon Punch, spread over the five runs.

Iwao the Impaler's icons (0.28) [bot: run 20261029-124921 t159-t198; rec: collect/20261028-190026
f0285-f0318]: Earth Impale is brown spikes rising from the ground, damage 2 (read as Spear at
confidence 0.15-0.8); his Swirl, stacked on an Earth Impale, read as Spear 7 times and as Swirl
once; his Spear is the Yari Master's drawing (read as Spear at 0.1-1.0, and on the Yari Master as
Sword 12 of 14 times). 'swirl' and 'spear' are Stage 1 labels too: the builder owns the templates
it added (array B), not whole labels, so the Stage 1 templates stay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .perceive import SLOT_PITCH, SLOT_Y0, SLOT_Y1, normalize

ASSETS = Path(__file__).parent / "assets"
LEGACY_LABELS = ("dragon_punch", "back_charge", "summon_double")   # what the 0.27 builder added
SAMPLES = {   # label: [(frame relative to ROOT, the unit's centre x at 1080p, slot from the bottom)]
    "dragon_punch": [
        ("runs/20261029-072323/frames/t0068_before.png", 1284, 0),
        ("runs/20261029-095311/frames/t0076_before.png", 1192, 0),
        ("runs/20261029-100218/frames/t0068_before.png", 1380, 1),
        ("runs/20261029-100218/frames/t0084_before.png", 1284, 1),
        ("runs/20261029-100218/frames/t0090_before.png", 632, 0),
        ("runs/20261029-100218/frames/t0106_before.png", 444, 0),
        ("runs/20261029-100218/frames/t0114_before.png", 732, 1),
        ("runs/20261029-100218/frames/t0124_before.png", 1240, 0),
        ("runs/20261029-100218/frames/t0134_before.png", 1004, 1),
        ("runs/20261029-101549/frames/t0065_before.png", 1332, 1),
        ("runs/20261029-101549/frames/t0071_before.png", 1096, 0),
        ("runs/20261029-101549/frames/t0085_before.png", 640, 0),
        ("runs/20261029-103242/frames/t0069_before.png", 1284, 0),
        ("runs/20261029-103242/frames/t0079_before.png", 1284, 0),
        ("runs/20261029-103242/frames/t0091_before.png", 1151, 0),
    ],
    "back_charge": [
        ("runs/20261029-095311/frames/t0085_before.png", 1152, 0),
        ("runs/20261029-100218/frames/t0071_before.png", 1340, 0),
        ("runs/20261029-100218/frames/t0072_before.png", 1292, 0),
        ("runs/20261029-100218/frames/t0073_before.png", 1240, 0),
        ("runs/20261029-100218/frames/t0074_before.png", 1192, 0),
        ("runs/20261029-100218/frames/t0094_before.png", 1012, 0),
        ("runs/20261029-100218/frames/t0095_before.png", 1012, 0),
        ("runs/20261029-100218/frames/t0096_before.png", 1012, 0),
        ("runs/20261029-100218/frames/t0097_before.png", 1012, 0),
        ("runs/20261029-101549/frames/t0094_before.png", 676, 0),
    ],
    "summon_double": [
        ("runs/20261029-072323/frames/t0064_before.png", 1332, 0),
        ("runs/20261029-072323/frames/t0065_before.png", 1332, 0),
        ("runs/20261029-095311/frames/t0070_before.png", 1384, 0),
        ("runs/20261029-095311/frames/t0071_before.png", 1384, 0),
        ("runs/20261029-100218/frames/t0061_before.png", 1384, 0),
        ("runs/20261029-100218/frames/t0062_before.png", 1384, 0),
        ("runs/20261029-101549/frames/t0059_before.png", 1332, 0),
        ("runs/20261029-101549/frames/t0060_before.png", 1332, 0),
        ("runs/20261029-103242/frames/t0064_before.png", 1332, 0),
        ("runs/20261029-103242/frames/t0065_before.png", 1332, 0),
    ],
    "earth_impale": [   # Iwao, alone (hero not adjacent) or under his Swirl
        ("runs/20261029-124921/frames/t0165_before.png", 1292, 0),
        ("runs/20261029-124921/frames/t0167_before.png", 1151, 0),
        ("runs/20261029-124921/frames/t0184_before.png", 1284, 0),
        ("runs/20261029-124921/frames/t0186_before.png", 1284, 0),
        ("runs/20261029-124921/frames/t0187_before.png", 1284, 0),
        ("collect/20261028-190026/f0289_idle.png", 1151, 0),
        ("collect/20261028-190026/f0291_CLICK.png", 1100, 0),
        ("collect/20261028-190026/f0293_CLICK.png", 1100, 0),
        ("collect/20261028-190026/f0296_idle.png", 1100, 0),
        ("collect/20261028-190026/f0297_CLICK.png", 1100, 0),
        ("collect/20261028-190026/f0313_idle.png", 1096, 0),
        ("collect/20261028-190026/f0316_CLICK.png", 1096, 0),
    ],
    "swirl": [   # Iwao: the Swirl he stacks on an Earth Impale when the hero is adjacent
        ("runs/20261029-124921/frames/t0186_before.png", 1284, 1),
        ("runs/20261029-124921/frames/t0187_before.png", 1284, 1),
        ("collect/20261028-190026/f0292_idle.png", 1100, 1),
        ("collect/20261028-190026/f0294_SPACE.png", 1100, 1),
        ("collect/20261028-190026/f0297_CLICK.png", 1100, 1),
        ("collect/20261028-190026/f0316_CLICK.png", 1096, 1),
    ],
    "spear": [   # Iwao's Spear (first ten), then the Yari Masters'
        ("runs/20261029-124921/frames/t0161_before.png", 1151, 0),
        ("runs/20261029-124921/frames/t0162_before.png", 1100, 0),
        ("runs/20261029-124921/frames/t0170_before.png", 1240, 0),
        ("runs/20261029-124921/frames/t0172_before.png", 1151, 0),
        ("runs/20261029-124921/frames/t0179_before.png", 1380, 0),
        ("runs/20261029-124921/frames/t0181_before.png", 1240, 0),
        ("runs/20261029-124921/frames/t0190_before.png", 1528, 0),
        ("runs/20261029-124921/frames/t0192_before.png", 1284, 0),
        ("collect/20261028-190026/f0308_D.png", 1332, 0),
        ("collect/20261028-190026/f0309_D.png", 1284, 0),
        ("runs/20261029-124921/frames/t0180_before.png", 820, 0),
        ("runs/20261029-124921/frames/t0181_before.png", 820, 0),
        ("collect/20261028-190026/f0308_D.png", 492, 0),
        ("collect/20261028-190026/f0309_D.png", 444, 0),
        ("collect/20261028-190026/f0310_W.png", 536, 0),
        ("collect/20261028-190026/f0313_idle.png", 536, 0),
        ("collect/20261028-190026/f0315_CLICK.png", 536, 0),
    ],
}


def interior(rgb: np.ndarray, cx: int, k: int) -> np.ndarray:
    """The icon template of bubble slot k over the unit at cx, as classify_icon sees it."""
    y0, y1 = SLOT_Y0 - SLOT_PITCH * k, SLOT_Y1 - SLOT_PITCH * k
    return cv2.resize(rgb[y0 + 12:y1 - 12, cx - 50:cx + 50], (25, 24), interpolation=cv2.INTER_AREA)


def build(root: str | Path) -> dict:
    root = Path(root)
    d = np.load(ASSETS / "icons.npz")
    # B marks the templates this builder added; files from 0.27 lack it, when it added only these
    built = d["B"] if "B" in d.files else np.isin(d["K"], LEGACY_LABELS)
    keep = ~built
    X, K = list(d["X"][keep]), list(d["K"][keep])
    B = [False] * len(K)
    added = {}
    for label, rows in SAMPLES.items():
        for rel, cx, k in rows:
            rgb = normalize(np.array(Image.open(root / rel).convert("RGB")))
            X.append(interior(rgb, cx, k))
            K.append(label)
            B.append(True)
        added[label] = len(rows)
    np.savez_compressed(ASSETS / "icons.npz", X=np.array(X, dtype=np.uint8), K=np.array(K), B=np.array(B))
    return {"kept": int(keep.sum()), "added": added}


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "."))
