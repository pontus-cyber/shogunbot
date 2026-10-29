"""Bubble icon templates for tiles first seen on Rei the Ruthless (0.27).

    python -m shogunbot.perception.build_icons ROOT    # ROOT holds runs/

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
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .perceive import SLOT_PITCH, SLOT_Y0, SLOT_Y1, normalize

ASSETS = Path(__file__).parent / "assets"
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
}


def interior(rgb: np.ndarray, cx: int, k: int) -> np.ndarray:
    """The icon template of bubble slot k over the unit at cx, as classify_icon sees it."""
    y0, y1 = SLOT_Y0 - SLOT_PITCH * k, SLOT_Y1 - SLOT_PITCH * k
    return cv2.resize(rgb[y0 + 12:y1 - 12, cx - 50:cx + 50], (25, 24), interpolation=cv2.INTER_AREA)


def build(root: str | Path) -> dict:
    root = Path(root)
    d = np.load(ASSETS / "icons.npz")
    keep = ~np.isin(d["K"], list(SAMPLES))
    X, K = list(d["X"][keep]), list(d["K"][keep])
    added = {}
    for label, rows in SAMPLES.items():
        for rel, cx, k in rows:
            rgb = normalize(np.array(Image.open(root / rel).convert("RGB")))
            X.append(interior(rgb, cx, k))
            K.append(label)
        added[label] = len(rows)
    np.savez_compressed(ASSETS / "icons.npz", X=np.array(X, dtype=np.uint8), K=np.array(K))
    return {"kept": int(keep.sum()), "added": added}


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "."))
