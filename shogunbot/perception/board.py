"""Floor-plate detection: finds the arena cells and what stands on them.

All coordinates are for a 1920x1080 client area; frames of other sizes are
resized first (see perceive.normalize).
"""
from __future__ import annotations

import numpy as np

PLATE_Y0, PLATE_Y1 = 764, 782     # rows sampled inside the floor plates
MIN_W, MAX_W = 100, 150           # plate width in px
PITCH = 140                       # cell spacing in px


def plate_runs(rgb: np.ndarray) -> list[tuple[int, int]]:
    """[(x0, x1)] of bright horizontal runs on the plate row."""
    band = rgb[PLATE_Y0:PLATE_Y1].astype(np.int16)
    lum = band.sum(axis=2).mean(axis=0)            # per-column brightness
    on = lum > 66
    runs, x = [], 0
    W = len(on)
    while x < W:
        if on[x]:
            x0 = x
            while x < W and on[x]:
                x += 1
            runs.append((x0, x))
        x += 1
    # merge tiny gaps (pixel art noise) then keep plate-sized runs
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 4:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    out = []
    for x0, x1 in merged:
        w = x1 - x0
        if MIN_W <= w <= MAX_W:
            out.append((x0, x1))
        elif w > MAX_W:  # two plates touching: split by pitch
            k = round(w / PITCH)
            for i in range(k):
                out.append((x0 + i * PITCH, x0 + i * PITCH + PITCH - 12))
    return out


def plate_kind(rgb: np.ndarray, x0: int, x1: int) -> str:
    """'hero' | 'enemy' | 'empty' from the plate's mean colour."""
    patch = rgb[PLATE_Y0:PLATE_Y1, x0 + 8:x1 - 8].reshape(-1, 3).astype(float)
    r, g, b = patch.mean(axis=0)
    # hero plate: teal-grey stone ends (145,168,157)/(97,127,121). Checked first,
    # because right after a swap its tick box is all red and the mean colour
    # no longer reads blue [run 20261028-044407: two stalls after swapping]
    p = patch.astype(int)
    teal = ((np.abs(p - (145, 168, 157)).max(axis=1) < 16) | (np.abs(p - (97, 127, 121)).max(axis=1) < 16)).mean()
    if teal > 0.12:
        return "hero"
    if r > g + 40 and r > b + 40:
        return "enemy"
    if b > g - 5 and b > 70:
        return "hero"
    return "empty"


def cells(rgb: np.ndarray) -> list[dict]:
    """Left-to-right cells: {x0, x1, cx, kind, rgb}."""
    out = []
    for x0, x1 in plate_runs(rgb):
        patch = rgb[PLATE_Y0:PLATE_Y1, x0 + 8:x1 - 8].reshape(-1, 3).mean(axis=0)
        out.append({"x0": x0, "x1": x1, "cx": (x0 + x1) // 2,
                    "kind": plate_kind(rgb, x0, x1), "rgb": [int(v) for v in patch]})
    return out
