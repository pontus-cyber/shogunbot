"""Consumables lying on the floor (core-requests item 15c, 0.50).

A slain enemy can drop a consumable [wiki]; it lies on its cell until the hero steps there, which puts it in the item
column when the column has a free slot, and what is left when the column is full is lost [user 2026-11-03 09:36]
(data/consumables.json rules 'floor'). The floor sprites are their own small drawings, not the column's: Edamame
Brew a square jar of green liquid, Kami Brew a light-blue flask shaped like a shield, the cards a 9 x 13 block frame
(the column draws the jar at about twice the size, in more detail) [rec: run 20261102-191231 t0653_after].

Every floor sprite sits on the screen's 4-px art grid at a fixed height, anywhere across its cell's plate: 439
Edamame Brews on 25 runs all had their liquid's top at y 728 and its left edge at a multiple of 4, from 68 px left of
the cell's centre to 28 px right of it [scan of 2026-11-03: the runs' before-frames]. So the floor band is read as
4 x 4 blocks, and each template is slid along the blocks of an empty cell. The regions grade their colours (Edamame's
glass rim is 167,144,108 in Hot Springs and 132,144,130 at Moonlit Port's night), so the match is a normalised
correlation per colour channel over the template's mask, which a gain or offset per channel leaves alone.

Templates: perception/assets/floor_items.npz, built by perception/build_floor.py from the frames in its SAMPLES:
Edamame Brew, Kami Brew, Mass Ice, Mass Poison and Rain of Mirrors. Not read yet (no template): Cool Up (no floor frame
known), Mass Curse and the Lucky Die (a red card with a mask and a small pale die face, seen on the floor in runs
20261101-053431 and -164934). When one shows up, add its frames to SAMPLES and rebuild.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSET = Path(__file__).parent / "assets" / "floor_items.npz"
GRID = 4                 # the art's pixel on screen; floor sprites sit on this grid [scan above]
BAND_Y0, BAND_Y1 = 696, 764    # floor band read (templates start at y 700-708 and end above the plates' top, 760)
REACH = 64               # a sprite's centre lies within this many px of its cell's centre (the plate's half width)
# A read counts at MIN_SCORE (the mean of the colour channels' correlations) with its weakest channel at CHANNEL_MIN or
# more [scan of 2026-11-03: every template on every empty cell of the 15,178 before-frames of the 94 runs; 1,841 cell
# reads pass, every one of 80 sampled within 0.06 of its threshold a real item of that kind]. Below them: an item's pale
# flash (its liquid or card drawn in one cream colour: Edamame 0.60-0.66 with a channel at 0.40-0.53, as Kami Brew up
# to 0.82 with one at 0.69 or less), a card's glint (a light stripe down its left side: Mass Ice 0.70-0.78, Mass Poison
# 0.62-0.69), plain backdrops (up to 0.69, a channel at 0.66 or less) and the cards without a template (Mass Curse 0.69
# as Kami Brew, the Lucky Die 0.65). Above them the region's grading: Edamame under Spirit Gateway's sky 0.65-0.72
# (channels 0.61-0.68), Kami Brew 0.82-0.85 (0.81-0.83), Mass Poison 0.82-0.87, Mass Ice 0.89, Rain of Mirrors 0.82-0.89.
# A flash or glint frame reads no item; the tracker keeps an item through one such read (bridge.FLOOR_MISS_MAX) and the
# blink frames read it again (perceive.merge_blink).
MIN_SCORE = {"green_potion": 0.65, "blue_potion": 0.80, "ice_card": 0.85, "skull_card": 0.80, "mirror_card": 0.80}
CHANNEL_MIN = {"green_potion": 0.60, "blue_potion": 0.75, "ice_card": 0.70, "skull_card": 0.60, "mirror_card": 0.70}

_T: dict | None = None


def templates() -> dict:
    """screen id -> (template blocks (rows, cols, 3), mask (rows, cols), top y, mask centre x in px)."""
    global _T
    if _T is None:
        _T = {}
        if ASSET.exists():
            z = np.load(ASSET, allow_pickle=False)
            for sid in [str(s) for s in z["sids"]]:
                t, m, y = z[f"{sid}_t"].astype(float), z[f"{sid}_m"].astype(bool), int(z[f"{sid}_y"])
                cols = np.nonzero(m.any(axis=0))[0]
                _T[sid] = (t, m, y, float((cols.min() + cols.max() + 1) / 2 * GRID))
    return _T


def band_blocks(rgb: np.ndarray) -> np.ndarray:
    """The floor band as 4 x 4 block means on the screen's grid: (rows, width // 4, 3)."""
    a = rgb[BAND_Y0:BAND_Y1, : rgb.shape[1] // GRID * GRID, :3].astype(float)
    rows, cols = a.shape[0] // GRID, a.shape[1] // GRID
    return a[: rows * GRID].reshape(rows, GRID, cols, GRID, 3).mean(axis=(1, 3))


def ncc(t: np.ndarray, m: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel normalised correlation of template t with windows w (n, rows, cols, 3) over the mask m: (mean over
    channels, min over channels), each of shape (n,)."""
    tv = t[m]                                        # (k, 3)
    wv = w[:, m]                                     # (n, k, 3)
    tc = tv - tv.mean(axis=0)
    wc = wv - wv.mean(axis=1, keepdims=True)
    num = (wc * tc).sum(axis=1)                      # (n, 3)
    den = np.sqrt((wc ** 2).sum(axis=1) * (tc ** 2).sum(axis=0)) + 1e-6
    r = num / den
    return r.mean(axis=1), r.min(axis=1)


def scores(B: np.ndarray, cx: int, width: int) -> dict:
    """Each template's best place on the cell centred at cx: screen id -> (score, its weakest channel's, sprite's left
    x)."""
    out = {}
    for sid, (t, m, y, mc) in templates().items():
        rows, cols = m.shape
        r0 = (y - BAND_Y0) // GRID
        if r0 < 0 or r0 + rows > B.shape[0]:
            continue
        lo = int(np.ceil((cx - REACH - mc) / GRID))
        hi = int(np.floor((cx + REACH - mc) / GRID))
        lo, hi = max(lo, 0), min(hi, width // GRID - cols)
        if hi < lo:
            continue
        win = np.lib.stride_tricks.sliding_window_view(B[r0:r0 + rows, lo:hi + cols], (rows, cols), axis=(0, 1))
        win = win[0].transpose(0, 2, 3, 1)           # (n, rows, cols, 3)
        mean, low = ncc(t, m, win)
        k = int(np.argmax(mean))
        out[sid] = (float(mean[k]), float(low[k]), (lo + k) * GRID)
    return out


def match_cell(B: np.ndarray, cx: int, width: int) -> tuple[str | None, float, int]:
    """The item on the cell centred at cx: the template with the highest score among those at their MIN_SCORE (and
    every channel at their CHANNEL_MIN): (screen id or None, score, sprite's left x)."""
    ok = [(sc, sid, px) for sid, (sc, low, px) in scores(B, cx, width).items()
          if sc >= MIN_SCORE.get(sid, 1.0) and low >= CHANNEL_MIN.get(sid, 1.0)]
    if not ok:
        return None, 0.0, 0
    sc, sid, px = max(ok)
    return sid, sc, px


def read(rgb: np.ndarray, cells: list[dict]) -> list[dict]:
    """Items on the floor of the empty cells: [{'x': cell, 'item': screen id, 'score', 'px'}]."""
    if not templates():
        return []
    B = band_blocks(rgb)
    out = []
    for i, c in enumerate(cells):
        if c.get("kind") != "empty":
            continue
        sid, score, px = match_cell(B, int(c["cx"]), rgb.shape[1])
        if sid is not None:
            out.append({"x": i, "item": sid, "score": round(score, 3), "px": px})
    return out
