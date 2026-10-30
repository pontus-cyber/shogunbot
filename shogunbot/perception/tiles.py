"""Tile faces: the dark 104 x 108 px squares that show a tile in the hand, on
reward pedestals and in queue bubbles.

The game is pixel art drawn at 4 screen px per art px (1080p), and faces sit
on that 4-px grid, so a face reads as a 26 x 27 art-pixel image:
  digit  art cols 4-9, rows 6-15: damage, a blocky 6 x 10 digit
  icon   art cols 12-25, rows 1-20
  marks  art rows 22-24: one slanted mark per cooldown point, grey when
         charged, red while recharging; cooldown 0 is a flat line
  dots   above the face (reward screens only): one per upgrade level,
         yellow once used
[measured on run 20261028-061420 f0014-f0034 and collect 20261027-164727 f0042-f0112]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FACE_W, FACE_H = 104, 108
ART = 4
ASSETS = Path(__file__).parent / "assets"
HAND_ROW = (880, 1030)          # y band of the hand faces (top y 900; 888 while hovered)
PEDESTAL_ROW = (380, 560)       # y band of tile-choice pedestal faces (top y 408)
_FILL_MIN, _FILL_MAX = 12, 70   # face background: dark grey (24-42 per channel), not black, not wood
_TINT_MIN = 0.05                # share of tinted px that marks an enchanted face


def _tint(a: np.ndarray) -> np.ndarray:
    """The maroon of a Free-Play face: interior (34,20,30), marks band (57,26,38), red-speckled
    border. Its band fails the grey rule (spread 32), so 0.25 cut the face 24 px short and lost it:
    the freeplay Arrow was never read in the hand and the bot could not play it (16 failed inputs
    in the boss fight of run 20261029-100218, t0060-t0143). Wood has B < G, grey faces R < B + 10."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (a.max(axis=2) < _FILL_MAX) & (g >= _FILL_MIN) & (b >= g + 6) & (r >= b + 10) & (r - g < 45)


def _fill(a: np.ndarray, tinted: bool = False) -> np.ndarray:
    mx, mn = a.max(axis=2), a.min(axis=2)
    grey = (mx < _FILL_MAX) & (mn >= _FILL_MIN) & (mx - mn < 26)
    return grey | _tint(a) if tinted else grey


def _runs(mask) -> list[tuple[int, int]]:
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def find_faces(rgb: np.ndarray, x0: int, x1: int, y0: int, y1: int,
               tinted: bool = False) -> list[tuple[int, int, int, int]]:
    """Face boxes (x0, y0, x1, y1) inside a region, left to right.

    Columns of a face always hold some background pixels (its top and bottom
    rows); the wood or scene between faces holds none, so column runs
    separate the faces. Each run is then cut to the face height by rows.
    tinted: also find enchanted (Free-Play) faces; used for the hand, where the
    wood and the black around it hold no such colour [0.26: of 5,490 recorded frames
    only the Free-Play faces of run 20261029-100218 changed, besides four JPEG map
    captures (collect/20261029-kbd-map, where the hand is not read) whose compression
    noise moved the boxes 11 px]."""
    a = rgb[y0:y1, x0:x1].astype(np.int16)
    d = _fill(a, tinted)
    tint = _tint(a) if tinted else None
    m = d.mean(axis=0)
    out = []
    for c0, c1 in _runs(m > 0.03):
        k = max(1, round((c1 - c0 + 16) / 120))   # faces touching: 120 px pitch
        if k == 1 and c1 - c0 > FACE_W:
            # a few stray fill columns next to a face: keep the face-wide window with the most
            # fill. A bird flying past the right pedestal added 12 such columns and shifted the
            # box 12 px left, so Shadow Dash read as an unknown tile with no damage digit
            # [run 20261028-153448 reward_1_tile_choice: columns 1068-1079 at 0.04-0.06 fill,
            # the face from 1080 at 0.47+]
            c0 = max(range(c0, c1 - FACE_W + 1), key=lambda s: (float(m[s:s + FACE_W].sum()), -s))
            c1 = c0 + FACE_W
        for j in range(k):
            cc0 = c0 + j * 120
            cc1 = min(c1, cc0 + FACE_W)
            if not FACE_W - 8 <= cc1 - cc0 <= FACE_W + 6:
                continue
            for r0, r1 in _runs(d[:, cc0:cc1].mean(axis=1) > 0.25):
                if FACE_H - 12 <= r1 - r0 <= FACE_H + 8:
                    if tint is not None and r1 - r0 < FACE_H and tint[r0:r1, cc0:cc1].mean() > _TINT_MIN:
                        r0 = r1 - FACE_H      # an enchanted face's speckled top border is no fill
                    out.append((x0 + cc0, y0 + r0, x0 + cc0 + FACE_W, y0 + r0 + FACE_H))
    return out


def grid(rgb: np.ndarray, box) -> np.ndarray:
    """27 x 26 x 3 art-pixel image of a face (block means of 4 x 4 px)."""
    x0, y0 = box[0], box[1]
    a = rgb[y0:y0 + FACE_H, x0:x0 + FACE_W].astype(np.float32)
    if a.shape[:2] != (FACE_H, FACE_W):
        a = np.pad(a, ((0, FACE_H - a.shape[0]), (0, FACE_W - a.shape[1]), (0, 0)))
    return a.reshape(27, ART, 26, ART, 3).mean(axis=(1, 3))


# ---------------------------------------------------------------- marks
def read_marks(g: np.ndarray) -> tuple[int, int]:
    """(cooldown, charged) from the marks band. A mark is 2 art px wide;
    grey = charged, red = still recharging. A flat line means cooldown 0.
    The marks are centred on the face at a 3 art px pitch: eight span art columns 2-24, so the band
    runs from column 1 to 24; 0.15-0.32.1 stopped at 23, cut the eighth mark to one column and read a
    cooldown of 8 as 7 [run 20261030-041652 t0237, t1026: the 4/8 Arrow, 7 grey marks and a red one].
    On the bot's own frames of all runs (fights, rewards, shops) the wider band changes only the reads of
    cooldown-8 tiles (also the Free-Play Arrow of runs 20261029-100218 and -103242) and of map frames, where
    the hand is not read; on the recordings, where the cursor sits on the hand, 7 faces under a hover
    outline or a click flash read differently (2 right only with the wider band, 2 only with the narrower,
    3 boxes off the face) [25,722 faces, 2026-10-30]."""
    band = g[22:25, 1:25]
    lum = band.mean(axis=2)
    lit = lum.max(axis=0) > 70
    runs = _runs(lit)
    if len(runs) == 1 and runs[0][1] - runs[0][0] >= 6:
        return 0, 0                               # flat line: no cooldown
    # a mark is exactly 2 art px wide, one blank px apart; the sparkle of a tile that has
    # just recharged adds a 1- or 3-px blob that read Swirl as 4 marks on 15 frames
    # [review of 0.15; runs 20261028-061420 f0023/f0029/f0071, 20261027-180145 f0083,
    # 20261028-044407 f0042, 20261028-082756 f0032]. Keep the longest evenly spaced chain.
    marks, chain = [], []
    for r in runs:
        if r[1] - r[0] != 2:
            chain = []
            continue
        if chain and r[0] - chain[-1][1] != 1:
            chain = []
        chain.append(r)
        if len(chain) > len(marks):
            marks = list(chain)
    charged = 0
    for c0, c1 in marks:
        px = band[:, c0:c1].reshape(-1, 3)
        px = px[px.mean(axis=1) > 70]
        r, gg = px[:, 0].mean(), px[:, 1].mean()
        if r <= gg + 30:
            charged += 1
    return len(marks), charged


# ---------------------------------------------------------------- digit
def _glyph(segs: str) -> np.ndarray:
    """6 x 10 blocky digit from 7-segment names (a top, b upper right, c lower
    right, d bottom, e lower left, f upper left, g middle), 2-px strokes."""
    m = np.zeros((10, 6), bool)
    if "a" in segs: m[0:2, :] = True
    if "g" in segs: m[4:6, :] = True
    if "d" in segs: m[8:10, :] = True
    if "f" in segs: m[0:6, 0:2] = True
    if "b" in segs: m[0:6, 4:6] = True
    if "e" in segs: m[4:10, 0:2] = True
    if "c" in segs: m[4:10, 4:6] = True
    return m


ONE = np.zeros((10, 6), bool)
ONE[0:2, 0:4] = True
ONE[0:10, 2:4] = True
ONE[8:10, :] = True
# '1' and '2' are measured [run 20261028-061420 f0034]; the rest follow the same
# 2-px seven-segment style and are a hypothesis until seen
DIGITS = {"1": ONE, "2": _glyph("abged"), "0": _glyph("abcdef"), "3": _glyph("abgcd"), "4": _glyph("fgbc"),
          "5": _glyph("afgcd"), "6": _glyph("afgedc"), "7": _glyph("abc"), "8": _glyph("abcdefg"),
          "9": _glyph("abcdfg")}
DIGITS_SEEN = {"1", "2"}


def read_digit(g: np.ndarray) -> tuple[int | None, float]:
    """(damage, confidence). None when no digit is drawn (N/A tiles)."""
    area = g[6:16, 4:10].mean(axis=2)
    if area.max() < 60:
        return None, 0.0
    m = area > 0.55 * area.max()
    best, d1, d2 = None, 99, 99
    for k, t in DIGITS.items():
        d = int((m != t).sum())
        if d < d1:
            best, d1, d2 = k, d, d1
        elif d < d2:
            d2 = d
    if d1 > 8:
        return None, 0.0
    conf = float(np.clip((d2 - d1) / 10, 0, 1)) * (1.0 if best in DIGITS_SEEN else 0.5)
    return int(best), conf


def is_ready(g: np.ndarray) -> bool:
    """Ready tiles draw the digit bright white; recharging ones are dimmed
    (digit max ~205 vs ~120 on run 20261028-061420 f0014-f0016)."""
    return float(g[6:16, 4:10].mean(axis=2).max()) > 170


# ---------------------------------------------------------------- icon
def icon_vec(g: np.ndarray) -> np.ndarray:
    """Icon region, scaled so its brightest pixel is 1: a dimmed (recharging)
    tile then matches the ready template."""
    ic = g[1:21, 12:26].astype(np.float32)
    bg = np.median(g[1:3, 1:25].reshape(-1, 3), axis=0)
    ic = np.clip(ic - bg, 0, None)
    mx = ic.mean(axis=2).max()
    return (ic / max(mx, 1.0)).ravel()


_T = None


def templates():
    global _T
    if _T is None:
        p = ASSETS / "faces.npz"
        if p.exists():
            d = np.load(p)
            _T = (d["V"].astype(np.float32), d["K"])
        else:
            _T = (np.zeros((0, 20 * 14 * 3), np.float32), np.array([], dtype="<U20"))
    return _T


def classify(g: np.ndarray, allowed=None) -> tuple[str | None, float, float]:
    """(tile id, distance, confidence) by nearest template, restricted to `allowed`."""
    V, K = templates()
    if not len(K):
        return None, 1e9, 0.0
    v = icon_vec(g)
    d = ((V - v) ** 2).mean(axis=1)
    if allowed is not None:
        d = np.where(np.isin(K, list(allowed)), d, np.inf)
    i = int(np.argmin(d))
    if not np.isfinite(d[i]):
        return None, 1e9, 0.0
    k = str(K[i])
    rest = d[(K != k) & np.isfinite(d)]
    other = float(rest.min()) if len(rest) else float(d[i]) * 3 + 0.02
    conf = float(np.clip((other - d[i]) / max(other, 1e-6), 0, 1))
    return k, float(d[i]), conf


# ---------------------------------------------------------------- dots
def read_dots(rgb: np.ndarray, box) -> tuple[int, int]:
    """(levels, used) from the dots above a face on reward screens: grey dots
    (~158,163,145) are free levels, golden ones (~200,156,24) are used
    [collect 20261027-164727 f0112]. Dots are 12 px wide on a 20 px pitch,
    centred over the face, 29 px above its top edge."""
    x0, y0 = box[0], box[1]
    y = y0 - 29
    if y < 3:
        return 0, 0
    cx = x0 + FACE_W // 2
    band = rgb[y - 3:y + 4, cx - 80:cx + 80].astype(np.int16)
    r, g, b = band[..., 0], band[..., 1], band[..., 2]
    grey = (b > 110) & (np.abs(r - g) < 16) & (r > 120)
    gold = (r > 150) & (r > g + 20) & (b < 60)
    col_grey = grey.mean(axis=0) > 0.5
    col_gold = gold.mean(axis=0) > 0.5
    levels = used = 0
    for mask, is_used in ((col_grey, False), (col_gold, True)):
        for c0, c1 in _runs(mask):
            if 7 <= c1 - c0 <= 16:
                levels += 1
                used += is_used
    return levels, used


# ---------------------------------------------------------------- face
def read_face(rgb: np.ndarray, box, allowed=None) -> dict:
    g = grid(rgb, box)
    tile, dist, conf = classify(g, allowed)
    dmg, dconf = read_digit(g)
    cd, charged = read_marks(g)
    return {"tile": tile, "dist": round(dist, 4), "conf": round(conf, 2), "damage": dmg,
            "damage_conf": round(dconf, 2), "cooldown": cd, "charged": charged, "ready": is_ready(g),
            "box": [int(v) for v in box], "cx": int(box[0] + FACE_W // 2), "cy": int(box[1] + FACE_H // 2)}
