"""The map between regions (M4 step 1) [user 2026-10-29; Claude's test of keyboard-only mode on the
Day 1 map left by run 20261029-072323, 2026-10-29: JPEG captures in collect/20261029-kbd-map].

Keyboard-only mode: the top of the three round buttons at the bottom right turns it on. The buttons
are then replaced by three dim key hints (SELECT [enter], INFO [I], MAP [M]); the map marks one
reachable node with an enter key-cap and a tooltip that names it ("WHISPERING CAVES", "HIBIKU
WASTELANDS", "DAMAGE UPGRADE & COMBO SHOP", "SACRIFICE TILE & GUARDING SHOP"). The up and down
arrows move the mark among the reachable nodes, Enter travels there. W and S, and the arrows in
mouse mode, do nothing. Any mouse click turns keyboard mode off again; moving the mouse does not.
Esc opens the pause menu in keyboard mode, where the pause button is hidden.

The tooltip is a neutral dark box (about 39,38,36) inside a near-white border; its text is the
shrine font (text.py) at half scale, 2 px per art pixel, so the shrine glyph table reads it ('W'
and 'I' match exactly). The hero's "HP: 10/10" bubble is the same kind of box.

From the hero's node after Bamboo Grove both shop nodes were reachable, and each shop led to both
Whispering Caves and Hibiku Wastelands (the paths cross) [obs: that map].
"""

from __future__ import annotations

import cv2
import numpy as np

from ..catalogue import CAT
from . import text

KEYBOARD_BUTTON = (1820 / 1920, 843 / 1080)   # top of the three round buttons (mouse mode)
HINT_BOX = (1670, 1830, 885, 1030)            # x0, x1, y0, y1: the three key hints
HINT_PX = 4000                                # neutral dim-grey px there: 6,236 in keyboard mode, 0 in
                                              # mouse mode, at most 2,463 on 600 recorded frames
MAP_AREA = (230, 1700, 160, 840)              # x0, x1, y0, y1: the parchment
TIP_MIN_PX = 5000                             # dark box px: tooltips 14,068-25,313, the HP bubble 8,580
TIP_FILL = 0.6                                # box fill: 0.83-0.85 (map art 0.24)
ART = 2                                       # px per art pixel in the tooltip font


def keyboard_mode(rgb: np.ndarray) -> bool:
    """The key hints of keyboard-only mode are on screen (normalized 1080p frame)."""
    x0, x1, y0, y1 = HINT_BOX
    reg = rgb[y0:y1, x0:x1].astype(np.int16)
    mx, mn = reg.max(axis=2), reg.min(axis=2)
    return int(((mx - mn < 12) & (mx >= 40) & (mx <= 80)).sum()) >= HINT_PX


def _boxes(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) of the dark tooltip boxes on the map. OpenCV labels the dark pixels
    (4-connected): 0.25-0.26 used scipy, which the bot's venv on the PC does not have, so the
    map step crashed after the boss [runs 20261029-120329 and -122345]."""
    x0, x1, y0, y1 = MAP_AREA
    reg = rgb[y0:y1, x0:x1].astype(np.int16)
    mx, mn = reg.max(axis=2), reg.min(axis=2)
    dark = ((mx - mn < 12) & (mx > 25) & (mx < 70)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    out = []
    for i in range(1, n):
        bx, by, w, h, px = (int(v) for v in stats[i])
        if px >= TIP_MIN_PX and px / (h * w) >= TIP_FILL:
            out.append((x0 + bx, y0 + by, x0 + bx + w, y0 + by + h))
    return sorted(out, key=lambda b: (b[1], b[0]))


def _lines(rgb: np.ndarray, box) -> list[str]:
    """The text lines in a tooltip box, read with the shrine glyphs at half scale."""
    x0, y0, x1, y1 = box
    t = rgb[y0:y1, x0:x1].astype(np.int16)
    ink = t.min(axis=2) > 170
    rows = ink.mean(axis=1) > 0.004
    out = []
    for r0, r1 in text._runs(rows):
        if r0 <= 2 or r1 >= ink.shape[0] - 2 or not 16 <= r1 - r0 <= 24:
            continue                                  # the border, or not a text line
        cols = np.nonzero(ink[r0:r1].any(axis=0))[0]
        cols = cols[(cols > 4) & (cols < ink.shape[1] - 4)]
        if not len(cols):
            continue
        c0 = int(cols.min())
        w = (int(cols.max()) + 1 - c0 + ART - 1) // ART * ART
        blk = ink[r0:r0 + 10 * ART, c0:c0 + w]
        if blk.shape != (10 * ART, w):
            continue
        art = blk.reshape(10, ART, w // ART, ART).mean(axis=(1, 3)) > 0.5
        out.append("".join(" " if g is None else text._match(g) for g in text.glyphs(art)))
    return out


def tooltips(rgb: np.ndarray) -> list[dict]:
    """Every dark box on the map with its text: [{'box', 'text'}]."""
    return [{"box": b, "text": " ".join(_lines(rgb, b)).strip()} for b in _boxes(rgb)]


def _dist(a: str, b: str) -> int:
    return text._dist(a, b)


def node_kind(name: str) -> tuple[str | None, str | None]:
    """('region', region id) or ('shop', None) for a tooltip's text, (None, None) when unsure."""
    t = " ".join(name.upper().split())
    known = sum(c.isalpha() for c in t)
    if known < 4 or known < 0.6 * len(t.replace(" ", "")) or t.startswith("HP"):
        return None, None                 # mostly unread glyphs (the hero's HP bubble: '??? ?????')
    if "SHOP" in t or _dist(t.split()[-1], "SHOP") <= 1:
        return "shop", None
    scored = sorted((_dist(t, r["name"].upper()), rid) for rid, r in CAT.regions.items() if rid != "camp")
    if scored and scored[0][0] <= max(2, len(t) // 5) and (len(scored) < 2 or scored[1][0] > scored[0][0]):
        return "region", scored[0][1]
    return None, None


def selected(rgb: np.ndarray) -> dict | None:
    """The marked node in keyboard mode: {'box', 'text', 'kind', 'region'}, or None when no
    tooltip other than the hero's HP bubble is read."""
    for tip in tooltips(rgb):
        kind, region = node_kind(tip["text"])
        if kind is not None:
            return {**tip, "kind": kind, "region": region}
    return None
