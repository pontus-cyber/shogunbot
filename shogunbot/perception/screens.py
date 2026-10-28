"""Non-combat screens: title, camp, map, reward/shrine.

Measured on the user's recorded run (collect/20261027-164727, 1920x1080):
  title   red key art, menu text; NEW RUN clicked at (0.5062, 0.8065)
  camp    red scene + hand panel; walk right twice, then GO pill
          (lit peach ~ (173,112,76)) at about (0.6849, 0.6269)
  map     tan parchment ~ (150,112,80) over >10% of the middle;
          first node (Bamboo Grove) clicked at (0.5047, 0.4704)
  reward  tile pick and shrine both carry a grey SKIP pill, ~288 px wide,
          rows ~720-790, x varies (1300-1672) with the layout
"""

from __future__ import annotations

import numpy as np

TITLE_NEW_RUN = (0.5062, 0.8065)
CAMP_GO = (0.6849, 0.6269)
MAP_FIRST_NODE = (0.5047, 0.4704)
MAP_LABEL = (0.5730, 0.4540)        # the 'BAMBOO GROVE' bubble next to the node
# inputs tried in turn on the map until the level loads
MAP_TRIES = [("click node", MAP_FIRST_NODE), ("click node", MAP_FIRST_NODE),
             ("key space", "space"), ("key d", "d"), ("click label", MAP_LABEL),
             ("key enter", "enter")]


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    best, s, n = (0, 0), 0, 0
    for i, v in enumerate(mask):
        if v:
            if n == 0:
                s = i
            n += 1
            if n > best[1] - best[0]:
                best = (s, i + 1)
        else:
            n = 0
    return best


def find_skip(rgb: np.ndarray) -> tuple[float, float] | None:
    """Centre of the grey SKIP pill as window fractions, or None."""
    a = rgb.astype(int)
    h, w = a.shape[:2]
    for y0 in range(690, 820, 6):
        band = a[y0:y0 + 20, 1150:1850]
        g = ((np.abs(band[..., 0] - band[..., 1]) < 14) & (np.abs(band[..., 1] - band[..., 2]) < 14)
             & (band[..., 0] > 55) & (band[..., 0] < 130))
        x0, x1 = _longest_run(g.mean(axis=0) > 0.5)
        if 240 <= x1 - x0 <= 340:
            # vertical extent of the pill at its centre column
            cx = 1150 + (x0 + x1) // 2
            col = a[650:860, cx - 60]
            gc = ((np.abs(col[:, 0] - col[:, 2]) < 14) & (col[:, 0] > 55) & (col[:, 0] < 130))
            ys = np.where(gc)[0]
            cy = 650 + int(np.median(ys)) if len(ys) else y0 + 10
            return (cx / w, cy / h)
    return None


def find_go(rgb: np.ndarray) -> tuple[float, float] | None:
    """Centre of the lit GO pill in camp, or None when it is not lit yet."""
    a = rgb[560:780, 1000:1800].astype(int)
    m = (np.abs(a[..., 0] - 173) < 16) & (np.abs(a[..., 1] - 113) < 16) & (np.abs(a[..., 2] - 78) < 16)
    if m.sum() < 3000:
        return None
    ys, xs = np.where(m)
    return ((1000 + xs.mean()) / rgb.shape[1], (560 + ys.mean()) / rgb.shape[0])


def features(rgb: np.ndarray) -> dict:
    a = rgb.astype(int)
    s = a[200:800, 150:1800]
    tan = ((np.abs(s[..., 0] - 150) < 20) & (np.abs(s[..., 1] - 112) < 20) & (np.abs(s[..., 2] - 80) < 20)).mean()
    s2 = a[200:700, 100:1800]
    red = (s2[..., 0] > s2[..., 1] * 1.7).mean()
    hand = a[900:1010, 780:1140]
    wood = ((hand[..., 0] > 150) & (hand[..., 1] > 80) & (hand[..., 1] < 150) & (hand[..., 2] < 100)).mean()
    return {"tan": float(tan), "red": float(red), "wood": float(wood), "mean": float(a.mean())}


def classify(rgb: np.ndarray) -> str | None:
    """'map' | 'reward' | 'title' | 'camp' | 'black' | None (not a menu screen)."""
    f = features(rgb)
    if f["mean"] < 3:
        return "black"
    if f["tan"] > 0.1:
        return "map"
    if find_skip(rgb) is not None:
        return "reward"
    if f["red"] > 0.5:
        return "camp" if f["wood"] > 0.08 else "title"
    return None
