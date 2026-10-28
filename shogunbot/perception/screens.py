"""Non-combat screens: title, camp, map, reward/shrine, pause menu, death.

Measured on the user's recorded run (collect/20261027-164727, 1920x1080):
  title   red key art, menu text; NEW RUN clicked at (0.5062, 0.8065)
  camp    red scene + hand panel; walk right twice, then GO pill
          (lit peach ~ (173,112,76)) at about (0.6849, 0.6269)
  map     tan parchment ~ (150,112,80) over >10% of the middle;
          first node (Bamboo Grove) clicked at (0.5047, 0.4704)
  reward  tile pick and shrine both carry a grey SKIP pill, ~288 px wide,
          rows ~720-790, x varies (1300-1672) with the layout

Giving up a run [user 2026-10-28; measured on captures in collect/20261028-121900-giveup]:
  pause   PAUSE MENU is the bottom of the three round HUD buttons (bottom right, disc
          x 1784-1855, y 972-1035, pause bars x 1804-1815 and 1824-1835). The map dims
          the HUD to ~0.6 brightness (runs/20261028-104830 map_1).
  menu    a near-white banner (229,226,219) across the full width at y 256-343 over a
          darkened screen, then rows of light text (~(173,172,163)) 40 px high, 80 px
          apart from y 472, centred at x 957. Pause menu: banner 'PAUSE' and five rows
          (RESUME, OPTIONS, GIVE UP, MAIN MENU, SAVE AND QUIT). Confirmation: banner
          'ARE YOU SURE YOU WANT TO GIVE UP?' and two rows (NO: GO BACK, YES: GIVE UP).
  death   red kanji in the middle; RESTART and BACK TO CAMP pills (118,68,58), x 1472-1808,
          y 878-942 and 972-1038 (also runs/20261027-182223 death.png, a real death).
          RESTART starts a new run straight in Bamboo Grove I.
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
PAUSE_BUTTON = (0.9477, 0.9269)     # centre of the PAUSE MENU disc (1820, 1001)
MENU_X = 957 / 1920                 # menu rows are centred here


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


def _row_runs(mask: np.ndarray, min_len: int, max_len: int = 10_000) -> list[tuple[int, int]]:
    """(start, end) of every run of True in a 1-D mask whose length is in [min_len, max_len]."""
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if min_len <= i - start <= max_len:
                out.append((start, i))
            start = None
    return out


def pause_button(rgb: np.ndarray) -> bool:
    """The PAUSE MENU disc is on screen: a warm brown disc with two darker bars, at
    full brightness (combat, rewards) or dimmed (map). Bars/fill brightness is 0.63 in
    both (149/236 combat, 90/141 map)."""
    a = rgb.astype(int)
    fill = np.concatenate([a[990:1016, 1788:1800].reshape(-1, 3), a[990:1016, 1817:1822].reshape(-1, 3),
                           a[990:1016, 1840:1852].reshape(-1, 3)])
    bars = np.concatenate([a[992:1016, 1806:1814].reshape(-1, 3), a[992:1016, 1826:1834].reshape(-1, 3)])
    out = np.concatenate([a[990:1016, 1764:1778].reshape(-1, 3), a[990:1016, 1862:1876].reshape(-1, 3)])
    if fill.std(axis=0).max() > 12 or bars.std(axis=0).max() > 12:
        return False
    f, b = fill.mean(axis=0), bars.mean(axis=0)
    if not (f[0] > 50 and f[0] > 1.5 * f[1]):
        return False
    return 0.5 < b.sum() / f.sum() < 0.75 and out.sum(axis=1).mean() < 60


def menu(rgb: np.ndarray) -> tuple[str, list[tuple[float, float]]] | None:
    """('pause' | 'confirm', centres of the text rows as window fractions) when the
    pause menu or the give-up confirmation is open, else None. Any other banner or
    row count is None, so the bot never clicks a row it has not recognised."""
    a = rgb.astype(int)
    h, w = rgb.shape[:2]
    band = a[262:338]
    banner = ((np.abs(band[..., 0] - 229) < 16) & (np.abs(band[..., 1] - 226) < 16)
              & (np.abs(band[..., 2] - 219) < 16))
    if banner.mean() < 0.75:              # 0.98 under 'PAUSE', 0.89 under the long question
        return None
    dark = np.where((band.sum(axis=2) < 200).any(axis=0))[0]
    if len(dark) == 0:
        return None
    title_w = int(dark.max() - dark.min())    # 139 'PAUSE', 911 'ARE YOU SURE ...'
    reg = a[400:880, 560:1360]
    light = (reg.min(axis=2) > 150) & (reg.max(axis=2) - reg.min(axis=2) < 40)
    rows = []
    for y0, y1 in _row_runs(light.sum(axis=1) > 15, 25, 60):
        xs = np.where(light[y0:y1].any(axis=0))[0]
        rows.append(((560 + (xs.min() + xs.max()) / 2) / w, (400 + (y0 + y1) / 2) / h))
    gaps = [(b[1] - a_[1]) * h for a_, b in zip(rows, rows[1:])]
    if not all(70 <= g <= 90 for g in gaps) or not all(abs(r[0] * w - 957) < 30 for r in rows):
        return None
    if len(rows) == 5 and title_w < 400:
        return "pause", rows
    if len(rows) == 2 and title_w > 600:
        return "confirm", rows
    return None


def find_restart(rgb: np.ndarray) -> tuple[float, float] | None:
    """Centre of the RESTART pill on the death screen (the upper of the two pills at the
    bottom right, with the red death kanji in the middle), else None."""
    a = rgb.astype(int)
    h, w = rgb.shape[:2]
    mid = a[380:500, 860:1060]
    if ((mid[..., 0] > 180) & (mid[..., 1] < 50)).mean() <= 0.05:
        return None
    r = a[850:1060, 1400:1900]
    pill = (np.abs(r[..., 0] - 118) < 12) & (np.abs(r[..., 1] - 68) < 12) & (np.abs(r[..., 2] - 58) < 12)
    runs = _row_runs(pill.sum(axis=1) > 120, 30, 90)
    if len(runs) != 2:
        return None
    y0, y1 = runs[0]
    xs = np.where(pill[y0:y1].sum(axis=0) > (y1 - y0) // 3)[0]
    if len(xs) == 0 or not 280 <= xs.max() - xs.min() <= 380:
        return None
    return ((1400 + (xs.min() + xs.max()) / 2) / w, (850 + (y0 + y1) / 2) / h)


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
