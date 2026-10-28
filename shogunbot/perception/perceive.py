"""Screen -> structured state for Stage 1.

perceive(rgb) returns a Perception with:
  screen       'combat' | 'boss' | 'reward' | 'shrine' | 'banner' | 'map' |
               'title' | 'death' | 'unknown'
  cells        left-to-right list of {kind, cx, unit}
  hero         {x, facing, hp, queue[...], hand[{tile, ready, cx}]}
  enemies      [{x, kind, facing, hp, queue[...], fire_now, intent}]
  boss_frac    boss HP bar fill 0..1 (boss screen only)
  wave_alert   True when the WAVE text is red (next wave spawns soon)
  confidence   per-field 0..1, so the logger can flag weak reads

Coordinates assume a 1920x1080 client area (normalize() resizes otherwise).
Every threshold here was measured on recorded frames; see tests/test_perception.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import screens
from .board import cells as find_cells

ASSETS = Path(__file__).parent / "assets"
_spr = np.load(ASSETS / "sprites.npz")
SPR_X = _spr["X"].reshape(len(_spr["X"]), -1).astype(np.float32) / 255
SPR_XF = _spr["X"][:, :, ::-1].reshape(len(_spr["X"]), -1).astype(np.float32) / 255
SPR_K = _spr["K"]
SPR_F = _spr["F"]
_ico = np.load(ASSETS / "icons.npz")
ICO_X = _ico["X"].reshape(len(_ico["X"]), -1).astype(np.float32) / 255
ICO_K = _ico["K"]

SPRITE_Y0, SPRITE_Y1 = 610, 765
SLOT_Y0, SLOT_Y1, SLOT_PITCH = 458, 576, 120
HP_ROWS = (799, 819)
HAND_Y0, HAND_Y1 = 905, 1000
HAND_PITCH = 120


def normalize(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[:2] != (1080, 1920):
        rgb = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_AREA)
    return rgb


# ---------------------------------------------------------------- sprites
def classify_sprite(crop: np.ndarray) -> tuple[str, int, float]:
    """(kind, facing, confidence) by flip-aware nearest neighbour."""
    v = cv2.resize(crop, (32, 40), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    d_same = ((SPR_X - v) ** 2).sum(1)
    d_flip = ((SPR_XF - v) ** 2).sum(1)
    i, j = int(d_same.argmin()), int(d_flip.argmin())
    if d_same[i] <= d_flip[j]:
        kind, facing, best = SPR_K[i], int(SPR_F[i]), d_same[i]
    else:
        kind, facing, best = SPR_K[j], -int(SPR_F[j]), d_flip[j]
    # confidence: gap to the best match of a different kind / opposite facing
    other_kind = min(d_same[SPR_K != kind].min(), d_flip[SPR_K != kind].min())
    conf_kind = float(np.clip((other_kind - best) / max(best, 1e-3), 0, 1))
    same_k = SPR_K == kind
    if kind in ("other",):
        return str(kind), 0, conf_kind
    # facing margin within kind
    fs = np.where(same_k, np.where(SPR_F == facing, d_same, d_flip), np.inf).min()
    fo = np.where(same_k, np.where(SPR_F == facing, d_flip, d_same), np.inf).min()
    conf_face = float(np.clip((fo - fs) / max(fs, 1e-3), 0, 1))
    return str(kind), facing, min(conf_kind, conf_face * 4)


def classify_icon(interior: np.ndarray) -> tuple[str, float]:
    v = cv2.resize(interior, (25, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    d = ((ICO_X - v) ** 2).sum(1)
    i = int(d.argmin())
    k = ICO_K[i]
    other = d[ICO_K != k].min()
    # orange bell = summon (only one template, so add a colour rule)
    hsv = cv2.cvtColor(interior, cv2.COLOR_RGB2HSV)
    orange = ((hsv[..., 0] > 8) & (hsv[..., 0] < 25) & (hsv[..., 1] > 150) & (hsv[..., 2] > 180)).mean()
    if orange > 0.12:
        return "summon", 1.0
    return str(k), float(np.clip((other - d[i]) / max(d[i], 1e-3), 0, 1))


# ---------------------------------------------------------------- bubbles
def _frame_colour(px: np.ndarray) -> str | None:
    r, g, b = px[..., 0].astype(int), px[..., 1].astype(int), px[..., 2].astype(int)
    red = ((r > 130) & (g < 95) & (b < 75)).mean()
    yel = ((r > 180) & (g > 120) & (b < 95)).mean()
    grey = ((np.abs(r - b) < 30) & (g >= r - 5) & (g > 60) & (g < 150)).mean()
    best = max(red, yel, grey)
    if best < 0.55:
        return None
    return "red" if best == red else "yellow" if best == yel else "grey"


def read_stack(rgb: np.ndarray, cx: int) -> list[dict]:
    """Bubble slots above a unit, bottom first: {type:'tile'|'plus', tile, colour, plus_colour}."""
    out = []
    for k in range(4):
        y0, y1 = SLOT_Y0 - SLOT_PITCH * k, SLOT_Y1 - SLOT_PITCH * k
        if y0 < 120:
            break
        col = None
        for dl in (-61, -63, -59, -65):
            for dr in (60, 62, 58, 64):
                border = np.concatenate([rgb[y0 + 10:y1 - 10, cx + dl], rgb[y0 + 10:y1 - 10, cx + dr]])
                col = _frame_colour(border)
                if col:
                    break
            if col:
                break
        if col is None:
            break
        interior = rgb[y0 + 12:y1 - 12, cx - 50:cx + 50]
        dark = (interior.astype(int).sum(axis=2) < 150).mean()
        if dark > 0.35:
            tile, conf = classify_icon(interior)
            out.append({"type": "tile", "tile": tile, "colour": col, "conf": round(conf, 2)})
        else:
            c = rgb[(y0 + y1) // 2 - 4:(y0 + y1) // 2 + 4, cx - 4:cx + 4].reshape(-1, 3).mean(0)
            plus = "yellow" if (c[0] > 180 and c[2] < 120) else "white"
            out.append({"type": "plus", "colour": col, "plus": plus})
    return out


def plate_marker(rgb: np.ndarray, cx: int) -> str | None:
    """Yellow move arrow ('move-left'/'move-right') or turn icon ('turn') on a plate."""
    reg = rgb[715:800, cx - 90:cx + 90].astype(int)
    y = (reg[..., 0] > 200) & (reg[..., 1] > 140) & (reg[..., 1] < 215) & (reg[..., 2] < 90)
    n = int(y.sum())
    if n < 1000:
        return None
    xs = np.nonzero(y)[1]
    dx = xs.mean() - 90
    if dx > 30:
        return "move-right"
    if dx < -30:
        return "move-left"
    return "turn"


def swap_ticks(rgb: np.ndarray, cx: int) -> tuple[int, int]:
    """(white, red) tick counts on the hero plate; red ticks = swap recharging."""
    reg = rgb[760:784, cx - 35:cx + 35].astype(int)
    white = reg.min(axis=2) > 150
    red = (reg[..., 0] > 150) & (reg[..., 1] < 80)

    def runs(m):
        col, n, prev = m.any(axis=0), 0, False
        for v in col:
            if v and not prev:
                n += 1
            prev = v
        return n
    return runs(white), runs(red)


def hp_squares(rgb: np.ndarray, cx: int, row_y: int) -> tuple[int, int]:
    """(green, red) square counts in one HP row under a plate."""
    row = rgb[row_y, cx - 70:cx + 70].astype(int)
    g = (row[:, 1] > row[:, 0] + 30) & (row[:, 1] > 80)
    r = (row[:, 0] > row[:, 1] + 40) & (row[:, 0] > 90)

    def runs(m):
        n, prev = 0, False
        for v in m:
            if v and not prev:
                n += 1
            prev = v
        return n
    return runs(g), runs(r)


# ---------------------------------------------------------------- hand
def hero_patch_facing(rgb: np.ndarray, cx: int) -> tuple[int, float]:
    """Hero facing from the grey shoulder patch: large on the back, barely
    visible on the front [user]. Patch pixels right of centre = back on the
    right = facing left. Agrees with every confident sprite read on Bamboo
    Grove I (82/82); on Bamboo Grove II's brighter backdrop the sprite
    classifier failed and the patch was right in every frame checked.
    Returns (facing or 0, confidence)."""
    c = rgb[SPRITE_Y0 + 66:SPRITE_Y0 + 98, cx - 38:cx + 38].astype(int)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    m = (g > r + 6) & (b >= r - 6) & (b <= g + 2) & (g > 35) & (g < 120)
    xs = np.nonzero(m)[1]
    if len(xs) < 15:
        return 0, 0.0
    dx = float(xs.mean() - 38)
    if abs(dx) <= 8:
        return 0, 0.0
    return (-1 if dx > 0 else 1), (1.0 if len(xs) >= 30 else 0.7)


def read_consumables(rgb: np.ndarray) -> list[dict]:
    """Bottles in the consumable column (left edge): a cork (148,97,63) with
    liquid below. Green liquid (55,107,31) = 'green_potion', blue (64,121,150)
    = 'blue_potion' (shield [user]). Green card with a skull = 'skull_card'
    (poisons an enemy, damage over time [user]). Measured on bot run
    20261027-182237 frame f0104."""
    a = rgb[690:1010, 85:155].astype(int)
    cork = (np.abs(a[..., 0] - 148) < 14) & (np.abs(a[..., 1] - 97) < 14) & (np.abs(a[..., 2] - 63) < 14)
    rows = np.nonzero(cork.sum(axis=1) >= 6)[0]
    out, last = [], -100
    for r in rows:
        if r - last < 40:
            continue
        last = r
        liq = a[r + 10:r + 75]
        green = ((np.abs(liq[..., 0] - 55) < 14) & (np.abs(liq[..., 1] - 107) < 14) & (np.abs(liq[..., 2] - 31) < 14)).sum()
        blue = ((np.abs(liq[..., 0] - 64) < 14) & (np.abs(liq[..., 1] - 121) < 14) & (np.abs(liq[..., 2] - 150) < 14)).sum()
        kind = "green_potion" if green >= 80 else "blue_potion" if blue >= 80 else None
        if kind:
            out.append({"item": kind, "fx": 120 / 1920, "fy": (690 + r + 35) / 1080})
    # skull card: green card with a pale-green skull (144,170,56), no cork
    skull = (np.abs(a[..., 0] - 144) < 14) & (np.abs(a[..., 1] - 170) < 14) & (np.abs(a[..., 2] - 56) < 14)
    srows = np.nonzero(skull.sum(axis=1) >= 3)[0]
    if len(srows) >= 8:
        groups, start, prev = [], srows[0], srows[0]
        for r in list(srows[1:]) + [10 ** 6]:
            if r - prev > 12:
                groups.append((start, prev))
                start = r
            prev = r
        for g0, g1 in groups:
            if skull[g0:g1 + 1].sum() >= 40:
                out.append({"item": "skull_card", "fx": 120 / 1920, "fy": (690 + (g0 + g1) / 2) / 1080})
    out.sort(key=lambda it: it["fy"])
    return out


def windup_pixels(rgb: np.ndarray, cx: int) -> int:
    """Pixels of the raised-sword colour (252,245,178) around a unit.

    An Ashigaru about to strike holds its sword up in this exact colour
    (>= 480 px on every such frame; idle poses and backgrounds stay < 150).
    [run 20261027-180145: 9 strikes came from bubbles read as red, all in this
    pose, so the pose is a second 'fire' signal next to the bubble colour.]
    """
    c = rgb[SPRITE_Y0 - 30:SPRITE_Y1 - 15, cx - 75:cx + 75].astype(int)
    return int(((np.abs(c[..., 0] - 252) < 8) & (np.abs(c[..., 1] - 245) < 8)
                & (np.abs(c[..., 2] - 178) < 10)).sum())


WINDUP_MIN = 400


def merge_blink(p: "Perception", frames: list) -> list[str]:
    """Bubbles of enemies about to act blink yellow over a red base [user].
    One frame can land on the red phase, so re-read the bubbles of every
    enemy without a fire/queue telegraph on extra frames taken across the
    blink cycle and upgrade it when any frame shows yellow.
    Returns notes on what was upgraded (for the log)."""
    notes = []
    for e in p.enemies:
        if e["intent"] in ("fire", "move-left", "move-right", "turn"):
            continue
        for f in frames:
            f = normalize(f)
            stack = read_stack(f, e["cx"])
            tiles = [t for t in stack if t["type"] == "tile"]
            plus = [t for t in stack if t["type"] == "plus"]
            if e["queue"] and tiles and tiles[0]["colour"] == "yellow":
                e["intent"], e["fire_now"] = "fire", True
                notes.append(f"enemy at {e['x']}: yellow on a later frame -> fire")
                break
            if e["intent"] is None and plus and plus[0]["plus"] == "yellow":
                e["intent"] = "queue"
                notes.append(f"enemy at {e['x']}: yellow plus on a later frame -> queue")
                break
    return notes


def read_hand(rgb: np.ndarray) -> list[dict]:
    """Tiles in the hand panel, left to right: {tile, ready, cx, fx, fy}."""
    out = []
    for cx in range(780, 1141, 60):
        box = rgb[HAND_Y0:HAND_Y1, cx - 45:cx + 45]
        hsv = cv2.cvtColor(box, cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1].mean()
        lum = box.astype(int).sum(axis=2).mean() / 3
        if sat > 70 or lum > 120:   # wood panel or empty
            continue
        if out and cx - out[-1]["cx"] < 100:
            continue
        interior = rgb[HAND_Y0 + 2:HAND_Y1 - 2, cx - 48:cx + 48]
        tile, conf = classify_icon(interior)
        # ready tiles draw the damage digit in bright white; recharging tiles are dimmed
        digit = rgb[915:975, cx - 45:cx]
        white = int((digit.min(axis=2) > 170).sum())
        out.append({"tile": tile, "ready": white > 300, "cx": cx,
                    "fx": cx / 1920, "fy": 0.885, "conf": round(conf, 2)})
    return out


# ---------------------------------------------------------------- screen
def screen_type(rgb: np.ndarray, cells_: list[dict]) -> str:
    a = rgb.astype(int)
    menu = screens.classify(rgb)
    if menu is not None:
        return menu
    if len(cells_) in (5, 7) and any(c["kind"] == "hero" for c in cells_):
        # boss bar: long red bar near y=115
        bar = a[108:124, 640:1280]
        if ((bar[..., 0] > 120) & (bar[..., 1] < 60)).mean() > 0.05 or (bar.sum(axis=2) < 60).mean() > 0.6 and \
                ((a[96:100, 640:1280].sum(axis=2) > 500).mean() > 0.6):
            return "boss"
        return "combat"
    # death: red kanji in the middle
    mid = a[380:500, 860:1060]
    if ((mid[..., 0] > 180) & (mid[..., 1] < 50)).mean() > 0.05:
        return "death"
    return "unknown"


def boss_fraction(rgb: np.ndarray) -> float:
    bar = rgb[108:124, 642:1278].astype(int)
    red = (bar[..., 0] > 120) & (bar[..., 1] < 70)
    cols = red.mean(axis=0) > 0.5
    return float(cols.mean())


def wave_alert(rgb: np.ndarray) -> bool:
    txt = rgb[105:150, 820:1100].astype(int)
    red = ((txt[..., 0] > 170) & (txt[..., 1] < 90)).mean()
    return bool(red > 0.02)


# ---------------------------------------------------------------- main
@dataclass
class Perception:
    screen: str
    cells: list = field(default_factory=list)
    hero: dict | None = None
    enemies: list = field(default_factory=list)
    hand: list = field(default_factory=list)
    items: list = field(default_factory=list)
    boss_frac: float | None = None
    wave_alert: bool = False
    confidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("screen", "cells", "hero", "enemies", "hand", "items", "boss_frac", "wave_alert", "confidence")}


def perceive(rgb: np.ndarray) -> Perception:
    rgb = normalize(rgb)
    cs = find_cells(rgb)
    screen = screen_type(rgb, cs)
    p = Perception(screen=screen)
    if screen not in ("combat", "boss"):
        return p
    conf = {}
    hero_i = next(i for i, c in enumerate(cs) if c["kind"] == "hero")
    cell_list = []
    for i, c in enumerate(cs):
        cx = c["cx"]
        entry = {"x": i, "cx": cx, "kind": c["kind"]}
        if c["kind"] in ("enemy", "hero"):
            crop = rgb[SPRITE_Y0:SPRITE_Y1, cx - 64:cx + 64]
            kind, facing, kc = classify_sprite(crop)
            stack = read_stack(rgb, cx)
            marker = plate_marker(rgb, cx)
            if c["kind"] == "hero":
                g1, r1 = hp_squares(rgb, cx, HP_ROWS[0])
                g2, r2 = hp_squares(rgb, cx, HP_ROWS[1])
                wt, rt = swap_ticks(rgb, cx)
                pf, pc = hero_patch_facing(rgb, cx)
                if pf:
                    facing, kind, kc = pf, "hero", pc
                p.hero = {"x": i, "cx": cx, "facing": facing if kind == "hero" else 0,
                          "hp": g1 + g2, "hp_max": g1 + g2 + r1 + r2,
                          "queue": [s["tile"] for s in stack if s["type"] == "tile"],
                          "swap_red": rt}
                conf["hero_facing"] = round(kc, 2) if kind == "hero" else 0.0
                conf["hero_hp"] = 1.0 if (g1 + g2 + r1 + r2) in (10, 5) else 0.3
            else:
                g, r = hp_squares(rgb, cx, HP_ROWS[0])
                tiles = [s for s in stack if s["type"] == "tile"]
                plus = [s for s in stack if s["type"] == "plus"]
                intent = None
                if marker:
                    intent = marker
                elif plus and plus[0]["plus"] == "yellow":
                    intent = "queue"
                elif tiles and tiles[0]["colour"] == "yellow":
                    intent = "fire"
                # HP pips are a more reliable kind signal than the sprite (attack poses, hit flashes)
                by_hp = {3: "ashigaru", 2: "tachi", 1: "charger"}.get(g + r)
                by_tile = {"sword": "ashigaru", "swirl": "tachi", "charge": "charger"}.get(tiles[0]["tile"]) if tiles else None
                if screen == "boss" and (g + r) == 0:
                    kind = "daisuke"
                elif kind == "daisuke" and screen == "boss":
                    pass
                elif by_hp:
                    kind = by_hp
                elif by_tile:
                    kind = by_tile
                if kind == "ashigaru" and tiles and intent != "fire" and windup_pixels(rgb, cx) >= WINDUP_MIN:
                    intent = "fire"
                    tiles[0] = dict(tiles[0], colour="yellow")
                en = {"x": i, "cx": cx, "kind": kind, "facing": facing, "hp": g, "hp_max": g + r,
                      "queue": [t["tile"] for t in tiles],
                      "fire_now": bool(tiles and tiles[0]["colour"] == "yellow"),
                      "intent": intent}
                p.enemies.append(en)
                conf[f"enemy{i}"] = round(kc, 2)
            entry["unit"] = kind
        cell_list.append(entry)
    p.cells = cell_list
    p.hand = read_hand(rgb)
    p.items = read_consumables(rgb)
    if screen == "boss":
        p.boss_frac = boss_fraction(rgb)
    p.wave_alert = wave_alert(rgb)
    p.confidence = conf
    return p
