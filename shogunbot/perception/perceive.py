"""Screen -> structured state for Stage 1.

perceive(rgb) returns a Perception with:
  screen       'combat' | 'boss' | 'reward' | 'shrine' | 'banner' | 'map' |
               'title' | 'death' | 'unknown'
  cells        left-to-right list of {kind, cx, unit}
  hero         {x, facing, hp, queue[...], hand[{tile, ready, cx}]}
  enemies      [{x, kind, facing, hp, queue[...], fire_now, intent, spawning, elite, alt}]
  boss_frac    boss HP bar fill 0..1 (boss screen only)
  wave         (wave, waves in the level) from the HUD counter "WAVE x/y", or None
  wave_alert   True when the WAVE text is salmon: the turn a wave pops in (56 of 70 salmon
               frames in the recordings), sometimes the turns before the next one
  confidence   per-field 0..1, so the logger can flag weak reads

Coordinates assume a 1920x1080 client area (normalize() resizes otherwise).
Every threshold here was measured on recorded frames; see tests/test_perception.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..catalogue import CAT
from ..model import DEFAULT_KIT
from . import hud, screens
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
# A unit appears in a red spawn beam, then flames, then its sprite. Red pixels (R >= 150, G <= 100,
# B <= 70, R - G >= 90) in the sprite box: flames 6,000-15,000, the beam 1,000-6,000, settled units
# 0-1,240. Under flames the sprite gives no facing (78 of 78 new Ashigaru/Tachi at 6,000+), so the
# bridge assumed the unit faced the hero, and about half face away. Below 2,000 the facing still
# reads (28 of 28) [scan of 3,804 units on every recorded frame, 0.20.1; user review of the 0.20
# series: enemy facing was the largest mismatch].
SPAWN_RED_PX = 2000
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


HERO_KIT = tuple(t.tile for t in DEFAULT_KIT)  # tiles the hero starts with (catalogue); the bot passes its live kit
# hand faces are matched among the kit and the reward pool, so a newly taken tile reads as itself
BASE_POOL = frozenset(t.id for t in CAT.base_pool())
# Enemies told apart by HP pips and by the tile in their bubble (catalogue), among the standard
# enemies of the regions the run has reached: a region's waves also bring the earlier regions'
# enemies [rec: collect/20261028-190026, Whispering Caves I and II had Ashigaru, Twin Tachi and
# Spike Chargers besides the Yari Master]. The bot passes the regions of its route (M4).
REGION = "bamboo_grove"
ROSTER = (REGION,)
KIND_BY_HP = CAT.kinds_by_hp(REGION)        # {3: ashigaru, 2: tachi, 1: charger}
KIND_BY_TILE = CAT.kinds_by_tile(REGION)    # {sword: ashigaru, swirl: tachi, charge: charger}
BASE_HP = {k: hp for hp, k in KIND_BY_HP.items()}


@lru_cache(maxsize=16)
def roster_tables(regions: tuple[str, ...] = ROSTER) -> tuple[dict, dict]:
    """(BASE_HP, KIND_BY_TILE) for the standard enemies of these regions, in route order. Kinds
    that share an HP (Hibiku Wastelands' Yumi Sniper and the Ashigaru, 3) are told apart by the
    tile in their bubble; without one the earlier region's kind is taken."""
    base, by_tile = {}, {}
    for r in regions:
        for e in CAT.region_enemies(r):
            base.setdefault(e.id, e.hp)
            if e.tile:
                by_tile.setdefault(e.tile, e.id)
    return base, by_tile
# Elite variants (Day 2 on) have extra HP [wiki: Quick, Heavy, Reactive Shield +1, Double-Striker +2]
ELITE_BONUS = tuple(sorted({t["hp_bonus"] for t in CAT.elite_traits.values() if t.get("hp_bonus")}))
# ... and an emblem at the top centre of their plate. Emblem-coloured pixels (brown-gold: R 150-230,
# G 85-150, B < 45, R - G 40-80) in the box: elites 80 (Double-Striker crown) and 214-256 (Quick);
# Day-1 Stage 1 plates 0-40 (4,801 reads). The yellow 1ST/2ND order markers are brighter (G > 150)
# and do not count [run 20261029-045856, the first Day 2 run; every recorded frame].
ELITE_Y0, ELITE_Y1, ELITE_HALF = 744, 784, 14
ELITE_PX = 60
SPRITE_KIND_CONF = 0.8   # the sprite's kind counts as evidence from this confidence on


def classify_icon(interior: np.ndarray, allowed: tuple | None = None) -> tuple[str, float]:
    v = cv2.resize(interior, (25, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    d = ((ICO_X - v) ** 2).sum(1)
    if allowed:
        # restrict to the tiles that can be there [run 20261028-061420: an Arrow in the
        # hand was read as Spear (conf 0.01) and the bot could not find its Arrow]
        d = np.where(np.isin(ICO_K, allowed), d, np.inf)
    i = int(d.argmin())
    k = ICO_K[i]
    rest = d[(ICO_K != k) & np.isfinite(d)]
    other = rest.min() if len(rest) else d[i] * 2
    if allowed:
        return str(k), float(np.clip((other - d[i]) / max(d[i], 1e-3), 0, 1))
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


# A bubble a few px off its unit's centre was read as no bubble: Rei's Dragon Punch at t71 of run
# 20261029-124921, 6 px right of her cell after her step (her punch then pushed the hero
# unpredicted), and the Yari Master's Spear at t95, next to the hero (it hit him and the Twin Tachi
# behind him). The centred check comes first, so a centred bubble reads as before; off centre only
# an enemy's frame counts (red or yellow on both sides, each side on its own, with the bubble's dark
# outline outside it): a shifted grey or one-sided match found bubbles in the bamboo and the red
# backdrops of later regions (hero bubbles on 37 frames, all false).
STACK_SHIFTS = (0, 4, -4, 8, -8, 12, -12)


def _frame_at(rgb: np.ndarray, y0: int, y1: int, cx: int) -> str | None:
    for dl in (-61, -63, -59, -65):
        for dr in (60, 62, 58, 64):
            border = np.concatenate([rgb[y0 + 10:y1 - 10, cx + dl], rgb[y0 + 10:y1 - 10, cx + dr]])
            col = _frame_colour(border)
            if col:
                return col
    return None


def _frame_off_centre(rgb: np.ndarray, y0: int, y1: int, cx: int) -> str | None:
    """An enemy's bubble centred within 2 px of cx: from its centre, a 4 px dark outline at 65-68
    px out, the coloured frame from 53 to 64 px [run 20261029-124921 t70-t71, Rei's bubble]."""
    rows = rgb[y0 + 10:y1 - 10]
    cols = [_frame_colour(rows[:, cx + d]) for d in (-59, -58, 58, 59)]
    if cols[0] not in ("red", "yellow") or any(c != cols[0] for c in cols):
        return None

    def dark(x: int) -> float:
        return float((rows[:, x].max(axis=1) < 60).mean())

    if max(dark(cx + d) for d in range(-70, -63)) < 0.8 or max(dark(cx + d) for d in range(64, 71)) < 0.8:
        return None
    return cols[0]


def read_stack(rgb: np.ndarray, cx: int) -> list[dict]:
    """Bubble slots above a unit, bottom first: {type:'tile'|'plus', tile, colour, plus_colour}."""
    out = []
    for k in range(4):
        y0, y1 = SLOT_Y0 - SLOT_PITCH * k, SLOT_Y1 - SLOT_PITCH * k
        if y0 < 120:
            break
        col = None
        for sh in STACK_SHIFTS:
            col = _frame_at(rgb, y0, y1, cx) if sh == 0 else _frame_off_centre(rgb, y0, y1, cx + sh)
            if col:
                cx += sh
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


def elite_pixels(rgb: np.ndarray, cx: int) -> int:
    """Emblem-coloured pixels at the top centre of a unit's plate (ELITE_PX and up: an elite)."""
    box = rgb[ELITE_Y0:ELITE_Y1, cx - ELITE_HALF:cx + ELITE_HALF + 1].astype(np.int16)
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    return int(((r >= 150) & (r <= 230) & (g >= 85) & (g <= 150) & (b < 45) & (r - g >= 40) & (r - g <= 80)).sum())


def decide_kind(pips: int, tile_kind: str | None, sprite_kind: str | None, elite: bool,
                base_hp: dict | None = None, elites: bool = True) -> tuple[str | None, str | None]:
    """(kind, alt) of a Stage 1 unit from its HP pips, the tile in its bubble, its sprite and
    the elite emblem. alt is a kind the read could not rule out (None when the evidence agrees).

    Day 1: the pips name the kind (the sprite misreads attack poses: 82 of 4,106 reads), and
    the tile icon never contradicted them (0 of 4,106). Day 2 brings elites with 1-2 extra HP,
    so 3 pips can be an Ashigaru or an elite Tachi: run 20261029-045856 read an elite Tachi
    loaded with Swirl as an Ashigaru (turns 16 and 21), swapped with it for the swap trick and
    took 2 damage twice [user review]. Now the tile icon decides whenever its kind fits the pips
    with or without an elite bonus, an elite (emblem, or pips no base kind has) is named by its
    sprite, and a sprite that names another kind the pips allow is kept as alt.

    base_hp: the kinds of the run's regions (roster_tables). elites False (Day 1, chosen at the
    camp): no unit has extra pips, so the pips alone name the kind [M4: in Whispering Caves the
    Yari Master's spear icon read as a Sword on 18 of 30 reads, which named the 4-pip unit an elite
    Ashigaru; collect/20261028-190026 f0186-f0280]."""
    allowed = (0,) + (ELITE_BONUS if elites else ())
    bonus = {k: pips - hp for k, hp in (base_hp or BASE_HP).items() if pips - hp in allowed}
    if not bonus:
        return None, None
    if tile_kind in bonus:
        return tile_kind, None
    # a tile no kind with these pips carries: kept as alt while elites can change the pips; on
    # Day 1 it is an icon misread [M4: the 4-pip Yari Master's spear read as a Sword named an
    # Ashigaru alt on 18 of 30 Caves reads, collect/20261028-190026 f0191-f0273]
    alt = tile_kind if elites and tile_kind in (base_hp or BASE_HP) else None
    if elite or 0 not in bonus.values():
        cands = [k for k, b in bonus.items() if b > 0] or list(bonus)
        if sprite_kind in cands:
            return sprite_kind, alt
        kind = min(cands, key=lambda k: bonus[k])           # +1 is the bonus of three of the four elite traits
        return kind, alt or next((c for c in cands if c != kind), None)
    base = next(k for k, b in bonus.items() if b == 0)
    if alt is None and sprite_kind in bonus and sprite_kind != base:
        alt = sprite_kind
    return base, alt


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


DIM_FACTOR = 0.475          # items' brightness while dimmed [runs 20261028-132238, -133837, -135635: 0.469-0.489]
DIM_P99 = (70, 115)         # 99th-percentile brightness of a dimmed item's window
DIM_FILLED = 0.28           # share of the window above 28 on a dimmed item (0.29-0.78); empty slots 0.13-0.27
ITEM_SLOTS_Y = (758, 857, 952)   # consumable column, measured on runs 20261027-182237 / 20261028-053637


def _near(a: np.ndarray, rgb: tuple, tol: int = 14) -> np.ndarray:
    return (np.abs(a[..., 0] - rgb[0]) < tol) & (np.abs(a[..., 1] - rgb[1]) < tol) & (np.abs(a[..., 2] - rgb[2]) < tol)


def _red_pips(a: np.ndarray, lit: np.ndarray) -> int:
    """Compact red blobs (9-120 px): the pips of the Lucky Die."""
    r_, g_, b_ = a[..., 0], a[..., 1], a[..., 2]
    red = ((r_ > g_ + 40) & (r_ > b_ + 40) & lit).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(red, connectivity=4)
    return int(sum(1 for i in range(1, n) if 9 <= st[i, cv2.CC_STAT_AREA] <= 120))


def read_consumables(rgb: np.ndarray) -> list[dict]:
    """Items in the consumable column (left edge), one per slot.

    Labels are the catalogue's screen ids (data/consumables.json maps each to
    its item and effect):
      green_potion  Edamame Brew   (cork + green liquid)
      blue_potion   Kami Brew      (cork + blue liquid)
      white_potion  Cool Up        (cork + pale liquid) [run 20261028-061420 item_unknown.png]
      skull_card    Mass Poison    (green card, pale skull)
      ice_card      Mass Ice       (navy card, cyan crystal)
      curse_card    Mass Curse     (red card, mask) [user] [collect 20261027-164727 f0083]
      dice_card     Lucky Die      (grey-white card, six red pips in 2 x 3) [user 2026-10-28]
                    [run 20261028-122652: 64-px pips in combat, 16-px pips under banners]
      mirror_card   Rain of Mirrors (grey card with diagonal light streaks) [user] [runs
                    20261028-155154, -155955, -160740: read, used and consumed; the grey card
                    first logged under this name was the Lucky Die]
    Anything else in a filled slot is returned as 'unknown' so the bot can log it.

    The game dims an item it will not let you use now to 47.5 % brightness: every
    item while a level ends, the combat cards but not Edamame Brew on a reward screen
    [run 20261027-182237 f0026]; empty slots never dim. 0.19 read dimmed items as
    'unknown' (Mass Ice, Cool Up) or as an empty slot (Edamame Brew) [runs
    20261028-132238, -133837, -135635 item_unknown.png]. A dimmed slot is brightened
    back before the rules run, and its read carries dim=True: the bot never uses it.
    """
    out = []
    for cy in ITEM_SLOTS_Y:
        a = rgb[cy - 45:cy + 45, 85:155].astype(int)
        mx = a.max(axis=2)
        p99 = float(np.percentile(mx, 99))
        if DIM_P99[0] <= p99 <= DIM_P99[1] and (mx > 28).mean() >= DIM_FILLED:
            # dimmed item: lit items peak at 150-255, dimmed ones at 72-107; an empty slot's
            # ring peaks at 64, a brown-tinted one at 73 with 23 % of the window lit
            # [items_old scan of all recorded frames, 0.20]
            bright = np.minimum(255, np.round(a / DIM_FACTOR)).astype(int)
            item = _classify_item(bright)
            if item not in ("unknown", "mirror_card"):
                out.append({"item": item, "fx": 120 / 1920, "fy": cy / 1080, "dim": True})
                continue
        if (mx > 60).mean() < 0.2:
            continue  # empty slot (dark ring)
        out.append({"item": _classify_item(a), "fx": 120 / 1920, "fy": cy / 1080})
    return out


def _classify_item(a: np.ndarray) -> str:
    """The item in one slot window (90 x 70 RGB, int) by the hue rules below."""
    # hue rules, so items dimmed during animations still read [run 20261027-164727]
    r_, g_, b_ = a[..., 0], a[..., 1], a[..., 2]
    lit = a.max(axis=2) > 25
    cork = int(((r_ > g_ + 25) & (g_ > b_ + 10) & (r_ > 60) & (r_ < 175) & lit)[:30].sum())
    green_px = int(((g_ > r_ + 25) & (g_ > b_ + 30) & lit).sum())
    blue_px = int(((b_ > r_ + 40) & (b_ >= g_ - 5) & lit).sum())
    red_px = int(((r_ > g_ + 40) & (r_ > b_ + 40) & lit).sum())
    skull = int(_near(a, (144, 170, 56), 30).sum())
    navy = float(((b_ > r_ + 25) & (b_ > g_ + 15) & (b_ < 120) & lit).mean())
    pips = _red_pips(a, lit)
    grey_px = int((((a.max(axis=2) - a.min(axis=2)) < 22) & lit).sum())
    if cork >= 6 and green_px >= 150:
        item = "green_potion"
    elif cork >= 6 and blue_px >= 150:
        item = "blue_potion"
    elif skull >= 40:
        item = "skull_card"
    elif navy >= 0.15:
        item = "ice_card"
    elif 5 <= pips <= 8 and grey_px >= 1500:
        # Lucky Die: 6 pips and 2,560-2,645 grey px [run 20261028-122652]; Mass Curse has
        # 14 red blobs on a dark red card [collect 20261027-164727 f0083]. Before 0.18.2 the
        # die read as curse_card in combat and as mirror_card under banners
        item = "dice_card"
    elif red_px >= 300:
        item = "curse_card"   # Mass Curse: red card with a mask [user]
    elif cork >= 6 and int(((a.min(axis=2) > 150) & (a.max(axis=2) - a.min(axis=2) < 30)).sum()) >= 300:
        # pale liquid: 848 px on run 20261028-061420 item_unknown.png; 0 on the other bottles.
        # Dimmed (reward screens) it reads after read_consumables brightens the slot.
        item = "white_potion"
    elif (lit.mean() > 0.35 and a.mean() < 150 and float(((a.max(axis=2) - a.min(axis=2)) < 22)[lit].mean()) > 0.7
          and cork < 6):
        item = "mirror_card"  # Rain of Mirrors: grey card [user] [runs 20261028-155154 .. -160740]
    else:
        item = "unknown"
    return item


def spawn_pixels(rgb: np.ndarray, cx: int) -> int:
    """Red spawn-effect pixels in a unit's sprite box (see SPAWN_RED_PX)."""
    a = rgb[SPRITE_Y0:SPRITE_Y1, max(0, cx - 64):cx + 64].astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return int(((r >= 150) & (g <= 100) & (b <= 70) & (r - g >= 90)).sum())


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
            if e["intent"] is None and plus:
                e["intent"] = "queue"
                notes.append(f"enemy at {e['x']}: a '+' on a later frame -> queue")
                break
    return notes


def read_hand(rgb: np.ndarray, kit_tiles=None) -> list[dict]:
    """Tiles in the hand panel, left to right, read from their faces
    (perception/tiles.py): {tile, ready, damage, cooldown, charged, cx, fx, fy, conf}.
    `charged` counts the grey cooldown marks, so it is the tile's current
    charge; `kit_tiles` restricts the icon match to the hero's own tiles."""
    from .reward_screen import read_hand as read_faces
    return read_faces(rgb, tuple(set(kit_tiles or HERO_KIT) | BASE_POOL))


# ---------------------------------------------------------------- screen
BOARD_CELLS = (5, 7, 9)   # level widths [wiki]; 9 cells first seen in Hot Springs and the Castle
                          # [collect/20261028-190026]


def screen_type(rgb: np.ndarray, cells_: list[dict]) -> str:
    """Reward screens first (their SKIP pill), then the fight HUD (screens.combat_hud), then
    the menus. 0.20.1 tested the menus first, and in the full-game recording only 315 of
    about 1,450 fight frames read as a fight: red backdrops read as camp or title, the cave
    as map [user review of collect/20261028-190026]. A frame with the fight HUD but no
    readable board is 'unknown': the bot waits instead of acting on a menu rule."""
    a = rgb.astype(int)
    board = len(cells_) in BOARD_CELLS and any(c["kind"] == "hero" for c in cells_)
    if a.mean() >= 3 and screens.find_skip(rgb) is None and screens.shop(rgb):
        return "shop"          # its floor reads as a board, a banner as the fight HUD [run 20261029-072042]
    if a.mean() >= 3 and screens.find_skip(rgb) is None and screens.combat_hud(rgb):
        if not board:
            return "unknown"
        return "boss" if screens.boss_bar(rgb) else "combat"
    menu = screens.classify(rgb)
    if menu is not None:
        return menu
    if board:
        return "boss" if screens.boss_bar(rgb) else "combat"
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
    wave: tuple | None = None
    wave_alert: bool = False
    confidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("screen", "cells", "hero", "enemies", "hand", "items", "boss_frac", "wave", "wave_alert",
                 "confidence")}


def boss_kind(regions: tuple[str, ...], cells: int) -> str:
    """The boss of the route's current region (its last) whose arena is this wide: Daisuke on 5
    cells, Rei the Ruthless on 7 [wiki]. 0.26 read Rei as Daisuke, and her Dragon Punch, which
    pushes the hero, as his Sword [third series: 4 of 5 bosses were Rei, every HP lost was hers].
    A boss the simulator lacks (Iwao) is planned as Daisuke, the stand-in, as before."""
    return CAT.boss_by_arena(regions[-1] if regions else REGION, cells) or "daisuke"


def perceive(rgb: np.ndarray, kit_tiles=None, regions: tuple[str, ...] = ROSTER, day: int | None = None) -> Perception:
    """kit_tiles: the hero's tile ids (catalogue), for reading the hand. regions: the regions of the
    run's route so far (whose enemies can appear). day: the day the run is on, when known (Day 1
    has no elites)."""
    base_hp, by_tile_kind = roster_tables(tuple(regions))
    rgb = normalize(rgb)
    cs = find_cells(rgb)
    screen = screen_type(rgb, cs)
    p = Perception(screen=screen)
    if screen in ("combat", "unknown"):
        p.wave = hud.read_wave(rgb)
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
                elif plus:
                    # the '+' bubble blinks between a small white '+' and a big yellow one, and both
                    # mean a load this phase: after a white read the unit loaded 367 times in 373
                    # (regular units 304 of 308, bosses 63 of 65; Iwao 14 of 14), after a yellow one
                    # 457 of 458 [bot: the logged turns of 29 runs, bots 0.3-0.28]. Until 0.29 only the
                    # yellow phase counted: Iwao's load was missed 10 times in 14 (his yellow '+' read
                    # as white), so at t159 and t162 of run 20261029-154200 he was planned as holding
                    # and the bot swapped into the corner beside him
                    intent = "queue"
                elif tiles and tiles[0]["colour"] == "yellow":
                    intent = "fire"
                # HP pips, the bubble's tile and the elite emblem decide the kind, the sprite only
                # among the kinds the pips allow (decide_kind). A boss is the only unit without
                # HP pips (its HP is the boss bar); pips win over its sprite: a summoned Charger in
                # the boss arena was read as a second Daisuke [run 20261028-053637, turns 101-161].
                # Which boss: the arena of the route's current region (boss_kind)
                by_tile = by_tile_kind.get(tiles[0]["tile"]) if tiles else None
                elite = day != 1 and elite_pixels(rgb, cx) >= ELITE_PX     # Day 1 has no elites
                alt = None
                if (g + r) == 0 and (screen == "boss" or (kind in CAT.enemies and CAT.enemies[kind].boss)):
                    # no pips: a boss. The fight's first frame shows no boss bar yet, so the screen
                    # reads as a fight and only the sprite (Daisuke's, the one boss sprite known)
                    # says boss [run 20261029-124921 t159: Iwao read as Daisuke]
                    kind = boss_kind(regions, len(cs))
                else:
                    by_pips, alt = decide_kind(g + r, by_tile, kind if kc >= SPRITE_KIND_CONF else None, elite,
                                               base_hp, elites=day != 1)
                    if by_pips:
                        kind = by_pips
                    elif by_tile:
                        kind = by_tile
                if kind == "ashigaru" and tiles and intent != "fire" and windup_pixels(rgb, cx) >= WINDUP_MIN:
                    intent = "fire"
                    tiles[0] = dict(tiles[0], colour="yellow")
                spawning = spawn_pixels(rgb, cx) >= SPAWN_RED_PX
                en = {"x": i, "cx": cx, "kind": kind, "facing": 0 if spawning else facing, "hp": g,
                      "hp_max": g + r, "queue": [t["tile"] for t in tiles],
                      "fire_now": bool(tiles and tiles[0]["colour"] == "yellow"),
                      "intent": intent, "spawning": spawning, "elite": elite, "alt": alt}
                p.enemies.append(en)
                conf[f"enemy{i}"] = round(kc, 2)
            entry["unit"] = kind
        cell_list.append(entry)
    p.cells = cell_list
    p.hand = read_hand(rgb, kit_tiles)
    p.items = read_consumables(rgb)
    if screen == "boss":
        p.boss_frac = boss_fraction(rgb)
    p.wave_alert = wave_alert(rgb)
    p.confidence = conf
    return p
