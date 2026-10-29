"""Reward screens between levels.

Two layouts carry the grey SKIP pill (screens.find_skip):
  tile_choice  two tiles on pedestals (top y 408-416, x 744 and 1080); click one
               to add it to the hand. After the first level of a region (Bamboo
               Grove I, Whispering Caves I, Moonlit Port ...).
  shrine       a statue with a white bubble naming an upgrade; click a hand
               tile (it moves into the statue's mouth and UPGRADE lights up),
               then click UPGRADE. After Bamboo Grove II and after the boss.
REROLL costs coins and is never used.
[collect 20261027-164727 f0042-f0043 and f0109-f0112; bot runs 20261027-180145 .. 20261028-061420]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..catalogue import CAT
from . import screens, text, tiles as T

UPGRADE_BUTTON = (0.5010, 0.6963)        # centre of the UPGRADE pill (962, 752)
_UPGRADE_BOX = (800, 1130, 718, 786)     # x0, x1, y0, y1 of the pill
STATUE_FACE = (912, 504)                 # top-left of a tile face in the statue's mouth
                                         # [run 20261028-092419 f0090, f0091, f0125, f0126; collect f0111]
UNKNOWN_DIST = 0.07                      # icon distance above which a tile is unknown (classes are >= 0.073 apart)
# The two pedestal faces sit at fixed spots and bob up and down: left x 744 and 1080, top y 408, 412
# or 416 on all 79 recorded tile-choice screens. A face is a dark square inside a light frame (cream
# in Bamboo Grove and Whispering Caves, grey at night in Moonlit Port). Found by colour runs alone,
# the left face merged with the Caves' maroon rock (Whispering Caves I, run 20261029-124921
# reward_4_unknown, collect/20261028-190026 f0218-f0220: read as 'unknown' and skipped) or with
# Moonlit Port's night sky (f0583-f0585: the box 24 px left, the offer unread).
PEDESTAL_X = (744, 1080)
PEDESTAL_Y = range(400, 428, 4)          # searched on the 4-px art grid
PEDESTAL_MIN = 0.6                       # face score: 0.76-0.84 on all 158 recorded pedestal faces,
                                         # at most 0.34 at these spots on the 221 shrine screens


@dataclass
class RewardScreen:
    kind: str                                         # 'tile_choice' | 'shrine' | 'unknown'
    offers: list = field(default_factory=list)        # tile_choice: pedestal tiles
    hand: list = field(default_factory=list)          # hand tiles with level dots
    upgrade: str | None = None                        # shrine: catalogue upgrade id
    upgrade_lines: list = field(default_factory=list)
    upgrade_match: tuple = ()
    upgrade_lit: bool = False
    statue: dict | None = None                        # shrine: the tile in the statue's mouth
    skip: tuple | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("kind", "offers", "hand", "upgrade", "upgrade_lines",
                                              "upgrade_match", "upgrade_lit", "statue", "skip")}


def upgrade_lit(rgb: np.ndarray, box=_UPGRADE_BOX) -> bool:
    """UPGRADE is dark maroon (72,32,40) with dark text until a tile sits in the
    statue. Then its text turns cream (232,224,208) on a brown pill (~114,66,56),
    or a peach pill (~175,116,82) while the cursor is on it. 0.15 looked for peach
    only, so the bot (cursor moved away) never saw it lit and skipped both shrines
    [run 20261028-092419 f0089 vs f0090, f0124 vs f0125; collect 20261027-164727 f0111].
    box: the shrine's pill, or the shop statue's (shop_screen.UPGRADE_BOX)."""
    x0, x1, y0, y1 = box
    a = rgb[y0:y1, x0:x1].astype(np.int16)
    text = int((a.min(axis=2) > 180).sum())                              # 3568 lit, 0 unlit
    pill = int(((a[..., 0] > 95) & (a[..., 0] - a[..., 2] > 35)).sum())  # brown or peach, not maroon or grey
    return text > 2000 and pill > 5000


def read_statue(rgb: np.ndarray, kit_tiles=None) -> dict | None:
    """The tile in the shrine statue's mouth (tile, damage, cooldown), or None
    when the mouth is empty. The face sits at a fixed spot on the dark statue,
    where find_faces cannot separate it from the stone."""
    g = T.grid(rgb, STATUE_FACE)
    tid, dist, conf = T.classify(g, kit_tiles)
    if tid is None or dist > UNKNOWN_DIST:
        return None                                  # empty mouth: best distance 0.073
    dmg, dconf = T.read_digit(g)
    cd, _ = T.read_marks(g)
    return {"tile": tid, "dist": round(dist, 4), "conf": round(conf, 2), "damage": dmg,
            "damage_conf": round(dconf, 2), "cooldown": cd}


def face_score(rgb: np.ndarray, x: int, y: int) -> float:
    """How much (x, y) looks like the top-left of a pedestal face: the share of dark pixels on
    the face's outer 4 px, and of light pixels in the frame 2-6 px outside it (corners left out,
    they are rounded); the lower of the two."""
    a = rgb.astype(np.int16)
    w, h = T.FACE_W, T.FACE_H
    box = a[y:y + h, x:x + w]
    edge = np.concatenate([box[:4].reshape(-1, 3), box[-4:].reshape(-1, 3),
                           box[4:-4, :4].reshape(-1, 3), box[4:-4, -4:].reshape(-1, 3)])
    ring = np.concatenate([a[y - 6:y - 2, x + 12:x + w - 12].reshape(-1, 3),
                           a[y + h + 2:y + h + 6, x + 12:x + w - 12].reshape(-1, 3),
                           a[y + 12:y + h - 12, x - 6:x - 2].reshape(-1, 3),
                           a[y + 12:y + h - 12, x + w + 2:x + w + 6].reshape(-1, 3)])
    dark = float((edge.max(axis=1) < 70).mean())
    light = float((ring.min(axis=1) > 90).mean())
    return min(dark, light)


def pedestal_faces(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Face boxes on the tile-choice pedestals, left to right: the fixed spots first (any
    backdrop), else faces found by colour runs (another layout)."""
    out = []
    for x in PEDESTAL_X:
        score, y = max((face_score(rgb, x, y), y) for y in PEDESTAL_Y)
        if score >= PEDESTAL_MIN:
            out.append((x, y, x + T.FACE_W, y + T.FACE_H))
    return out if len(out) == len(PEDESTAL_X) else T.find_faces(rgb, 600, 1320, *T.PEDESTAL_ROW)


def identify(g: np.ndarray, damage: int | None, cooldown: int, candidates=None) -> tuple[str | None, float, float]:
    """Tile id of a face. Unupgraded tiles show their catalogue damage and
    cooldown, which narrows the candidates before the icon decides."""
    pool = list(candidates) if candidates is not None else [t.id for t in CAT.tiles.values() if t.obtainable]
    exact = [t for t in pool if CAT.tile(t).cooldown == cooldown and (damage is None or CAT.tile(t).damage == damage)]
    tid, dist, conf = T.classify(g, exact or pool)
    if tid is None or dist > UNKNOWN_DIST:
        return None, dist, 0.0
    return tid, dist, conf if exact else conf * 0.5


def read_hand(rgb: np.ndarray, kit_tiles=None, dots: bool = False) -> list[dict]:
    """Hand tiles left to right: tile, damage, cooldown, charge, ready, click spot
    (and level dots on reward screens). `kit_tiles` restricts the icon match."""
    out = []
    for b in T.find_faces(rgb, 560, 1360, *T.HAND_ROW, tinted=True):
        g = T.grid(rgb, b)
        tid, dist, conf = T.classify(g, kit_tiles)
        dmg, dconf = T.read_digit(g)
        cd, charged = T.read_marks(g)
        f = {"tile": tid if dist <= UNKNOWN_DIST else None, "dist": round(dist, 4), "conf": round(conf, 2),
             "damage": dmg, "damage_conf": round(dconf, 2), "cooldown": cd, "charged": charged,
             "ready": T.is_ready(g), "cx": b[0] + T.FACE_W // 2,
             "fx": (b[0] + T.FACE_W / 2) / 1920, "fy": (b[1] + T.FACE_H / 2) / 1080}
        if dots:
            f["levels"], f["used"] = T.read_dots(rgb, b)
        out.append(f)
    return out


def read(rgb: np.ndarray, kit_tiles=None) -> RewardScreen:
    skip = screens.find_skip(rgb)
    hand = read_hand(rgb, kit_tiles, dots=True)
    bubble = text.find_bubble(rgb)
    if bubble is not None:
        lines = text.read_text(rgb, bubble)
        uid, d, second = text.match_upgrade([s for s, _ in lines])
        unknown = sum(s.count("?") for s, _ in lines)
        total = max(1, sum(len(s.replace(" ", "")) for s, _ in lines))
        # an exact read of known glyphs is trusted ("-1" vs "-2" differ by one glyph);
        # a read with unknown glyphs needs a clear winner
        ok = uid is not None and ((d == 0 and unknown == 0) or
                                  (d <= 1 and second - d >= 2 and unknown / total <= 0.35))
        return RewardScreen("shrine", hand=hand, upgrade=uid if ok else None, upgrade_lines=lines,
                            upgrade_match=(uid, d, second, unknown), upgrade_lit=upgrade_lit(rgb),
                            statue=read_statue(rgb, kit_tiles), skip=skip)
    boxes = pedestal_faces(rgb)
    if len(boxes) == 2:
        offers = []
        pool = [t.id for t in CAT.base_pool()]
        for b in boxes:
            g = T.grid(rgb, b)
            dmg, dconf = T.read_digit(g)
            cd, _ = T.read_marks(g)
            tid, dist, conf = identify(g, dmg, cd, pool)
            if tid is None:
                tid, dist, conf = identify(g, dmg, cd)   # outside the base pool
            levels, used = T.read_dots(rgb, b)
            offers.append({"tile": tid, "dist": round(dist, 4), "conf": round(conf, 2), "damage": dmg,
                           "cooldown": cd, "levels": levels, "used": used,
                           "fx": (b[0] + T.FACE_W / 2) / 1920, "fy": (b[1] + T.FACE_H / 2) / 1080})
        return RewardScreen("tile_choice", offers=offers, hand=hand, skip=skip)
    return RewardScreen("unknown", hand=hand, skip=skip)
