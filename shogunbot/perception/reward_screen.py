"""Reward screens between levels.

Two layouts carry the grey SKIP pill (screens.find_skip):
  tile_choice  two tiles on pedestals (top y 408, x 744 and 1080); click one
               to add it to the hand. After Bamboo Grove I.
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
UNKNOWN_DIST = 0.07                      # icon distance above which a tile is unknown (classes are >= 0.073 apart)


@dataclass
class RewardScreen:
    kind: str                                         # 'tile_choice' | 'shrine' | 'unknown'
    offers: list = field(default_factory=list)        # tile_choice: pedestal tiles
    hand: list = field(default_factory=list)          # hand tiles with level dots
    upgrade: str | None = None                        # shrine: catalogue upgrade id
    upgrade_lines: list = field(default_factory=list)
    upgrade_match: tuple = ()
    upgrade_lit: bool = False
    skip: tuple | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("kind", "offers", "hand", "upgrade", "upgrade_lines",
                                              "upgrade_match", "upgrade_lit", "skip")}


def upgrade_lit(rgb: np.ndarray) -> bool:
    """UPGRADE is maroon (74,36,41) until a tile sits in the statue, then peach
    (~175,116,82) [collect 20261027-164727 f0109 vs f0111]."""
    x0, x1, y0, y1 = _UPGRADE_BOX
    a = rgb[y0:y1, x0:x1].astype(np.int16)
    peach = (np.abs(a[..., 0] - 175) < 18) & (np.abs(a[..., 1] - 116) < 18) & (np.abs(a[..., 2] - 81) < 18)
    return int(peach.sum()) > 3000


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
    for b in T.find_faces(rgb, 560, 1360, *T.HAND_ROW):
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
                            upgrade_match=(uid, d, second, unknown), upgrade_lit=upgrade_lit(rgb), skip=skip)
    boxes = T.find_faces(rgb, 600, 1320, *T.PEDESTAL_ROW)
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
