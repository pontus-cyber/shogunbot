"""The shop between regions (M4 step 1) [run 20261029-072042; collect/20261028-190026, shop
visits at f0172, f0329, f0587, f0931, f1365 and f1599].

Layout at 1080p (normalize first), measured on those frames:
- The statue on the left sells an upgrade. Its speech bubble names it in the shrine font: one
  line ("-2 COOLDOWN", "+1 DAMAGE") or, for an enchantment, the name, the red cooldown cost and
  a description in a smaller font that is not read ("POISON / +2 COOLDOWN", "SHOCKWAVE / +1
  COOLDOWN"). text.read_text reads it and text.match_upgrade names it. The bubble stands at
  x 300-790, y 85-350; the shrine's search region starts at x 640, so the shop has its own.
  After a purchase the bubble shrinks to a dash (f0931).
- Its price, a coin and white digits, sits under the statue (x 440-630, y 640-700).
- The counter on the right holds four or five slots in a row (x 950-1850): an offer with its
  price (a coin and digits), a padlock with a skull price (it unlocks a slot), locked slots
  without a price, or empty ones (sold). Prices are white digits at y 385-445.
- Coins are the second counter at the top left (a coin, then white digits, y 60-130); skulls
  the first (y 10-60).
- Every number is in the HUD's italic font: hud.py's glyph table (learned on the wave counter)
  reads them, 9 art px tall (36 px). A white skull icon reads as '?'.
- GO at the bottom right (screens.SHOP_GO); the shop is recognised by its GO pill (screens.shop).
- Buying the statue's upgrade works as at a shrine: click a hand tile, it moves into the statue's
  mouth (a tile face at STATUE_FACE) and UPGRADE lights up (UPGRADE_BOX); click UPGRADE, the
  coins drop by the price and the tile goes back to the hand upgraded; the price then goes up
  (20 -> 40 for -2 COOLDOWN). Clicking the tile in the mouth puts it back in the hand. [collect/
  20261028-190026: the user bought at f0172-f0178 (27 -> 7 coins), f0329-f0331, f0591-f0594,
  f0938 and f1375-f1376]
What the counter sells is not identified yet (no icon templates), so only the statue is bought from.
"""

from __future__ import annotations

import numpy as np

from . import hud, text, tiles as T
from .reward_screen import UNKNOWN_DIST, upgrade_lit as _lit

BUBBLE_REGION = (300, 820, 20, 400)     # x0, x1, y0, y1: the statue's bubble [f0172 .. f1599]
BUBBLE_MIN_PX = 12000                   # one line: 26,512-33,152 white px; enchantments ~90,000
PRICE_BOX = (440, 630, 600, 720)        # x0, x1, y0, y1: the statue's price
COUNTER_BOX = (950, 1860, 375, 455)     # the counter's price row
COINS_BOX = (150, 300, 60, 130)         # the coin counter's digits (the coin icon is left of x 150)
SKULLS_BOX = (150, 300, 10, 60)
INK_MIN = 200                           # white digits: every channel above this
GLYPH_H = (32, 40)                      # digit and skull glyph heights, px (36 measured)
LABEL_GAP = 30                          # glyphs closer than this belong to one label: 4 px between digits,
                                        # 16 between the skull icon and its digit; slots 100+ apart
# The statue's mouth and UPGRADE pill, 456 px left of and 4 px below the shrine's [collect/20261028-190026:
# the tile faces at f0173, f0176, f0177 and f0330 match a Swirl here within 0.0002; an empty mouth reads
# 0.044-0.045 from its nearest tile, and an enchanted Swirl at f1376 0.042, so the face read counts only
# below MOUTH_DIST and a lit UPGRADE decides otherwise; lit: 3,568-4,780 cream text px and 14,992-15,504
# pill px on the 7 frames with a tile in the mouth, at most 3,208 and 0 on the 57 others]
STATUE_FACE = (456, 508)
UPGRADE_BOX = (344, 674, 722, 790)
UPGRADE_BUTTON = (0.2651, 0.7000)       # the pill's centre (509, 756)
MOUTH_DIST = 0.03


def _glyphs(rgb: np.ndarray, box) -> list[tuple[int, int, np.ndarray]]:
    """(x0, x1, bitmap) of the white glyphs in box, left to right, each cropped to its rows."""
    x0, x1, y0, y1 = box
    a = rgb[y0:y1, x0:x1].astype(np.int16)
    ink = a.min(axis=2) > INK_MIN
    out = []
    for c0, c1 in hud._runs(ink.any(axis=0), gap=2):
        g = ink[:, c0:c1]
        ys = np.nonzero(g.any(axis=1))[0]
        g = g[ys.min():ys.max() + 1]
        if GLYPH_H[0] <= g.shape[0] <= GLYPH_H[1] and g.shape[1] <= 40:
            out.append((x0 + c0, x0 + c1, g))
    return out


def _labels(glyphs) -> list[list[tuple[int, int, np.ndarray]]]:
    out: list = []
    for g in glyphs:
        if out and g[0] - out[-1][-1][1] < LABEL_GAP:
            out[-1].append(g)
        else:
            out.append([g])
    return out


def _number(glyphs) -> int | None:
    s = "".join(hud._match(hud._key(g)) for _, _, g in glyphs)
    return int(s) if s.isdigit() else None


def number_in(rgb: np.ndarray, box) -> int | None:
    """The one number in box, or None."""
    labels = _labels(_glyphs(rgb, box))
    return _number(labels[0]) if len(labels) == 1 else None


def statue(rgb: np.ndarray) -> dict:
    """The statue's upgrade: {'upgrade': catalogue id or None, 'lines': text read, 'price'}."""
    box = text.find_bubble(rgb, BUBBLE_REGION, BUBBLE_MIN_PX)
    lines = text.read_text(rgb, box) if box else []
    up = None
    if lines:
        up, dist, second = text.match_upgrade([t for t, _ in lines])
        if dist >= second:          # no clear match
            up = None
    return {"upgrade": up, "lines": [t for t, _ in lines], "price": number_in(rgb, PRICE_BOX)}


def counter(rgb: np.ndarray) -> list[dict]:
    """The counter's prices left to right: {'x', 'price', 'currency'}. A label that starts with
    the white skull icon costs skulls (a padlock: unlocks a slot), else coins."""
    out = []
    for lab in _labels(_glyphs(rgb, COUNTER_BOX)):
        first = hud._match(hud._key(lab[0][2]))
        currency = "skull" if first == "?" and len(lab) > 1 else "coin"
        digits = lab[1:] if currency == "skull" else lab
        out.append({"x": int((lab[0][0] + lab[-1][1]) // 2), "price": _number(digits), "currency": currency})
    return out


def statue_tile(rgb: np.ndarray, kit_tiles=None) -> dict | None:
    """The tile in the statue's mouth (tile, damage, cooldown), or None when the mouth reads empty."""
    g = T.grid(rgb, STATUE_FACE)
    tid, dist, conf = T.classify(g, kit_tiles)
    if tid is None or dist > min(MOUTH_DIST, UNKNOWN_DIST):
        return None
    dmg, _ = T.read_digit(g)
    cd, _ = T.read_marks(g)
    return {"tile": tid, "dist": round(dist, 4), "damage": dmg, "cooldown": cd}


def upgrade_lit(rgb: np.ndarray) -> bool:
    """UPGRADE lights up (cream text on a brown or peach pill) while a tile sits in the mouth."""
    return _lit(rgb, UPGRADE_BOX)


def read(rgb: np.ndarray) -> dict:
    """Everything the shop shows that is read: the statue's upgrade and price, the counter's
    prices, coins and skulls (None where unread)."""
    st = statue(rgb)
    return {"upgrade": st["upgrade"], "upgrade_text": st["lines"], "upgrade_price": st["price"],
            "counter": counter(rgb), "coins": number_in(rgb, COINS_BOX), "skulls": number_in(rgb, SKULLS_BOX)}


def describe(d: dict) -> str:
    """One line for the console."""
    if not d:
        return "nothing read"
    up = " / ".join(d.get("upgrade_text") or []) or "-"
    price = d.get("upgrade_price")
    offers = ", ".join(f"{o['price'] if o['price'] is not None else '?'} {o['currency']}s"
                       for o in d.get("counter") or []) or "none read"
    return (f"statue {up}" + (f" for {price}" if price is not None else "") +
            f"; counter {offers}; coins {d.get('coins') if d.get('coins') is not None else '?'}")
