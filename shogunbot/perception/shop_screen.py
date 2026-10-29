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

from ..catalogue import CAT
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
SALE_INK = (235, 165, 205, 60)          # a SALE price's gold digits: R >= 235, G 165-205, B <= 60 (measured
                                        # 244-247, 185-189, 23-27 on rec2 f0790; the coin icon's light
                                        # shade 230,184,95 and its body 220,151,47 stay out)
ICON_Y = (395, 445)                     # the currency icon left of a price's digits [rec2 f0177, f0790;
ICON_W = 48                             # run 20261029-184142 shop_1]: heart 162,45,37 and 203,80,47;
ICON_PX = 60                            # coin 220,151,47, 187,113,32 and 230,184,95 (464 heart px and
                                        # 512-592 coin px on those frames)
TIP_SLOT_TOL = 70                       # px between a tooltip box's centre and its slot's price (24 measured)
SLOT_Y = 375                            # a counter slot, where the cursor shows its tooltip and a click buys it:
                                        # your clicks that bought were at y 369-419 [rec2 f0181, f0447, f0449,
                                        # f0458, f0460, f0786, f0791]; the slots span y 255-445
# The statue's mouth and UPGRADE pill, 456 px left of and 4 px below the shrine's [collect/20261028-190026:
# the tile faces at f0173, f0176, f0177 and f0330 match a Swirl here within 0.0002; an empty mouth reads
# 0.044-0.045 from its nearest tile, and an enchanted Swirl at f1376 0.042, so the face read counts only
# below MOUTH_DIST and a lit UPGRADE decides otherwise; lit: 3,568-4,780 cream text px and 14,992-15,504
# pill px on the 7 frames with a tile in the mouth, at most 3,208 and 0 on the 57 others]
STATUE_FACE = (456, 508)
UPGRADE_BOX = (344, 674, 722, 790)
UPGRADE_BUTTON = (0.2651, 0.7000)       # the pill's centre (509, 756)
MOUTH_DIST = 0.03


def _glyphs(rgb: np.ndarray, box, sale: bool = False) -> list[tuple[int, int, np.ndarray]]:
    """(x0, x1, bitmap) of the white glyphs in box, left to right, each cropped to its rows. sale: the
    gold digits of a price on SALE count too (the counter)."""
    x0, x1, y0, y1 = box
    a = rgb[y0:y1, x0:x1].astype(np.int16)
    ink = a.min(axis=2) > INK_MIN
    if sale:
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        ink |= (r >= SALE_INK[0]) & (g >= SALE_INK[1]) & (g <= SALE_INK[2]) & (b <= SALE_INK[3])
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


def _icon(rgb: np.ndarray, x: int) -> str:
    """The currency icon just left of a price's first digit at x: 'heart' (red) or 'coin' (orange)."""
    a = rgb[ICON_Y[0]:ICON_Y[1], max(0, x - ICON_W):x - 2].astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    heart = int(((r >= 145) & (r <= 215) & (g >= 35) & (g <= 95) & (b >= 25) & (b <= 60) & (r - g >= 90)).sum())
    coin = int(((r >= 180) & (g >= 105) & (g <= 160) & (b >= 20) & (b <= 65) & (r - g >= 50)).sum())
    return "heart" if heart > max(ICON_PX, coin) else "coin"


def counter(rgb: np.ndarray) -> list[dict]:
    """The counter's prices left to right: {'x', 'price', 'currency'}. A label that starts with
    the white skull icon costs skulls (a padlock: unlocks a slot); else a red heart before the digits
    costs hearts (HP: a blood service such as REROLL), an orange coin coins. A price on SALE is gold
    [rec2 f0790: 7 coins under 'SALE!!']; 0.24-0.31 read white digits only, and a heart price as coins
    [run 20261029-184142 shop_1: a 1-heart reroll read as 1 coin]."""
    out = []
    for lab in _labels(_glyphs(rgb, COUNTER_BOX, sale=True)):
        first = hud._match(hud._key(lab[0][2]))
        currency = "skull" if first == "?" and len(lab) > 1 else _icon(rgb, lab[0][0])
        digits = lab[1:] if currency == "skull" else lab
        out.append({"x": int((lab[0][0] + lab[-1][1]) // 2), "price": _number(digits), "currency": currency})
    return out


def tooltip(rgb: np.ndarray) -> dict | None:
    """The counter's tooltip, which shows while the cursor is on a slot: {'box', 'title', 'kind', 'id'},
    or None. The title names the offer between dashes ('- QUICK RECOVERY -'), in the shrine font at
    3 px per art pixel (text.read_title); a blue line under it says what it is ('SKILL'), then the
    description in the small font. kind 'skill' or 'consumable' with the catalogue id when the title
    matches one name clearly, else 'other' (SHOP UPGRADE, REROLL, BLOOD EXCHANGE) with id None
    [rec2, collect/20261029-160032 f0177-f1753; rec1 f0178-f1610]."""
    box, title = text.read_title(rgb, text.TIP_AREA)
    if box is None:
        return None
    name = title.strip(" -")
    kind, oid = identify(name)
    return {"box": box, "title": name, "kind": kind, "id": oid}


def tip_over(tip: dict, x: int) -> bool:
    """The tooltip stands over the slot whose price is centred at x: its box is centred 24 px left of the
    price on every recorded tooltip (rec2: x 1160 -> box 864-1408, 1328 -> 1032-1576, 1076 -> 780-1324,
    1244 -> 948-1492, 1498 -> 1200-1744); the slots are 168 px apart."""
    x0, _, x1, _ = tip["box"]
    return abs((x0 + x1) / 2 - x) <= TIP_SLOT_TOL


def identify(name: str) -> tuple[str, str | None]:
    """('skill', id) or ('consumable', id) for a tooltip title by the nearest catalogue name (edit
    distance, '?' matching any letter, as the shrine's upgrades), ('other', None) when no name is
    clearly nearest."""
    t = " ".join(name.upper().split())
    scored = sorted([(text._dist(t, s.name.upper()), "skill", s.id) for s in CAT.skills.values()] +
                    [(text._dist(t, c.name.upper()), "consumable", c.id) for c in CAT.consumables.values()])
    if not scored or not t:
        return "other", None
    d, kind, oid = scored[0]
    if d <= max(1, len(t) // 6) and (len(scored) < 2 or scored[1][0] > d):
        return kind, oid
    return "other", None


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
    named = {o["x"]: o.get("title") for o in d.get("goods") or []}
    offers = ", ".join((f"{named[o['x']]} " if named.get(o["x"]) else "") +
                       f"{o['price'] if o['price'] is not None else '?'} {o['currency']}s"
                       for o in d.get("counter") or []) or "none read"
    return (f"statue {up}" + (f" for {price}" if price is not None else "") +
            f"; counter {offers}; coins {d.get('coins') if d.get('coins') is not None else '?'}")
