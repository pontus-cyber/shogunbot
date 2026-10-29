"""Shrine text: the white speech bubble over the statue names the upgrade on
offer ("-1 COOLDOWN", "+2 DAMAGE / +3 COOLDOWN"). Gains are dark, costs red.

The bubble font is pixel art on the 4-px grid, 10 art px tall. Lines are cut
into glyphs at blank columns and each glyph is matched against glyphs.json,
learned from frames with known text (build_glyphs). An unknown glyph reads
'?', and the line set is then matched to the catalogue's upgrade texts by
edit distance, so an unseen letter still lands on the right upgrade.
[measured on runs 20261027-182237 .. 20261028-061420, collect 20261027-164727]

The bubble's bottom sits on the statue (y 344) and it grows upward with its
text: top y 244 for one line, 186 for two, and 111 or 51 for an enchantment
shrine, which prints the name, the red cooldown cost and a description in a
smaller font (20 px rows, not read). 0.19 searched from y 150 only and read
nothing, or only the cost, on both enchantment shrines it met [runs
20261028-132238 and -132952 reward_3_shrine].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..catalogue import CAT

ASSETS = Path(__file__).parent / "assets"
BUBBLE_REGION = (640, 1290, 20, 380)    # x0, x1, y0, y1 searched for the white bubble; widest seen
                                        # 708-1220, tallest 51-344 [run 20261028-132952 reward_3_shrine]
GLYPH_SAMPLES = {
    # frame: the lines its bubble shows (the shrine font's glyphs are learned from these)
    "runs/20261027-182237/frames/f0135_CLICK.png": ["+1 UPGRADE SLOT", "-1 COOLDOWN"],
    "runs/20261028-044407/frames/f0097_CLICK.png": ["-1 COOLDOWN"],
    "runs/20261028-053637/frames/f0103_CLICK.png": ["+1 DAMAGE", "+1 COOLDOWN"],
    "runs/20261028-061420/frames/f0089_CLICK.png": ["+2 DAMAGE", "+3 COOLDOWN"],
    "collect/20261027-164727/f0109_idle.png": ["-2 COOLDOWN"],
    "runs/20261028-130456/frames/reward_2_shrine.png": ["-1 DAMAGE", "-4 COOLDOWN"],
    "runs/20261028-132238/frames/reward_3_shrine.png": ["FREE-PLAY", "+3 COOLDOWN"],
    "runs/20261028-132952/frames/reward_3_shrine.png": ["PERFECT STRIKE", "+1 COOLDOWN"],
}


def find_bubble(rgb: np.ndarray, region=BUBBLE_REGION, min_px: int = 25000):
    """(x0, y0, x1, y1) of the near-white bubble (~233,232,224) or None. region: x0, x1, y0, y1
    searched (the shop's statue stands left of the shrine's: shop_screen.BUBBLE_REGION)."""
    x0, x1, y0, y1 = region
    reg = rgb[y0:y1, x0:x1].astype(np.int16)
    white = (reg.min(axis=2) > 210) & (reg.max(axis=2) - reg.min(axis=2) < 20)
    if white.sum() < min_px:         # the bubble is a solid block (~32 000 px or more); tooltips are dark
        return None
    band = _runs(white.mean(axis=1) > 0.25, gap=6)
    if not band:
        return None
    r0, r1 = max(band, key=lambda r: r[1] - r[0])        # the tallest block of white rows
    cols = np.nonzero(white[r0:r1].mean(axis=0) > 0.25)[0]   # columns measured on those rows only
    if not len(cols):
        return None
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    if white[r0:r1, c0:c1].sum() < 0.5 * (r1 - r0) * (c1 - c0):
        return None
    return (x0 + c0, y0 + r0, x0 + c1, y0 + r1)


def _ink(t: np.ndarray):
    dark = t.max(axis=2) < 120
    red = (t[..., 0] > t[..., 1] + 50) & (t[..., 0] > 100)
    return dark | red, red


def _runs(mask, gap: int = 0):
    """(start, end) of the True runs; runs closer than `gap` are merged."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    if gap:
        merged = []
        for r in out:
            if merged and r[0] - merged[-1][1] < gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        out = merged
    return out


def lines(rgb: np.ndarray, box) -> list[tuple[np.ndarray, bool]]:
    """Text lines in the bubble as (art-pixel ink matrix 10 x w, is_red)."""
    x0, y0, x1, y1 = box
    t = rgb[y0 + 8:y1 - 4, x0 + 8:x1 - 8].astype(np.int16)
    out = []
    if t.size == 0:
        return out
    ink, red = _ink(t)
    for r0, r1 in _runs(ink.mean(axis=1) > 0.005):
        if not 32 <= r1 - r0 <= 48:
            continue                      # bubble outline, statue teeth
        cols = np.nonzero(ink[r0:r1].any(axis=0))[0]
        c0 = int(cols.min())
        w = (int(cols.max()) + 1 - c0 + 3) // 4 * 4
        blk = ink[r0:r0 + 40, c0:c0 + w]
        if blk.shape != (40, w):
            continue
        art = blk.reshape(10, 4, w // 4, 4).mean(axis=(1, 3)) > 0.5
        is_red = red[r0:r1].sum() > 0.5 * ink[r0:r1].sum()
        out.append((art, bool(is_red)))
    return out


def glyphs(art: np.ndarray) -> list[np.ndarray | None]:
    """Cut a line into glyphs at blank columns; None marks a word space."""
    out = []
    runs = _runs(art.any(axis=0))
    for i, (c0, c1) in enumerate(runs):
        if i and c0 - runs[i - 1][1] >= 3:
            out.append(None)
        out.append(art[:, c0:c1])
    return out


def _key(g: np.ndarray) -> str:
    return "/".join("".join("#" if v else "." for v in row) for row in g)


_GLYPHS = None


def glyph_table() -> dict[str, str]:
    global _GLYPHS
    if _GLYPHS is None:
        p = ASSETS / "glyphs.json"
        _GLYPHS = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _GLYPHS


def _match(g: np.ndarray) -> str:
    table = glyph_table()
    k = _key(g)
    if k in table:
        return table[k]
    best, bd = "?", 4
    for kk, ch in table.items():
        rows = kk.split("/")
        if len(rows[0]) != g.shape[1]:
            continue
        d = sum(a != b for a, b in zip(k.replace("/", ""), kk.replace("/", "")))
        if d < bd:
            best, bd = ch, d
    return best


def read_text(rgb: np.ndarray, box=None) -> list[tuple[str, bool]]:
    """[(line text, is_red)] read from the shrine bubble."""
    box = box or find_bubble(rgb)
    if box is None:
        return []
    out = []
    for art, is_red in lines(rgb, box):
        s = "".join(" " if g is None else _match(g) for g in glyphs(art))
        out.append((s, is_red))
    return out


def _dist(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if (ca == cb or ca == "?") else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def match_upgrade(text_lines: list[str]) -> tuple[str | None, int, int]:
    """(upgrade id, distance, runner-up distance) for the lines read."""
    got = " / ".join(t.strip() for t in text_lines)
    scored = sorted((_dist(got, " / ".join(u.lines)), u.id) for u in CAT.upgrades.values())
    if not scored:
        return None, 99, 99
    best_d, best = scored[0]
    second = scored[1][0] if len(scored) > 1 else 99
    return best, best_d, second


def build_glyphs(samples: list[tuple[np.ndarray, list[str]]]) -> dict:
    """Learn glyphs from (frame, known lines) pairs; writes assets/glyphs.json.
    Glyphs learned before are kept; a sample that would relabel one is an error."""
    table: dict[str, str] = dict(glyph_table())
    for rgb, known in samples:
        box = find_bubble(rgb)
        got = lines(rgb, box) if box else []
        if len(got) != len(known):
            raise ValueError(f"read {len(got)} lines, expected {len(known)}: {known}")
        for (art, _), text in zip(got, known):
            gs = glyphs(art)
            chars = list(text)
            if len(gs) != len(chars):
                raise ValueError(f"{text!r}: {len(gs)} glyph slots for {len(chars)} characters")
            for g, ch in zip(gs, chars):
                if g is None:
                    if ch != " ":
                        raise ValueError(f"{text!r}: space where {ch!r} was expected")
                    continue
                k = _key(g)
                if table.get(k, ch) != ch:
                    raise ValueError(f"{text!r}: glyph already learned as {table[k]!r}, labelled {ch!r} here")
                table[k] = ch
    (ASSETS / "glyphs.json").write_text(json.dumps(table, indent=0, sort_keys=True), encoding="utf-8")
    global _GLYPHS
    _GLYPHS = table
    return table


def build(root: Path) -> dict:
    """Rebuild glyphs.json from GLYPH_SAMPLES, frames under `root` (holds runs/ and collect/)."""
    from PIL import Image
    samples = []
    for rel, known in GLYPH_SAMPLES.items():
        p = Path(root) / rel
        if not p.exists():
            print(f"missing {rel}: the glyphs learned from it before are kept")
            continue
        samples.append((np.array(Image.open(p).convert("RGB")), known))
    return build_glyphs(samples)


if __name__ == "__main__":
    # python -m shogunbot.perception.text ROOT     (ROOT holds runs/ and collect/)
    import sys
    t = build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
    print(len(t), "glyphs:", "".join(sorted(set(t.values()))))
