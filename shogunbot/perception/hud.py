"""The fight HUD's wave counter, "WAVE x/y", at the top centre of every level that has
waves (all regions; boss levels have no counter).

The text is the pixel font on the 4-px grid, in one of three colours: cream (193,196,185)
on most frames, salmon (205,84,50) while a new wave pops in, grey (122,119,113) on some
frames (meaning not known). Text pixels are those with a brightest channel above
INK_MIN; the band's background stays below 60. Glyphs are cut at blank columns. The
first four must be "WAVE" at its normal size (widths 40, 24, 28, 24 px): while a wave
pops in, the text is drawn larger and is not read. The rest are digits and "/", matched
to templates in assets/wave_glyphs.json, learned from WAVE_SAMPLES
(python -m shogunbot.perception.hud ROOT).
[4,960 recorded frames (Stage 1 runs and collect/20261028-190026): 2,998 reads, all on
fight screens, none contradicting the level's sequence; levels of 3 to 15 waves]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent / "assets"
WAVE_BAND = (86, 166, 600, 1320)        # y0, y1, x0, x1 searched for the counter
INK_MIN = 110                           # brightest channel of a text pixel: grey text 120-130
                                        # [collect/20261028-190026 f1621], background < 60
WAVE_WIDTHS = (40, 24, 28, 24)          # "W", "A", "V", "E" at normal size, px
WIDTH_TOL = 4
WAVE_SAMPLES = {
    # frame: the counter it shows (every digit and "/" appear in these)
    "collect/20261028-190026/f0014_idle.png": "1/4",
    "collect/20261028-190026/f0017_CLICK.png": "2/4",
    "collect/20261028-190026/f0025_idle.png": "3/4",
    "collect/20261028-190026/f0050_idle.png": "1/5",
    "collect/20261028-190026/f0755_idle.png": "1/6",
    "collect/20261028-190026/f1037_idle.png": "7/15",
    "collect/20261028-190026/f1045_idle.png": "8/15",
    "collect/20261028-190026/f1050_idle.png": "9/15",
    "collect/20261028-190026/f1061_idle.png": "10/15",
    "collect/20261028-190026/f1064_idle.png": "11/15",
}


def _runs(mask, gap: int = 1) -> list[list[int]]:
    """[start, end) of True runs; runs closer than `gap` are merged."""
    out: list[list[int]] = []
    s = None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append([s, i])
            s = None
    if s is not None:
        out.append([s, len(mask)])
    merged: list[list[int]] = []
    for r in out:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def glyph_bitmaps(rgb: np.ndarray) -> list[np.ndarray] | None:
    """The counter's glyphs (ink bitmaps cropped to their rows), or None when the band
    holds no single line of text clear of its edges."""
    y0, y1, x0, x1 = WAVE_BAND
    a = rgb[y0:y1, x0:x1].astype(np.int16)
    ink = a.max(axis=2) > INK_MIN
    rows = _runs(ink.any(axis=1), gap=3)
    if len(rows) != 1:
        return None
    r0, r1 = rows[0]
    if r0 == 0 or r1 == ink.shape[0]:
        return None
    sub = ink[r0:r1]
    cols = _runs(sub.any(axis=0), gap=2)
    if not cols or cols[0][0] == 0 or cols[-1][1] == ink.shape[1]:
        return None
    out = []
    for c0, c1 in cols:
        g = sub[:, c0:c1]
        ys = np.nonzero(g.any(axis=1))[0]
        out.append(g[ys.min():ys.max() + 1])
    return out


def _key(bits: np.ndarray) -> str:
    """Art-pixel key of a glyph bitmap (4-px blocks from its own top-left)."""
    b = np.asarray(bits, dtype=np.uint8)
    h, w = b.shape
    H, W = max(1, (h + 2) // 4), max(1, (w + 2) // 4)
    pad = np.zeros((H * 4, W * 4), np.uint8)
    pad[:min(h, H * 4), :min(w, W * 4)] = b[:H * 4, :W * 4]
    art = pad.reshape(H, 4, W, 4).mean(axis=(1, 3)) > 0.5
    return "/".join("".join("#" if v else "." for v in row) for row in art)


_TABLE: dict | None = None


def glyph_table() -> dict[str, str]:
    global _TABLE
    if _TABLE is None:
        p = ASSETS / "wave_glyphs.json"
        _TABLE = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _TABLE


def _match(key: str) -> str:
    table = glyph_table()
    if key in table:
        return table[key]
    best, bd = "?", 4
    flat = key.replace("/", "")
    shape = (key.count("/") + 1, len(key.split("/")[0]))
    for kk, ch in table.items():
        if (kk.count("/") + 1, len(kk.split("/")[0])) != shape:
            continue
        d = sum(a != b for a, b in zip(flat, kk.replace("/", "")))
        if d < bd:
            best, bd = ch, d
    return best


def read_wave(rgb: np.ndarray) -> tuple[int, int] | None:
    """(wave, waves in the level) from the counter, or None when it is not readable."""
    gs = glyph_bitmaps(rgb)
    if gs is None or len(gs) < 7:
        return None
    if any(abs(g.shape[1] - w) > WIDTH_TOL for g, w in zip(gs[:4], WAVE_WIDTHS)):
        return None
    txt = "".join(_match(_key(g)) for g in gs[4:])
    if txt.count("/") != 1 or "?" in txt:
        return None
    cur, total = txt.split("/")
    if not cur or not total:
        return None
    cur_i, total_i = int(cur), int(total)
    if not 1 <= cur_i <= total_i <= 30:
        return None
    return cur_i, total_i


def build(root: Path) -> dict:
    """Learn the digit and "/" glyphs from WAVE_SAMPLES (frames under `root`); writes
    assets/wave_glyphs.json. A sample that relabels a glyph learned before is an error."""
    from PIL import Image
    table: dict[str, str] = dict(glyph_table())
    for rel, text in WAVE_SAMPLES.items():
        p = Path(root) / rel
        if not p.exists():
            print(f"missing {rel}: the glyphs learned from it before are kept")
            continue
        gs = glyph_bitmaps(np.array(Image.open(p).convert("RGB")))
        if gs is None or len(gs) != 4 + len(text):
            raise ValueError(f"{rel}: {0 if gs is None else len(gs)} glyphs for 'WAVE' + {text!r}")
        for g, ch in zip(gs[4:], text):
            k = _key(g)
            if table.get(k, ch) != ch:
                raise ValueError(f"{rel}: glyph already learned as {table[k]!r}, labelled {ch!r} here")
            table[k] = ch
    (ASSETS / "wave_glyphs.json").write_text(json.dumps(table, indent=0, sort_keys=True), encoding="utf-8")
    global _TABLE
    _TABLE = table
    return table


if __name__ == "__main__":
    # python -m shogunbot.perception.hud ROOT     (ROOT holds runs/ and collect/)
    import sys
    t = build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
    print(len(t), "glyphs:", "".join(sorted(set(t.values()))))
