"""Floor item templates: perception/assets/floor_items.npz (0.50, core-requests item 15c).

    python -m shogunbot.perception.build_floor ROOT     # ROOT holds runs/

Each SAMPLES entry is a frame with an item on the floor: (screen id, run, frame, the sprite window's left x). The
windows sit on the screen's 4-px grid at each item's fixed height (GEOMETRY). The frames were found by a colour scan
of the runs' before-frames (2026-11-03): the jar's green liquid as a 20 x 24 px blob with its top at y 728, the
shield flask's blue liquid as a 24 x 20 blob at y 724 (20 x 20 at 728 where the region's grading cuts its top row),
Mass Ice's cyan crystal as a 32 x 20 blob at y 716; Mass Poison is one frame [run 20261031-165514 t0206]; Rain of
Mirrors' beige card with diagonal streaks matched the Mass Poison template at 0.70-0.81 (its card frame), up to 3 frames
from each of 21 runs whose card interior shows the streaks (block brightness spread 8-20). Samples
are spread over the regions (up to 8 per stage, round robin over the runs), so the sprite's blocks agree across
backdrops while the backdrop's do not.

A template is the median of its samples' 4 x 4 blocks, each sample's colour channels first scaled to the median's
over the core blocks (the regions grade colours by channel). Its mask: for a card, the 9 x 13 block frame without
its corners (the card is opaque; one sample cannot tell the backdrop apart); for the others, the blocks whose colour
varies by at most SPREAD_MAX across the samples (median over the samples of the largest channel deviation), above
the floor's front edge (y 756), as one connected piece. A sample that correlates below KEEP_MIN with the first
template is dropped and the template built again (3 of the 40 Edamame samples were bamboo leaves).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .floor import ASSET, GRID, ncc
from .perceive import normalize

# screen id -> (window top y, rows, cols, shape, core blocks (r0, r1, c0, c1) for the channel gains)
GEOMETRY = {
    "green_potion": (708, 13, 13, "sprite", (5, 10, 3, 9)),
    "blue_potion": (704, 14, 11, "sprite", (5, 10, 3, 8)),
    "ice_card": (704, 13, 9, "card", (3, 10, 2, 7)),
    "skull_card": (704, 13, 9, "card", (3, 10, 2, 7)),
    "mirror_card": (704, 13, 9, "card", (3, 10, 2, 7)),
}
SPREAD_MAX = 8.0
FLOOR_EDGE_Y = 756
KEEP_MIN = 0.80

SAMPLES = [
    ("green_potion", "20261029-072323", "t0045_before", 1464),
    ("green_potion", "20261029-124921", "t0044_before", 460),
    ("green_potion", "20261029-154200", "t0048_before", 460),
    ("green_potion", "20261029-184142", "t0031_before", 460),
    ("green_potion", "20261031-080407", "t0048_before", 812),
    ("green_potion", "20261031-102644", "t0036_before", 860),
    ("green_potion", "20261031-114119", "t0032_before", 716),
    ("green_potion", "20261031-214756", "t0044_before", 1232),
    ("green_potion", "20261031-114119", "t0108_before", 1140),
    ("green_potion", "20261101-021522", "t0138_before", 1116),
    ("green_potion", "20261101-122050", "t0106_before", 1172),
    ("green_potion", "20261031-114119", "t0105_before", 1140),
    ("green_potion", "20261101-021522", "t0144_before", 1168),
    ("green_potion", "20261031-114119", "t0109_before", 1140),
    ("green_potion", "20261101-021522", "t0137_before", 1116),
    ("green_potion", "20261031-114119", "t0101_before", 1140),
    ("green_potion", "20261030-115221", "t0183_before", 1032),
    ("green_potion", "20261031-114119", "t0207_before", 1308),
    ("green_potion", "20261031-154544", "t0458_before", 416),
    ("green_potion", "20261031-180506", "t0173_before", 964),
    ("green_potion", "20261031-203835", "t0234_before", 528),
    ("green_potion", "20261031-233208", "t0241_before", 960),
    ("green_potion", "20261101-164934", "t0187_before", 1284),
    ("green_potion", "20261030-115221", "t0193_before", 884),
    ("green_potion", "20261101-004025", "t0580_before", 1228),
    ("green_potion", "20261101-021522", "t0789_before", 740),
    ("green_potion", "20261101-053431", "t1582_before", 792),
    ("green_potion", "20261101-075048", "t0658_before", 416),
    ("green_potion", "20261101-105146", "t0555_before", 908),
    ("green_potion", "20261101-135625", "t0550_before", 1088),
    ("green_potion", "20261101-151701", "t0592_before", 912),
    ("green_potion", "20261102-145513", "t0526_before", 1196),
    ("green_potion", "20261102-065902", "t0741_before", 640),
    ("green_potion", "20261102-145513", "t0739_before", 1204),
    ("green_potion", "20261102-065902", "t0740_before", 640),
    ("green_potion", "20261102-145513", "t0742_before", 1204),
    ("green_potion", "20261102-065902", "t0742_before", 592),
    ("green_potion", "20261102-145513", "t0733_before", 1304),
    ("green_potion", "20261102-065902", "t0739_before", 640),
    ("green_potion", "20261102-065902", "t0749_before", 688),
    ("blue_potion", "20261031-102644", "t0067_before", 1288),
    ("blue_potion", "20261031-165514", "t0038_before", 1032),
    ("blue_potion", "20261101-034756", "t0044_before", 732),
    ("blue_potion", "20261101-092253", "t0004_before", 1160),
    ("blue_potion", "20261031-102644", "t0041_before", 1336),
    ("blue_potion", "20261031-165514", "t0040_before", 1032),
    ("blue_potion", "20261101-034756", "t0043_before", 732),
    ("blue_potion", "20261031-165514", "t0037_before", 1032),
    ("blue_potion", "20261030-115221", "t0113_before", 436),
    ("blue_potion", "20261031-144722", "t0111_before", 1180),
    ("blue_potion", "20261031-185210", "t0132_before", 1124),
    ("blue_potion", "20261030-115221", "t0112_before", 388),
    ("blue_potion", "20261031-185210", "t0125_before", 1124),
    ("blue_potion", "20261031-114119", "t0162_before", 1320),
    ("blue_potion", "20261031-123442", "t0167_before", 1020),
    ("blue_potion", "20261031-134157", "t0217_before", 496),
    ("blue_potion", "20261031-165514", "t0356_before", 1032),
    ("blue_potion", "20261031-194343", "t0246_before", 1276),
    ("blue_potion", "20261031-203835", "t0246_before", 724),
    ("blue_potion", "20261101-004025", "t0261_before", 596),
    ("blue_potion", "20261101-034756", "t0176_before", 1180),
    ("blue_potion", "20261031-203835", "t0436_before", 776),
    ("blue_potion", "20261031-233208", "t0340_before", 920),
    ("blue_potion", "20261101-053431", "t0427_before", 1004),
    ("blue_potion", "20261101-092253", "t0376_before", 780),
    ("blue_potion", "20261101-105146", "t0386_before", 908),
    ("blue_potion", "20261101-122050", "t0511_before", 1316),
    ("blue_potion", "20261101-151701", "t0375_before", 840),
    ("blue_potion", "20261101-164934", "t0335_before", 652),
    ("blue_potion", "20261101-004025", "t0510_before", 912),
    ("blue_potion", "20261101-034756", "t0568_before", 860),
    ("blue_potion", "20261101-053431", "t0554_before", 792),
    ("blue_potion", "20261101-135625", "t0560_before", 1476),
    ("blue_potion", "20261101-151701", "t0599_before", 796),
    ("blue_potion", "20261102-191231", "t0653_before", 1332),
    ("blue_potion", "20261101-004025", "t0511_before", 912),
    ("blue_potion", "20261101-034756", "t0569_before", 860),
    ("blue_potion", "20261103-020829", "t0897_before", 1272),
    ("ice_card", "20261031-165514", "t0101_before", 1136),
    ("ice_card", "20261031-165514", "t0102_before", 1136),
    ("ice_card", "20261031-165514", "t0100_before", 1136),
    ("ice_card", "20261031-165514", "t0104_before", 1136),
    ("ice_card", "20261031-165514", "t0098_before", 1136),
    ("ice_card", "20261031-165514", "t0105_before", 1136),
    ("ice_card", "20261031-165514", "t0086_before", 1136),
    ("ice_card", "20261031-134157", "t0212_before", 1004),
    ("ice_card", "20261031-144722", "t0205_before", 560),
    ("ice_card", "20261031-185210", "t0202_before", 684),
    ("ice_card", "20261031-233208", "t0203_before", 604),
    ("ice_card", "20261101-105146", "t0252_before", 536),
    ("ice_card", "20261101-122050", "t0294_before", 992),
    ("ice_card", "20261031-134157", "t0213_before", 1004),
    ("ice_card", "20261031-144722", "t0206_before", 560),
    ("skull_card", "20261031-165514", "t0206_before", 836),
    ("mirror_card", "20261028-155154", "t0045_before", 528),
    ("mirror_card", "20261028-155154", "t0046_before", 528),
    ("mirror_card", "20261028-155955", "t0047_before", 936),
    ("mirror_card", "20261028-155955", "t0050_before", 888),
    ("mirror_card", "20261028-155955", "t0054_before", 840),
    ("mirror_card", "20261028-160740", "t0020_before", 636),
    ("mirror_card", "20261029-045856", "t0043_before", 1172),
    ("mirror_card", "20261029-045856", "t0050_before", 1172),
    ("mirror_card", "20261029-045856", "t0052_before", 1172),
    ("mirror_card", "20261030-213120", "t0137_before", 1036),
    ("mirror_card", "20261030-213120", "t0222_before", 840),
    ("mirror_card", "20261030-213120", "t0235_before", 840),
    ("mirror_card", "20261031-080407", "t0210_before", 936),
    ("mirror_card", "20261031-080407", "t0213_before", 936),
    ("mirror_card", "20261031-080407", "t0218_before", 888),
    ("mirror_card", "20261031-154544", "t0570_before", 656),
    ("mirror_card", "20261031-165514", "t0099_before", 1136),
    ("mirror_card", "20261031-165514", "t0307_before", 1400),
    ("mirror_card", "20261031-165514", "t0328_before", 1496),
    ("mirror_card", "20261031-185210", "t0128_before", 864),
    ("mirror_card", "20261031-233208", "t0166_before", 1112),
    ("mirror_card", "20261031-233208", "t0169_before", 1164),
    ("mirror_card", "20261031-233208", "t0172_before", 1112),
    ("mirror_card", "20261101-021522", "t0225_before", 908),
    ("mirror_card", "20261101-021522", "t0238_before", 1008),
    ("mirror_card", "20261101-021522", "t0250_before", 908),
    ("mirror_card", "20261101-034756", "t0196_before", 1356),
    ("mirror_card", "20261101-053431", "t0226_before", 1284),
    ("mirror_card", "20261101-053431", "t0237_before", 1184),
    ("mirror_card", "20261101-053431", "t0244_before", 1136),
    ("mirror_card", "20261101-092253", "t0259_before", 720),
    ("mirror_card", "20261101-092253", "t0268_before", 672),
    ("mirror_card", "20261101-092253", "t0279_before", 720),
    ("mirror_card", "20261101-122050", "t0201_before", 864),
    ("mirror_card", "20261101-122050", "t0211_before", 964),
    ("mirror_card", "20261101-122050", "t0215_before", 1012),
    ("mirror_card", "20261101-135625", "t0368_before", 1268),
    ("mirror_card", "20261101-135625", "t0379_before", 1412),
    ("mirror_card", "20261101-135625", "t0401_before", 1316),
    ("mirror_card", "20261101-164934", "t0060_before", 596),
    ("mirror_card", "20261101-164934", "t0254_before", 1336),
    ("mirror_card", "20261101-164934", "t0578_before", 1360),
    ("mirror_card", "20261102-065902", "t0742_before", 964),
    ("mirror_card", "20261102-065902", "t0744_before", 964),
    ("mirror_card", "20261102-065902", "t0745_before", 964),
]


def window(rgb: np.ndarray, y: int, x: int, rows: int, cols: int) -> np.ndarray:
    a = rgb[y:y + rows * GRID, x:x + cols * GRID, :3].astype(float)
    return a.reshape(rows, GRID, cols, GRID, 3).mean(axis=(1, 3))


def card_mask(rows: int, cols: int) -> np.ndarray:
    m = np.ones((rows, cols), bool)
    for r, c in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)):
        m[r, c] = False
    return m


def largest_piece(m: np.ndarray) -> np.ndarray:
    seen = np.zeros_like(m)
    best = np.zeros_like(m)
    for r0, c0 in zip(*np.nonzero(m)):
        if seen[r0, c0]:
            continue
        piece, stack = [], [(r0, c0)]
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            piece.append((r, c))
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < m.shape[0] and 0 <= cc < m.shape[1] and m[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        if len(piece) > best.sum():
            best = np.zeros_like(m)
            for r, c in piece:
                best[r, c] = True
    return best


def template(blocks: list, geo: tuple) -> tuple[np.ndarray, np.ndarray]:
    y, rows, cols, shape, (r0, r1, c0, c1) = geo
    B = np.stack(blocks)
    ref = np.median(B, axis=0)
    gains = (B[:, r0:r1, c0:c1].reshape(len(B), -1, 3).mean(axis=1) + 1) / (ref[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0) + 1)
    N = B / gains[:, None, None, :]
    med = np.median(N, axis=0)
    if shape == "card" or len(B) < 2:
        return med, card_mask(rows, cols)
    spread = np.median(np.abs(N - med).max(axis=3), axis=0)
    m = spread <= SPREAD_MAX
    m[(FLOOR_EDGE_Y - y) // GRID:] = False
    return med, largest_piece(m)


def build(root: Path) -> dict:
    out, report = {}, {}
    for sid, geo in GEOMETRY.items():
        y, rows, cols = geo[:3]
        blocks = []
        for s, run, frame, x in SAMPLES:
            if s != sid:
                continue
            f = root / "runs" / run / "frames" / f"{frame}.png"
            if not f.exists():
                continue
            rgb = normalize(np.asarray(Image.open(f).convert("RGB")))
            blocks.append(window(rgb, y, x, rows, cols))
        if not blocks:
            continue
        t, m = template(blocks, geo)
        score = ncc(t, m, np.stack(blocks))[0]
        keep = [b for b, sc in zip(blocks, score) if sc >= KEEP_MIN]
        if len(keep) < len(blocks) and keep:
            t, m = template(keep, geo)
        out[sid] = (t, m, y)
        report[sid] = {"samples": len(blocks), "kept": len(keep), "mask_blocks": int(m.sum())}
    return out, report


def main(argv: list[str]) -> None:
    root = Path(argv[0]) if argv else Path(".")
    out, report = build(root)
    arrays = {"sids": np.array(sorted(out))}
    for sid, (t, m, y) in out.items():
        arrays[f"{sid}_t"] = t.astype(np.float32)
        arrays[f"{sid}_m"] = m
        arrays[f"{sid}_y"] = np.array(y)
    np.savez_compressed(ASSET, **arrays)
    for sid, r in sorted(report.items()):
        print(f"{sid}: {r['kept']} of {r['samples']} samples, {r['mask_blocks']} mask blocks")
    print(f"wrote {ASSET}")


if __name__ == "__main__":
    main(sys.argv[1:])
