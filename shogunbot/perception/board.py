"""Floor-plate detection: finds the arena cells and what stands on them.

All coordinates are for a 1920x1080 client area; frames of other sizes are
resized first (see perceive.normalize).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

PLATE_Y0, PLATE_Y1 = 764, 782     # rows sampled inside the floor plates
MIN_W, MAX_W = 100, 150           # plate width in px
PITCH = 140                       # cell spacing in px
PLATE_W = 128                     # a plate's width; 12 px between plates
BOARD_CELLS = (5, 7, 9)           # level widths [wiki]; 9 cells first seen in Hot Springs and the Castle
                                  # [collect/20261028-190026]
# Moonlit Port's boards are dark: an empty plate's body reads 60-70 on the plate rows (summed RGB),
# against 66 for plate_runs, so its empty cells broke up and the fight read as 'unknown' (210 of the
# 248 frames of its levels in collect/20261028-190026, f0335-f0582), or worse, its bright plates alone
# made a board with holes, or a shorter one (dark_cells, open_end). On the cell grid found from its
# bright plates, 828 empty-plate bodies read 60-70 and 965 cells outside the board below 10; 52 fell
# between 10 and 50 [those frames].
DARK_BODY_MIN = 40
DARK_PLATE_MAX = 100              # a dark plate's body: at most this (the plates beyond a board's end)
GAP_DARK = 20                     # a gap column between dark plates: 2-12 on those frames
GAP_MAX = 40                      # a gap inside a dark board: its darkest pixel at most this
HERO_TEAL = 0.2                   # _tealish share that marks the hero's plate on a dark board
ITEM_X = 180                      # the item column at the screen's left edge (x 80-150) reaches the plate
                                  # rows; a potion there reads like a plate [collect/20261028-190026 f1718]
ROW_TOL = 70                      # neighbouring plates of a board are PITCH apart within this: a missing plate
                                  # makes 280; a spawn flame's glow moved a plate's edge by 12 px [collect/
                                  # 20261029-160032 f0047, Bamboo Grove II], a tooltip's bar over the first
                                  # plate by 64 [f1370, Hot Springs], the item column by 48 [f1893]
EDGE_BODY_MIN = 60                # Moonlit Port III's camera panned a 9-cell board to x0 140, its first plate
EDGE_STD = 25                     # 78 px right of the item column, so the row read as running on under the
                                  # column and the bot saw no board for 90 s [run 20261031-123442
                                  # unknown_1793447769.png, after t225]. That first cell is taken when at
                                  # least EDGE_BODY_MIN px of its body lie right of ITEM_X, the board is then
                                  # 9 wide (nothing can lie further left) and the body is even (std <= EDGE_STD:
                                  # 0.9-2.7 empty and 14 on an enemy there; 43 on the lit ground left of a
                                  # 7-cell Bamboo Grove board, collect/20261029-160032 f0415, which it must not take)


def plate_runs(rgb: np.ndarray) -> list[tuple[int, int]]:
    """[(x0, x1)] of bright horizontal runs on the plate row."""
    band = rgb[PLATE_Y0:PLATE_Y1].astype(np.int16)
    lum = band.sum(axis=2).mean(axis=0)            # per-column brightness
    on = lum > 66
    runs, x = [], 0
    W = len(on)
    while x < W:
        if on[x]:
            x0 = x
            while x < W and on[x]:
                x += 1
            runs.append((x0, x))
        x += 1
    # merge tiny gaps (pixel art noise) then keep plate-sized runs
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 4:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    out = []
    for x0, x1 in merged:
        w = x1 - x0
        if MIN_W <= w <= MAX_W:
            out.append((x0, x1))
        elif w > MAX_W:  # two plates touching: split by pitch
            k = round(w / PITCH)
            for i in range(k):
                out.append((x0 + i * PITCH, x0 + i * PITCH + PITCH - 12))
    return out


def plate_kind(rgb: np.ndarray, x0: int, x1: int) -> str:
    """'hero' | 'enemy' | 'empty' from the plate's mean colour."""
    patch = rgb[PLATE_Y0:PLATE_Y1, x0 + 8:x1 - 8].reshape(-1, 3).astype(float)
    r, g, b = patch.mean(axis=0)
    # hero plate: teal-grey stone ends (145,168,157)/(97,127,121). Checked first,
    # because right after a swap its tick box is all red and the mean colour
    # no longer reads blue [run 20261028-044407: two stalls after swapping]
    p = patch.astype(int)
    teal = ((np.abs(p - (145, 168, 157)).max(axis=1) < 16) | (np.abs(p - (97, 127, 121)).max(axis=1) < 16)).mean()
    if teal > 0.12:
        return "hero"
    if r > g + 40 and r > b + 40:
        return "enemy"
    if b > g - 5 and b > 70:
        return "hero"
    return "empty"


def _phase(lum: np.ndarray, runs: list[tuple[int, int]]) -> int | None:
    """The cell grid's phase (x0 mod PITCH) from the plates that are bright enough to read as runs and
    from the dark gaps between plates: 12 px of near-black columns with plate on both sides."""
    votes: Counter = Counter()
    for x0, x1 in runs:
        if PLATE_W - 8 <= x1 - x0 <= PLATE_W + 4:
            votes[x0 % PITCH] += 2
    dark = lum < GAP_DARK
    x, W = 1, lum.shape[0]
    while x < W - 1:
        if dark[x] and not dark[x - 1]:
            x1 = x
            while x1 < W and dark[x1]:
                x1 += 1
            if 10 <= x1 - x <= 14 and x1 < W and lum[x - 4:x].min() >= DARK_BODY_MIN and lum[x1:x1 + 4].min() >= DARK_BODY_MIN:
                votes[(x1 - PITCH) % PITCH] += 1     # the next plate starts where the gap ends
            x = x1
        x += 1
    if not votes:
        return None
    # neighbouring phases are one plate edge read a pixel apart: count them together
    best = max(range(PITCH), key=lambda p: (votes[p] + 0.5 * (votes[(p - 1) % PITCH] + votes[(p + 1) % PITCH]), votes[p]))
    return best if votes[best] else None


def _tealish(rgb: np.ndarray, x0: int) -> float:
    """Share of a plate's pixels in the dark teal of the hero's plate ends as Moonlit Port lights them
    (green and blue above red): 0.25-0.34 on its hero plates, whatever the swap ticks show (red after a
    swap, white in a flash), 0 on its empty and enemy plates [collect/20261028-190026 f0341-f0350]."""
    p = rgb[PLATE_Y0:PLATE_Y1, x0 + 8:x0 + PLATE_W - 8].reshape(-1, 3).astype(np.int16)
    return float(((p[:, 1] > p[:, 0] + 10) & (p[:, 2] > p[:, 0] + 8) & (p[:, 1] > 55)).mean())


def _rows(rgb: np.ndarray, phase: int) -> tuple[np.ndarray, list[int], list[bool]]:
    """The plate-row brightness, the grid cells (x0) of this phase, and which hold a plate's body."""
    lum = rgb[PLATE_Y0:PLATE_Y1].astype(np.int16).sum(axis=2).mean(axis=0)
    grid = list(range(phase, lum.shape[0] - PLATE_W + 1, PITCH))
    present = [x0 >= 1 and _body(lum, x0) >= DARK_BODY_MIN for x0 in grid]
    return lum, grid, present


def _body(lum: np.ndarray, x0: int) -> float:
    """A grid cell's plate-body brightness. A cell that reaches into the item column is judged on its part
    right of ITEM_X: the potions light the plate rows (326-537 at x 90-140), and once the camera had panned
    a 9-cell board 48 px left after the hero stepped right, the grid cell left of the board lay over them,
    read as a plate, and the board as running on under the column, so the bot saw no board for 90 s and
    stopped [run 20261030-115221 stuck.png, after t233, Moonlit Port III; x 150-230 there read 3-13].
    A cell with less than 20 px right of the column holds no plate the bot can see."""
    a, b = max(x0 + 10, ITEM_X), x0 + PLATE_W - 10
    if b - a < 20:
        return 0.0
    return float(lum[a:b].mean())


def _edge_cell(lum: np.ndarray, x0: int, width: int) -> bool:
    """The cell at x0, partly under the item column, is a 9-cell board's first plate (EDGE_BODY_MIN)."""
    a, b = max(x0 + 10, ITEM_X), x0 + PLATE_W - 10
    return width == max(BOARD_CELLS) and b - a >= EDGE_BODY_MIN and float(lum[a:b].std()) <= EDGE_STD


def _gap_dark(band: np.ndarray, x0: int) -> bool:
    """The gap right of the plate at x0 is dark somewhere in its middle (a move arrow lights only its
    top rows; the lighthouse beam and a tooltip light it all) [collect/20261028-190026 f0353, f0402]."""
    return band[:, x0 + PLATE_W + 2:x0 + PITCH - 2].min() <= GAP_MAX


def dark_cells(rgb: np.ndarray, runs: list[tuple[int, int]]) -> list[dict] | None:
    """The cells of a dark board, or None when the read is not certain. The cell grid (140 px apart)
    comes from the bright plates and the dark gaps (_phase); the board is the one unbroken row of grid
    cells whose plate body reads at least DARK_BODY_MIN that has a board's width and exactly one hero
    plate (_tealish at least HERO_TEAL). Cells in the item column (left of ITEM_X) count as unknown: a
    row that reaches them with a plate's body there is not taken. Every gap inside the row must be
    dark (_gap_dark), so a frame where the lighthouse beam or a tooltip lights the plate row stays
    unread (the next one is read) rather than misread."""
    lum = rgb[PLATE_Y0:PLATE_Y1].astype(np.int16).sum(axis=2).mean(axis=0)
    phase = _phase(lum, runs)
    if phase is None:
        return None
    lum, grid, present = _rows(rgb, phase)
    rows, cur = [], []
    for i, (x0, ok) in enumerate(zip(grid, present)):
        if ok and x0 >= ITEM_X:
            cur.append(i)
        else:
            if cur:
                rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)
    boards = []
    for r in rows:
        if r[0] > 0 and present[r[0] - 1]:
            if not _edge_cell(lum, grid[r[0] - 1], len(r) + 1):
                continue                      # it runs on under the item column: its width is unknown
            r = [r[0] - 1] + r
        heroes = [i for i in r if _tealish(rgb, grid[i]) >= HERO_TEAL]
        if len(r) in BOARD_CELLS and len(heroes) == 1:
            boards.append((r, heroes[0]))
    if len(boards) != 1:
        return None
    r, hero = boards[0]
    band = rgb[PLATE_Y0:PLATE_Y1].astype(np.int16).sum(axis=2)
    if not all(_gap_dark(band, grid[i]) for i in r[:-1]):
        return None
    out = _cells(rgb, [(grid[i], grid[i] + PLATE_W) for i in r])
    for c in out:
        c["kind"] = "hero" if c["x0"] == grid[hero] else ("enemy" if c["kind"] == "enemy" else "empty")
    return out


def open_end(rgb: np.ndarray, cs: list[dict]) -> bool:
    """A board read from its bright plates that stops next to a dark plate (its body DARK_BODY_MIN to
    DARK_PLATE_MAX, across a dark gap): Moonlit Port's empty plates beyond the lit ones, so the board
    is wider than read (collect/20261028-190026 f0513: 7 read, 9 there; f1407: 5 read, 7 there).
    The item column is not looked at."""
    band = rgb[PLATE_Y0:PLATE_Y1].astype(np.int16).sum(axis=2)
    lum = band.mean(axis=0)
    for x0, gap_at in ((cs[0]["x0"] - PITCH, cs[0]["x0"] - PITCH), (cs[-1]["x0"] + PITCH, cs[-1]["x0"])):
        if x0 < ITEM_X or x0 + PLATE_W > lum.shape[0]:
            continue
        body = lum[x0 + 10:x0 + PLATE_W - 10].mean()
        if DARK_BODY_MIN <= body <= DARK_PLATE_MAX and _gap_dark(band, gap_at):
            return True
    return False


def _cells(rgb: np.ndarray, runs: list[tuple[int, int]]) -> list[dict]:
    out = []
    for x0, x1 in runs:
        patch = rgb[PLATE_Y0:PLATE_Y1, x0 + 8:x1 - 8].reshape(-1, 3).mean(axis=0)
        out.append({"x0": x0, "x1": x1, "cx": (x0 + x1) // 2,
                    "kind": plate_kind(rgb, x0, x1), "rgb": [int(v) for v in patch]})
    return out


def is_board(cs: list[dict]) -> bool:
    """A board's width, a hero, and no plate missing in the row: Moonlit Port's bright plates alone (its
    enemies' and the hero's, with dark empty plates between them) made boards of 5 with holes in them,
    read as fights of the wrong width (12 frames of collect/20261028-190026, e.g. f0440: 9 cells)."""
    return (len(cs) in BOARD_CELLS and any(c["kind"] == "hero" for c in cs)
            and all(abs(b["x0"] - a["x0"] - PITCH) < ROW_TOL for a, b in zip(cs, cs[1:])))


def cells(rgb: np.ndarray) -> list[dict]:
    """Left-to-right cells: {x0, x1, cx, kind, rgb}. When the bright runs make no board, or one that
    stops next to a dark plate (open_end), the dark board's grid is tried (Moonlit Port, dark_cells);
    if that is not certain either, no cells (the screen reads as 'unknown' and the bot looks again)."""
    runs = plate_runs(rgb)
    out = _cells(rgb, runs)
    if is_board(out) and not open_end(rgb, out):
        return out
    dark = dark_cells(rgb, runs)
    if dark and is_board(dark):
        return dark
    return [] if is_board(out) else out
