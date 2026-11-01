"""Hero queue bubble templates: perception/assets/hero_icons.npz (0.47).

    python -m shogunbot.perception.build_hero_icons ROOT [RUN ...]   # ROOT holds runs/; RUN: leave these runs out

The hero's bubbles draw his tiles as his hand does, with the tile's damage digit and an enchant's tint. Until 0.46
perceive read them with the enemy bubble icons (icons.npz), which draw several tiles their own way, and with the
gated icons closed (the Shuriken among them): on 1,715 frames of ten runs of 2026-10-31 and 2026-11-01 whose bubble
count matched the queue the bot built, 669 of 3,038 bubbles read as another tile, chiefly the Charge as Spear (118),
the Shuriken as Back Charge (93), the Arrow as Spear (92), the Spear as Arrow (89) and the Swirl as Arrow (56); a
Charge read as Spear when a hand face was unread made the planned queue [spear] where the game had [charge], three
times in run 20261101-105146 (t387, t639, t669: core-requests item 6). Restricted to the kit's tiles 15% still read
wrong.

Labels come from the bot's own clicks: the queue its turns built since its last attack (a click the game refused,
logged as a loop_break 'did not leave the hand' on the next turn, is dropped), on frames whose bubble count equals
that queue. Each slot is cut as read_stack finds it (the frame's shift; the icon at the bubble's centre). Up to
PER_BUCKET samples per (tile, damage digit, enchant), spread over the runs. perceive opens these templates only on
the hero, restricted to his kit's tiles (perceive.classify_hero_icon).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .perceive import (SLOT_PITCH, SLOT_Y0, SLOT_Y1, STACK_SHIFTS, _frame_at, _frame_off_centre, normalize,
                       perceive, plus_sign)

ASSETS = Path(__file__).parent / "assets"
PER_BUCKET = 6     # samples kept per (tile, damage, enchant)
PER_RUN = 2        # ... of which at most this many from one run


def built_queues(run: Path) -> dict[int, tuple[list[str], dict, str | None]]:
    """turn -> (the hero tiles the bot had queued at the turn's start, bottom first; its kit by label; the turn's
    'before' frame)."""
    rows = [json.loads(line) for line in open(run / "turns.jsonl", encoding="utf-8")]
    refused = set()
    ev = run / "events.jsonl"
    if ev.exists():
        for line in open(ev, encoding="utf-8"):
            e = json.loads(line)
            if e.get("event") == "loop_break" and "did not leave the hand" in (e.get("reason") or ""):
                refused.add(e.get("turn"))
    out, q, level = {}, [], None
    for r in rows:
        if r["level"] != level:
            q, level = [], r["level"]
        kit = {k["label"]: k for k in r["state"]["hero"].get("kit") or [] if "label" in k}   # logged since 0.2x
        out[r["turn"]] = (list(q), kit, (r.get("frames") or {}).get("before"))
        a = r["decision"]["action"]
        if a == "attack":
            q = []
        elif a.startswith("queue:") and (r["turn"] + 1) not in refused:
            q.append(a[6:])
    return out


def bubble_crops(rgb: np.ndarray, cx: int) -> list[np.ndarray]:
    """The hero's tile bubbles, bottom first, as read_stack cuts them (25 x 24 px icon interiors)."""
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
        if (interior.astype(int).sum(axis=2) < 150).mean() > 0.35:
            out.append(cv2.resize(interior, (25, 24), interpolation=cv2.INTER_AREA))
        elif plus_sign(rgb, y0, y1, cx) is None:
            break
    return out


def build(root: str | Path, leave_out: tuple[str, ...] = ()) -> dict:
    root = Path(root)
    runs = [p for p in sorted((root / "runs").iterdir()) if (p / "turns.jsonl").exists() and p.name not in leave_out]
    X, K, S = [], [], []
    bucket, per_run = defaultdict(int), defaultdict(int)
    for run in runs:
        for turn, (queue, kit, frame) in sorted(built_queues(run).items()):
            if not queue or not frame or any(lb not in kit for lb in queue):
                continue
            keys = [(kit[lb]["tile"], kit[lb]["damage"], kit[lb].get("enchant")) for lb in queue]
            if all(bucket[k] >= PER_BUCKET or per_run[(k, run.name)] >= PER_RUN for k in keys):
                continue
            path = run / frame.replace("\\", "/")
            if not path.exists():
                continue
            rgb = normalize(np.array(Image.open(path).convert("RGB")))
            p = perceive(rgb, [k["tile"] for k in kit.values()])
            if not p.hero:
                continue
            crops = bubble_crops(rgb, p.hero["cx"])
            if len(crops) != len(queue):
                continue
            for k, crop in zip(keys, crops):
                if bucket[k] >= PER_BUCKET or per_run[(k, run.name)] >= PER_RUN:
                    continue
                bucket[k] += 1
                per_run[(k, run.name)] += 1
                X.append(crop)
                K.append(k[0])
                S.append(f"{run.name}/{frame.replace(chr(92), '/')}")
    np.savez_compressed(ASSETS / "hero_icons.npz", X=np.array(X, dtype=np.uint8), K=np.array(K), S=np.array(S))
    by_tile = defaultdict(int)
    for k in K:
        by_tile[k] += 1
    return {"templates": len(K), "by_tile": dict(by_tile), "runs": len(runs), "left_out": list(leave_out)}


if __name__ == "__main__":
    print(json.dumps(build(sys.argv[1] if len(sys.argv) > 1 else ".", tuple(sys.argv[2:])), indent=1))
