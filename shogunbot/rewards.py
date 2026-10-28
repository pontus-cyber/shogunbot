"""Reward choices by planner value.

Every option on a reward screen (each offered tile, each tile an upgrade
could go on, and skipping) becomes a kit. Each kit plays the rest of the
stage in the simulator (selfplay, the planner against the rules engine) on
the same fixed seeds, so the comparison is paired and repeatable, and the
kit with the best expected result is taken. A result is the HP left at the
end of the stage, or -HERO_MAX_HP for a death: damage-first, like the
planner [user].

After the boss the next region is not simulated yet, so Bamboo Grove II and
the boss stand in for "the fights ahead".
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from .catalogue import CAT
from .model import HERO_MAX_HP, Kit, KitTile
from .selfplay import run_once

SEEDS = tuple(range(9000, 9048))  # fixed: the same 48 simulated runs score every option
DEPTH = 3                         # planner depth inside the valuation (live play uses 4-6). Depth 2
                                  # undervalued the dash tiles against depth 3 [valuation study 2026-10-28]
NEXT_LEVEL = {"bamboo-1": "bamboo-2", "bamboo-2": "boss", "boss": "bamboo-2"}
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))


@dataclass
class Option:
    name: str                   # 'skip', 'take:<tile>', 'upgrade:<label>'
    kit: Kit
    value: float = 0.0
    wins: int = 0
    detail: dict = field(default_factory=dict)


STALL_PENALTY = 3.0   # a simulated run that hit the turn limit alive: HP kept, minus this


def score(res) -> float:
    if res.win:
        return float(res.hp)
    if res.timeout:
        return float(res.hp) - STALL_PENALTY
    return -float(HERO_MAX_HP)


def value(kit: Kit, start: str, hp: int, seeds=SEEDS, depth: int = DEPTH) -> tuple[float, int]:
    """(mean score, wins) of this kit over the seeds, from level `start` with `hp`."""
    rs = [run_once(s, depth, kit=kit, start=start, hp=hp) for s in seeds]
    return sum(score(r) for r in rs) / len(rs), sum(r.win for r in rs)


def upgraded(kt: KitTile, up) -> KitTile | None:
    """The tile after a shrine upgrade, or None when it cannot take it."""
    if up.removes_tile or up.unknown_effect:
        return None
    if up.uses_level and kt.used >= kt.levels:
        return None
    if up.kind == "enchant":
        if kt.enchant is not None:
            return None   # one enchantment per tile [wiki]
        return KitTile(kt.label, kt.tile, kt.damage, max(0, kt.cooldown + up.cooldown), up.id,
                       kt.levels, kt.used + 1)
    return KitTile(kt.label, kt.tile, max(0, kt.damage + up.damage), max(0, kt.cooldown + up.cooldown),
                   kt.enchant, kt.levels + up.slots, kt.used + (1 if up.uses_level else 0))


def tile_choice_options(kit: Kit, offers: list[dict]) -> list[Option]:
    opts = [Option("skip", kit)]
    for i, o in enumerate(offers):
        t = o.get("tile")
        if t is None or t not in CAT.tiles or not CAT.tile(t).sim:
            continue      # unknown or not simulated: never taken blind
        cat = CAT.tile(t)
        dmg = o["damage"] if o.get("damage") is not None else cat.damage
        cd = o["cooldown"] if o.get("cooldown") is not None else cat.cooldown
        new = kit.add(t, damage=dmg, cooldown=cd, levels=o.get("levels") or 3, used=o.get("used") or 0)
        opts.append(Option(f"take:{t}", new, detail={"offer": i}))
    return opts


def upgrade_options(kit: Kit, upgrade_id: str) -> list[Option]:
    opts = [Option("skip", kit)]
    up = CAT.upgrade(upgrade_id)
    for kt in kit:
        new = upgraded(kt, up)
        if new is not None:
            opts.append(Option(f"upgrade:{kt.label}", kit.replace(kt.label, new), detail={"label": kt.label}))
    return opts


def _job(args):
    kit_list, start, hp, seed, depth = args
    r = run_once(seed, depth, kit=Kit.from_list(kit_list), start=start, hp=hp)
    return score(r), r.win


def _score_all(opts: list[Option], start: str, hp: int, seeds, depth: int, workers: int) -> None:
    jobs = [(o.kit.to_list(), start, hp, s, depth) for o in opts for s in seeds]
    results = None
    if workers > 1 and len(jobs) > len(seeds):
        try:
            # spawn: the same on Windows (the PC) and in tests
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
                results = list(ex.map(_job, jobs, chunksize=max(1, len(jobs) // (workers * 4))))
        except Exception:
            results = None   # fall back to one process
    if results is None:
        results = [_job(j) for j in jobs]
    n = len(seeds)
    for i, o in enumerate(opts):
        part = results[i * n:(i + 1) * n]
        o.value = round(sum(v for v, _ in part) / n, 3)
        o.wins = sum(w for _, w in part)


def decide(opts: list[Option], level: str, hp: int, seeds=SEEDS, depth: int = DEPTH,
           workers: int = WORKERS) -> tuple[Option, list[Option], float]:
    """Score every option on the same seeds; the best one wins. On a tie the
    reward is taken (it is free), and among takes the first offered."""
    t0 = time.time()
    start = NEXT_LEVEL.get(level, "bamboo-2")
    hp = max(1, min(HERO_MAX_HP, hp))
    if len(opts) > 1:
        _score_all(opts, start, hp, tuple(seeds), depth, workers)
    ranked = sorted(opts, key=lambda o: (-o.value, o.name == "skip"))
    return ranked[0], opts, round(time.time() - t0, 1)
