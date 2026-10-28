"""Reward choices: take every free reward, pick the target by simulation.

Rules [user, after run 20261028-082756, where two shrine upgrades were
skipped on noise]:
  1. A pure-gain upgrade (+damage, -cooldown, +upgrade slot, nothing worse)
     is always taken. The simulation only picks which tile gets it.
  2. A trade-off (e.g. +2 damage +3 cooldown, an enchantment that costs
     cooldown) and a tile choice are scored against skipping on the same
     seeded fights, in a scenario harder than the rest of Stage 1 so the
     options separate.
  3. Skip only when every option is a net loss: worse than skipping by more
     than the noise (paired difference + 2 standard errors below zero).
Never offered for a decision: Sacrifice a Tile (gives coins, loses a tile)
and upgrades whose effect is unknown.

Scoring: each option plays the 'stress' scenario (selfplay.py) on the same
48 seeds with common random numbers, so a seed is the same fight for every
option. A run scores the HP left minus 0.02 per turn (faster wins count
when HP ends equal), -10 for a death, HP - 3 for a stalled run.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from .catalogue import CAT
from .model import HERO_MAX_HP, Kit, KitTile
from .selfplay import run_once

SEEDS = tuple(range(9000, 9048))  # fixed: the same 48 simulated fights score every option
DEPTH = 3                         # planner depth inside the valuation (live play uses 4-6). Depth 2
                                  # undervalued the dash tiles against depth 3 [valuation study 2026-10-28]
SCENARIO = "stress"               # harder than the rest of Stage 1, where every option won at ~full HP
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
TURN_COST = 0.02                  # per simulated turn
STALL_PENALTY = 3.0               # a simulated run that hit the turn limit alive: HP kept, minus this
NOISE_Z = 2.0                     # a net loss is worse than skipping by more than 2 standard errors


@dataclass
class Option:
    name: str                   # 'skip', 'take:<tile>', 'upgrade:<label>'
    kit: Kit
    value: float = 0.0          # mean score
    wins: int = 0
    scores: list = field(default_factory=list, repr=False)
    diff: float | None = None   # mean paired difference to skipping
    se: float | None = None     # its standard error
    detail: dict = field(default_factory=dict)

    @property
    def net_loss(self) -> bool:
        return self.diff is not None and self.diff + NOISE_Z * (self.se or 0.0) < 0

    def summary(self) -> dict:
        d = {"name": self.name, "scored": bool(self.scores), "value": self.value, "wins": self.wins}
        if self.diff is not None:
            d.update(diff=self.diff, se=self.se, net_loss=self.net_loss)
        return d


@dataclass
class Decision:
    best: Option
    options: list
    rule: str
    seconds: float = 0.0


def score(res) -> float:
    if res.win:
        return float(res.hp) - TURN_COST * res.turns
    if res.timeout:
        return float(res.hp) - STALL_PENALTY - TURN_COST * res.turns
    return -float(HERO_MAX_HP)


# ---------------------------------------------------------------- upgrades
def pure_gain(up) -> bool:
    """Nothing gets worse: no damage lost, no cooldown added, no tile removed."""
    return (up.kind == "stat" and not up.removes_tile and not up.unknown_effect
            and up.damage >= 0 and up.cooldown <= 0 and up.slots >= 0
            and (up.damage > 0 or up.cooldown < 0 or up.slots > 0))


def offerable(up) -> bool:
    """Upgrades the bot ever takes: not Sacrifice (coins for a tile), not unknown effects."""
    return not up.removes_tile and not up.unknown_effect


def upgraded(kt: KitTile, up) -> KitTile | None:
    """The tile after a shrine upgrade, or None when it cannot take it or it
    would change nothing (e.g. -1 cooldown on a cooldown-0 tile)."""
    if not offerable(up):
        return None
    if up.uses_level and kt.used >= kt.levels:
        return None
    if up.kind == "enchant":
        if kt.enchant is not None:
            return None   # one enchantment per tile [wiki]
        return KitTile(kt.label, kt.tile, kt.damage, max(0, kt.cooldown + up.cooldown), up.id,
                       kt.levels, kt.used + 1)
    new = KitTile(kt.label, kt.tile, max(0, kt.damage + up.damage), max(0, kt.cooldown + up.cooldown),
                  kt.enchant, kt.levels + up.slots, kt.used + (1 if up.uses_level else 0))
    if (new.damage, new.cooldown, new.levels) == (kt.damage, kt.cooldown, kt.levels):
        return None
    return new


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


# ---------------------------------------------------------------- simulation
def _job(args):
    kit_list, hp, seed, depth, scenario = args
    r = run_once(seed, depth, kit=Kit.from_list(kit_list), hp=hp, scenario=scenario, start="bamboo-2")
    return score(r), r.win


def evaluate(opts: list[Option], hp: int, seeds=SEEDS, depth: int = DEPTH, workers: int = WORKERS,
             scenario: str = SCENARIO) -> None:
    """Fill each option's per-seed scores, mean, wins, and its paired
    difference to 'skip' (when skip is among the options)."""
    seeds = tuple(seeds)
    hp = max(1, min(HERO_MAX_HP, hp))
    jobs = [(o.kit.to_list(), hp, s, depth, scenario) for o in opts for s in seeds]
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
        o.scores = [v for v, _ in part]
        o.value = round(sum(o.scores) / n, 3)
        o.wins = sum(w for _, w in part)
    paired(opts)


def paired(opts: list[Option]) -> None:
    base = next((o for o in opts if o.name == "skip" and o.scores), None)
    if base is None:
        return
    for o in opts:
        if o is base or len(o.scores) != len(base.scores):
            continue
        d = [a - b for a, b in zip(o.scores, base.scores)]
        m = sum(d) / len(d)
        var = sum((x - m) ** 2 for x in d) / max(1, len(d) - 1)
        o.diff, o.se = round(m, 3), round(math.sqrt(var / len(d)), 3)


def _best(cands: list[Option]) -> Option:
    """Highest mean; on a tie the first (kit order, or the first offer)."""
    return max(cands, key=lambda o: (o.value, -cands.index(o)))


# ---------------------------------------------------------------- decisions
def decide_upgrade(kit: Kit, upgrade_id: str, hp: int, *, seeds=SEEDS, depth: int = DEPTH,
                   workers: int = WORKERS, scenario: str = SCENARIO, evaluator=None) -> Decision:
    t0 = time.time()
    evaluator = evaluator or evaluate
    up = CAT.upgrade(upgrade_id)
    opts = upgrade_options(kit, upgrade_id)
    skip, cands = opts[0], opts[1:]
    if not offerable(up):
        return Decision(skip, opts, "not a free reward (sacrifice or unknown effect)", 0.0)
    if not cands:
        return Decision(skip, opts, "no tile can take it", 0.0)
    if pure_gain(up):
        if len(cands) == 1:
            return Decision(cands[0], opts, "pure gain: always taken; one tile can take it", 0.0)
        if up.damage == 0 and up.cooldown == 0:
            # only adds an upgrade slot: no fight changes, so no simulation can tell
            # tiles apart; give it to the strongest tile, the one most worth upgrading again
            best = max(cands, key=lambda o: (CAT.tile(kit[o.detail["label"]].tile).ready_value,
                                             -kit[o.detail["label"]].free_levels, -cands.index(o)))
            return Decision(best, opts, "pure gain: always taken; slot goes to the strongest tile", 0.0)
        evaluator(cands, hp, seeds, depth, workers, scenario)
        return Decision(_best(cands), opts, "pure gain: always taken; simulation picked the tile",
                        round(time.time() - t0, 1))
    evaluator(opts, hp, seeds, depth, workers, scenario)
    return _tradeoff(opts, "trade-off", t0)


def decide_tile(kit: Kit, offers: list[dict], hp: int, *, seeds=SEEDS, depth: int = DEPTH,
                workers: int = WORKERS, scenario: str = SCENARIO, evaluator=None) -> Decision:
    t0 = time.time()
    evaluator = evaluator or evaluate
    opts = tile_choice_options(kit, offers)
    if len(opts) == 1:
        return Decision(opts[0], opts, "no offered tile the simulator can play", 0.0)
    evaluator(opts, hp, seeds, depth, workers, scenario)
    return _tradeoff(opts, "tile choice", t0)


def _tradeoff(opts: list[Option], what: str, t0: float) -> Decision:
    """Skip only when every option is a net loss; otherwise take the best option that is
    not one, also when it scored below skipping but within the noise [user, after run
    20261028-082756]. The rule text says which case decided: run 20261028-154317 took
    -1 damage -4 cooldown on Arrow at -0.20 +- 0.29 below skipping under the line 'best
    option that is not a net loss', which read as if it had beaten skipping [user review
    of the 0.20 series]."""
    skip, cands = opts[0], opts[1:]
    secs = round(time.time() - t0, 1)
    keep = [o for o in cands if not o.net_loss]
    if not keep:
        return Decision(skip, opts, f"{what}: every option a net loss, skipped", secs)
    best = _best(keep)
    if best.diff is None:
        return Decision(best, opts, f"{what}: best option that is not a net loss", secs)
    margin = f"{best.diff:+.2f} +- {best.se or 0.0:.2f} HP against skipping"
    if best.diff >= 0:
        return Decision(best, opts, f"{what}: {best.name} scores {margin}, the best option", secs)
    below = "every option scored below skipping; " if all(o.diff is not None and o.diff < 0 for o in cands) else ""
    return Decision(best, opts, f"{what}: {below}{best.name} scores {margin}, a loss within the noise, "
                                f"so it is taken (skip only when every option is a net loss)", secs)
