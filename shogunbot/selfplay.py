"""Offline Stage 1 self-play: the planner against the simulator.

Zero-cost tuning bench. Wave composition and spawn rules are approximations
of what was observed live (domain model section 3); replace them with logged
data as runs accumulate.

    python -m shogunbot selfplay -n 200 --depth 3
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .model import ENEMY_KINDS, Enemy, Hero, State
from .planner import WEIGHTS, auto_depth, plan
from .rules import step

# (level, cells, hero start cell, waves: list of enemy-kind lists)
LEVELS = [
    ("bamboo-1", 5, 2, 4, ["ashigaru", "tachi"]),
    ("bamboo-2", 7, 3, 5, ["ashigaru", "tachi", "charger"]),
]
SPAWN_WHEN_ALIVE_AT_MOST = 1  # Normal room: next wave when <= this many remain ([web] 1-2)


@dataclass
class Result:
    win: bool
    reached: str
    hp: int
    turns: int


def _spawn(rng: random.Random, s: State, kinds: list[str], uid0: int) -> list[Enemy]:
    cap = (s.cells + 1) // 2
    out = []
    for i, k in enumerate(kinds):
        if len(s.alive()) + len(out) >= cap:
            break
        empties = [x for x in range(s.cells)
                   if x != s.hero.x and all(e.x != x for e in s.alive() + out)]
        if not empties:
            break
        x = rng.choice(empties)
        toward = 1 if s.hero.x > x else -1
        facing = toward if rng.random() < 0.6 else -toward  # spawn-facing conflict: mix both
        out.append(Enemy(k, x, facing, ENEMY_KINDS[k][0], [], uid=uid0 + i))
    return out


def play_level(rng, s: State, waves: list[list[str]], depth: int, w: dict, max_turns=200) -> tuple[State, bool]:
    pending = list(waves)
    uid = 1
    last_size = len(pending[0])
    s.enemies = _spawn(rng, s, pending.pop(0), uid)
    uid += 10
    s.wave = 1
    s.waves_total = len(waves)
    for _ in range(max_turns):
        if s.hero.hp <= 0:
            return s, False
        alive_n = len(s.alive())
        if pending and (alive_n == 0 or (alive_n <= SPAWN_WHEN_ALIVE_AT_MOST and last_size >= 3)):
            last_size = len(pending[0])
            s.enemies += _spawn(rng, s, pending.pop(0), uid)
            uid += 10
            s.wave += 1
        if not s.alive() and not pending:
            return s, True
        d = plan(s, depth or auto_depth(s), w)
        outs = step(s, d.action)
        r, acc = rng.random(), 0.0
        for p, s2, _ in outs:
            acc += p
            if r <= acc:
                break
        s = s2
    return s, False


def boss(rng, hp: int, depth: int, w: dict) -> tuple[State, bool]:
    s = State(5, Hero(1, 1, hp=hp), [Enemy("daisuke", 3, -1, 15, [], uid=99)], level="boss")
    for _ in range(300):
        if s.hero.hp <= 0:
            return s, False
        if not any(e.kind == "daisuke" for e in s.alive()):
            return s, True
        d = plan(s, depth or auto_depth(s), w)
        outs = step(s, d.action)
        r, acc = rng.random(), 0.0
        for p, s2, _ in outs:
            acc += p
            if r <= acc:
                break
        s = s2
    return s, False


def run_once(seed: int, depth: int = 3, w: dict = WEIGHTS) -> Result:
    rng = random.Random(seed)
    hp = 10
    turns = 0
    for name, cells, hx, nwaves, pool in LEVELS:
        waves = [[rng.choice(pool) for _ in range(rng.randint(1, 3))] for _ in range(nwaves)]
        s = State(cells, Hero(hx, 1, hp=hp), [], level=name)
        s, ok = play_level(rng, s, waves, depth, w)
        hp, turns = s.hero.hp, turns + s.turn
        if not ok:
            return Result(False, name, hp, turns)
    s, ok = boss(rng, hp, depth, w)
    return Result(ok, "boss", s.hero.hp, turns + s.turn)


def bench(n: int = 50, depth: int = 3, w: dict = WEIGHTS, seed0: int = 0) -> dict:
    t = time.time()
    res = [run_once(seed0 + i, depth, w) for i in range(n)]
    wins = sum(r.win for r in res)
    reached = {}
    for r in res:
        if not r.win:
            reached[r.reached] = reached.get(r.reached, 0) + 1
    return {"runs": n, "wins": wins, "win_rate": round(wins / n, 3),
            "deaths_by_level": reached,
            "avg_hp_on_win": round(sum(r.hp for r in res if r.win) / max(1, wins), 2),
            "avg_turns": round(sum(r.turns for r in res) / n, 1),
            "seconds": round(time.time() - t, 1)}
