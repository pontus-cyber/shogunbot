"""Offline Stage 1 self-play: the planner against the simulator.

Zero-cost tuning bench. Level sizes, wave counts and enemy pools come from
the catalogue (data/regions.json); wave composition and spawn rules are
approximations of what was observed live (domain model section 3). M3
replaces them with a generator learned from logs.

    python -m shogunbot selfplay -n 200 --depth 3
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .catalogue import CAT
from .model import DEFAULT_KIT, ENEMY_KINDS, HERO_MAX_HP, Enemy, Hero, Kit, State
from .planner import WEIGHTS, auto_depth, plan
from .rules import ENEMY_TILE_DAMAGE, _hits_hero, enemy_options, step

_INTENT = {"turn": "turn", "queue": "queue", "fire": "fire", "wait": "hold"}


def telegraph(s: State, rng: random.Random) -> None:
    """Show the planner what the game shows: every enemy's next action.

    The game draws each enemy's coming action before the hero moves, so live
    the first enemy phase is known. Here each enemy's action is drawn from its
    preset table on the board as it stands (the same 'yellow before the hero
    acts' rule as step) and written as its intent; step then plays it."""
    for e in s.alive():
        e.intent = None
        e.primed = (e.armed and e.wind >= 1 and ENEMY_TILE_DAMAGE.get(e.queue[0], 0) > 0
                    and _hits_hero(s, e, e.queue[0]))
    for e in s.alive():
        if e.frozen > 0:
            continue
        opts = enemy_options(s, e)
        r, acc, act = rng.random(), 0.0, opts[-1][1]
        for p, a in opts:
            acc += p
            if r <= acc:
                act = a
                break
        if act[0] == "move":
            e.intent = "move-right" if act[1] > 0 else "move-left"
        else:
            e.intent = _INTENT.get(act[0])
        if e.kind == "daisuke" and act[0] == "queue":
            e.intent = "queue"   # his tile type is not telegraphed; step draws it again
    for e in s.alive():
        e.primed = False


def _levels(region: str = "bamboo_grove") -> list[tuple]:
    """(level id, cells, hero start cell, waves, enemy pool) for the wave levels."""
    out, pool = [], []
    for lv in CAT.regions[region]["level_list"]:
        pool = pool + [e for e in lv.get("new_enemies", []) if e not in pool]
        if lv["waves"]:
            out.append((lv["id"], lv["cells"], lv["hero_cell"], lv["waves"], list(pool)))
    return out


LEVELS = _levels()                 # bamboo-1: 5 cells, 4 waves; bamboo-2: 7 cells, 5 waves
_BOSS_LV = CAT.levels["boss"]
_BOSS = CAT.enemy(_BOSS_LV["boss"])
SPAWN_WHEN_ALIVE_AT_MOST = 1  # Normal room: next wave when <= this many remain ([web] 1-2)


@dataclass
class Result:
    win: bool
    reached: str
    hp: int
    turns: int
    timeout: bool = False   # turn limit hit with the hero alive (the planner stalled)


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


def play_level(rng, s: State, waves: list[list[str]], depth: int, w: dict, max_turns=200,
               telegraphs: bool = True) -> tuple[State, bool]:
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
        if telegraphs:
            telegraph(s, rng)
        d = plan(s, depth or auto_depth(s), w)
        outs = step(s, d.action)
        r, acc = rng.random(), 0.0
        for p, s2, _ in outs:
            acc += p
            if r <= acc:
                break
        s = s2
    return s, False


def boss(rng, hp: int, depth: int, w: dict, kit: Kit = DEFAULT_KIT, telegraphs: bool = True) -> tuple[State, bool]:
    s = State(_BOSS_LV["cells"], Hero(_BOSS_LV["hero_cell"], 1, hp=hp, kit=kit),
              [Enemy(_BOSS.id, 3, -1, _BOSS.hp, [], uid=99)], level="boss")
    for _ in range(300):
        if s.hero.hp <= 0:
            return s, False
        if not any(e.kind == _BOSS.id for e in s.alive()):
            return s, True
        if telegraphs:
            telegraph(s, rng)
        d = plan(s, depth or auto_depth(s), w)
        outs = step(s, d.action)
        r, acc = rng.random(), 0.0
        for p, s2, _ in outs:
            acc += p
            if r <= acc:
                break
        s = s2
    return s, False


def run_once(seed: int, depth: int = 3, w: dict = WEIGHTS, kit: Kit = DEFAULT_KIT,
             start: str = "bamboo-1", hp: int = HERO_MAX_HP, telegraphs: bool = True) -> Result:
    """One simulated Stage 1 run from level `start` with this kit and HP.
    telegraphs=True shows the planner each enemy's next action, as the game does."""
    rng = random.Random(seed)
    turns = 0
    started = False
    for name, cells, hx, nwaves, pool in LEVELS:
        started = started or name == start
        waves = [[rng.choice(pool) for _ in range(rng.randint(1, 3))] for _ in range(nwaves)]
        if not started:
            continue
        s = State(cells, Hero(hx, 1, hp=hp, kit=kit), [], level=name)
        s, ok = play_level(rng, s, waves, depth, w, telegraphs=telegraphs)
        hp, turns = s.hero.hp, turns + s.turn
        if not ok:
            return Result(False, name, hp, turns, timeout=hp > 0)
    s, ok = boss(rng, hp, depth, w, kit, telegraphs=telegraphs)
    return Result(ok, "boss", s.hero.hp, turns + s.turn, timeout=not ok and s.hero.hp > 0)


def bench(n: int = 50, depth: int = 3, w: dict = WEIGHTS, seed0: int = 0, kit: Kit = DEFAULT_KIT,
          telegraphs: bool = True) -> dict:
    t = time.time()
    res = [run_once(seed0 + i, depth, w, kit, telegraphs=telegraphs) for i in range(n)]
    wins = sum(r.win for r in res)
    reached = {}
    for r in res:
        if not r.win and not r.timeout:
            reached[r.reached] = reached.get(r.reached, 0) + 1
    return {"runs": n, "wins": wins, "win_rate": round(wins / n, 3),
            "deaths_by_level": reached, "stalls": sum(r.timeout for r in res),
            "avg_hp_on_win": round(sum(r.hp for r in res if r.win) / max(1, wins), 2),
            "avg_turns": round(sum(r.turns for r in res) / n, 1),
            "seconds": round(time.time() - t, 1)}
