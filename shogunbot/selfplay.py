"""Offline self-play: the planner against the simulator.

Zero-cost bench, and the engine behind reward valuation (rewards.py).
Level sizes, enemy pools and HP come from the catalogue (data/regions.json,
data/enemies.json). Since 0.22 the waves follow the generator learned from the
recordings (data/waves.json, shogunbot.levelgen): each wave's units by level and
wave number, spawn cells weighted by distance from the hero, facing by kind, and
the next wave at once when no enemy is left or, with one left, by the learned
per-turn chance (for sure on the sixth turn in a row). GENERATOR = "handset" plays
the 0.21 rules (1-3 random units, next wave at 0 alive or at 1 after a wave of 3+).

Common random numbers: everything about a run's opposition is fixed by its
seed alone, so two kits played on the same seed meet the same fight:
  - each wave's enemies, and the draws that settle their cells and facings
    when they come, are made up front from the seed;
  - enemy choices (telegraphs, preset outcomes) on turn t of a level use a
    random stream seeded by (seed, level, t), whatever happened before, and
    whether the next wave comes on turn t a stream of its own (_spawn_rng).
The only differences between two kits' runs then come from how the hero
plays them.

Scenarios:
  stage1  the rest of Stage 1 from a given level (Bamboo Grove I, II, boss)
  stress  harder fights than Stage 1, so reward options separate: a Bamboo
          Grove II board with Day 5+ enemy HP [wiki] and 7 waves (Stage 1 has 5)
          drawn from its learned templates, then Daisuke at Day 5+ HP (20) [wiki].
          On the valuation's 48 seeds (depth 3, full HP; score = HP left - 0.02
          per turn, -10 for a death): Swirl+Arrow wins 46 and scores 5.64,
          Swirl+Arrow+Spear wins 48 and scores 7.54 (0.22). With the hand-set
          generator (0.16, 16 seeds): Swirl+Arrow won 12, Swirl+Arrow+Spear 16

    python -m shogunbot selfplay -n 200 --depth 3
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .catalogue import CAT
from .model import BOSS_KINDS, DEFAULT_KIT, ENEMY_KINDS, HERO_MAX_HP, Enemy, Hero, Kit, State
from .planner import STALL_TURNS, WEIGHTS, auto_depth, plan, progress_key, stall_budget, stall_weights
from . import behaviour, levelgen, rules
from .rules import ENEMY_TILE_DAMAGE, _hits_hero, enemy_options, step

_INTENT = {"turn": "turn", "queue": "queue", "fire": "fire", "wait": "hold"}


def telegraph(s: State, rng: random.Random) -> None:
    """Show the planner what the game shows: every enemy's next action.

    The game draws each enemy's coming action before the hero moves, so live
    the first enemy phase is known. Here each enemy's action is drawn from the
    learned behaviour model (behaviour.py; the 0.16 preset table when
    GAME_BEHAVIOUR is "presets") on the board as it stands, and written as
    its intent; step then plays it."""
    for e in s.alive():
        e.intent = None
        e.primed = (e.armed and e.wind >= 1 and ENEMY_TILE_DAMAGE.get(e.queue[0], 0) > 0
                    and _hits_hero(s, e, e.queue[0]))
    for e in sorted(s.alive(), key=lambda e: e.x):
        if e.frozen > 0:
            continue
        drawn = behaviour.intents(s, e) if GAME_BEHAVIOUR == "learned" else None
        if drawn:
            r, acc, it = rng.random(), 0.0, drawn[-1][1]
            for p, i in drawn:
                acc += p
                if r <= acc:
                    it = i
                    break
            e.intent = it
            continue
        opts = enemy_options(s, e, presets=True)
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
        if e.kind in BOSS_KINDS and act[0] == "queue":
            e.intent = "queue"   # a boss's tile type is not telegraphed; step draws it again
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
SPAWN_WHEN_ALIVE_AT_MOST = 1  # handset rules: next wave when <= this many remain ([web] 1-2)
GENERATOR = "learned"         # "learned": data/waves.json; "handset": the 0.21 rules, for comparisons
SCENARIOS = ("stage1", "stress")
GAME_BEHAVIOUR = "learned"   # how simulated enemies choose; "presets" = the 0.16 tables. The planner's
                             # own model of them is rules.BEHAVIOUR, so the two can be compared
STRESS_WAVES = 7


def enemy_hp(kind: str, day5: bool) -> int:
    e = CAT.enemy(kind)
    return (e.hp_day5 if day5 and e.hp_day5 else e.hp) or ENEMY_KINDS[kind][0]


@dataclass
class Result:
    win: bool
    reached: str
    hp: int
    turns: int
    timeout: bool = False   # turn limit hit with the hero alive (the planner stalled)


@dataclass
class LevelPlan:
    """One level's opposition, fixed by the seed. waves: per wave, the enemies as
    (kind, a uniform draw for the cell, a uniform draw for the facing); the cell and
    facing are settled at spawn time from those draws, where the hero stands then.
    learned: the data/waves.json level the waves follow."""
    name: str
    cells: int
    hero_x: int
    waves: list
    day5: bool = False
    boss: bool = False
    learned: str = ""


def plan_levels(seed: int, start: str = "bamboo-1", scenario: str = "stage1") -> list[LevelPlan]:
    """The levels a run plays, drawn from the seed only (common random numbers)."""
    g = random.Random(seed)
    out = []

    def wave(level, i, pool, lo, hi):
        templates = levelgen.wave_templates(level, i) if GENERATOR == "learned" else []
        if templates:
            r = g.random() * sum(n for _, n in templates)
            kinds = list(templates[-1][0])
            for units, n in templates:
                r -= n
                if r <= 0:
                    kinds = list(units)
                    break
            g.shuffle(kinds)
        else:
            kinds = [g.choice(pool) for _ in range(g.randint(lo, hi))]
        return [(k, g.random(), g.random()) for k in kinds]

    if scenario == "stress":
        name, cells, hx, nwaves, pool = LEVELS[-1]
        out.append(LevelPlan("stress-" + name, cells, hx, [wave(name, i + 1, pool, 2, 3) for i in range(STRESS_WAVES)],
                             True, learned=name))
        out.append(LevelPlan("stress-boss", _BOSS_LV["cells"], _BOSS_LV["hero_cell"], [], True, True))
        return out
    started = False
    for name, cells, hx, nwaves, pool in LEVELS:
        started = started or name == start
        waves = [wave(name, i + 1, pool, 1, 3) for i in range(nwaves)]
        if started:
            out.append(LevelPlan(name, cells, hx, waves, learned=name))
    out.append(LevelPlan("boss", _BOSS_LV["cells"], _BOSS_LV["hero_cell"], [], False, True))
    return out


def _spawn(s: State, wave: list, uid0: int, day5: bool, learned: str = "") -> list[Enemy]:
    """Place a wave: each unit on a free cell drawn with the learned weight for its distance
    from the hero (a cell next to him about a third as often), facing him with the learned
    chance for its kind; at most (cells + 1) / 2 enemies alive [wiki]. The handset
    generator draws cells uniformly and faces 60% toward the hero."""
    cap = (s.cells + 1) // 2
    out = []
    for i, (k, r_cell, r_face) in enumerate(wave):
        if len(s.alive()) + len(out) >= cap:
            break
        taken = {s.hero.x} | {e.x for e in s.alive() + out}
        free = [c for c in range(s.cells) if c not in taken]
        if not free:
            break
        ws = [levelgen.cell_weight(learned, abs(c - s.hero.x)) if learned and GENERATOR == "learned" else 1.0
              for c in free]
        r, x = r_cell * sum(ws), free[-1]
        for c, wgt in zip(free, ws):
            r -= wgt
            if r <= 0:
                x = c
                break
        toward = 1 if s.hero.x > x else -1
        p = levelgen.p_toward(k) if learned and GENERATOR == "learned" else 0.6
        facing = toward if r_face < p else -toward
        out.append(Enemy(k, x, facing, enemy_hp(k, day5), [], uid=uid0 + i))
    return out


def _spawn_rng(seed: int, level_i: int, t: int) -> random.Random:
    """The next-wave draws, a stream of their own so enemy choices stay on _turn_rng's."""
    return random.Random((seed * 2_000_003 + level_i * 30_011 + t * 7 + 1) & 0xFFFFFFFF)


def _turn_rng(seed: int, level_i: int, t: int) -> random.Random:
    return random.Random((seed * 1_000_003 + level_i * 10_007 + t) & 0xFFFFFFFF)


class _Stall:
    """Turns since an enemy last lost HP, died or appeared. From STALL_TURNS on, the
    planner uses stall weights (planner.stall_weights), as the bot does live
    [user 2026-10-28], and searches one ply deeper than the fixed self-play depth
    (live already searches 4-6 plies)."""
    def __init__(self):
        self.key, self.quiet = None, 0

    def settings(self, s: State, depth: int, w: dict) -> tuple[int, dict]:
        k = progress_key(s)
        self.quiet = 0 if k != self.key else self.quiet + 1
        self.key = k
        if self.quiet < STALL_TURNS:
            return depth, w
        return (depth or auto_depth(s)) + 1, stall_weights(w, s.hero.hp, stall_budget(self.quiet))


def _play_turn(s: State, rng: random.Random, depth: int, w: dict, telegraphs: bool,
               stall: _Stall | None = None, trace: list | None = None) -> State:
    if telegraphs:
        telegraph(s, rng)
    if stall is not None:
        depth, w = stall.settings(s, depth, w)
    d = plan(s, depth or auto_depth(s), w)
    if trace is not None:
        trace.append(_frame(s, d.action))
    outs = step(s, d.action, validate=False)
    r, acc = rng.random(), 0.0
    for p, s2, _ in outs:
        acc += p
        if r <= acc:
            break
    return s2


def _frame(s: State, action: str) -> list:
    """One turn of a recorded fight (run_once(trace=...)), for the live viewer's replay: level,
    wave, board, hero HP, the hero's action, the enemies' telegraphs, the hero's queue."""
    tele = " ".join(f"{e.x}:{e.intent}" for e in s.alive() if e.intent and e.intent != "hold")
    return [s.level, s.wave, s.render(), s.hero.hp, action, tele, " ".join(s.hero.queue)]


def play_level(seed: int, level_i: int, lp: LevelPlan, s: State, depth: int, w: dict, max_turns=200,
               telegraphs: bool = True, trace: list | None = None) -> tuple[State, bool]:
    pending = list(lp.waves)
    uid = 1
    last_size = len(pending[0])
    s.enemies = _spawn(s, pending.pop(0), uid, lp.day5, lp.learned)
    uid += 10
    s.wave = 1
    s.waves_total = len(lp.waves)
    stall = _Stall()
    age = ones = 0          # turns since the wave came; turns in a row with at most one enemy left
    for t in range(max_turns):
        if s.hero.hp <= 0:
            return s, False
        alive_n = len(s.alive())
        if t > 0:
            age += 1
            ones = ones + 1 if alive_n <= 1 else 0
        if GENERATOR == "learned" and lp.learned:
            due = _spawn_rng(seed, level_i, t).random() < levelgen.p_next_wave(alive_n, ones, age)
        else:
            due = alive_n == 0 or (alive_n <= SPAWN_WHEN_ALIVE_AT_MOST and last_size >= 3)
        if pending and due:
            last_size = len(pending[0])
            s.enemies += _spawn(s, pending.pop(0), uid, lp.day5, lp.learned)
            uid += 10
            s.wave += 1
            age = ones = 0
        if not s.alive() and not pending:
            return s, True
        s = _play_turn(s, _turn_rng(seed, level_i, t), depth, w, telegraphs, stall, trace)
    return s, False


def boss(seed: int, level_i: int, lp: LevelPlan, hp: int, depth: int, w: dict, kit: Kit = DEFAULT_KIT,
         telegraphs: bool = True, max_turns: int = 300, trace: list | None = None) -> tuple[State, bool]:
    s = State(lp.cells, Hero(lp.hero_x, 1, hp=hp, kit=kit, shield=bool(kit.fx.get("start_shield"))),
              [Enemy(_BOSS.id, 3, -1, enemy_hp(_BOSS.id, lp.day5), [], uid=99)], level="boss")
    stall = _Stall()
    for t in range(max_turns):
        if s.hero.hp <= 0:
            return s, False
        if not any(e.kind == _BOSS.id for e in s.alive()):
            return s, True
        s = _play_turn(s, _turn_rng(seed, level_i, t), depth, w, telegraphs, stall, trace)
    return s, False


def run_once(seed: int, depth: int = 3, w: dict = WEIGHTS, kit: Kit = DEFAULT_KIT,
             start: str = "bamboo-1", hp: int = HERO_MAX_HP, telegraphs: bool = True,
             scenario: str = "stage1", trace: list | None = None) -> Result:
    """One simulated run with this kit and HP: the rest of Stage 1 from level
    `start`, or the 'stress' scenario. telegraphs=True shows the planner each
    enemy's next action, as the game does. trace, a list, gets every turn
    (_frame) and each level's end; recording changes nothing in the run."""
    turns = 0
    for i, lp in enumerate(plan_levels(seed, start, scenario)):
        if lp.boss:
            s, ok = boss(seed, i, lp, hp, depth, w, kit, telegraphs, trace=trace)
        else:
            s = State(lp.cells, Hero(lp.hero_x, 1, hp=hp, kit=kit, shield=bool(kit.fx.get("start_shield"))), [],
                      level=lp.name)   # Fortress: a shield at every fight's start [wiki]
            s, ok = play_level(seed, i, lp, s, depth, w, telegraphs=telegraphs, trace=trace)
        if trace is not None:
            trace.append(_frame(s, "cleared" if ok else "died" if s.hero.hp <= 0 else "stalled"))
        hp, turns = s.hero.hp, turns + s.turn
        if not ok:
            return Result(False, lp.name, hp, turns, timeout=hp > 0)
    return Result(True, lp.name, hp, turns)


def bench(n: int = 50, depth: int = 3, w: dict = WEIGHTS, seed0: int = 0, kit: Kit = DEFAULT_KIT,
          telegraphs: bool = True, scenario: str = "stage1") -> dict:
    t = time.time()
    res = [run_once(seed0 + i, depth, w, kit, telegraphs=telegraphs, scenario=scenario) for i in range(n)]
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
