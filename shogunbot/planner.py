"""Expectimax planner over the Stage 1 rules engine.

Scoring weights live in WEIGHTS so a run log can record exactly which
weights produced a decision, and tuning between runs is a config change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import HERO_TILES, State
from .rules import _first_unit_ahead, legal_hero_actions, step, threats

WEIGHTS: dict[str, float] = {
    "hero_hp": 1000.0,           # damage-first [user]: one HP outweighs any kill combo
    "enemy_hp": -25.0,
    "enemy_alive": -40.0,
    "wave_clear": 150.0,
    "death": -100000.0,
    "threat_dmg": -45.0,         # per point of damage an armed enemy would deal right now
    "open_charge_lane": -30.0,   # loaded Charger with a clear lane to the hero
    "swirl_ready": 8.0,
    "arrow_ready": 6.0,
    "swap_ready": 6.0,
    "queued_tile": 5.0,          # tiles already loaded are tempo
    "enemy_facing_away": 4.0,
    "cornered_no_swap": -12.0,   # at an edge with swap on cooldown and an enemy adjacent
    "kill": 30.0,                # per kill along the path
    "hp_lost_path": -600.0,      # extra per HP lost along the path (on top of hero_hp)
    "hp_lost_now": -1500.0,      # extra per expected HP lost THIS turn: certain damage now is
                                 # never traded for kills while a zero-damage move exists [user]
    "turn_cost": -4.0,           # per turn spent, so the planner does not dither
    "lethal_loaded": 20.0,       # per enemy the loaded queue would kill if fired now
    "lethal_ready": 6.0,         # per enemy that queue + ready tiles could kill in reach
    "turn_action": -3.0,         # turning costs tempo and can drop a swap line
    "at_edge": -6.0,             # standing on an edge cell while enemies are alive
    "spawn_unready": -25.0,      # per tile not ready when the next wave is about to spawn
}

SPAWN_THRESHOLD = 1  # [web] Normal rooms spawn the next wave at 1-2 alive; HYP: 1


def evaluate(s: State, w: dict[str, float] = WEIGHTS) -> tuple[float, dict]:
    h = s.hero
    terms: dict[str, float] = {}
    if h.hp <= 0:
        return w["death"], {"death": w["death"]}
    alive = s.alive()
    terms["hero_hp"] = w["hero_hp"] * h.hp
    terms["enemy_hp"] = w["enemy_hp"] * sum(e.hp for e in alive)
    terms["enemy_alive"] = w["enemy_alive"] * len(alive)
    more_waves = s.waves_total and s.wave < s.waves_total
    if not alive and not more_waves:
        terms["wave_clear"] = w["wave_clear"]
    if more_waves and len(alive) <= SPAWN_THRESHOLD:
        # the next wave arrives now: arriving enemies meet whatever is recharged
        unready = sum(1 for t in HERO_TILES if not h.ready(t) and t not in h.queue)
        terms["spawn_unready"] = w["spawn_unready"] * (unready + (0 if h.swap_ready else 1))
    terms["threat_dmg"] = w["threat_dmg"] * sum(t["dmg"] for t in threats(s))
    lanes = 0
    for e in alive:
        if e.kind == "charger" and e.armed and _first_unit_ahead(s, e.x, e.facing, e.uid) is h:
            lanes += 1
    terms["open_charge_lane"] = w["open_charge_lane"] * lanes
    terms["swirl_ready"] = w["swirl_ready"] * (h.ready("swirl"))
    terms["arrow_ready"] = w["arrow_ready"] * (h.ready("arrow"))
    terms["swap_ready"] = w["swap_ready"] * h.swap_ready
    terms["queued_tile"] = w["queued_tile"] * len(h.queue)
    away = 0
    for e in alive:
        d = 1 if h.x > e.x else -1
        if e.facing != d:
            away += 1
    terms["enemy_facing_away"] = w["enemy_facing_away"] * away
    # damage potential: what the loaded queue would kill, and what queue+ready tiles could kill
    loaded = _queue_damage(s, h.queue)
    terms["lethal_loaded"] = w["lethal_loaded"] * sum(1 for e in alive if loaded.get(e.uid, 0) >= e.hp)
    full = list(h.queue) + [t for t in HERO_TILES if h.ready(t)]
    potential = _queue_damage(s, full[:3])
    terms["lethal_ready"] = w["lethal_ready"] * sum(1 for e in alive if potential.get(e.uid, 0) >= e.hp)
    at_edge = h.x in (0, s.cells - 1)
    adjacent = any(abs(e.x - h.x) == 1 for e in alive)
    if at_edge and alive:
        terms["at_edge"] = w["at_edge"]
    if at_edge and adjacent and not h.swap_ready:
        terms["cornered_no_swap"] = w["cornered_no_swap"]
    return sum(terms.values()), terms


def _queue_damage(s: State, queue: list[str]) -> dict[int, int]:
    """Damage per enemy uid if this queue fired now (sequential, no kills removed)."""
    h = s.hero
    dmg: dict[int, int] = {}
    hp = {e.uid: e.hp for e in s.alive()}
    for tile in queue:
        d = HERO_TILES[tile][0]
        if tile == "swirl":
            for cx in (h.x + h.facing, h.x - h.facing):
                e = s.enemy_at(cx)
                if e and hp[e.uid] > 0:
                    dmg[e.uid] = dmg.get(e.uid, 0) + d
                    hp[e.uid] -= d
        elif tile == "arrow":
            nx = h.x + h.facing
            while s.in_bounds(nx):
                e = s.enemy_at(nx)
                if e and hp[e.uid] > 0:
                    dmg[e.uid] = dmg.get(e.uid, 0) + d
                    hp[e.uid] -= d
                    break
                nx += h.facing
    return dmg


@dataclass
class Decision:
    action: str
    value: float
    candidates: list[dict] = field(default_factory=list)  # [{action, value, p_damage, exp_damage}]
    predicted: list[dict] = field(default_factory=list)   # top outcomes of the chosen action
    depth: int = 2
    nodes: int = 0


GAMMA = 0.85  # discount per turn: damage now is worse than the same damage later
# tie-break order when values are equal: commit, then hold, then reposition, turn last
PRIORITY = {"attack": 0, "queue:swirl": 1, "queue:arrow": 2, "wait": 3, "left": 4, "right": 4, "turn": 5}


def _action_value(s: State, a: str, depth: int, w: dict, stats: dict):
    v = 0.0
    p_dmg = 0.0
    exp_dmg = 0.0
    outs = step(s, a)
    for p, s2, ev in outs:
        lost = sum(d for _, _, d in ev.hero_damage)
        if lost:
            p_dmg += p
            exp_dmg += p * lost
        bonus = w["kill"] * len(ev.kills) + w["hp_lost_path"] * lost + w["turn_cost"]
        if a == "turn":
            bonus += w["turn_action"]
        v += p * (bonus + GAMMA * _search(s2, depth - 1, w, stats))
    return v, p_dmg, exp_dmg, outs


def _search(s: State, depth: int, w: dict, stats: dict) -> float:
    stats["nodes"] += 1
    if depth == 0 or s.hero.hp <= 0 or not s.alive():
        return evaluate(s, w)[0]
    memo = stats.setdefault("memo", {})
    k = (depth, s.key())
    if k in memo:
        return memo[k]
    v = max(_action_value(s, a, depth, w, stats)[0] for a in legal_hero_actions(s))
    memo[k] = v
    return v


def plan(s: State, depth: int = 3, w: dict[str, float] = WEIGHTS) -> Decision:
    stats = {"nodes": 0}
    cands = []
    for a in legal_hero_actions(s):
        v, p_dmg, exp_dmg, outs = _action_value(s, a, depth, w, stats)
        v += w.get("hp_lost_now", 0.0) * exp_dmg
        cands.append({"action": a, "value": round(v, 2), "p_damage": round(p_dmg, 3),
                      "exp_damage": round(exp_dmg, 3), "_outs": outs})
    cands.sort(key=lambda c: (-c["value"], PRIORITY.get(c["action"], 9)))
    best = cands[0]
    predicted = [
        {"p": round(p, 3), "board": s2.render(), "hero_hp": s2.hero.hp, "events": ev.to_dict(),
         "state": s2.to_dict()}
        for p, s2, ev in best["_outs"][:3]
    ]
    for c in cands:
        c.pop("_outs")
    return Decision(best["action"], best["value"], cands, predicted, depth, stats["nodes"])


def auto_depth(s: State) -> int:
    """Deeper search when the board is small enough to afford it."""
    n = len(s.alive())
    return 6 if n <= 1 else 5 if n == 2 else 4
