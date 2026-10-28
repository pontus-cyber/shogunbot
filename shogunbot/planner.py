"""Expectimax planner over the Stage 1 rules engine.

Scoring weights live in WEIGHTS so a run log can record exactly which
weights produced a decision, and tuning between runs is a config change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import QUEUE_MAX, State
from .rules import SPEC, _first_unit_ahead, _furthest_unit_ahead, legal_hero_actions, step, threats

WEIGHTS: dict[str, float] = {
    "hero_hp": 1000.0,           # damage-first [user]: one HP outweighs any kill combo
    "enemy_hp": -25.0,
    "enemy_alive": -40.0,
    "wave_clear": 150.0,
    "death": -100000.0,
    "threat_dmg": -45.0,         # per point of damage an armed enemy would deal right now
    "open_charge_lane": -30.0,   # loaded Charger with a clear lane to the hero
    "tile_ready": 1.0,           # x each ready tile's catalogue ready_value (Swirl 8, Arrow 6)
    "swap_ready": 6.0,
    "queued_tile": 12.0,         # loaded tiles are tempo; must beat the lost "ready" bonus
                                 # or the bot never loads while waiting [run 20261028-053637]
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
    q, ch = h.queue, h.charge
    if more_waves and len(alive) <= SPAWN_THRESHOLD:
        # the next wave arrives now: arriving enemies meet whatever is recharged
        unready = sum(1 for lb, cd in h.kit.cooldowns if lb not in q and ch.get(lb, 0) < cd)
        terms["spawn_unready"] = w["spawn_unready"] * (unready + (0 if h.swap_ready else 1))
    terms["threat_dmg"] = w["threat_dmg"] * sum(t["dmg"] for t in threats(s))
    lanes = 0
    for e in alive:
        if e.kind == "charger" and e.armed and _first_unit_ahead(s, e.x, e.facing, e.uid) is h:
            lanes += 1
    terms["open_charge_lane"] = w["open_charge_lane"] * lanes
    terms["tiles_ready"] = w["tile_ready"] * sum(rv for lb, cd, rv in h.kit.ready_values
                                                 if lb not in q and ch.get(lb, 0) >= cd)
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
    full = list(h.queue) + h.ready_labels()
    potential = _queue_damage(s, full[:QUEUE_MAX])
    terms["lethal_ready"] = w["lethal_ready"] * sum(1 for e in alive if potential.get(e.uid, 0) >= e.hp)
    at_edge = h.x in (0, s.cells - 1)
    adjacent = any(abs(e.x - h.x) == 1 for e in alive)
    if at_edge and alive:
        terms["at_edge"] = w["at_edge"]
    if at_edge and adjacent and not h.swap_ready:
        terms["cornered_no_swap"] = w["cornered_no_swap"]
    return sum(terms.values()), terms


def _queue_damage(s: State, queue: list[str]) -> dict[int, int]:
    """Damage per enemy uid if this queue fired now from the hero's cell.

    A quick estimate for scoring: tiles fire in order, units already brought
    to 0 HP by earlier tiles are passed over, the hero does not move (dashes
    strike their first target in place) and curses are ignored."""
    h = s.hero
    dmg: dict[int, int] = {}
    if not queue:
        return dmg
    hp = {e.uid: e.hp for e in s.enemies if e.hp > 0}
    byl = h.kit.by_label
    hx, f = h.x, h.facing
    for label in queue:
        kt = byl.get(label)
        if kt is None:
            continue
        d = kt.damage
        k, offs, dr, rng, cannot_kill = SPEC[kt.tile]
        st = f * dr
        hits = []
        if k == "cells":
            for off in offs:
                e = s.enemy_at(hx + f * off)
                if e is not None and hp[e.uid] > 0:
                    hits.append(e)
        elif k == "first" or k == "dash_strike":
            nx = hx + st
            while 0 <= nx < s.cells and (rng is None or abs(nx - hx) <= rng):
                e = s.enemy_at(nx)
                if e is not None and hp[e.uid] > 0:
                    hits.append(e)
                    break
                nx += st
        elif k == "furthest":
            u = _furthest_unit_ahead(s, hx, st)
            if u is not None and hp.get(getattr(u, "uid", None), 0) > 0:
                hits.append(u)
        elif k == "dash_through":
            nx = hx + st
            while s.in_bounds(nx) and s.unit_at(nx) is None:
                nx += st
            while s.in_bounds(nx) and (e := s.enemy_at(nx)) is not None and hp[e.uid] > 0:
                hits.append(e)
                nx += st
            if not s.in_bounds(nx):
                hits = []
        elif k == "damaged":
            hits = [e for e in s.alive() if 0 < hp[e.uid] < e.max_hp]
        for e in hits:
            dd = min(d, hp[e.uid] - 1) if cannot_kill else d
            if dd <= 0:
                continue
            dmg[e.uid] = dmg.get(e.uid, 0) + dd
            hp[e.uid] -= dd
    return dmg


@dataclass
class Decision:
    action: str
    value: float
    candidates: list[dict] = field(default_factory=list)  # [{action, value, p_damage, exp_damage}]
    predicted: list[dict] = field(default_factory=list)   # top outcomes of the chosen action
    depth: int = 2
    nodes: int = 0


GAMMA = 0.97  # future damage counts almost as much as damage now [user: HP is the run's most valuable resource]
# tie-break order when values are equal: commit, then load (kit order), then hold,
# then reposition, turn last
PRIORITY = {"attack": 0, "wait": 3, "left": 4, "right": 4, "turn": 5}


def _priority(s: State, a: str) -> float:
    if a.startswith("queue:"):
        labels = s.hero.kit.labels
        lb = a[6:]
        if lb not in labels:
            return 9
        return 1 + labels.index(lb) / len(labels)   # in kit order, between attack and wait
    return PRIORITY.get(a, 9)


def _action_value(s: State, a: str, depth: int, w: dict, stats: dict):
    v = 0.0
    p_dmg = 0.0
    exp_dmg = 0.0
    outs = step(s, a, validate=False)   # a comes from legal_hero_actions
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
    cands.sort(key=lambda c: (-c["value"], _priority(s, c["action"])))
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


LIVE_BUDGET_S = 3.0   # a third tile triples planning time (346 -> 827-1000 ms on logged boss states)


def plan_live(s: State, w: dict[str, float] = WEIGHTS) -> Decision:
    """Live planning. With the starting two tiles: plan(s, auto_depth(s)), as
    before. With more tiles the tree grows, so the search deepens one ply at a
    time and stops before a ply that would overrun LIVE_BUDGET_S (estimated
    from how much the last ply grew)."""
    import time
    target = auto_depth(s)
    if len(s.hero.kit) <= 2:
        return plan(s, target, w)
    t0 = time.time()
    depth = min(3, target)
    d = plan(s, depth, w)
    prev, growth = time.time() - t0, 6.0
    while depth < target:
        if (time.time() - t0) + prev * growth > LIVE_BUDGET_S:
            break
        t1 = time.time()
        d2 = plan(s, depth + 1, w)
        cur = time.time() - t1
        growth = max(2.0, cur / max(prev, 1e-3))
        prev, depth, d = cur, depth + 1, d2
    return d
