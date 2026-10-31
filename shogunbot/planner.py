"""Expectimax planner over the Stage 1 rules engine.

Scoring weights live in WEIGHTS so a run log can record exactly which
weights produced a decision, and tuning between runs is a config change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import QUEUE_MAX, State
from .rules import SPEC, _first_unit_ahead, _furthest_unit_ahead, draw_intents, legal_hero_actions, step, threats

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
SHIELD_HP = 2        # HYP: a shield is worth one hit, about 2 HP (the Wanderer's tiles deal 1-4)

STALL_TURNS = 15     # [user 2026-10-28] turns in which no enemy lost HP, died or appeared
# [user 2026-10-31: "HP is the priority, it is needed in the long run, but ease the lever to 3
# instead of 1"; run 20261031-102644 waited 450 turns at Iwao, where every way forward cost 2 HP]
# the stall budget grows with the quiet turns: (quiet turns, HP the trade may cost)
STALL_STEPS = ((STALL_TURNS, 1), (40, 2), (60, 3))
STALL_MIN_HP = 3     # a trade never takes the hero below this (below it, the 1-HP budget as before)
STALL_STUCK_TURNS = 200   # quiet turns after which the bot ends the run as 'stuck'


def stall_push(candidates: list[dict], action: str, budget: int, hp_weight: float = 1000.0) -> str | None:
    """From a 2-HP stall budget on, attack when the attack line costs at most the budget.

    Run 20261031-102644 (0.36, Iwao, t180-t629): the hero waited at the wall for 450 turns
    with Shuriken, Swirl and Charge loaded. Every turn the search scored waiting highest
    (3781) because its line expected Iwao to walk into reach, but loaded Iwao held for 459
    turns; the attack line (it kills the 1-HP Yari Master and charges Iwao) scored 1415,
    about 2 HP lower. Waiting never makes progress, so once the stall budget allows the
    trade [user 2026-10-31] the attack is taken when it is within budget * hp_weight
    (+ half an HP) of the best line. Returns the action to play instead, or None."""
    if budget < 2 or action == "attack":
        return None
    best = max((c["value"] for c in candidates), default=None)
    att = next((c for c in candidates if c["action"] == "attack"), None)
    if best is None:
        return None
    if att is None:
        # nothing loaded: load the best tile within budget, so the next turn has an attack to push.
        # Run 20261031-180506 (0.41, Moonlit Port II t219-t419): with no tile queued there was no
        # 'attack' line, and the hero walked back and forth with a 3-HP Guardian for 200 turns while
        # queue:arrow#2 (0 expected damage) scored about 220 below 'left' every turn [0.44]
        if action.startswith("queue:"):
            return None
        qs = [c for c in candidates if c["action"].startswith("queue:")
              and c["value"] >= best - hp_weight * (budget + 0.5) and c.get("exp_damage", 0) <= budget]
        return max(qs, key=lambda c: c["value"])["action"] if qs else None
    if att["value"] >= best - hp_weight * (budget + 0.5) and att.get("exp_damage", 0) <= budget:
        return "attack"
    return None


def stall_budget(quiet: int) -> int:
    """HP a stalled fight may trade for progress after `quiet` quiet turns (0: not stalled)."""
    budget = 0
    for turns, hp in STALL_STEPS:
        if quiet >= turns:
            budget = hp
    return budget


def stall_weights(w: dict[str, float], hp: int, budget: int = 1) -> dict[str, float]:
    """Weights for a stalled fight: after STALL_TURNS quiet turns a 1-HP trade that
    makes progress is allowed [user 2026-10-28]. Seen in self-play (seeds 9008 and
    9023, Bamboo Grove II at depth 3): a loaded Charger waits behind the last other
    enemy, killing that enemy lets the Charger hit for 1, and damage-first scoring
    dodged forever. Here 2 damage dealt (one tile hit) outweighs 1 HP lost, immediate
    or later, and any line that drops the hero more than 1 HP below its current HP
    is heavily penalised. A zero-damage kill still scores higher than any trade.
    A hit still waiting at the end of the search (an armed enemy with the hero in
    its footprint, a loaded Charger with a clear lane) counts about as much as a hit
    taken: otherwise every line kills on the last ply, the Charger's answer falls
    past the horizon, and the kill is put off again on every turn [self-play seed
    9008 turn 40, depth 3].
    `budget` (stall_budget) widens the trade to 2 and 3 HP as the stall lasts, never below
    STALL_MIN_HP [user 2026-10-31]; HP still comes first: a cheaper line always scores higher."""
    floor = max(hp - budget, min(hp - 1, STALL_MIN_HP))
    return {**w, "enemy_hp": -1000.0, "hp_lost_now": 0.0, "threat_dmg": -1000.0, "open_charge_lane": -600.0,
            "hp_floor": floor, "below_floor": -5000.0}


def progress_key(s: State) -> tuple:
    """Changes whenever an enemy loses HP, dies or appears: the stall counter's reset."""
    return tuple(sorted((e.kind, e.hp) for e in s.alive()))


def evaluate(s: State, w: dict[str, float] = WEIGHTS) -> tuple[float, dict]:
    h = s.hero
    terms: dict[str, float] = {}
    if h.hp <= 0:
        return w["death"], {"death": w["death"]}
    alive = s.alive()
    terms["hero_hp"] = w["hero_hp"] * h.hp
    terms["enemy_hp"] = w["enemy_hp"] * sum(e.hp for e in alive)
    shields = sum(1 for e in alive if e.shield)
    if shields:
        # a shield takes one hit, about one tile's damage: breaking it is progress (scales with the stall weights)
        terms["enemy_shield"] = w["enemy_hp"] * SHIELD_HP * shields
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
    floor = w.get("hp_floor")
    if floor is not None and h.hp < floor:          # stall weights only
        terms["below_floor"] = w["below_floor"] * (floor - h.hp)
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
    shielded = {e.uid for e in s.enemies if e.hp > 0 and e.shield}
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
            if e.uid in shielded:
                shielded.discard(e.uid)      # the shield takes this hit [wiki: Tiles, Shield]
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
    pv: list[dict] = field(default_factory=list)          # principal variation (explain_decision)
    reason: list[str] = field(default_factory=list)       # best vs second best, in plain words


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


MAX_BRANCHES = 8    # telegraph combinations at the first simulated turn
DEEP_BRANCHES = 3   # ... and at later turns


def _search(s: State, depth: int, w: dict, stats: dict) -> float:
    stats["nodes"] += 1
    if depth == 0 or s.hero.hp <= 0 or not s.alive():
        return evaluate(s, w)[0]
    memo = stats.setdefault("memo", {})
    k = (depth, s.key())
    if k in memo:
        return memo[k]
    # the next turn's telegraphs in full; turns after that only their likeliest few
    # (the boss fight reached 12 s per move with 8 everywhere) [replanning 759 logged turns]
    ply = stats.get("root_depth", depth) - depth
    branches = draw_intents(s, MAX_BRANCHES if ply <= 1 else DEEP_BRANCHES)
    if branches:
        v = sum(p * max(_action_value(sc, a, depth, w, stats)[0] for a in legal_hero_actions(sc))
                for p, sc in branches)
    else:
        v = max(_action_value(s, a, depth, w, stats)[0] for a in legal_hero_actions(s))
    memo[k] = v
    return v


def plan(s: State, depth: int = 3, w: dict[str, float] = WEIGHTS) -> Decision:
    """Best hero action. Live, the screen shows every telegraph; an enemy
    whose telegraph is unknown is averaged over the behaviour model (the
    hero has to commit without seeing it)."""
    stats = {"nodes": 0, "root_depth": depth}
    branches = draw_intents(s) or [(1.0, s)]
    cands = []
    for a in legal_hero_actions(s):
        v = p_dmg = exp_dmg = 0.0
        outs = []
        for pb, sb in branches:
            vb, pd, ed, ob = _action_value(sb, a, depth, w, stats)
            v += pb * vb
            p_dmg += pb * pd
            exp_dmg += pb * ed
            outs += [(pb * p, s2, ev) for p, s2, ev in ob]
        outs.sort(key=lambda t: -t[0])
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
    d = Decision(best["action"], best["value"], cands, predicted, depth, stats["nodes"])
    d._stats, d._weights = stats, w      # the search's memo, for explain_decision
    return d


def auto_depth(s: State) -> int:
    """Deeper search when the board is small enough to afford it."""
    n = len(s.alive())
    return 6 if n <= 1 else 5 if n == 2 else 4


LIVE_BUDGET_S = 3.0   # a third tile triples planning time (346 -> 827-1000 ms on logged boss states)


def plan_live(s: State, w: dict[str, float] = WEIGHTS, explain: bool = False) -> Decision:
    """Live planning: the search deepens one ply at a time up to auto_depth and
    stops before a ply that would overrun LIVE_BUDGET_S (estimated from how
    much the last ply grew). Since 0.17 for every kit: the learned behaviour
    branches more than the old presets [replanning 759 logged turns]."""
    import time
    target = auto_depth(s)
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
    if explain:
        # after the move is chosen, so planning time and decisions are unchanged
        explain_decision(s, d)
    return d


# ------------------------------------------------------------------ explanation (live viewer)
def _best_here(s: State, depth: int, w: dict, stats: dict) -> str | None:
    """The action the search rated best at this node. Child values come from the
    search's memo, so this costs a few rule steps, not a new search."""
    best, best_key = None, None
    for a in legal_hero_actions(s):
        v = _action_value(s, a, depth, w, stats)[0]
        k = (v, -_priority(s, a))
        if best_key is None or k > best_key:
            best, best_key = a, k
    return best


def _verb(action: str) -> str:
    k, _, arg = action.partition(" ")
    if k == "move":
        return "steps left" if arg.startswith("-") else "steps right"
    return {"fire": "fires", "turn": "turns", "queue": f"loads {arg}", "wait": "holds", "hold": "holds",
            "frozen": "is frozen"}.get(k, action)


def principal_variation(s: State, d: Decision, action: str | None = None, max_plies: int = 5):
    """The likely line after `action` (default: the chosen move): at each turn the
    likeliest telegraphs, the move the search rated best there, and the likeliest
    outcome. Returns (plies, final state)."""
    stats, w = getattr(d, "_stats", None), getattr(d, "_weights", WEIGHTS)
    if stats is None:
        return [], s
    pb, cur = max(draw_intents(s) or [(1.0, s)], key=lambda b: b[0])
    a, depth, prob, line = action or d.action, d.depth, pb, []
    while a is not None and len(line) < max_plies:
        p, s2, ev = max(step(cur, a, validate=False), key=lambda o: o[0])
        prob *= p
        by_uid = {e.uid: e for e in cur.enemies}
        line.append({
            "action": a,
            "telegraphs": [{"x": e.x, "kind": e.kind, "intent": e.intent} for e in cur.alive()],
            "enemies": [{"x": by_uid[u].x, "kind": by_uid[u].kind, "does": _verb(act)}
                        for u, act in ev.enemy_actions if u in by_uid],
            "hero_damage": sum(dmg for _, _, dmg in ev.hero_damage),
            "kills": [k for k, _, _ in ev.kills],
            "board": s2.render(), "hero_hp": s2.hero.hp, "p": round(prob, 3)})
        depth -= 1
        cur = s2
        if depth <= 0 or s2.hero.hp <= 0 or not s2.alive():
            break
        ply = stats["root_depth"] - depth
        pb, cur = max(draw_intents(s2, MAX_BRANCHES if ply <= 1 else DEEP_BRANCHES) or [(1.0, s2)],
                      key=lambda b: b[0])
        prob *= pb
        a = _best_here(cur, depth, w, stats)
    return line, cur


# plain words for the score terms: (weight key for counting, words when better, words when worse)
def _pl(n, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


_WORDS = {
    "hp": ("hero_hp", lambda n: f"keeps {_pl(n, 'more HP', 'more HP')}", lambda n: f"keeps {_pl(n, 'less HP', 'less HP')}"),
    "kills": ("enemy_alive", lambda n: f"{_pl(n, 'fewer enemy', 'fewer enemies')} left",
              lambda n: f"{_pl(n, 'more enemy', 'more enemies')} left"),
    "damage": ("enemy_hp", lambda n: f"{n} less enemy HP left", lambda n: f"{n} more enemy HP left"),
    "wave_clear": (None, lambda n: "clears the level", lambda n: "does not clear the level"),
    "threat_dmg": ("threat_dmg", lambda n: f"{n} less damage aimed at you at the end of the line",
                   lambda n: f"{n} more damage aimed at you at the end of the line"),
    "open_charge_lane": ("open_charge_lane", lambda n: "no Charger lane open to you",
                         lambda n: "a Charger lane open to you"),
    "tiles_ready": (None, lambda n: "more tiles ready", lambda n: "fewer tiles ready"),
    "swap_ready": (None, lambda n: "swap still ready", lambda n: "swap used up"),
    "queued_tile": ("queued_tile", lambda n: f"{_pl(n, 'more tile', 'more tiles')} loaded",
                    lambda n: f"{_pl(n, 'fewer tile', 'fewer tiles')} loaded"),
    "enemy_facing_away": ("enemy_facing_away", lambda n: f"{_pl(n, 'more enemy', 'more enemies')} facing away",
                          lambda n: f"{_pl(n, 'fewer enemy', 'fewer enemies')} facing away"),
    "lethal_loaded": ("lethal_loaded", lambda n: f"the loaded tiles would kill {n} more",
                      lambda n: f"the loaded tiles would kill {n} fewer"),
    "lethal_ready": ("lethal_ready", lambda n: f"loaded plus ready tiles could kill {n} more",
                     lambda n: f"loaded plus ready tiles could kill {n} fewer"),
    "at_edge": (None, lambda n: "not on an edge cell", lambda n: "on an edge cell"),
    "cornered_no_swap": (None, lambda n: "not cornered", lambda n: "cornered without a swap"),
    "spawn_unready": ("spawn_unready", lambda n: f"{_pl(n, 'more tile', 'more tiles')} recharged for the next wave",
                      lambda n: f"{_pl(n, 'fewer tile', 'fewer tiles')} recharged for the next wave"),
    "below_floor": (None, lambda n: "stays within the stall budget", lambda n: "goes past the stall budget"),
    "death": (None, lambda n: "survives", lambda n: "dies"),
}


def _line_terms(line: list, leaf: State, cand: dict, w: dict) -> dict:
    t = dict(evaluate(leaf, w)[1])
    hp_points = t.pop("hero_hp", 0.0) + w["hp_lost_path"] * sum(p["hero_damage"] for p in line) \
        + w.get("hp_lost_now", 0.0) * cand["exp_damage"]
    out = {"hp": hp_points,
           "kills": t.pop("enemy_alive", 0.0) + w["kill"] * sum(len(p["kills"]) for p in line),
           "damage": t.pop("enemy_hp", 0.0)}
    out.update(t)
    out["_hp"] = max(leaf.hero.hp, 0)
    out["_alive"] = len(leaf.alive())
    out["_enemy_hp"] = sum(e.hp for e in leaf.alive())
    return out


def _words(k: str, points: float, tb: dict, ts: dict, w: dict) -> str | None:
    if k not in _WORDS:
        return None
    unit, better, worse = _WORDS[k]
    if k == "hp":
        n = tb["_hp"] - ts["_hp"]
        if n == 0:
            return "takes less damage this turn" if points > 0 else "takes more damage this turn"
        return (better if n > 0 else worse)(abs(n))
    if k in ("kills", "damage"):
        n = ts["_alive"] - tb["_alive"] if k == "kills" else ts["_enemy_hp"] - tb["_enemy_hp"]
        if n == 0:
            return None
        return (better if n > 0 else worse)(abs(n))
    if unit is None:
        return (better if points > 0 else worse)(0)
    wt = w.get(unit, 0.0)
    n = abs(round(points / wt, 1)) if wt else 0
    n = int(n) if n == int(n) else n
    return (better if points > 0 else worse)(n)


def explain_decision(s: State, d: Decision, max_plies: int = 5) -> Decision:
    """Fill d.pv (the likely line of the chosen move) and d.reason (why it beat the
    second best: the biggest score-term differences between their likely lines)."""
    w = getattr(d, "_weights", WEIGHTS)
    ranked = sorted(d.candidates, key=lambda c: (-c["value"], _priority(s, c["action"])))
    if not ranked:
        return d
    best = ranked[0]
    line_b, leaf_b = principal_variation(s, d, best["action"], max_plies)
    d.pv = line_b
    if len(ranked) < 2:
        d.reason = [f"{best['action']} is the only possible move"]
        return d
    second = ranked[1]
    line_s, leaf_s = principal_variation(s, d, second["action"], max_plies)
    tb, ts = _line_terms(line_b, leaf_b, best, w), _line_terms(line_s, leaf_s, second, w)
    diffs = sorted(((k, tb.get(k, 0.0) - ts.get(k, 0.0)) for k in set(tb) | set(ts) if not k.startswith("_")),
                   key=lambda kv: -abs(kv[1]))
    if best["value"] == second["value"]:
        reason = [f"{best['action']} and {second['action']} score the same ({best['value']:.0f}); "
                  f"the tie-break order picks {best['action']} (attack, load, wait, move, turn)."]
    else:
        reason = [f"{best['action']} scores {best['value']:.0f}, {best['value'] - second['value']:.0f} more than "
                  f"{second['action']} ({second['value']:.0f}). On the likely lines:"]
    for k, pts in diffs:
        if abs(pts) < 1 or len(reason) > 4:
            break
        txt = _words(k, pts, tb, ts, w)
        if txt:
            reason.append(f"{'+' if pts > 0 else '-'} {txt} ({pts:+.0f})")
    # timing: later turns count GAMMA (0.97) less each, so the same result sooner scores higher
    kb = {}
    for i, ply in enumerate(line_b, 1):
        for k in ply["kills"]:
            kb.setdefault(k, i)
    ks = {}
    for i, ply in enumerate(line_s, 1):
        for k in ply["kills"]:
            ks.setdefault(k, i)
    for kind in sorted(set(kb) & set(ks), key=lambda k: kb[k]):
        if kb[kind] != ks[kind] and len(reason) <= 5:
            n = abs(ks[kind] - kb[kind])
            reason.append(f"{'+' if kb[kind] < ks[kind] else '-'} kills the {kind} {n} turn{'s' if n > 1 else ''} "
                          f"{'sooner' if kb[kind] < ks[kind] else 'later'}")
    hit_b = next((i for i, ply in enumerate(line_b, 1) if ply["hero_damage"]), None)
    hit_s = next((i for i, ply in enumerate(line_s, 1) if ply["hero_damage"]), None)
    if hit_b and hit_s and hit_b != hit_s and len(reason) <= 5:
        reason.append(f"{'+' if hit_b > hit_s else '-'} the first hit on you comes {abs(hit_b - hit_s)} turn(s) "
                      f"{'later' if hit_b > hit_s else 'sooner'}")
    if len(reason) == 1 and best["value"] != second["value"]:
        reason.append("the likely lines end the same; the difference is in the less likely outcomes")
    d.reason = reason
    return d
