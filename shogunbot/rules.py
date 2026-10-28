"""Stage 1 rules engine: hero actions and enemy presets.

Hypotheses from the domain model (tagged [hyp] there) are marked HYP here,
so a simulator mismatch in the logs points straight at the rule to fix.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

from .model import (
    ENEMY_KINDS, ENEMY_TILE_COOLDOWN, ENEMY_TILE_DAMAGE, HERO_TILES, QUEUE_MAX, SWAP_COOLDOWN,
    Enemy, Hero, State,
)

HERO_ACTIONS = ("left", "right", "turn", "wait", "queue:swirl", "queue:arrow", "attack")


@dataclass
class Events:
    """What happened during one full turn (hero action + enemy phase)."""
    hero_damage: list[tuple[str, int, int]] = field(default_factory=list)  # (enemy kind, uid, dmg)
    enemy_damage: list[tuple[str, int, int, str]] = field(default_factory=list)  # (kind, uid, dmg, source)
    kills: list[tuple[str, int, str]] = field(default_factory=list)  # (kind, uid, killer)
    enemy_actions: list[tuple[int, str]] = field(default_factory=list)  # (uid, action)
    swapped_with: int | None = None
    invalid: str | None = None

    def clone(self) -> "Events":
        return Events(list(self.hero_damage), list(self.enemy_damage), list(self.kills),
                      list(self.enemy_actions), self.swapped_with, self.invalid)

    def to_dict(self) -> dict:
        return {
            "hero_damage": self.hero_damage,
            "enemy_damage": self.enemy_damage,
            "kills": self.kills,
            "enemy_actions": self.enemy_actions,
            "swapped_with": self.swapped_with,
            "invalid": self.invalid,
        }


# ---------------------------------------------------------------------------
# hero
# ---------------------------------------------------------------------------

def legal_hero_actions(s: State) -> list[str]:
    out = []
    h = s.hero
    for a in HERO_ACTIONS:
        if a in ("left", "right"):
            d = -1 if a == "left" else 1
            nx = h.x + d
            if not s.in_bounds(nx):
                continue
            occ = s.enemy_at(nx)
            if occ is not None and not (h.facing == d and h.swap_ready):
                continue
        elif a.startswith("queue:"):
            t = a.split(":", 1)[1]
            if len(h.queue) >= QUEUE_MAX or not h.ready(t):
                continue
        elif a == "attack" and not h.queue:
            continue
        out.append(a)
    return out


def _damage_enemy(s: State, e: Enemy, dmg: int, source: str, ev: Events) -> None:
    if e.hp <= 0 or dmg <= 0:
        return
    e.hp -= dmg
    ev.enemy_damage.append((e.kind, e.uid, dmg, source))
    if e.hp <= 0:
        ev.kills.append((e.kind, e.uid, source))


def _first_unit_ahead(s: State, x: int, facing: int, skip_uid: int | None = None):
    nx = x + facing
    while s.in_bounds(nx):
        u = s.unit_at(nx)
        if u is not None and not (isinstance(u, Enemy) and u.uid == skip_uid):
            return u
        nx += facing
    return None


def _hero_fire(s: State, ev: Events) -> None:
    h = s.hero
    for tile in list(h.queue):  # bottom first
        dmg = HERO_TILES[tile][0]
        if tile == "swirl":
            for cx in (h.x + h.facing, h.x - h.facing):
                e = s.enemy_at(cx)
                if e:
                    _damage_enemy(s, e, dmg, "hero:swirl", ev)
        elif tile == "arrow":
            u = _first_unit_ahead(s, h.x, h.facing)
            if isinstance(u, Enemy):
                _damage_enemy(s, u, dmg, "hero:arrow", ev)
        h.charge[tile] = 0
    h.queue = []


def apply_hero(s: State, action: str, ev: Events) -> State:
    """Apply the hero's action (no enemy phase). Returns a new state."""
    s = s.clone()
    h = s.hero
    fired: set[str] = set()
    if action in ("left", "right"):
        d = -1 if action == "left" else 1
        nx = h.x + d
        occ = s.enemy_at(nx)
        if occ is not None:
            # swap (Wanderer special): both keep facing
            occ.x, h.x = h.x, nx
            h.swap_charge = 0
            ev.swapped_with = occ.uid
        else:
            h.x = nx
    elif action == "turn":
        h.facing = -h.facing
    elif action == "wait":
        pass
    elif action.startswith("queue:"):
        h.queue.append(action.split(":", 1)[1])
    elif action == "attack":
        fired = set(h.queue)
        _hero_fire(s, ev)
    # a turn passes: recharge everything not queued and not just fired
    for t, (_, cd) in HERO_TILES.items():
        if t not in h.queue and t not in fired and h.charge[t] < cd:
            h.charge[t] += 1
    if ev.swapped_with is None and h.swap_charge < SWAP_COOLDOWN:
        h.swap_charge += 1
    s.enemies = [e for e in s.enemies if e.hp > 0]
    return s


# ---------------------------------------------------------------------------
# enemies
# ---------------------------------------------------------------------------

def _toward(e: Enemy, hx: int) -> int:
    return 1 if hx > e.x else -1


def _hits_hero(s: State, e: Enemy, tile: str) -> bool:
    hx = s.hero.x
    if tile == "sword":
        return e.x + e.facing == hx
    if tile == "swirl":
        return abs(e.x - hx) == 1
    if tile == "charge":
        # HYP/[user]: only with a clear lane to the hero
        return _first_unit_ahead(s, e.x, e.facing, skip_uid=e.uid) is s.hero
    return False


def enemy_options(s: State, e: Enemy) -> list[tuple[float, tuple]]:
    """Preset table (domain model section 6). Returns [(prob, action)].

    action tuples: ("turn",) ("queue", tile) ("fire",) ("move", d) ("wait",)
    """
    if e.frozen > 0:
        return [(1.0, ("frozen",))]
    hx = s.hero.x
    d = _toward(e, hx)
    dist = abs(e.x - hx)
    facing_hero = e.facing == d

    def move_or_wait(direction: int) -> tuple:
        nx = e.x + direction
        if s.in_bounds(nx) and s.unit_at(nx) is None:
            return ("move", direction)
        return ("wait",)

    if e.intent is not None:
        opts = _intent_options(s, e, move_or_wait)
        if opts is not None:
            return opts

    if e.kind == "daisuke":
        return _daisuke_options(s, e, d, dist, facing_hero, move_or_wait)

    if e.armed and e.primed:
        return [(1.0, ("fire",))]  # yellow at the start of the turn (R2)

    _, tile = ENEMY_KINDS[e.kind]
    needs_facing = tile in ("sword", "charge")
    if needs_facing and not facing_hero:
        return [(1.0, ("turn",))]
    if not e.armed and e.cd > 0:
        # recharging: Ashigaru/Tachi reposition toward the hero, others hold
        if e.kind in ("ashigaru", "tachi") and dist > 1:
            return [(1.0, move_or_wait(d))]
        return [(0.5, ("wait",)), (0.5, move_or_wait(-d))] if dist == 1 else [(1.0, ("wait",))]
    if not e.armed:
        if dist == 1 and e.kind == "ashigaru":
            # HYP: after firing, Ashigaru was seen both re-arming and stepping away
            return [(0.6, ("queue", tile)), (0.4, move_or_wait(-d))]
        return [(1.0, ("queue", tile))]
    if e.kind == "charger":
        # fires only if its lane was already clear when the hero's turn began,
        # and not in the phase right after it loaded (R1)
        return [(1.0, ("fire",))] if e.primed else [(1.0, ("wait",))]
    if e.primed:
        return [(1.0, ("fire",))]
    if _hits_hero(s, e, e.queue[0]):
        return [(1.0, ("wait",))]  # hero just walked in: red now, yellow next phase
    return [(1.0, move_or_wait(d))]


def _intent_options(s: State, e: Enemy, move_or_wait):
    """First enemy phase after a screen read: the game shows what every enemy
    does next, so the phase is deterministic given the hero's action.

    [obs, replay of run 20261027-164727] a red (loaded, not yellow) enemy does
    not fire this phase even when the hero steps into its footprint; it turns
    yellow and fires the phase after.
    Daisuke's queued tile type is not telegraphed, so 'queue' falls back to
    his preset table.
    """
    it = e.intent
    if it == "fire" and e.armed:
        return [(1.0, ("fire",))]
    # Chargers follow R1 too: the red read in run 20261027-180145 (turn 60) was the
    # red phase of a blinking bubble [user]; run 20261028-044407 (turns 46, 55, 72)
    # showed red Chargers with a clear lane holding, then charging the turn after.
    if it == "turn":
        return [(1.0, ("turn",))]
    if it in ("move-left", "move-right"):
        return [(1.0, move_or_wait(-1 if it == "move-left" else 1))]
    if it == "queue":
        if e.kind == "daisuke":
            # the new tile's type is not telegraphed: his queue options, no fire
            dopts = [(p, a) for p, a in _daisuke_options(s, e, _toward(e, s.hero.x), abs(e.x - s.hero.x),
                                                          e.facing == _toward(e, s.hero.x), move_or_wait)
                     if a[0] == "queue"]
            if not dopts:
                dopts = [(0.4, ("queue", "sword")), (0.4, ("queue", "swirl")), (0.2, ("queue", "summon"))]
            tot = sum(p for p, _ in dopts)
            return [(p / tot, a) for p, a in dopts]
        return [(1.0, ("queue", ENEMY_KINDS[e.kind][1]))]
    if it in ("hold", "fire"):
        return [(1.0, ("wait",))]
    return None


def _daisuke_options(s, e, d, dist, facing_hero, move_or_wait):
    # [web] distance-based combo choice; probabilities are placeholders (HYP)
    if e.armed and e.primed and e.queue != ["charge"]:
        return [(1.0, ("fire",))]
    if not facing_hero:
        return [(1.0, ("turn",))]
    if e.armed:
        if e.queue == ["charge"]:
            # Charge + Sword is loaded in two phases, sometimes with a hold between
            # [run 20261028-053637 turns 120-123, 130-133]
            return [(0.5, ("queue", "sword")), (0.5, ("wait",))]
        if _hits_hero(s, e, e.queue[0]):
            return [(1.0, ("wait",))]
        if e.queue[0] == "charge":
            # lane blocked: waits or steps
            return [(0.5, ("wait",)), (0.5, move_or_wait(d))]
        return [(1.0, move_or_wait(d))]
    if dist >= 2:
        return [(0.6, ("queue", "charge")), (0.2, ("queue", "summon")), (0.2, move_or_wait(d))]
    return [(0.4, ("queue", "sword")), (0.4, ("queue", "swirl")), (0.2, ("queue", "summon"))]


def _hurt_hero(h, dmg: int, e: Enemy, ev: Events) -> None:
    """A shield (Kami Brew) negates the next attack received [wiki]."""
    if h.shield:
        h.shield = False
        ev.enemy_actions.append((e.uid, "hit absorbed by shield"))
        return
    h.hp -= dmg
    ev.hero_damage.append((e.kind, e.uid, dmg))


def _enemy_fire(s: State, e: Enemy, ev: Events) -> None:
    h = s.hero
    for tile in list(e.queue):
        if tile not in ENEMY_TILE_DAMAGE:
            tile = "sword"  # unknown icon read: assume the common melee hit
        dmg = ENEMY_TILE_DAMAGE[tile]
        if tile == "sword":
            tx = e.x + e.facing
            if h.x == tx:
                _hurt_hero(h, dmg, e, ev)
            else:
                t = s.enemy_at(tx)
                if t:
                    _damage_enemy(s, t, dmg, f"{e.kind}:sword", ev)
        elif tile == "swirl":
            for tx in (e.x + 1, e.x - 1):
                if h.x == tx:
                    _hurt_hero(h, dmg, e, ev)
                else:
                    t = s.enemy_at(tx)
                    if t:
                        _damage_enemy(s, t, dmg, f"{e.kind}:swirl", ev)
        elif tile == "charge":
            u = _first_unit_ahead(s, e.x, e.facing, skip_uid=e.uid)
            if u is not None:
                stop = (u.x if isinstance(u, Enemy) else h.x) - e.facing
                e.x = stop
                if u is h:
                    _hurt_hero(h, dmg, e, ev)
                else:
                    _damage_enemy(s, u, dmg, f"{e.kind}:charge", ev)
        elif tile == "summon":
            empties = [x for x in range(s.cells) if s.unit_at(x) is None]
            if empties:
                # HYP: random cell; planner assumes the cell farthest from the hero
                x = max(empties, key=lambda c: abs(c - h.x))
                uid = max([u.uid for u in s.enemies] + [0]) + 1
                s.enemies.append(Enemy("charger", x, 1 if h.x > x else -1, ENEMY_KINDS["charger"][0], [], uid))
    e.cd = max((ENEMY_TILE_COOLDOWN.get(t, 0) for t in e.queue), default=0)
    e.queue = []


def _apply_enemy_action(s: State, uid: int, action: tuple, ev: Events) -> None:
    e = next((x for x in s.enemies if x.uid == uid and x.hp > 0), None)
    if e is None:
        return
    ev.enemy_actions.append((uid, " ".join(map(str, action))))
    kind = action[0]
    if kind != "fire" and e.cd > 0:
        e.cd -= 1
    if kind not in ("queue", "fire"):
        e.wind = min(e.wind + 1, 2)
    if kind == "frozen":
        e.frozen -= 1
    elif kind == "turn":
        e.facing = -e.facing
    elif kind == "queue":
        tile = action[1]
        e.queue = e.queue + tile.split("+")
        e.wind = 0
    elif kind == "fire":
        _enemy_fire(s, e, ev)
    elif kind == "move":
        nx = e.x + action[1]
        if s.in_bounds(nx) and s.unit_at(nx) is None:
            e.x = nx


def enemy_phase(s: State, max_outcomes: int = 12) -> list[tuple[float, State, Events]]:
    """All enemy-phase outcomes as (prob, state, events).

    Enemies act left to right (HYP: order unknown). Each enemy's options are
    computed on the state as it is when its turn comes, so a lane opened by an
    earlier kill is seen by a later Charger.
    """
    order = [e.uid for e in sorted(s.alive(), key=lambda e: e.x)]
    frontier: list[tuple[float, State, Events]] = [(1.0, s.clone(), Events())]
    for uid in order:
        nxt = []
        for p, st, ev in frontier:
            e = next((x for x in st.enemies if x.uid == uid and x.hp > 0), None)
            if e is None:
                nxt.append((p, st, ev))
                continue
            opts = enemy_options(st, e)
            for q, act in opts:
                st2 = st.clone() if len(opts) > 1 else st
                ev2 = ev.clone() if len(opts) > 1 else ev
                _apply_enemy_action(st2, uid, act, ev2)
                st2.enemies = [x for x in st2.enemies if x.hp > 0]
                nxt.append((p * q, st2, ev2))
        nxt.sort(key=lambda t: -t[0])
        frontier = nxt[:max_outcomes]
    total = sum(p for p, _, _ in frontier) or 1.0
    for _, st, _ in frontier:
        for e in st.enemies:
            e.intent = None  # telegraphs only cover the coming phase
            e.primed = False
    return [(p / total, st, ev) for p, st, ev in frontier]


def step(s: State, action: str) -> list[tuple[float, State, Events]]:
    """Full turn: hero action then enemy phase."""
    ev0 = Events()
    if action not in legal_hero_actions(s):
        ev0.invalid = action
        return [(1.0, s.clone(), ev0)]
    # Yellow is decided BEFORE the hero acts: a loaded enemy (past its one-phase
    # wind-up, R1) with the hero in its footprint at the start of the turn fires
    # this phase whatever the hero does (R2). That is what makes the swap trick
    # work: the swapped enemy keeps its facing and fires away from the hero
    # [user; run 20261028-053637]. A red enemy instead turns or waits.
    primed = {e.uid for e in s.alive()
              if e.armed and e.wind >= 1 and e.queue[0] in ("sword", "swirl", "charge")
              and _hits_hero(s, e, e.queue[0])}
    s1 = apply_hero(s, action, ev0)
    for e in s1.enemies:
        e.primed = e.uid in primed
    out = []
    for p, s2, ev in enemy_phase(s1):
        merged = ev0.clone()
        merged.hero_damage += ev.hero_damage
        merged.enemy_damage += ev.enemy_damage
        merged.kills += ev.kills
        merged.enemy_actions += ev.enemy_actions
        s2.turn = s.turn + 1
        out.append((p, s2, merged))
    return out


def threats(s: State) -> list[dict]:
    """Armed enemies whose current queue would hit the hero if fired now."""
    out = []
    for e in s.alive():
        if e.armed and _hits_hero(s, e, e.queue[0]):
            dmg = sum(ENEMY_TILE_DAMAGE[t] for t in e.queue if t != "summon")
            out.append({"uid": e.uid, "kind": e.kind, "dmg": dmg})
    return out
