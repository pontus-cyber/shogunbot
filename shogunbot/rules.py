"""Rules engine: hero actions and enemy presets.

Tile footprints come from the catalogue (data/tiles.json, field 'attack'),
so hero and enemy attacks share one resolver and a new tile is a data
change. What an enemy does next is its telegraph: read from the screen for
the coming phase, and for later simulated turns drawn from the behaviour
model learned from the run logs (behaviour.py, data/behaviour.json) on the
board at the start of the turn, as the game decides it. The 0.16 hand-set
presets remain behind BEHAVIOUR = "presets" for comparison. Hypotheses from
the domain model (tagged [hyp] there) are marked HYP here, so a simulator
mismatch in the logs points straight at the rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import behaviour
from .catalogue import CAT
from .model import (
    BOSS_KINDS, DEFAULT_KIT, ENEMY_KINDS, ENEMY_TILE_COOLDOWN, ENEMY_TILE_DAMAGE, QUEUE_MAX, SWAP_COOLDOWN,
    Enemy, Hero, State,
)

MOVES = ("left", "right", "turn", "wait")
PAIRED = frozenset(k for k in BOSS_KINDS if CAT.enemy(k).pair > 1)   # bosses whose units share one HP bar
HERO_ACTIONS = MOVES + DEFAULT_KIT.queue_actions + ("attack",)   # default kit; see hero_actions()
POISON_TICK = next(c.effect["damage"] for c in CAT.consumables.values() if c.effect["type"] == "poison_all")
POISON_TURNS = next(c.effect["turns"] for c in CAT.consumables.values() if c.effect["type"] == "poison_all")
ICE_TURNS = next(c.effect["turns"] for c in CAT.consumables.values() if c.effect["type"] == "freeze_all")


def hero_actions(hero: Hero) -> tuple[str, ...]:
    return MOVES + hero.kit.queue_actions + ("attack",)


# tile id -> (kind, offsets, dir, range, cannot_kill): the catalogue footprint, unpacked for hot loops
SPEC: dict[str, tuple] = {
    t.id: (t.kind, tuple(t.attack.get("offsets", ())), t.attack.get("dir", 1), t.attack.get("range"),
           bool(t.attack.get("cannot_kill")))
    for t in CAT.tiles.values()}


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
# footprints (shared by hero and enemies)
# ---------------------------------------------------------------------------

def _first_unit_ahead(s: State, x: int, facing: int, skip_uid: int | None = None, rng: int | None = None):
    nx = x + facing
    while s.in_bounds(nx) and (rng is None or abs(nx - x) <= rng):
        u = s.unit_at(nx)
        if u is not None and not (isinstance(u, Enemy) and u.uid == skip_uid):
            return u
        nx += facing
    return None


def _furthest_unit_ahead(s: State, x: int, facing: int):
    found, nx = None, x + facing
    while s.in_bounds(nx):
        u = s.unit_at(nx)
        if u is not None:
            found = u
        nx += facing
    return found


def tile_hits(s: State, x: int, facing: int, tile: str, skip_uid: int | None = None) -> list:
    """Units a tile fired from cell x with this facing would hit, ignoring
    movement during the attack (used for threat checks)."""
    spec = CAT.tile(tile).attack
    k = spec["kind"]
    d = facing * spec.get("dir", 1)
    if k == "cells":
        return [u for off in spec["offsets"] if (u := s.unit_at(x + facing * off)) is not None]
    if k in ("first", "dash_strike"):
        u = _first_unit_ahead(s, x, d, skip_uid, spec.get("range"))
        return [u] if u is not None else []
    if k == "furthest":
        u = _furthest_unit_ahead(s, x, d)
        return [u] if u is not None else []
    return []


# ---------------------------------------------------------------------------
# hero
# ---------------------------------------------------------------------------

def legal_hero_actions(s: State) -> list[str]:
    out = []
    h = s.hero
    for a in MOVES:
        if a in ("left", "right"):
            d = -1 if a == "left" else 1
            nx = h.x + d
            if not s.in_bounds(nx):
                continue
            occ = s.enemy_at(nx)
            if occ is not None and not ((h.facing == d or h.kit.swap_both_ways) and h.swap_ready):
                continue
        out.append(a)
    q = h.queue
    if len(q) < QUEUE_MAX:
        ch = h.charge
        for (lb, cd), a in zip(h.kit.cooldowns, h.kit.queue_actions):
            if lb not in q and ch.get(lb, 0) >= cd:
                out.append(a)
    if q:
        out.append("attack")
    return out


def _damage_enemy(s: State, e: Enemy, dmg: int, source: str, ev: Events, cannot_kill: bool = False) -> None:
    if e.hp <= 0 or dmg <= 0:
        return
    if e.cursed:
        dmg *= 2          # Mass Curse: double damage on the next hit [wiki] [user]
        e.cursed = False
    if cannot_kill:
        dmg = min(dmg, e.hp - 1)   # Nagiboku leaves at least 1 HP [wiki]
        if dmg <= 0:
            return
    e.hp -= dmg
    ev.enemy_damage.append((e.kind, e.uid, dmg, source))
    if e.hp <= 0:
        ev.kills.append((e.kind, e.uid, source))
    if e.kind in PAIRED:
        # The Twins share one HP bar: a hit on either takes it from both, and both die at 0 [wiki]
        # [rec: collect/20261028-190026 f0580, collect/20261029-160032 f0775]
        for o in s.enemies:
            if o is not e and o.kind == e.kind and o.hp > 0:
                o.hp = e.hp
                if o.hp <= 0:
                    ev.kills.append((o.kind, o.uid, source))


def _push(s: State, u, d: int, ev: Events, collision: int, src: str, by: Enemy | None,
          shielded: bool = False) -> None:
    """Dragon Punch: u slides along d as far as it can [wiki: Tiles]. A unit in the way stops it
    and both take `collision` damage; the edge of the board stops it without damage [bot: Rei,
    runs 20261029-100218 t68/t79/t143 and -103242 t91 (a unit), -101549 t91 (the wall)]. When a
    shield took the punch it also takes the collision, but the hero still slides [bot: run
    20261029-072323 t69]."""
    if u is None or u.hp <= 0:
        return
    nx = u.x + d
    while s.in_bounds(nx) and s.unit_at(nx) is None:
        u.x = nx
        nx += d
    w = s.unit_at(nx) if s.in_bounds(nx) else None
    if w is None or collision <= 0:
        return
    for v in (u, w):
        if v is s.hero:
            if not (v is u and shielded) and by is not None:
                _hurt_hero(s.hero, collision, by, ev)
        else:
            _damage_enemy(s, v, collision, src + "+collision", ev)


def _skill_bonus(s: State, e: Enemy, moved: bool) -> int:
    """Extra damage from the build's skills [wiki: Shop & Skills]: Back Stabber +1 on an enemy facing
    away from the hero, Sniper +1 from 4 or more cells away, Close Combat +1 on an adjacent enemy unless
    the tile moved the hero; a level adds its amount again (two Snipers give +2 [wiki])."""
    fx = s.hero.kit.fx
    if not fx:
        return 0
    h = s.hero
    dist = abs(e.x - h.x)
    b = 0
    if "damage_from_behind" in fx and dist and e.facing == (1 if e.x > h.x else -1):
        b += fx["damage_from_behind"]
    if "damage_far" in fx and dist >= 4:
        b += fx["damage_far"]
    if "damage_adjacent" in fx and dist == 1 and not moved:
        b += fx["damage_adjacent"]
    return b


def _hero_hit(s: State, e: Enemy, kt, spec: dict, ev: Events, moved: bool = False, boost: int = 0) -> bool:
    """One hero tile hitting one enemy, with the tile's enchantment and the build's skills (moved:
    the tile moved the hero before it struck; boost: Dynamic Boost's extra damage this turn).
    Returns True on an exact kill (Perfect Strike)."""
    src = "hero:" + kt.tile
    hp0 = e.hp
    _damage_enemy(s, e, kt.damage + _skill_bonus(s, e, moved) + boost, src, ev, spec.get("cannot_kill", False))
    ench = kt.enchant
    if ench is None:
        return False
    if ench == "shockwave":
        for nx in (e.x - 1, e.x + 1):   # enemies next to the target take 1 [wiki]
            n = s.enemy_at(nx)
            if n is not None and n is not e:
                _damage_enemy(s, n, 1, src + "+shockwave", ev)
    if e.hp > 0:
        if ench == "poison":
            e.poison = max(e.poison, POISON_TURNS)
        elif ench == "ice":
            e.frozen = max(e.frozen, ICE_TURNS)
            e.intent = None
        elif ench == "curse":
            e.cursed = True
    return ench == "perfect_strike" and hp0 > 0 and e.hp == 0


def _hero_attack(s: State, kt, ev: Events, boost: int = 0, dyn: int = 0) -> bool:
    """Fire one kit tile from the hero's current cell. Returns True when the tile recharges at once
    (Perfect Strike). boost: extra damage on every hit (Dynamic Boost after a tile moved the hero this
    turn); dyn: Dynamic Boost's bonus, which a dash tile also gets on its own strike once it has moved
    the hero [hyp: the wiki says 'if you moved with a tile in the same turn']."""
    h = s.hero
    spec = CAT.tile(kt.tile).attack
    k = spec["kind"]
    d = h.facing * spec.get("dir", 1)
    recharge = False
    for _ in range(2 if kt.enchant == "double_strike" else 1):
        if k == "cells":
            for off in spec["offsets"]:
                e = s.enemy_at(h.x + h.facing * off)
                if e:
                    recharge |= _hero_hit(s, e, kt, spec, ev, boost=boost)
                    if spec.get("push"):
                        _push(s, e, h.facing * (1 if off > 0 else -1), ev, spec.get("collision", 1),
                              "hero:" + kt.tile, None)
        elif k in ("first", "furthest"):
            u = (_first_unit_ahead(s, h.x, d, rng=spec.get("range")) if k == "first"
                 else _furthest_unit_ahead(s, h.x, d))
            if isinstance(u, Enemy):
                recharge |= _hero_hit(s, u, kt, spec, ev, boost=boost)
        elif k == "damaged":
            for e in [e for e in s.alive() if e.hp < e.max_hp]:
                recharge |= _hero_hit(s, e, kt, spec, ev, boost=boost)
        elif k == "dash_strike":
            # HYP: the hero stops next to the first unit and strikes it; with
            # nothing in the way it dashes to the last cell (catalogue no_target)
            u = _first_unit_ahead(s, h.x, d)
            if u is None:
                if spec.get("no_target") == "edge":
                    h.x = s.cells - 1 if d > 0 else 0
            else:
                moved = h.x != u.x - d
                h.x = u.x - d
                recharge |= _hero_hit(s, u, kt, spec, ev, moved, max(boost, dyn) if moved else boost)
        elif k == "dash":
            # the Dash: forward as far as possible [wiki: Tiles]
            while s.in_bounds(h.x + d) and s.unit_at(h.x + d) is None:
                h.x += d
        elif k == "swap_toss":
            _swap_cells(s, h.x + h.facing, h.x - h.facing)
        elif k == "dash_through":
            # HYP: passes every unit of the first group ahead, lands on the
            # first empty cell behind them and strikes each one it passed
            nx = h.x + d
            while s.in_bounds(nx) and s.unit_at(nx) is None:
                nx += d
            passed = []
            while s.in_bounds(nx) and s.enemy_at(nx) is not None:
                passed.append(s.enemy_at(nx))
                nx += d
            if passed and s.in_bounds(nx):
                h.x = nx
                for e in passed:
                    recharge |= _hero_hit(s, e, kt, spec, ev, True, max(boost, dyn))
    return recharge


def _hero_fire(s: State, ev: Events) -> set[str]:
    """Fire the queue bottom first. Returns the labels that do not recharge
    from zero (Perfect Strike)."""
    h = s.hero
    keep = set()
    # Dynamic Boost: attacks deal more once a tile has moved the hero this turn [wiki] [rec2 f1158-f1159:
    # level 2, a Charge carried the hero to the Guardian and the 4-damage Swirl behind it showed 6]
    dyn, moved = h.kit.fx.get("damage_if_moved", 0), False
    for label in list(h.queue):
        kt = h.kit.by_label.get(label)
        if kt is None:
            continue
        x0 = h.x
        if _hero_attack(s, kt, ev, dyn if moved else 0, dyn):
            keep.add(label)
        else:
            h.charge[label] = 0
        moved = moved or h.x != x0
    h.queue = []
    return keep


def free_action(s: State, action: str) -> bool:
    """Queueing a Freeplay-enchanted tile does not pass the turn [wiki]."""
    if not action.startswith("queue:"):
        return False
    kt = s.hero.kit.by_label.get(action[6:])
    return kt is not None and kt.enchant == "freeplay"


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
            if h.kit.fx.get("special_damage"):
                _damage_enemy(s, occ, h.kit.fx["special_damage"], "hero:swap", ev)   # Damaging Move [wiki]
        else:
            h.x = nx
    elif action == "turn":
        h.facing = -h.facing
    elif action == "wait":
        pass
    elif action.startswith("queue:"):
        h.queue.append(action[6:])
        if free_action(s, action):
            return s   # no turn passes, nothing recharges
    elif action == "attack":
        fired = set(h.queue) - _hero_fire(s, ev)
    # a turn passes: recharge everything not queued and not just fired
    ch, q = h.charge, h.queue
    for lb, cd in h.kit.cooldowns:
        c = ch.get(lb, 0)
        if c < cd and lb not in q and lb not in fired:
            ch[lb] = c + 1
    if ev.swapped_with is None and h.swap_charge < h.kit.swap_cooldown:
        h.swap_charge += 1
    extra = h.kit.fx.get("wait_recharge") if action == "wait" else 0
    if extra:
        # Mindfulness: waiting recharges an extra cooldown of the tiles and the special move [wiki]
        for lb, cd in h.kit.cooldowns:
            if lb not in q:
                ch[lb] = min(cd, ch.get(lb, 0) + extra)
        h.swap_charge = min(h.kit.swap_cooldown, h.swap_charge + extra)
    s.enemies = [e for e in s.enemies if e.hp > 0]
    return s


# ---------------------------------------------------------------------------
# enemies
# ---------------------------------------------------------------------------

def _toward(e: Enemy, hx: int) -> int:
    return 1 if hx > e.x else -1


def _hits_hero(s: State, e: Enemy, tile: str) -> bool:
    """Would e's tile hit the hero now? For a unit whose kind the read could not settle
    (e.alt), also where the other kind's tile would reach: the planner then treats a swap
    with a primed Ashigaru that may be a Tachi as unsafe [user review of run 20261029-045856]."""
    if _tile_hits_hero(s, e, tile):
        return True
    return e.alt is not None and _tile_hits_hero(s, e, ENEMY_KINDS[e.alt][1])


def _tile_hits_hero(s: State, e: Enemy, tile: str) -> bool:
    if tile not in ENEMY_TILE_DAMAGE:
        return False
    k, offs, dr, rng, _ = SPEC[tile]
    if k == "cells":
        hx, ex, f = s.hero.x, e.x, e.facing
        for off in offs:
            if ex + f * off == hx:
                return True
        return False
    if k in ("first", "dash_strike"):
        # HYP/[user]: a charge needs a clear lane to the hero
        return _first_unit_ahead(s, e.x, e.facing * dr, skip_uid=e.uid, rng=rng) is s.hero
    if k == "furthest":
        return _furthest_unit_ahead(s, e.x, e.facing * dr) is s.hero
    if k == "aim":
        # the Volley: aimed at the hero's cell when the turn it fires begins, so the hero is always in reach
        # until then; once aimed, only that cell is hit [data/tiles.json 'volley']
        return e.aim is None or e.aim == s.hero.x
    return False


BEHAVIOUR = "learned"   # "presets": the hand-set tables of 0.16 instead of the learned model (M3)


def enemy_options(s: State, e: Enemy, presets: bool = False) -> list[tuple[float, tuple]]:
    """What an enemy does in this enemy phase. Returns [(prob, action)].

    With a telegraph (read from the screen, or drawn at the start of the
    turn by the planner and self-play) the action is that telegraph. Without
    one: the learned behaviour model (behaviour.py, from the run logs), or
    with presets=True the 0.16 hand-set tables (domain model section 7).

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

    if not presets and BEHAVIOUR == "learned":
        drawn = behaviour.intents(s, e)
        if drawn:
            out, saved = [], e.intent
            try:
                for p, it in drawn:
                    e.intent = it
                    for q, a in _intent_options(s, e, move_or_wait) or [(1.0, ("wait",))]:
                        out.append((p * q, a))
            finally:
                e.intent = saved
            return out

    if e.kind == "daisuke":
        return _daisuke_options(s, e, d, dist, facing_hero, move_or_wait)
    if e.kind in BOSS_KINDS:
        return _boss_options(s, e, d, dist, facing_hero, move_or_wait)

    if e.armed and e.primed:
        return [(1.0, ("fire",))]  # yellow at the start of the turn (R2)

    _, tile = ENEMY_KINDS[e.kind]
    if CAT.tile(tile).directional and not facing_hero:
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
        if e.kind in BOSS_KINDS:
            return _boss_queue(s, e, move_or_wait)
        return [(1.0, ("queue", ENEMY_KINDS[e.kind][1]))]
    if it in ("hold", "fire"):
        return [(1.0, ("wait",))]
    return None


def _boss_queue(s: State, e: Enemy, move_or_wait) -> list[tuple[float, tuple]]:
    """What a boss loads on a 'queue' telegraph (the bubble shows it only a phase later). The
    second half of a combo follows the first (catalogue 'combos': Daisuke's Charge then Sword, 29
    of 29 logged; Rei's Charge then Dragon Punch 10 of 10, Back Charge then Summon 2 of 2; Iwao's
    Earth Impale then Swirl with the hero next to him, 3 of 3); otherwise what he loaded in the logs
    at this distance with this queue (behaviour.json), else the boss's catalogue 'loads' by distance
    (Rei, Iwao), else Daisuke's presets."""
    kind = CAT.enemy(e.kind)
    b = kind.behaviour
    dist = abs(e.x - s.hero.x)
    # a 'queue' telegraph over the first half of a combo adds its second half: he decided at the
    # start of the turn, so Iwao's Swirl comes even if the hero has stepped away since (dist 1)
    nxt = kind.combo_next(e.queue, 1)
    if nxt:
        return [(1.0, ("queue", nxt))]
    pair = _pair_load(s, e, b) if not e.queue else None
    if pair:
        return pair
    m = behaviour.model() if BEHAVIOUR == "learned" else None
    learned = m.boss_tile(abs(e.x - s.hero.x), e.queue, e.kind) if m else None
    if learned:
        return [(p, ("queue", t)) for p, t in learned]
    loads = b.get("loads", {}).get(str(min(dist, 3))) if not e.queue else None
    if loads:
        tot = sum(loads.values())
        return [(n / tot, ("queue", t)) for t, n in sorted(loads.items())]
    if e.kind != "daisuke":
        return [(1.0, ("queue", ENEMY_KINDS[e.kind][1]))]
    dopts = [(p, a) for p, a in _daisuke_options(s, e, _toward(e, s.hero.x), dist,
                                                  e.facing == _toward(e, s.hero.x), move_or_wait)
             if a[0] == "queue"]
    if not dopts:
        dopts = [(0.4, ("queue", "sword")), (0.4, ("queue", "swirl")), (0.2, ("queue", "summon"))]
    tot = sum(p for p, _ in dopts)
    return [(p / tot, a) for p, a in dopts]


def _pair_load(s: State, e: Enemy, b: dict) -> list[tuple[float, tuple]] | None:
    """The Twins' loads (catalogue 'behaviour') [wiki: The Twins] [rec: both recordings' fights]: with
    both on one side of the hero the rear one loads Spear, and the front one, next to him, Shadow Dash,
    or the Dash (Phantom Leap) followed by Swap Toss when he stands on the wall's cell; with the hero
    between them each loads a melee tile in the recorded proportions."""
    if "outside" not in b:
        return None
    mates = [o for o in s.alive() if o is not e and o.kind == e.kind]
    if not mates:
        return None
    hx, o = s.hero.x, mates[0]
    if (e.x - hx) * (o.x - hx) > 0:                       # both on one side of the hero
        if abs(e.x - hx) > abs(o.x - hx):
            return [(1.0, ("queue", b["outside"]["rear"]))]
        if abs(e.x - hx) == 1:
            at_wall = hx in (0, s.cells - 1)
            return [(1.0, ("queue", (b["outside"]["front_at_wall"] if at_wall else b["outside"]["front"])[0]))]
        return None
    tot = sum(b["between"].values())
    return [(n / tot, ("queue", t)) for t, n in sorted(b["between"].items())]


def _boss_options(s, e, d, dist, facing_hero, move_or_wait):
    """A boss without a telegraph, from the catalogue (Rei, Iwao) [bot: Rei's five fights, runs
    20261029-072323 ... -103242; Iwao: run 20261029-124921 t159-t198]: a summon or a finished combo
    fires one phase after loading (R1); a combo's first tile is followed by its second; a single
    tile with the hero in its footprint fires (R2), otherwise she walks toward him (a loaded Dragon
    Punch was walked up and fired when adjacent, run 100218 t88-t91); unloaded, she turns to the
    hero or loads. Iwao ('after_fire': away) steps away from the hero the turn after he fires, or
    holds when blocked, and loads the turn after (8 of 8)."""
    kind = CAT.enemy(e.kind)
    if e.armed:
        q = e.queue
        nxt = kind.combo_next(q, dist)
        if nxt:
            return [(1.0, ("queue", nxt))] if e.wind >= 1 else [(1.0, ("wait",))]
        if e.wind >= 1 and (len(q) > 1 or CAT.tile(q[0]).kind == "summon"):
            return [(1.0, ("fire",))]
        if e.primed:
            return [(1.0, ("fire",))]
        if not facing_hero and CAT.tile(q[0]).directional:
            return [(1.0, ("turn",))]
        if _hits_hero(s, e, q[0]):
            return [(1.0, ("wait",))]     # the hero walked in: red now, yellow next phase
        return [(1.0, move_or_wait(d))]
    if kind.behaviour.get("after_fire") == "away" and e.fage == 0:
        return [(1.0, move_or_wait(-d))]
    if not facing_hero:
        return [(1.0, ("turn",))]
    return _boss_queue(s, e, move_or_wait)


def _daisuke_options(s, e, d, dist, facing_hero, move_or_wait):
    # [web] distance-based combo choice; probabilities are placeholders (HYP)
    if e.armed and e.primed and e.queue != ["charge"]:
        return [(1.0, ("fire",))]
    if e.armed and e.queue[0] == "summon" and e.wind >= 1:
        # a loaded Summon holds one phase (red), then fires (yellow), facing or not
        # [runs 20261028-053637 t93-94, t116-117, t136-137, t146-147; 20261028-061420 t93-94, t115-116]
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
    """A shield (Kami Brew) negates the next attack received [wiki]. Skills [wiki: Shop & Skills]:
    Iron Skin takes 1 off each hit but never to 0; Reactive Shield raises a shield when the hero is hit
    (not while one is up; also on a 0-damage hit)."""
    if h.shield:
        h.shield = False
        ev.enemy_actions.append((e.uid, "hit absorbed by shield"))
        return
    fx = h.kit.fx
    if fx:
        if fx.get("damage_taken") and dmg > 0:
            dmg = max(1, dmg + fx["damage_taken"])
        if "shield_on_damage" in fx:
            h.shield = True
    h.hp -= dmg
    ev.hero_damage.append((e.kind, e.uid, dmg))


def _enemy_hit(s: State, e: Enemy, u, dmg: int, tile: str, ev: Events) -> None:
    if u is s.hero:
        _hurt_hero(s.hero, dmg, e, ev)
    elif u is not None:
        # Unfriendly Fire: enemies deal +1 damage to each other [wiki]
        _damage_enemy(s, u, dmg + s.hero.kit.fx.get("enemy_friendly_fire", 0), f"{e.kind}:{tile}", ev)


def _swap_cells(s: State, a: int, b: int) -> None:
    """Swap Toss: whatever stands on cells a and b changes places (a cell off the board holds nothing)."""
    ua = s.unit_at(a) if s.in_bounds(a) else None
    ub = s.unit_at(b) if s.in_bounds(b) else None
    if ua is not None and s.in_bounds(b):
        ua.x = b
    if ub is not None and s.in_bounds(a):
        ub.x = a


def _enemy_fire(s: State, e: Enemy, ev: Events) -> None:
    h = s.hero
    if e.alt is not None:
        # the read could not rule out alt: its tile hits the hero where e's own would not
        at = ENEMY_KINDS[e.alt][1]
        if (at in ENEMY_TILE_DAMAGE and _tile_hits_hero(s, e, at)
                and not any(_tile_hits_hero(s, e, t) for t in e.queue)):
            _hurt_hero(h, ENEMY_TILE_DAMAGE[at], e, ev)
    for tile in list(e.queue):
        if tile not in ENEMY_TILE_DAMAGE:
            tile = "sword"  # unknown icon read: assume the common melee hit
        dmg = ENEMY_TILE_DAMAGE[tile]
        spec = CAT.tile(tile).attack
        k = spec["kind"]
        if k == "cells":
            # friendly fire: Twin Tachi and Daisuke's Swirl hit allies too [wiki]; Rei punches
            # her own Twin Tachi into the hero [wiki] [bot: run 20261029-103242 t91]
            for off in spec["offsets"]:
                u = s.unit_at(e.x + e.facing * off)
                shielded = u is h and h.shield
                _enemy_hit(s, e, u, dmg, tile, ev)
                if spec.get("push") and u is not None:
                    _push(s, u, e.facing * (1 if off > 0 else -1), ev, spec.get("collision", 1),
                          f"{e.kind}:{tile}", e, shielded)
        elif k in ("first", "furthest"):
            fn = _first_unit_ahead if k == "first" else _furthest_unit_ahead
            _enemy_hit(s, e, fn(s, e.x, e.facing * spec.get("dir", 1)), dmg, tile, ev)
        elif k == "dash_strike":
            dd = e.facing * spec.get("dir", 1)
            u = _first_unit_ahead(s, e.x, dd, skip_uid=e.uid)
            if u is not None:
                e.x = u.x - dd
                _enemy_hit(s, e, u, dmg, tile, ev)
            elif spec.get("no_target") == "edge":
                # nothing ahead: the dash ends at the wall. A boss that the hero swapped past fires
                # his Charge away from him and ends at the wall, 25 of 25 logged times [bot: Daisuke
                # and Rei, runs 20261028-044407 t94 ... 20261029-100218 t135]
                e.x = s.cells - 1 if dd > 0 else 0
        elif k == "dash_through":
            # Shadow Dash: through the units ahead to the first empty cell behind them, 1 damage to
            # each; nothing happens without an empty cell [wiki: Tiles] [rec: the Twins carried the
            # front one through the hero, collect/20261028-190026 f0562-f0565, -160032 f0755-f0761]
            dd = e.facing * spec.get("dir", 1)
            nx, passed = e.x + dd, []
            while s.in_bounds(nx) and s.unit_at(nx) is not None and s.unit_at(nx) is not e:
                passed.append(s.unit_at(nx))
                nx += dd
            if passed and s.in_bounds(nx):
                e.x = nx
                for u in passed:
                    _enemy_hit(s, e, u, dmg, tile, ev)
        elif k == "dash":
            # the Dash (the Twins' Phantom Leap): forward as far as possible [wiki: Tiles]; HYP: it
            # stops at the first unit or the wall
            dd = e.facing * spec.get("dir", 1)
            while s.in_bounds(e.x + dd) and s.unit_at(e.x + dd) is None:
                e.x += dd
        elif k == "swap_toss":
            # Swap Toss: the contents of the cells ahead and behind change places [wiki: Tiles]; the
            # hero next to a Twin landed behind it [rec: collect/20261029-160032 f0770-f0772]
            _swap_cells(s, e.x + e.facing, e.x - e.facing)
        elif k == "aim":
            # the Volley strikes the cell aimed at as the turn began, whatever unit stands there: the hero who
            # stayed (run 20261030-115221 t174, -2), nobody once he stepped off, or the Sniper itself after
            # the hero swapped it onto the cell [rec: collect/20261029-160032 f0199-f0200, f0222-f0223]
            cell = e.aim if e.aim is not None else h.x
            _enemy_hit(s, e, s.unit_at(cell) if s.in_bounds(cell) else None, dmg, tile, ev)
        elif k == "summon":
            # the summoner's unit (Daisuke: Spike Chargers, Rei: Twin Tachi) [wiki]; HYP: random
            # cells, the planner assumes those farthest from the hero
            kind = CAT.enemy(e.kind).summons or spec.get("unit", "charger")
            for _ in range(spec.get("count", 1)):
                empties = [x for x in range(s.cells) if s.unit_at(x) is None]
                if not empties:
                    break
                x = max(empties, key=lambda c: abs(c - h.x))
                uid = max([u.uid for u in s.enemies] + [0]) + 1
                s.enemies.append(Enemy(kind, x, 1 if h.x > x else -1, ENEMY_KINDS[kind][0], [], uid))
    e.cd = max((ENEMY_TILE_COOLDOWN.get(t, 0) for t in e.queue), default=0)
    e.queue = []
    e.aim = None


def _apply_enemy_action(s: State, uid: int, action: tuple, ev: Events) -> None:
    e = next((x for x in s.enemies if x.uid == uid and x.hp > 0), None)
    if e is None:
        return
    ev.enemy_actions.append((uid, " ".join(map(str, action))))
    kind = action[0]
    # history for the behaviour model: the telegraph's class (a blocked move still counts as a move)
    e.last = behaviour.history_class(e.intent) or {"fire": "fire", "queue": "queue", "turn": "turn",
                                                     "move": "move"}.get(kind, "hold")
    e.fage = 0 if e.last == "fire" else (e.fage + 1 if e.fage is not None else None)
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


def _poison_tick(s: State, ev: Events) -> None:
    """End of turn: every poisoned enemy takes 1 damage [wiki]."""
    for e in s.alive():
        if e.poison > 0:
            e.poison -= 1
            _damage_enemy(s, e, POISON_TICK, "poison", ev)
    s.enemies = [x for x in s.enemies if x.hp > 0]


def enemy_phase(s: State, max_outcomes: int = 12) -> list[tuple[float, State, Events]]:
    """All enemy-phase outcomes as (prob, state, events).

    Enemies act left to right (HYP: order unknown). Each enemy's options are
    computed on the state as it is when its turn comes, so a lane opened by an
    earlier kill is seen by a later Charger.
    """
    order = [e.uid for e in sorted(s.alive(), key=lambda e: e.x)]
    poisoned = any(e.poison > 0 for e in s.enemies)   # only the hero's action applies poison
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
    for _, st, ev in frontier:
        if poisoned:
            _poison_tick(st, ev)
        for e in st.enemies:
            e.intent = None  # telegraphs only cover the coming phase
            e.primed = False
            e.aim = None     # an aim lasts the turn it was taken in
    return [(p / total, st, ev) for p, st, ev in frontier]


MAX_BRANCHES = 8   # joint telegraphs kept per chance node (most likely first)


def draw_intents(s: State, max_branches: int = MAX_BRANCHES) -> list[tuple[float, State]] | None:
    """The game decides every enemy's action at the start of the hero's turn
    and shows it as a telegraph (domain model section 6). For enemies whose
    telegraph is not known (simulated turns after the first), draw it from the
    learned behaviour model on the board as it is now, BEFORE the hero acts:
    [(probability, state with telegraphs)], or None when nothing is missing."""
    if BEHAVIOUR != "learned" or behaviour.model() is None:
        return None
    todo = [e for e in s.alive() if e.intent is None and e.frozen <= 0]
    if not todo:
        return None
    combos = [(1.0, {})]
    for e in todo:
        opts = behaviour.intents(s, e) or [(1.0, "hold")]
        combos = [(p * q, {**c, e.uid: it}) for p, c in combos for q, it in opts]
        combos.sort(key=lambda t: -t[0])
        combos = combos[:max_branches]
    tot = sum(p for p, _ in combos) or 1.0
    out = []
    for p, c in combos:
        sc = s.clone()
        for e in sc.enemies:
            if e.uid in c:
                e.intent = c[e.uid]
        out.append((p / tot, sc))
    return out


def step(s: State, action: str, validate: bool = True) -> list[tuple[float, State, Events]]:
    """Full turn: hero action then enemy phase. The planner passes
    validate=False for actions it took from legal_hero_actions itself.
    Missing telegraphs are drawn first, on the board before the hero acts."""
    ev0 = Events()
    if validate and action not in legal_hero_actions(s):
        ev0.invalid = action
        return [(1.0, s.clone(), ev0)]
    if free_action(s, action):
        s1 = apply_hero(s, action, ev0)
        return [(1.0, s1, ev0)]
    branches = draw_intents(s)
    if branches:
        out = []
        for pb, sb in branches:
            out += [(pb * p, s2, ev) for p, s2, ev in step(sb, action, validate=False)]
        return out
    # Yellow is decided BEFORE the hero acts: a loaded enemy (past its one-phase
    # wind-up, R1) with the hero in its footprint at the start of the turn fires
    # this phase whatever the hero does (R2). That is what makes the swap trick
    # work: the swapped enemy keeps its facing and fires away from the hero
    # [user; run 20261028-053637]. A red enemy instead turns or waits.
    primed = {e.uid for e in s.alive()
              if e.armed and e.wind >= 1 and ENEMY_TILE_DAMAGE.get(e.queue[0], 0) > 0
              and _hits_hero(s, e, e.queue[0])}
    # an aimed tile (Volley) takes the hero's cell as the turn begins, before he acts: the warning sign at his
    # feet [rec: collect/20261029-160032 f0199, f0203, f0215, f0222; bot: run 20261030-115221 t174, t178]
    aims = {e.uid: s.hero.x for e in s.alive()
            if e.uid in primed and e.aim is None and (SPEC[e.queue[0]][0] == "aim" or (
                e.alt is not None and SPEC[ENEMY_KINDS[e.alt][1]][0] == "aim"))}
    s1 = apply_hero(s, action, ev0)
    for e in s1.enemies:
        e.primed = e.uid in primed
        if e.uid in aims:
            e.aim = aims[e.uid]
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
