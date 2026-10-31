"""Game state for the simulator.

Game facts (tile damage and cooldowns, enemy HP and tiles, the hero's kit)
come from the catalogue (shogunbot/data/*.json, see catalogue.py). Cells are
indexed 0..n-1 from the left; facing is +1 (right) or -1 (left).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

from .catalogue import CAT

Facing = Literal[-1, 1]

# ---- hero -------------------------------------------------------------------
HERO_ID = "wanderer"
_HERO = CAT.hero(HERO_ID)
HERO_MAX_HP: int = _HERO.hp
SWAP_COOLDOWN: int = _HERO.special["cooldown"]
QUEUE_MAX: int = CAT.hero_rules["queue_max"]
START_LEVELS: int = CAT.hero_rules["starting_levels"]


@dataclass(frozen=True)
class KitTile:
    """One tile the hero owns, with its current (possibly upgraded) stats."""
    label: str              # unique in the kit: the tile id, '#2' added for a second copy
    tile: str               # catalogue tile id
    damage: int
    cooldown: int
    enchant: str | None = None
    levels: int = START_LEVELS   # max level: the dots above the tile
    used: int = 0                # levels used by upgrades

    @property
    def free_levels(self) -> int:
        return self.levels - self.used


def skill_max_hp(sid: str, level: int) -> int:
    """Max HP a skill adds at this level: Healthy +2 a level; Fortress and Iron Skin cost some [wiki]."""
    sk = CAT.skill(sid)
    add = sk.amount(level) if sk.effect.get("type") == "max_hp" else 0
    return int(add or 0) + int(sk.effect.get("max_hp", 0))


def skill_special_cooldown(sid: str, level: int) -> int:
    """Change to the special move's cooldown: Quick Recovery -1, -2, -3 by level; Mamushi Move +1 [wiki]."""
    sk = CAT.skill(sid)
    add = sk.amount(level) if sk.effect.get("type") == "special_cooldown" else 0
    return int(add or 0) + int(sk.effect.get("special_cooldown", 0))


class Kit:
    """The hero's build: tiles in hand order and the skills bought in shops ((id, level) pairs,
    data/skills.json). Immutable, so search nodes share one."""
    __slots__ = ("tiles", "by_label", "labels", "queue_actions", "cooldowns", "ready_values", "skills",
                 "skill_levels", "max_hp", "swap_cooldown", "swap_both_ways", "fx", "_hash")

    def __init__(self, tiles, skills=()):
        self.tiles: tuple[KitTile, ...] = tuple(tiles)
        self.by_label = {t.label: t for t in self.tiles}
        if len(self.by_label) != len(self.tiles):
            raise ValueError(f"duplicate kit labels: {[t.label for t in self.tiles]}")
        self.labels = tuple(t.label for t in self.tiles)
        self.queue_actions = tuple("queue:" + lb for lb in self.labels)
        # (label, cooldown) and (label, cooldown, planner ready value), for the hot loops
        self.cooldowns = tuple((t.label, t.cooldown) for t in self.tiles)
        self.ready_values = tuple((t.label, t.cooldown, CAT.tile(t.tile).ready_value) for t in self.tiles)
        lv: dict[str, int] = {}
        for sid, n in (skills.items() if isinstance(skills, dict) else skills):
            if sid not in CAT.skills:
                raise ValueError(f"unknown skill {sid}")
            lv[sid] = max(1, min(int(n), CAT.skill(sid).max_level))
        self.skills: tuple[tuple[str, int], ...] = tuple(sorted(lv.items()))
        self.skill_levels = dict(self.skills)
        # the special move's cooldown never drops below 1 [wiki: Quick Recovery]
        self.max_hp = HERO_MAX_HP + sum(skill_max_hp(sid, n) for sid, n in self.skills)
        self.swap_cooldown = max(1, SWAP_COOLDOWN + sum(skill_special_cooldown(sid, n) for sid, n in self.skills))
        # Two-Way Move: the swap also works with a unit behind the hero [wiki] [rec2 f0491-f0492]
        self.swap_both_ways = any(CAT.skill(sid).effect.get("type") == "special_both_ways" for sid, _ in self.skills)
        # the build's simulated skill effects: effect type -> amount at its level (rules.py reads these)
        fx: dict[str, int] = {}
        for sid, n in self.skills:
            sk = CAT.skill(sid)
            if sk.sim:
                a = sk.amount(n)
                fx[sk.effect["type"]] = fx.get(sk.effect["type"], 0) + (int(a) if isinstance(a, (int, float)) else 1)
        self.fx = fx
        self._hash = hash((self.tiles, self.skills))

    def level(self, sid: str) -> int:
        """The level of a skill in the build, 0 when not owned."""
        return self.skill_levels.get(sid, 0)

    def with_skill(self, sid: str) -> "Kit":
        """A new kit with the skill bought: a copy of an owned skill raises its level, up to its max [wiki]."""
        lv = dict(self.skills)
        lv[sid] = min(lv.get(sid, 0) + 1, CAT.skill(sid).max_level)
        return Kit(self.tiles, lv)

    def __getitem__(self, label: str) -> KitTile:
        return self.by_label[label]

    def __contains__(self, label: str) -> bool:
        return label in self.by_label

    def __iter__(self):
        return iter(self.tiles)

    def __len__(self) -> int:
        return len(self.tiles)

    def __eq__(self, other) -> bool:
        return isinstance(other, Kit) and other.tiles == self.tiles and other.skills == self.skills

    def __hash__(self) -> int:
        return self._hash

    def __deepcopy__(self, memo):
        return self  # immutable

    def __repr__(self) -> str:
        return "Kit(" + ", ".join([f"{t.label}:{t.damage}/{t.cooldown}" + (f"+{t.enchant}" if t.enchant else "")
                                   for t in self.tiles] + [f"{sid} {n}" for sid, n in self.skills]) + ")"

    def to_list(self) -> list[dict]:
        """The tiles as dicts, then one {'skill', 'level'} dict per skill (logs, the fight cache's key)."""
        return [asdict(t) for t in self.tiles] + [{"skill": sid, "level": n} for sid, n in self.skills]

    @staticmethod
    def from_list(items) -> "Kit":
        items = list(items)
        return Kit((KitTile(**d) for d in items if "skill" not in d),
                   [(d["skill"], d["level"]) for d in items if "skill" in d])

    @staticmethod
    def of(*tile_ids: str) -> "Kit":
        """Catalogue stats for each tile id, in order."""
        k = Kit(())
        for t in tile_ids:
            k = k.add(t)
        return k

    def add(self, tile_id: str, damage: int | None = None, cooldown: int | None = None,
            levels: int = START_LEVELS, used: int = 0, enchant: str | None = None) -> "Kit":
        """A new kit with this tile appended (a second copy gets label 'id#2')."""
        t = CAT.tile(tile_id)
        label, n = tile_id, 1
        while label in self.by_label:
            n += 1
            label = f"{tile_id}#{n}"
        kt = KitTile(label, tile_id, t.damage if damage is None else damage,
                     t.cooldown if cooldown is None else cooldown, enchant, levels, used)
        return Kit(self.tiles + (kt,), self.skills)

    def replace(self, label: str, new: KitTile) -> "Kit":
        return Kit((new if t.label == label else t for t in self.tiles), self.skills)

    def remove(self, label: str) -> "Kit":
        return Kit((t for t in self.tiles if t.label != label), self.skills)


def default_kit(hero: str = HERO_ID, tile_set: int = 0) -> Kit:
    """The hero's starting tiles from the catalogue (Wanderer set 1: Swirl, Arrow)."""
    return Kit.of(*CAT.hero(hero).tile_sets[tile_set])


DEFAULT_KIT = default_kit()
# name: (damage, cooldown) of the default kit, for code that predates the dynamic kit
HERO_TILES: dict[str, tuple[int, int]] = {t.label: (t.damage, t.cooldown) for t in DEFAULT_KIT}

# ---- enemies ----------------------------------------------------------------
# enemy id: (max_hp, default tile), for every unit the simulator has presets for
ENEMY_KINDS: dict[str, tuple[int, str]] = {e.id: (e.hp, e.tile) for e in CAT.enemies.values() if e.sim}
_ENEMY_TILES = sorted({t for e in CAT.enemies.values() if e.sim for t in e.tiles})
# tile id: damage / cooldown of the tiles those units carry. Sword and Summon
# cooldowns are hypotheses (catalogue cooldown_source); logged re-arm timing settles them.
ENEMY_TILE_DAMAGE: dict[str, int] = {t: CAT.tile(t).damage or 0 for t in _ENEMY_TILES}
ENEMY_TILE_COOLDOWN: dict[str, int] = {t: CAT.tile(t).cooldown or 0 for t in _ENEMY_TILES}
ENEMY_TAG: dict[str, str] = {e.id: e.tag for e in CAT.enemies.values()}
# simulated bosses (no HP pips: HP from the boss bar) [catalogue]: Daisuke, and Rei since 0.27
BOSS_KINDS: frozenset[str] = frozenset(e.id for e in CAT.enemies.values() if e.boss and e.sim)
# Heavy [wiki: Traits, 'cannot be turned or moved']: The Statue and its Summoner never move or turn in their fights
# [rec: collect/20261028-190026 f0857-f0921, collect/20261029-160032 f1071-f1120]; the hero cannot swap with them,
# and Rain of Mirrors does not turn them
HEAVY_KINDS: frozenset[str] = frozenset(e.id for e in CAT.enemies.values() if e.sim and "heavy" in e.traits)


@dataclass
class Enemy:
    kind: str
    x: int
    facing: Facing
    hp: int
    queue: list[str] = field(default_factory=list)  # bottom first
    uid: int = 0
    frozen: int = 0  # turns left frozen (Mass Ice, Ice enchantment)
    cd: int = 0      # turns until it may queue again (HYP)
    # telegraphed action for the coming enemy phase, read from the screen:
    # 'fire' | 'queue' | 'turn' | 'move-left' | 'move-right' | 'hold' | None (unknown)
    intent: str | None = None
    # Charger only: loaded with a clear lane to the hero when the hero's turn
    # began. It charges in that enemy phase; a lane opened by the hero's own
    # action is used the phase after [run 20261027-180145, turns 59-60].
    primed: bool = False
    # phases since the queue was completed. R1: a unit fires at the earliest one
    # phase after loading (red, then yellow). Units read from the screen start at 1.
    wind: int = 1
    cursed: bool = False  # Mass Curse / Curse enchantment: next hit deals double damage
    poison: int = 0       # turns of poison left: 1 damage at the end of each turn [wiki]
    # history for the learned behaviour model (behaviour.py): the class of the
    # last action it telegraphed (fire, queue, turn, move, hold; None = unknown,
    # e.g. just spawned) and phases since it last fired (None = not seen firing)
    last: str | None = None
    fage: int | None = None
    # a kind the screen read could not rule out (an Ashigaru that may be an elite Tachi): when
    # it fires, the hero is also hit where alt's tile would reach (rules._hits_hero, _enemy_fire)
    # [user review of run 20261029-045856]
    alt: str | None = None
    elite: bool = False   # an elite variant (Day 2 on): emblem on its plate, extra HP; traits not modelled
    # an aimed tile (the Yumi Sniper's Volley): the cell it strikes this turn, the hero's cell when the turn
    # began, shown by a warning sign; None until the turn it fires (rules.step sets it)
    aim: int | None = None
    # a shield (the Shielder's Shield or Ally Shield, Kowa's Shield): negates the next attack received [wiki:
    # Tiles]; a blue orb in front of the unit on screen (perception.shields)
    shield: bool = False

    @property
    def armed(self) -> bool:
        return bool(self.queue)

    @property
    def max_hp(self) -> int:
        return ENEMY_KINDS.get(self.kind, (self.hp, None))[0] or self.hp

    def clone(self) -> "Enemy":
        return Enemy(self.kind, self.x, self.facing, self.hp, list(self.queue), self.uid, self.frozen, self.cd,
                     self.intent, self.primed, self.wind, self.cursed, self.poison, self.last, self.fage,
                     self.alt, self.elite, self.aim, self.shield)

    def key(self) -> tuple:
        return (self.kind, self.x, self.facing, self.hp, tuple(self.queue), self.frozen, self.cd, self.intent,
                self.primed, min(self.wind, 1), self.cursed, self.poison, self.last,
                None if self.fage is None else min(self.fage, 2), self.alt, self.aim, self.shield)


@dataclass
class Hero:
    x: int
    facing: Facing
    hp: int = HERO_MAX_HP
    queue: list[str] = field(default_factory=list)  # kit labels, bottom first
    # current charge per kit label; a tile is ready when charge == its cooldown.
    # None = all full (level start).
    charge: dict[str, int] | None = None
    swap_charge: int | None = None   # None = ready (the kit's swap cooldown)
    shield: bool = False  # Kami Brew: negates the next attack received [wiki]
    kit: Kit = DEFAULT_KIT

    def __post_init__(self) -> None:
        if self.charge is None:
            self.charge = {t.label: t.cooldown for t in self.kit.tiles}
        if self.swap_charge is None:
            self.swap_charge = self.kit.swap_cooldown

    def ready(self, label: str) -> bool:
        t = self.kit.by_label.get(label)
        return t is not None and label not in self.queue and self.charge.get(label, 0) >= t.cooldown

    def ready_labels(self) -> list[str]:
        q, ch = self.queue, self.charge
        return [lb for lb, cd in self.kit.cooldowns if lb not in q and ch.get(lb, 0) >= cd]

    @property
    def swap_ready(self) -> bool:
        return self.swap_charge >= self.kit.swap_cooldown

    def clone(self) -> "Hero":
        return Hero(self.x, self.facing, self.hp, list(self.queue), dict(self.charge), self.swap_charge,
                    self.shield, self.kit)

    def key(self) -> tuple:
        # the kit is constant within one search, so it is not part of the key
        return (self.x, self.facing, self.hp, tuple(self.queue), tuple(sorted(self.charge.items())),
                self.swap_charge, self.shield)


@dataclass
class State:
    cells: int
    hero: Hero
    enemies: list[Enemy] = field(default_factory=list)
    level: str = ""          # e.g. "bamboo-1", "bamboo-2", "boss"
    wave: int = 0
    waves_total: int = 0
    turn: int = 0

    def clone(self) -> "State":
        return State(self.cells, self.hero.clone(), [e.clone() for e in self.enemies],
                     self.level, self.wave, self.waves_total, self.turn)

    def key(self) -> tuple:
        return (self.cells, self.hero.key(), tuple(sorted(e.key() for e in self.enemies if e.hp > 0)),
                self.wave, self.waves_total)

    # ---- helpers ------------------------------------------------------------
    def unit_at(self, x: int):
        if self.hero.x == x:
            return self.hero
        for e in self.enemies:
            if e.x == x and e.hp > 0:
                return e
        return None

    def enemy_at(self, x: int) -> Enemy | None:
        for e in self.enemies:
            if e.x == x and e.hp > 0:
                return e
        return None

    def in_bounds(self, x: int) -> bool:
        return 0 <= x < self.cells

    def alive(self) -> list[Enemy]:
        return [e for e in self.enemies if e.hp > 0]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hero"]["kit"] = self.hero.kit.to_list()
        return d

    @staticmethod
    def from_dict(d: dict) -> "State":
        hd = dict(d["hero"])
        kit = Kit.from_list(hd.pop("kit")) if hd.get("kit") else DEFAULT_KIT
        hero = Hero(**hd, kit=kit)
        enemies = [Enemy(**e) for e in d.get("enemies", [])]
        rest = {k: v for k, v in d.items() if k not in ("hero", "enemies")}
        return State(hero=hero, enemies=enemies, **rest)

    def render(self) -> str:
        """One-line board, e.g. '. A> <H T< .' for logs."""
        out = []
        for x in range(self.cells):
            if self.hero.x == x:
                out.append(("<" if self.hero.facing < 0 else "") + "H" + (">" if self.hero.facing > 0 else ""))
                continue
            e = self.enemy_at(x)
            if e is None:
                out.append(".")
            else:
                tag = ENEMY_TAG.get(e.kind, "?")
                tag += str(e.hp)
                if e.shield:
                    tag += "o"      # shielded
                if e.armed:
                    tag += "*"
                out.append(("<" if e.facing < 0 else "") + tag + (">" if e.facing > 0 else ""))
        return " ".join(out)
