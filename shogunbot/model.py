"""Game state for Stage 1 (Bamboo Grove), Wanderer with the default kit.

Every number here comes from the Stage 1 domain model
(project doc claude/stage1-domain-model.md). Cells are indexed 0..n-1 from
the left; facing is +1 (right) or -1 (left).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Facing = Literal[-1, 1]

# ---- tiles ------------------------------------------------------------------
# name: (damage, cooldown)
HERO_TILES: dict[str, tuple[int, int]] = {
    "swirl": (2, 3),  # ahead and behind
    "arrow": (2, 5),  # first unit ahead, any range
}
SWAP_COOLDOWN = 4
HERO_MAX_HP = 10
QUEUE_MAX = 3

# enemy tile name: damage
ENEMY_TILE_DAMAGE: dict[str, int] = {
    "sword": 2,   # cell directly ahead
    "swirl": 2,   # ahead and behind
    "charge": 1,  # dash forward to first unit ahead, then strike it
    "summon": 0,
}

# HYP: enemy tiles recharge like hero tiles (wiki cooldowns: Swirl 3, Charge 4).
# Ashigaru "Sword" cooldown is unknown; logged re-arm timing will settle it.
ENEMY_TILE_COOLDOWN: dict[str, int] = {"sword": 2, "swirl": 3, "charge": 4, "summon": 6}

# enemy kind: (max_hp, default tile)
ENEMY_KINDS: dict[str, tuple[int, str]] = {
    "ashigaru": (3, "sword"),
    "tachi": (2, "swirl"),
    "charger": (1, "charge"),
    "daisuke": (15, "sword"),
}


@dataclass
class Enemy:
    kind: str
    x: int
    facing: Facing
    hp: int
    queue: list[str] = field(default_factory=list)  # bottom first
    uid: int = 0
    frozen: int = 0  # unused in default kit, kept for later
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
    cursed: bool = False  # Mass Curse: next hit deals double damage

    @property
    def armed(self) -> bool:
        return bool(self.queue)

    def clone(self) -> "Enemy":
        return Enemy(self.kind, self.x, self.facing, self.hp, list(self.queue), self.uid, self.frozen, self.cd, self.intent, self.primed, self.wind, self.cursed)

    def key(self) -> tuple:
        return (self.kind, self.x, self.facing, self.hp, tuple(self.queue), self.frozen, self.cd, self.intent, self.primed, min(self.wind, 1), self.cursed)


@dataclass
class Hero:
    x: int
    facing: Facing
    hp: int = HERO_MAX_HP
    queue: list[str] = field(default_factory=list)  # bottom first
    # current charge per tile; a tile is ready when charge == cooldown
    charge: dict[str, int] = field(default_factory=lambda: {k: v[1] for k, v in HERO_TILES.items()})
    swap_charge: int = SWAP_COOLDOWN
    shield: bool = False  # Kami Brew: negates the next attack received [wiki]

    def ready(self, tile: str) -> bool:
        return tile not in self.queue and self.charge[tile] >= HERO_TILES[tile][1]

    @property
    def swap_ready(self) -> bool:
        return self.swap_charge >= SWAP_COOLDOWN

    def clone(self) -> "Hero":
        return Hero(self.x, self.facing, self.hp, list(self.queue), dict(self.charge), self.swap_charge, self.shield)

    def key(self) -> tuple:
        return (self.x, self.facing, self.hp, tuple(self.queue), tuple(sorted(self.charge.items())), self.swap_charge, self.shield)


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
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "State":
        hero = Hero(**d["hero"])
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
                tag = {"ashigaru": "A", "tachi": "T", "charger": "C", "daisuke": "D"}.get(e.kind, "?")
                tag += str(e.hp)
                if e.armed:
                    tag += "*"
                out.append(("<" if e.facing < 0 else "") + tag + (">" if e.facing > 0 else ""))
        return " ".join(out)
