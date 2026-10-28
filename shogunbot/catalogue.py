"""Game catalogue: tiles, enemies, traits, consumables, upgrades, regions, heroes.

The JSON files in shogunbot/data/ hold every game fact the bot uses, each
with its source tag ([wiki], [obs], [bot], [user], [hyp]). Code reads game
facts through this module only, so a new region or tile is a data change.

    from shogunbot.catalogue import CAT
    CAT.tile("swirl").damage                         # 2
    CAT.enemy("ashigaru").hp                         # 3
    CAT.consumable_by_screen("green_potion").effect  # {'type': 'heal', 'amount': 3}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Tile:
    id: str
    name: str
    damage: int | None
    cooldown: int | None
    attack: dict = field(compare=False, hash=False)
    sim: bool = False
    free_play: bool = False
    skull_cost: int | None = None
    obtainable: bool = False
    text: str = ""
    ready_override: float | None = None

    @property
    def kind(self) -> str:
        return self.attack.get("kind", "other")

    @property
    def directional(self) -> bool:
        """True when the tile only reaches the facing side, so its user has to
        face the target before loading it (Sword, Charge); False for symmetric
        footprints (Swirl) and summons."""
        k = self.kind
        if k == "cells":
            offs = self.attack.get("offsets", [])
            return set(offs) != {-o for o in offs}
        return k in ("first", "furthest", "dash_strike", "dash_through")

    @property
    def base_pool(self) -> bool:
        """Available without unlocking: the tile-choice pool of a fresh profile."""
        return self.obtainable and self.skull_cost is None

    @property
    def ready_value(self) -> float:
        """Planner bonus for having this tile ready to queue.

        3 per damage point, +2 per extra cell the footprint covers, +2 for a dash
        (it doubles as a move). Gives Swirl 8 and Arrow 6, the hand-tuned
        weights the planner used through 0.13."""
        if self.ready_override is not None:
            return self.ready_override
        return ready_value(self.kind, self.attack, self.damage or 0)


def ready_value(kind: str, attack: dict, damage: int) -> float:
    v = 3.0 * damage
    if kind == "cells":
        v += 2.0 * (len(attack.get("offsets", [])) - 1)
    elif kind in ("dash_strike", "dash_through"):
        v += 2.0
    return v


@dataclass(frozen=True)
class EnemyKind:
    id: str
    name: str
    hp: int | None
    hp_day5: int | None
    traits: tuple
    region: str | None
    tiles: tuple
    tag: str
    standard: bool
    boss: bool
    sim: bool
    summons: str | None = None
    text: str = ""

    @property
    def tile(self) -> str | None:
        """The tile the simulator loads by default (the first listed)."""
        return self.tiles[0] if self.tiles else None


@dataclass(frozen=True)
class Consumable:
    id: str
    name: str
    kind: str
    effect: dict = field(compare=False, hash=False)
    screen_id: str | None = None
    look: str = ""
    text: str = ""
    sell: int | None = None


@dataclass(frozen=True)
class Upgrade:
    id: str
    name: str
    kind: str                 # 'stat' | 'enchant'
    lines: tuple              # shrine bubble text, upper case
    seen: bool
    damage: int = 0
    cooldown: int = 0
    slots: int = 0
    uses_level: bool = True
    removes_tile: bool = False
    unknown_effect: bool = False
    text: str = ""

    @property
    def enchant(self) -> str | None:
        return self.id if self.kind == "enchant" else None

    @property
    def screen_text(self) -> str:
        return " ".join(self.lines)


@dataclass(frozen=True)
class Hero:
    id: str
    name: str
    hp: int | None
    tile_sets: tuple
    special: dict = field(compare=False, hash=False)


class Catalogue:
    def __init__(self, root: Path = DATA):
        def load(name):
            return json.loads((root / name).read_text(encoding="utf-8"))

        t = load("tiles.json")
        self.tiles: dict[str, Tile] = {}
        for d in t["tiles"]:
            self.tiles[d["id"]] = Tile(
                id=d["id"], name=d["name"], damage=d.get("damage"), cooldown=d.get("cooldown"),
                attack=d.get("attack", {"kind": "other"}), sim=bool(d.get("sim")),
                free_play=bool(d.get("free_play")), skull_cost=d.get("skull_cost"),
                obtainable=bool(d.get("obtainable")), text=d.get("text", ""),
                ready_override=d.get("ready_value"))

        e = load("enemies.json")
        self.traits = {d["id"]: d for d in e["traits"]}
        self.elite_traits = {d["id"]: d for d in e["elite_traits"]}
        self.enemies: dict[str, EnemyKind] = {}
        for d in e["enemies"]:
            self.enemies[d["id"]] = EnemyKind(
                id=d["id"], name=d["name"], hp=d.get("hp"), hp_day5=d.get("hp_day5"),
                traits=tuple(d.get("traits", [])), region=d.get("region"), tiles=tuple(d.get("tiles", [])),
                tag=d.get("tag", "?"), standard=bool(d.get("standard")), boss=bool(d.get("boss")),
                sim=bool(d.get("sim")), summons=d.get("summons"), text=d.get("text", ""))

        c = load("consumables.json")
        self.consumable_rules = c["rules"]
        self.consumables: dict[str, Consumable] = {}
        for d in c["consumables"]:
            self.consumables[d["id"]] = Consumable(
                id=d["id"], name=d["name"], kind=d["kind"], effect=d["effect"],
                screen_id=d.get("screen_id"), look=d.get("look", ""), text=d.get("text", ""),
                sell=d.get("sell"))

        u = load("upgrades.json")
        self.upgrade_rules = u["rules"]
        self.upgrades: dict[str, Upgrade] = {}
        for d in u["stat_upgrades"]:
            self.upgrades[d["id"]] = Upgrade(
                id=d["id"], name=d["name"], kind="stat", lines=tuple(d["lines"]), seen=bool(d.get("seen")),
                damage=d.get("damage", 0), cooldown=d.get("cooldown", 0), slots=d.get("slots", 0),
                uses_level=d.get("uses_level", True), removes_tile=bool(d.get("removes_tile")),
                unknown_effect=bool(d.get("unknown_effect")))
        for d in u["enchantments"]:
            self.upgrades[d["id"]] = Upgrade(
                id=d["id"], name=d["name"], kind="enchant", lines=tuple(d["lines"]), seen=bool(d.get("seen")),
                cooldown=d.get("cooldown", 0), text=d.get("text", ""))
        self.shop_upgrades = u["shop_upgrades"]

        r = load("regions.json")
        self.region_rules = r["rules"]
        self.regions = {d["id"]: d for d in r["regions"]}
        self.levels = {lv["id"]: dict(lv, region=reg["id"])
                       for reg in r["regions"] for lv in reg.get("level_list", [])}

        h = load("heroes.json")
        self.hero_rules = h["rules"]
        self.heroes: dict[str, Hero] = {}
        for d in h["heroes"]:
            self.heroes[d["id"]] = Hero(id=d["id"], name=d["name"], hp=d.get("hp"),
                                        tile_sets=tuple(tuple(s) for s in d["tile_sets"]),
                                        special=d.get("special", {}))
        self._check()

    # ---- lookups -----------------------------------------------------------
    def tile(self, tid: str) -> Tile:
        return self.tiles[tid]

    def enemy(self, eid: str) -> EnemyKind:
        return self.enemies[eid]

    def hero(self, hid: str) -> Hero:
        return self.heroes[hid]

    def consumable(self, cid: str) -> Consumable:
        return self.consumables[cid]

    def consumable_by_screen(self, screen_id: str) -> Consumable | None:
        return next((c for c in self.consumables.values() if c.screen_id == screen_id), None)

    def upgrade(self, uid: str) -> Upgrade:
        return self.upgrades[uid]

    def region_enemies(self, region: str) -> list[EnemyKind]:
        """Standard enemies introduced in this region (bosses excluded)."""
        return [self.enemies[i] for i in self.regions[region]["enemies"]]

    def kinds_by_hp(self, region: str) -> dict[int, str]:
        """HP -> enemy id for a region's standard enemies (perception reads HP pips)."""
        out = {}
        for e in self.region_enemies(region):
            if e.hp in out:
                raise ValueError(f"{region}: {e.id} and {out[e.hp]} share {e.hp} HP; pips cannot tell them apart")
            out[e.hp] = e.id
        return out

    def kinds_by_tile(self, region: str) -> dict[str, str]:
        """Default tile -> enemy id for a region's standard enemies (perception reads bubble icons)."""
        return {e.tile: e.id for e in self.region_enemies(region) if e.tile}

    def base_pool(self) -> list[Tile]:
        return [t for t in self.tiles.values() if t.base_pool]

    def _check(self) -> None:
        for e in self.enemies.values():
            for t in e.tiles:
                if t not in self.tiles:
                    raise ValueError(f"enemy {e.id}: unknown tile {t}")
        for h in self.heroes.values():
            for s in h.tile_sets:
                for t in s:
                    if t not in self.tiles:
                        raise ValueError(f"hero {h.id}: unknown tile {t}")
        for reg in self.regions.values():
            for i in reg.get("enemies", []) + (reg.get("boss") or []):
                if i not in self.enemies:
                    raise ValueError(f"region {reg['id']}: unknown enemy {i}")


@lru_cache(maxsize=1)
def load() -> Catalogue:
    return Catalogue()


CAT = load()
