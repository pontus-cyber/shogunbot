"""Perception -> model State, plus the memory the screen does not show.

The screen shows positions, HP, facing, queues, telegraphs, which hand tiles
are ready, and swap ticks. It does not show how far a recharging tile has
come, so the Tracker carries hero tile charge between turns using the same
rules the simulator applies, and reconciles it with the ready flags it sees.
"""

from __future__ import annotations

from .model import HERO_TILES, SWAP_COOLDOWN, ENEMY_KINDS, Enemy, Hero, State
from .rules import apply_hero, Events

DAISUKE_HP = 15
DAISUKE_TILES = ("sword", "swirl", "charge", "summon")


class Tracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.charge = {k: v[1] for k, v in HERO_TILES.items()}
        self.facing = 1
        self.facing_known = False
        self.facing_disagree = 0
        self.level = ""
        self.notes: list[str] = []
        self.prev: list = []          # enemies of the last state (for facing persistence)
        self.swapped_uid_x: int | None = None

    # ------------------------------------------------------------------
    def state(self, p) -> State | None:
        """Build a State from a combat/boss Perception. None if no hero."""
        if p.hero is None:
            return None
        self.notes = []
        cells = len(p.cells)
        if p.screen == "boss" or (cells == 5 and self.level in ("bamboo-2", "boss")):
            # the boss arena shows a turn or two before the boss bar appears
            level = "boss"
        else:
            level = "bamboo-1" if cells == 5 else "bamboo-2" if cells == 7 else f"cells-{cells}"
        if level != self.level:
            # new level: every tile starts fully recharged [wiki]
            self.charge = {k: v[1] for k, v in HERO_TILES.items()}
            self.level = level
            self.facing_known = False
            self.prev = []

        h = p.hero
        facing = self._hero_facing(h, p.confidence.get("hero_facing", 0.0))

        queue = [t for t in h["queue"] if t in HERO_TILES]
        charge = dict(self.charge)
        seen = {t["tile"]: t["ready"] for t in p.hand if t["tile"] in HERO_TILES}
        for tile, (_, cd) in HERO_TILES.items():
            if tile in queue:
                continue
            if tile in seen:
                if seen[tile]:
                    if charge[tile] < cd:
                        self.notes.append(f"{tile} ready earlier than tracked ({charge[tile]}/{cd})")
                    charge[tile] = cd
                elif charge[tile] >= cd:
                    self.notes.append(f"{tile} shown recharging but tracked full; set {cd - 1}")
                    charge[tile] = cd - 1
        self.charge = charge

        hp = h["hp"] if h["hp_max"] else 0
        swap_charge = SWAP_COOLDOWN - min(SWAP_COOLDOWN, h.get("swap_red", 0))
        hero = Hero(h["x"], facing, hp=hp, queue=queue, charge=charge, swap_charge=swap_charge)

        enemies = []
        for i, e in enumerate(p.enemies):
            kind = e["kind"] if e["kind"] in ENEMY_KINDS else "ashigaru"
            if kind != e["kind"]:
                self.notes.append(f"enemy at {e['x']} read as {e['kind']}; assumed ashigaru")
            if kind == "daisuke":
                hp_e = max(1, round((p.boss_frac or 1.0) * DAISUKE_HP))
            else:
                hp_e = e["hp"] or ENEMY_KINDS[kind][0]
            intent = e.get("intent") or "hold"
            queue = list(e["queue"])
            if kind == "daisuke":
                # his tiles are Sword, Swirl, Charge, Summon; other reads are icon misses
                # [run 20261027-182237 crashed on a 'spear' read]
                fixed = [t if t in DAISUKE_TILES else "sword" for t in queue]
                if fixed != queue:
                    self.notes.append(f"daisuke queue read {queue}; set to {fixed}")
                queue = fixed
            elif queue and any(t != ENEMY_KINDS[kind][1] for t in queue):
                # regular enemies only carry their own tile; an odd read is an animation overlay
                self.notes.append(f"enemy at {e['x']} queue read {queue}; set to {ENEMY_KINDS[kind][1]}")
                queue = [ENEMY_KINDS[kind][1]]
            facing_e = self._enemy_facing(kind, e, queue, intent, h["x"],
                                          p.confidence.get(f"enemy{e['x']}", 0.0))
            enemies.append(Enemy(kind, e["x"], facing_e, hp_e, queue, uid=i + 1, intent=intent))
        bosses = [en for en in enemies if en.kind == "daisuke"]
        if len(bosses) > 1:
            # one boss only: keep the read nearest his last known cell, the rest are
            # summoned Chargers whose pips were hidden [run 20261028-053637]
            last = next((q[1] for q in self.prev if q[0] == "daisuke"), None)
            keep = min(bosses, key=lambda en: abs(en.x - last)) if last is not None else max(bosses, key=lambda en: len(en.queue))
            for en in bosses:
                if en is not keep:
                    en.kind, en.hp = "charger", 1
                    en.queue = [t for t in en.queue if t == "charge"][:1]
            self.notes.append(f"{len(bosses)} Daisuke reads; kept cell {keep.x}")
        self.prev = [(en.kind, en.x, en.facing, en.intent) for en in enemies]
        self.swapped_uid_x = None

        s = State(cells, hero, enemies, level=level,
                  wave=1, waves_total=0 if level == "boss" else 99)
        return s

    # ------------------------------------------------------------------ facing
    def _hero_facing(self, h: dict, conf: float) -> int:
        """The hero only changes facing by the turn action, so after the first
        confident read the tracked value is used. A confident contrary read on
        two consecutive frames overrides it (e.g. a turn key that did not land).
        [run 20261027-180145: 9 hero-facing misreads from the sprite alone]"""
        seen = h["facing"]
        if not self.facing_known:
            if seen and conf >= 0.5:
                self.facing, self.facing_known = seen, True
            elif seen:
                self.facing = seen
            return self.facing
        if seen and seen != self.facing and conf >= 0.9:
            self.facing_disagree += 1
            if self.facing_disagree >= 2:
                self.notes.append(f"hero facing re-read as {seen}")
                self.facing, self.facing_disagree = seen, 0
        else:
            self.facing_disagree = 0
        return self.facing

    def _enemy_facing(self, kind: str, e: dict, queue: list, intent: str, hero_x: int, conf: float) -> int:
        """Game logic first, sprite last [run 20261027-180145: wind-up poses
        and chargers next to the hero were misread and cost 5 HP].

        1. A loaded Sword/Charge enemy, or one telegraphing 'fire', faces the
           hero: enemies queue only once they face the hero, and a yellow
           bubble means the hero is in the footprint. A Charger facing the hero
           never turns away [user].
        2. An enemy telegraphing 'turn' is facing away from the hero now.
        3. Otherwise facing persists from the last frame (enemies change facing
           only by a telegraphed turn), matched by kind within one cell.
        4. New units: the sprite classifier.
        """
        toward = 1 if hero_x > e["x"] else -1
        directional = kind in ("ashigaru", "charger") or (kind == "daisuke" and queue and queue[0] in ("sword", "charge"))
        if directional and (queue or intent == "fire") and self.swapped_uid_x != e["x"]:
            return toward
        if intent == "turn":
            return -toward
        prev = [q for q in self.prev if q[0] == kind and abs(q[1] - e["x"]) <= 1]
        if prev:
            pk = min(prev, key=lambda q: abs(q[1] - e["x"]))
            f = -pk[2] if pk[3] == "turn" else pk[2]
            if e["facing"] and e["facing"] != f and conf >= 0.95:
                self.notes.append(f"enemy at {e['x']}: sprite says {e['facing']} with high confidence, kept {f}")
            return f
        return e["facing"] or toward

    def acted(self, s: State, action: str) -> None:
        """Advance tracked charge and facing through the hero's action."""
        s2 = apply_hero(s, action, Events())
        self.charge = dict(s2.hero.charge)
        if action == "turn":
            self.facing = -s.hero.facing
        if s2.hero.x != s.hero.x and s.enemy_at(s2.hero.x) is not None:
            # swap: the enemy now stands where the hero was and keeps its facing
            self.swapped_uid_x = s.hero.x
            self.prev = [(k, s.hero.x if x == s2.hero.x else x, f, None) for k, x, f, _ in self.prev]
