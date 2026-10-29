"""Perception -> model State, plus the memory the screen does not show.

The screen shows positions, HP, facing, queues, telegraphs, the hand tiles
with their charge (grey vs red cooldown marks), and swap ticks. The Tracker
keeps the hero's kit (which tiles, with what upgrades), the order of the
hero's queue, and charge for tiles it cannot see, and reconciles them with
each read.
"""

from __future__ import annotations

from itertools import permutations

from . import behaviour
from .catalogue import CAT
from .model import BOSS_KINDS, DEFAULT_KIT, SWAP_COOLDOWN, ENEMY_KINDS, Enemy, Hero, Kit, KitTile, State
from .rules import apply_hero, Events

BOSS_HP = {k: CAT.enemy(k).hp for k in BOSS_KINDS}         # the bar's full width [wiki]: Daisuke 15, Rei 13
BOSS_TILES = {k: CAT.enemy(k).tiles for k in BOSS_KINDS}
# sprites that do not show the facing: a new one is taken to face the hero. The Spike Charger is
# symmetric [0.20.1]; Iwao faces the viewer: in run 20261029-124921 (t159-t198) the hero was on his
# left on all 40 turns, and his sprite read 'facing right' on 12 of them, flipping 12 times
FRONT_FACING = ("charger", "iwao")
# hero tiles whose queue bubble shows another tile's art: a queued Katana has the enemy
# Sword's icon [run 20261028-092419 t117-t150: read as Sword or Spear, the bot thought
# Katana was still in the hand and tried to queue it 47 times]
BUBBLE_ALIAS = {"sword": "katana"}


def h_queue(p) -> list:
    return list((p.hero or {}).get("queue", []))


def match_faces(kit: Kit, faces: list[dict], charge: dict | None = None) -> dict[str, dict]:
    """Assign hand faces to kit labels by tile id. With copies of a tile (swirl,
    swirl#2) the assignment that best fits each copy's cooldown, damage and
    tracked charge wins, also when only some copies are in view."""
    charge = charge or {}
    out = {}
    tiles = []
    for kt in kit:
        if kt.tile not in tiles:
            tiles.append(kt.tile)
    for tile in tiles:
        labels = [kt.label for kt in kit if kt.tile == tile]
        fs = [f for f in faces if f.get("tile") == tile]
        if not fs:
            continue

        def cost(lb, f):
            kt = kit[lb]
            c = 0.0
            if f.get("cooldown") is not None and f["cooldown"] != kt.cooldown:
                c += 10
            if f.get("damage") is not None and f.get("damage_conf", 0) >= 0.75 and f["damage"] != kt.damage:
                c += 10
            want = charge.get(lb, kt.cooldown)
            return c + abs(f.get("charged", want) - want)

        if len(labels) == 1 and len(fs) == 1:
            out[labels[0]] = fs[0]
            continue
        best, best_c = None, None
        if len(labels) <= len(fs):
            for perm in permutations(fs, len(labels)):
                c = sum(cost(lb, f) for lb, f in zip(labels, perm))
                if best_c is None or c < best_c:
                    best, best_c = list(zip(labels, perm)), c
        else:
            for perm in permutations(labels, len(fs)):
                c = sum(cost(lb, f) for lb, f in zip(perm, fs))
                if best_c is None or c < best_c:
                    best, best_c = list(zip(perm, fs)), c
        out.update(dict(best))
    return out


def kit_from_faces(kit: Kit, faces: list[dict], notes: list | None = None) -> Kit:
    """The kit with stats confirmed or corrected by a full hand read (reward
    screens and level starts show every tile). Tiles the read finds that the
    kit lacks are added; kit tiles the read misses are kept."""
    notes = notes if notes is not None else []
    tiles = list(kit.tiles)
    matched = match_faces(kit, faces)
    for i, kt in enumerate(tiles):
        f = matched.get(kt.label)
        if f is None:
            continue
        dmg = f["damage"] if f.get("damage") is not None and f.get("damage_conf", 0) >= 0.75 else kt.damage
        cd = f.get("cooldown", kt.cooldown)
        levels = f.get("levels") or kt.levels
        used = f.get("used", kt.used) if f.get("levels") else kt.used
        if (dmg, cd, levels, used) != (kt.damage, kt.cooldown, kt.levels, kt.used):
            notes.append(f"{kt.label}: read {dmg}/{cd} L{used}/{levels}, kit had "
                         f"{kt.damage}/{kt.cooldown} L{kt.used}/{kt.levels}")
            tiles[i] = KitTile(kt.label, kt.tile, dmg, cd, kt.enchant, levels, used)
    new = Kit(tiles)
    seen = set(id(f) for f in matched.values())
    extra = [f for f in faces if f.get("tile") and id(f) not in seen and f["tile"] in CAT.tiles]
    if len(faces) <= len(kit):
        # as many faces as kit tiles: a kit tile nobody sees next to a face no kit tile
        # matches means the kit is wrong (e.g. a reward assumed taken that was not);
        # a confident read replaces it, otherwise the kit is kept [review of 0.15]
        unseen = [kt for kt in new if kt.label not in matched]
        for f in extra:
            sure = f.get("conf", 0) >= 0.5 and f.get("dist", 1.0) <= 0.03
            if unseen and sure and len(faces) == len(kit):
                old = unseen.pop(0)
                new = new.remove(old.label).add(
                    f["tile"], damage=f.get("damage") if f.get("damage_conf", 0) >= 0.75 else None,
                    cooldown=f.get("cooldown"), levels=f.get("levels") or 3, used=f.get("used") or 0)
                notes.append(f"kit had {old.label} but the hand shows {f['tile']}; replaced")
            else:
                notes.append(f"hand face read as {f['tile']} matches no kit tile; kit kept")
        return new
    for f in extra[:len(faces) - len(kit)]:
        t = f["tile"]
        new = new.add(t, damage=f.get("damage") if f.get("damage") is not None else None,
                      cooldown=f.get("cooldown"), levels=f.get("levels") or 3, used=f.get("used") or 0)
        notes.append(f"hand shows {t} the kit did not have; added")
    return new


STAGE1_LEVELS = ("bamboo-1", "bamboo-2", "boss")


def level_name(stage: int, i: int) -> str:
    """Stage 1 keeps its names; later stages are 'stage<k>-1', 'stage<k>-2', 'stage<k>-boss' until
    levels are named by region (M4 step 2)."""
    if stage == 1:
        return STAGE1_LEVELS[i]
    return f"stage{stage}-{'boss' if i == 2 else i + 1}"


class Tracker:
    def __init__(self, kit: Kit = DEFAULT_KIT) -> None:
        self.kit = kit
        self.reset()

    def set_kit(self, kit: Kit) -> None:
        """A reward changed the hero's tiles; new tiles start charged."""
        self.kit = kit
        self.charge = {t.label: max(t.cooldown, 0) for t in kit}
        self.queue_track = []

    def reset(self) -> None:
        self.charge = {t.label: t.cooldown for t in self.kit}
        self.queue_track: list[str] = []   # labels the bot queued, bottom first
        self.facing = 1
        self.facing_known = False
        self.facing_disagree = 0
        self.level = ""
        self.stage, self.level_i = 1, -1   # the run's stage and level within it (-1: no fight yet)
        self.advance = False               # a reward screen ended the level: the next fight is the next level
        self.notes: list[str] = []
        self.prev: list = []          # enemies of the last state (for facing persistence)
        self.fresh: list = []         # (x, kind, facing) of the units new in the last state
        self.hist: list = []          # enemies when the bot last acted: (kind, x, telegraph, fage), for the
                                      # behaviour model's history features (behaviour.carry)
        self.freeze_left = 0          # turns of Mass Ice left
        self.shield = False           # Kami Brew active
        self.cursed = False           # Mass Curse on every enemy until our next attack
        self.swapped_uid_x: int | None = None

    # ------------------------------------------------------------------
    def next_level(self) -> None:
        """A reward screen ended the current level (bot, replay): the next fight starts the
        next one. Levels only move forward [user review of run 20261029-072323: Rei the
        Ruthless's 7-cell arena, read by its width after the boss died, started a new level
        'bamboo-2' at the boss shrine]."""
        if self.level_i >= 0:
            self.advance = True

    def _level(self, screen: str, cells: int) -> str:
        stage, i = self.stage, self.level_i
        if self.advance:
            self.advance = False
            stage, i = (stage + 1, 0) if i >= 2 else (stage, i + 1)
        elif i < 0:                          # the run's first fight: by the board
            i = 2 if screen == "boss" else 1 if cells == 7 and stage == 1 else 0
        if screen == "boss":                 # the boss bar names this stage's boss
            i = 2
        elif stage == 1 and i == 0 and cells == 7:
            i = 1                            # Bamboo Grove II without its reward screen read
        elif stage == 1 and i == 1 and cells == 5:
            i = 2                            # Daisuke's arena shows a turn or two before the boss bar
        self.stage, self.level_i = stage, i
        return level_name(stage, i)

    def state(self, p) -> State | None:
        """Build a State from a combat/boss Perception. None if no hero."""
        if p.hero is None:
            return None
        self.notes = []
        cells = len(p.cells)
        level = self._level(p.screen, cells)
        if level != self.level:
            # new level: the hand shows every tile, so confirm the kit (a reward may
            # have changed it); every tile starts fully recharged [wiki]
            faces = [f for f in p.hand if f.get("tile")]
            if faces and not h_queue(p):
                before = self.kit
                self.kit = kit_from_faces(self.kit, faces, self.notes)
                if self.kit != before:
                    self.notes.append(f"kit at level start: {self.kit}")
            self.charge = {t.label: t.cooldown for t in self.kit}
            self.queue_track = []
            self.level = level
            self.facing_known = False
            self.prev = []
            self.hist = []

        h = p.hero
        facing = self._hero_facing(h, p.confidence.get("hero_facing", 0.0))

        kit = self.kit
        queue = self._queue(h, p.hand)
        charge = dict(self.charge)
        faces = match_faces(kit, p.hand, charge)
        hand_read = any(f.get("tile") for f in p.hand)
        unseen = []
        for kt in kit:
            tile, cd = kt.label, kt.cooldown
            if tile in queue:
                continue
            f = faces.get(tile)
            if f is None:
                if hand_read:
                    unseen.append(tile)
                continue
            if "charged" in f and f.get("cooldown") == cd:
                # the marks show the charge itself: grey = charged [run 20261028-061420 f0014-f0020]
                if f["charged"] != charge.get(tile):
                    self.notes.append(f"{tile} charge read {f['charged']}/{cd}, tracked {charge.get(tile)}")
                    charge[tile] = f["charged"]
            elif f["ready"]:
                if charge[tile] < cd:
                    self.notes.append(f"{tile} ready earlier than tracked ({charge[tile]}/{cd})")
                charge[tile] = cd
            elif charge[tile] >= cd:
                self.notes.append(f"{tile} shown recharging but tracked full; set {cd - 1}")
                charge[tile] = cd - 1
        self.charge = charge
        # a kit tile that is neither in the hand nor queued cannot be played now: plan it
        # as recharging. Planned as ready, the bot chose it and found no face to click
        # [review of 0.15]. The tracked charge itself is kept.
        plan_charge = dict(charge)
        for lb in unseen:
            cd = kit[lb].cooldown
            if plan_charge.get(lb, cd) >= cd:
                plan_charge[lb] = cd - 1
                self.notes.append(f"{lb} is not in the hand and not queued; planned as recharging")

        hp = h["hp"] if h["hp_max"] else 0
        # the red ticks under the hero count the turns his swap still needs (Quick Recovery shortens it)
        swap_charge = kit.swap_cooldown - min(kit.swap_cooldown, h.get("swap_red", 0))
        hero = Hero(h["x"], facing, hp=hp, queue=queue, charge=plan_charge, swap_charge=swap_charge,
                    shield=self.shield, kit=kit)

        enemies = []
        for i, e in enumerate(p.enemies):
            kind = e["kind"] if e["kind"] in ENEMY_KINDS else "ashigaru"
            if kind != e["kind"]:
                self.notes.append(f"enemy at {e['x']} read as {e['kind']}; assumed ashigaru")
            if kind in BOSS_KINDS:
                hp_e = max(1, round((p.boss_frac or 1.0) * BOSS_HP[kind]))
            else:
                hp_e = e["hp"] or ENEMY_KINDS[kind][0]
            intent = e.get("intent") or "hold"
            queue = list(e["queue"])
            if kind in BOSS_KINDS:
                # a boss loads only his own tiles (Daisuke: Sword, Swirl, Charge, Summon); other
                # reads are icon misses, set to his first tile [run 20261027-182237 crashed on a
                # 'spear' read]
                own = BOSS_TILES[kind]
                fixed = [t if t in own else own[0] for t in queue]
                if fixed != queue:
                    self.notes.append(f"{kind} queue read {queue}; set to {fixed}")
                queue = fixed
            elif queue and any(t != ENEMY_KINDS[kind][1] for t in queue):
                # regular enemies only carry their own tile; an odd read is an animation overlay
                self.notes.append(f"enemy at {e['x']} queue read {queue}; set to {ENEMY_KINDS[kind][1]}")
                queue = [ENEMY_KINDS[kind][1]]
            facing_e = self._enemy_facing(kind, e, queue, intent, h["x"],
                                          p.confidence.get(f"enemy{e['x']}", 0.0))
            alt = e.get("alt") if e.get("alt") in ENEMY_KINDS and e.get("alt") != kind else None
            if alt:
                self.notes.append(f"enemy at {e['x']}: read as {kind}, may be {alt}")
            en = Enemy(kind, e["x"], facing_e, hp_e, queue, uid=i + 1, intent=intent, alt=alt,
                       elite=bool(e.get("elite")))
            if self.freeze_left > 0:
                en.frozen, en.intent = self.freeze_left, None
            en.cursed = self.cursed
            enemies.append(en)
        bosses = [en for en in enemies if en.kind in BOSS_KINDS]
        if len(bosses) > 1:
            # one boss only: keep the read nearest his last known cell, the rest are
            # summoned units whose pips were hidden [run 20261028-053637: Chargers]
            last = next((q[1] for q in self.prev if q[0] in BOSS_KINDS), None)
            keep = min(bosses, key=lambda en: abs(en.x - last)) if last is not None else max(bosses, key=lambda en: len(en.queue))
            unit = CAT.enemy(keep.kind).summons or "charger"
            for en in bosses:
                if en is not keep:
                    en.kind, en.hp = unit, ENEMY_KINDS[unit][0]
                    en.queue = [t for t in en.queue if t == ENEMY_KINDS[unit][1]][:1]
            self.notes.append(f"{len(bosses)} {keep.kind} reads; kept cell {keep.x}")
        self.fresh = [(en.x, en.kind, en.facing) for en in self._unmatched(enemies)]
        self.prev = [(en.kind, en.x, en.facing, en.intent) for en in enemies]
        self.swapped_uid_x = None
        # what each enemy did last turn and how long since it fired: the learned model's
        # history features, matched the way shogunbot.mine matches logged turns
        for j, i in behaviour.match(self.hist, [(en.kind, en.x) for en in enemies]).items():
            _, _, it, fage = self.hist[i]
            enemies[j].last, enemies[j].fage = behaviour.carry(it, fage)

        s = State(cells, hero, enemies, level=level,
                  wave=1, waves_total=0 if level.endswith("boss") else 99)
        return s

    # ------------------------------------------------------------------ queue
    def _queue(self, h: dict, hand: list[dict]) -> list[str]:
        """The hero's queue, bottom first, as kit labels.

        Queued tiles leave the hand [run 20261028-061420 f0025], so the kit
        tiles missing from the hand are the queued ones; the number of bubbles
        over the hero confirms the count. Their order comes from the bubble
        icons where those name kit tiles, else from what the bot queued."""
        kit = self.kit
        bubbles = [BUBBLE_ALIAS.get(b, b) for b in h.get("queue", [])]
        in_hand = match_faces(kit, hand, self.charge)
        missing = [lb for lb in kit.labels if lb not in in_hand]
        track = [x for x in self.queue_track if x in kit]

        def rank(lb):
            return track.index(lb) if lb in track else len(track) + kit.labels.index(lb)

        if len(missing) == len(bubbles):
            # also with an empty hand: a 3-tile kit with 3 bubbles is a full queue [review of 0.15]
            tracked = sorted(missing, key=rank)
            if len(missing) > 4:
                return tracked

            def fit(order):
                # bubbles matched first; ties (misread bubbles, copies of a tile) go to
                # the order the bot queued in [review of 0.15]
                return (sum(kit[lb].tile == name for lb, name in zip(order, bubbles)),
                        sum(a == b for a, b in zip(order, tracked)))

            order = list(max(permutations(tracked), key=fit)) if tracked else []
            if fit(order)[0] < len(bubbles):
                self.notes.append(f"queue bubbles read {bubbles}; queue taken as {order}")
            return order
        # counts disagree (an animation, a misread face): the bubble icons, as kit labels
        q, left = [], sorted(kit.labels, key=lambda lb: (lb in in_hand, rank(lb)))
        for name in bubbles:
            lb = next((x for x in left if kit[x].tile == name), None)
            if lb is not None:
                q.append(lb)
                left.remove(lb)
        if hand:
            self.notes.append(f"{len(bubbles)} queue bubbles but {len(missing)} kit tiles off the hand")
        return q

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
           only by a telegraphed turn), from the unit of that kind that could
           have become this one (_could_be).
        4. New units: the sprite classifier; a Spike Charger faces the hero.

        Rules 3 and 4 were checked on the 345 new units of 21 runs (true facing
        from the next turn's telegraph). Rule 3 used to match any unit of the
        kind within one cell and handed 9 new units the facing of the unit they
        replaced. The Charger's sprite is symmetric: its read was right on 20 of
        42 new Chargers, and new Chargers face away 34 times in 65, so no read
        helps; facing the hero is the cautious guess, and the next turn's
        telegraph shows it [0.20.1, user review of the 0.20 series].
        """
        toward = 1 if hero_x > e["x"] else -1
        directional = kind in ("ashigaru", "charger") or (
            kind in BOSS_KINDS and queue and queue[0] in ("sword", "charge", "dragon_punch", "spear"))
        if directional and (queue or intent == "fire") and self.swapped_uid_x != e["x"]:
            return toward
        if intent == "turn":
            return -toward
        prev = [q for q in self.prev if q[0] == kind and self._could_be(q, e["x"])]
        if prev:
            pk = min(prev, key=lambda q: abs(q[1] - e["x"]))
            f = -pk[2] if pk[3] == "turn" else pk[2]
            if e["facing"] and e["facing"] != f and conf >= 0.95:
                self.notes.append(f"enemy at {e['x']}: sprite says {e['facing']} with high confidence, kept {f}")
            return f
        if kind in FRONT_FACING:
            return toward
        return e["facing"] or toward

    def _unmatched(self, enemies: list) -> list:
        """The enemies that none of last state's units can have become: new units (a
        wave, a summon), or every enemy at a level's first state. One old unit accounts
        for one enemy: same cell first, then a telegraphed move, a charge or a dash, or
        one cell for a unit whose telegraph is not known (after a swap or a mirror)."""
        free = list(range(len(self.prev)))
        left = []
        for en in enemies:
            i = next((i for i in free if self.prev[i][0] == en.kind and self.prev[i][1] == en.x), None)
            if i is None:
                left.append(en)
            else:
                free.remove(i)
        out = []
        for en in left:
            cand = [i for i in free if self.prev[i][0] == en.kind and (
                self._could_be(self.prev[i], en.x) or (self.prev[i][3] is None and abs(self.prev[i][1] - en.x) == 1))]
            if cand:
                free.remove(min(cand, key=lambda i: abs(self.prev[i][1] - en.x)))
            else:
                out.append(en)
        return out

    @staticmethod
    def _could_be(q: tuple, x: int) -> bool:
        """Could last turn's unit q = (kind, x, facing, intent) be the one now at x?
        A unit changes cell only by a telegraphed one-cell move, a Charger's charge
        along its facing, or a boss's dash; the hero's swap is applied to self.prev
        in advance()."""
        kind, px, f, intent = q
        if px == x or kind in BOSS_KINDS:
            return True
        step = x - px
        if abs(step) == 1 and intent == ("move-right" if step > 0 else "move-left"):
            return True
        return kind == "charger" and intent == "fire" and (step > 0) == (f > 0)

    def freeze(self, turns: int) -> None:
        self.freeze_left = turns

    def mirror(self) -> None:
        """Rain of Mirrors turned every enemy: flip the remembered facings, or the
        facing-persistence rule would undo the flip."""
        self.prev = [(k, x, -f, None) for k, x, f, _ in self.prev]

    def acted(self, s: State, action: str) -> None:
        """Advance tracked charge and facing through the hero's action."""
        if self.freeze_left > 0:
            self.freeze_left -= 1
        if action == "attack":
            self.cursed = False  # approximation: our attack uses the curse up
        if self.shield:
            from .rules import step
            outs = step(s, action)
            if outs and not max(outs, key=lambda t: t[0])[1].hero.shield:
                self.shield = False  # the likeliest outcome used it up
        s2 = apply_hero(s, action, Events())
        self.charge = dict(s2.hero.charge)
        self.queue_track = list(s2.hero.queue)
        # where each enemy stands once the hero has acted (a swap moves one), and its telegraph
        self.hist = [(e.kind, s.hero.x if (e.x == s2.hero.x and s2.hero.x != s.hero.x) else e.x, e.intent, e.fage)
                     for e in s.alive()]
        if action == "turn":
            self.facing = -s.hero.facing
        if s2.hero.x != s.hero.x and s.enemy_at(s2.hero.x) is not None:
            # swap: the enemy now stands where the hero was and keeps its facing
            self.swapped_uid_x = s.hero.x
            self.prev = [(k, s.hero.x if x == s2.hero.x else x, f, None) for k, x, f, _ in self.prev]
