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
from .perception.tiles import DOTS_PER_ROW
from .model import (BOSS_KINDS, DEFAULT_KIT, SWAP_COOLDOWN, ENEMY_KINDS, HEAVY_KINDS, UNFREEZABLE_KINDS, Enemy, Hero,
                    Kit, KitTile, State)
from .rules import CHAINS, ICE_TURNS, PHASED, SPEC, apply_hero, chained, Events, _enemy_fire

BOSS_HP = {k: CAT.enemy(k).hp for k in BOSS_KINDS}         # the bar's full width [wiki]: Daisuke 15, Rei 13
BOSS_TILES = {k: CAT.enemy(k).tiles for k in BOSS_KINDS}
# a Quick unit counts as loaded for the fire rule (Tracker.state) on a read of its own tile at least this sure: the
# Shinobi's Shuriken read 0.96-1.0 when it fired (rec1 f0694, f0711, f0712); reads of tiles it does not carry 0.02-0.16
QUICK_READ_CONF = 0.5
HERO_SHIELD_GONE = 0.005   # perceive.hero_shield below this: no shield around the hero (Tracker.state)
HERO_FACING_WEAK_N = 3     # weak contrary hero-facing reads in a row that override the tracked facing [0.42]
# sprites that do not show the facing: a new one is taken to face the hero. The Spike Charger is
# symmetric [0.20.1]; Iwao faces the viewer: in run 20261029-124921 (t159-t198) the hero was on his
# left on all 40 turns, and his sprite read 'facing right' on 12 of them, flipping 12 times
FRONT_FACING = ("charger", "iwao", "the_twins", "the_statue", "summoner", "kowa", "hideyoshi")   # the Twins' facing read
# '?' on most of their frames [rec: collect/20261028-190026 f0544-f0580]; The Statue (cell 6) and its Summoner (cell 0)
# face the hero from the start and never turn [rec: collect/20261028-190026 f0857-f0921]; Kowa starts on cell 4 facing
# the hero on cell 2 (rec1 f1274, rec2 f1475), and his moves keep his facing, so it follows from his turns; Hideyoshi
# starts on cell 4 facing the hero on cell 2 (rec1 f1528, rec2 f1691), and his facing follows his turns and his Mirror
# (Tracker._chain_rules)
# hero tiles whose queue bubble shows another tile's art: a queued Katana has the enemy
# Sword's icon [run 20261028-092419 t117-t150: read as Sword or Spear, the bot thought
# Katana was still in the hand and tried to queue it 47 times]
BUBBLE_ALIAS = {"sword": "katana"}
# a hero bubble read at this confidence or more names its tile (perceive.classify_hero_icon): on 3,489 bubbles of ten
# runs left out of hero_icons.npz, 98.8% of the right reads and 29 of the 91 wrong ones (0.8% of all) [0.47]; below it
# the order the bot queued in decides
BUBBLE_SURE = 0.3


def h_queue(p) -> list:
    return list((p.hero or {}).get("queue", []))


def hand_unsettled(p, kit: Kit) -> str | None:
    """Why the hand read and the hero's queue bubbles do not account for the kit, or None.

    Every kit tile is in the hand (bright or dimmed) or queued, so faces + bubbles = tiles on a
    settled screen. Two animations break that, and the screen's overall change stays under
    grab_stable's threshold while they play [user review of run 20261030-213120, item 2]: a tile that
    comes ready flashes (a white bar sweeps over its face: t62 read the Arrow as nothing and the
    Charge as an unknown face, t95 the Swirl as nothing), and after an attack the fired tiles fly back
    into the hand (t241, t291; t168 read an empty hand while the Arrow flashed). Of the 1,339 saved
    fight frames of the bot's runs, 9 read so, all in one of these two moments."""
    faces = list(getattr(p, "hand", None) or [])
    unread = sum(1 for f in faces if f.get("tile") is None)
    if unread:
        return f"{unread} hand face(s) unread"
    bubbles = h_queue(p)
    if len(faces) + len(bubbles) < len(kit):
        # (more than the kit is a tile taken at the last reward: the tracker adds it at the level's start)
        return f"{len(faces)} hand faces and {len(bubbles)} queue bubbles for {len(kit)} kit tiles"
    return None


def match_faces(kit: Kit, faces: list[dict], charge: dict | None = None, guess: bool = True,
                skip: tuple = ()) -> dict[str, dict]:
    """Assign hand faces to kit labels by tile id. With copies of a tile (swirl,
    swirl#2) the assignment that best fits each copy's cooldown, damage and
    tracked charge wins, also when only some copies are in view. guess: a face
    whose icon was not recognised goes to the one kit label left over (not in
    skip, e.g. queued) whose cooldown and damage it shows (_guess_unread)."""
    charge = charge or {}
    out = {}
    tiles = []
    for kt in kit:
        if kt.tile not in tiles:
            tiles.append(kt.tile)
    for tile in tiles:
        labels = [kt.label for kt in kit if kt.tile == tile]
        if len(labels) > 1 and skip:
            # copies: a queued copy is not in the hand (the tracker passes the queue as skip), so the faces go to the others
            labels = [lb for lb in labels if lb not in skip] or labels
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
            if f.get("levels"):
                # reward screens show the level dots: two Arrows at 2/5 with 4 and 3 levels swapped labels at
                # every read of run 20261102-232559 (reward_2_shrine: 'arrow: read 2/5 L0/4, kit had 2/5 L0/3')
                c += (f["levels"] != kt.levels) + (f.get("used", kt.used) != kt.used)
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
    if guess:
        _guess_unread(kit, faces, out, skip)
    return out


def _guess_unread(kit: Kit, faces: list[dict], out: dict, skip: tuple = ()) -> None:
    """An upgraded or enchanted face can miss its icon template (tile None, conf about 0.42): in run
    20261031-154544 the Swirl and the Shuriken were 'not found in hand' 30 times, the shrine lost 4
    upgrades, and in run 20261031-180506 swirl#2 stayed 'planned as recharging' for 200 turns while
    the hero had nothing to attack with [0.43]. Such a face is given to the only kit label not yet
    matched (and not in skip) whose cooldown it shows, and whose damage too when that read is sure;
    with no such label or more than one it stays unread."""
    free = [kt for kt in kit if kt.label not in out and kt.label not in skip]
    for f in faces:
        if f.get("tile") or f.get("cooldown") is None:
            continue
        cands = [kt for kt in free if kt.cooldown == f["cooldown"]
                 and (f.get("damage") is None or f.get("damage_conf", 0) < 0.75 or kt.damage == f["damage"])]
        if len(cands) == 1:
            kt = cands[0]
            f["tile"], f["guessed"] = kt.tile, True
            out[kt.label] = f
            free.remove(kt)


def kit_from_faces(kit: Kit, faces: list[dict], notes: list | None = None) -> Kit:
    """The kit with stats confirmed or corrected by a full hand read (reward
    screens and level starts show every tile). Tiles the read finds that the
    kit lacks are added; kit tiles the read misses are kept."""
    notes = notes if notes is not None else []
    tiles = list(kit.tiles)
    matched = match_faces(kit, faces, guess=False)
    for i, kt in enumerate(tiles):
        f = matched.get(kt.label)
        if f is None:
            continue
        dmg = f["damage"] if f.get("damage") is not None and f.get("damage_conf", 0) >= 0.75 else kt.damage
        cd = f.get("cooldown", kt.cooldown)
        levels = f.get("levels") or kt.levels
        if f.get("levels") and f["levels"] < kt.levels and f["levels"] % DOTS_PER_ROW == 0:
            # a read of full dot rows only, the kit has more: the stacked row above was missed (levels never go
            # down; before 0.49 one row was read, 4 for the Swirl's 6 at run 20261102-202249 reward_9) [item 16b]
            levels = kt.levels
        used = f.get("used", kt.used) if f.get("levels") else kt.used
        if (dmg, cd, levels, used) != (kt.damage, kt.cooldown, kt.levels, kt.used):
            notes.append(f"{kt.label}: read {dmg}/{cd} L{used}/{levels}, kit had "
                         f"{kt.damage}/{kt.cooldown} L{kt.used}/{kt.levels}")
            tiles[i] = KitTile(kt.label, kt.tile, dmg, cd, kt.enchant, levels, used)
    new = Kit(tiles, kit.skills)          # the hand shows tiles only: the skills stay
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


def level_name(stage: int, i: int, boss_i: int = 2) -> str:
    """Stage 1 keeps its names; later stages are 'stage<k>-1', 'stage<k>-2', ... and 'stage<k>-boss'
    (boss_i: the boss level's index in its region, 2 for a region of three levels)."""
    if stage == 1:
        return STAGE1_LEVELS[i]
    return f"stage{stage}-{'boss' if i >= boss_i else i + 1}"


def region_boss_index(region: str | None) -> int:
    """The boss level's index in a region: its level count from the catalogue less one (Moonlit Port
    has three wave levels before The Twins [rec: collect/20261028-190026, -20261029-160032]); 2 when
    the region is not known."""
    reg = CAT.regions.get(region) if region else None
    return max(1, int(reg["levels"]) - 1) if reg and reg.get("levels") else 2


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
        self.facing_weak = 0
        self.level = ""
        self.stage, self.level_i = 1, -1   # the run's stage and level within it (-1: no fight yet)
        self.route: list[str] = ["bamboo_grove"]   # the run's regions in order (bot, replay)
        self.advance = False               # a reward screen ended the level: the next fight is the next level
        self.notes: list[str] = []
        self.prev: list = []          # enemies of the last state (for facing persistence)
        self.prev_queues: list = []   # their queues, in the same order (a bubble that flashed as it fired)
        self.prev_hp: dict = {}       # cell -> HP of the last state's units
        self.fresh: list = []         # (x, kind, facing) of the units new in the last state
        self.hist: list = []          # enemies when the bot last acted: (kind, x, telegraph, fage), for the
                                      # behaviour model's history features (behaviour.carry)
        self.freeze_left = 0          # turns of Mass Ice left
        self.shield = False           # Kami Brew active
        self.last_hp: int | None = None   # the hero's HP in the last state
        self.hero_x_after: int | None = None   # the hero's cell after his last action, before the enemies moved
        self.cursed = False           # Mass Curse on every enemy until our next attack
        self.swapped_uid_x: int | None = None
        self.mirrored = False               # Rain of Mirrors turned every enemy since the last state (rule 1 waits)
        self.cycle: dict | None = None      # The Statue when the bot last acted: its tile count, stack, telegraph, holds
        self.loaded_tile: dict = {}         # kind -> the tile it loaded last, for a kind that alternates (Kowa)
        self._cycle_now: dict | None = None
        self._queue_conf: dict = {}         # cell -> the confidence of each tile read in its unit's bubbles
        self._orbs: set = set()             # cells whose unit shows the Shielder's fire orb
        self._canon: dict = {}              # kind -> its HP-pip tie group's first kind on this route (_same)
        self.chain_prev: dict = {}          # kind -> (queue, telegraph) of a combo boss when the bot last acted
        self.bombs: dict = {}               # cell -> enemy phases left: Hideyoshi's bombs on the floor (State.bombs)
        self.bombs_new: dict = {}           # bombs his combo places in the coming enemy phase (acted), cell -> phases left
        self.rings: list = []               # the Shogun's Corrupted Barrage rings (State.rings), as of the hero's turn
        self.rings_new: list = []           # rings his combo places in the coming enemy phase (acted)
        self.boss_phase: dict = {}          # kind -> the phase of a boss with more than one bar (the Shogun)
        self.boss_frac: dict = {}           # kind -> his bar's last read

    # ------------------------------------------------------------------
    def next_level(self) -> None:
        """A reward screen ended the current level (bot, replay): the next fight starts the
        next one. Levels only move forward [user review of run 20261029-072323: Rei the
        Ruthless's 7-cell arena, read by its width after the boss died, started a new level
        'bamboo-2' at the boss shrine]."""
        if self.level_i >= 0:
            self.advance = True

    def start_region(self, region: str) -> bool:
        """The map took the hero to a region of the route: its first fight is that region's first level, whatever reward
        screens were read. Hideyoshi's Keep shows none after him, the map comes at once (rec1 f1594), so 0.48 played the
        Castle's wave level as 'stage6-boss', with no wave to come and the boss rules on, in all three runs that reached it
        (20261102-232559, 20261103-020829, -035113: map_travel, then no level_start until the Shogun) [core-requests item
        17a]. Levels only move forward. Returns whether the tracker moved."""
        if region not in self.route:
            return False
        stage = self.route.index(region) + 1
        if stage < self.stage or (stage == self.stage and self.level_i >= 0):
            return False
        self.stage, self.level_i, self.advance = stage, -1, False
        return True

    def boss_index(self, stage: int | None = None) -> int:
        """The boss level's index in a stage's region (the route: the regions the run travels, set by
        the bot as it goes and by the replay from the recording's route)."""
        stage = self.stage if stage is None else stage
        return region_boss_index(self.route[stage - 1] if 0 < stage <= len(self.route) else None)

    def _level_pos(self, screen: str, cells: int) -> tuple[int, int]:
        """(stage, level index) of the fight on screen, without moving the tracker there."""
        stage, i = self.stage, self.level_i
        if self.advance:
            stage, i = (stage + 1, 0) if i >= self.boss_index(stage) else (stage, i + 1)
        elif i < 0:                          # the run's first fight: by the board
            i = self.boss_index(stage) if screen == "boss" else 1 if cells == 7 and stage == 1 else 0
        if screen == "boss":                 # the boss bar names this stage's boss
            i = self.boss_index(stage)
        elif stage == 1 and i == 0 and cells == 7:
            i = 1                            # Bamboo Grove II without its reward screen read
        elif stage == 1 and i == 1 and cells == 5:
            i = 2                            # Daisuke's arena shows a turn or two before the boss bar
        return stage, i

    def peek_level(self, screen: str, cells: int) -> str:
        """The name state() would give the fight on screen; the tracker stays where it is."""
        stage, i = self._level_pos(screen, cells)
        return level_name(stage, i, self.boss_index(stage))

    def _level(self, screen: str, cells: int) -> str:
        stage, i = self._level_pos(screen, cells)
        self.advance = False
        self.stage, self.level_i = stage, i
        return level_name(stage, i, self.boss_index(stage))

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
            # a shield does not outlast its fight: run 20261102-232559 carried a Reactive Shield from Spirit Gateway II
            # into III (t434) and from Kowa into Keep I (t796), and each first hit there cost HP planned as blocked (-2 at
            # t440 and t803, 4 of the run's 9 unpredicted HP) [core-requests item 17c]; Fortress gives one at every
            # fight's start [wiki]
            self.shield = bool(self.kit.fx.get("start_shield"))
            self.queue_track = []
            self.level = level
            self.facing_known = False
            self.prev = []
            self.prev_queues = []
            self.prev_hp = {}
            self.hero_x_after = None
            self.hist = []
            self.cycle = self._cycle_now = None
            self.loaded_tile = {}
            self.bombs, self.bombs_new = {}, {}
            self.rings, self.rings_new = [], []
            self.boss_phase, self.boss_frac = {}, {}
            self.chain_prev = {}

        h = p.hero
        facing = self._hero_facing(h, p.confidence.get("hero_facing", 0.0))

        kit = self.kit
        queue = self._queue(h, p.hand)
        charge = dict(self.charge)
        faces = match_faces(kit, p.hand, charge, skip=tuple(queue))
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
                read = f["charged"]
                if read >= cd > 0 and f.get("ready") is False:
                    # a dimmed face is recharging whatever its marks say [run 20261030-041652 t237: the 4/8
                    # Arrow's eighth mark fell outside the marks band, so it read 7 of 7 while dimmed]
                    read = cd - 1
                if read != charge.get(tile):
                    self.notes.append(f"{tile} charge read {read}/{cd}, tracked {charge.get(tile)}")
                    charge[tile] = read
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
        if self.shield and self.last_hp and 0 < hp < self.last_hp:
            # a shield negates the next attack [wiki], so HP lost means it is gone: one the predicted line did not
            # use stayed on (recording 2's Kami Brew at The Twins, f0758, was still tracked at Spirit Gateway I
            # f0875, where a Charger's 1 was planned as blocked)
            self.shield = False
            self.notes.append(f"HP {self.last_hp} -> {hp}: no shield")
        if self.shield and self.hero_x_after is not None and h["x"] != self.hero_x_after:
            # an enemy moved him (Dragon Punch, the Hook, the Smoke Bomb), so it hit him and the shield took the hit:
            # run 20261029-095311 t76, Rei's punch pushed the hero to the wall without HP lost, the tracked shield
            # stayed on, and t82's -2 was planned as -1
            self.shield = False
            self.notes.append(f"hero moved from {self.hero_x_after} to {h['x']} by an enemy: no shield")
        if "shield_on_damage" in kit.fx and self.last_hp and 0 < hp < self.last_hp:
            # Reactive Shield: HP lost gives the hero a shield [wiki: skills], which stays until an attack hits it. Run
            # 20261102-145513 t529: the shield from an earlier hit took the Grappler's Hook and its Sword cut the hero
            # (-2, a new shield after it); untracked, the Hook was planned to hit (-1), its shield to take the Sword
            # (the same at t740, the Shadow Charger's Charge and Sword) [core-requests item 11]
            self.shield = True
            self.notes.append(f"HP {self.last_hp} -> {hp}: Reactive Shield")
        if self.shield and h.get("shield_frac") is not None and h["shield_frac"] < HERO_SHIELD_GONE:
            # no sphere around the hero: no shield (perceive.hero_shield). Run 20261103-052436 at the Shogun: a Reactive
            # Shield gained from his Dragon Punch was used up by its collision (t820, t841), and the next Spear cut for
            # 2 planned as blocked (t823, t846); the frames show no sphere [core-requests items 17c, 19]
            self.shield = False
            self.notes.append("no shield around the hero on the screen")
        if hp:
            self.last_hp = hp
        # the red ticks under the hero count the turns his swap still needs (Quick Recovery shortens it)
        swap_charge = kit.swap_cooldown - min(kit.swap_cooldown, h.get("swap_red", 0))
        hero = Hero(h["x"], facing, hp=hp, queue=queue, charge=plan_charge, swap_charge=swap_charge,
                    shield=self.shield, kit=kit)

        enemies = []
        from .perception.perceive import pip_ties
        ties = pip_ties(tuple(self.route))
        self._canon = {k: min({k} | set(o)) for k, o in ties.items()}
        warn = list(getattr(p, "warn", []) or [])
        aimed_fire = any(e["queue"] and SPEC.get(e["queue"][0], ("",))[0] == "aim" and e.get("intent") == "fire"
                         for e in p.enemies)
        sure = set()          # cells whose unit's bubble showed its own tiles, read sure (the Quick rule below)
        for i, e in enumerate(p.enemies):
            kind = e["kind"] if e["kind"] in ENEMY_KINDS else "ashigaru"
            if kind != e["kind"]:
                self.notes.append(f"enemy at {e['x']} read as {e['kind']}; assumed ashigaru")
            if kind in ties and not e["queue"] and not e.get("spawning"):
                # a unit named by its tile keeps that name while its bubble is empty: the Yumi Sniper has the
                # Ashigaru's 3 HP pips and shows its Volley only while loaded [run 20261030-115221 t175-t176: the
                # Sniper that had fired at t174 read as an Ashigaru again]; the Archer and the Tachi share 2
                was = next((q[0] for q in self.prev if q[0] in ties[kind] and self._could_be(q, e["x"])), None)
                if was:
                    self.notes.append(f"enemy at {e['x']}: kept as {was} (read as {kind}, no tile in its bubble)")
                    kind = was
            if kind in PHASED:
                hp_e = self._phased_hp(kind, p.boss_frac)
            elif kind in BOSS_KINDS:
                hp_e = max(1, round((p.boss_frac or 1.0) * BOSS_HP[kind]))
            else:
                hp_e = e["hp"] or ENEMY_KINDS[kind][0]
            intent = e.get("intent") or "hold"
            queue = list(e["queue"])
            if kind in BOSS_KINDS:
                # a boss loads only his own tiles (Daisuke: Sword, Swirl, Charge, Summon); other
                # reads are icon misses, set to his first tile [run 20261027-182237 crashed on a
                # 'spear' read]. Hideyoshi's stack is always a part of one of his combos (CAT chain_fix)
                own = BOSS_TILES[kind]
                last_phase = kind not in PHASED or CAT.enemy(kind).phase(hp_e) == CAT.enemy(kind).phases
                if kind in PHASED and any(t not in own for t in queue) and self.chain_prev.get(kind, ([], None))[0]:
                    # the Shogun's yellow frames: flames over the bubble read as the red bell (rec2 f1935, his Spear):
                    # the stack he loaded is the one he fires
                    was = list(self.chain_prev[kind][0])
                    self.notes.append(f"{kind} queue read {queue}; kept {was}")
                    queue = was
                fixed = CAT.enemy(kind).chain_fix(queue) if kind in CHAINS and last_phase else queue
                fixed = [t if t in own else own[0] for t in fixed]
                if fixed != queue:
                    self.notes.append(f"{kind} queue read {queue}; set to {fixed}")
                queue = fixed
            elif queue and any(t not in CAT.enemy(kind).tiles for t in queue):
                # regular enemies only carry their own tiles (the Grappler and the Shinobi two each); an odd
                # read is an animation overlay
                self.notes.append(f"enemy at {e['x']} queue read {queue}; set to {ENEMY_KINDS[kind][1]}")
                queue = [ENEMY_KINDS[kind][1]]
            elif queue and min(e.get("queue_conf") or [1.0]) >= QUICK_READ_CONF:
                sure.add(e["x"])
            if e.get("flash") and not queue and kind not in BOSS_KINDS:
                # its bubble washed out as it fired (perceive.firing_flash): it fires what it held the turn before, else
                # its one tile (runs 20261102-065902 t748, 20261102-232559 t241: read as holding, -2 each)
                held = [j for j, q in enumerate(self.prev) if self._same(q[0], kind) and self._could_be(q, e["x"])]
                j = min(held, key=lambda j: abs(self.prev[j][1] - e["x"]), default=None)
                queue = list(self.prev_queues[j]) if j is not None and j < len(self.prev_queues) else []
                if not queue and len(CAT.enemy(kind).tiles) == 1:
                    queue = [CAT.enemy(kind).tiles[0]]
                if queue:
                    intent = "fire"
                    self.notes.append(f"enemy at {e['x']}: its bubble flashed as it fired; fires {queue}")
            seq = CAT.enemy(kind).behaviour.get("sequence") if kind not in BOSS_KINDS else None
            if seq and intent == "fire" and 0 < len(queue) < len(seq) and queue == seq[:len(queue)]:
                # the Grappler loads its Sword the phase after its Hook and fires both (behaviour.intents): a fire
                # read with the Hook alone missed the Sword's bubble (run 20261101-053431 t461: the Hook was planned
                # alone, the hero's Kami Brew took it, and the Sword cut him for 2 unpredicted)
                self.notes.append(f"enemy at {e['x']}: fires with {queue} read; its sequence is {seq}")
                queue = list(seq)
            aim = None
            if queue and SPEC.get(queue[0], ("",))[0] == "aim" and warn and (intent == "fire" or not aimed_fire):
                # the warning sign: an aimed Volley fires this turn at that cell, the hero's, whatever the
                # bubble's blink showed [rec: collect/20261029-160032 f0199, f0204, f0216, f0222, f0264 read red]
                aim = min(warn, key=lambda c: abs(c - h["x"]))
                if intent != "fire":
                    self.notes.append(f"enemy at {e['x']}: warning sign at {aim}; fires this turn")
                intent = "fire"
            facing_e = self._enemy_facing(kind, e, queue, intent, h["x"],
                                          p.confidence.get(f"enemy{e['x']}", 0.0))
            alt = e.get("alt") if e.get("alt") in ENEMY_KINDS and e.get("alt") != kind else None
            if alt:
                self.notes.append(f"enemy at {e['x']}: read as {kind}, may be {alt}")
            en = Enemy(kind, e["x"], facing_e, hp_e, queue, uid=i + 1, intent=intent, alt=alt,
                       elite=bool(e.get("elite")), aim=aim, shield=bool(e.get("shield")))
            if self.freeze_left > 0 and kind not in UNFREEZABLE_KINDS:
                en.frozen, en.intent = self.freeze_left, None     # (Hideyoshi is Unfreezable [wiki])
            en.cursed = self.cursed
            enemies.append(en)
        bosses = [en for en in enemies if en.kind in BOSS_KINDS]
        if len(bosses) > 1 and len(bosses) <= CAT.enemy(bosses[0].kind).pair and len({b.kind for b in bosses}) == 1:
            pass                              # The Twins: two units, one HP bar [wiki]
        elif len(bosses) > 1:
            # one boss only: keep the read nearest his last known cell, the rest are
            # summoned units whose pips were hidden [run 20261028-053637: Chargers]
            last = next((q[1] for q in self.prev if q[0] in BOSS_KINDS), None)
            keep = min(bosses, key=lambda en: abs(en.x - last)) if last is not None else max(bosses, key=lambda en: len(en.queue))
            # (The Statue summons through its Summoner: the Summoner's unit)
            unit = (CAT.enemy(keep.kind).summons or next((e.summons for e in CAT.enemies.values()
                                                          if e.behaviour.get("synced_with") == keep.kind), None)
                    or "charger")
            minder = next((k for k, e in CAT.enemies.items() if e.behaviour.get("summoner_of") == keep.kind), None)
            for en in bosses:
                if en is not keep:
                    # a unit that stood there last turn keeps its kind and HP: the Summoner on cell 0 read as
                    # The Statue while the Hook pulled the hero past it, and a summoned Tachi on cell 3 the
                    # turn after it came [rec: collect/20261028-190026 f0868-f0869, f0879]
                    was = next((q for q in self.prev if q[0] not in BOSS_KINDS and q[1] == en.x), None)
                    k = was[0] if was else minder if minder and en.x in (0, cells - 1) else unit
                    en.kind, en.hp = k, (self.prev_hp.get(en.x) if was else None) or ENEMY_KINDS[k][0]
                    en.queue = [t for t in en.queue if t in CAT.enemy(k).tiles][:1]
            self.notes.append(f"{len(bosses)} {keep.kind} reads; kept cell {keep.x}")
        self.fresh = [(en.x, en.kind, en.facing) for en in self._unmatched(enemies)]
        self.prev = [(en.kind, en.x, en.facing, en.intent) for en in enemies]
        self.prev_queues = [list(en.queue) for en in enemies]
        self.prev_hp = {en.x: en.hp for en in enemies}
        self.swapped_uid_x = None
        self.mirrored = False
        # what each enemy did last turn and how long since it fired: the learned model's
        # history features, matched the way shogunbot.mine matches logged turns
        # a unit whose pips tie with another kind's keeps its history when its read kind flips: the Shielder named by its
        # Ally Shield on its hold phase lost the '+' of the phase before, and its rule fired it a phase early (run
        # 20261101-053431 t0673, read as a Yari Master at t0672)
        hist = [(self._canon.get(q[0], q[0]),) + tuple(q[1:]) for q in self.hist]
        for j, i in behaviour.match(hist, [(self._canon.get(en.kind, en.kind), en.x) for en in enemies]).items():
            _, _, it, fage = self.hist[i]
            enemies[j].last, enemies[j].fage = behaviour.carry(it, fage)
        self._queue_conf = {e["x"]: list(e.get("queue_conf") or []) for e in p.enemies}
        self._orbs = {e["x"] for e in p.enemies if e.get("orb")}
        self._cycle(enemies)

        s = State(cells, hero, enemies, level=level,
                  wave=1, waves_total=0 if level.endswith("boss") else 99, bombs=self._bombs(p, cells),
                  rings=self._rings(cells))
        from .rules import _hits_hero
        for en in enemies:
            if (en.kind in behaviour.QUICK and en.queue and en.intent in ("hold", None) and en.x in sure
                    and _hits_hero(s, en, en.queue[0])):
                # Quick: loaded, it fires the phase after it loaded once the hero is the first unit ahead, whatever
                # one frame of its bubble shows (the Shinobi's Shuriken 3 of 3, red on the frames read) [rec:
                # collect/20261028-190026 f0694-f0695, f0711-f0713; collect/20261029-160032 f0892-f0895]. Only on a
                # sure read of its own tile: a ghost read set to its Shuriken (rec2 f1051, 'swap_toss' at 0.06) or
                # a later region's unit with its pips (rec2 f1294, f1331, f1854) fired no Shuriken
                self.notes.append(f"enemy at {en.x}: quick and loaded, the hero in its reach; fires this turn")
                en.intent = "fire"
        self._guard_facing(s)
        self._rule_telegraphs(s)
        self._chain_rules(s)
        return s

    def _phased_hp(self, kind: str, frac) -> int:
        """A boss with more than one bar (the Shogun: two of 40, pooled in the simulator): the bars still to come plus this
        one's read. A bar that gains more than half its width starts the next phase (no heal comes near that) [rec:
        collect/20261028-190026 f1794 0.072 -> f1797 1.000; collect/20261029-160032 f1957 0.097 -> f1960 1.000]."""
        k = CAT.enemy(kind)
        ph = self.boss_phase.get(kind, 1)
        last = self.boss_frac.get(kind)
        if frac is None:                    # no bar read on this frame (the fight's first frames): the last read
            frac = 1.0 if last is None else last
        if last is not None and frac - last > 0.5 and ph < k.phases:
            ph += 1
            self.loaded_tile[kind] = "phase2"     # his opening comes next (rules._boss_queue)
            self.notes.append(f"{kind}: his bar refilled ({last:.2f} -> {frac:.2f}); phase {ph}")
        self.boss_phase[kind], self.boss_frac[kind] = ph, frac
        return max(1, round(frac * k.hp)) + k.hp * (k.phases - ph)

    def _rings(self, cells: int) -> list:
        """The Shogun's Corrupted Barrage rings, [centre, next distance, phases to wait]: those his last combo placed
        (acted) and those tracked from before, already grown by the phase just played."""
        out = [list(r) for r in self.rings] + [list(r) for r in self.rings_new]
        self.rings, self.rings_new = [list(r) for r in out], []
        return out

    def _bombs(self, p, cells: int) -> list:
        """The bombs on the floor, [cell, enemy phases left]: those Hideyoshi's last combo placed (acted: 2 phases left,
        at the cell a red bomb shows near the predicted one), those tracked from before, and a red bomb read that neither
        accounts for, taken to explode at the end of the coming phase (its age unknown: the cautious case). A tracked
        bomb not seen is kept: it blinks red, black and white (rec1 f1536-f1539, rec2 f1719-f1725)."""
        seen = [c for c in (getattr(p, "bombs", None) or []) if 0 <= c < cells]
        out = dict(self.bombs)
        for c, n in self.bombs_new.items():
            near = [x for x in seen if abs(x - c) <= 1 and x not in out]
            at = min(near, key=lambda x: abs(x - c)) if near else c
            if at != c:
                self.notes.append(f"bomb read on {at}, predicted on {c}")
            out[at] = n
        for c in seen:
            if c not in out:
                self.notes.append(f"a bomb on {c} not placed by a tracked combo; taken to explode this turn")
                out[c] = 1
        self.bombs, self.bombs_new = out, {}
        return [[c, n] for c, n in sorted(out.items())]

    def _chain_rules(self, s: State) -> None:
        """Hideyoshi's telegraph and facing from his rules (behaviour.cunning_intent), and his last combo (he never uses
        the same twice in a row: Enemy.loaded). His bubble blinks yellow and red on a fire turn, so a red read of a whole
        combo says nothing; a fire read of part of one missed the rest of its bubbles. A turn marker on his plate means he
        faces away from the hero, and a '+' with nothing loaded that he faces him (he turns first): either sets the
        tracked facing, which his sprite does not show (no templates past Stage 1)."""
        for e in s.enemies:
            if not chained(e):
                continue
            kind = CAT.enemy(e.kind)
            if e.last == "fire" and e.queue:
                # the phase after a combo he has nothing loaded (the next one starts with a '+'): a stack read now is the
                # fired one's bubbles playing out (rec2 f1703, f1719; rec1 f1544, f1580)
                self.notes.append(f"{e.kind} at {e.x}: fired last turn; {e.queue} read is its bubbles playing out")
                e.queue = []
            prevq, previt = self.chain_prev.get(e.kind, (None, None))
            nxt = kind.chain_next(prevq) if prevq else None
            if previt == "queue" and nxt and len(e.queue) <= len(prevq) and e.queue == prevq[:len(e.queue)]:
                # he never drops a combo: after a '+' over part of one he holds its next tile too, so a stack read short
                # of that missed bubbles (rec1 f1555: on cell 0 at the screen's left edge his bubble read nothing)
                self.notes.append(f"{e.kind} at {e.x}: {e.queue} read after {prevq} and a '+'; holds {prevq + [nxt]}")
                e.queue = prevq + [nxt]
            if e.queue:
                self.loaded_tile[e.kind] = e.queue[0]
            e.loaded = self.loaded_tile.get(e.kind)
            toward = 1 if s.hero.x > e.x else -1
            if not e.queue and e.intent in ("turn", "queue"):
                f = -toward if e.intent == "turn" else toward
                if f != e.facing:
                    self.notes.append(f"{e.kind} at {e.x}: read {e.intent} with nothing loaded; faces {f}")
                    e.facing = f
                    self.prev = [(k, x, f if k == e.kind else ff, it) for k, x, ff, it in self.prev]
            if e.intent == "fire" and e.queue and not kind.chain_full(e.queue):
                nxt, q = kind.chain_next(e.queue), list(e.queue)
                while nxt:
                    q.append(nxt)
                    nxt = kind.chain_next(q)
                if kind.chain_full(q):
                    self.notes.append(f"{e.kind} at {e.x}: fires with {e.queue} read; the combo is {q}")
                    e.queue = q
                    continue
            if e.intent in ("hold", "queue", "fire", None):
                rule = behaviour.cunning_intent(s, e, behaviour.features(s, e, e.last, e.fage))
                if rule != e.intent and not (rule == "turn" and e.intent == "queue"):
                    self.notes.append(f"{e.kind} at {e.x}: read {e.intent} with {e.queue or 'nothing'} loaded; "
                                      f"his rule says {rule}")
                    e.intent = rule

    def _guard_facing(self, s: State) -> None:
        """The Shielder's facing from its rule (behaviour.guard_intent), where its sprite says nothing (no templates past
        Stage 1): it turns to a shieldless ally behind it, so a turn telegraph with shieldless allies on one side only
        means it faces away from them now; and it loads Ally Shield for the first shieldless ally ahead, so the phase
        its Ally Shield first shows with shieldless allies on one side only, it faces them. Run 20261101-053431
        t0671-t0675: its turn was taken as turning to the hero, so it was planned facing him, and the Yari Master on
        cell 4 it shielded at t0674 stood 'behind' it."""
        changed = False
        for e in s.enemies:
            if not CAT.enemy(e.kind).behaviour.get("guard"):
                continue
            left, right = behaviour.shieldless_ally(s, e, -1), behaviour.shieldless_ally(s, e, 1)
            if (left is None) == (right is None):
                continue
            side = -1 if left is not None else 1
            if e.intent == "turn":
                f, why = -side, "turns to the shieldless ally"
            elif e.queue and e.queue[0] == "ally_shield" and e.last == "queue":
                f, why = side, "loaded Ally Shield for the shieldless ally"
            else:
                continue
            if f != e.facing:
                self.notes.append(f"{e.kind} at {e.x}: {why}; faces {f}")
                e.facing, changed = f, True
        if changed:
            facing = {en.x: en.facing for en in s.enemies}
            self.prev = [(k, x, facing.get(x, f) if CAT.enemy(k).behaviour.get("guard") else f, it)
                         for k, x, f, it in self.prev]

    def _rule_telegraphs(self, s: State) -> None:
        """Kowa's and the Shielder's telegraphs from their rules (behaviour.coward_intent, guard_intent) where the
        screen does not show them, and the tile Kowa loaded last (he alternates Shield and Smoke Bomb). The phase after
        a use they rest (Kowa 12 of 12, the Shielder in every use seen): the empty bubble left over and the cast flash
        read as '+' then (rec1 f1284, f1311; f1306, f1318), and a Shielder's idle bubble on 38 frames (rec1 f0960-f0963,
        f1120-f1121; rec2 f1186-f1189). A tile loaded in an earlier phase is used on the rule: a use turn's bubble
        blinks yellow and red, so a red read says nothing (rec1 f1317 and f1334 read red on their only frame, f1295
        and f1342 on their last), and the Shielder's fire orb shows it (perceive.orb_pixels). A marker on the plate
        and a yellow bubble are kept as read."""
        for e in s.enemies:
            b = CAT.enemy(e.kind).behaviour
            coward, guard = e.kind in BOSS_KINDS and b.get("coward"), b.get("guard")
            if not (coward or guard):
                continue
            if coward:
                if e.queue:
                    self.loaded_tile[e.kind] = e.queue[-1]
                e.loaded = self.loaded_tile.get(e.kind)
            if (guard and e.queue and e.queue[0] == "ally_shield" and e.fage != 0 and e.intent in ("fire", "hold", "queue", None)
                    and behaviour.shieldless_ally(s, e, e.facing) is None):
                # no shieldless ally ahead: it holds the tile, its bubble blinking and its orb lit (guard_intent); a
                # marker on the plate is kept as read (run 20261101-021522 t0839: a loaded Shielder stepped right)
                if e.intent != "hold":
                    self.notes.append(f"{e.kind} at {e.x}: read {e.intent} with Ally Shield loaded and no shieldless "
                                      f"ally ahead; holds")
                e.intent = "hold"
                continue
            if e.queue and e.x in self._orbs and e.intent != "fire":
                self.notes.append(f"{e.kind} at {e.x}: the fire orb; fires this turn")
                e.intent = "fire"
                continue
            if e.fage == 0:
                if e.intent != "hold":
                    self.notes.append(f"{e.kind} at {e.x}: read {e.intent} the phase after a use; rests")
                e.intent = "hold"
                continue
            if e.queue and e.intent in ("hold", "queue", None):
                f = behaviour.features(s, e, e.last, e.fage)
                rule = behaviour.coward_intent(s, e, f) if coward else behaviour.guard_intent(s, e, f)
                if rule in ("hold", "fire") and rule != e.intent:
                    self.notes.append(f"{e.kind} at {e.x}: read {e.intent} with {e.queue[0]} loaded; its rule says {rule}")
                    e.intent = rule

    def _cycle(self, enemies: list) -> None:
        """The Statue's telegraph from its 9-turn cycle (behaviour.cycle_intent), and its stack kept from turn to
        turn: its bubbles bob in 4-px steps and shake on the fire turn, so a single read can name a tile wrongly or
        miss the yellow [rec: collect/20261028-190026 f0857-f0921, collect/20261029-160032 f1071-f1120]. What it did
        in the phase just played comes from its tile count against the last turn's (one more: it loaded; none after
        three: it fired; the same: it rested or held). The Summoner follows it."""
        st = next((e for e in enemies if CAT.enemy(e.kind).behaviour.get("cycle")), None)
        if st is None:
            self._cycle_now = None
            return
        conf = list(self._queue_conf.get(st.x, [])) + [0.0] * len(st.queue)
        read, prev = [(t, conf[i]) for i, t in enumerate(st.queue)], self.cycle
        q = len(read)
        if prev is None:
            # its first turn rests [wiki: 'needing an extra turn to add a Tile']; a '+' on the first frame read
            # means that turn was missed and it loads now
            last, holds, slots = ("hold" if q == 0 and st.intent == "queue" else None), 0, read
        else:
            pq = prev["q"]
            # each slot keeps its most confident read: a tile just loaded glows and the stack shakes on the fire
            # turn (rec1 f0863: Hook, Spear read Arrow 0.08, Sword 0.24; f0913: Spear read Sword 0.01), settled
            # frames read 1.0
            slots = [max(a, b, key=lambda tc: tc[1]) for a, b in zip(prev["slots"], read)] + read[len(prev["slots"]):]
            if q == pq + 1:
                last = "queue"
            elif q == 0 and pq >= 3:
                last = "fire"
            elif q == pq:
                last = "hold"
            else:
                last, slots = prev["intent"], read
                self.notes.append(f"the_statue: {q} tiles read after {pq}")
            slots = slots[:q]
            holds = prev["holds"] + 1 if (q >= 3 and pq >= 3) else 0
        stack = [t for t, _ in slots]
        it = behaviour.cycle_intent(q, last, holds)
        if st.intent != it:
            self.notes.append(f"the_statue: read {st.intent}, its cycle says {it}")
        if stack != list(st.queue):
            self.notes.append(f"the_statue: stack read {st.queue}, kept {stack}")
        st.queue, st.last, st.intent = stack, last, it
        st.wind = holds if q >= 3 else 1
        self._cycle_now = {"q": q, "slots": slots, "intent": it, "holds": holds}
        for e in enemies:
            if CAT.enemy(e.kind).behaviour.get("synced_with") == st.kind:
                # the Summoner loads its bell with the Statue's first load and summons with its second
                want = ("queue" if it == "queue" and q == 0 and not e.queue else
                        "fire" if it == "queue" and q == 1 else "hold")
                if want == "fire" and not e.queue:
                    e.queue = ["summon"]
                if e.intent != want:
                    self.notes.append(f"summoner: read {e.intent}, the Statue's cycle says {want}")
                e.intent = want

    # ------------------------------------------------------------------ queue
    def _queue(self, h: dict, hand: list[dict]) -> list[str]:
        """The hero's queue, bottom first, as kit labels.

        Queued tiles leave the hand [run 20261028-061420 f0025], so the kit
        tiles missing from the hand are the queued ones; the number of bubbles
        over the hero confirms the count. Their order comes from the bubble
        icons where those name kit tiles, else from what the bot queued."""
        kit = self.kit
        bubbles = [BUBBLE_ALIAS.get(b, b) for b in h.get("queue", [])]
        conf = list(h.get("queue_conf") or [])[:len(bubbles)]
        sure = [c >= BUBBLE_SURE for c in conf] + [True] * (len(bubbles) - len(conf))
        in_hand = match_faces(kit, hand, self.charge, guess=False)   # queue count: as before 0.43
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
                # sure bubbles matched first; ties (unsure or misread bubbles, copies of a tile) go to the order the
                # bot queued in [review of 0.15], unsure reads last [0.47: run 20261101-105146 t386, a Charge read
                # as Spear at 0.24 put the Charge before the Spear]
                return (sum(kit[lb].tile == name for lb, name, ok in zip(order, bubbles, sure) if ok),
                        sum(a == b for a, b in zip(order, tracked)),
                        sum(kit[lb].tile == name for lb, name in zip(order, bubbles)))

            order = list(max(permutations(tracked), key=fit)) if tracked else []
            if fit(order)[2] < len(bubbles):
                self.notes.append(f"queue bubbles read {bubbles}; queue taken as {order}")
            return order
        if track and len(track) == len(bubbles) and all(
                kit[lb].tile == name or not ok for lb, name, ok in zip(track, bubbles, sure)):
            # counts disagree (an unread hand face, an animation) but the bubbles fit the queue the bot built: that
            # queue [0.47: run 20261101-105146 t638-t639 and t667-t669, a hand face unread and the Charge read as
            # Spear at 0.03-0.05, so the plan fired a Spear the game did not have]
            if hand:
                self.notes.append(f"{len(bubbles)} queue bubbles but {len(missing)} kit tiles off the hand; "
                                  f"queue as the bot built it: {track}")
            return list(track)
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
        # a weaker contrary read (perceive's grey-patch fallback, 0.5) that holds on HERO_FACING_WEAK_N
        # reads in a row also overrides: in run 20261031-165514 (t568-t605) the tracked 'right' was never
        # corrected and the hero attacked the wall five times [0.42]
        if seen and seen != self.facing and conf >= 0.5:
            self.facing_weak += 1
            if self.facing_weak >= HERO_FACING_WEAK_N:
                self.notes.append(f"hero facing re-read as {seen} (weak reads)")
                self.facing, self.facing_weak = seen, 0
        else:
            self.facing_weak = 0
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
        # The Shogun's first bar has no combos: 'he runs up, attacks with the tile, backs off' [wiki], loaded facing the hero
        # like the other bosses (behaviour.shogun_intent turns him first when he faces away). Run 20261103-020829 t895-t897:
        # tracked facing away since his Mirror (t887), he loaded the Sword, stepped right twice toward the hero and cut him
        # (-2 planned as a miss) [core-requests, found while doing item 11]
        phase1 = kind in PHASED and self.boss_phase.get(kind, 1) < CAT.enemy(kind).phases
        # the Moonlit Port units (0.31) load like the Ashigaru: Sword and Arrow strike ahead; the Twins' loads
        # all point at the hero (Tetsubo, Shadow Dash, the Dash and Swap Toss with the hero next to them)
        directional = kind in ("ashigaru", "charger", "guardian", "ashigaru_archer") or (
            kind in BOSS_KINDS and (kind not in CHAINS or phase1) and queue and queue[0] in (
                "sword", "charge", "dragon_punch", "spear", "tetsubo", "shadow_dash", "dash", "swap_toss"))
        # (Hideyoshi loads facing the hero, but the hero may pass him while he loads, and his Mirror turns him: his
        # facing persists from his turns and his combos' moves, Tracker._chain_rules and acted)
        if (not directional and kind not in BOSS_KINDS and queue and not CAT.enemy(kind).behaviour.get("guard")
                and (CAT.tile(queue[0]).directional or CAT.tile(queue[0]).attack.get("dir"))):
            # every regular unit loading a tile that strikes ahead (the Grappler's Hook, the Yari Master's Spear, the
            # Shinobi's Shuriken and Smoke Bomb) faces the hero: run 20261101-004025 t336 and t341 (0.45), a Grappler
            # tracked facing away hooked the hero and cut him (-3 at t341), run 20261031-233208 t285 (-3), and the
            # Yari Masters of runs -233208 t519 and 20261101-004025 t497, t501 (-2 each) [core-requests items 3, 4]
            directional = True
        if directional and (queue or intent == "fire") and self.swapped_uid_x != e["x"] and not self.mirrored:
            # (after Rain of Mirrors a loaded unit faces away and a yellow one fires away, as after a swap: run
            # 20261101-021522 f0209-f0210, the Yari Master's Spear went right after the card, not into the hero)
            return toward
        if intent == "turn" and not CAT.enemy(kind).behaviour.get("guard"):
            # (the Shielder turns to an ally, not to the hero: _guard_facing)
            return -toward
        prev = [q for q in self.prev if self._same(q[0], kind) and self._could_be(q, e["x"])]
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
            i = next((i for i in free if self._same(self.prev[i][0], en.kind) and self.prev[i][1] == en.x), None)
            if i is None:
                left.append(en)
            else:
                free.remove(i)
        out = []
        for en in left:
            cand = [i for i in free if self._same(self.prev[i][0], en.kind) and (
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

    def _same(self, a: str, b: str) -> bool:
        """Could a unit read as kind a be the one now read as b? The same kind, or kinds whose HP pips tie on this
        route (the Ashigaru Archer and the Tachi; the Yari Master, the Shinobi and the Shielder), which only the tile
        in the bubble tells apart."""
        return a == b or (a in self._canon and self._canon.get(a) == self._canon.get(b))

    def freeze(self, turns: int) -> None:
        self.freeze_left = turns

    def item_used(self, item: str) -> None:
        """An item used up (gone from its slot): its effect on the tracked state, by the catalogue's effect type.
        The bot calls it after a use; the replay too, for the items a recording used (0.36: recording 1's Mass Ice
        at f0726 froze the Shinobi on cell 0 for three turns, which 0.35's replay did not know)."""
        cons = CAT.consumable_by_screen(item)
        kind = cons.effect.get("type") if cons else None
        if kind == "freeze_all":
            self.freeze(ICE_TURNS)        # Mass Ice: all enemies frozen for 3 turns [catalogue]
        elif kind == "shield":
            self.shield = True            # Kami Brew: negates the next attack [wiki]
        elif kind == "curse_all":
            self.cursed = True            # Mass Curse: next hit on each enemy is doubled [wiki]
        elif kind == "turn_all":
            self.mirror()                 # Rain of Mirrors: every enemy turned around [wiki]
        elif kind == "recharge_all":
            # Cool Up: every tile full again (the hand shows it too; a tile left dimmed by a slow frame is planned
            # as recharging otherwise) [catalogue]
            self.charge = {t.label: t.cooldown for t in self.kit}
        # Lucky Die rerolls every enemy's intent [wiki] [user 2026-10-28]: nothing to model, the next frame shows
        # the new telegraphs; brews change HP and charge, which the next frame shows too

    def mirror(self) -> None:
        """Rain of Mirrors turned every enemy: flip the remembered facings, or the
        facing-persistence rule would undo the flip."""
        self.prev = [(k, x, f if k in HEAVY_KINDS else -f, None) for k, x, f, _ in self.prev]
        self.mirrored = True

    def recharging(self, label: str) -> None:
        """The game refused to queue this tile (it stayed in the hand): it is recharging, whatever was read
        or tracked [user review of run 20261030-041652, fix 1]."""
        kt = self.kit.by_label.get(label)
        if kt is not None and kt.cooldown > 0:
            self.charge[label] = min(self.charge.get(label, kt.cooldown), kt.cooldown - 1)

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
        self.hero_x_after = s2.hero.x
        self.charge = dict(s2.hero.charge)
        self.queue_track = list(s2.hero.queue)
        # bombs: one phase closer (a count reaching 0 explodes in the coming phase); Hideyoshi's combo, fired in the
        # coming phase, moves him (Origin, Mirror, the dashes: his facing after it is what persists) and may move the
        # hero (Origin and Mirror change places with a unit: no hit, so a shield stays) and place a bomb
        self.bombs = {c: n - 1 for c, n in self.bombs.items() if n - 1 > 0}
        self.bombs_new = {}
        # rings: the coming enemy phase strikes each at its next distance and grows it (rules._ring_tick)
        rs = State(s.cells, s.hero.clone(), [], rings=[list(r) for r in self.rings])
        from .rules import _ring_tick
        _ring_tick(rs, Events())
        self.rings, self.rings_new = rs.rings, []
        fired = {}
        self.chain_prev = {e.kind: (list(e.queue), e.intent) for e in s.alive() if e.kind in CHAINS}
        for e in s2.alive():
            if e.kind in CHAINS and e.intent == "fire" and e.queue:
                c = s2.clone()
                ce = next(x for x in c.enemies if x.uid == e.uid)
                c.bombs = []
                _enemy_fire(c, ce, Events())
                fired[e.kind] = (e.x, ce.x, ce.facing)
                self.bombs_new.update({x: n - 1 for x, n in c.bombs})
                for r in c.rings:
                    rr = State(c.cells, c.hero.clone(), [], rings=[list(r)])
                    from .rules import _ring_tick
                    _ring_tick(rr, Events())
                    self.rings_new.extend(rr.rings)
                if c.hero.x != s2.hero.x:
                    self.hero_x_after = c.hero.x
        # where each enemy stands once the hero has acted (a swap moves one), and its telegraph
        self.hist = [(e.kind, s.hero.x if (e.x == s2.hero.x and s2.hero.x != s.hero.x) else e.x, e.intent, e.fage)
                     for e in s.alive()]
        self.cycle = self._cycle_now
        if action == "turn":
            self.facing = -s.hero.facing
        if s2.hero.x != s.hero.x and s.enemy_at(s2.hero.x) is not None:
            # swap: the enemy now stands where the hero was and keeps its facing. The others keep their
            # telegraphs: 0.18-0.32 cleared every unit's, so a unit telegraphing a turn kept its old facing
            # once the hero swapped past another (run 20261029-184142 t31: the Ashigaru on cell 6 turned, the
            # sprite read it, the tracker kept it facing right; also runs 102441 t16, 154200 t159)
            self.swapped_uid_x = s.hero.x
            self.prev = [(k, s.hero.x, f, None) if x == s2.hero.x else (k, x, f, i) for k, x, f, i in self.prev]
        if fired:
            # his combo's moves: where he stands and which way he faces after it (Mirror turns him: 4 of 4, rec1 f1544,
            # f1568, f1580; rec2 f1712); the next frame's facing persists from there
            self.prev = [(k, fired[k][1], fired[k][2], None) if k in fired else (k, x, f, i) for k, x, f, i in self.prev]
