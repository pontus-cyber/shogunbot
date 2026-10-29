"""0.28: Iwao the Impaler, the Whispering Caves boss, first met by the bot in run 20261029-124921
(0.27.1 played him as Daisuke, the stand-in: his icons were rewritten as Swords and both of his hits,
-2 at t172 and t192, were unpredicted [user review]). His fight in the user's recording
(collect/20261028-190026 f0285-f0318) is the second sample. Frame tests skip when the frames are
missing."""

import numpy as np
from test_bot_flow import FakeGW, frame, make_bot

from shogunbot import behaviour
from shogunbot.bridge import Tracker
from shogunbot.catalogue import CAT
from shogunbot.model import Enemy, Hero, Kit, State
from shogunbot.perception.perceive import perceive, read_stack
from shogunbot.planner import plan
from shogunbot.rules import _boss_queue, step

R = "runs/20261029-124921/frames/"
C = "collect/20261028-190026/"
KIT = ("swirl", "arrow", "back_charge")
ROUTE = ("bamboo_grove", "whispering_caves")


def _read(rel, kit=KIT, regions=ROUTE):
    rgb = frame(rel)
    p = perceive(rgb, list(kit), regions=regions, day=1)
    return rgb, p


def _state(rel):
    _, p = _read(rel)
    return Tracker(Kit.of(*KIT)).state(p)


def _board(iwao, hero_x, hp=10, facing=1, cells=7):
    hero = Hero(hero_x, facing, hp=hp, kit=Kit.of("swirl", "arrow"))
    return State(cells, hero, [iwao], level="stage2-boss", wave=0, waves_total=0)


def _iwao(x, queue=(), intent=None, facing=-1, hp=25, wind=1, last=None, fage=None):
    e = Enemy("iwao", x, facing, hp, list(queue), uid=1, intent=intent, wind=wind)
    e.last, e.fage = last, fage
    return e


# ---- perception and the tracker ---------------------------------------------------------------
def test_the_seven_cell_caves_arena_is_iwao():
    _, p = _read(R + "t0172_before.png")
    assert (len(p.cells), [(e["kind"], e["x"]) for e in p.enemies]) == (7, [("iwao", 4)])
    s = Tracker(Kit.of(*KIT)).state(p)
    assert s.enemies[0].hp == max(1, round(p.boss_frac * 25))      # HP 25 [wiki]


def test_his_icons_read_as_his_tiles():
    # Spear is the Yari Master's drawing; Earth Impale is brown spikes; Swirl sits on an Earth Impale
    reads = {}
    for rel in (R + "t0165_before.png", R + "t0172_before.png", R + "t0175_before.png", R + "t0186_before.png",
                C + "f0297_CLICK.png"):
        rgb, p = _read(rel)
        boss = next(e for e in p.enemies if e["kind"] == "iwao")
        reads[rel.split("/")[-1][:5]] = [s["tile"] for s in read_stack(rgb, boss["cx"])]
    assert reads == {"t0165": ["earth_impale"], "t0172": ["spear"], "t0175": ["summon"],
                     "t0186": ["earth_impale", "swirl"], "f0297": ["earth_impale", "swirl"]}


def test_a_new_iwao_faces_the_hero():
    # his sprite faces the viewer: at t159 it read 'facing right' with the hero on his left
    _, p = _read(R + "t0159_before.png")
    s = Tracker(Kit.of(*KIT)).state(p)
    assert (s.hero.x, s.enemies[0].x, s.enemies[0].facing) == (2, 4, -1)


# ---- the hits of run 20261029-124921 ----------------------------------------------------------
def test_his_spear_hits_two_cells_ahead_and_the_planner_steps_out():
    # t172: Spear telegraphed, the hero two cells away; 0.27.1 read a Sword (adjacent only), stayed and
    # lost 2. Stepping left leaves the Spear's reach
    s = _state(R + "t0172_before.png")
    assert [(e.kind, e.x, e.queue, e.intent) for e in s.enemies] == [("iwao", 4, ["spear"], "fire")]
    assert step(s, "wait")[0][1].hero.hp == s.hero.hp - 2
    d = plan(s, 3)
    assert next(c for c in d.candidates if c["action"] == d.action)["exp_damage"] == 0
    assert step(s, d.action)[0][1].hero.hp == s.hero.hp


def test_at_t192_every_move_takes_damage_so_the_kami_brew_is_due(tmp_path):
    # the hero stood next to him facing away from him (no swap this turn) with his Spear about to fire
    s = _state(R + "t0192_before.png")
    d = plan(s, 3)
    assert min(c["exp_damage"] for c in d.candidates) >= 1.0
    b = make_bot(tmp_path, FakeGW([(np.zeros((1080, 1920, 3), np.uint8), [])]))
    assert b._damage_item(s, {"blue_potion", "green_potion"}) == "blue_potion"


# ---- his tiles and pattern [wiki] [bot: run 20261029-124921 t159-t198] ------------------------
def test_earth_impale_hits_two_cells_ahead_and_behind():
    for hero_x, hit in ((1, True), (5, True), (2, False), (4, False)):
        s = _board(_iwao(3, ["earth_impale"], "fire"), hero_x)
        assert (step(s, "wait")[0][1].hero.hp == 8) is hit, hero_x


def test_swirl_is_added_to_earth_impale_only_with_the_hero_next_to_him():
    def drawn(hero_x):
        e = _iwao(3, ["earth_impale"], last="hold", fage=2)
        return behaviour.intents(_board(e, hero_x), e)
    assert drawn(2) == [(1.0, "queue")]          # adjacent: he adds a Swirl (3 of 3)
    assert drawn(1) == [(1.0, "fire")]           # two cells: Earth Impale fires alone (R2)
    assert drawn(0) == [(1.0, "move-left")]      # out of reach: he steps toward the hero
    e = _iwao(3, ["earth_impale"], "queue")
    assert _boss_queue(_board(e, 2), e, None) == [(1.0, ("queue", "swirl"))]


def test_after_firing_he_steps_away_then_loads():
    e = _iwao(3, fage=0, last="fire")
    assert behaviour.intents(_board(e, 1), e) == [(1.0, "move-right")]
    e = _iwao(6, fage=0, last="fire")
    assert behaviour.intents(_board(e, 4), e) == [(1.0, "hold")]          # the wall
    e = _iwao(4, fage=1, last="move")
    assert behaviour.intents(_board(e, 1), e) == [(1.0, "queue")]


def test_a_finished_combo_and_a_bell_fire_a_phase_later():
    e = _iwao(5, ["earth_impale", "swirl"], last="hold", fage=3)
    assert behaviour.intents(_board(e, 4), e) == [(1.0, "fire")]
    e = _iwao(5, ["summon"], last="hold", fage=3)
    assert behaviour.intents(_board(e, 2), e) == [(1.0, "fire")]


def test_his_summon_brings_a_yari_master():
    s = _board(_iwao(5, ["summon"], "fire"), 2)
    s2 = step(s, "wait")[0][1]
    assert [e.kind for e in s2.enemies if e.kind != "iwao"] == ["yari_master"]
    assert CAT.enemy("iwao").combo_next(["earth_impale"], 2) is None


def test_his_loads_follow_the_distance():
    e = _iwao(4, intent="queue")
    opts = dict((a[1], p) for p, a in _boss_queue(_board(e, 2), e, None))
    assert set(opts) == {"spear", "summon", "earth_impale"} and opts["spear"] > opts["earth_impale"]


# ---- bubbles a few px off their unit [user review of run 20261029-124921, point 5] ------------
def test_a_bubble_off_its_units_centre_is_read():
    # t71: Rei had just stepped next to the hero; her Dragon Punch bubble sat 6 px right of her cell
    # and 0.27.1 read no bubble, so her punch (1 damage and a push to the wall) was unpredicted.
    # t95 (Whispering Caves I): the Yari Master's Spear next to the hero, the same; it hit the hero
    # and the Twin Tachi behind him (-2, the Tachi died)
    _, p = _read(R + "t0071_before.png", regions=("bamboo_grove",))      # Stage 1's boss
    assert [(e["kind"], e["x"], e["queue"]) for e in p.enemies] == [("rei", 2, ["dragon_punch"])]
    _, p = _read(R + "t0095_before.png")
    assert [(e["kind"], e["x"], e["queue"]) for e in p.enemies] == [("tachi", 0, ["swirl"]), ("yari_master", 3, ["spear"])]
