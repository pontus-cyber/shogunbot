"""0.20: spend every item on the run's last level, confirm an HP drop on a second frame,
and read items that the level-end dimming darkens. Frame-based tests skip when the
recorded frames are missing."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_bot_flow import FakeGW, frame, make_bot

from shogunbot.bridge import Tracker
from shogunbot.model import Enemy, Hero, Kit, State

BLANK = np.zeros((1080, 1920, 3), np.uint8)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def _bot(tmp_path, img=BLANK):
    return make_bot(tmp_path, FakeGW([(img, [])]))


def _boss(hero, *enemies):
    return State(7, hero, list(enemies), level="boss", wave=1, waves_total=1)


WAIT = SimpleNamespace(action="wait")
BOXED = dict(swap_charge=0, charge={"swirl": 0, "arrow": 0})    # no swap, no tile ready: no escape


# ---- 1. use it or lose it on the last level [user 2026-10-28] ------------------------
def test_mass_ice_when_an_enemy_is_armed_or_adjacent(tmp_path):
    b, tr = _bot(tmp_path), Tracker()
    near = _boss(Hero(2, 1), Enemy("ashigaru", 3, -1, 3, uid=1))
    armed = _boss(Hero(0, 1), Enemy("ashigaru", 5, -1, 3, ["sword"], uid=1))
    calm = _boss(Hero(0, 1), Enemy("ashigaru", 5, -1, 3, uid=1))
    assert b._spend_item(near, WAIT, tr, {"ice_card"})[0] == "ice_card"
    assert b._spend_item(armed, WAIT, tr, {"ice_card"})[0] == "ice_card"
    assert b._spend_item(calm, WAIT, tr, {"ice_card"}) is None
    frozen = _boss(Hero(2, 1), Enemy("ashigaru", 3, -1, 3, ["sword"], uid=1, frozen=2))
    assert b._spend_item(frozen, WAIT, tr, {"ice_card"}) is None        # already frozen


def test_kami_brew_is_not_spent_on_an_armed_enemy_alone(tmp_path):
    # [user review of run 20261029-124921] 0.20-0.27.1 spent it on the last level as soon as an
    # enemy was armed (t161, Iwao armed); the unavoidable hits came at t172 and t192
    b, tr = _bot(tmp_path), Tracker()
    armed = _boss(Hero(0, 1), Enemy("ashigaru", 5, -1, 3, ["sword"], uid=1))
    assert b._spend_item(armed, WAIT, tr, {"blue_potion"}) is None


def test_kami_brew_when_every_move_takes_damage(tmp_path):
    # the chain for a turn where every move takes damage: Mass Ice first, Kami Brew last, and no
    # second shield while one is up
    b = _bot(tmp_path)
    s = _boss(Hero(1, 1, hp=7, **BOXED), Enemy("ashigaru", 0, 1, 3, ["sword"], uid=1, intent="fire", wind=2),
              Enemy("ashigaru", 2, -1, 3, ["sword"], uid=2, intent="fire", wind=2))
    assert b._damage_item(s, {"blue_potion"}) == "blue_potion"
    assert b._damage_item(s, {"ice_card", "blue_potion"}) == "ice_card"
    s.hero.shield = True
    assert b._damage_item(s, {"blue_potion"}) is None


def test_lucky_die_when_an_enemy_telegraphs_fire(tmp_path):
    b, tr = _bot(tmp_path), Tracker()
    fire = _boss(Hero(0, 1), Enemy("ashigaru", 5, -1, 3, ["sword"], uid=1, intent="fire"))
    queue = _boss(Hero(0, 1), Enemy("ashigaru", 5, -1, 3, uid=1, intent="queue"))
    assert b._spend_item(fire, WAIT, tr, {"dice_card"})[0] == "dice_card"
    assert b._spend_item(queue, WAIT, tr, {"dice_card"}) is None


def test_cool_up_when_the_best_tile_is_recharging(tmp_path):
    # Swirl is the default kit's strongest tile (catalogue ready value 8 against Arrow's 6)
    b, tr = _bot(tmp_path), Tracker()
    enemy = Enemy("ashigaru", 5, -1, 3, uid=1)
    empty = _boss(Hero(0, 1, charge={"swirl": 1, "arrow": 5}), enemy)
    full = _boss(Hero(0, 1, charge={"swirl": 3, "arrow": 0}), enemy)
    assert b._spend_item(empty, WAIT, tr, {"white_potion"})[0] == "white_potion"
    assert b._spend_item(full, WAIT, tr, {"white_potion"}) is None


def test_mass_curse_on_the_first_attack_that_hits_the_boss(tmp_path):
    b, tr = _bot(tmp_path), Tracker()
    boss = Enemy("daisuke", 4, -1, 15, uid=1)
    loaded = _boss(Hero(1, 1, queue=["arrow"], charge={"swirl": 3, "arrow": 0}), boss)
    attack = SimpleNamespace(action="attack")
    assert b._spend_item(loaded, attack, tr, {"curse_card"})[0] == "curse_card"
    assert b._spend_item(loaded, WAIT, tr, {"curse_card"}) is None        # not attacking
    away = _boss(Hero(1, -1, queue=["arrow"], charge={"swirl": 3, "arrow": 0}), boss)
    assert b._spend_item(away, attack, tr, {"curse_card"}) is None        # facing away: no hit


def test_edamame_is_neither_spent_on_the_last_level_nor_drunk_before_planning(tmp_path):
    # [user 2026-10-29] removed from 'use it or lose it' (items carry over past Stage 1); 0.20-0.23
    # drank it below full HP on the last level and at HP 7 or lower elsewhere
    b, tr = _bot(tmp_path), Tracker()
    p = SimpleNamespace(items=[{"item": "green_potion", "fx": 0.06, "fy": 0.88}])
    hurt = _boss(Hero(0, 1, hp=3), Enemy("ashigaru", 5, -1, 3, uid=1))
    assert b._pre_plan_item(p, hurt, {}) is None
    assert b._spend_item(hurt, WAIT, tr, {"green_potion"}) is None


# ---- Edamame Brew [user 2026-10-29]: (b) whenever all 3 HP heal, or (a) against a hit that would
# kill or leave HP within one enemy hit of 0, when the brew prevents it. 0.24-0.27.1 required both, so
# run 20261029-124921 held two brews from HP 7 down to 3 [user review]
def _fight(hero, *enemies):
    return State(7, hero, list(enemies), level="bamboo-2", wave=1, waves_total=5)


def test_edamame_whenever_all_its_hp_heal():
    from shogunbot.bot import edamame_reason
    for hp, drink in ((2, True), (7, True), (8, False), (10, False)):
        calm = _fight(Hero(0, 1, hp=hp), Enemy("ashigaru", 5, -1, 3, uid=1, intent="hold"))
        assert bool(edamame_reason(calm, "wait", 10)) is drink, hp


def test_edamame_at_hp_7_in_whispering_caves_i():
    # run 20261029-124921 t97: HP 7, two Edamame Brews in the slots, no hit coming [user review]
    from shogunbot.bot import edamame_reason
    from shogunbot.perception.perceive import perceive
    p = perceive(frame("runs/20261029-124921/frames/t0097_before.png"), ["swirl", "arrow", "back_charge"],
                 regions=("bamboo_grove", "whispering_caves"), day=1)
    assert [i["item"] for i in p.items if not i.get("dim")].count("green_potion") == 2
    s = Tracker(Kit.of("swirl", "arrow", "back_charge")).state(p)
    assert s.hero.hp == 7 and "heal" in edamame_reason(s, "wait", 10)


def test_edamame_against_a_hit_when_not_all_its_hp_heal():
    # (a) on its own: with 5 max HP and 3 left, a Sword (2) would leave 1; the brew heals 2 of its 3
    from shogunbot.bot import edamame_reason
    s = _fight(Hero(0, 1, hp=3, **BOXED), Enemy("ashigaru", 1, -1, 3, ["sword"], uid=1, intent="fire", wind=2))
    assert "leave 1 HP" in edamame_reason(s, "wait", 5)
    calm = _fight(Hero(0, 1, hp=3), Enemy("ashigaru", 5, -1, 3, uid=1, intent="hold"))
    assert edamame_reason(calm, "wait", 5) is None             # no hit coming and not all of it heals


def test_edamame_against_a_hit_on_the_next_turn():
    # loaded this phase (R1: it cannot fire yet); it fires next turn and the hero cannot get away
    from shogunbot.bot import edamame_reason
    s = _fight(Hero(0, 1, hp=3, **BOXED), Enemy("ashigaru", 1, -1, 3, ["sword"], uid=1, intent="hold", wind=0))
    assert "next" in edamame_reason(s, "wait", 5)


def test_edamame_against_a_kill_only_when_the_brew_prevents_it():
    # two Swords at once: 4 damage. Max HP 5 and 4 makes (b) false, so (a) alone decides
    from shogunbot.bot import edamame_reason
    both = lambda hp: _fight(Hero(1, 1, hp=hp, **BOXED),
                             Enemy("ashigaru", 0, 1, 3, ["sword"], uid=1, intent="fire", wind=2),
                             Enemy("ashigaru", 2, -1, 3, ["sword"], uid=2, intent="fire", wind=2))
    assert "kill" in edamame_reason(both(3), "wait", 5)        # 3 - 4 < 0 without; 5 - 4 = 1 with the brew
    assert edamame_reason(both(2), "wait", 4) is None          # 4 - 4 = 0 with the brew: it would not save him


# ---- 4. an HP drop needs a second frame [user 2026-10-28] -----------------------------
def test_hp_flicker_is_not_taken_as_damage(tmp_path):
    # run 20261028-132952 turn 78 (f0090): a sword animation over the hero hid two pips, read
    # 8 of 8; the next frame (f0091) shows 10 of 10
    tr = Tracker(Kit.of("swirl", "arrow", "shadow_dash"))
    b = _bot(tmp_path, frame("runs/20261028-132952/frames/f0091_CLICK.png"))
    assert b._confirm_hp_drop(8, 10, 10, tr, 78) == 10
    b = _bot(tmp_path, frame("runs/20261028-132952/frames/f0090_A.png"))   # the flicker again
    assert b._confirm_hp_drop(8, 10, 10, tr, 78) == 10                    # 8 pips < 10 seen: not trusted


def test_real_hp_drop_is_confirmed(tmp_path):
    # run 20261028-061420 turn 51: 9 of 10 pips on both frames
    b = _bot(tmp_path, frame("runs/20261028-061420/frames/f0062_CLICK.png"))
    assert b._confirm_hp_drop(9, 10, 10, Tracker(), 51) == 9


# ---- 2. dimmed items: the game dims an item it will not let you use now ----------------
# (all of them while a level ends; on a reward screen the combat cards but not the potion)
DIMMED = {   # frame: {slot y: (item, dimmed)}
    "runs/20261028-132238/frames/item_unknown.png": {952: ("ice_card", True)},
    "runs/20261028-133837/frames/item_unknown.png": {857: ("white_potion", True), 952: ("green_potion", True)},
    "runs/20261028-135635/frames/item_unknown.png": {952: ("white_potion", True)},
    "runs/20261028-061420/frames/f0034_CLICK.png": {952: ("white_potion", True)},
    "collect/20261027-164727/f0106_idle.png": {857: ("green_potion", True), 952: ("curse_card", True)},
    "runs/20261027-182237/frames/f0026_CLICK.png": {857: ("skull_card", True), 952: ("green_potion", False)},
    "runs/20261028-132238/frames/item_ice_card.png": {952: ("ice_card", False)},
}


@pytest.mark.parametrize("rel", list(DIMMED))
def test_dimmed_items_read_as_themselves(rel):
    from shogunbot.perception.perceive import read_consumables
    items = read_consumables(frame(rel))
    assert {round(i["fy"] * 1080): (i["item"], bool(i.get("dim"))) for i in items} == DIMMED[rel]
