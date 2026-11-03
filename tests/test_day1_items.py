"""0.49: the loop's Day 1 requests 15 (a death at 3 HP in Hot Springs III: Dynamic Boost, Cool Up, floor items),
16 (upgrade slots) and 17 (the first run to the Shogun's Castle) [claude/core-requests.md]. Run-log tests skip when the
logs are missing."""
import json

import pytest

from test_bot_flow import DATA, _stub_windows_modules

from shogunbot.model import Enemy, Hero, Kit, State
from shogunbot.planner import plan_live

_stub_windows_modules()


def _logged(run: str, turn: int) -> State:
    path = DATA / f"runs/{run}/turns.jsonl"
    if not path.exists():
        pytest.skip("run log not available")
    row = next(json.loads(l) for l in open(path) if json.loads(l)["turn"] == turn)
    return State.from_dict(row["state"])


# ---------------------------------------------------------------- item 15a
def test_the_step_into_the_hooks_line_is_no_longer_planned():
    # run 20261102-191231 (0.47.1) t653, Hot Springs III at 3 HP: the hero stepped left into the line of a loaded
    # Grappler (Hook then Sword) four cells away, planning to kill it at t654 with Charge, Back Charge and Shuriken: 0.47.1
    # gave the Charge's own strike Dynamic Boost (3 planned, 2 real), the Grappler lived, pulled him and cut him: dead.
    # Since 0.48 a dash strike gets no boost (the Shuriken after a dash that moved the hero still does: 4 of 4 clean
    # logged cases), and the step is planned as the death it was
    s = _logged("20261102-191231", 653)
    d = plan_live(s)
    assert d.action != "left"
    assert next(c for c in d.candidates if c["action"] == "left")["value"] < -50000


# ---------------------------------------------------------------- item 15b
def test_cool_up_is_used_when_a_recharged_kit_saves_hp():
    # cornered on cell 0 by an armed Ashigaru facing him, the swap recharging: every move takes 2. Cool Up recharges the
    # swap too [catalogue: recharge_all], and the swap past the Ashigaru leaves it facing away
    from shogunbot.bot import Bot
    kit = Kit.of("swirl", "arrow")
    hero = Hero(0, 1, hp=4, kit=kit)
    hero.swap_charge = 0
    s = State(5, hero, [Enemy("ashigaru", 1, -1, 3, ["sword"], uid=1, intent="fire")], level="stage5-1", wave=1,
              waves_total=5)
    assert Bot._coolup_gain(s) >= 1.0
    hero.swap_charge = kit.swap_cooldown                    # nothing to recharge: no use
    assert Bot._coolup_gain(s) == 0.0


# ---------------------------------------------------------------- item 17d
def test_a_statue_upgrade_past_cooldown_8_is_not_an_option():
    # run 20261102-232559 (0.48) reward_11_shrine after Spirit Gateway I: Double-Strike (+4 cooldown) went to the
    # Arrow (cooldown 5): three clicks, the tile never went into the statue, the shrine was skipped. All 14 logged
    # upgrades that would have raised a tile above cooldown 8 failed (10 of them Double-Strike on an Arrow), all 19
    # to exactly 8 were taken [wiki: cooldown 0-8; data/upgrades.json rules.cooldown_cap]
    from shogunbot.model import KitTile
    from shogunbot.rewards import upgrade_options
    kit = Kit([KitTile("swirl", "swirl", 3, 2, None, 4, 3), KitTile("arrow", "arrow", 2, 5, None, 4, 0),
               KitTile("arrow#2", "arrow", 4, 5, None, 3, 2), KitTile("shuriken", "shuriken", 1, 3, None, 4, 0),
               KitTile("swirl#2", "swirl", 2, 3, None, 4, 0)])
    names = [o.name for o in upgrade_options(kit, "double_strike")]
    assert names == ["skip", "upgrade:swirl", "upgrade:shuriken", "upgrade:swirl#2"]
    # Free-Play (+3) takes an Arrow to exactly 8: taken in runs 20261028-155154, 20261029-100218 and others
    assert "upgrade:arrow" in [o.name for o in upgrade_options(kit, "freeplay")]


# ---------------------------------------------------------------- item 16
def test_a_slot_goes_to_a_tile_with_no_free_slot_first_the_cooldown_1_one():
    # run 20261102-202249 (0.47.1) reward_8_shrine: '+1 UPGRADE SLOT' went to the Swirl, which had free slots, while
    # the Arrow at cooldown 1 had used all 3 [user 2026-11-02 20:47: "Arrow needed upgrade room ... with a chance of
    # making arrow cooldown free"]
    from shogunbot.model import KitTile
    from shogunbot.rewards import decide_upgrade

    def boom(*a, **k):
        raise AssertionError("no simulation expected")
    kit = Kit([KitTile("swirl", "swirl", 3, 3, None, 5, 1), KitTile("arrow", "arrow", 3, 1, None, 3, 3),
               KitTile("spear", "spear", 3, 6, None, 4, 1), KitTile("swirl#2", "swirl", 2, 3, None, 4, 0)])
    d = decide_upgrade(kit, "slot_1", 10, evaluator=boom)
    assert d.best.name == "upgrade:arrow" and "cooldown 1" in d.rule
    full_spear = kit.replace("spear", KitTile("spear", "spear", 3, 6, None, 4, 4))
    assert decide_upgrade(full_spear, "slot_1", 10, evaluator=boom).best.name == "upgrade:arrow"
    # no tile at cooldown 1 among the full ones: the strongest full tile (catalogue ready value: Spear 8, Arrow 6)
    no_cd1 = full_spear.replace("arrow", KitTile("arrow", "arrow", 3, 4, None, 3, 3))
    assert decide_upgrade(no_cd1, "slot_1", 10, evaluator=boom).best.name == "upgrade:spear"
    # every tile with a free slot: the strongest tile, as before (Spear and Swirl 8; the Spear has fewer free slots)
    free = kit.replace("arrow", KitTile("arrow", "arrow", 3, 1, None, 4, 3))
    assert decide_upgrade(free, "slot_1", 10, evaluator=boom).best.name == "upgrade:spear"


def test_stacked_level_dots_are_read_and_a_missed_row_keeps_the_kits_count():
    # above 4 levels the dots stack in a second row 20 px higher; one row read the Swirl's 6 levels as 4 [user
    # 2026-11-02 20:49; run 20261102-202249 reward_9_shrine]
    import numpy as np
    from PIL import Image
    from shogunbot.bridge import kit_from_faces
    from shogunbot.model import KitTile
    from shogunbot.perception.reward_screen import read_hand
    path = DATA / "runs/20261102-202249/frames/reward_9_shrine.png"
    if not path.exists():
        pytest.skip("frame not available")
    hand = read_hand(np.asarray(Image.open(path).convert("RGB")), dots=True)
    assert [(f["tile"], f["damage"], f["levels"], f["used"]) for f in hand] == [
        ("swirl", 2, 4, 0), ("arrow", 3, 3, 3), ("swirl", 3, 6, 1), ("spear", 3, 4, 1)]
    # a read of one full row (4) where the kit has 6 keeps 6; a read below a full row is trusted
    kit = Kit([KitTile("swirl", "swirl", 3, 3, None, 6, 1), KitTile("arrow", "arrow", 3, 5, None, 3, 0)])
    faces = [{"tile": "swirl", "damage": 3, "damage_conf": 1.0, "cooldown": 3, "levels": 4, "used": 1},
             {"tile": "arrow", "damage": 3, "damage_conf": 1.0, "cooldown": 5, "levels": 2, "used": 0}]
    new = kit_from_faces(kit, faces)
    assert new["swirl"].levels == 6 and new["arrow"].levels == 2


def test_copies_of_a_tile_are_told_apart_by_their_dots():
    # run 20261102-232559 reward_2_shrine: two Arrows at 2/5, 4 and 3 levels, swapped labels ('arrow: read 2/5 L0/4,
    # kit had 2/5 L0/3'), so a click meant for one copy could take the other
    from shogunbot.bridge import match_faces
    from shogunbot.model import KitTile
    kit = Kit([KitTile("arrow", "arrow", 2, 5, None, 4, 0), KitTile("arrow#2", "arrow", 2, 5, None, 3, 0)])
    faces = [{"tile": "arrow", "damage": 2, "damage_conf": 1.0, "cooldown": 5, "levels": 3, "used": 0, "fx": 0.4},
             {"tile": "arrow", "damage": 2, "damage_conf": 1.0, "cooldown": 5, "levels": 4, "used": 0, "fx": 0.5}]
    m = match_faces(kit, faces)
    assert m["arrow"]["fx"] == 0.5 and m["arrow#2"]["fx"] == 0.4


# ---------------------------------------------------------------- item 17b
def test_hp_comes_first_at_the_shop_when_hp_is_low():
    from shogunbot import rewards
    brew = {"kind": "consumable", "id": "edamame_brew", "price": 5, "currency": "coin"}
    healthy = {"kind": "skill", "id": "healthy", "price": 15, "currency": "coin"}
    assert rewards.shop_priority(good=brew) is None and rewards.shop_priority(good=healthy) == 2
    assert rewards.shop_priority(good=brew, low_hp=True) == 0
    assert rewards.shop_priority(good=healthy, low_hp=True) == 0.5 < rewards.shop_priority("cooldown_minus_2")
    cool = {"kind": "consumable", "id": "cool_up", "price": 5, "currency": "coin"}
    assert rewards.shop_priority(good=cool, low_hp=True) is None             # not a heal


def test_the_shop_buys_the_brew_first_at_low_hp(tmp_path, monkeypatch):
    # run 20261102-232559 (0.48): the shop after Hideyoshi at HP 2 of 10 with 69 coins bought no HP (its counter's
    # Edamame Brew 'not bought (consumable)'). Recording 2's shop frame f0783 stands in for that shop
    from test_bot_flow import frame, make_bot
    from test_series import _GW, _Rec
    from shogunbot.bridge import Tracker
    b = make_bot(tmp_path, _GW([(frame("collect/20261029-160032/f0783_idle.png"), [])]))
    b.rec, b.shop_n = _Rec(), 1
    calls = []
    monkeypatch.setattr(b, "_shop_statue", lambda f, o, h, tr, hp, c, n: (calls.append(("statue", c)), (f, c))[1])
    monkeypatch.setattr(b, "_shop_skill", lambda f, g, tr, hp, c: (calls.append((g["id"], c)), (f, c - g["price"]))[1])
    monkeypatch.setattr(b, "_shop_brew", lambda f, g, tr, hp, cap, c: (calls.append((g["id"], c)), (f, c - g["price"]))[1])
    goods = [{"x": 1160, "price": 15, "currency": "coin", "title": "HEALTHY", "kind": "skill", "id": "healthy"},
             {"x": 1498, "price": 5, "currency": "coin", "title": "EDAMAME BREW", "kind": "consumable",
              "id": "edamame_brew"}]
    offer = {"upgrade": "cooldown_minus_2", "upgrade_price": 20, "coins": 69, "goods": goods}
    b._shop_buy(b.gw.grab(), offer, [], Tracker(Kit.of("swirl", "arrow")), 2, 10)
    assert calls == [("edamame_brew", 69), ("healthy", 64), ("statue", 49)]
    calls.clear()
    b._shop_buy(b.gw.grab(), offer, [], Tracker(Kit.of("swirl", "arrow")), 8, 10)    # HP 8 of 10: no brew
    assert calls == [("statue", 69), ("healthy", 69)]


# ---------------------------------------------------------------- item 17c
def test_the_copy_the_bot_queued_is_the_one_off_the_hand():
    # run 20261102-232559 t426-t431: the bot queued the 2-damage Arrow; the 4-damage copy left in the hand read with an
    # unsure digit, was matched to 'arrow', and the queue was tracked as 'arrow#2': t431 planned the Shinobi killed by 4
    # before its Shuriken, the 2 left it at 2 HP and it threw (-1)
    from shogunbot.bridge import Tracker
    from shogunbot.model import KitTile
    tr = Tracker(Kit([KitTile("arrow", "arrow", 2, 5, None, 4, 0), KitTile("arrow#2", "arrow", 4, 5, None, 3, 2),
                      KitTile("swirl", "swirl", 3, 2, None, 4, 3)]))
    tr.queue_track = ["arrow"]
    hand = [{"tile": "arrow", "damage": 2, "damage_conf": 0.4, "cooldown": 5, "charged": 5, "ready": True},
            {"tile": "swirl", "damage": 3, "damage_conf": 1.0, "cooldown": 2, "charged": 2, "ready": True}]
    assert tr._queue({"queue": ["arrow"], "queue_conf": [0.9]}, hand) == ["arrow"]
    # a sure digit on the face left in the hand wins over the bot's own record
    hand[0].update(damage=2, damage_conf=1.0)
    tr.queue_track = ["arrow#2"]
    assert tr._queue({"queue": ["arrow"], "queue_conf": [0.9]}, hand) == ["arrow#2"]
    hand[0].update(damage=4, damage_conf=1.0)
    assert tr._queue({"queue": ["arrow"], "queue_conf": [0.9]}, hand) == ["arrow"]


def test_a_queued_copy_gets_no_hand_face():
    # with the queue known, the face in the hand belongs to the copy that is not queued (else it is planned recharging)
    from shogunbot.bridge import match_faces
    from shogunbot.model import KitTile
    kit = Kit([KitTile("arrow", "arrow", 2, 5, None, 4, 0), KitTile("arrow#2", "arrow", 4, 5, None, 3, 2)])
    face = {"tile": "arrow", "damage": None, "cooldown": 5, "charged": 5, "ready": True}
    assert list(match_faces(kit, [face], skip=("arrow",))) == ["arrow#2"]
    assert list(match_faces(kit, [face], skip=("arrow#2",))) == ["arrow"]


def test_a_shield_does_not_outlast_its_fight():
    # run 20261102-232559: a Reactive Shield from Spirit Gateway II was still tracked in III, and the Ashigaru's Sword at
    # t440 was planned as blocked (-2); the same from Kowa into Keep I (t803, -2)
    from test_bot_flow import frame
    from shogunbot.bridge import Tracker
    from shogunbot.perception.perceive import perceive
    route = ("bamboo_grove", "whispering_caves", "moonlit_port", "spirit_gateway")
    kit = Kit.of("swirl", "arrow", "shuriken").with_skill("reactive_shield")
    tr = Tracker(kit)
    tr.route, tr.stage, tr.level_i, tr.level = list(route), 4, 1, "stage4-2"
    tr.shield = True
    tr.next_level()
    s = tr.state(perceive(frame("runs/20261102-232559/frames/t0440_before.png"), ["swirl", "arrow", "shuriken"],
                          regions=route, day=1))
    assert s.level == "stage4-3" and not s.hero.shield


def test_the_shoguns_first_bar_loads_facing_the_hero():
    # run 20261103-020829 t897: tracked facing away since his Mirror, the Shogun loaded the Sword, stepped right twice
    # toward the hero and cut him (-2 planned as a miss). On his second bar his facing persists from his combos
    from shogunbot.bridge import Tracker
    tr = Tracker()
    tr.prev = [("the_shogun", 2, -1, "move-right")]
    e = {"x": 3, "facing": -1}
    tr.boss_phase = {"the_shogun": 1}
    assert tr._enemy_facing("the_shogun", e, ["sword"], "fire", 4, 0.5) == 1
    tr.boss_phase = {"the_shogun": 2}
    assert tr._enemy_facing("the_shogun", e, ["sword"], "fire", 4, 0.5) == -1


def test_a_bubble_that_flashed_as_it_fired_is_a_fire():
    # run 20261102-232559 t241, Moonlit Port III: the Guardian next to the hero fired its Sword while its bubble washed
    # out yellow and white; read as an empty bubble, it was planned as holding and the wait cost 2 (also run
    # 20261102-065902 t748, an Ashigaru)
    from test_bot_flow import frame
    from shogunbot.bridge import Tracker
    from shogunbot.perception.perceive import perceive
    route = ("bamboo_grove", "whispering_caves", "moonlit_port")
    tr = Tracker(Kit.of("swirl", "arrow", "shuriken"))
    tr.route, tr.stage, tr.level_i, tr.level = list(route), 3, 2, "stage3-3"
    tr.prev = [("yumi_sniper", 0, 1, "queue"), ("guardian", 1, 1, "move-right"), ("yari_master", 4, -1, "queue"),
               ("yari_master", 6, -1, "hold")]                       # t240 as logged
    tr.prev_queues = [[], ["sword"], [], ["spear"]]
    p = perceive(frame("runs/20261102-232559/frames/t0241_before.png"), ["swirl", "arrow", "shuriken"],
                 regions=route, day=1)
    g = next(e for e in p.enemies if e["x"] == 2)
    assert g["kind"] == "guardian" and g["flash"] and g["intent"] == "fire"
    s = tr.state(p)
    en = s.enemy_at(2)
    assert en.queue == ["sword"] and en.intent == "fire" and en.facing == 1
    d = plan_live(s)
    assert next(c for c in d.candidates if c["action"] == "wait")["exp_damage"] >= 2
