"""M2: reward screens, kit tracking and reward choice.

Frame-based tests read recorded frames from the run folders (runs/, collect/)
next to this repo on the PC, or from $SHOGUNBOT_DATA; they are skipped when
the frames are not there (the frames are never committed).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from shogunbot import rewards
from shogunbot.bridge import Tracker, kit_from_faces, match_faces
from shogunbot.catalogue import CAT
from shogunbot.model import DEFAULT_KIT, Kit, KitTile

DATA = Path(os.environ.get("SHOGUNBOT_DATA", Path(__file__).resolve().parents[1]))


def frame(rel):
    p = DATA / rel
    if not p.exists():
        pytest.skip(f"recorded frame not available: {rel}")
    from PIL import Image
    return np.array(Image.open(p).convert("RGB"))


# ---- screen reading on recorded frames ----------------------------------------
TILE_CHOICES = {
    "runs/20261027-180145/frames/f0030_CLICK.png": [("arrow", 2, 5, 3), ("katana", 2, 0, 2)],
    "runs/20261027-182237/frames/f0026_CLICK.png": [("back_charge", 1, 3, 3), ("shuriken", 1, 3, 2)],
    "runs/20261028-044407/frames/f0034_CLICK.png": [("shuriken", 1, 3, 2), ("charge", 1, 4, 4)],
    "runs/20261028-053637/frames/f0032_CLICK.png": [("charge", 1, 4, 4), ("swirl", 2, 3, 2)],
    "runs/20261028-061420/frames/f0034_CLICK.png": [("back_charge", 1, 3, 3), ("swirl", 2, 3, 3)],
    "collect/20261027-164727/f0042_CLICK.png": [("shadow_dash", 1, 5, 4), ("spear", 2, 5, 2)],
    "runs/20261028-082756/frames/reward_1_tile_choice.png": [("spear", 2, 5, 3), ("katana", 2, 0, 4)],
    # a bird flying past the right pedestal shifted its box 12 px; 0.20 read 'None None/5 L2'
    "runs/20261028-153448/frames/reward_1_tile_choice.png": [("back_charge", 1, 3, 4), ("shadow_dash", 1, 5, 2)],
}
SHRINES = {
    "runs/20261027-182237/frames/f0135_CLICK.png": "slot_1_cooldown_minus_1",
    "runs/20261028-044407/frames/f0097_CLICK.png": "cooldown_minus_1",
    "runs/20261028-044407/frames/win.png": "cooldown_minus_1",
    "runs/20261028-053637/frames/f0103_CLICK.png": "damage_1_cooldown_1",
    "runs/20261028-061420/frames/f0089_CLICK.png": "damage_2_cooldown_3",
    "runs/20261028-061420/frames/win.png": "cooldown_minus_1",
    "collect/20261027-164727/f0109_idle.png": "cooldown_minus_2",
    "runs/20261028-082756/frames/reward_2_shrine.png": "damage_1",                 # first "+1 DAMAGE" seen
    "runs/20261028-082756/frames/reward_3_shrine.png": "slot_1_cooldown_minus_1",
    # 0.20: the bubble grows upward with its text; 0.19 searched from y 150 and missed these
    "runs/20261028-130456/frames/reward_2_shrine.png": "cooldown_minus_4_damage_minus_1",  # '-4' unread in 0.19
    "runs/20261028-132238/frames/reward_3_shrine.png": "freeplay",          # FREE-PLAY / +3 COOLDOWN
    "runs/20261028-132952/frames/reward_3_shrine.png": "perfect_strike",    # PERFECT STRIKE / +1 COOLDOWN
    "collect/20261027-164727/f0154_idle.png": "freeplay",
    "runs/20261028-133837/frames/reward_3_shrine.png": "cooldown_minus_1",
    "runs/20261028-134818/frames/reward_3_shrine.png": "damage_1_cooldown_1",
}


@pytest.mark.parametrize("rel", list(TILE_CHOICES))
def test_tile_choice_offers_read(rel):
    from shogunbot.perception import reward_screen
    rs = reward_screen.read(frame(rel), ("swirl", "arrow"))
    assert rs.kind == "tile_choice"
    assert [(o["tile"], o["damage"], o["cooldown"], o["levels"]) for o in rs.offers] == TILE_CHOICES[rel]
    assert rs.skip is not None and [h["tile"] for h in rs.hand if h["tile"]]


@pytest.mark.parametrize("rel", list(SHRINES))
def test_shrine_upgrade_read(rel):
    from shogunbot.perception import reward_screen
    rs = reward_screen.read(frame(rel), ("swirl", "arrow", "spear"))
    assert rs.kind == "shrine" and rs.upgrade == SHRINES[rel] and not rs.upgrade_lit


def test_shrine_lights_up_with_a_tile_in_the_statue():
    from shogunbot.perception import reward_screen
    assert reward_screen.read(frame("collect/20261027-164727/f0111_CLICK.png"), ("swirl", "arrow", "spear")).upgrade_lit


def test_hand_marks_are_the_charge():
    # run 20261028-061420: turn 5 (f0014) arrow 3/5 swirl 0/3; turn 6 (f0015) 4/5, 1/3; turn 11 (f0020) 2/5, 0/3
    from shogunbot.perception.perceive import read_hand
    for rel, want in [("runs/20261028-061420/frames/f0014_A.png", {"arrow": 3, "swirl": 0}),
                      ("runs/20261028-061420/frames/f0015_S.png", {"arrow": 4, "swirl": 1}),
                      ("runs/20261028-061420/frames/f0020_S.png", {"arrow": 2, "swirl": 0})]:
        hand = read_hand(frame(rel))
        assert {h["tile"]: h["charged"] for h in hand} == want


def test_spear_taken_in_bamboo_grove_one_joins_the_kit():
    # collect 20261027-164727: the user took a Spear after Bamboo Grove I; the hand at the
    # next shrine shows Spear, Swirl, Arrow
    from shogunbot.perception import reward_screen
    rs = reward_screen.read(frame("collect/20261027-164727/f0109_idle.png"), ("swirl", "arrow", "spear"))
    notes = []
    kit = kit_from_faces(DEFAULT_KIT, rs.hand, notes)
    assert sorted(kit.labels) == ["arrow", "spear", "swirl"] and kit["spear"].levels == 2


# ---- kit logic -------------------------------------------------------------------
def test_kit_from_faces_keeps_the_kit_when_a_face_is_misread():
    faces = [{"tile": "swirl", "cooldown": 3, "charged": 3}, {"tile": "spear", "cooldown": 5, "charged": 5}]
    notes = []
    assert kit_from_faces(DEFAULT_KIT, faces, notes) == DEFAULT_KIT and notes


def test_second_copy_faces_go_to_the_closest_charge():
    kit = Kit.of("swirl", "arrow", "swirl")
    faces = [{"tile": "swirl", "cooldown": 3, "charged": 1}, {"tile": "swirl", "cooldown": 3, "charged": 3}]
    m = match_faces(kit, faces, {"swirl": 3, "swirl#2": 1, "arrow": 5})
    assert m["swirl"]["charged"] == 3 and m["swirl#2"]["charged"] == 1


def test_upgrade_options_respect_levels_and_enchantments():
    kit = Kit([KitTile("swirl", "swirl", 2, 3, None, 3, 3), KitTile("arrow", "arrow", 2, 5, "poison", 3, 0)])
    names = [o.name for o in rewards.upgrade_options(kit, "cooldown_minus_1")]
    assert names == ["skip", "upgrade:arrow"]            # swirl has no free level
    names = [o.name for o in rewards.upgrade_options(kit, "slot_1")]
    assert names == ["skip", "upgrade:swirl", "upgrade:arrow"]   # slot upgrades use no level
    names = [o.name for o in rewards.upgrade_options(kit, "ice")]
    assert names == ["skip"]                              # arrow holds poison; swirl has no level
    up = rewards.upgraded(KitTile("arrow", "arrow", 2, 5), CAT.upgrade("cooldown_minus_4_damage_minus_1"))
    assert (up.damage, up.cooldown, up.used) == (1, 1, 1)


def test_unknown_or_unsimulated_offers_are_never_taken():
    offers = [{"tile": None, "damage": 2, "cooldown": 3}, {"tile": "trap", "damage": 3, "cooldown": 4}]
    assert [o.name for o in rewards.tile_choice_options(DEFAULT_KIT, offers)] == ["skip"]


# ---- reward rules (0.16) ----------------------------------------------------------
def fake_eval(table):
    """An evaluator that gives each option fixed per-seed scores: table[name] -> list."""
    def ev(opts, hp, seeds, depth, workers, scenario):
        for o in opts:
            o.scores = list(table[o.name])
            o.value = round(sum(o.scores) / len(o.scores), 3)
            o.wins = len(o.scores)
        rewards.paired(opts)
    return ev


def test_pure_gain_classification():
    gains = {u.id for u in CAT.upgrades.values() if rewards.pure_gain(u)}
    assert gains == {"damage_1", "cooldown_minus_1", "cooldown_minus_2", "slot_1", "slot_1_cooldown_minus_1"}
    assert not rewards.offerable(CAT.upgrade("sacrifice")) and not rewards.offerable(CAT.upgrade("warriors_gamble"))
    assert not any(rewards.pure_gain(u) for u in CAT.upgrades.values() if u.kind == "enchant")


def test_pure_gain_upgrades_are_always_taken():
    # [run 20261028-082756] '+1 DAMAGE' and '+1 UPGRADE SLOT -1 COOLDOWN' were skipped because
    # skip scored 9.98 vs 9.96 and 9.77 vs 9.44: noise, every simulated run won at ~full HP
    kit = Kit.of("swirl", "arrow", "spear")
    skip_wins = {"skip": [10.0] * 4, "upgrade:swirl": [9.0] * 4, "upgrade:arrow": [8.0] * 4,
                 "upgrade:spear": [9.5] * 4}
    for uid in ("damage_1", "slot_1_cooldown_minus_1", "cooldown_minus_2"):
        d = rewards.decide_upgrade(kit, uid, 10, evaluator=fake_eval(skip_wins))
        assert d.best.name == "upgrade:spear" and "pure gain" in d.rule, (uid, d.rule)


def test_pure_gain_on_the_only_tile_needs_no_simulation():
    kit = Kit([KitTile("swirl", "swirl", 2, 3, None, 3, 3), KitTile("arrow", "arrow", 2, 5, None, 3, 0)])
    def boom(*a):
        raise AssertionError("no simulation expected")
    assert rewards.decide_upgrade(kit, "damage_1", 10, evaluator=boom).best.name == "upgrade:arrow"


def test_slot_only_goes_to_the_strongest_tile():
    kit = Kit.of("arrow", "swirl")      # Swirl ready value 8 > Arrow 6
    def boom(*a):
        raise AssertionError("no simulation expected")
    assert rewards.decide_upgrade(kit, "slot_1", 10, evaluator=boom).best.name == "upgrade:swirl"


def test_upgrade_that_changes_nothing_is_not_an_option():
    kit = Kit.of("katana", "swirl")     # Katana already has cooldown 0
    assert [o.name for o in rewards.upgrade_options(kit, "cooldown_minus_1")] == ["skip", "upgrade:swirl"]


def test_tradeoff_skips_only_when_every_option_is_a_net_loss():
    kit = Kit.of("swirl", "arrow")
    worse = {"skip": [10, 9, 10, 8], "upgrade:swirl": [8, 7, 8, 6], "upgrade:arrow": [9, 8, 9, 7]}
    d = rewards.decide_upgrade(kit, "damage_2_cooldown_3", 10, evaluator=fake_eval(worse))
    assert d.best.name == "skip" and "net loss" in d.rule
    noisy = {"skip": [10, 9, 10, 8], "upgrade:swirl": [8, 7, 8, 6], "upgrade:arrow": [10, 10, 7, 8]}
    d = rewards.decide_upgrade(kit, "damage_2_cooldown_3", 10, evaluator=fake_eval(noisy))
    assert d.best.name == "upgrade:arrow"     # -0.5 +- 0.87: not a net loss, so taken


def test_rule_text_says_when_the_option_taken_scored_below_skipping():
    # [run 20261028-154317, Bamboo Grove II '-1 DAMAGE / -4 COOLDOWN'] every option scored below
    # skipping, Arrow -0.20 +- 0.29; it was taken as intended (no option a net loss), but the
    # log said 'best option that is not a net loss' [user review of the 0.20 series]
    kit = Kit.of("swirl", "arrow")
    below = {"skip": [10, 9, 10, 8], "upgrade:swirl": [9, 9, 9, 7], "upgrade:arrow": [10, 9, 9, 8]}
    d = rewards.decide_upgrade(kit, "cooldown_minus_4_damage_minus_1", 10, evaluator=fake_eval(below))
    assert d.best.name == "upgrade:arrow"
    assert d.rule.startswith("trade-off: every option scored below skipping; upgrade:arrow scores -0.25")
    assert "a loss within the noise, so it is taken" in d.rule
    above = {"skip": [9, 9, 9, 8], "upgrade:swirl": [10, 9, 10, 9], "upgrade:arrow": [9, 9, 9, 8]}
    d = rewards.decide_upgrade(kit, "damage_2_cooldown_3", 10, evaluator=fake_eval(above))
    assert d.best.name == "upgrade:swirl" and d.rule == "trade-off: upgrade:swirl scores +0.75 +- 0.25 HP against skipping, the best option"


def test_tile_choice_takes_an_offer_unless_all_are_net_losses():
    offers = [{"tile": "katana", "damage": 2, "cooldown": 0, "levels": 4},
              {"tile": "spear", "damage": 2, "cooldown": 5, "levels": 3}]
    close = {"skip": [9.9, 9.8, 10.0, 9.7], "take:katana": [9.8, 9.9, 9.6, 9.7], "take:spear": [9.7, 9.8, 9.9, 9.5]}
    d = rewards.decide_tile(DEFAULT_KIT, offers, 10, evaluator=fake_eval(close))
    assert d.best.name == "take:katana"


def test_valuation_runs_are_repeatable_and_share_the_fight():
    from shogunbot.selfplay import plan_levels, run_once
    assert plan_levels(9001, scenario="stress") == plan_levels(9001, scenario="stress")
    a = run_once(9001, 1, kit=Kit.of("swirl", "arrow"), scenario="stress")
    b = run_once(9001, 1, kit=Kit.of("swirl", "arrow"), scenario="stress")
    assert a == b


def test_tracker_queue_comes_from_the_hand():
    # Swirl and Arrow queued: the hand is empty and two bubbles sit over the hero
    from shogunbot.perception.perceive import perceive
    p = perceive(frame("runs/20261028-061420/frames/f0025_SPACE.png"))
    tr = Tracker()
    s = tr.state(p)
    assert s.hero.queue == ["swirl", "arrow"]


def test_sparkle_after_a_recharge_is_not_a_cooldown_mark():
    # a tile that has just recharged sparkles over its marks; 0.15 read Swirl as 4 marks
    # on 15 frames [review of 0.15; these frames]
    from shogunbot.perception.perceive import read_hand
    for rel in ["runs/20261028-061420/frames/f0023_CLICK.png", "runs/20261028-061420/frames/f0029_CLICK.png",
                "runs/20261028-061420/frames/f0071_CLICK.png", "runs/20261027-180145/frames/f0083_CLICK.png",
                "runs/20261028-044407/frames/f0042_CLICK.png", "runs/20261028-082756/frames/f0032_CLICK.png"]:
        hand = read_hand(frame(rel))
        assert {h["tile"]: h["cooldown"] for h in hand} == {"swirl": 3, "arrow": 5}, rel


# ---- tracker: kit tiles the hand does not show [review of 0.15] ----------------------
def face(tile, charged=None, **kw):
    t = CAT.tiles[tile]
    f = {"tile": tile, "cooldown": t.cooldown, "charged": t.cooldown if charged is None else charged,
         "damage": t.damage, "damage_conf": 1.0, "conf": 0.9, "dist": 0.01, "fx": 0.5, "fy": 0.88}
    f["ready"] = f["charged"] >= f["cooldown"]
    f.update(kw)
    return f


def fake_perception(hand, bubbles=(), cells=7):
    from types import SimpleNamespace
    hero = {"x": 3, "facing": 1, "hp": 10, "hp_max": 10, "queue": list(bubbles), "swap_red": 0}
    return SimpleNamespace(hero=hero, hand=hand, cells=list(range(cells)), screen="combat", enemies=[],
                           confidence={"hero_facing": 1.0}, boss_frac=None, items=[])


def test_copies_of_a_tile_are_told_apart_by_cooldown():
    kit = Kit.of("swirl", "arrow").add("swirl", cooldown=2)
    faces = [face("swirl", cooldown=2, charged=2), face("swirl")]
    m = match_faces(kit, faces)
    assert m["swirl#2"]["cooldown"] == 2 and m["swirl"]["cooldown"] == 3


def test_kit_tile_the_hand_does_not_show_is_replaced_by_a_confident_read():
    # the kit had Spear (a reward assumed taken) but the hand shows Katana
    notes = []
    kit = kit_from_faces(Kit.of("swirl", "arrow", "spear"), [face("swirl"), face("arrow"), face("katana")], notes)
    assert kit.labels == ("swirl", "arrow", "katana") and kit["katana"].cooldown == 0 and notes


def test_unsure_read_does_not_replace_a_kit_tile():
    kit = Kit.of("swirl", "arrow", "spear")
    faces = [face("swirl"), face("arrow"), face("katana", conf=0.2, dist=0.08)]
    assert kit_from_faces(kit, faces, []) == kit


def test_kit_tile_with_no_face_is_planned_as_recharging():
    # planned as ready, the bot chose it, found no face to click and stalled
    tr = Tracker(Kit.of("swirl", "arrow", "spear"))
    s = tr.state(fake_perception([face("swirl"), face("arrow")]))
    assert "spear" not in s.hero.ready_labels() and sorted(s.hero.ready_labels()) == ["arrow", "swirl"]
    assert tr.charge["spear"] == 5          # the tracked charge is kept


def test_full_queue_with_an_empty_hand():
    tr = Tracker(Kit.of("swirl", "arrow", "spear"))
    assert tr._queue({"queue": ["arrow", "swirl", "spear"]}, []) == ["arrow", "swirl", "spear"]


def test_misread_bubble_keeps_the_bubbles_that_were_read():
    tr = Tracker(Kit.of("swirl", "arrow", "spear"))
    assert tr._queue({"queue": ["arrow", "katana", "swirl"]}, []) == ["arrow", "spear", "swirl"]


def test_misread_duplicate_bubble_falls_back_to_the_queued_order():
    # Spear read as a second Arrow: the order the bot queued in decides
    tr = Tracker(Kit.of("swirl", "arrow", "spear"))
    tr.queue_track = ["spear", "arrow"]
    assert tr._queue({"queue": ["arrow", "arrow"]}, [face("swirl")]) == ["spear", "arrow"]


# ---- run 20261028-092419 (0.15): first live shrines ------------------------------------
LIT = {
    "runs/20261028-092419/frames/f0089_CLICK.png": (False, None),
    "runs/20261028-092419/frames/f0090_CLICK.png": (True, "swirl"),     # cursor moved away: brown, not peach
    "runs/20261028-092419/frames/f0091_CLICK.png": (True, "katana"),
    "runs/20261028-092419/frames/f0124_CLICK.png": (False, None),
    "runs/20261028-092419/frames/f0125_CLICK.png": (True, "katana"),
    "runs/20261028-092419/frames/f0126_CLICK.png": (True, "arrow"),
    "collect/20261027-164727/f0111_CLICK.png": (True, "swirl"),         # cursor on UPGRADE: peach
}


@pytest.mark.parametrize("rel", list(LIT))
def test_upgrade_lit_and_the_tile_in_the_statue(rel):
    from shogunbot.perception import reward_screen
    rs = reward_screen.read(frame(rel), ("swirl", "arrow", "katana", "spear"))
    lit, tile = LIT[rel]
    assert rs.kind == "shrine" and rs.upgrade_lit == lit
    assert (rs.statue or {}).get("tile") == tile


def test_queued_katana_is_not_queued_again():
    # boss turns 117-150: hand empty, Katana's bubble read as another tile; 0.15 kept
    # trying to queue Katana (47 act_failed)
    from shogunbot.perception.perceive import perceive
    from shogunbot.rules import legal_hero_actions
    tr = Tracker(Kit.of("swirl", "arrow", "katana"))
    s = tr.state(perceive(frame("runs/20261028-092419/frames/t0118_before.png"), ["swirl", "arrow", "katana"]))
    assert s.hero.queue == ["arrow", "swirl", "katana"]
    assert not [a for a in legal_hero_actions(s) if a.startswith("queue:")]


def test_glyph_samples_read_back_exactly():
    # every frame the shrine font is learned from reads back as its labelled lines
    from shogunbot.perception import text
    for rel, known in text.GLYPH_SAMPLES.items():
        rgb = frame(rel)
        assert [s for s, _ in text.read_text(rgb)] == known, rel


def test_enchantment_shrines_are_trade_offs_the_simulator_scores():
    from shogunbot.catalogue import CAT
    for uid in ("freeplay", "perfect_strike"):
        up = CAT.upgrade(uid)
        assert up.kind == "enchant" and not rewards.pure_gain(up) and rewards.offerable(up)
        assert len(rewards.upgrade_options(Kit.of("swirl", "arrow"), uid)) == 3   # skip + two tiles
