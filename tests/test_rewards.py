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
}
SHRINES = {
    "runs/20261027-182237/frames/f0135_CLICK.png": "slot_1_cooldown_minus_1",
    "runs/20261028-044407/frames/f0097_CLICK.png": "cooldown_minus_1",
    "runs/20261028-044407/frames/win.png": "cooldown_minus_1",
    "runs/20261028-053637/frames/f0103_CLICK.png": "damage_1_cooldown_1",
    "runs/20261028-061420/frames/f0089_CLICK.png": "damage_2_cooldown_3",
    "runs/20261028-061420/frames/win.png": "cooldown_minus_1",
    "collect/20261027-164727/f0109_idle.png": "cooldown_minus_2",
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


def test_decide_is_repeatable_and_takes_on_ties():
    offers = [{"tile": "katana", "damage": 2, "cooldown": 0, "levels": 2}]
    a = rewards.decide(rewards.tile_choice_options(DEFAULT_KIT, offers), "bamboo-1", 10,
                       seeds=(1, 2), depth=1, workers=1)
    b = rewards.decide(rewards.tile_choice_options(DEFAULT_KIT, offers), "bamboo-1", 10,
                       seeds=(1, 2), depth=1, workers=1)
    assert [(o.name, o.value) for o in a[1]] == [(o.name, o.value) for o in b[1]]
    opts = [rewards.Option("skip", DEFAULT_KIT, 5.0), rewards.Option("take:katana", DEFAULT_KIT, 5.0)]
    assert sorted(opts, key=lambda o: (-o.value, o.name == "skip"))[0].name == "take:katana"


def test_tracker_queue_comes_from_the_hand():
    # Swirl and Arrow queued: the hand is empty and two bubbles sit over the hero
    from shogunbot.perception.perceive import perceive
    p = perceive(frame("runs/20261028-061420/frames/f0025_SPACE.png"))
    tr = Tracker()
    s = tr.state(p)
    assert s.hero.queue == ["swirl", "arrow"]
