"""0.38: the map after Moonlit Port is lighter paper (168,138,105) and read as 'unknown'
(tan 0.066 < 0.1), which stopped run 20261031-114119 on the map. Frame-based tests skip
when the frames are missing."""

import pytest
from test_bot_flow import frame

from shogunbot.perception import screens


@pytest.mark.parametrize("rel", [
    "runs/20261031-114119/frames/stuck.png",          # after Moonlit Port's shop, HP 10/12
    "collect/20261029-160032/f1515_CLICK.png",
    "collect/20261029-160032/f0797_idle.png",
    "collect/20261028-190026/f1361_CLICK.png",        # LOCATION CLEARED banner
    "collect/20261028-190026/f0597_CLICK.png",
])
def test_light_paper_map_reads_as_map(rel):
    assert screens.classify(frame(rel)) == "map"


@pytest.mark.parametrize("rel", [
    "runs/20261031-114119/frames/t0140_after.png",    # Moonlit Port fight: paper 0.29, HUD on
    "runs/20261030-041652/frames/t0181_after.png",
])
def test_paper_coloured_fight_is_not_a_map(rel):
    assert screens.classify(frame(rel)) != "map"


# 0.39: Moonlit Port III's camera panned a 9-cell board to x0 140, under the item column's edge
# [run 20261031-123442, after t225: no board for 90 s, the run stopped]
def test_panned_nine_cell_board_under_the_item_column():
    from shogunbot.perception import board
    cs = board.cells(frame("runs/20261031-123442/frames/unknown_1793447769.png"))
    assert [c["x0"] for c in cs] == list(range(140, 1261, 140))
    assert [c["kind"] for c in cs][6:] == ["enemy", "empty", "hero"]


def test_a_row_that_may_run_on_under_the_item_column_stays_unread():
    from shogunbot.perception import board
    # a 5-cell row whose next plate lies under the potions: the width is unknown
    assert not board.is_board(board.cells(frame("collect/20261029-160032/f0547_idle.png")))


# 0.41: the yellow move arrow over a red enemy plate made one too-wide run whose split pieces
# set the grid 92 px off [run 20261031-144722, Moonlit Port II wave 5/5, after t256]
def test_split_run_does_not_set_the_grid():
    from shogunbot.perception import board
    cs = board.cells(frame("runs/20261031-144722/frames/stuck.png"))
    assert [(c["x0"], c["kind"]) for c in cs] == [(520, "empty"), (660, "empty"), (800, "empty"),
                                                   (940, "enemy"), (1080, "hero")]


# 0.42: Spirit Gateway's red light made the hero's shoulder patch grey; the facing went unread and the
# tracker kept 'right' while he faced the wall [run 20261031-165514 t568-t605: 12 -> 2 HP]
def test_hero_facing_under_red_light():
    from shogunbot.perception import perceive
    for rel in ("runs/20261031-165514/frames/t0572_before.png", "runs/20261031-165514/frames/t0572_after.png"):
        assert perceive.perceive(frame(rel)).hero["facing"] == -1


def test_weak_contrary_hero_facing_reads_override_after_three():
    from shogunbot.bridge import Tracker
    tr = Tracker()
    tr.facing, tr.facing_known = 1, True
    assert tr._hero_facing({"facing": -1}, 0.5) == 1
    assert tr._hero_facing({"facing": -1}, 0.5) == 1
    assert tr._hero_facing({"facing": -1}, 0.5) == -1
    assert tr._hero_facing({"facing": 1}, 0.06) == -1          # a weak read below 0.5 never counts


# 0.43: an unrecognised hand face goes to the one free kit label with its cooldown and damage
# [run 20261031-154544: 'queue:swirl: tile not found in hand' 18 times, the face read tile None, 2/5]
def test_unread_face_goes_to_the_only_label_with_its_stats():
    from shogunbot.bridge import match_faces
    from shogunbot.model import Kit
    kit = Kit.from_list([{"label": "swirl", "tile": "swirl", "damage": 2, "cooldown": 5},
                         {"label": "arrow", "tile": "arrow", "damage": 2, "cooldown": 8},
                         {"label": "spear", "tile": "spear", "damage": 2, "cooldown": 5},
                         {"label": "shuriken", "tile": "shuriken", "damage": 1, "cooldown": 3}])
    face = {"tile": None, "conf": 0.42, "damage": 2, "damage_conf": 0.8, "cooldown": 5, "charged": 5,
            "ready": True, "cx": 960}
    assert match_faces(kit, [dict(face)]) == {}                              # swirl or spear: unread
    got = match_faces(kit, [dict(face)], skip=("spear",))                   # the spear is queued
    assert list(got) == ["swirl"] and got["swirl"]["guessed"]
    assert match_faces(kit, [dict(face)], skip=("spear",), guess=False) == {}
