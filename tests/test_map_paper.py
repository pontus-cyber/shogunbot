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
