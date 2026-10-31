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
