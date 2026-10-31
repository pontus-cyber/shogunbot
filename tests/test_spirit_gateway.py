"""0.36: --full goes on to region 4, Spirit Gateway and The Statue [user review of run 20261030-213120: "raise
FULL_LAST_STAGE to 4 (Spirit Gateway, The Statue)"]. Both recordings took Spirit Gateway after Moonlit Port's shop
(collect/20261028-190026 f0597, collect/20261029-160032 f0798): three wave levels of 7, 7 and 9 cells, then The
Statue's 7-cell arena. Frame tests skip when the frames are missing."""

from test_bot_flow import _stub_windows_modules, frame

from shogunbot.bridge import Tracker

_stub_windows_modules()

REC1 = "collect/20261028-190026/"
REC2 = "collect/20261029-160032/"


def test_the_map_names_spirit_gateway_and_the_bot_prefers_it():
    from shogunbot import bot as B
    from shogunbot.perception import map_screen as M
    sel = M.selected(frame(REC1 + "f0597_CLICK.png"))          # the tooltip over the node after the shop
    assert (sel["kind"], sel["region"]) == ("region", "spirit_gateway")
    assert "spirit_gateway" in B.MAP_REGIONS and B.FULL_LAST_STAGE == 4


def test_spirit_gateway_counts_three_wave_levels_before_the_statue():
    tr = Tracker()
    tr.route = ["bamboo_grove", "whispering_caves", "moonlit_port", "spirit_gateway"]
    tr.stage, tr.level_i = 4, -1
    names = []
    for screen, cells in (("combat", 7), ("combat", 7), ("combat", 9), ("boss", 7)):
        if names:
            tr.next_level()
        names.append(tr._level(screen, cells))
    assert names == ["stage4-1", "stage4-2", "stage4-3", "stage4-boss"]
