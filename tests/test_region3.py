"""Region 3 (0.31): --full goes on to Moonlit Port and The Twins [user 2026-10-29]. Moonlit Port has three
wave levels before its boss (7, 5 and 9 cells, 5 waves each; The Twins' arena 7) [rec: collect/
20261028-190026 f0335-f0582 and collect/20261029-160032 f0462-f0778; regions.json 'observed_day1']."""

from shogunbot.bridge import Tracker, level_name, region_boss_index


def test_moonlit_port_has_three_levels_before_its_boss():
    assert region_boss_index("moonlit_port") == 3
    assert region_boss_index("whispering_caves") == 2 and region_boss_index(None) == 2
    assert level_name(3, 2, 3) == "stage3-3" and level_name(3, 3, 3) == "stage3-boss"
    assert level_name(2, 2) == "stage2-boss" and level_name(1, 2) == "boss"


def test_the_tracker_names_region_3_levels_by_the_route():
    tr = Tracker()
    tr.route = ["bamboo_grove", "whispering_caves", "moonlit_port"]
    names = []
    # Bamboo Grove I, II, boss; Caves I, II, Iwao; Moonlit Port I, II, III, The Twins: a reward after each
    for screen, cells in [("combat", 5), ("combat", 7), ("boss", 5), ("combat", 5), ("combat", 7), ("boss", 7),
                          ("combat", 7), ("combat", 5), ("combat", 9), ("boss", 7)]:
        names.append(tr._level(screen, cells))
        tr.next_level()
    assert names == ["bamboo-1", "bamboo-2", "boss", "stage2-1", "stage2-2", "stage2-boss",
                     "stage3-1", "stage3-2", "stage3-3", "stage3-boss"]
