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


# ---- The Twins [wiki: The Twins] [rec: collect/20261028-190026 f0544-f0580, collect/20261029-160032 f0727-f0775]

from shogunbot.model import Enemy, Hero, State
from shogunbot.rules import Events, _damage_enemy, _enemy_fire, _pair_load
from shogunbot.catalogue import CAT


def _twins(lx, rx, hx, hero_facing=1, hp=35):
    return State(7, Hero(hx, hero_facing), [Enemy("the_twins", lx, 1, hp, uid=1), Enemy("the_twins", rx, -1, hp, uid=2)],
                 level="stage3-boss")


def test_the_twins_share_one_hp_bar():
    s = _twins(0, 6, 3)
    ev = Events()
    _damage_enemy(s, s.enemies[0], 5, "hero:spear", ev)
    assert [e.hp for e in s.enemies] == [30, 30]
    _damage_enemy(s, s.enemies[1], 40, "hero:spear", ev)
    assert [e.hp for e in s.enemies] == [-10, -10] and len(ev.kills) == 2     # both die on one hit


def test_shadow_dash_carries_the_front_twin_through_the_hero():
    # rec1 f0562 -> f0565: the front Twin on cell 3 (the hero on 2) loaded Shadow Dash and ended on cell 1
    s = _twins(3, 6, 2)
    s.enemies[0].facing, s.enemies[0].queue = -1, ["shadow_dash"]
    ev = Events()
    _enemy_fire(s, s.enemies[0], ev)
    assert s.enemies[0].x == 1 and s.hero.hp == 9


def test_the_dash_then_swap_toss_puts_the_hero_back_between_them():
    # rec2 f0770 -> f0772: the front Twin on cell 4, the hero on 5 next to it; the Dash cannot move it,
    # Swap Toss sends the hero to cell 3, behind it
    s = _twins(0, 4, 5, hero_facing=-1)
    s.enemies[1].facing, s.enemies[1].queue = 1, ["dash", "swap_toss"]
    _enemy_fire(s, s.enemies[1], Events())
    assert s.enemies[1].x == 4 and s.hero.x == 3 and s.hero.hp == 10


def test_what_the_twins_load():
    b = CAT.enemy("the_twins").behaviour
    s = _twins(3, 6, 2)                                   # both right of the hero: outside
    assert _pair_load(s, s.enemies[1], b) == [(1.0, ("queue", "spear"))]          # the rear one
    assert _pair_load(s, s.enemies[0], b) == [(1.0, ("queue", "shadow_dash"))]    # the front one, next to him
    s = _twins(0, 5, 6)                                   # the hero on the wall's cell (rec2 f0739)
    assert _pair_load(s, s.enemies[1], b) == [(1.0, ("queue", "dash"))]
    assert CAT.enemy("the_twins").combo_next(["dash"], 1) == "swap_toss"
    s = _twins(0, 6, 3)                                   # between them: melee
    assert {a[1] for _, a in _pair_load(s, s.enemies[0], b)} == {"spear", "tetsubo", "swirl"}
