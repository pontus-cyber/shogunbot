"""Catalogue footprints and enchantments in the simulator.

Footprints follow the wiki Tiles page text [wiki]; where it is silent the
rule is a hypothesis [hyp] (see notes in data/tiles.json) and the first live
use will confirm or correct it.
"""

from shogunbot.model import Enemy, Hero, Kit, KitTile, State
from shogunbot.rules import Events, apply_hero, legal_hero_actions, step


def board(cells, hero_x, hero_f, kit_tiles, enemies, queue=()):
    kit = Kit.of(*kit_tiles)
    h = Hero(hero_x, hero_f, kit=kit, queue=list(queue))
    es = [Enemy(k, x, f, hp, [], uid=i + 1) for i, (k, x, f, hp) in enumerate(enemies)]
    return State(cells, h, es)


def fire(s):
    ev = Events()
    return apply_hero(s, "attack", ev), ev


def hp_by_x(s):
    return {e.x: e.hp for e in s.alive()}


def test_katana_hits_the_cell_ahead_only():
    s = board(5, 2, 1, ["katana"], [("ashigaru", 3, -1, 3), ("ashigaru", 1, 1, 3)], ["katana"])
    s2, _ = fire(s)
    assert hp_by_x(s2) == {3: 1, 1: 3}


def test_katana_has_no_cooldown():
    s = board(5, 2, 1, ["katana"], [("ashigaru", 4, -1, 3)], ["katana"])
    s2, _ = fire(s)
    assert "queue:katana" in legal_hero_actions(s2)


def test_spear_pierces_two_cells():
    s = board(5, 0, 1, ["spear"], [("tachi", 1, -1, 2), ("ashigaru", 2, -1, 3), ("charger", 3, -1, 1)], ["spear"])
    s2, ev = fire(s)
    assert hp_by_x(s2) == {2: 1, 3: 1} and len(ev.kills) == 1


def test_shuriken_hits_the_first_target_for_one():
    s = board(7, 0, 1, ["shuriken"], [("ashigaru", 4, -1, 3), ("charger", 6, -1, 1)], ["shuriken"])
    s2, _ = fire(s)
    assert hp_by_x(s2) == {4: 2, 6: 1}


def test_charge_dashes_next_to_the_first_target_and_strikes():
    s = board(7, 0, 1, ["charge"], [("ashigaru", 4, -1, 3)], ["charge"])
    s2, _ = fire(s)
    assert s2.hero.x == 3 and s2.hero.facing == 1 and hp_by_x(s2) == {4: 2}


def test_charge_without_target_dashes_to_the_edge():
    # [hyp] data/tiles.json charge.no_target = edge
    s = board(7, 2, 1, ["charge"], [("ashigaru", 0, 1, 3)], ["charge"])
    s2, _ = fire(s)
    assert s2.hero.x == 6


def test_back_charge_dashes_backward_keeping_facing():
    s = board(7, 5, 1, ["back_charge"], [("tachi", 1, 1, 2)], ["back_charge"])
    s2, _ = fire(s)
    assert s2.hero.x == 2 and s2.hero.facing == 1 and hp_by_x(s2) == {1: 1}


def test_shadow_dash_passes_a_group_and_hits_each():
    # [hyp] lands on the first empty cell behind the group, strikes every unit passed
    s = board(7, 1, 1, ["shadow_dash"], [("ashigaru", 2, -1, 3), ("tachi", 3, -1, 2)], ["shadow_dash"])
    s2, _ = fire(s)
    assert s2.hero.x == 4 and hp_by_x(s2) == {2: 2, 3: 1}


def test_shadow_dash_blocked_at_the_edge_does_nothing():
    s = board(5, 2, 1, ["shadow_dash"], [("ashigaru", 3, -1, 3), ("tachi", 4, -1, 2)], ["shadow_dash"])
    s2, _ = fire(s)
    assert s2.hero.x == 2 and hp_by_x(s2) == {3: 3, 4: 2}


def test_queue_fires_from_where_the_dash_left_the_hero():
    # Charge to the Ashigaru (cell 4), then Swirl fires from cell 3: it finishes the
    # Ashigaru and misses the Tachi at 0 that it would have hit from the start cell
    s = board(7, 1, 1, ["charge", "swirl"], [("tachi", 0, 1, 2), ("ashigaru", 4, -1, 3)], ["charge", "swirl"])
    s2, _ = fire(s)
    assert s2.hero.x == 3 and hp_by_x(s2) == {0: 2}


def test_earth_impale_lightning_nagiboku_scar_strike():
    s = board(7, 3, 1, ["earth_impale"], [("ashigaru", 1, 1, 3), ("ashigaru", 4, -1, 3), ("ashigaru", 5, -1, 3)],
              ["earth_impale"])
    assert hp_by_x(fire(s)[0]) == {1: 1, 4: 3, 5: 1}
    s = board(7, 0, 1, ["lightning"], [("ashigaru", 2, -1, 3), ("ashigaru", 5, -1, 3)], ["lightning"])
    assert hp_by_x(fire(s)[0]) == {2: 3, 5: 1}
    s = board(5, 2, 1, ["nagiboku"], [("charger", 1, 1, 1), ("tachi", 3, -1, 2)], ["nagiboku"])
    assert hp_by_x(fire(s)[0]) == {1: 1, 3: 1}       # cannot kill
    s = board(7, 0, 1, ["scar_strike"], [("ashigaru", 2, -1, 2), ("ashigaru", 5, -1, 3)], ["scar_strike"])
    assert hp_by_x(fire(s)[0]) == {2: 1, 5: 3}


def test_second_copy_gets_its_own_label_and_cooldown():
    kit = Kit.of("swirl", "arrow", "swirl")
    h = Hero(2, 1, kit=kit, charge={"swirl": 0, "arrow": 5, "swirl#2": 3})
    s = State(5, h, [Enemy("ashigaru", 4, -1, 3, uid=1)])
    acts = legal_hero_actions(s)
    assert "queue:swirl#2" in acts and "queue:swirl" not in acts


def enchanted(tile, ench, **kw):
    return Kit([KitTile(tile, tile, kw.get("damage", 2), kw.get("cooldown", 3), ench)])


def test_double_strike_hits_twice():
    s = State(5, Hero(2, 1, kit=enchanted("katana", "double_strike"), queue=["katana"]),
              [Enemy("ashigaru", 3, -1, 3, uid=1)])
    s2, _ = fire(s)
    assert not s2.alive()


def test_poison_ticks_at_the_end_of_the_turn():
    s = State(7, Hero(0, 1, kit=enchanted("arrow", "poison"), queue=["arrow"]),
              [Enemy("ashigaru", 5, -1, 5, uid=1)])
    s.enemies[0].intent = "hold"
    (_, s2, ev), = step(s, "attack")
    assert s2.enemies[0].hp == 5 - 2 - 1 and s2.enemies[0].poison == 2
    assert ("ashigaru", 1, 1, "poison") in ev.enemy_damage


def test_ice_freezes_the_target():
    s = State(5, Hero(2, 1, kit=enchanted("katana", "ice"), queue=["katana"]),
              [Enemy("ashigaru", 3, -1, 3, uid=1)])
    s2, _ = fire(s)
    assert s2.enemies[0].frozen == 3


def test_curse_doubles_the_following_hit():
    kit = Kit([KitTile("katana", "katana", 1, 0, "curse"), KitTile("arrow", "arrow", 1, 5)])
    s = State(5, Hero(2, 1, kit=kit, queue=["katana", "arrow"]), [Enemy("ashigaru", 3, -1, 3, uid=1)])
    s2, _ = fire(s)
    assert not s2.alive()   # 1, then 1 doubled to 2


def test_shockwave_hits_the_targets_neighbours():
    s = State(7, Hero(1, 1, kit=enchanted("katana", "shockwave"), queue=["katana"]),
              [Enemy("ashigaru", 2, -1, 3, uid=1), Enemy("tachi", 3, -1, 2, uid=2)])
    s2, _ = fire(s)
    assert hp_by_x(s2) == {2: 1, 3: 1}


def test_perfect_strike_recharges_on_an_exact_kill():
    s = State(5, Hero(2, 1, kit=enchanted("arrow", "perfect_strike", cooldown=5), queue=["arrow"]),
              [Enemy("tachi", 4, -1, 2, uid=1)])
    s2, _ = fire(s)
    assert not s2.alive() and s2.hero.charge["arrow"] == 5


def test_freeplay_queue_does_not_pass_the_turn():
    s = State(5, Hero(2, 1, kit=enchanted("swirl", "freeplay")), [Enemy("ashigaru", 3, -1, 3, ["sword"], uid=1)])
    s.enemies[0].intent = "fire"
    (_, s2, ev), = step(s, "queue:swirl")
    assert s2.hero.hp == 10 and s2.hero.queue == ["swirl"] and not ev.enemy_actions
