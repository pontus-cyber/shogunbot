"""0.27: Rei the Ruthless, the alternate Bamboo Grove boss [third series: 4 of 5 bosses were Rei and
every HP lost was hers]. Her Dragon Punch pushes the target as far back as possible, with collision
damage when another unit is in the path [wiki: Tiles]; the turns below are where the runs showed
it. Frame tests skip when the frames are missing."""

from test_bot_flow import frame

from shogunbot.bridge import Tracker
from shogunbot.model import Enemy, Hero, Kit, State
from shogunbot.perception.perceive import perceive, read_stack
from shogunbot.rules import enemy_options, step

R100218 = "runs/20261029-100218/frames/"


def _board(units, hero_x, hp=10, facing=-1, shield=False, cells=7):
    hero = Hero(hero_x, facing, hp=hp, kit=Kit.of("swirl", "arrow"), shield=shield)
    return State(cells, hero, units, level="boss", wave=0, waves_total=0)


def _rei(x, facing, queue, intent="fire", hp=13):
    return Enemy("rei", x, facing, hp, list(queue), uid=1, intent=intent)


def _one(s, action="wait"):
    outs = step(s, action)
    assert len(outs) == 1, [o[1].render() for o in outs]
    return outs[0][1]


# ---- the rules, each from a turn of the third series ------------------------------------------
def test_dragon_punch_slides_the_hero_into_a_tachi():
    # run 20261029-100218 t79: the Tachi steps 1 -> 2, her punch slides the hero 5 -> 3 into it:
    # 1 for the punch and 1 for the collision, and the Tachi takes 1 [observed: HP 7 -> 5, Tachi 2 -> 1]
    t = Enemy("tachi", 1, 1, 2, ["swirl"], uid=2, intent="move-right")
    s2 = _one(_board([t, _rei(6, -1, ["dragon_punch"])], hero_x=5, hp=7))
    assert (s2.hero.x, s2.hero.hp) == (3, 5)
    assert next(e for e in s2.enemies if e.kind == "tachi").hp == 1


def test_a_unit_next_to_the_hero_stops_the_slide_at_once():
    # run 20261029-100218 t143: the Tachi steps 3 -> 4 next to the hero; the punch cannot move him
    t = Enemy("tachi", 3, 1, 2, ["swirl"], uid=2, intent="move-right")
    s2 = _one(_board([t, _rei(6, -1, ["dragon_punch"], hp=2)], hero_x=5, hp=5))
    assert (s2.hero.x, s2.hero.hp) == (5, 3)
    assert next(e for e in s2.enemies if e.kind == "tachi").hp == 1


def test_charge_then_punch():
    # run 20261029-100218 t68: Charge 6 -> 4 strikes (1), the punch slides the hero 3 -> 1 into the
    # Tachi on 0 (1 + 1) [observed: HP 10 -> 7, Rei on 4, the Tachi 2 -> 1]
    t = Enemy("tachi", 0, 1, 2, [], uid=2, intent="queue")
    s2 = _one(_board([t, _rei(6, -1, ["charge", "dragon_punch"])], hero_x=3))
    rei = next(e for e in s2.enemies if e.kind == "rei")
    assert (s2.hero.x, s2.hero.hp, rei.x) == (1, 7, 4)
    assert next(e for e in s2.enemies if e.kind == "tachi").hp == 1


def test_the_wall_stops_the_slide_without_damage():
    # run 20261029-101549 t91: Rei on 0 facing right, Charge + punch, the hero on 4 slides to the
    # last cell [observed: HP 10 -> 8, the hero on 6, Rei on 3]
    s2 = _one(_board([_rei(0, 1, ["charge", "dragon_punch"])], hero_x=4))
    assert (s2.hero.x, s2.hero.hp, s2.enemies[0].x) == (6, 8, 3)


def test_a_shield_takes_the_punch_and_the_collision_not_the_slide():
    # run 20261029-072323 t69: Kami Brew up; the hero slid 5 -> 2 into the Tachi on 1, HP unchanged,
    # the Tachi 2 -> 1
    ts = [Enemy("tachi", 0, 1, 2, ["swirl"], uid=2, intent="hold"), Enemy("tachi", 1, 1, 2, ["swirl"], uid=3, intent="hold")]
    s2 = _one(_board(ts + [_rei(6, -1, ["dragon_punch"])], hero_x=5, shield=True))
    assert (s2.hero.x, s2.hero.hp, s2.hero.shield) == (2, 10, False)
    assert {e.x: e.hp for e in s2.enemies if e.kind == "tachi"} == {0: 2, 1: 1}


def test_she_punches_her_tachi_into_the_hero():
    # run 20261029-103242 t91: the Tachi between them takes the punch and the collision and dies;
    # the hero takes the collision [observed: HP 8 -> 7, the Tachi gone]
    t = Enemy("tachi", 3, 1, 2, [], uid=2, intent="hold")
    s2 = _one(_board([t, _rei(4, -1, ["dragon_punch"], hp=3)], hero_x=2, hp=8, facing=1))
    assert s2.hero.hp == 7 and not any(e.kind == "tachi" for e in s2.enemies)


def test_a_boss_charge_with_nothing_ahead_ends_at_the_wall():
    # run 20261028-061420 t84: the hero swaps past Daisuke; his Charge + Sword fires away from the
    # hero and he ends on cell 0 [25 of 25 logged swaps past a charging boss]
    d = Enemy("daisuke", 3, -1, 11, ["charge", "sword"], uid=1, intent="fire")
    s = State(5, Hero(2, 1, hp=10, kit=Kit.of("swirl", "arrow")), [d], level="boss", wave=0, waves_total=0)
    s2 = _one(s, "right")
    assert (s2.hero.x, s2.hero.hp, s2.enemies[0].x) == (3, 10, 0)


def test_the_red_bell_brings_two_twin_tachi():
    # every Rei fight opened with it: two Twin Tachi [runs 20261029-072323 t65 ... -103242 t65]
    s2 = _one(_board([_rei(6, -1, ["summon_double"])], hero_x=4))
    assert sorted((e.kind, e.x) for e in s2.enemies if e.kind != "rei") == [("tachi", 0), ("tachi", 1)]


def test_charge_alone_is_followed_by_the_punch():
    # Charge was followed by Dragon Punch 10 of 10 times, a phase apart [Rei's five fights]
    r = _rei(6, -1, ["charge"], intent="queue")
    opts = enemy_options(_board([r], hero_x=2), r)
    assert opts == [(1.0, ("queue", "dragon_punch"))]


# ---- perception and the tracker ---------------------------------------------------------------
def test_her_bubbles_read_as_her_tiles():
    reads = {}
    for t in (61, 67, 71, 73, 88, 118):
        p = perceive(frame(R100218 + f"t{t:04d}_before.png"))
        boss = next(e for e in p.enemies if e["kind"] == "rei")
        reads[t] = [s["tile"] for s in read_stack(frame(R100218 + f"t{t:04d}_before.png"), boss["cx"])]
    assert reads == {61: ["summon_double"], 67: ["charge", "dragon_punch"], 71: ["back_charge"],
                     73: ["back_charge", "summon"], 88: ["dragon_punch"], 118: ["summon"]}


def test_the_seven_cell_arena_is_rei_and_the_tracker_uses_her_hp():
    p = perceive(frame(R100218 + "t0068_before.png"))
    assert [(e["kind"], e["x"]) for e in p.enemies] == [("tachi", 0), ("rei", 6)]
    s = Tracker().state(p)
    rei = next(e for e in s.enemies if e.kind == "rei")
    assert rei.queue == ["charge", "dragon_punch"]
    assert rei.hp == max(1, round(p.boss_frac * 13))


def test_daisukes_five_cell_arena_is_still_daisuke():
    # run 20261029-102441, the series' one Daisuke fight
    p = perceive(frame("runs/20261029-102441/frames/f0077_CLICK.png"))
    assert (p.screen, len(p.cells), [(e["kind"], e["x"]) for e in p.enemies]) == ("boss", 5, [("daisuke", 3)])
