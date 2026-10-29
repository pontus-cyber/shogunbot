"""0.23: elite enemies (Day 2) and enemies whose kind the read cannot settle [user review of
run 20261029-045856, the first Day 2 run: an elite Tachi (3 HP pips, Swirl loaded) was read
as an Ashigaru; the bot swapped with it for the swap trick and its Swirl hit the hero, -2 at
turns 16 and 21]. Frame-based tests skip when the frames are missing."""

import pytest
from test_bot_flow import frame

from shogunbot.bridge import Tracker
from shogunbot.model import Enemy, Hero, Kit, State
from shogunbot.perception.perceive import decide_kind, perceive
from shogunbot.planner import WEIGHTS, plan_live
from shogunbot.rules import step

DAY2 = "runs/20261029-045856/frames/"


def _reads(rel):
    p = perceive(frame(rel))
    return {e["x"]: (e["kind"], e["hp"], e["hp_max"], e["queue"], e["elite"], e["alt"]) for e in p.enemies}


def test_the_elite_tachi_of_turn_16_reads_as_a_tachi_with_its_swirl():
    r = _reads(DAY2 + "t0016_before.png")
    assert r[1] == ("tachi", 3, 3, ["swirl"], True, None)      # was ashigaru, Swirl rewritten to Sword
    assert r[0] == ("tachi", 2, 2, [], False, None)            # a plain Tachi
    assert r[4] == ("ashigaru", 3, 3, [], False, None)


def test_the_elite_tachi_of_turn_21_reads_as_a_tachi_with_its_swirl():
    r = _reads(DAY2 + "t0021_before.png")
    assert r[2] == ("tachi", 1, 3, ["swirl"], True, None)
    assert r[4] == ("ashigaru", 3, 3, ["sword"], False, None)


def test_an_unloaded_elite_tachi_and_a_double_striker_ashigaru():
    assert _reads(DAY2 + "t0016_after.png")[2][:2] == ("tachi", 3)          # emblem + sprite, no tile
    assert _reads(DAY2 + "f0012_CLICK.png")[4][:5] == ("ashigaru", 5, 5, [], True)   # 5 pips: +2


def test_day1_reads_unchanged_where_the_sprite_disagrees():
    # an Ashigaru in an attack pose reads as a Tachi by its sprite (conf 1.0); its Sword decides
    assert _reads("runs/20261027-180145/frames/f0061_SPACE.png")[3][0] == "ashigaru"


def test_decide_kind():
    # (pips, the kind of the tile in the bubble, the sprite's kind, emblem) -> (kind, alt)
    assert decide_kind(3, None, None, False) == ("ashigaru", None)          # Day 1: pips decide
    assert decide_kind(3, "tachi", None, False) == ("tachi", None)          # a Swirl fits an elite Tachi
    assert decide_kind(3, "ashigaru", "tachi", False) == ("ashigaru", None) # a Sword settles a sprite misread
    assert decide_kind(3, None, "tachi", True) == ("tachi", None)           # emblem: the sprite names it
    assert decide_kind(3, None, "tachi", False) == ("ashigaru", "tachi")    # unsettled: alt kept
    assert decide_kind(1, None, "tachi", False) == ("charger", None)        # 1 pip: only a Charger
    assert decide_kind(5, None, None, False) == ("ashigaru", None)          # no base kind has 5: elite
    assert decide_kind(4, None, None, True)[1] is not None                  # Ashigaru +1 or Tachi +2
    assert decide_kind(0, None, None, False) == (None, None)


def _swap_board(alt):
    """Hero at 2 facing left; at 1 a loaded unit facing him, telegraphing fire (yellow)."""
    hero = Hero(2, -1, hp=10, kit=Kit.of("swirl", "arrow"))
    e = Enemy("ashigaru", 1, 1, 3, ["sword"], uid=1, intent="fire", alt=alt)
    return State(5, hero, [e], level="bamboo-1", wave=1, waves_total=4)


def test_a_swap_with_a_primed_unit_that_may_be_a_tachi_is_unsafe():
    # swapping puts the hero behind it: a Sword fires away from him, a Swirl hits him too
    sure = sum(p * (10 - s2.hero.hp) for p, s2, _ in step(_swap_board(None), "left"))
    unsure = sum(p * (10 - s2.hero.hp) for p, s2, _ in step(_swap_board("tachi"), "left"))
    assert sure == 0 and unsure == 2
    assert plan_live(_swap_board("tachi"), WEIGHTS).action != "left"


def test_the_bridge_carries_alt_and_elite():
    from test_facing import _p
    tr = Tracker()
    p = _p([{"x": 1, "kind": "ashigaru", "facing": 1, "queue": ["sword"], "alt": "tachi", "elite": False},
            {"x": 4, "kind": "tachi", "facing": -1, "hp": 3, "hp_max": 3, "elite": True}], hero_x=2)
    s = tr.state(p)
    assert (s.enemies[0].alt, s.enemies[1].elite, s.enemies[1].hp) == ("tachi", True, 3)
    assert any("may be tachi" in n for n in tr.notes)


def test_replanning_turns_16_and_21_avoids_the_swaps():
    # the logged states of run 20261029-045856 with the unit read as it is now
    import json
    from test_bot_flow import DATA
    path = DATA / "runs/20261029-045856/turns.jsonl"
    if not path.exists():
        pytest.skip("run logs not available")
    recs = {json.loads(l)["turn"]: json.loads(l) for l in open(path, encoding="utf-8")}
    for turn, x, swap in ((16, 1, "left"), (21, 2, "right")):
        s = State.from_dict(recs[turn]["state"])
        e = s.enemy_at(x)
        e.kind, e.queue, e.elite = "tachi", ["swirl"], True
        d = plan_live(s, WEIGHTS)
        assert d.action != swap
        assert next(c for c in d.candidates if c["action"] == d.action)["exp_damage"] == 0
