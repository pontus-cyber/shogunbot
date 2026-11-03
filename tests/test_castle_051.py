"""0.51: the Shogun's Castle's unpredicted hits [claude/core-requests.md item 19]: the Shogun's Dragon Punch (read as Charge,
Back Charge or Shuriken until 0.50), his first bar's loaded tile held behind a unit, and the Warden (read as a Spike Charger
until 0.50), whose explosion on death hit the hero. Frame and run-log tests skip when the frames or logs are missing."""
import json

import pytest

from test_bot_flow import DATA, _stub_windows_modules, frame

from shogunbot import behaviour
from shogunbot.model import Enemy, Hero, State
from shogunbot.planner import plan_live
from shogunbot.rules import step

_stub_windows_modules()

ROUTE = ("bamboo_grove", "hibiku_wastelands", "moonlit_port", "spirit_gateway", "hot_springs", "hideyoshis_keep",
         "shoguns_castle")
DP = "dragon_punch"


def _p(rel):
    from shogunbot.perception.perceive import perceive
    return perceive(frame(rel), regions=ROUTE, day=1)


def _logged(run: str, turn: int) -> tuple[dict, State]:
    path = DATA / f"runs/{run}/turns.jsonl"
    if not path.exists():
        pytest.skip("run log not available")
    row = next(json.loads(l) for l in open(path) if json.loads(l)["turn"] == turn)
    return row, State.from_dict(row["state"])


def _top(s, action):
    return max(step(s, action), key=lambda t: t[0])


def _shogun(p):
    return next(e for e in p.enemies if e["kind"] == "the_shogun")


# ---------------------------------------------------------------- his Dragon Punch
def test_his_dragon_punch_reads_as_his():
    # a '1' and a white fist, his bubbles higher than the standard slot: 0.48-0.50 read it as Charge on his yellow frames
    # (run 20261103-052436 t820, t841) and as Back Charge or Shuriken on his red ones (run -085026 t863-t870, the frames
    # the templates were not built from)
    for rel in ("runs/20261103-052436/frames/t0820_before.png", "runs/20261103-052436/frames/t0841_before.png",
                "runs/20261103-085026/frames/t0866_before.png", "runs/20261103-085026/frames/t0870_before.png"):
        assert _shogun(_p(rel))["queue"] == [DP], rel
    # his Charge under a '+' still reads as Charge (052436 t813)
    assert _shogun(_p("runs/20261103-052436/frames/t0813_before.png"))["queue"] == ["charge"]


def test_his_dragon_punch_pushes_the_hero_into_the_summoner():
    # run 20261103-085026 t870 (0.49): the Shogun on cell 2 facing the hero on 3 punched him to cell 5, against the Summoner
    # on 6: 1 for the punch and 1 for the collision (HP 5 -> 3); read as Back Charge, it was planned as no damage
    row, s = _logged("20261103-085026", 870)
    sh = next(e for e in s.enemies if e.kind == "the_shogun")
    sh.queue = [DP]
    _, s2, ev = _top(s, "attack")
    assert (s2.hero.x, s2.hero.hp) == (5, 3)
    assert [d for _, _, d in ev.hero_damage] == [1, 1]


# ---------------------------------------------------------------- his first bar: a unit between
def _st(hx, ex, q, others=()):
    e = Enemy("the_shogun", ex, 1 if hx > ex else -1, 70, list(q), uid=1, last="queue", fage=2)
    return State(7, Hero(hx, -1 if hx > ex else 1, hp=10), [e] + list(others), level="stage7-boss")


def _rule(s):
    e = s.enemies[0]
    return behaviour.shogun_intent(s, e, behaviour.features(s, e, e.last, e.fage))


def test_a_unit_between_him_and_the_hero_holds_his_tile():
    # 43 of 43 phases of his first bar with a unit between him and the hero and his tile short of the hero, he held it
    # (runs 20261103-020829, -085026, -113809); 0.48-0.50 planned him walking up or firing into the unit
    ym = Enemy("yari_master", 2, 1, 4, [], uid=2, intent="hold")
    assert _rule(_st(5, 1, [DP], [ym])) == "hold"                        # 085026 t866
    assert _rule(_st(5, 1, [DP])) == "move-right"                        # nothing between: he steps toward the hero
    gr = Enemy("grappler", 3, -1, 5, [], uid=2, intent="hold")
    assert _rule(_st(4, 0, ["spear"], [gr])) == "hold"                    # 113809 t823: the cell next to him free
    # his Charge and Spear behind an Ashigaru: held while the hero is out of the Spear's reach, fired once he is in it
    # (020829 t866-t868)
    ash = Enemy("ashigaru", 3, 1, 3, [], uid=2, intent="hold")
    assert _rule(_st(5, 2, ["charge", "spear"], [ash])) == "hold"
    assert _rule(_st(4, 2, ["charge", "spear"], [ash])) == "fire"
    assert _rule(_st(3, 2, [DP])) == "fire"                              # next to him: the punch


# ---------------------------------------------------------------- the Warden
def test_the_warden_is_read_by_its_lantern():
    # the Castle's wave level: Wardens on cells 4 and 7 (run 20261103-085026 t839), read as Spike Chargers until 0.50;
    # its Blight Charger stays a Charger (t845, cell 1), and the Spike Chargers of run -113809 t729 too
    p = _p("runs/20261103-085026/frames/t0839_before.png")
    assert [(e["x"], e["kind"]) for e in p.enemies if e["hp_max"] == 1] == [(4, "warden"), (7, "warden")]
    p = _p("runs/20261103-085026/frames/t0845_before.png")
    assert [(e["x"], e["kind"]) for e in p.enemies if e["hp_max"] == 1] == [(1, "charger"), (3, "warden"), (5, "charger")]
    p = _p("runs/20261103-113809/frames/t0729_before.png")
    assert [(e["x"], e["kind"]) for e in p.enemies if e["hp_max"] == 1] == [(6, "charger"), (8, "charger")]


def _wardens(s, cells):
    for e in s.enemies:
        if e.kind == "charger" and e.x in cells:
            e.kind, e.queue = "warden", []
    return s


def test_a_warden_explodes_on_the_units_beside_it():
    # run 20261103-085026 t841: the hero on cell 6 killed the Warden on 7 with his Katana: 2 to him and 2 to the Yari Master
    # on 8, which his Charge then killed (HP 8 -> 6); t847: his Shadow Dash put him on cell 4 and his Spear killed the
    # Warden on 3: 2 to him and 2 to the Yari Master on 2 (HP 6 -> 4). Both planned as no damage until 0.50
    row, s = _logged("20261103-085026", 841)
    _, s2, ev = _top(_wardens(s, (4, 7)), "attack")
    assert (s2.hero.x, s2.hero.hp) == (7, 6) and s2.enemy_at(8) is None
    row, s = _logged("20261103-085026", 847)
    _, s2, ev = _top(_wardens(s, (3,)), "attack")
    assert (s2.hero.x, s2.hero.hp) == (4, 4)
    assert ("yari_master", 3, "warden:explosion") in ev.kills


def test_the_planner_kills_no_warden_from_next_to_it():
    # the same two boards: a move that costs nothing is planned instead
    for turn, cells in ((841, (4, 7)), (847, (3,))):
        _, s = _logged("20261103-085026", turn)
        d = plan_live(_wardens(s, cells))
        assert d.action != "attack"
        assert next(c for c in d.candidates if c["action"] == d.action)["exp_damage"] == 0


def test_an_explosion_kills_a_warden_beside_it_and_that_one_explodes_too():
    # [wiki: 'explodes dealing 2 damage to any unit in adjacent cells']: a chain along the row
    w1 = Enemy("warden", 2, 1, 1, [], uid=1, intent="hold")
    w2 = Enemy("warden", 3, 1, 1, [], uid=2, intent="hold")
    t = Enemy("tachi", 4, -1, 2, [], uid=3, intent="hold")
    s = State(7, Hero(1, 1, hp=10), [w1, w2, t])
    from shogunbot.rules import _damage_enemy, Events
    ev = Events()
    _damage_enemy(s, w1, 1, "test", ev)
    assert s.hero.hp == 8 and w2.hp <= 0 and t.hp <= 0
