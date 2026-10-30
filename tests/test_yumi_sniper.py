"""0.35: the Yumi Sniper and its Volley [user review of run 20261030-115221, unpredicted hit at Moonlit Port I t174].

At t174 the bot attacked from cell 2 and took 2 it had not predicted: the 3-pip unit on cell 0 was a Yumi Sniper,
which 0.34 read as an Ashigaru (its Volley icon read as Spear and the tracker set it to a Sword), so its Volley
counted as a Sword two cells away. The Volley is aimed at the hero's cell as the turn it fires begins (a yellow
warning sign at his feet); the hero acts first, then the arrows strike that cell and whatever unit stands there
[data/tiles.json 'volley': rec: collect/20261029-160032; bot: run 20261030-115221]. The Sniper came to Moonlit
Port although the route went through Whispering Caves: regions.json lists it as a visitor there.
Frame tests skip when the frames are missing."""

import json

import numpy as np
import pytest

from test_bot_flow import DATA, frame

from shogunbot.bridge import Tracker
from shogunbot.model import Enemy, Hero, State
from shogunbot.perception.perceive import decide_kind, perceive, roster_tables, roster_visitors
from shogunbot.planner import plan_live
from shogunbot.rules import Events, _enemy_fire, step

ROUTE = ("bamboo_grove", "whispering_caves", "moonlit_port")
RUN = "runs/20261030-115221/"


def _p(rel):
    return perceive(frame(rel), regions=ROUTE, day=1)


# ---------------------------------------------------------------- reading

def test_the_sniper_is_a_visitor_of_moonlit_port_named_only_by_its_volley():
    assert roster_visitors(ROUTE) == frozenset({"yumi_sniper"})
    assert roster_visitors(("bamboo_grove", "hibiku_wastelands")) == frozenset()     # at home it is a native
    base, by_tile = roster_tables(ROUTE)
    assert base["yumi_sniper"] == 3 and by_tile["volley"] == "yumi_sniper"
    v = roster_visitors(ROUTE)
    assert decide_kind(3, None, None, False, base, elites=False, visitors=v) == ("ashigaru", None)
    assert decide_kind(3, "yumi_sniper", None, False, base, elites=False, visitors=v) == ("yumi_sniper", None)
    assert decide_kind(3, "ashigaru", None, False, base, elites=False, visitors=v) == ("ashigaru", None)


def test_the_volley_icon_reads_on_the_held_out_frames_of_run_115221():
    # the icon's templates come from recording 2 only; every Volley bubble of run 115221 read as Spear in 0.34
    for name, x in [("t0172_after", 0), ("t0173_before", 0), ("t0177_before", 0), ("t0190_before", 6),
                    ("t0209_after", 1), ("stuck", 0)]:
        p = _p(RUN + f"frames/{name}.png")
        e = next(e for e in p.enemies if e["x"] == x)
        assert (e["kind"], e["queue"]) == ("yumi_sniper", ["volley"]), name


def test_the_warning_sign_marks_the_aimed_cell():
    p = _p(RUN + "frames/t0174_before.png")
    assert p.warn == [2] and p.hero["x"] == 2
    sniper = next(e for e in p.enemies if e["x"] == 0)
    assert sniper["kind"] == "yumi_sniper" and sniper["intent"] == "fire"
    assert _p(RUN + "frames/f0223_SPACE.png").warn == [1]           # t178: the hero had stepped to cell 1
    assert _p(RUN + "frames/t0173_before.png").warn == []           # red: no sign yet
    # recording 2, Hibiku Wastelands: the sign under the hero while the bubble's blink showed red
    p = perceive(frame("collect/20261029-160032/f0199_A.png"), regions=("bamboo_grove", "hibiku_wastelands"), day=1)
    assert p.warn == [p.hero["x"]]


def test_the_tracker_aims_the_volley_and_keeps_the_sniper_when_its_bubble_is_empty():
    tr = Tracker()
    tr.route = list(ROUTE)
    s = tr.state(_p(RUN + "frames/t0174_before.png"))
    u = next(e for e in s.enemies if e.x == 0)
    assert (u.kind, u.queue, u.intent, u.aim) == ("yumi_sniper", ["volley"], "fire", 2)
    tr.prev = [(e.kind, e.x, e.facing, e.intent) for e in s.enemies]
    s = tr.state(_p(RUN + "frames/t0175_before.png"))               # fired: no bubble, 3 pips
    u = next(e for e in s.enemies if e.x == 0)
    assert u.kind == "yumi_sniper" and not u.queue


# ---------------------------------------------------------------- the rules

def _board(hx, sniper_x, aim=None, intent="fire", others=()):
    sn = Enemy("yumi_sniper", sniper_x, 1 if hx > sniper_x else -1, 3, ["volley"], uid=1, intent=intent, aim=aim)
    return State(7, Hero(hx, 1), [sn, *others], level="stage3-1", waves_total=5, wave=3)


def _outcomes(s, action):
    return [(p, s2.hero.hp, {e.uid: e.hp for e in s2.enemies}) for p, s2, _ in step(s, action)]


def test_the_volley_strikes_the_cell_aimed_at_as_the_turn_began():
    # stay: 2 [bot: run 20261030-115221 t174]; step off: nothing [rec: collect/20261029-160032 f0199-f0200]
    assert {hp for _, hp, _ in _outcomes(_board(3, 0), "wait")} == {8}
    assert {hp for _, hp, _ in _outcomes(_board(3, 0), "right")} == {10}
    assert {hp for _, hp, _ in _outcomes(_board(3, 0, aim=3), "left")} == {10}
    # red (loaded this turn or the last): it holds, and nothing is aimed yet
    assert {hp for _, hp, _ in _outcomes(_board(3, 0, intent="hold"), "wait")} == {10}


def test_the_volley_hits_whatever_stands_on_the_cell():
    # rec2 f0222 -> f0223: the hero on cell 5 swapped with the Sniper on 6; its Volley hit itself for 2
    s = _board(5, 6, aim=5)
    s.hero.facing = 1
    for _, hp, ehp in _outcomes(s, "right"):
        assert hp == 10 and ehp[1] == 1
    # an enemy the volley lands on takes it too
    s = _board(3, 0, aim=4, others=[Enemy("guardian", 4, -1, 5, uid=2)])
    _enemy_fire(s, s.enemies[0], Events())
    assert s.enemies[1].hp == 3 and s.hero.hp == 10 and s.enemies[0].aim is None


def test_a_sniper_killed_by_the_heros_action_does_not_fire():
    # t178: the Swirl (4 there) killed the Sniper on cell 0 before its volley at cell 1 [bot: run 20261030-115221];
    # here the default kit's Swirl (2) on a Sniper with 2 HP left
    s = _board(1, 0, aim=1)
    s.enemies[0].hp = 2
    s.hero.queue = ["swirl"]
    for _, hp, ehp in _outcomes(s, "attack"):
        assert hp == 10 and 1 not in ehp


def test_a_unit_that_may_be_a_sniper_aims_its_volley_too():
    # on a route through Hibiku Wastelands the Sniper is a native: a 3-pip unit without a tile is planned as an
    # Ashigaru that may be a Sniper (decide_kind's alt). Its possible Volley is aimed as the turn begins like a
    # Sniper's, so stepping off dodges it; 0.35's first replay counted it as a hit wherever the hero went
    # (recording 2 f0909)
    sus = Enemy("ashigaru", 0, 1, 3, ["sword"], uid=1, intent="fire", alt="yumi_sniper")
    s = State(7, Hero(3, 1), [sus], level="stage3-1", waves_total=5, wave=3)
    assert {s2.hero.hp for _, s2, _ in step(s, "right")} == {10}
    assert {s2.hero.hp for _, s2, _ in step(s, "wait")} == {8}


def test_kinds_the_pips_cannot_tell_apart():
    from shogunbot.perception.perceive import pip_ties
    assert pip_ties(ROUTE) == {"ashigaru": {"yumi_sniper"}, "yumi_sniper": {"ashigaru"},
                               "tachi": {"ashigaru_archer"}, "ashigaru_archer": {"tachi"}}
    assert pip_ties(("bamboo_grove", "whispering_caves")) == {}


def test_a_loaded_sniper_fires_the_turn_after_its_red_turn():
    # R1/R2 on the Volley: loaded at t172 ('+'), red at t173, fires at t174 at wherever the hero stood then
    s = _board(3, 0, intent="hold")
    s.enemies[0].wind = 0                    # just loaded
    hp_next = set()
    for _, s2, _ in step(s, "wait"):
        for _, s3, _ in step(s2, "wait"):
            hp_next.add(s3.hero.hp)
    assert hp_next == {8}


def test_run_115221_t174_the_planner_now_swaps_instead_of_attacking():
    # the logged board of t174 (the Sniper as the frames show it): attacking took the volley (-2); the swap with
    # the yellow Yari Master takes nothing (his Spear then points away, the volley lands on him)
    path = DATA / RUN / "turns.jsonl"
    if not path.exists():
        pytest.skip("run log not available")
    r = next(json.loads(l) for l in path.open(encoding="utf-8") if json.loads(l)["turn"] == 174)
    s = State.from_dict(r["state"])
    sn = next(e for e in s.enemies if e.x == 0)
    sn.kind, sn.queue, sn.aim = "yumi_sniper", ["volley"], 2
    d = plan_live(s)
    by = {c["action"]: c["exp_damage"] for c in d.candidates}
    assert by["attack"] == pytest.approx(2.0)
    assert d.action == "right" and by["right"] == 0.0


def test_mine_relabels_the_snipers_of_run_115221():
    from shogunbot import mine
    if not (DATA / RUN / "turns.jsonl").exists():
        pytest.skip("run log not available")
    run = next(r for r in mine.load_runs(DATA) if r.id == "20261030-115221")
    kinds = {(t["turn"], e["x"]): (e["kind"], e["queue"]) for t in run.turns for e in t["state"]["enemies"]}
    assert kinds[(174, 0)] == ("yumi_sniper", ["volley"])
    assert kinds[(209, 2)][0] == "yumi_sniper" and kinds[(210, 1)][0] == "yumi_sniper"
    assert kinds[(170, 1)][0] == "ashigaru"                         # wave 2's unit was an Ashigaru
