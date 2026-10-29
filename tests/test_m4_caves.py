"""0.26: Whispering Caves (M4 e1). The bot passes perception the regions of its route and the
day: a region's waves also bring the earlier regions' enemies, and Day 1 has no elites, so the
HP pips name the kind. With the Stage 1 roster the Yari Masters of Caves I and II read as elite
Ashigaru [collect/20261028-190026 f0186-f0274]. Frame tests skip when the frames are missing."""

from test_bot_flow import frame

from shogunbot.bridge import Tracker
from shogunbot.model import ENEMY_KINDS, Enemy, Hero, Kit, State
from shogunbot.perception.perceive import decide_kind, perceive, roster_tables
from shogunbot.rules import step

REC = "collect/20261028-190026/"
CAVES = ("bamboo_grove", "whispering_caves")


def _reads(rel, regions=CAVES, day=1):
    p = perceive(frame(rel), regions=regions, day=day)
    return {e["x"]: (e["kind"], e["hp"], e["hp_max"], e["queue"], e["alt"]) for e in p.enemies}


def test_the_caves_roster_adds_the_yari_master_to_stage_1():
    base, by_tile = roster_tables(CAVES)
    assert base == {"ashigaru": 3, "tachi": 2, "charger": 1, "yari_master": 4}
    assert by_tile["spear"] == "yari_master" and by_tile["sword"] == "ashigaru"
    assert roster_tables(("bamboo_grove",))[0] == {"ashigaru": 3, "tachi": 2, "charger": 1}


def test_decide_kind_on_day_1():
    base, _ = roster_tables(CAVES)
    assert decide_kind(4, None, None, False, base, elites=False) == ("yari_master", None)
    # the spear read as a Sword (18 of 30 Caves reads): no Ashigaru has 4 pips on Day 1
    assert decide_kind(4, "ashigaru", None, False, base, elites=False) == ("yari_master", None)
    # an Ashigaru in an attack pose reads as a Tachi by its sprite; on Day 1 3 pips settle it
    assert decide_kind(3, None, "tachi", False, base, elites=False) == ("ashigaru", None)
    assert decide_kind(5, None, None, False, base, elites=False) == (None, None)   # no kind has 5


def test_kinds_that_share_pips_are_told_apart_by_the_tile():
    base, by_tile = roster_tables(("bamboo_grove", "hibiku_wastelands"))
    assert decide_kind(3, by_tile["volley"], None, False, base, elites=False) == ("yumi_sniper", None)
    assert decide_kind(3, None, None, False, base, elites=False)[0] == "ashigaru"   # earlier region


def test_caves_i_yari_masters_read_as_themselves():
    r = _reads(REC + "f0204_CLICK.png")
    assert r[2] == ("yari_master", 4, 4, [], None)
    assert r[3] == ("ashigaru", 3, 3, [], None)
    assert r[4] == ("yari_master", 2, 4, ["spear"], None)


def test_the_yari_masters_spear_reads_as_a_spear():
    # 0.26-0.27.1 read this Spear as a Sword (18 of 30 Caves reads; test_decide_kind_on_day_1 keeps
    # that case); 0.28 added the Yari Master's and Iwao's Spear to the icon templates
    assert _reads(REC + "f0198_idle.png")[4] == ("yari_master", 4, 4, ["spear"], None)


def test_the_stage_1_roster_reads_them_as_elite_ashigaru():
    # what 0.25 saw, and why the bot now passes its route
    r = _reads(REC + "f0204_CLICK.png", regions=("bamboo_grove",), day=None)
    assert r[2][0] == "ashigaru" and r[4][0] == "ashigaru"


def test_the_tracker_plans_a_yari_master_with_its_spear():
    s = Tracker().state(perceive(frame(REC + "f0198_idle.png"), regions=CAVES, day=1))
    y = next(e for e in s.enemies if e.x == 4)
    assert (y.kind, y.hp, y.queue, y.alt) == ("yari_master", 4, ["spear"], None)


def _board(units, hero_x=0, hero_facing=1):
    hero = Hero(hero_x, hero_facing, hp=10, kit=Kit.of("swirl", "arrow"))
    return State(7, hero, units, level="stage2-1", wave=1, waves_total=4)


def test_the_spear_hits_the_two_cells_ahead_allies_too():
    # catalogue: Spear hits the 2 cells directly ahead; the Yari Master attacks even if it hits
    # allies [wiki]. Loaded and yellow (fire) at 3 facing left: an Ashigaru at 2, the hero at 1.
    y = Enemy("yari_master", 3, -1, 4, ["spear"], uid=1, intent="fire")
    a = Enemy("ashigaru", 2, 1, 3, [], uid=2, intent="hold")
    outs = step(_board([y, a], hero_x=1), "wait")
    assert len(outs) == 1
    _, s2, ev = outs[0]
    assert s2.hero.hp == 8
    assert next(e for e in s2.enemies if e.uid == 2).hp == 1
    assert ENEMY_KINDS["yari_master"] == (4, "spear")


def test_the_spear_does_not_reach_three_cells():
    y = Enemy("yari_master", 3, -1, 4, ["spear"], uid=1, intent="fire")
    _, s2, _ = step(_board([y], hero_x=0), "wait")[0]
    assert s2.hero.hp == 10


def test_a_bot_run_replays_with_the_route_it_logged(tmp_path):
    # 0.28: run 20261029-124921 (--full) logged its map travel and day; the replay reads them, so its
    # Caves levels are read with the Caves roster as the bot read them
    from shogunbot.replay import logged_route
    run = tmp_path / "20261029-124921"
    (run / "frames").mkdir(parents=True)
    (run / "events.jsonl").write_text(
        '{"event": "day_select", "day": 1}\n'
        '{"event": "map_travel", "node": "SLOT UPGRADE ? DANCER S?OP", "node_kind": "shop", "region": null}\n'
        '{"event": "map_travel", "node": "W?ISPERING CA?ES", "node_kind": "region", "region": "whispering_caves"}\n',
        encoding="utf-8")
    assert logged_route(run / "frames") == (["bamboo_grove", "whispering_caves"], 1)
    assert logged_route(tmp_path) == (None, None)
