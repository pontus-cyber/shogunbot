"""0.21: waves from the HUD counter "WAVE x/y" (perception.hud, shogunbot.waves) and the
screen classifier on the full-game recording [user review of collect/20261028-190026: red
backdrops read as camp, Moonlit Port and the Keep as unknown; "the HUD 'WAVE x/y' is in
every region: use this run for M3 step 2"]. Frame-based tests skip when the frames are
missing."""

import pytest
from test_bot_flow import frame
from test_facing import _p

from shogunbot.bridge import Tracker
from shogunbot.perception import hud
from shogunbot.waves import Waves

FULL = "collect/20261028-190026/"


# ------------------------------------------------------------------ the counter
@pytest.mark.parametrize("rel,text", sorted(hud.WAVE_SAMPLES.items()))
def test_counter_reads_the_samples(rel, text):
    cur, total = text.split("/")
    assert hud.read_wave(frame(rel)) == (int(cur), int(total))


@pytest.mark.parametrize("rel,want", [
    (FULL + "f0602_idle.png", (1, 5)),                       # salmon, Spirit Gateway's red backdrop
    (FULL + "f1621_idle.png", (1, 10)),                      # grey, the Shogun's Castle
    (FULL + "f1043_S.png", (7, 15)),                         # grey, Hot Springs' 15-wave level
    ("runs/20261028-104830/frames/f0064_A.png", (3, 5)),     # grey, Bamboo Grove II
])
def test_counter_reads_every_colour(rel, want):
    assert hud.read_wave(frame(rel)) == want


@pytest.mark.parametrize("rel", [
    FULL + "f0012_idle.png",                                 # level start: no counter yet
    FULL + "f0426_idle.png",                                 # 'WAVE 5/5' popping in, drawn larger
    FULL + "f0880_idle.png",                                 # boss: name and HP bar
    FULL + "f0001_idle.png",                                 # title
    FULL + "f0326_CLICK.png",                                # map
    "runs/20261028-153448/frames/t0057_before.png",          # boss intro banner
])
def test_no_counter_no_read(rel):
    assert hud.read_wave(frame(rel)) is None


# ------------------------------------------------------------------ screens
@pytest.mark.parametrize("rel,want", [
    (FULL + "f0001_idle.png", {"title"}),
    (FULL + "f0004_idle.png", {"camp"}),
    (FULL + "f0010_idle.png", {"map"}),
    (FULL + "f0326_CLICK.png", {"map"}),
    (FULL + "f0190_CLICK.png", {"combat"}),                  # Whispering Caves
    (FULL + "f0602_idle.png", {"combat"}),                   # Spirit Gateway: read as camp in 0.20.1
    (FULL + "f1012_idle.png", {"combat"}),                   # Hot Springs: read as camp in 0.20.1
    (FULL + "f1650_D.png", {"combat"}),                      # the Shogun's Castle, grey counter
    (FULL + "f0880_idle.png", {"boss"}),                     # The Statue: read as camp in 0.20.1
    (FULL + "f1300_SPACE.png", {"boss"}),                    # Kowa the Coward
    (FULL + "f1760_SPACE.png", {"boss"}),                    # The Shogun
    (FULL + "f0340_D.png", {"unknown", "combat"}),           # Moonlit Port: plates too dark to read yet
    (FULL + "f0300_CLICK.png", {"unknown", "boss"}),         # Iwao's red flash: read as title in 0.20.1
    # Stage 1: unchanged (all 3,096 Stage 1 frames classify as in 0.20.1)
    ("collect/20261027-164727/f0005_D.png", {"camp"}),       # camp floor: five plates and the hero
    ("collect/20261027-164727/f0001_idle.png", {"title"}),
    ("collect/20261027-164727/f0009_idle.png", {"map"}),
    ("collect/20261027-164727/f0042_CLICK.png", {"reward"}),
    ("collect/20261027-164727/f0114_idle.png", {"boss"}),
    ("runs/20261027-182223/frames/death.png", {"death"}),
])
def test_screen_types(rel, want):
    from shogunbot.perception.perceive import perceive
    p = perceive(frame(rel))
    assert p.screen in want


def test_perception_carries_the_counter():
    from shogunbot.perception.perceive import perceive
    assert perceive(frame(FULL + "f1012_idle.png")).wave == (1, 15)


# ------------------------------------------------------------------ new units
def test_new_units_are_the_ones_no_old_unit_can_have_become():
    tr = Tracker()
    tr.state(_p([{"x": 1, "kind": "tachi", "facing": 1, "intent": "move-right"},
                 {"x": 5, "kind": "ashigaru", "facing": -1, "intent": "hold"}]))
    assert len(tr.fresh) == 2                                # a level's first state: all new
    tr.state(_p([{"x": 2, "kind": "tachi", "facing": 1},     # the telegraphed move
                 {"x": 5, "kind": "ashigaru", "facing": -1},
                 {"x": 6, "kind": "ashigaru", "facing": -1}]))
    assert tr.fresh == [(6, "ashigaru", -1)]


def test_one_old_unit_accounts_for_one_new_read():
    tr = Tracker()
    tr.state(_p([{"x": 2, "kind": "ashigaru", "facing": -1, "intent": "move-right"}]))
    tr.state(_p([{"x": 2, "kind": "ashigaru", "facing": -1}, {"x": 4, "kind": "ashigaru", "facing": -1}],
                hero_x=0))
    assert tr.fresh == [(4, "ashigaru", -1)]
    tr = Tracker()
    tr.state(_p([{"x": 2, "kind": "ashigaru", "facing": -1, "intent": "move-right"}], hero_x=0))
    tr.state(_p([{"x": 2, "kind": "ashigaru", "facing": -1}, {"x": 3, "kind": "ashigaru", "facing": -1}],
                hero_x=0))
    assert len(tr.fresh) == 1                                # one of the two is new


def test_a_unit_with_unknown_telegraph_may_have_moved_one_cell():
    tr = Tracker()
    tr.state(_p([{"x": 4, "kind": "tachi", "facing": -1, "intent": "move-left"}]))
    tr.mirror()                                              # Rain of Mirrors: telegraphs forgotten
    tr.state(_p([{"x": 5, "kind": "tachi", "facing": 1}]))
    assert tr.fresh == []


# ------------------------------------------------------------------ wave log
def _alive(n):
    return [object()] * n


def test_hud_counts_a_wave_that_arrives_as_the_last_enemy_dies():
    w = Waves()
    assert [e[1]["wave"] for e in w.update((1, 4), _alive(1), [(3, "ashigaru", -1)], 0)] == [1]
    assert w.update((1, 4), _alive(1), [], 1) == []
    # the last enemy died and wave 2 arrived on the same turn: one living enemy before and after
    ev = w.update((2, 4), _alive(1), [(0, "tachi", 1)], 2)
    assert len(ev) == 1 and ev[0][0] == "wave_spawn"
    f = ev[0][1]
    assert (f["wave"], f["total"], f["source"], f["survivors"]) == (2, 4, "hud", 0)
    assert f["spawned"] == [{"x": 0, "kind": "tachi", "facing": 1}]   # new units only
    assert (w.wave, w.total) == (2, 4)


def test_without_the_counter_a_rise_in_enemies_is_a_wave():
    w = Waves()
    w.update((1, 5), _alive(1), [(3, "ashigaru", -1)], 0)
    ev = w.update(None, _alive(3), [(0, "tachi", 1), (6, "charger", -1)], 1)
    assert ev[0][1]["wave"] == 2 and ev[0][1]["source"] == "count" and ev[0][1]["survivors"] == 1
    assert w.update((2, 5), _alive(3), [], 2) == []          # the HUD agrees: nothing more


def test_the_counter_corrects_a_wave_counted_by_mistake():
    w = Waves()
    w.update((1, 5), _alive(2), [(1, "tachi", 1), (4, "tachi", -1)], 0)
    w.update(None, _alive(1), [], 1)
    w.update(None, _alive(2), [(4, "tachi", -1)], 2)        # a unit missed on one frame, seen again
    assert w.wave == 2
    ev = w.update((1, 5), _alive(2), [], 3)
    assert ev == [("wave_corrected", {"turn": 3, "counted": 2, "hud": 1})] and w.wave == 1


def test_units_seen_a_turn_before_the_counter_still_join_the_wave():
    w = Waves()
    w.update((1, 4), _alive(1), [(3, "ashigaru", -1)], 0)
    w.update(None, _alive(1), [(0, "tachi", 1)], 4)         # the counter unreadable, no rise
    ev = w.update((2, 4), _alive(1), [], 5)
    assert ev[0][1]["spawned"] == [{"x": 0, "kind": "tachi", "facing": 1}]


def test_runlog_records_the_total_and_source(tmp_path):
    import json
    from shogunbot.runlog import RunLogger
    log = RunLogger(str(tmp_path), "none")
    log.start_run({"test": True})
    log.level_start("bamboo-1", 10, 0)
    log.wave_seen(1, [{"x": 3, "kind": "ashigaru", "facing": -1}], 0, 0, total=4, source="hud", survivors=0)
    log.level_end(10, "next")
    ev = [json.loads(l) for l in open(log.run_dir / "events.jsonl", encoding="utf-8")]
    sp = next(e for e in ev if e["event"] == "wave_spawn")
    assert (sp["total"], sp["source"], sp["survivors"]) == (4, "hud", 0)
    lv = next(e for e in ev if e["event"] == "level_end")
    assert lv["waves_total"] == 4
