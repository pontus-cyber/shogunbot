"""0.22: the level generator learned from the recordings (shogunbot.levelgen, data/waves.json)
and self-play's use of it [M3 step 2]. The frame-based test skips when the frames are missing."""

import random

import pytest
from test_bot_flow import DATA

from shogunbot import levelgen as L
from shogunbot import selfplay as SP
from shogunbot.model import Enemy, Hero, State


def test_learned_waves_match_the_counter_and_the_rosters():
    g = L.generator()
    assert g["levels"]["bamboo-1"]["waves"] == 4 and g["levels"]["bamboo-2"]["waves"] == 5
    first = {u for t, _ in L.wave_templates("bamboo-1", 1) for u in t}
    assert first == {"ashigaru", "tachi"} and all(len(t) == 1 for t, _ in L.wave_templates("bamboo-1", 1))
    sizes = {len(t) for t, _ in L.wave_templates("bamboo-2", 3)}
    assert sizes <= {2, 3}
    # a wave number never seen (the stress scenario's 7th) takes the nearest learned one
    assert L.wave_templates("bamboo-2", 7) == L.wave_templates("bamboo-2", 5)
    assert L.wave_templates("no-such-level", 1) == []


def test_next_wave_timing():
    assert L.p_next_wave(0, 0, 1) == 1.0                    # nobody left: at once
    assert L.p_next_wave(1, 6, 6) == 1.0                    # sixth turn in a row at one: for sure
    assert L.p_next_wave(1, 1, 2) == 0.0                    # a young wave never comes early
    assert 0 < L.p_next_wave(1, 2, 10) < 0.5
    assert L.p_next_wave(2, 0, 5) == 0.0 and L.p_next_wave(3, 0, 20) == 0.0


def test_cells_next_to_the_hero_are_drawn_less_often():
    assert L.cell_weight("bamboo-2", 1) < 0.5 < L.cell_weight("bamboo-2", 2)
    assert L.cell_weight("bamboo-2", 9) == L.cell_weight("bamboo-2", max(
        int(k) for k in L.generator()["levels"]["bamboo-2"]["cell_weight_by_distance"]))


def test_self_play_waves_follow_the_learned_templates():
    for seed in range(20):
        plan = SP.plan_levels(seed)
        assert plan == SP.plan_levels(seed)                 # common random numbers
        b1 = plan[0]
        assert b1.learned == "bamboo-1" and [len(w) for w in b1.waves][:2] == [1, 1]
        assert all(1 <= len(w) <= 2 for w in b1.waves[2:])
        assert all(2 <= len(w) <= 3 for w in plan[1].waves)


def test_spawn_avoids_the_hero_and_follows_the_facing_draw():
    s = State(7, Hero(3, 1, hp=10), [Enemy("ashigaru", 4, -1, 3, [], uid=1)], level="bamboo-2")
    wave = [("tachi", r, 0.0) for r in (0.0, 0.5, 0.99)]
    out = SP._spawn(s, wave, 10, False, "bamboo-2")
    assert len(out) == 3 and not {e.x for e in out} & {3, 4} and len({e.x for e in out}) == 3
    assert all(e.facing == (1 if e.x < 3 else -1) for e in out)        # r_face 0: toward the hero
    away = SP._spawn(s, [("tachi", 0.0, 0.999)], 10, False, "bamboo-2")[0]
    assert away.facing == (-1 if away.x < 3 else 1)
    # never more than (cells + 1) / 2 alive [wiki]
    assert len(SP._spawn(s, [("tachi", random.random(), 0.5) for _ in range(6)], 10, False, "bamboo-2")) == 3


def test_frame_waves_reads_a_recording():
    folder = DATA / "runs/20261028-121403/frames"       # 0.17, stopped by hand at Bamboo Grove II
    if not (folder / "actions.jsonl").exists():
        pytest.skip("recorded frames not available")
    r = L.frame_waves(folder)
    b1 = [w for w in r["waves"] if w["level"] == "bamboo-1"]
    assert [w["wave"] for w in b1] == [1, 2, 3, 4]
    assert {w["total"] for w in b1} == {4} and {w["source"] for w in b1} == {"hud"}
    assert [len(w["spawned"]) for w in b1][:2] == [1, 1]
