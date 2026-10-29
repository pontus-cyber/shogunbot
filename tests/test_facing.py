"""0.20.1: enemy facing of new units [user review of the 0.20 series: enemy facing was the
largest prediction mismatch, 35 turns]. Of 345 new units in 21 runs, 86 got a wrong facing:
52 read under spawn flames (no sprite), 25 Spike Chargers (symmetric sprite), 9 inherited from
the unit they replaced. Frame-based tests skip when the frames are missing."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_bot_flow import frame, make_bot

from shogunbot.bridge import Tracker


def _p(enemies, hero_x=3, cells=7):
    hero = {"x": hero_x, "facing": 1, "hp": 10, "hp_max": 10, "queue": [], "swap_red": 0}
    ens = [dict({"hp": 2, "hp_max": 2, "queue": [], "intent": None, "fire_now": False, "cx": 0}, **e) for e in enemies]
    return SimpleNamespace(hero=hero, hand=[], cells=list(range(cells)), screen="combat", enemies=ens,
                           confidence={"hero_facing": 1.0}, boss_frac=None, items=[])


def test_spawn_flames_are_flagged_and_give_no_facing():
    # run 20261028-153448 turn 8: two Tachis spawning in flames at cells 0 and 1; turn 9 the
    # same Tachis show their sprites (and the turn telegraph: they spawned facing away)
    from shogunbot.perception.perceive import perceive
    p8 = perceive(frame("runs/20261028-153448/frames/t0008_before.png"))
    assert [(e["x"], e["spawning"], e["facing"]) for e in p8.enemies] == [(0, True, 0), (1, True, 0)]
    p9 = perceive(frame("runs/20261028-153448/frames/t0009_before.png"))
    assert [(e["x"], e["spawning"], e["facing"]) for e in p9.enemies] == [(0, False, -1), (1, False, -1)]


def test_settled_units_are_not_flagged():
    # Ashigaru (orange hat), Charger (red body) and Tachi on settled frames
    from shogunbot.perception.perceive import perceive
    for rel in ("runs/20261028-153448/frames/t0025_before.png", "runs/20261028-154317/frames/t0018_before.png"):
        assert not any(e["spawning"] for e in perceive(frame(rel)).enemies), rel


def test_a_new_unit_does_not_take_the_facing_of_the_unit_it_replaced():
    # last turn a Tachi at cell 1 faced the hero and held; this turn a new Tachi stands at
    # cell 0. It cannot be the old one (no move telegraph), so its sprite decides.
    tr = Tracker()
    tr.state(_p([{"x": 1, "kind": "tachi", "facing": 1, "intent": "hold"}]))
    s = tr.state(_p([{"x": 0, "kind": "tachi", "facing": -1}]))
    assert s.enemies[0].facing == -1
    # one that telegraphed the move keeps its facing whatever the sprite says
    tr = Tracker()
    tr.state(_p([{"x": 1, "kind": "tachi", "facing": 1, "intent": "move-left"}]))
    s = tr.state(_p([{"x": 0, "kind": "tachi", "facing": -1}]))
    assert s.enemies[0].facing == 1


def test_could_be_follows_moves_charges_and_daisuke():
    cb = Tracker._could_be
    assert cb(("tachi", 2, 1, "hold"), 2)
    assert cb(("tachi", 2, 1, "move-right"), 3) and not cb(("tachi", 2, 1, "move-left"), 3)
    assert not cb(("ashigaru", 2, 1, "hold"), 3)
    assert cb(("charger", 1, 1, "fire"), 4) and not cb(("charger", 1, 1, "fire"), 0)
    assert cb(("daisuke", 4, -1, "fire"), 1)


def test_a_new_charger_is_taken_to_face_the_hero():
    # the Spike Charger's sprite is symmetric: its read was right on 20 of 42 new Chargers
    tr = Tracker()
    s = tr.state(_p([{"x": 5, "kind": "charger", "facing": 1, "hp": 1, "hp_max": 1}]))
    assert s.enemies[0].facing == -1          # the hero is at 3
    tr.state(_p([{"x": 5, "kind": "charger", "facing": 1, "hp": 1, "hp_max": 1, "intent": "turn"}]))
    s = tr.state(_p([{"x": 5, "kind": "charger", "facing": 1, "hp": 1, "hp_max": 1, "intent": "hold"}]))
    assert s.enemies[0].facing == -1          # turned toward the hero after the turn telegraph


class _Seq:
    """A game window whose grab() serves frames in order (the last one repeats)."""
    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def grab(self):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return f

    def move_mouse_away(self):
        pass


def test_bot_waits_for_the_flames_before_reading_facing(tmp_path, monkeypatch):
    from shogunbot.perception.perceive import perceive
    monkeypatch.setattr("time.sleep", lambda s: None)
    flames = frame("runs/20261028-153448/frames/t0008_before.png")
    clear = frame("runs/20261028-153448/frames/t0009_before.png")
    b = make_bot(tmp_path, _Seq([flames, clear]))
    f, p = b._wait_spawns(flames, perceive(flames), Tracker(), 8)
    assert f is clear and not any(e["spawning"] for e in p.enemies)


def test_bot_stops_waiting_after_the_limit(tmp_path, monkeypatch):
    from shogunbot import bot as B
    from shogunbot.perception.perceive import perceive
    flames = frame("runs/20261028-153448/frames/t0008_before.png")
    clock = iter(np.arange(0.0, 100.0, 1.0))
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(B.time, "time", lambda: next(clock))
    b = make_bot(tmp_path, _Seq([flames]))
    f, p = b._wait_spawns(flames, perceive(flames), Tracker(), 8)
    assert all(e["spawning"] for e in p.enemies)    # still flames: the bot goes on with what it has


def test_a_turn_telegraph_survives_a_swap_past_another_unit():
    # run 20261029-184142 t31: the Ashigaru on cell 6 telegraphed a turn while the hero swapped with the Charger
    # on cell 4; it turned (its sprite reads facing left on t0031_after), but the swap cleared every unit's
    # telegraph, so the tracker kept it facing right and both t31 and t32 counted a facing miss [user review,
    # note 3]. Only the swapped unit's telegraph is cleared now
    from shogunbot.model import Kit
    from shogunbot.perception.perceive import normalize, perceive
    R = "runs/20261029-184142/frames/"
    kit = Kit.of("swirl", "arrow", "spear")
    tr = Tracker(kit)
    tr.route = ["bamboo_grove"]
    see = lambda f: perceive(normalize(frame(R + f)), [t.tile for t in kit], regions=("bamboo_grove",), day=1)
    s = tr.state(see("t0031_before.png"))
    assert [(e.kind, e.x, e.facing, e.intent) for e in s.enemies] == [("charger", 4, 1, "turn"), ("ashigaru", 6, 1, "turn")]
    tr.acted(s, "right")                                      # the swap with the Charger
    s2 = tr.state(see("t0031_after.png"))
    assert [(e.kind, e.x, e.facing) for e in s2.enemies] == [("charger", 3, -1), ("ashigaru", 6, -1)]
