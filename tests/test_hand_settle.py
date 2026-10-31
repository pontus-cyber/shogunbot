"""0.36 [user review of run 20261030-213120, item 2: hand read misses at t62 and t168].

Both came from animations in the hand that grab_stable cannot see (the whole screen hardly changes):
at t62 the Arrow came ready and a white bar swept over its face, so the hand read one unknown face (the
Charge, cut by the flash, conf 0.11) and no Arrow; the planner chose queue:arrow, the tile was not found,
and on the next frame the loop breaker took the unchanged board for a refused click and planned the
Arrow as recharging. At t168 (Iwao) the Arrow flashed again: the hand read empty, so the queue bubbles
could not be matched to kit tiles (the Katana's bubble reads as Spear), the Katana was taken for a hand
tile and queue:katana was planned while it sat in the queue. A hand read whose faces and queue bubbles do
not account for the kit is now read again before planning (bot.HAND_WAIT_S), and an action that could not
be sent no longer counts as refused by the game. Frame tests skip when the frames are missing."""

import json

import pytest
from test_bot_flow import _stub_windows_modules, frame, make_bot
from test_series import _GW, _Rec

from shogunbot.bridge import Tracker, hand_unsettled
from shogunbot.model import Kit
from shogunbot.perception.perceive import perceive

_stub_windows_modules()

RUN = "runs/20261030-213120/frames/"
T62_KIT = Kit.of("swirl", "arrow", "charge")
T168_KIT = Kit.of("swirl", "arrow", "charge", "katana")
CAVES = ("bamboo_grove", "whispering_caves")


def _p(name, kit, regions=("bamboo_grove",)):
    return perceive(frame(RUN + name), [t.tile for t in kit], regions=regions, day=1)


def _events(b):
    return [json.loads(l) for l in open(b.log.run_dir / "events.jsonl", encoding="utf-8")]


@pytest.mark.parametrize("name,kit,regions,why", [
    ("t0062_before.png", T62_KIT, ("bamboo_grove",), "unread"),         # the Arrow's ready flash (t62, 1st read)
    ("t0062_after.png", T62_KIT, ("bamboo_grove",), None),              # 3 s later: Charge, Arrow, Swirl
    ("t0167_after.png", T168_KIT, CAVES, "0 hand faces and 3 queue bubbles"),   # t168, 1st read: empty hand
    ("t0168_before.png", T168_KIT, CAVES, None),                        # t168, 2nd read: the Arrow, 3 bubbles
    ("t0095_before.png", T62_KIT, ("bamboo_grove",), "2 hand faces and 0 queue bubbles"),   # the Swirl's flash
])
def test_a_hand_read_that_does_not_account_for_the_kit_is_unsettled(name, kit, regions, why):
    got = hand_unsettled(_p(name, kit, regions), kit)
    assert (got is None) if why is None else (why in got)


def test_a_tile_taken_at_the_last_reward_does_not_hold_the_level_start():
    # run 213120 t23, Bamboo Grove II's first turn: the Charge taken at the tile choice is a third face while the
    # tracker's kit still holds two tiles (it adds the Charge at the level's start, Tracker.state)
    p = _p("t0023_before.png", T62_KIT)
    assert [f["tile"] for f in p.hand] == ["swirl", "arrow", "charge"]
    assert hand_unsettled(p, Kit.of("swirl", "arrow")) is None


def _run_on(tmp_path, monkeypatch, frames, kit, do=None):
    from shogunbot import bot as B
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(B, "BLINK_SAMPLES", 0)
    monkeypatch.setattr(B, "Tracker", lambda: Tracker(kit))

    class GW(_GW):
        """Serves the frames in turn, one per grab_stable, and stays on the last."""
        def __init__(self, frames):
            super().__init__([(f, []) for f in frames])
            self.inputs = []

        def grab_stable(self, **kw):
            f = self.steps[self.i][0]
            self.i = min(self.i + 1, len(self.steps) - 1)
            return f

        def click(self, fx, fy):
            self.inputs.append(("click", round(fx * 1920)))
            raise KeyboardInterrupt

        def key(self, name):
            self.inputs.append(("key", name))
            raise KeyboardInterrupt

    gw = GW(frames)
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after = _Rec(), False
    if do is not None:
        monkeypatch.setattr(B.Bot, "do", do)
    return b, gw, b.run()


def test_the_bot_reads_the_hand_again_before_planning(tmp_path, monkeypatch):
    # t62: the flashing frame, then the settled one; 0.35 planned on the first and queue:arrow failed
    b, gw, res = _run_on(tmp_path, monkeypatch, [frame(RUN + "t0062_before.png"), frame(RUN + "t0062_before.png"),
                                                 frame(RUN + "t0062_after.png")], T62_KIT)
    assert res == "stopped" and len(gw.inputs) == 1
    ev = _events(b)
    wait = [e for e in ev if e["event"] == "hand_wait"]
    assert len(wait) == 1 and wait[0]["settled"] and "unread" in wait[0]["reason"]
    assert not any(e["event"] == "act_failed" for e in ev)
    kind, x = gw.inputs[0]
    if kind == "click":                     # a queued tile: one of the faces the settled frame shows
        assert any(abs(x - cx) < 40 for cx in (840, 960, 1080))


def test_t168_waits_out_the_empty_hand_and_does_not_queue_the_queued_katana(tmp_path, monkeypatch):
    # t168 (Iwao): the first read had an empty hand (the Arrow's flash), the second the Arrow and 3 bubbles
    b, gw, res = _run_on(tmp_path, monkeypatch, [frame(RUN + "t0167_after.png"), frame(RUN + "t0168_before.png")],
                         T168_KIT)
    assert res == "stopped" and len(gw.inputs) == 1
    ev = _events(b)
    assert [e["settled"] for e in ev if e["event"] == "hand_wait"] == [True]
    assert not any(e["event"] == "act_failed" for e in ev)
    first = json.loads(open(b.log.run_dir / "turns.jsonl", encoding="utf-8").readline())
    assert first["hero"]["queue"] == ["charge", "swirl", "katana"] and first["decision"]["action"] != "queue:katana"


def test_a_hand_that_never_settles_is_planned_on_after_the_wait(tmp_path, monkeypatch):
    from shogunbot import bot as B
    monkeypatch.setattr(B, "HAND_WAIT_S", 0.0)
    b, gw, res = _run_on(tmp_path, monkeypatch, [frame(RUN + "t0062_before.png")], T62_KIT)
    assert res == "stopped"
    wait = [e for e in _events(b) if e["event"] == "hand_wait"]
    assert len(wait) == 1 and not wait[0]["settled"]


def test_an_action_that_could_not_be_sent_is_not_taken_for_a_refused_click(tmp_path, monkeypatch):
    # t62, second read: 0.35 logged 'the tile did not leave the hand: planned as recharging' for the Arrow,
    # which had never been clicked
    calls = []

    def do(self, action, p, frame_, kit=None, charge=None):
        calls.append(action)
        if len(calls) == 1:
            return None                     # 'tile not found in hand'
        raise KeyboardInterrupt

    b, gw, res = _run_on(tmp_path, monkeypatch, [frame(RUN + "t0062_after.png")], T62_KIT, do=do)
    assert res == "stopped" and len(calls) == 2 and calls[1] != calls[0]
    reasons = [e["reason"] for e in _events(b) if e["event"] == "loop_break"]
    assert reasons == ["action could not be sent"]
