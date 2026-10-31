"""0.36 [user review of run 20261030-213120, item 4: 11 enemy kind mismatches, facing 20].

The 11: six 2-pip units planned as Tachi with the Archer as alt until the Archer's Arrow showed, three
Yumi Snipers (a visitor of Moonlit Port, named only by its Volley) planned as Ashigaru until it showed,
and the two Twins at t303, read during their intro ('THE WANDERER V.S. THE TWINS'), before the boss bar,
as two 3-HP Ashigaru. None was a tile misread. A kind change to the alt the planner played is no longer
logged as a misprediction (as in replay), and a boss level now waits for its bar (bot.BOSS_INTRO_S).
Frame tests skip when the frames are missing."""

import json

from test_bot_flow import _stub_windows_modules, frame, make_bot
from test_series import _GW, _Rec

from shogunbot.bridge import Tracker
from shogunbot.model import Enemy, Hero, Kit, State
from shogunbot.perception.perceive import perceive
from shogunbot.runlog import state_diff

_stub_windows_modules()

RUN = "runs/20261030-213120/frames/"
ROUTE = ["bamboo_grove", "whispering_caves", "moonlit_port"]


def test_a_kind_the_planner_kept_as_alt_is_no_misprediction():
    # t184: the unit on cell 2, read 'T2' (Tachi, alt Archer), loaded an Arrow the next turn
    pred = State(7, Hero(3, 1), [Enemy("tachi", 2, 1, 2, ["swirl"], uid=1, alt="ashigaru_archer")])
    obs = State(7, Hero(3, 1), [Enemy("ashigaru_archer", 2, 1, 2, ["arrow"], uid=1)])
    assert state_diff(pred, obs) == []
    # a kind the planner did not keep still counts (t195: the Sniper, a visitor, planned as an Ashigaru)
    pred = State(7, Hero(5, -1), [Enemy("ashigaru", 1, 1, 3, ["sword"], uid=1)])
    obs = State(7, Hero(5, -1), [Enemy("yumi_sniper", 1, 1, 3, ["volley"], uid=1)])
    assert state_diff(pred, obs) == ["cell1.kind: pred=ashigaru obs=yumi_sniper"]


class _MoonlitTracker(Tracker):
    """A tracker past Moonlit Port III's reward: the next fight is The Twins."""
    @property
    def route(self):
        return ROUTE

    @route.setter
    def route(self, v):
        pass


def test_the_boss_intro_is_waited_out(tmp_path, monkeypatch):
    from shogunbot import bot as B
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(B, "BLINK_SAMPLES", 0)
    kit = Kit.of("katana", "charge", "arrow", "swirl")

    def tracker():
        tr = _MoonlitTracker(kit)
        tr.stage, tr.level_i, tr.advance, tr.level = 3, 2, True, "stage3-3"
        return tr

    monkeypatch.setattr(B, "Tracker", tracker)
    monkeypatch.setattr(B, "perceive", lambda f, k=None, regions=(), day=None: perceive(f, k, regions=tuple(ROUTE), day=1))
    intro, bar = frame(RUN + "t0303_before.png"), frame(RUN + "t0303_after.png")

    class GW(_GW):
        def __init__(self):
            super().__init__([(intro, []), (intro, []), (bar, [])])

        def grab_stable(self, **kw):
            f = self.steps[self.i][0]
            self.i = min(self.i + 1, len(self.steps) - 1)
            return f

        def click(self, fx, fy):
            raise KeyboardInterrupt

        def key(self, name):
            raise KeyboardInterrupt

    b = make_bot(tmp_path, GW())
    b.rec, b.restart_after = _Rec(), False
    assert b.run() == "stopped"
    ev = [json.loads(l) for l in open(b.log.run_dir / "events.jsonl", encoding="utf-8")]
    wait = [e for e in ev if e["event"] == "boss_intro_wait"]
    assert len(wait) == 1 and wait[0]["bar"] and wait[0]["level"] == "stage3-boss"
    first = json.loads(open(b.log.run_dir / "turns.jsonl", encoding="utf-8").readline())
    assert first["level"] == "stage3-boss"
    assert sorted((e["x"], e["kind"]) for e in first["state"]["enemies"]) == [(0, "the_twins"), (6, "the_twins")]
