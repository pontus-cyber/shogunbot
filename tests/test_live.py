"""0.19: the live decision viewer. The planner explains its move (principal variation and
plain-word reasons) without changing it, the bot writes runs/<id>/live.json, and
`python -m shogunbot watch` serves it."""

import json
import os
import threading
import time
import urllib.request

from test_stall import STALLED

from shogunbot.live import LIVE_FILE, LiveView
from shogunbot.model import Enemy, Hero, State
from shogunbot.planner import WEIGHTS, explain_decision, plan, plan_live
from shogunbot.runlog import RunLogger


def test_principal_variation_is_the_zero_damage_double_kill():
    # self-play seed 9008, Bamboo Grove II turn 40 (tests/test_stall.py): the line the
    # search found at 4 plies, with the telegraphs it expected at each turn
    s = State.from_dict(STALLED)
    d = explain_decision(s, plan(s, 4, WEIGHTS))
    assert [p["action"] for p in d.pv] == ["queue:swirl", "right", "right", "attack"]
    assert sorted(d.pv[-1]["kills"]) == ["charger", "tachi"] and d.pv[-1]["hero_hp"] == 10
    assert all(p["telegraphs"] for p in d.pv) and d.reason


def test_explaining_does_not_change_the_decision():
    s = State.from_dict(STALLED)
    plain, explained = plan_live(s, WEIGHTS), plan_live(s, WEIGHTS, explain=True)
    assert (plain.action, plain.value, plain.candidates) == (explained.action, explained.value, explained.candidates)
    assert not plain.pv and explained.pv


def test_reason_names_the_damage_the_best_move_avoids():
    # the hero on the edge, a loaded Ashigaru next to it fires this turn (R2): only the swap
    # (moving into it) avoids the 2 damage
    s = State(5, Hero(0, 1, hp=10), [Enemy("ashigaru", 1, -1, 3, ["sword"], uid=1, intent="fire", wind=2,
                                           last="queue", fage=None)], level="bamboo-1", wave=1, waves_total=4)
    d = explain_decision(s, plan(s, 3, WEIGHTS))
    ranked = sorted(d.candidates, key=lambda c: -c["value"])
    assert ranked[0]["action"] == "right" and ranked[0]["exp_damage"] == 0 and ranked[1]["exp_damage"] == 2
    assert d.reason[0].startswith("right scores")
    assert d.reason[1].startswith("+ keeps 2 more HP")


def _decision():
    s = State.from_dict(STALLED)
    return s, explain_decision(s, plan(s, 4, WEIGHTS))


def test_live_view_writes_a_complete_document(tmp_path):
    s, d = _decision()
    lv = LiveView()
    lv.start(tmp_path, "run-x", "0.19")
    rec = {"turn": 39, "level": "bamboo-2", "board": s.render(), "decision": {"action": "wait"},
           "predicted": [{"p": 1.0, "board": s.render(), "hero_hp": 10}], "observed": {"board": s.render(), "hero_hp": 10},
           "matched_outcome": 0, "matched_p": 1.0, "sim_diff": [], "hp_lost": 0, "hp_lost_predicted": 0}
    lv.turn(s=s, d=d, turn=40, plan_ms=12.0, last_record=rec)
    lv.reward({"layout": "shrine", "upgrade": "damage_1", "level": "bamboo-2", "hp": 10, "choice": "upgrade:swirl",
               "rule": "pure gain: always taken", "options": [{"name": "upgrade:swirl", "scored": False}]})
    lv.event("give_up", {"step": "done", "screen": "camp"})
    lv.end("win", {"turns": 86, "hp_end": 10})
    doc = json.loads((tmp_path / LIVE_FILE).read_text())
    assert doc["status"] == "ended" and doc["result"] == "win" and doc["turn"] == 40
    assert doc["board"]["hero"]["x"] == 0 and len(doc["board"]["enemies"]) == 2
    assert doc["decision"]["action"] == d.action and doc["decision"]["candidates"][0]["delta"] == 0
    assert [p["action"] for p in doc["decision"]["pv"]] == ["queue:swirl", "right", "right", "attack"]
    assert doc["history"][0]["match"] is True and doc["reward"]["choice"] == "upgrade:swirl"
    assert "camp" in doc["events"][-1]["text"]


def test_live_view_never_stops_the_bot(tmp_path, monkeypatch):
    s, d = _decision()
    lv = LiveView()
    lv.start(tmp_path, "run-y", "0.19")

    def locked(*a, **k):
        raise PermissionError("the viewer has the file open")
    monkeypatch.setattr("time.sleep", lambda t: None)
    monkeypatch.setattr(os, "replace", locked)
    lv.turn(s=s, d=d, turn=1, plan_ms=1.0)          # skipped quietly
    monkeypatch.undo()
    lv.turn(s=s, d=d, turn=2, plan_ms=1.0)
    assert json.loads((tmp_path / LIVE_FILE).read_text())["turn"] == 2


def test_logger_hands_finished_turns_and_events_to_the_viewer(tmp_path):
    s, d = _decision()
    seen = []
    log = RunLogger(str(tmp_path), "none")
    log.on_event = lambda kind, data: seen.append(kind)
    log.start_run({"test": True})
    log.turn(state=s, confidence={}, perceive_ms=1.0, decision=d, plan_ms=1.0)
    assert log.last_turn is None
    log.turn(state=s, confidence={}, perceive_ms=1.0, decision=d, plan_ms=1.0)
    assert log.last_turn["turn"] == s.turn and log.last_turn["observed"] is not None
    assert seen[0] == "run_start"


def test_watch_serves_the_page_and_follows_the_newest_run(tmp_path):
    from shogunbot.watch import make_server
    for run, seq in (("20261028-100000", 1), ("20261028-110000", 2)):
        (tmp_path / run).mkdir()
        (tmp_path / run / LIVE_FILE).write_text(json.dumps({"run_id": run, "seq": seq, "status": "playing"}))
        time.sleep(0.02)
    os.utime(tmp_path / "20261028-110000" / LIVE_FILE, None)
    for run, expect in ((None, "20261028-110000"), ("20261028-100000", "20261028-100000")):
        srv = make_server(tmp_path, run, port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
            doc = json.loads(urllib.request.urlopen(base + "/live.json", timeout=5).read())
            assert "Shogunbot live" in page and doc["run_id"] == expect
        finally:
            srv.shutdown()
            srv.server_close()


def test_bot_turn_writes_the_live_feed(tmp_path, monkeypatch):
    # one real combat frame through the whole loop: perceive, plan with explain, log, live.json
    from test_bot_flow import frame, make_bot
    from test_series import _GW, _Rec
    img = frame("runs/20261028-122652/frames/f0003_A.png")
    gw = _GW([(img, [])])
    b = make_bot(tmp_path, gw)              # stubs the Windows-only modules before bot.py loads
    from shogunbot import bot as B
    b.rec, b.restart_after = _Rec(), False
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2               # two loop passes, then F8
    monkeypatch.setattr(B, "_stop_requested", stop)
    monkeypatch.setattr(B, "BLINK_SAMPLES", 0)
    monkeypatch.setattr(B.Bot, "_wait_change", lambda self, before, timeout=2.5: True)
    assert b.run() == "stopped"
    doc = json.loads((b.log.run_dir / LIVE_FILE).read_text())
    assert doc["status"] == "ended" and doc["result"] == "stopped"
    assert doc["decision"]["candidates"] and doc["decision"]["pv"] and doc["decision"]["reason"]
    assert doc["board"]["hero"]["hp"] == 10 and doc["history"]
