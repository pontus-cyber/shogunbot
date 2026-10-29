"""0.23: sequential reward valuation and its live feed [user review of run 20261029-045856:
decisions took 45-73 s; stop each option once its gap to the leader or to skip exceeds about
3 SE, and show the valuation in the live viewer; no faster decision that is a worse one,
user 2026-10-29]."""

import json
import random
import time
from functools import partial

from shogunbot import rewards
from shogunbot.live import LIVE_FILE, LiveView
from shogunbot.model import DEFAULT_KIT, Kit

OFFERS = [{"tile": "katana", "damage": 2, "cooldown": 0, "levels": 4, "fx": 0.41, "fy": 0.43},
          {"tile": "spear", "damage": 2, "cooldown": 5, "levels": 3, "fx": 0.59, "fy": 0.43}]


def _base(seed):
    return random.Random(seed).uniform(3.0, 8.0)


def fake_job(gaps, args):
    """A fight without simulation: skip scores a seeded base, an option the base plus the gap
    of each tile in its kit, by label or 'label/cooldown' (a number, or a function of the seed).
    (score, win, hp, turns, stalled, level reached)."""
    kit_list, hp, seed, depth, scenario = args
    labels = {t["label"] for t in kit_list} | {f"{t['label']}/{t['cooldown']}" for t in kit_list}
    v = _base(seed)
    for label, gap in gaps.items():
        if label in labels:
            v += gap(seed) if callable(gap) else gap
    won = v > 0
    return v, won, max(0, round(v)), 60, False, "stress-boss"


def noisy(mean, sd):
    return lambda seed: mean + random.Random(seed * 7 + 1).gauss(0.0, sd)


EVALUATE = rewards.evaluate     # the real one: some tests put a wrapper in its place


def here(gaps, **kw):
    """rewards.evaluate in this process with fake fights (the decide_* functions pass workers)."""
    def ev(opts, hp, seeds, depth, workers, scenario, **more):
        return EVALUATE(opts, hp, seeds, depth, 1, scenario, job=partial(fake_job, gaps), **kw, **more)
    return ev


def _decide(gaps, stop_z=rewards.STOP_Z):
    return rewards.decide_tile(DEFAULT_KIT, OFFERS, 10, evaluator=here(gaps, stop_z=stop_z))


def test_an_option_far_behind_stops_and_skip_stops_when_settled():
    d = _decide({"katana": noisy(3.0, 0.5), "spear": noisy(0.2, 0.5)})
    by = {o.name: o for o in d.options}
    assert d.best.name == "take:katana"
    assert len(by["take:spear"].scores) == 24 and by["take:spear"].out
    assert "behind take:katana after 24 seeds" in by["take:spear"].stopped
    assert len(by["skip"].scores) == 24 and "settled after 24 seeds" in by["skip"].stopped
    assert d.stats["fights"] == 72 and "settled after 24 of 48 seeds" in d.rule
    assert _decide({"katana": noisy(3.0, 0.5), "spear": noisy(0.2, 0.5)}, stop_z=None).best.name == "take:katana"


def test_a_net_loss_stops_early_and_every_net_loss_skips():
    d = _decide({"katana": noisy(-3.0, 0.5), "spear": noisy(-2.5, 0.5)})
    assert d.best.name == "skip" and "every option a net loss" in d.rule
    assert all(o.out and "a net loss" in o.stopped and len(o.scores) == 24 for o in d.options[1:])


def test_close_options_play_every_seed_and_decide_as_before():
    gaps = {"katana": noisy(0.6, 2.0), "spear": noisy(0.5, 2.0)}
    seq, full = _decide(gaps), _decide(gaps, stop_z=None)
    assert seq.best.name == full.best.name
    by = {o.name: o for o in seq.options}
    assert len(by["take:katana"].scores) == 48 and len(by["take:spear"].scores) == 48


def test_seeds_that_happen_to_agree_exactly_do_not_stop_an_option():
    # a 0.02 lead on every seed has no spread: SD_FLOOR keeps both options playing
    d = _decide({"katana": 0.02, "spear": 0.0})
    by = {o.name: o for o in d.options}
    assert len(by["take:spear"].scores) == 48 and not by["take:spear"].stopped
    assert d.best.name == "take:katana"


def test_pure_gain_stops_early_only_when_settled():
    # [run 20261028-155955, boss shrine -1 cooldown] a 16-seed cap picked Arrow where 48 seeds pick
    # Spear: pure gains play up to 48 seeds and stop on the same test as every other decision
    kit = Kit.of("swirl", "arrow", "spear")
    close = rewards.decide_upgrade(kit, "cooldown_minus_1", 10, evaluator=here({"spear/4": noisy(0.3, 1.0)}))
    assert "pure gain" in close.rule and close.stats["of"] == 48
    assert max(len(o.scores) for o in close.options[1:]) == 48 and close.best.name == "upgrade:spear"
    clear = rewards.decide_upgrade(kit, "cooldown_minus_1", 10, evaluator=here({"spear/4": noisy(3.0, 0.5)}))
    assert clear.best.name == "upgrade:spear" and clear.stats["fights"] == 72 and "settled after 24 of 48" in clear.rule


def test_progress_feed_reports_running_results_and_an_example_fight():
    seen = []
    gaps = {"katana": noisy(3.0, 0.5), "spear": noisy(0.2, 0.5)}
    d = rewards.decide_tile(DEFAULT_KIT, OFFERS, 10, evaluator=here(gaps), progress=seen.append)
    first, last = seen[0], seen[-1]
    assert first["phase"] == "running" and first["fights"] == 0 and len(first["options"]) == 3
    assert last["phase"] == "valued" and last["fights"] == d.stats["fights"] and last["looks"][0] == 24
    rows = {r["name"]: r for r in last["options"]}
    assert rows["take:katana"]["seeds"] == 24 and rows["take:katana"]["diff"] > 2 and not rows["take:katana"]["net_loss"]
    assert rows["take:spear"]["stopped"] and last["leader"] == "take:katana"
    ex = rows["take:katana"]["example"]
    assert ex["seed"] == last["example_seed"] and ex["win"] and ex["turns"] == 60
    assert len(last["example_waves"]) == 7 and last["scenario"]["boss_hp"] == 20
    assert d.stats["feeds"] == len(seen)


def slow_job(args):
    time.sleep(1.2)
    return fake_job({"katana": 3.0}, args)


def test_worker_pool_feeds_the_viewer_while_fights_run():
    # the PC plays the fights in worker processes; the feed must not wait for them
    seen = []
    opts = rewards.tile_choice_options(DEFAULT_KIT, OFFERS[:1])
    try:
        rewards.evaluate(opts, 10, seeds=rewards.SEEDS[:2], workers=2, job=slow_job, progress=seen.append)
    finally:
        rewards.close_pool()
    assert [len(o.scores) for o in opts] == [2, 2]
    running = [s for s in seen if s["phase"] == "running"]
    assert len(running) >= 3 and running[-1]["fights"] >= 2        # fed every FEED_S, not per fight


def test_live_view_merges_the_valuation_and_measures_its_cost(tmp_path):
    lv = LiveView()
    lv.start(tmp_path, "run-v", "0.23")
    lv.valuation({"layout": "tile_choice", "phase": "starting", "offers": OFFERS}, reset=True)
    lv.valuation({"phase": "running", "options": [{"name": "skip", "seeds": 3}]})
    lv.valuation({"phase": "decided", "choice": "take:katana", "click": "the Katana pedestal (left)"})
    doc = json.loads((tmp_path / LIVE_FILE).read_text())
    v = doc["valuation"]
    assert doc["status"] == "reward" and v["layout"] == "tile_choice" and v["click"].startswith("the Katana")
    assert v["options"][0]["seeds"] == 3 and lv.val_writes == 3 and lv.val_s > 0
    lv.valuation({"layout": "shrine", "phase": "starting"}, reset=True)
    assert "click" not in json.loads((tmp_path / LIVE_FILE).read_text())["valuation"]


def test_bot_shows_the_valuation_the_choice_and_the_click(tmp_path, monkeypatch):
    from test_bot_flow import FakeGW, frame, make_bot
    from shogunbot.bridge import Tracker
    reward = frame("runs/20261028-061420/frames/f0034_CLICK.png")   # Back Charge | Swirl
    after = frame("collect/20261027-164727/f0043_idle.png")
    monkeypatch.setattr(rewards, "evaluate", here({"swirl#2": noisy(3.0, 0.5)}))
    monkeypatch.setattr("time.sleep", lambda s: None)
    gw = FakeGW([(reward, [(1080, 408, 1184, 516)]), (after, [])])
    b = make_bot(tmp_path, gw)
    b.live = LiveView()
    b.live.start(tmp_path, "run-b", "0.23")
    assert b._reward(reward, Tracker(), "bamboo-1", 9) == "taken"
    v = json.loads((tmp_path / LIVE_FILE).read_text())["valuation"]
    assert v["phase"] == "done" and v["outcome"] == "taken" and v["choice"] == "take:swirl"
    assert v["click"] == "the Swirl pedestal (right)" and v["offers"][0]["tile"] == "back_charge"
    assert v["kit"] == ["swirl 2/3", "arrow 2/5"] and v["options"][0]["seeds"] == 24
    ev = [json.loads(l) for l in open(b.log.run_dir / "events.jsonl", encoding="utf-8")]
    dec = next(e for e in ev if e["event"] == "reward_decision")
    assert dec["fights"] == 72 and dec["seeds"] == 24 and dec["seeds_max"] == 48 and dec["feeds"] >= 2


def test_a_recorded_fight_plays_the_same_and_keeps_every_turn():
    # the viewer's replay [user 2026-10-29: watch the fights]: recording changes nothing
    from shogunbot.selfplay import run_once
    kit = Kit.of("swirl", "arrow")
    trace = []
    plain = run_once(9001, 1, kit=kit, scenario="stress", start="bamboo-2")
    traced = run_once(9001, 1, kit=kit, scenario="stress", start="bamboo-2", trace=trace)
    assert plain == traced
    ends = [r for r in trace if r[4] in ("cleared", "died", "stalled")]
    assert len(trace) == traced.turns + len(ends) and ends[-1][4] == ("cleared" if traced.win else ends[-1][4])
    level, wave, board, hp, action, telegraphs, queue = trace[0]
    assert level == "stress-bamboo-2" and wave == 1 and "H" in board and hp == 10 and action


def test_the_example_fight_carries_its_turns_and_the_console_line_reads():
    def job(args):
        f = fake_job({"katana": noisy(3.0, 0.5)}, args)
        return f + ([["stress-bamboo-2", 1, ". H> <A3", 10, "wait", "", ""]],)
    seen = []
    opts = rewards.tile_choice_options(DEFAULT_KIT, OFFERS)
    rewards.evaluate(opts, 10, workers=1, job=job, progress=seen.append)
    last = seen[-1]
    rows = {r["name"]: r for r in last["options"]}
    assert rows["take:katana"]["example"]["trace"][0][2] == ". H> <A3"
    line = rewards.progress_line(last)
    assert line.startswith(f"{last['elapsed']:.0f}s, {last['fights']} fights: skip 24/48")
    assert "take:katana 24/48" in line and "take:spear 24/48" in line and "stopped" in line
