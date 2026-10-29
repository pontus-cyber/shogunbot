"""0.18: series of runs. A won run is given up from the pause menu, which leads back to
the camp [run 20261028-130456]; a run given up in a fight, or a death, leads to the death
screen, where the next run clicks RESTART [user 2026-10-28]. Frames come from
collect/20261028-121900-giveup (captured while the path was tested by hand) and the
bot's own runs; tests are skipped when the frames are missing.
"""

import json

import pytest

from test_bot_flow import FakeGW, Rec, frame, make_bot

GIVEUP = "collect/20261028-121900-giveup/"
SERIES = "runs/20261028-130456/frames/"   # 0.18.2 giving up its won run from the Stage 2 map
PAUSE_BOX = (1784, 972, 1855, 1035)      # the PAUSE MENU disc
GIVE_UP_ROW = (858, 612, 1057, 652)      # third of the five pause menu rows
YES_ROW = (810, 532, 1105, 572)          # second of the two confirmation rows
RESTART_BOX = (1472, 878, 1808, 942)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_detectors_tell_the_screens_apart():
    from shogunbot.perception import screens as S
    shots = {
        "combat": frame(GIVEUP + "f0004_new_run.png"),
        "map": frame("runs/20261028-104830/frames/map_1.png"),           # HUD dimmed on the map
        "shrine": frame("runs/20261028-104830/frames/f0112_CLICK.png"),
        "pause": frame(GIVEUP + "f0001_pause_menu.png"),
        "confirm": frame(GIVEUP + "f0002_confirm.png"),
        "death_give_up": frame(GIVEUP + "f0003_death_after_give_up.png"),
        "death_slain": frame("runs/20261027-182223/frames/death.png"),    # a real death
        "death_fading_in": frame("runs/20261028-053637/frames/death.png"),  # no buttons yet
        # the bot's own frames from its first give-up [run 20261028-130456]
        "stage2_map": frame(SERIES + "f0100_CLICK.png"),
        "pause_live": frame(SERIES + "f0101_CLICK.png"),
        "confirm_live": frame(SERIES + "f0102_CLICK.png"),
        "camp_after_give_up": frame(SERIES + "give_up_failed.png"),
    }
    assert {k for k, f in shots.items() if S.pause_button(f)} == {"combat", "map", "shrine", "stage2_map"}
    menus = {k: S.menu(f) for k, f in shots.items()}
    assert {k for k, m in menus.items() if m} == {"pause", "confirm", "pause_live", "confirm_live"}
    assert menus["pause_live"][0] == "pause" and menus["confirm_live"][0] == "confirm"
    assert S.classify(shots["camp_after_give_up"]) == "camp"
    px = lambda rows: [(round(x * 1920), round(y * 1080)) for x, y in rows]
    assert menus["pause"][0] == "pause" and px(menus["pause"][1]) == \
        [(958, 472), (958, 552), (958, 632), (958, 712), (958, 792)]
    assert menus["confirm"][0] == "confirm" and px(menus["confirm"][1]) == [(958, 472), (958, 552)]
    restart = {k: S.find_restart(f) for k, f in shots.items()}
    assert {k for k, r in restart.items() if r} == {"death_give_up", "death_slain"}
    assert px([restart["death_give_up"]]) == [(1640, 908)] == px([restart["death_slain"]])


def test_won_run_is_given_up_from_the_pause_menu_and_lands_in_camp(tmp_path):
    # [run 20261028-130456] from the Stage 2 map, YES: GIVE UP goes straight to the camp;
    # 0.18.2 waited for a death screen and stopped the series although the give-up worked
    gw = FakeGW([(frame(SERIES + "f0100_CLICK.png"), [PAUSE_BOX]),
                 (frame(SERIES + "f0101_CLICK.png"), [GIVE_UP_ROW]),
                 (frame(SERIES + "f0102_CLICK.png"), [YES_ROW]),
                 (frame(SERIES + "give_up_failed.png"), [])])
    b = make_bot(tmp_path, gw)
    assert b._give_up() is True
    assert gw.clicks == [(1820, 1001), (958, 632), (958, 552)]


def test_run_given_up_in_a_fight_ends_on_the_death_screen(tmp_path, monkeypatch):
    from shogunbot import bot as B
    monkeypatch.setattr(B, "GIVE_UP_WAIT_S", 0.2)   # not a map: the pause click waits half of this
    gw = FakeGW([(frame(GIVEUP + "f0004_new_run.png"), [PAUSE_BOX]),
                 (frame(GIVEUP + "f0001_pause_menu.png"), [GIVE_UP_ROW]),
                 (frame(GIVEUP + "f0002_confirm.png"), [YES_ROW]),
                 (frame(GIVEUP + "f0003_death_after_give_up.png"), [])])
    b = make_bot(tmp_path, gw)
    assert b._give_up() is True
    assert gw.clicks == [(1820, 1001), (958, 632), (958, 552)]


def test_give_up_clicks_no_menu_row_it_has_not_recognised(tmp_path):
    # the pause button 'opens' something that is not the pause menu: no row is clicked
    gw = FakeGW([(frame("runs/20261028-104830/frames/map_1.png"), [PAUSE_BOX]),
                 (frame("runs/20261028-104830/frames/f0112_CLICK.png"), [])])
    b = make_bot(tmp_path, gw)
    assert b._give_up() is False
    assert gw.clicks == [(1820, 1001)] * 3


def test_give_up_stops_when_the_confirmation_does_not_show(tmp_path):
    # GIVE UP clicked twice without the confirmation: stop, never click YES's spot
    gw = FakeGW([(frame("runs/20261028-104830/frames/map_1.png"), [PAUSE_BOX]),
                 (frame(GIVEUP + "f0001_pause_menu.png"), [])])
    b = make_bot(tmp_path, gw)
    assert b._give_up() is False
    assert gw.clicks == [(1820, 1001), (958, 632), (958, 632)]


class _GW(FakeGW):
    def focus(self):
        return True


class _Rec(Rec):
    def start(self, run_dir):
        pass

    def flush(self, *a, **k):
        pass


def test_new_run_clicks_restart_on_the_death_screen(tmp_path, monkeypatch):
    # [run 20261027-182223] the second run of a series ended after 1 s on the death screen
    from shogunbot import bot as B
    monkeypatch.setattr(B, "DEATH_SCREEN_MAX_S", 0.05)  # end the test run on the next death frame
    gw = _GW([(frame("runs/20261027-182223/frames/death.png"), [RESTART_BOX]),
              (frame("runs/20261028-053637/frames/death.png"), [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after = _Rec(), False
    assert b.run() == "stuck"
    assert gw.clicks == [(1640, 908)]


def test_series_continues_after_a_win_only_when_the_give_up_worked(monkeypatch, capsys):
    from shogunbot import bot as B
    script = [("win", True), ("death", None), ("win", False), ("win", True)]
    seen = []

    class FakeBot:
        def __init__(self, frame_policy, autostart):
            self.gave_up, self.restart_after = None, False

        def run(self):
            res, self.gave_up = script[len(seen)]
            seen.append(self.restart_after)
            return res

    monkeypatch.setattr(B, "Bot", FakeBot)
    B.play(runs=4)
    assert seen == [True, True, True]          # the third run won but could not be given up
    assert "could not be given up" in capsys.readouterr().out


def test_one_logger_starts_every_run_of_a_series_from_zero(tmp_path):
    # 0.18 review: start_run kept the last run's stats, so run 2's summary held run 1's turns
    from shogunbot.runlog import RunLogger
    log = RunLogger(str(tmp_path), "none")
    first = log.start_run({"n": 1})
    log.level_start("bamboo-1", 10, 4)
    log.stats.turns = 25
    s1 = log.end_run("win")
    second = log.start_run({"n": 2})
    s2 = log.end_run("death")
    assert second != first and (s1["turns"], s2["turns"]) == (25, 0)
    assert s2["levels"] == [] and s2["minutes"] < 1


# ---- 0.23: the camp's day selector [user 2026-10-29: Day 1 until the elite traits are simulated]
DAY2 = "runs/20261029-045856/frames/"
DAY_LEFT_BOX = (805, 215, 875, 290)
GO_BOX = (1185, 640, 1400, 720)


def _day1(day2):
    """A synthetic Day 1 camp from the recorded Day 2 one (no Day 1 frame exists yet): the left
    arrow dark, the right one lit, no red elite rule."""
    import numpy as np
    a = day2.copy()
    left, right = a[205:300, 800:880], a[205:300, 1045:1130]

    def near(reg, rgb, tol):
        return (np.abs(reg.astype(int) - rgb) < tol).all(axis=2)
    lit_l, body_l = near(left, (188, 143, 106), 18), near(left, (119, 69, 59), 6)
    dim_r, body_r = near(right, (118, 68, 58), 6), near(right, (75, 39, 45), 6)
    left[lit_l], left[body_l] = (118, 68, 58), (75, 39, 45)
    right[dim_r], right[body_r] = (188, 143, 106), (119, 69, 59)
    rules = a[230:285, 150:690]
    rules[(rules[..., 0] > 150) & (rules[..., 1] < 110)] = (40, 20, 25)
    return a


def test_day_selector_reads_the_recorded_day2_camp():
    from shogunbot.perception import screens as S
    sel = S.day_selector(frame(DAY2 + "f0007_CLICK.png"))
    assert (sel["left"], sel["right"], sel["elite"]) == (True, False, True)
    assert S.day_selector(frame(DAY2 + "f0001_D.png")) is None          # hero not at GO yet
    one = S.day_selector(_day1(frame(DAY2 + "f0007_CLICK.png")))
    assert (one["left"], one["right"], one["elite"]) == (False, True, False)


def test_camp_steps_down_to_day1_before_go(tmp_path, monkeypatch):
    from shogunbot import bot as B
    day2 = frame(DAY2 + "f0007_CLICK.png")
    gw = _GW([(day2, [DAY_LEFT_BOX]), (_day1(day2), [GO_BOX]), (frame(DAY2 + "f0009_CLICK.png"), [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after = _Rec(), False
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 12
    monkeypatch.setattr(B, "_stop_requested", stop)
    monkeypatch.setattr("time.sleep", lambda s: None)
    b.run()
    assert gw.clicks[0] == (841, 252)                     # the day arrow first
    assert (1315, 677) in gw.clicks and gw.i == 2          # then GO, onto the map
    ev = [json.loads(l) for l in open(b.log.run_dir / "events.jsonl", encoding="utf-8")]
    assert any(e["event"] == "day_select" and e["clicks"] == 1 for e in ev)


def test_camp_stops_when_the_day_selector_does_not_react(tmp_path, monkeypatch):
    from shogunbot import bot as B
    gw = _GW([(frame(DAY2 + "f0007_CLICK.png"), [])])     # clicks change nothing
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after = _Rec(), False
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert b.run() == "stuck"
    assert gw.clicks == [(841, 252)] * B.DAY_CLICKS_MAX     # never GO on Day 2
