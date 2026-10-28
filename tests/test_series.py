"""0.18: series of runs. A won run is given up from the pause menu and the next run
clicks RESTART on the death screen [user 2026-10-28]. Frames come from
collect/20261028-121900-giveup (captured while the path was tested by hand) and the
bot's own runs; tests are skipped when the frames are missing.
"""

import pytest

from test_bot_flow import FakeGW, Rec, frame, make_bot

GIVEUP = "collect/20261028-121900-giveup/"
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
    }
    assert {k for k, f in shots.items() if S.pause_button(f)} == {"combat", "map", "shrine"}
    menus = {k: S.menu(f) for k, f in shots.items()}
    assert {k for k, m in menus.items() if m} == {"pause", "confirm"}
    px = lambda rows: [(round(x * 1920), round(y * 1080)) for x, y in rows]
    assert menus["pause"][0] == "pause" and px(menus["pause"][1]) == \
        [(958, 472), (958, 552), (958, 632), (958, 712), (958, 792)]
    assert menus["confirm"][0] == "confirm" and px(menus["confirm"][1]) == [(958, 472), (958, 552)]
    restart = {k: S.find_restart(f) for k, f in shots.items()}
    assert {k for k, r in restart.items() if r} == {"death_give_up", "death_slain"}
    assert px([restart["death_give_up"]]) == [(1640, 908)] == px([restart["death_slain"]])


def test_won_run_is_given_up_from_the_pause_menu(tmp_path):
    gw = FakeGW([(frame("runs/20261028-104830/frames/map_1.png"), [PAUSE_BOX]),
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
