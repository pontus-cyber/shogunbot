"""A game window off the primary display, the only one DXGI capture reads, is put back there: at the
start of a run since 0.28 [runs 20261029-045632, -045723, -124858: 'capture region (-2559, 0, 1278,
1399) is outside the primary output (2560x1440)'; user review of run 20261029-124921], and before
every capture since 0.29, at its former size [run 20261029-153952: the window left the display at
turn 14 of Bamboo Grove I and the run stopped; user review of run 20261029-154200]."""

import pytest

from shogunbot.display import GAME_CLIENT, NotOnPrimary, home_rect, on_primary, placed

PRIMARY = (2560, 1440)
OFF = (-2559, 0, 1278, 1399)               # the client area of runs 124858 and 153952
OFF_OUTER = (-2567, -31, -1273, 1407)      # its window: 8 px borders and a 31 px title bar
HOME = (444, 77, 1920, 1080)               # where the game sits on the user's PC [window_info 2026-10-29]


def test_a_window_on_the_primary_display_stays():
    assert on_primary((0, 0, 2560, 1440), PRIMARY)
    assert on_primary(HOME, PRIMARY)
    assert not on_primary(OFF, PRIMARY)


def test_the_window_goes_back_where_it_was_at_its_size():
    x, y, w, h = home_rect(OFF, OFF_OUTER, PRIMARY, HOME)
    assert (x, y, w, h) == (444 - 8, 77 - 31, 1920 + 16, 1080 + 39)


def test_without_a_former_place_it_goes_to_the_middle_at_the_games_size():
    # run 124858 started with the window already off the display: no place on it was seen yet
    x, y, w, h = home_rect(OFF, OFF_OUTER, PRIMARY)
    client = (x + 8, y + 31, w - 16, h - 39)
    assert client[2:] == GAME_CLIENT and on_primary(client, PRIMARY)
    assert client[:2] == ((2560 - 1920) // 2, (1440 - 1080) // 2)


class Win:
    """A window that leaves the display and comes back when moved (or not)."""
    def __init__(self, movable=True, back_after=None):
        self.r, self.movable, self.back_after, self.moves, self.t = OFF, movable, back_after, 0, 0.0

    def rect(self):
        if self.back_after is not None and self.t >= self.back_after:
            self.r = HOME
        return self.r

    def move_home(self):
        self.moves += 1
        if self.movable:
            self.r = HOME
        return self.r

    def sleep(self, s):
        self.t += s

    def clock(self):
        return self.t


def _placed(w, events, stop=None, wait_s=300.0):
    return placed(w.rect, w.move_home, lambda: PRIMARY, lambda e, **f: events.append((e, f)), stop,
                  wait_s=wait_s, sleep=w.sleep, clock=w.clock)


def test_a_window_that_left_mid_run_is_moved_back_and_logged():
    w, events = Win(), []
    assert _placed(w, events) == HOME
    assert w.moves == 1 and events == [("window_moved", {"before": list(OFF), "after": list(HOME)})]
    events.clear()
    assert _placed(w, events) == HOME and events == []          # on the display: nothing to do


def test_a_window_that_cannot_be_moved_is_waited_for():
    w, events = Win(movable=False, back_after=7.0), []
    assert _placed(w, events) == HOME
    assert [e for e, _ in events] == ["window_move_failed", "window_back"]
    assert w.t == 8.0                                           # looked every 2 s


def test_the_wait_ends_with_the_old_error_or_on_f8():
    w, events = Win(movable=False), []
    with pytest.raises(NotOnPrimary):
        _placed(w, events, wait_s=30.0)
    assert w.moves == 4                                         # tried again every 10 s
    with pytest.raises(KeyboardInterrupt):
        _placed(Win(movable=False), [], stop=lambda: True)
