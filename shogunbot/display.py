"""Where the game window goes: DXGI capture reads the primary display only, so a game window on
another monitor is put back onto it [runs 20261029-045632, -045723 and -124858 stopped at the start
with 'capture region (-2559, 0, 1278, 1399) is outside the primary output (2560x1440)'; run
20261029-153952 stopped the same way at turn 14 of Bamboo Grove I, the window having left the
primary display during the run; user reviews of runs 20261029-124921 and -154200: move it back
instead of erroring]. The window found off the primary display had also changed size: a 1278 x 1399
client area is the left half of a 2560 x 1440 monitor, where the game draws its 16:9 picture with
bars that perception cannot read. So the window goes back at its size too. No Win32 here (testable
off Windows): winio.GameWindow makes the calls."""

from __future__ import annotations

import time

# The game's client area on the user's PC: 1920 x 1080 at (444, 77) on the 2560 x 1440 primary
# display [window_info 2026-10-29]; every recorded frame has this size.
GAME_CLIENT = (1920, 1080)
WAIT_OFF_PRIMARY_S = 300.0   # how long a capture waits for a window that could not be moved back
POLL_S, TRIES_EVERY = 2.0, 5  # while waiting: look every 2 s, try moving it again every 10 s


class NotOnPrimary(RuntimeError):
    pass


def on_primary(client: tuple[int, int, int, int], primary: tuple[int, int]) -> bool:
    """client: the client area (x, y, width, height) in screen px; primary: the primary display's
    (width, height), whose top-left is the screen origin."""
    x, y, w, h = client
    pw, ph = primary
    return x >= 0 and y >= 0 and x + w <= pw and y + h <= ph


def home_rect(client: tuple[int, int, int, int], outer: tuple[int, int, int, int],
              primary: tuple[int, int], home: tuple[int, int, int, int] | None = None
              ) -> tuple[int, int, int, int]:
    """The window rect (x, y, width, height) for SetWindowPos that puts the client area back on the
    primary display: where it was last seen there (`home`, a client rect) when that is known,
    else at the game's size (GAME_CLIENT, or the display when smaller) in the middle of the display.
    client: the client area now; outer: the window rect now (left, top, right, bottom), borders and
    title bar included, which give the frame's size."""
    x, y, w, h = client
    left, top = x - outer[0], y - outer[1]            # left border; title bar and top border
    right, bottom = outer[2] - (x + w), outer[3] - (y + h)
    pw, ph = primary
    if home is not None and on_primary(home, primary):
        hx, hy, hw, hh = home
    else:
        hw, hh = min(GAME_CLIENT[0], pw), min(GAME_CLIENT[1], ph)
        hx, hy = (pw - hw) // 2, (ph - hh) // 2
    return hx - left, hy - top, hw + left + right, hh + top + bottom


def placed(rect, move_home, primary, notify=None, should_stop=None, wait_s: float = WAIT_OFF_PRIMARY_S,
           sleep=time.sleep, clock=time.monotonic) -> tuple[int, int, int, int]:
    """The game's client rect, on the primary display; checked before every capture. A window that has
    left the display [run 20261029-153952, turn 14] is put back (move_home) and the move logged
    (notify 'window_moved'); when that fails, the capture waits up to wait_s for the window to come
    back, trying again every 10 s (notify 'window_move_failed' once, 'window_back' when it is back),
    and F8 still stops the run (should_stop). Only then NotOnPrimary, the old error.
    rect(), primary(): the client rect and the display size now; move_home(): moves the window and
    returns its client rect."""
    def tell(event, **fields):
        if notify is not None:
            try:
                notify(event, **fields)
            except Exception:
                pass

    r = rect()
    if on_primary(r, primary()):
        return r
    t_end, told = clock() + wait_s, False
    while True:
        try:
            after = move_home()
            if on_primary(after, primary()):
                tell("window_moved", before=list(r), after=list(after))
                return after
            err = f"moved, but the client area is at {after}"
        except Exception as e:
            err = repr(e)
        if not told:
            tell("window_move_failed", before=list(r), error=err, waiting_s=wait_s)
            told = True
        for _ in range(TRIES_EVERY):
            if should_stop is not None and should_stop():
                raise KeyboardInterrupt("stop requested while the game window was off the primary display")
            if clock() > t_end:
                raise NotOnPrimary(f"the game window stayed off the primary display {primary()} for "
                                   f"{wait_s:.0f} s (last at {rect()}); move it there and start again")
            sleep(POLL_S)
            cur = rect()
            if on_primary(cur, primary()):
                tell("window_back", rect=list(cur))
                return cur
