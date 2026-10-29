"""Where the game window goes: DXGI capture reads the primary display only, so a game window on
another monitor is moved onto it before the run [runs 20261029-045632, -045723 and -124858 stopped
with 'capture region (-2559, 0, 1278, 1399) is outside the primary output (2560x1440)'; user review
of run 20261029-124921: move it instead of erroring]. Pure arithmetic; winio.GameWindow does the
Win32 calls."""

from __future__ import annotations


def on_primary(client: tuple[int, int, int, int], primary: tuple[int, int]) -> bool:
    """client: the client area (x, y, width, height) in screen px; primary: the primary display's
    (width, height), whose top-left is the screen origin."""
    x, y, w, h = client
    pw, ph = primary
    return x >= 0 and y >= 0 and x + w <= pw and y + h <= ph


def move_to(client: tuple[int, int, int, int], outer: tuple[int, int, int, int],
            primary: tuple[int, int]) -> tuple[int, int] | None:
    """The window's new top-left (screen px) that puts its client area on the primary display, or
    None when it is there already. outer: the window rect (left, top, right, bottom), borders and
    title bar included. The whole window goes to the origin when it fits; otherwise the client area
    does, with the borders off screen."""
    if on_primary(client, primary):
        return None
    x, y, w, h = client
    bx, by = x - outer[0], y - outer[1]          # the client area's offset inside the window
    ow, oh = outer[2] - outer[0], outer[3] - outer[1]
    pw, ph = primary
    if ow <= pw and oh <= ph:
        return 0, 0
    return -bx, -by
