"""Windows glue: find the game window, capture its client area, send input.

Reuses local_game_mcp (installed in the same venv) for DXGI capture and
scan-code SendInput, so the bot sees and presses exactly what the MCP did.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np

from local_game_mcp import windowing
from local_game_mcp.backends import sendinput
from local_game_mcp.capture import CaptureManager

from .display import NotOnPrimary, home_rect, on_primary, placed

TITLE_REGEX = r"(?i)shogun"
HOLD_S = 0.06
SWP_NOZORDER, SWP_NOACTIVATE = 0x0004, 0x0010
SW_RESTORE = 9


class GameNotFound(RuntimeError):
    pass


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


_U32 = None


def _user32():
    """A user32 of our own with declared argument types (window handles are pointer-sized), so the
    types set here never touch local_game_mcp's calls."""
    global _U32
    if _U32 is None:
        from ctypes import wintypes
        u = ctypes.WinDLL("user32", use_last_error=True)
        u.GetSystemMetrics.argtypes, u.GetSystemMetrics.restype = [ctypes.c_int], ctypes.c_int
        u.GetWindowRect.argtypes, u.GetWindowRect.restype = [wintypes.HWND, ctypes.POINTER(_RECT)], wintypes.BOOL
        u.IsZoomed.argtypes, u.IsZoomed.restype = [wintypes.HWND], wintypes.BOOL
        u.IsIconic.argtypes, u.IsIconic.restype = [wintypes.HWND], wintypes.BOOL
        u.ShowWindow.argtypes, u.ShowWindow.restype = [wintypes.HWND, ctypes.c_int], wintypes.BOOL
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_uint]
        u.SetWindowPos.restype = wintypes.BOOL
        _U32 = u
    return _U32


def primary_size() -> tuple[int, int]:
    """The primary display in physical px (the process is DPI-aware): SM_CXSCREEN, SM_CYSCREEN."""
    u = _user32()
    return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))


def outer_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = _RECT()
    _user32().GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def cursor_pos() -> tuple[int, int] | None:
    """The cursor's screen position (physical pixels: the process is DPI-aware), or None."""
    try:
        pt = _POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)
    except Exception:
        pass
    return None


class GameWindow:
    def __init__(self, title_regex: str = TITLE_REGEX):
        sendinput.set_dpi_awareness()
        self.title_regex = title_regex
        self.cap = CaptureManager()
        self.last_click = None    # (target x, y, cursor x, y) of the last click, screen pixels
        self.home = None          # the client rect last seen on the primary display
        self.notify = None        # callback(event, **fields): the bot logs window moves with it
        self.should_stop = None   # callback: True when the user asks to stop (F8) during a wait

    # ---- window ---------------------------------------------------------
    def hwnd(self) -> int:
        h = windowing.find_window(self.title_regex)
        if not h:
            raise GameNotFound("Shogun Showdown window not found; is the game running?")
        return h

    def rect(self) -> tuple[int, int, int, int]:
        return windowing.client_rect_screen(self.hwnd())

    def _move_home(self) -> tuple[int, int, int, int]:
        """Put the window back on the primary display where its client area was last seen there,
        at that size (display.home_rect); a maximised or minimised window is restored first.
        Returns the client rect after the move."""
        h = self.hwnd()
        u = _user32()
        if u.IsZoomed(h) or u.IsIconic(h):
            u.ShowWindow(h, SW_RESTORE)
            time.sleep(0.4)
        x, y, w, hh = home_rect(self.rect(), outer_rect(h), primary_size(), self.home)
        u.SetWindowPos(h, 0, x, y, w, hh, SWP_NOZORDER | SWP_NOACTIVATE)
        time.sleep(1.0)                # let the game redraw at its new place
        return self.rect()

    def ensure_on_primary(self) -> tuple | None:
        """At the start of a run (and of `collect`): move the window onto the primary display when
        its client area is elsewhere. Returns (from, to) client rects when it moved, None when it
        was there already; NotOnPrimary when moving failed."""
        before = self.rect()
        if on_primary(before, primary_size()):
            self.home = before
            return None
        after = self._move_home()
        if not on_primary(after, primary_size()):
            raise NotOnPrimary(f"the game window is at {after}, not on the primary display {primary_size()}")
        self.home = after
        return before, after

    def placed_rect(self) -> tuple[int, int, int, int]:
        """The client rect, on the primary display (display.placed): checked before every capture."""
        r = placed(self.rect, self._move_home, primary_size, self.notify, self.should_stop)
        self.home = r
        return r

    def focus(self) -> bool:
        h = self.hwnd()
        info = windowing.get_info(h)
        if info.elevated:
            raise RuntimeError("game runs elevated; Windows drops synthetic input (UIPI)")
        return windowing.focus(h)

    def is_foreground(self) -> bool:
        return windowing.foreground_hwnd() == self.hwnd()

    # ---- capture --------------------------------------------------------
    def grab(self) -> np.ndarray:
        """Client-area frame as RGB uint8 (H, W, 3), with the window on the primary display."""
        bgra = self.cap.grab_region(self.placed_rect())
        return np.ascontiguousarray(bgra[..., 2::-1])

    def grab_stable(self, settle_s: float = 0.25, timeout_s: float = 4.0, tol: float = 1.5) -> np.ndarray:
        """Wait until two frames settle_s apart differ by < tol mean abs (animations done)."""
        t_end = time.time() + timeout_s
        prev = self.grab()
        while True:
            time.sleep(settle_s)
            cur = self.grab()
            if float(np.mean(np.abs(cur.astype(np.int16) - prev.astype(np.int16)))) < tol:
                return cur
            if time.time() > t_end:
                return cur
            prev = cur

    # ---- input ----------------------------------------------------------
    def _ensure_focus(self) -> None:
        if not self.is_foreground():
            self.focus()
            time.sleep(0.1)

    def key(self, name: str) -> None:
        self._ensure_focus()
        sendinput.send(sendinput.key_input(name, True))
        time.sleep(HOLD_S)
        sendinput.send(sendinput.key_input(name, False))

    def click(self, fx: float, fy: float) -> None:
        """Click at a fraction (0-1) of the client area."""
        self._ensure_focus()
        x, y, w, h = self.rect()
        sx, sy = x + int(fx * w), y + int(fy * h)
        # Unity UI needs a hover before the press: move, nudge, settle, then click
        sendinput.send(sendinput.mouse_move_input(sx - 3, sy))
        time.sleep(0.12)
        sendinput.send(sendinput.mouse_move_input(sx, sy))
        time.sleep(0.15)
        # where the cursor really is when the button goes down [user review of run
        # 20261029-072042: clicks where the cursor did not seem to move]
        cur = cursor_pos()
        self.last_click = (sx, sy) + (cur if cur is not None else (None, None))
        sendinput.send(sendinput.mouse_button_input("left", True))
        time.sleep(0.1)
        sendinput.send(sendinput.mouse_button_input("left", False))
        # stay on the target: the game reads the cursor once per frame, and a
        # release seen at another position cancels the click [obs 2026-10-27]
        time.sleep(0.4)

    def move_mouse_away(self) -> None:
        """Park the cursor so hover tooltips do not cover the board."""
        time.sleep(0.2)
        x, y, w, h = self.rect()
        sendinput.send(sendinput.mouse_move_input(x + w - 5, y + 5))
