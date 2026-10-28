"""Windows glue: find the game window, capture its client area, send input.

Reuses local_game_mcp (installed in the same venv) for DXGI capture and
scan-code SendInput, so the bot sees and presses exactly what the MCP did.
"""

from __future__ import annotations

import time

import numpy as np

from local_game_mcp import windowing
from local_game_mcp.backends import sendinput
from local_game_mcp.capture import CaptureManager

TITLE_REGEX = r"(?i)shogun"
HOLD_S = 0.06


class GameNotFound(RuntimeError):
    pass


class GameWindow:
    def __init__(self, title_regex: str = TITLE_REGEX):
        sendinput.set_dpi_awareness()
        self.title_regex = title_regex
        self.cap = CaptureManager()

    # ---- window ---------------------------------------------------------
    def hwnd(self) -> int:
        h = windowing.find_window(self.title_regex)
        if not h:
            raise GameNotFound("Shogun Showdown window not found; is the game running?")
        return h

    def rect(self) -> tuple[int, int, int, int]:
        return windowing.client_rect_screen(self.hwnd())

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
        """Client-area frame as RGB uint8 (H, W, 3)."""
        bgra = self.cap.grab_region(self.rect())
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
