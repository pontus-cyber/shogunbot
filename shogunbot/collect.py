"""Frame collector: saves one frame per player input, labelled with the input.

Polls the keyboard and mouse (GetAsyncKeyState, no extra packages). On each
press while the game is the foreground window it saves the frame from the
moment just before the press, i.e. the board the player decided on, and
logs the action. Every logged line is a (state, action) pair, which is the
training and validation data the bot needs.

Also saves a frame when the screen settles after a long pause (level
banners, reward screens) so non-combat screens are covered.

    python -m shogunbot collect            # Ctrl+C to stop

Output: collect/<timestamp>/
    f0001_D.png ...      frame before the input, input in the name
    actions.jsonl        {"frame", "t", "action", "click": [fx, fy]}
"""

from __future__ import annotations

import ctypes
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .winio import GameWindow

user32 = ctypes.WinDLL("user32", use_last_error=True)

# virtual-key code -> label
KEYS = {
    0x41: "A", 0x44: "D", 0x57: "W", 0x53: "S",
    0x20: "SPACE", 0x0D: "ENTER", 0x1B: "ESC",
    0x25: "LEFT", 0x27: "RIGHT", 0x26: "UP", 0x28: "DOWN",
    0x51: "Q", 0x45: "E", 0x31: "1", 0x32: "2", 0x33: "3",
    0x01: "CLICK",
}


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _pressed(vk: int) -> bool:
    """True if the key went down since the last call (low bit), so a quick
    tap during a slow capture or save is not missed."""
    st = user32.GetAsyncKeyState(vk)
    return bool(st & 0x0001)


def _cursor() -> tuple[int, int]:
    p = _POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def run(out_root: str = "collect", poll_s: float = 0.015, refresh_s: float = 0.05,
        settle_s: float = 1.5, min_change: float = 2.0) -> None:
    gw = GameWindow()
    out = Path(out_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "actions.jsonl", "a", encoding="utf-8")
    print(f"recording to {out}  (one frame per keypress/click; Ctrl+C to stop)")

    for vk in KEYS:
        _pressed(vk)  # clear stale press bits
    latest = None          # most recent frame = pre-input snapshot
    latest_t = 0.0
    last_saved = None
    last_input_t = time.time()
    n = 0
    warned = False
    t0 = time.time()

    jobs: queue.Queue = queue.Queue()

    def _writer() -> None:
        while True:
            item = jobs.get()
            if item is None:
                return
            arr, path = item
            Image.fromarray(arr).save(path, compress_level=1)

    threading.Thread(target=_writer, daemon=True).start()

    def save(frame, tag: str, extra: dict) -> None:
        nonlocal n, last_saved
        n += 1
        name = f"f{n:04d}_{tag}.png"
        jobs.put((frame, out / name))
        last_saved = frame
        log.write(json.dumps({"frame": name, "t": round(time.time() - t0, 2), **extra}) + "\n")
        log.flush()
        print(name)

    try:
        while True:
            now = time.time()
            try:
                fg = gw.is_foreground()
            except Exception:
                fg = False
            if not fg:
                if not warned:
                    print("waiting for Shogun Showdown to be the active window on the primary monitor")
                    warned = True
                for vk in KEYS:
                    _pressed(vk)
                latest = None
                time.sleep(0.3)
                continue
            if warned:
                print("game active, recording")
                warned = False

            if now - latest_t >= refresh_s:
                try:
                    latest = gw.grab()
                    latest_t = now
                except Exception:
                    latest = None

            for vk, label in KEYS.items():
                if _pressed(vk) and latest is not None:
                    extra = {"action": label}
                    if label == "CLICK":
                        x, y, w, h = gw.rect()
                        cx, cy = _cursor()
                        extra["click"] = [round((cx - x) / w, 4), round((cy - y) / h, 4)]
                    save(latest, label, extra)
                    last_input_t = now

            if latest is not None and now - last_input_t > settle_s:
                if last_saved is None or float(np.mean(np.abs(
                        latest.astype(np.int16) - last_saved.astype(np.int16)))) >= min_change:
                    save(latest, "idle", {"action": None})
                last_input_t = now

            time.sleep(poll_s)
    except KeyboardInterrupt:
        jobs.put(None)
        time.sleep(1.0)
        log.close()
        print(f"stopped; {n} frames in {out}")
