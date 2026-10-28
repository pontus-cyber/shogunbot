"""M2: the bot's reward flow against recorded frames, with a fake game window.

The window serves recorded frames and switches to the 'after' frame when the
bot clicks where the game would react. Skipped when frames are missing.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

DATA = Path(os.environ.get("SHOGUNBOT_DATA", Path(__file__).resolve().parents[1]))


def frame(rel):
    p = DATA / rel
    if not p.exists():
        pytest.skip(f"recorded frame not available: {rel}")
    from PIL import Image
    return np.array(Image.open(p).convert("RGB"))


def _stub_windows_modules():
    """bot.py imports winio, which needs the Windows-only local_game_mcp."""
    if "local_game_mcp" in sys.modules:
        return
    pkg = types.ModuleType("local_game_mcp")
    win = types.ModuleType("local_game_mcp.windowing")
    cap = types.ModuleType("local_game_mcp.capture")
    back = types.ModuleType("local_game_mcp.backends")
    si = types.ModuleType("local_game_mcp.backends.sendinput")
    win.find_window = lambda rx: 1
    cap.CaptureManager = object
    for name in ("set_dpi_awareness", "send", "key_input", "mouse_move_input", "mouse_button_input"):
        setattr(si, name, lambda *a, **k: None)
    back.sendinput = si
    sys.modules.update({"local_game_mcp": pkg, "local_game_mcp.windowing": win, "local_game_mcp.capture": cap,
                        "local_game_mcp.backends": back, "local_game_mcp.backends.sendinput": si})


class FakeGW:
    """Serves `screens[i]`; a click inside a target box moves to the next screen."""
    def __init__(self, steps):
        self.steps = steps          # [(frame, [(x0, y0, x1, y1) boxes that advance]), ...]
        self.i = 0
        self.clicks = []

    def grab(self):
        return self.steps[self.i][0]

    grab_stable = lambda self, **kw: self.grab()

    def click(self, fx, fy):
        x, y = fx * 1920, fy * 1080
        self.clicks.append((round(x), round(y)))
        for x0, y0, x1, y1 in self.steps[self.i][1]:
            if x0 <= x <= x1 and y0 <= y <= y1 and self.i + 1 < len(self.steps):
                self.i += 1
                return

    def move_mouse_away(self):
        pass

    def key(self, name):
        pass


class Rec:
    def record(self, *a, **k):
        pass


def make_bot(tmp_path, gw):
    _stub_windows_modules()
    from shogunbot import bot as B
    from shogunbot.runlog import RunLogger
    b = B.Bot.__new__(B.Bot)
    b.gw, b.rec, b.verbose, b.autostart = gw, Rec(), False, True
    b.log = RunLogger(str(tmp_path), "none")
    b.log.start_run({"test": True})
    b.reward_n = 0
    return b


@pytest.fixture(autouse=True)
def fast_valuation(monkeypatch):
    from shogunbot import rewards
    monkeypatch.setattr(rewards, "SEEDS", (1, 2))
    orig = rewards.decide
    monkeypatch.setattr(rewards, "decide", lambda opts, level, hp, **kw: orig(opts, level, hp, seeds=(1, 2),
                                                                             depth=1, workers=1))
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_tile_choice_takes_the_chosen_tile(tmp_path, monkeypatch):
    from shogunbot import rewards
    from shogunbot.bridge import Tracker
    reward = frame("runs/20261028-061420/frames/f0034_CLICK.png")   # Back Charge | Swirl
    after = frame("collect/20261027-164727/f0043_idle.png")          # transition
    # force the choice so the test does not depend on simulated values
    orig = rewards.decide

    def pick_swirl(opts, level, hp, **kw):
        best, opts, secs = orig(opts, level, hp, seeds=(1,), depth=1, workers=1)
        return next(o for o in opts if o.name == "take:swirl"), opts, secs
    monkeypatch.setattr(rewards, "decide", pick_swirl)
    gw = FakeGW([(reward, [(1080, 408, 1184, 516)]), (after, [])])
    b = make_bot(tmp_path, gw)
    tr = Tracker()
    assert b._reward(reward, tr, "bamboo-1", 9) == "taken"
    assert gw.clicks[0] == (1132, 462)                       # the right pedestal
    assert tr.kit.labels == ("swirl", "arrow", "swirl#2") and tr.kit["swirl#2"].levels == 3


def test_shrine_puts_the_tile_in_the_statue_then_upgrades(tmp_path, monkeypatch):
    from shogunbot import rewards
    from shogunbot.bridge import Tracker
    from shogunbot.model import Kit
    shrine = frame("collect/20261027-164727/f0109_idle.png")       # -2 COOLDOWN; hand spear, swirl, arrow
    lit = frame("collect/20261027-164727/f0111_CLICK.png")         # swirl in the statue, UPGRADE lit
    done = frame("collect/20261027-164727/f0112_idle.png")         # tile pops out
    orig = rewards.decide

    def pick_swirl(opts, level, hp, **kw):
        best, opts, secs = orig(opts, level, hp, seeds=(1,), depth=1, workers=1)
        return next(o for o in opts if o.name == "upgrade:swirl"), opts, secs
    monkeypatch.setattr(rewards, "decide", pick_swirl)
    gw = FakeGW([(shrine, [(908, 900, 1012, 1008)]), (lit, [(800, 718, 1130, 786)]), (done, [])])
    b = make_bot(tmp_path, gw)
    tr = Tracker(Kit.of("swirl", "arrow", "spear"))
    assert b._reward(shrine, tr, "bamboo-2", 10) == "taken"
    assert gw.clicks[:2] == [(960, 954), (962, 752)]          # swirl in the hand, then UPGRADE
    assert (tr.kit["swirl"].cooldown, tr.kit["swirl"].used) == (1, 1)


def test_unreadable_screen_skips(tmp_path):
    from shogunbot.bridge import Tracker
    shrine = frame("runs/20261028-061420/frames/f0089_CLICK.png")
    gw = FakeGW([(shrine, [])])
    b = make_bot(tmp_path, gw)
    from shogunbot.perception import reward_screen
    rs = reward_screen.read(shrine)
    rs.upgrade = None
    b._read_reward = lambda tr: (shrine, rs)
    assert b._reward(shrine, Tracker(), "bamboo-2", 10) == "skipped"
    assert gw.clicks == [(1444, 749)]                         # SKIP
