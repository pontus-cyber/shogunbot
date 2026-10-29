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


def prefer(name):
    """Evaluator that scores `name` above every other option on every seed."""
    from shogunbot import rewards

    def ev(opts, hp, seeds, depth, workers, scenario, **kw):
        for o in opts:
            o.scores = [10.0 if o.name == name else 9.0] * 4
            o.value, o.wins = sum(o.scores) / 4, 4
        rewards.paired(opts)
    return ev


@pytest.fixture(autouse=True)
def fast_valuation(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_tile_choice_takes_the_chosen_tile(tmp_path, monkeypatch):
    from shogunbot import rewards
    from shogunbot.bridge import Tracker
    reward = frame("runs/20261028-061420/frames/f0034_CLICK.png")   # Back Charge | Swirl
    after = frame("collect/20261027-164727/f0043_idle.png")          # transition
    # force the choice so the test does not depend on simulated values
    monkeypatch.setattr(rewards, "evaluate", prefer("take:swirl"))
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
    monkeypatch.setattr(rewards, "evaluate", prefer("upgrade:swirl"))
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


def test_pure_gain_shrine_is_taken_even_when_skip_scores_higher(tmp_path, monkeypatch):
    # [run 20261028-082756] '+1 DAMAGE' after Bamboo Grove II was skipped on noise in 0.15
    from shogunbot import rewards
    from shogunbot.bridge import Tracker
    from shogunbot.model import Kit
    shrine = frame("runs/20261028-082756/frames/reward_2_shrine.png")   # hand spear, arrow, swirl
    lit = frame("collect/20261027-164727/f0111_CLICK.png")
    done = frame("collect/20261027-164727/f0112_idle.png")
    monkeypatch.setattr(rewards, "evaluate", prefer("skip"))
    gw = FakeGW([(shrine, [(788, 900, 1132, 1008)]), (lit, [(800, 718, 1130, 786)]), (done, [])])
    b = make_bot(tmp_path, gw)
    tr = Tracker(Kit.of("swirl", "arrow", "spear"))
    assert b._reward(shrine, tr, "bamboo-2", 10) == "taken"
    assert tr.kit != Kit.of("swirl", "arrow", "spear") and gw.clicks[-1] == (962, 752)


def test_queueing_a_tile_that_is_not_in_the_hand_sends_nothing(tmp_path):
    # 0.15 waited for a face it could not click; do() now reports it and the loop plans again
    # [review of 0.15]
    from types import SimpleNamespace
    from shogunbot.model import Kit
    blank = np.zeros((1080, 1920, 3), np.uint8)
    gw = FakeGW([(blank, [])])
    b = make_bot(tmp_path, gw)
    p = SimpleNamespace(hand=[{"tile": "swirl", "cooldown": 3, "charged": 3, "ready": True, "fx": 0.45, "fy": 0.88}])
    assert b.do("queue:spear", p, blank, Kit.of("swirl", "arrow", "spear")) is None
    assert gw.clicks == []


def test_shrine_upgrade_is_taken_with_the_cursor_moved_away(tmp_path, monkeypatch):
    # [run 20261028-092419 f0089-f0091] UPGRADE lit brown (not peach) after Swirl went into
    # the statue; 0.15 missed it, clicked again and skipped the shrine
    from shogunbot import rewards
    from shogunbot.bridge import Tracker
    from shogunbot.model import Kit
    shrine = frame("runs/20261028-092419/frames/f0089_CLICK.png")   # +1 DAMAGE +1 COOLDOWN; arrow, katana, swirl
    lit = frame("runs/20261028-092419/frames/f0090_CLICK.png")      # swirl in the statue
    done = frame("collect/20261027-164727/f0112_idle.png")
    monkeypatch.setattr(rewards, "evaluate", prefer("upgrade:swirl"))
    gw = FakeGW([(shrine, [(1028, 900, 1132, 1008)]), (lit, [(800, 718, 1130, 786)]), (done, [])])
    b = make_bot(tmp_path, gw)
    tr = Tracker(Kit.of("swirl", "arrow", "katana"))
    assert b._reward(shrine, tr, "bamboo-2", 10) == "taken"
    assert gw.clicks == [(1080, 954), (962, 752)]
    assert (tr.kit["swirl"].damage, tr.kit["swirl"].cooldown, tr.kit["swirl"].used) == (3, 4, 1)


def test_wrong_tile_in_the_statue_is_swapped_for_the_chosen_one(tmp_path, monkeypatch):
    # [run 20261028-092419 f0124-f0126] the hand closes up after a tile leaves it; a click at
    # the old spot put Arrow in the statue. The bot must find Katana again and click it.
    from shogunbot import rewards
    from shogunbot.bridge import Tracker
    from shogunbot.model import Kit
    shrine = frame("runs/20261028-092419/frames/f0124_CLICK.png")   # +1 DAMAGE; arrow, katana, swirl
    wrong = frame("runs/20261028-092419/frames/f0126_CLICK.png")    # arrow in the statue; swirl, katana in hand
    right = frame("runs/20261028-092419/frames/f0125_CLICK.png")    # katana in the statue
    done = frame("collect/20261027-164727/f0112_idle.png")
    monkeypatch.setattr(rewards, "evaluate", prefer("upgrade:katana"))
    gw = FakeGW([(shrine, [(908, 900, 1012, 1008)]), (wrong, [(968, 900, 1072, 1008)]),
                 (right, [(800, 718, 1130, 786)]), (done, [])])
    b = make_bot(tmp_path, gw)
    tr = Tracker(Kit.of("swirl", "arrow", "katana"))
    assert b._reward(shrine, tr, "boss", 9) == "taken"
    assert gw.clicks == [(960, 954), (1020, 954), (962, 752)]
    assert tr.kit["katana"].damage == 3


def test_chosen_copy_in_the_statue_is_upgraded_when_a_tooltip_hides_upgrade(tmp_path, monkeypatch):
    # [run 20261028-133837, boss shrine '-1 COOLDOWN'] kit swirl (cooldown 2), arrow, swirl#2
    # (cooldown 3). The chosen Swirl went into the statue, but Arrow slid under the cursor and its
    # tooltip covered UPGRADE (f0123). 0.19 read UPGRADE as unlit, looked for that Swirl in the hand,
    # found only the other copy ('tile not found in hand') and skipped the upgrade.
    from shogunbot import rewards
    from shogunbot.bridge import Tracker
    from shogunbot.model import Kit, KitTile
    shrine = frame("runs/20261028-133837/frames/f0122_CLICK.png")   # hand swirl 2/2, arrow, swirl 2/3
    hidden = frame("runs/20261028-133837/frames/f0123_CLICK.png")   # swirl in the statue, tooltip over UPGRADE
    done = frame("collect/20261027-164727/f0112_idle.png")
    monkeypatch.setattr(rewards, "evaluate", prefer("upgrade:swirl"))
    gw = FakeGW([(shrine, [(788, 900, 892, 1008)]), (hidden, [(800, 718, 1130, 786)]), (done, [])])
    b = make_bot(tmp_path, gw)
    kit = Kit.of("swirl", "arrow", "swirl")
    kit = kit.replace("swirl", KitTile("swirl", "swirl", 2, 2, None, 3, 1))
    tr = Tracker(kit)
    assert b._reward(shrine, tr, "boss", 10) == "taken"
    assert gw.clicks == [(840, 954), (962, 752)]              # the cooldown-2 Swirl, then UPGRADE
    assert (tr.kit["swirl"].cooldown, tr.kit["swirl"].used) == (1, 2)
    assert (tr.kit["swirl#2"].cooldown, tr.kit["swirl#2"].used) == (3, 0)


def test_statue_check_tells_two_copies_apart():
    from types import SimpleNamespace
    from shogunbot import bot as B
    from shogunbot.bridge import Tracker
    from shogunbot.model import Kit
    tr = Tracker(Kit.of("swirl", "arrow").add("swirl", cooldown=2))
    rs = lambda statue, lit=False: SimpleNamespace(statue=statue, upgrade_lit=lit, hand=[])
    assert B.Bot._in_statue(rs({"tile": "swirl", "cooldown": 2}), tr, "swirl#2")
    assert not B.Bot._in_statue(rs({"tile": "swirl", "cooldown": 2}), tr, "swirl")     # the other copy
    assert B.Bot._in_statue(rs({"tile": "arrow", "cooldown": 5}), tr, "arrow")         # one copy: tile is enough
    assert not B.Bot._in_statue(rs(None), tr, "arrow")                                 # nothing read, not lit
