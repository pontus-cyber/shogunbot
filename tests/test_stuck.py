"""0.33 [user review of run 20261030-041652, fixes 1-3]: the Arrow had become 4/8 at the Whispering Caves II
shrine (+1 damage +1 cooldown), but its eighth cooldown mark fell outside the marks band, so the faces read 4/7
and corrected the kit at the Iwao level start (t133). From then on the tracker counted the Arrow full one turn
early: its click was refused 54 times at Iwao from t143 and on every turn from t237 on Moonlit Port I (over 800
in a row, for hours), while the game's turn counter stood still. The band now reaches the eighth mark, a dimmed
face is planned as recharging and is never clicked, a queued tile that stays in the hand is banned on that board,
and a run whose inputs change nothing stops as 'stuck'. Frame-based tests skip when the frames are missing."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_bot_flow import FakeGW, _stub_windows_modules, frame, make_bot
from test_rewards import face, fake_perception
from test_series import _GW, _Rec

from shogunbot.bridge import Tracker, kit_from_faces
from shogunbot.model import Kit, KitTile
from shogunbot.perception import reward_screen
from shogunbot.perception.perceive import normalize, read_hand

_stub_windows_modules()        # bot.py imports winio, which needs the Windows-only local_game_mcp

R041652 = "runs/20261030-041652/frames/"
KIT_TILES = ["swirl", "arrow", "katana"]


def _events(b):
    return [json.loads(l) for l in open(b.log.run_dir / "events.jsonl", encoding="utf-8")]


def _misread_kit():
    """The kit the tracker held on Moonlit Port I: the Arrow at 4/7 (the cut read), really 4/8."""
    return Kit([KitTile("swirl", "swirl", 2, 1, None, 3, 1), KitTile("arrow", "arrow", 4, 7, "poison", 3, 3),
                KitTile("katana", "katana", 2, 0, None, 4, 0), KitTile("swirl#2", "swirl", 2, 3, None, 2, 0)])


# ---- fix 3: the marks band reaches the eighth mark ----------------------------------------
@pytest.mark.parametrize("name,charged", [("t0236_before.png", 6), ("t0237_before.png", 7), ("t1026_before.png", 7)])
def test_the_eighth_cooldown_mark_is_read(name, charged):
    # 7 grey marks and a red one at t237 and t1026 (6 and 2 at t236); 0.15-0.32 read 7 marks, all grey
    arrow = next(f for f in read_hand(normalize(frame(R041652 + name)), KIT_TILES) if f["tile"] == "arrow")
    assert (arrow["damage"], arrow["cooldown"], arrow["charged"], arrow["ready"]) == (4, 8, charged, False)


def test_the_reward_screen_reads_the_4_8_arrow():
    # the Iwao shrine (cooldown -2), where 0.32.1 kept the kit's 4/7
    rs = reward_screen.read(normalize(frame(R041652 + "reward_6_shrine.png")), tuple(KIT_TILES))
    arrow = next(f for f in rs.hand if f["tile"] == "arrow")
    assert (arrow["damage"], arrow["cooldown"]) == (4, 8)
    assert kit_from_faces(_misread_kit(), rs.hand)["arrow"].cooldown == 8


def test_the_freeplay_arrow_reads_its_eight_marks():
    # run 20261029-100218 t62: the Arrow with Free play (cooldown 8), full; 0.32 read it 7 of 7
    arrow = next(f for f in read_hand(normalize(frame("runs/20261029-100218/frames/t0062_before.png")),
                                      ["swirl", "arrow", "shuriken"]) if f["tile"] == "arrow")
    assert (arrow["cooldown"], arrow["charged"], arrow["ready"]) == (8, 8, True)


# ---- fix 1: trust the screen ----------------------------------------------------------------
def test_a_dimmed_face_with_full_marks_is_planned_as_recharging():
    # t237 as 0.32.1 read it: 7 of 7 marks against the kit's 7, the digit dimmed
    tr = Tracker(_misread_kit())
    s = tr.state(fake_perception([face("swirl", cooldown=1, charged=1), face("katana"),
                                  face("swirl", cooldown=3, charged=3),
                                  face("arrow", cooldown=7, charged=7, damage=4, ready=False)]))
    assert tr.charge["arrow"] == 6 and "arrow" not in s.hero.ready_labels()
    # a bright face with full marks stays ready
    s = Tracker(_misread_kit()).state(fake_perception([face("katana"), face("arrow", cooldown=7, charged=7, damage=4)]))
    assert "arrow" in s.hero.ready_labels()


def test_a_refused_tile_is_tracked_as_recharging():
    tr = Tracker(_misread_kit())
    tr.recharging("arrow")
    assert tr.charge["arrow"] == 6
    tr.recharging("katana")                 # cooldown 0: nothing to wait for
    assert tr.charge["katana"] == 0


def test_a_dimmed_face_is_never_clicked(tmp_path):
    blank = np.zeros((1080, 1920, 3), np.uint8)
    gw = FakeGW([(blank, [])])
    b = make_bot(tmp_path, gw)
    p = SimpleNamespace(hand=[face("arrow", cooldown=8, charged=7, damage=4, ready=False, fx=0.594)])
    assert b.do("queue:arrow", p, blank, _misread_kit(), {"arrow": 7}) is None
    assert gw.clicks == []
    assert any(e["event"] == "act_failed" and "dimmed" in e["reason"] for e in _events(b))


# ---- fixes 1 and 2 through the main loop: a screen that never changes -------------------------
def test_a_screen_that_never_changes_ends_the_run_as_stuck(tmp_path, monkeypatch):
    from shogunbot import bot as B
    img = frame(R041652 + "t1026_before.png")
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(B, "BLINK_SAMPLES", 0)
    monkeypatch.setattr(B, "NO_CHANGE_STOP", 8)
    monkeypatch.setattr(B.Bot, "_wait_change", lambda self, before, timeout=2.5: False)
    gw = _GW([(img, [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after = _Rec(), False
    assert b.run() == "stuck"
    ev = _events(b)
    stop = [e for e in ev if e["event"] == "no_change_stop"]
    assert len(stop) == 1 and stop[0]["inputs"] == 8
    assert (b.log.run_dir / "frames" / "stuck_no_change.png").exists()
    # the dimmed Arrow (7 of 8) is never clicked
    arrow = next(f for f in read_hand(normalize(img), KIT_TILES) if f["tile"] == "arrow")
    assert all(abs(x - arrow["fx"] * 1920) > 40 for x, y in gw.clicks)
    # a tile queued on a board that then stays the same is not sent a third time on it
    sent = [e["action"] for e in ev if e["event"] == "no_visible_change"]
    assert all(not (a.startswith("queue:") and sent[i:i + 3] == [a] * 3) for i, a in enumerate(sent))
