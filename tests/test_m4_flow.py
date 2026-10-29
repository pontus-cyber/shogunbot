"""0.24 [user review of runs 20261029-072042 and -072323]: levels move forward with the reward
screens, every run starts at the camp, the shop is read and left by GO, and a map that does not
react stops the run. Frame-based tests skip when the recorded frames are missing."""

import json
from types import SimpleNamespace

import pytest
from test_bot_flow import FakeGW, _stub_windows_modules, frame, make_bot
from test_series import _GW, _Rec

from shogunbot.bridge import Tracker

_stub_windows_modules()        # bot.py imports winio, which needs the Windows-only local_game_mcp

R072042 = "runs/20261029-072042/frames/"
R072323 = "runs/20261029-072323/frames/"
SERIES = "runs/20261028-130456/frames/"     # 0.18.2 giving up a won run from the Stage 2 map
FULL = "collect/20261028-190026/"
PAUSE_BOX = (1784, 972, 1855, 1035)
GIVE_UP_ROW = (858, 612, 1057, 652)
YES_ROW = (810, 532, 1105, 572)
BACK_TO_CAMP_BOX = (1472, 975, 1808, 1035)
SHOP_GO_BOX = (1544, 724, 1759, 787)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def _events(b):
    return [json.loads(l) for l in open(b.log.run_dir / "events.jsonl", encoding="utf-8")]


def _stop_after(monkeypatch, n):
    from shogunbot import bot as B
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > n
    monkeypatch.setattr(B, "_stop_requested", stop)


# ---- levels follow the reward screens [user review of run 20261029-072323] ------------------
def _levels(tr, seq):
    out = []
    for screen, cells in seq:
        if screen == "reward":
            tr.next_level()
        else:
            out.append(tr._level(screen, cells))
    return out


def test_a_boss_arena_without_its_bar_stays_the_boss_level():
    # run 072323: Rei the Ruthless fights in a 7-cell arena; after he died the bar went and
    # 0.23.1 named the 7-cell board 'bamboo-2' by its width, so the boss shrine was logged as a
    # new level with 0 turns and the boss level ended as 'next'
    seq = [("combat", 5), ("reward", 0), ("combat", 7), ("reward", 0),
           ("boss", 7), ("boss", 7), ("combat", 7), ("reward", 0)]
    assert _levels(Tracker(), seq) == ["bamboo-1", "bamboo-2", "boss", "boss", "boss"]


def test_levels_after_stage1_count_on_by_stage():
    seq = [("combat", 5), ("reward", 0), ("combat", 7), ("reward", 0), ("combat", 5), ("boss", 5),
           ("reward", 0), ("combat", 5), ("reward", 0), ("combat", 7), ("reward", 0), ("boss", 7)]
    assert _levels(Tracker(), seq) == ["bamboo-1", "bamboo-2", "boss", "boss",
                                       "stage2-1", "stage2-2", "stage2-boss"]


def test_without_reward_reads_stage1_falls_back_to_the_width():
    # replays of recordings whose reward frames were not read: 5, then 7, then Daisuke's 5-cell
    # arena a turn before his bar shows
    seq = [("combat", 5), ("combat", 7), ("combat", 7), ("combat", 5), ("boss", 5)]
    assert _levels(Tracker(), seq) == ["bamboo-1", "bamboo-2", "bamboo-2", "boss", "boss"]
    assert _levels(Tracker(), [("combat", 7), ("boss", 7)]) == ["bamboo-2", "boss"]   # joined mid-run


def test_run_072323_frames_name_the_levels():
    from shogunbot.perception.perceive import perceive
    tr, got = Tracker(), []
    for name in ("t0055_before.png", "reward_seen_2.png", "t0063_before.png", "t0073_after.png", "win.png"):
        p = perceive(frame(R072323 + name))
        if p.screen == "reward":
            tr.next_level()
        else:
            got.append(tr.state(p).level)
    assert got == ["bamboo-2", "boss", "boss"]


def test_a_reward_ends_the_level_as_cleared(tmp_path, monkeypatch):
    # 0.23.1 ended a level only when the next one started ('next'), or with the run's result
    from shogunbot import bot as B
    b = make_bot(tmp_path, _GW([(frame(SERIES + "f0100_CLICK.png"), [])]))
    b.rec, b.restart_after = _Rec(), False
    start_run = b.log.start_run

    def start_in_a_level(meta):
        rid = start_run(meta)
        b.log.level_start("bamboo-1", 10, 4)
        return rid
    b.log.start_run = start_in_a_level
    monkeypatch.setattr(B.Bot, "_reward", lambda self, *a, **k: "taken")
    screens = iter(["reward"] + ["unknown"] * 50)
    monkeypatch.setattr(B, "perceive", lambda fr, kit=None: SimpleNamespace(
        screen=next(screens), hero=None, enemies=[], items=[], cells=[], hand=[], wave=None, confidence={}))
    _stop_after(monkeypatch, 3)
    b.run()
    ends = [e for e in _events(b) if e["event"] == "level_end"]
    assert [(e["level"], e["outcome"]) for e in ends] == [("bamboo-1", "cleared")]


# ---- every run starts at the camp [user 2026-10-29] ------------------------------------------
def test_a_run_that_starts_on_a_map_gives_it_up_first(tmp_path, monkeypatch):
    # run 072042 began on the Stage 2 map the last single run had left and sent Stage 1 map inputs
    gw = _GW([(frame(R072042 + "map_1.png"), [PAUSE_BOX]),
              (frame(SERIES + "f0101_CLICK.png"), [GIVE_UP_ROW]),
              (frame(SERIES + "f0102_CLICK.png"), [YES_ROW]),
              (frame(SERIES + "give_up_failed.png"), [])])          # the camp
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after, b.start_at_camp = _Rec(), False, True
    _stop_after(monkeypatch, 3)
    b.run()
    assert gw.clicks[:3] == [(1820, 1001), (958, 632), (958, 552)]   # pause, GIVE UP, YES
    assert gw.i == 3
    ev = _events(b)
    assert any(e["event"] == "start_elsewhere" and e["screen"] == "map" for e in ev)
    assert not any(e["event"] == "menu" and e.get("screen") == "map" for e in ev)   # no map input


def test_a_run_that_starts_in_the_shop_gives_it_up_first(tmp_path, monkeypatch):
    # 072042 after the user clicked a shop node: 0.23.1 read the shop as a fight with HP 0
    gw = _GW([(frame(R072042 + "unknown_1793254918.png"), [PAUSE_BOX]),
              (frame(SERIES + "f0101_CLICK.png"), [GIVE_UP_ROW]),
              (frame(SERIES + "f0102_CLICK.png"), [YES_ROW]),
              (frame(SERIES + "give_up_failed.png"), [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after, b.start_at_camp = _Rec(), False, True
    _stop_after(monkeypatch, 3)
    b.run()
    assert gw.clicks[:3] == [(1820, 1001), (958, 632), (958, 552)]
    assert not any(c == (1651, 755) for c in gw.clicks)              # GO is not clicked


def test_a_run_that_starts_on_the_death_screen_goes_back_to_camp(tmp_path, monkeypatch):
    # RESTART would keep the last run's day; the camp is where Day 1 is chosen
    gw = _GW([(frame("runs/20261027-182223/frames/death.png"), [BACK_TO_CAMP_BOX]),
              (frame(SERIES + "give_up_failed.png"), [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after, b.start_at_camp = _Rec(), False, True
    _stop_after(monkeypatch, 2)
    b.run()
    assert gw.clicks[0] == (1640, 1004) and gw.i == 1


def test_the_start_stops_when_the_give_up_fails(tmp_path, monkeypatch):
    from shogunbot import bot as B
    monkeypatch.setattr(B, "GIVE_UP_WAIT_S", 0.2)
    gw = _GW([(frame(R072042 + "map_1.png"), [])])                  # the pause menu never opens
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after, b.start_at_camp = _Rec(), False, True
    assert b.run() == "stuck"
    assert gw.clicks == [(1820, 1001)] * 3


# ---- the shop: read, then GO [user 2026-10-29] -----------------------------------------------
def test_shop_reads_the_offers_and_leaves_by_go(tmp_path, monkeypatch):
    gw = _GW([(frame(FULL + "f0172_CLICK.png"), [SHOP_GO_BOX]), (frame(FULL + "f0183_CLICK.png"), [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after, b.shop_n = _Rec(), False, 0
    assert b._shop(gw.grab()) is True
    assert gw.clicks == [(1651, 755)]
    shop = next(e for e in _events(b) if e["event"] == "shop")
    assert (shop["upgrade"], shop["upgrade_price"], shop["coins"], shop["skulls"]) == ("cooldown_minus_2", 20, 27, 57)
    assert [(o["price"], o["currency"]) for o in shop["counter"]] == [(15, "coin"), (1, "skull")]


def test_shop_reads_on_every_recorded_shop():
    # the six shop visits of the full-game recording and the one of run 072042, first frames
    from shogunbot.perception import shop_screen as SS
    from shogunbot.perception.perceive import normalize
    want = {FULL + "f0172_CLICK.png": ("cooldown_minus_2", 20, 27), FULL + "f0329_CLICK.png": ("damage_1", 20, 28),
            FULL + "f0587_idle.png": ("damage_1", 20, 37), FULL + "f0932_idle.png": ("cooldown_minus_2", 20, 45),
            FULL + "f1365_idle.png": ("cooldown_minus_2", 20, 54), FULL + "f1599_idle.png": ("shockwave", 20, 40),
            R072042 + "unknown_1793254918.png": ("poison", 20, 29)}
    for rel, (up, price, coins) in want.items():
        d = SS.read(normalize(frame(rel)))
        assert (d["upgrade"], d["upgrade_price"], d["coins"]) == (up, price, coins), rel


def test_shop_stops_when_go_does_not_leave(tmp_path):
    gw = _GW([(frame(FULL + "f0172_CLICK.png"), [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after, b.shop_n = _Rec(), False, 0
    assert b._shop(gw.grab()) is False
    assert gw.clicks == [(1651, 755)] * 3


# ---- a map that does not react stops the run ------------------------------------------------
def test_map_inputs_stop_after_the_limit(tmp_path, monkeypatch):
    from shogunbot import bot as B
    gw = _GW([(frame(R072042 + "map_1.png"), [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after = _Rec(), False          # start_at_camp off: the map is played
    assert b.run() == "stuck"
    tries = [e for e in _events(b) if e["event"] == "menu" and e.get("screen") == "map"]
    assert len(tries) == B.MAP_TRIES_MAX
    assert any(e["event"] == "map_stuck" for e in _events(b))


# ---- --stage1 and --full ------------------------------------------------------------------
def test_the_final_level_follows_the_mode(tmp_path):
    from shogunbot import bot as B
    b = make_bot(tmp_path, FakeGW([(frame(SERIES + "f0100_CLICK.png"), [])]))
    assert b.last_level == "boss"
    b.stage1 = False
    assert b.last_level == f"stage{B.FULL_LAST_STAGE}-boss"


def test_cli_play_modes(monkeypatch):
    from shogunbot import __main__ as M
    seen = {}
    import shogunbot.bot as B
    monkeypatch.setattr(B, "play", lambda **kw: seen.update(kw))
    M.main(["play"])
    assert seen["stage1"] is True
    M.main(["play", "--full"])
    assert seen["stage1"] is False
    M.main(["play", "--stage1", "--runs", "3"])
    assert seen["stage1"] is True and seen["runs"] == 3


# ---- where the cursor was at each click [user review of run 20261029-072042] -----------------
def test_clicks_log_the_cursor_on_the_map_and_on_a_miss(tmp_path):
    gw = FakeGW([(frame(SERIES + "f0100_CLICK.png"), [])])
    b = make_bot(tmp_path, gw)
    f = gw.grab()

    def at(cx, cy):
        def click(fx, fy):
            gw.last_click = (round(fx * 1920), round(fy * 1080), cx, cy)
        return click
    gw.click = at(1500, 900)
    b.click(0.5, 0.5, f, screen="camp")                     # missed by far: logged
    gw.click = at(960, 541)
    b.click(0.5, 0.5, f, screen="map")                      # on target: logged on the map only
    gw.click = at(960, 540)
    b.click(0.5, 0.5, f, screen="camp")                     # on target elsewhere: not logged
    got = [(e["screen"], e["ok"]) for e in _events(b) if e["event"] == "cursor"]
    assert got == [("camp", False), ("map", True)]


# ---- the map between regions in keyboard-only mode [user 2026-10-29; captures 20261029-kbd-map] ----
KBD = "collect/20261029-kbd-map/"
KEYBOARD_BOX = (1785, 808, 1855, 878)


class KeyGW(_GW):
    """FakeGW whose steps may also move on keys: steps [(frame, click boxes, {key: next step})]."""
    def __init__(self, steps):
        super().__init__([(st[0], st[1]) for st in steps])
        self.keymap = [st[2] if len(st) > 2 else {} for st in steps]
        self.keys = []

    def key(self, name):
        self.keys.append(name)
        nxt = self.keymap[self.i].get(name)
        if nxt is not None:
            self.i = nxt


def test_map_reader_names_the_marked_node():
    from shogunbot.perception import map_screen as M
    reads = {}
    for name in ("k1_map_kbd_shop_top", "k2_map_mouse_at_shop", "k3_map_kbd_caves", "k4_map_kbd_hibiku"):
        f = frame(KBD + name + ".jpg")
        sel = M.selected(f)
        reads[name] = (M.keyboard_mode(f), sel and sel["kind"], sel and sel["region"])
    assert reads == {"k1_map_kbd_shop_top": (True, "shop", None), "k2_map_mouse_at_shop": (False, None, None),
                     "k3_map_kbd_caves": (True, "region", "whispering_caves"),
                     "k4_map_kbd_hibiku": (True, "region", "hibiku_wastelands")}
    assert M.keyboard_mode(frame(KBD + "k5_shop_kbd.jpg")) is True
    assert M.keyboard_mode(frame(R072042 + "map_1.png")) is False


def test_map_after_a_boss_looks_for_the_preferred_region(tmp_path):
    # mouse mode: click the keyboard button; Hibiku is marked, down does not move the mark, up
    # marks the Caves, Enter travels there
    gw = KeyGW([(frame(KBD + "k2_map_mouse_at_shop.jpg"), [KEYBOARD_BOX]),
                (frame(KBD + "k4_map_kbd_hibiku.jpg"), [], {"up": 2}),
                (frame(KBD + "k3_map_kbd_caves.jpg"), [], {"enter": 3}),
                (frame(FULL + "f0186_idle.png"), [])])
    b = make_bot(tmp_path, gw)
    look = {}
    for _ in range(8):
        if gw.i == 3:
            break
        b._map_step(gw.grab(), look)
    assert gw.clicks == [(1820, 843)] and gw.keys == ["down", "up", "enter"]
    travel = next(e for e in _events(b) if e["event"] == "map_travel")
    assert travel["region"] == "whispering_caves" and len(travel["seen"]) == 2


def test_map_takes_a_shop_at_once(tmp_path):
    gw = KeyGW([(frame(KBD + "k1_map_kbd_shop_top.jpg"), [], {"enter": 1}), (frame(FULL + "f0172_CLICK.png"), [])])
    b = make_bot(tmp_path, gw)
    b._map_step(gw.grab(), {})
    assert gw.keys == ["enter"] and gw.i == 1


def test_keyboard_mode_left_on_is_turned_off_by_a_click(tmp_path, monkeypatch):
    # the shop in keyboard mode: one click on an empty spot first, then the shop as usual
    from shogunbot import bot as B
    spot = (round(B.MOUSE_MODE_SPOT[0] * 1920), round(B.MOUSE_MODE_SPOT[1] * 1080))
    gw = _GW([(frame(KBD + "k5_shop_kbd.jpg"), [(spot[0] - 5, spot[1] - 5, spot[0] + 5, spot[1] + 5)]),
              (frame(FULL + "f0172_CLICK.png"), [SHOP_GO_BOX]), (frame(FULL + "f0183_CLICK.png"), [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after = _Rec(), False
    _stop_after(monkeypatch, 3)
    b.run()
    assert gw.clicks[:2] == [spot, (1651, 755)]
    assert any(e["event"] == "mouse_mode" for e in _events(b))


def test_give_up_in_keyboard_mode_opens_the_menu_with_esc(tmp_path):
    gw = KeyGW([(frame(KBD + "k3_map_kbd_caves.jpg"), [], {"esc": 1}),
                (frame(KBD + "k0_pause_kbd.jpg"), [GIVE_UP_ROW]),
                (frame(SERIES + "f0102_CLICK.png"), [YES_ROW]),
                (frame(SERIES + "give_up_failed.png"), [])])
    b = make_bot(tmp_path, gw)
    assert b._give_up(wait_map=False) is True
    assert gw.keys == ["esc"] and gw.clicks == [(958, 632), (958, 552)]
