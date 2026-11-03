"""0.48: --full goes on to region 7, the Shogun's Castle and the Shogun, the last region of the Day 1 route [standing rule,
user 2026-11-01 16:50; claude/core-requests.md item 14], and items 12, 13 and 11 of that file. Both recordings took the
Castle after the Keep's shop (collect/20261028-190026 f1613 'THE SHOGUN?S CASTLE'): one wave level of 9 cells and 10 waves,
then the Shogun's 7-cell arena with his two Summoners (rec1 f1739-f1830, rec2 f1916-f2007), then the VICTORY screen. Frame
tests skip when the frames are missing."""

import glob
from types import SimpleNamespace

import pytest
from test_bot_flow import DATA, _stub_windows_modules, frame, make_bot
from test_m4_flow import PAUSE_BOX, GIVE_UP_ROW, YES_ROW, SERIES, _events, _stop_after
from test_series import _GW, _Rec

from shogunbot import behaviour
from shogunbot.bridge import Tracker
from shogunbot.catalogue import CAT
from shogunbot.model import Enemy, Hero, State
from shogunbot.rules import step

_stub_windows_modules()

REC1 = "collect/20261028-190026/"
REC2 = "collect/20261029-160032/"
ROUTE1 = ("bamboo_grove", "whispering_caves", "moonlit_port", "spirit_gateway", "hot_springs", "hideyoshis_keep",
          "shoguns_castle")
ROUTE2 = ("bamboo_grove", "hibiku_wastelands", "moonlit_port", "spirit_gateway", "hot_springs", "hideyoshis_keep",
          "shoguns_castle")
O, B, M, C, S, P, X, U = ("origin_of_symmetry", "back_charge", "mirror", "charge", "swirl", "spear", "corrupted_barrage",
                          "shuriken")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)


def _f(rec: str, n: int):
    hit = glob.glob(f"{DATA}/{rec}f{n:04d}_*.png")
    if not hit:
        pytest.skip(f"recorded frame not available: {rec}f{n:04d}")
    return frame(rec + hit[0].split("/")[-1])


def _p(rec, n, route=ROUTE1):
    from shogunbot.perception.perceive import perceive
    return perceive(_f(rec, n), ["swirl", "arrow"], regions=route, day=1)


def _units(p):
    return [(e["x"], e["kind"], e["queue"]) for e in p.enemies]


# ---------------------------------------------------------------- the route
def test_the_map_names_the_castle_and_the_bot_goes_there():
    from shogunbot import bot as Bt
    from shogunbot.perception import map_screen
    sel = map_screen.selected(_f(REC1, 1613))       # 'THE SHOGUN?S CASTLE': the apostrophe reads as '?'
    assert (sel["kind"], sel["region"]) == ("region", "shoguns_castle")
    assert "shoguns_castle" in Bt.MAP_REGIONS and Bt.FULL_LAST_STAGE == 7


def test_the_castle_has_one_wave_level_then_the_shogun():
    tr = Tracker()
    tr.route = list(ROUTE1)
    tr.stage, tr.level_i = 7, -1
    names = []
    for screen, cells in (("combat", 9), ("boss", 7)):
        if names:
            tr.next_level()
        names.append(tr._level(screen, cells))
    assert names == ["stage7-1", "stage7-boss"]
    assert CAT.boss_by_arena("shoguns_castle", 7) == "the_shogun"
    k = CAT.enemy("the_shogun")
    assert (k.hp, k.hp_day5, k.phases, k.max_hp) == (40, 50, 2, 80) and {"quick", "unfreezable"} <= set(k.traits)
    assert [k.phase(h) for h in (80, 41, 40, 1)] == [1, 1, 2, 2]
    assert CAT.enemy("shogun_summoner").hp == 30 and "heavy" in CAT.enemy("shogun_summoner").traits
    from shogunbot.replay import UNSHOWN_HP
    assert {"the_shogun", "shogun_summoner"} <= UNSHOWN_HP          # no pips: the replay scores them on cell and kind


def test_the_map_to_the_castle_starts_its_wave_level_without_a_reward_screen():
    # core-requests item 17a: no reward screen after Hideyoshi (the map comes at once, rec1 f1594), so runs 20261102-232559,
    # 20261103-020829 and -035113 (0.48) played the Castle's 9-cell wave level as 'stage6-boss', with no waves to come
    tr = Tracker()
    tr.route = list(ROUTE1[:6])
    tr.stage, tr.level_i = 6, tr.boss_index(6)            # Hideyoshi's fight; no reward screen after it
    tr.route.append("shoguns_castle")                       # the map's travel (bot._map_travel)
    assert tr.start_region("shoguns_castle")
    assert tr._level("combat", 9) == "stage7-1"
    tr.next_level()                                         # its reward screen
    assert tr._level("boss", 7) == "stage7-boss"
    assert not tr.start_region("hideyoshis_keep")          # levels only move forward
    assert not tr.start_region("shoguns_castle")           # already in it


# ---------------------------------------------------------------- perception
def test_his_summoners_are_told_from_him_and_his_tiles_read():
    # rec1 f1743: the Summoners on the end cells load Ally Shield, the Shogun on cell 4 his Mirror (Hideyoshi's drawing)
    assert _units(_p(REC1, 1743)) == [(0, "shogun_summoner", ["ally_shield"]), (4, "the_shogun", [M]),
                                      (6, "shogun_summoner", ["ally_shield"])]
    # frames outside the template samples: Charge and Swirl yellow (f1808), Back Charge and Mirror (f1813), Charge and
    # Spear in phase 1 (f1782), on the left end in phase 2 he is still the Shogun (f1814: no Summoners left)
    assert _units(_p(REC1, 1808)) == [(3, "the_shogun", [C, S])]
    assert _units(_p(REC1, 1813)) == [(2, "the_shogun", [B, M])]
    assert [u for u in _units(_p(REC1, 1782)) if u[1] == "the_shogun"] == [(2, "the_shogun", [C, P])]
    assert [u[1] for u in _units(_p(REC1, 1814))] == ["the_shogun"]
    assert _units(_p(REC2, 1942, ROUTE2))[1] == (4, "the_shogun", [C])
    assert _units(_p(REC2, 1989, ROUTE2)) == [(3, "the_shogun", [C])]


def test_the_flames_over_his_yellow_frames_are_not_the_bell():
    # the orange flames over his fire frames passed the bell's colour rule: rec1 f1753 his Mirror read 'summon', the
    # tracker took his first tile (the Sword) and the replay planned a 2-HP cut that never came; f1807 his Charge (with
    # the Swirl), f1816 his Shuriken
    assert [u for u in _units(_p(REC1, 1753)) if u[1] == "the_shogun"] == [(2, "the_shogun", [M])]
    assert _units(_p(REC1, 1807)) == [(3, "the_shogun", [C, S])]
    assert _units(_p(REC1, 1816)) == [(0, "the_shogun", [U])]


def test_a_grey_slot_over_an_enemy_bubble_is_the_backdrop():
    # enemy bubbles are framed red or yellow; the grey sky over his yellow Mirror read as a '+' (a load, not the fire:
    # rec1 f1744, he mirrored from cell 4 to 2), and over his Summoners' yellow Ally Shield (f1745, which also passed the
    # bell's colour rule); the Keep's cages over a Yari Master on cell 0 gave its Spear a '+' on 28 frames (f1564-f1591)
    assert [(u[0], u[2]) for u in _units(_p(REC1, 1744))] == [(0, ["ally_shield"]), (4, [M]), (6, ["ally_shield"])]
    assert [e["intent"] for e in _p(REC1, 1744).enemies] == [None, "fire", None]
    p = _p(REC1, 1745)
    assert [(e["x"], e["queue"], e["intent"]) for e in p.enemies if e["kind"] == "shogun_summoner"] == [
        (0, ["ally_shield"], "fire"), (6, ["ally_shield"], "fire")]
    p = _p(REC1, 1570, ROUTE1[:6])
    assert [(e["x"], e["kind"], e["queue"], e["intent"]) for e in p.enemies][0] == (0, "yari_master", [P], None)
    # rec2 f1601: the Keep's cages over a unit with no bubble read as a grey '+'
    p = _p(REC2, 1601, ROUTE2[:6])
    assert all(e["intent"] is None for e in p.enemies if e["x"] == 2)


def test_a_summoner_whose_bell_glows_is_not_read_as_him():
    # a firing bell lights the Summoner's sprite red (rec1 f1765 cell 0: 5,378 red px, f1792 cell 6; rec2 f1945 cell 0
    # 1,824): with his read off the end cells, the end cells' pipless reads are Summoners, and the bell stays theirs
    assert [u for u in _units(_p(REC1, 1765)) if u[0] in (0, 1)] == [(0, "shogun_summoner", ["summon"]),
                                                                     (1, "the_shogun", [])]
    assert [u[1] for u in _units(_p(REC1, 1792)) if u[0] in (0, 2, 6)] == ["shogun_summoner", "the_shogun",
                                                                           "shogun_summoner"]
    assert [u[1] for u in _units(_p(REC2, 1945, ROUTE2)) if u[0] in (0, 1)] == ["shogun_summoner", "the_shogun"]


def test_the_victory_screen_and_its_back_to_camp_pill():
    from shogunbot.perception import screens
    from shogunbot.perception.perceive import normalize
    won, btn = screens.victory(normalize(_f(REC2, 2013)))
    assert won and abs(btn[0] - 0.854) < 0.01 and abs(btn[1] - 0.885) < 0.01
    assert screens.victory(normalize(_f(REC2, 2011))) == (True, None)          # the kanji first, the pill 8.8 s later
    for n in (1779, 1802, 1830):          # his yellow fire bubbles on the dark arena, and the fight's last frame: not it
        assert not screens.victory(normalize(_f(REC1, n)))[0]
    assert not screens.victory(normalize(_f("collect/20261027-164727/", 146)))[0]    # Bamboo Grove's golden sky


# ---------------------------------------------------------------- the simulator
def _st(hx, hf, ex, ef, q, hp_boss=80, intent=None, last="queue", fage=None, others=(), hp=10):
    e = Enemy("the_shogun", ex, ef, hp_boss, list(q), uid=1, intent=intent, last=last, fage=fage)
    return State(7, Hero(hx, hf, hp=hp), [e] + list(others), level="stage7-boss")


def _top(s, action="wait"):
    return max(step(s, action), key=lambda t: t[0])


def _boss(s):
    return next(e for e in s.enemies if e.kind == "the_shogun")


def test_the_corrupted_barrage_ring_and_how_the_hero_dodged_it():
    # rec1 f1801-f1807: phase 2, the Shogun on cell 2 facing the hero on 1 fires Origin and the Barrage: he lands on the
    # centre (3), pillars strike cells 2 and 4 (f1803), warning signs mark 1 and 5 (f1804), then 0 and 6 (f1805); the
    # hero stepped to 0 (A) and back to 1 (D) and lost nothing
    _, s2, _ = _top(_st(1, 1, 2, -1, [O, X], hp_boss=32, intent="fire"))
    assert (_boss(s2).x, s2.hero.hp, s2.rings) == (3, 10, [[3, 2, 0]])
    s2.enemies[0].intent = "hold"
    _, s3, _ = _top(s2, "left")
    assert (s3.hero.x, s3.hero.hp, s3.rings) == (0, 10, [[3, 3, 0]])
    s3.enemies[0].intent = "hold"
    _, s4, _ = _top(s3, "right")
    assert (s4.hero.x, s4.hero.hp, s4.rings) == (1, 10, [])
    # staying on cell 1 takes the ring's 3
    s2.enemies[0].intent = "hold"
    _, s3b, _ = _top(s2, "wait")
    assert s3b.hero.hp == 7


def test_the_hit_that_empties_his_first_bar_starts_phase_2():
    # rec1 f1794-f1797: his first bar emptied, the bar refilled, the Summoners and the summoned units were gone, he kept
    # his cell. Pooled: a hit from 42 stops at 40; the planner sees damage, not 40 HP more
    sm = Enemy("shogun_summoner", 0, 1, 30, [], uid=2, intent="hold")
    ym = Enemy("yari_master", 5, -1, 4, [], uid=3, intent="hold")
    s = _st(2, 1, 3, -1, [], hp_boss=42, intent="hold", others=[sm, ym])
    s.hero.queue = ["swirl"]
    _, s2, ev = _top(s, "attack")
    assert _boss(s2).hp == 40 and [e.kind for e in s2.enemies] == ["the_shogun"]
    assert [k for k, _, _ in ev.kills] == ["shogun_summoner", "yari_master"]
    assert _boss(s2).queue == [X]        # on the centre his opening Barrage needs no Origin (rec2 f1984)


def test_his_phase_2_movesets():
    from shogunbot.rules import _boss_queue
    e = Enemy("the_shogun", 2, 1, 30, [], uid=1, intent="queue")
    s = State(7, Hero(4, -1), [e])
    e.loaded = "phase2"
    assert _boss_queue(s, e, None) == [(1.0, ("queue", O))]            # off the centre: Origin first
    e.queue = [O]
    assert _boss_queue(s, e, None) == [(1.0, ("queue", X))]
    e.queue, e.loaded = [], "back_charge"                               # after Back Charge and Mirror: the Shuriken
    assert _boss_queue(s, e, None) == [(1.0, ("queue", U))]
    e.loaded = U                                                        # then any moveset but that one
    assert sorted(t for _, (_, t) in _boss_queue(s, e, None)) == sorted([C, O])
    e.loaded = C
    assert sorted(t for _, (_, t) in _boss_queue(s, e, None)) == sorted([B, O])


def test_his_phase_1_rules_and_his_summoners():
    f = lambda s, e: behaviour.features(s, e, e.last, e.fage)       # noqa: E731
    # the fight's first phase he rests (rec1 f1741, rec2 f1917-f1919)
    s = _st(2, 1, 4, -1, [], last=None)
    assert behaviour.shogun_intent(s, s.enemies[0], f(s, s.enemies[0])) == "hold"
    # a Charge gets its Spear (rec1 f1775-f1778); a Spear that does not reach the hero: a step toward him (f1757-f1759)
    s = _st(1, 1, 4, -1, [C])
    assert behaviour.shogun_intent(s, s.enemies[0], f(s, s.enemies[0])) == "queue"
    s = _st(1, 1, 5, -1, [P])
    assert behaviour.shogun_intent(s, s.enemies[0], f(s, s.enemies[0])) == "move-left"
    s = _st(3, 1, 4, -1, [S])                                           # the hero next to him: Quick, it fires
    assert behaviour.shogun_intent(s, s.enemies[0], f(s, s.enemies[0])) == "fire"
    # his Summoners: a load, held a phase, then fired; Ally Shield while he has none, else the bell
    sm = Enemy("shogun_summoner", 0, 1, 30, [], uid=2, last="hold")
    s = _st(2, 1, 4, -1, [], others=[sm])
    assert behaviour.summoner_intent(s, sm, f(s, sm)) == "queue"
    from shogunbot.rules import _intent_options
    sm.intent = "queue"
    assert _intent_options(s, sm, None) == [(1.0, ("queue", "ally_shield"))]
    s.enemies[0].shield = True
    assert _intent_options(s, sm, None) == [(1.0, ("queue", "summon"))]


def test_the_tracker_reads_his_bars_as_one_pool():
    tr = Tracker()
    assert tr._phased_hp("the_shogun", 1.0) == 80
    assert tr._phased_hp("the_shogun", 0.072) == 43                    # rec1 f1794
    assert tr._phased_hp("the_shogun", 1.0) == 40                      # f1797: the bar refilled, phase 2
    assert tr._phased_hp("the_shogun", 0.726) == 29


# ---------------------------------------------------------------- the end of a Day 1 run
def test_a_run_that_starts_on_the_victory_screen_goes_back_to_camp(tmp_path, monkeypatch):
    # rec2 f2013: the VICTORY screen with its BACK TO CAMP pill; f2016: the camp
    gw = _GW([(_f(REC2, 2013), [(1472, 924, 1807, 987)]), (_f(REC2, 2016), [])])
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after, b.start_at_camp = _Rec(), False, True
    _stop_after(monkeypatch, 2)
    b.run()
    assert gw.clicks[0] == (1640, 956) and gw.i == 1
    assert any(e["event"] == "menu" and e.get("screen") == "victory" for e in _events(b))


# ---------------------------------------------------------------- core-requests items 12 and 13
def test_the_heros_plate_after_a_swap_on_a_bright_board(tmp_path):
    # item 12, run 20261102-021456 stuck.png (Bamboo Grove II, t31): the swap ticks red, the plate's mean colour (97, 71,
    # 70) read 'empty', no hero, 90 s 'unknown' and the run stopped as stuck; his plate's stone ends are teal (0.32)
    from shogunbot.perception.perceive import perceive
    p = perceive(frame("runs/20261102-021456/frames/stuck.png"), ["swirl", "arrow"], regions=("bamboo_grove",), day=1)
    assert p.screen == "combat" and p.hero["x"] == 6
    assert [(e["x"], e["kind"]) for e in p.enemies] == [(0, "ashigaru"), (1, "ashigaru"), (5, "charger")]


def test_the_teal_plate_counts_only_when_neither_board_reader_makes_a_board():
    # found by the 0.48 replay gate: on Moonlit Port's dark 9-cell board (run 20261031-114119 f0310) the bright plates
    # made 5 cells with a 184-px gap and no hero; with the teal rule tried first they were a board, and the Ashigaru on
    # cell 4 lost its Sword and its step; the dark reader reads the 9 cells
    from shogunbot.perception.perceive import perceive
    p = perceive(frame("runs/20261031-114119/frames/f0310_SPACE.png"), ["swirl", "arrow"],
                 regions=("bamboo_grove", "whispering_caves", "moonlit_port"), day=1)
    assert len(p.cells) == 9 and p.hero["x"] == 2
    assert [(e["x"], e["kind"], e["queue"], e["intent"]) for e in p.enemies][2] == (4, "ashigaru", ["sword"], "move-left")


def test_a_run_that_starts_on_a_fight_it_cannot_read_gives_it_up(tmp_path, monkeypatch):
    # item 13: runs 20261102-031727, -045205 and -055546 found that frozen board as 'unknown' and stopped after 90 s with
    # 0 turns, leaving it for the next; a fight screen (its HUD) that stays unreadable at a run start is given up
    from shogunbot import bot as B
    monkeypatch.setattr(B, "UNKNOWN_FRAME_S", 0.0)
    fight = frame("runs/20261102-021456/frames/stuck.png")
    gw = _GW([(fight, [PAUSE_BOX]), (frame(SERIES + "f0101_CLICK.png"), [GIVE_UP_ROW]),
              (frame(SERIES + "f0102_CLICK.png"), [YES_ROW]), (frame(SERIES + "give_up_failed.png"), [])])
    real = B.perceive

    def unreadable(fr, kit=None, **kw):
        p = real(fr, kit, **kw)
        if p.screen == "combat":
            return SimpleNamespace(screen="unknown", hero=None, enemies=[], items=[], cells=[], hand=[], wave=None,
                                   confidence={})
        return p
    monkeypatch.setattr(B, "perceive", unreadable)
    b = make_bot(tmp_path, gw)
    b.rec, b.restart_after, b.start_at_camp = _Rec(), False, True
    _stop_after(monkeypatch, 4)
    b.run()
    assert gw.clicks[:3] == [(1820, 1001), (958, 632), (958, 552)]     # pause, GIVE UP, YES
    assert any(e["event"] == "start_elsewhere" and e["screen"] == "combat" for e in _events(b))


# ---------------------------------------------------------------- core-requests item 11
def test_the_last_edamame_is_kept_for_the_final_region():
    # runs 20261102-004152 and -145513 drank every brew at HP 6-7 before Hideyoshi's Keep and reached him at 1 HP
    from shogunbot.bot import edamame_reason
    s = State(7, Hero(3, 1, hp=6), [Enemy("ashigaru", 6, -1, 3, [], uid=1, intent="hold")], level="stage5-1")
    assert edamame_reason(s, "wait", 10) is not None                    # all 3 HP heal
    assert edamame_reason(s, "wait", 10, reserve=True) is None          # the last brew: kept
    # (a) still spends it: a hit that would leave HP within one enemy hit
    s = State(7, Hero(3, 1, hp=3), [Enemy("ashigaru", 4, -1, 3, ["sword"], uid=1, intent="fire")], level="stage5-1")
    assert edamame_reason(s, "wait", 10, reserve=True) is not None


def test_reactive_shield_after_an_hp_loss_is_tracked():
    # item 11, run 20261102-145513 t529: a Reactive Shield from an earlier hit took the Grappler's Hook, its Sword cut the
    # hero for 2 and gave him a new shield (t0529_before shows the shield; t530 HP 7). Untracked, the Hook was planned to
    # hit (-1) and its new shield to take the Sword
    from shogunbot.model import Kit
    from shogunbot.perception.perceive import perceive
    route = ("bamboo_grove", "whispering_caves", "moonlit_port", "spirit_gateway", "hot_springs")
    kit = Kit.of("swirl", "arrow", "back_charge").with_skill("reactive_shield")
    tr = Tracker(kit)
    tr.route, tr.stage, tr.level_i = list(route), 5, -1
    tr.next_level()
    run = "runs/20261102-145513/frames/"
    s1 = tr.state(perceive(frame(run + "t0529_before.png"), ["swirl", "arrow", "back_charge"], regions=route, day=1))
    tr.last_hp = 9
    tr.acted(s1, "wait")
    s2 = tr.state(perceive(frame(run + "t0530_before.png"), ["swirl", "arrow", "back_charge"], regions=route, day=1))
    assert s2.hero.hp == 7 and s2.hero.shield
