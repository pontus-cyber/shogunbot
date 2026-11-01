"""0.47: --full goes on to region 6, Hideyoshi's Keep and Hideyoshi the Cunning [user 2026-11-01 16:50, claude/core-requests.md
item 8: "Region 6: Hideyoshi's Keep (boss Hideyoshi) - extend --full"]. Both recordings took the Keep after Hot Springs'
shop (collect/20261028-190026 f1379, collect/20261029-160032 f1532): one wave level of 7 cells and 10 waves, where the Shadow
Charger comes, then Hideyoshi's 7-cell arena [wiki: Hideyoshi's Keep: '1 Level and 1 Boss Level']. His fights: rec1
f1526-f1592, rec2 f1689-f1735. Frame tests skip when the frames are missing."""

from test_bot_flow import DATA, _stub_windows_modules, frame

from shogunbot import behaviour, rules
from shogunbot.bridge import Tracker
from shogunbot.catalogue import CAT
from shogunbot.model import Enemy, Hero, State
from shogunbot.rules import Events, _enemy_fire, step

_stub_windows_modules()

REC1 = "collect/20261028-190026/"
REC2 = "collect/20261029-160032/"
ROUTE1 = ("bamboo_grove", "whispering_caves", "moonlit_port", "spirit_gateway", "hot_springs", "hideyoshis_keep")
ROUTE2 = ("bamboo_grove", "hibiku_wastelands", "moonlit_port", "spirit_gateway", "hot_springs", "hideyoshis_keep")
O, S, E, B, M, P, D, X = "origin_of_symmetry", "swirl", "earth_impale", "back_charge", "mirror", "spear", "dash", "bomb"


def _f(rec: str, n: int):
    """Frame n of a recording, whatever input its name carries."""
    import glob
    import pytest
    hit = glob.glob(f"{DATA}/{rec}f{n:04d}_*.png")
    if not hit:
        pytest.skip(f"recorded frame not available: {rec}f{n:04d}")
    return frame(rec + hit[0].split("/")[-1])


def _p(rec, n, route=ROUTE1):
    from shogunbot.perception.perceive import perceive
    return perceive(_f(rec, n), ["swirl", "arrow"], regions=route, day=1)


# ---------------------------------------------------------------- the route
def test_the_map_names_the_keep_and_the_bot_goes_there():
    from shogunbot import bot as Bt
    from shogunbot.perception import map_screen
    sel = map_screen.selected(_f(REC1, 1379))       # 'HIDEYOSHI?S KEEP': the apostrophe reads as '?'
    assert (sel["kind"], sel["region"]) == ("region", "hideyoshis_keep")
    assert "hideyoshis_keep" in Bt.MAP_REGIONS and Bt.FULL_LAST_STAGE == 6


def test_the_keep_has_one_wave_level_then_hideyoshi():
    tr = Tracker()
    tr.route = list(ROUTE1)
    tr.stage, tr.level_i = 6, -1
    names = []
    for screen, cells in (("combat", 7), ("boss", 7)):
        if names:
            tr.next_level()
        names.append(tr._level(screen, cells))
    assert names == ["stage6-1", "stage6-boss"]
    assert CAT.boss_by_arena("hideyoshis_keep", 7) == "hideyoshi"
    assert CAT.enemy("hideyoshi").hp == 45 and {"quick", "unfreezable"} <= set(CAT.enemy("hideyoshi").traits)


# ---------------------------------------------------------------- the simulator
def _st(hx, hf, ex, ef, q, intent=None, last="queue", fage=None, others=(), bombs=None, hp=10, boss_hp=45):
    e = Enemy("hideyoshi", ex, ef, boss_hp, list(q), uid=1, intent=intent, last=last, fage=fage)
    return State(7, Hero(hx, hf, hp=hp), [e] + list(others), level="stage6-boss", bombs=bombs or [])


def _top(s, action="wait"):
    return max(step(s, action), key=lambda t: t[0])


def _boss(s):
    return next(e for e in s.enemies if e.kind == "hideyoshi")


def test_origin_takes_the_centre_from_the_hero_and_the_swirl_follows():
    # rec2 f1702-f1703: the hero on cell 3, Hideyoshi on 4 facing him; Origin put Hideyoshi on 3 and the hero on 4, and the
    # Swirl hit the hero there (his Kami Brew took it); the Earth Impale reached cells 1 and 5
    _, s2, ev = _top(_st(3, 1, 4, -1, [O, S, E], intent="fire"))
    assert (_boss(s2).x, _boss(s2).facing, s2.hero.x, s2.hero.hp) == (3, -1, 4, 8)
    # rec2 f1729-f1730: the hero on the centre, Hideyoshi on the wall: the hero lands on the wall, out of reach
    _, s2, _ = _top(_st(3, 1, 6, -1, [O, S, E], intent="fire"))
    assert (_boss(s2).x, s2.hero.x, s2.hero.hp) == (3, 6, 10)


def test_back_charge_mirror_spear_lands_on_the_far_wall_facing_back_and_spears_the_hero():
    # rec2 f1711-f1713: Hideyoshi on 3 facing the hero on 4: Back Charge to the wall (cell 0), Mirror to 6 facing left, the
    # Spear's two cells reach the hero on 4 (11 -> 9)
    _, s2, ev = _top(_st(4, -1, 3, 1, [B, M, P], intent="fire"))
    assert (_boss(s2).x, _boss(s2).facing, s2.hero.hp) == (6, -1, 8)


def test_back_charge_strikes_the_first_unit_behind_and_stops_beside_it():
    # rec1 f1566-f1568: on 3 facing the hero on 6, his Back Charge struck the Yari Master on cell 0 (4 -> 3) and he stopped
    # on 1; Mirror then put him on 5 facing left, and the Spear hit cells 4 and 3, nothing
    ym = Enemy("yari_master", 0, 1, 4, [], uid=2, intent="hold")
    _, s2, _ = _top(_st(6, -1, 3, 1, [B, M, P], intent="fire", others=[ym]))
    assert (_boss(s2).x, _boss(s2).facing, s2.hero.hp) == (5, -1, 10)
    assert next(e for e in s2.enemies if e.kind == "yari_master").hp == 3


def test_swirl_mirror_swirl_hits_both_ends():
    # rec1 f1579-f1582: Hideyoshi on 5 facing the hero on 6, the Yari Master on 0 at 3 HP: the first Swirl hit the hero
    # (5 -> 3), Mirror put Hideyoshi on 1 facing left, the second Swirl hit the Yari Master (3 -> 1)
    ym = Enemy("yari_master", 0, 1, 3, [], uid=2, intent="hold")
    _, s2, _ = _top(_st(6, -1, 5, 1, [S, M, S], intent="fire", others=[ym], hp=5))
    assert (_boss(s2).x, _boss(s2).facing, s2.hero.hp) == (1, -1, 3)
    assert next(e for e in s2.enemies if e.kind == "yari_master").hp == 1


def test_dash_bomb_back_charge_and_the_bomb_two_phases_later():
    # rec2 f1717-f1726: Hideyoshi on 6 facing the hero on 5: the Dash went nowhere, the bomb landed under the hero, the
    # Back Charge met the wall; the hero stepped left twice and the bomb exploded on cells 4-6, hitting Hideyoshi (3)
    _, s2, _ = _top(_st(5, 1, 6, -1, [D, X, B], intent="fire", boss_hp=7))
    assert (_boss(s2).x, s2.hero.x, s2.hero.hp, s2.bombs) == (6, 5, 10, [[5, 2]])
    _, s3, _ = _top(s2, "left")
    assert s3.bombs == [[5, 1]] and _boss(s3).hp == 7
    _, s4, ev = _top(s3, "left")
    assert s4.bombs == [] and _boss(s4).hp == 4 and s4.hero.hp == 10
    # rec1 f1535-f1540: from cell 4 facing the hero on 2, the Dash stopped on 3, the bomb landed under the hero, the Back
    # Charge took Hideyoshi to the wall; a hero who stays on the bomb's cell or beside it takes 3
    _, s2, _ = _top(_st(2, 1, 4, -1, [D, X, B], intent="fire"))
    assert (_boss(s2).x, s2.bombs) == (6, [[2, 2]])
    s2.enemies[0].intent = "hold"
    _, s3, _ = _top(s2, "right")
    s3.enemies[0].intent = "hold"
    _, s4, _ = _top(s3, "wait")
    assert s4.bombs == [] and s4.hero.hp == 7                      # the hero on 3, beside the bomb on 2


def test_mirror_onto_a_unit_changes_places_with_it():
    # [wiki: Hideyoshi the Cunning, strategy: 'Backward Charge or Mirror, which teleport the hero to his position'] [hyp]
    s = _st(6, -1, 0, 1, [S, M, S], intent="fire")
    _, s2, _ = _top(s)
    assert (_boss(s2).x, _boss(s2).facing, s2.hero.x, s2.hero.hp) == (6, -1, 0, 10)


def test_his_rules():
    # the fight's first phase he rests (rec1 f1529, rec2 f1692); a whole combo fires the phase after its last tile (Quick);
    # part of one gets its next tile; nothing loaded and facing away: he turns (rec2 f1704); else he loads
    first = _st(2, 1, 4, -1, [], last=None)
    assert behaviour.intents(first, first.enemies[0]) == [(1.0, "hold")]
    for q, want in (([O, S, E], "fire"), ([S, M], "queue"), (["summon_double"], "fire"), ([D], "queue")):
        s = _st(2, 1, 4, -1, q)
        assert behaviour.intents(s, s.enemies[0]) == [(1.0, want)], q
    away = _st(5, 1, 4, -1, [], last="fire", fage=0)
    assert behaviour.intents(away, away.enemies[0]) == [(1.0, "turn")]
    # the next tile of a combo; a new combo is never the one he used last [wiki]
    s = _st(2, 1, 4, -1, [B, M])
    assert rules._boss_queue(s, s.enemies[0], None) == [(1.0, ("queue", P))]
    s = _st(2, 1, 4, -1, [], last="fire", fage=0)
    s.enemies[0].loaded = D
    opts = rules._boss_queue(s, s.enemies[0], None)
    assert {a[1] for _, a in opts} == {O, B, S, "summon_double"} and abs(sum(p for p, _ in opts) - 1) < 1e-9
    # its first tile names the combo he uses
    s = _st(2, 1, 4, -1, [], intent="queue", last="hold")
    for p, s2, _ in step(s, "wait"):
        assert _boss(s2).loaded == _boss(s2).queue[0]


def test_he_is_unfreezable():
    from shogunbot.model import Kit, UNFREEZABLE_KINDS
    assert "hideyoshi" in UNFREEZABLE_KINDS
    kit = Kit.of("swirl", "arrow")
    kit = kit.replace("swirl", kit["swirl"].__class__(**{**kit["swirl"].__dict__, "enchant": "ice"}))
    s = _st(3, 1, 4, -1, [D])
    s.hero.kit, s.hero.queue = kit, ["swirl"]
    s.hero.charge = {t.label: t.cooldown for t in kit}
    _, s2, _ = _top(s, "attack")
    assert _boss(s2).frozen == 0 and _boss(s2).hp == 43


def test_the_shadow_charger_loads_charge_then_sword_and_fires_the_phase_after():
    # rec1 f1414-f1420, rec2 f1668-f1677: Charge, then the Sword over it, then fire with no red phase (Quick); the Charge
    # dashes to the first unit ahead and strikes it (1), the Sword the cell ahead (2)
    def mk(q, last="queue"):
        return State(7, Hero(3, -1, hp=10), [Enemy("shadow_charger", 0, 1, 5, list(q), uid=1, last=last)], level="stage6-1")
    s = mk([], last="turn")
    assert behaviour.intents(s, s.enemies[0]) == [(1.0, "queue")]
    s = mk(["charge"])
    assert behaviour.intents(s, s.enemies[0]) == [(1.0, "queue")]
    s = mk(["charge", "sword"])
    assert behaviour.intents(s, s.enemies[0]) == [(1.0, "fire")]
    s.enemies[0].intent = "fire"
    _, s2, ev = _top(s)
    assert (s2.enemies[0].x, s2.hero.hp) == (2, 7)


# ---------------------------------------------------------------- perception
def _boss_read(rec, n, route):
    p = _p(rec, n, route)
    e = next(e for e in p.enemies if e["kind"] == "hideyoshi")
    return CAT.enemy("hideyoshi").chain_fix(e["queue"]), e


def test_his_icons_read_on_frames_outside_the_samples():
    # (perception/build_icons.SAMPLES holds other frames of the same fights)
    for rec, n, route, want in ((REC2, 1699, ROUTE2, [O, S, E]), (REC2, 1709, ROUTE2, [B, M]),
                                (REC2, 1727, ROUTE2, [O, S]), (REC1, 1548, ROUTE1, [D, X]),
                                (REC1, 1577, ROUTE1, [S, M, S]), (REC1, 1584, ROUTE1, [O])):
        q, e = _boss_read(rec, n, route)
        assert q == want, (rec, n, e["queue"])


def test_a_five_pip_unit_with_a_charge_at_the_bottom_is_a_shadow_charger():
    # rec1 f1416: the Shadow Charger on cell 3 with its Charge; rec2 f1676: Charge and Sword (0.46 read both as Guardians)
    p = _p(REC1, 1416)
    assert next(e for e in p.enemies if e["x"] == 3)["kind"] == "shadow_charger"
    p = _p(REC2, 1676, ROUTE2)
    e = next(e for e in p.enemies if e["x"] == 3)
    assert (e["kind"], e["queue"]) == ("shadow_charger", ["charge", "sword"])


def test_the_red_bomb_is_read_on_the_floor_and_spawn_beams_are_not():
    assert _p(REC1, 1536).bombs == [2] and _p(REC2, 1719, ROUTE2).bombs == [5]
    assert _p(REC1, 1465).bombs == [] and _p(REC2, 1576, ROUTE2).bombs == []     # spawn beams on three cells
    assert _p(REC1, 1537).bombs == []                                           # its black phase is not read


# ---------------------------------------------------------------- the tracker
def test_the_tracker_follows_his_combos_his_facing_and_his_bomb():
    """rec2 f1711-f1728, the bot's inputs replayed as waits and steps (the hero's own queue does not matter here)."""
    tr = Tracker()
    tr.route = list(ROUTE2)
    tr.stage, tr.level_i, tr.advance = 6, 0, True
    steps = ((1711, "wait"), (1713, "wait"), (1715, "wait"), (1716, "wait"), (1717, "wait"),
             (1722, "left"), (1725, "left"), (1728, "wait"))
    seen = {}
    for n, act in steps:
        s = tr.state(_p(REC2, n, ROUTE2))
        h = _boss(s)
        seen[n] = (h.x, h.facing, h.intent, list(h.queue), [list(b) for b in s.bombs])
        tr.acted(s, act)
    assert seen[1711][2:4] == ("fire", [B, M, P])
    assert seen[1713][:4] == (6, -1, "queue", [])          # Mirror turned him: he faces the hero, no turn first
    assert seen[1717][2:4] == ("fire", [D, X, B])
    assert seen[1722][4] == [[5, 2]] and seen[1725][4] == [[5, 1]] and seen[1728][4] == []


def test_a_stack_read_short_after_a_plus_keeps_the_combo():
    """rec1 f1554-f1555: [Origin, Swirl] and a '+' on cell 0, then nothing read at the screen's left edge: he holds the
    Earth Impale too and fires (he never drops a combo)."""
    tr = Tracker()
    tr.route = list(ROUTE1)
    tr.stage, tr.level_i, tr.advance = 6, 0, True
    s = tr.state(_p(REC1, 1554))
    assert (_boss(s).queue, _boss(s).intent) == ([O, S], "queue")
    tr.acted(s, "wait")
    s = tr.state(_p(REC1, 1555))
    assert (_boss(s).queue, _boss(s).intent) == ([O, S, E], "fire")
