"""0.45: --full goes on to region 5, Hot Springs and Kowa the Coward [user 2026-10-31, claude/core-requests.md: "Region 5
(Hot Springs, Kowa): FULL_LAST_STAGE 5, add hot_springs to MAP_REGIONS; Shielder (Shield / Ally Shield: a shield negates
the next hit) in the simulator; Kowa's data"]. Both recordings took Hot Springs after Spirit Gateway's shop
(collect/20261028-190026 f0944, collect/20261029-160032 f1148): three wave levels of 7, 5 and 9 cells, then Kowa's
7-cell arena. Frame tests skip when the frames are missing."""

from test_bot_flow import DATA, _stub_windows_modules, frame

from shogunbot.bridge import Tracker

_stub_windows_modules()

REC1 = "collect/20261028-190026/"
REC2 = "collect/20261029-160032/"
ROUTE1 = ("bamboo_grove", "whispering_caves", "moonlit_port", "spirit_gateway", "hot_springs")
ROUTE2 = ("bamboo_grove", "hibiku_wastelands", "moonlit_port", "spirit_gateway", "hot_springs")


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


def test_the_map_names_hot_springs_and_the_bot_prefers_it():
    from shogunbot import bot as B
    from shogunbot.perception import map_screen as M
    for rec, n in ((REC1, 944), (REC2, 1148)):                  # the tooltip over the node after Spirit Gateway's shop
        sel = M.selected(_f(rec, n))
        assert (sel["kind"], sel["region"]) == ("region", "hot_springs")
    assert "hot_springs" in B.MAP_REGIONS and B.FULL_LAST_STAGE >= 5          # region 6 since 0.47


def test_hot_springs_counts_three_wave_levels_before_kowa():
    tr = Tracker()
    tr.route = list(ROUTE1)
    tr.stage, tr.level_i = 5, -1
    names = []
    for screen, cells in (("combat", 7), ("combat", 5), ("combat", 9), ("boss", 7)):
        if names:
            tr.next_level()
        names.append(tr._level(screen, cells))
    assert names == ["stage5-1", "stage5-2", "stage5-3", "stage5-boss"]


# ---------------------------------------------------------------- the simulator: shields
from shogunbot import behaviour
from shogunbot.model import Enemy, Hero, Kit, State
from shogunbot.planner import _queue_damage, evaluate
from shogunbot.rules import Events, _enemy_fire, _poison_tick, step


def test_a_shield_negates_the_next_hit_and_is_gone():
    # Shield: 'Gains a shield that negates the next attack received' [wiki: Tiles]
    s = State(7, Hero(1, 1, queue=["arrow"]), [Enemy("ashigaru", 4, -1, 3, uid=1, shield=True)])
    for _, s2, ev in step(s, "attack"):
        a = s2.enemies[0]
        assert (a.hp, a.shield) == (3, False) and not ev.enemy_damage
    s2.hero.queue = ["arrow"]                                     # the next hit lands
    for _, s3, ev in step(s2, "attack"):
        assert s3.enemies[0].hp == 1 and ev.enemy_damage


def test_a_poison_tick_breaks_a_shield():
    # 'Poison is especially great as it can immediately break his shield' [wiki: Kowa the Coward]
    e = Enemy("ashigaru", 4, -1, 3, uid=1, shield=True, poison=2)
    s = State(7, Hero(1, 1), [e])
    _poison_tick(s, Events())
    assert (e.hp, e.shield, e.poison) == (3, False, 1)
    _poison_tick(s, Events())
    assert e.hp == 2


def test_a_shielded_hit_brings_no_enchantment_and_the_planner_counts_the_shield():
    from shogunbot.model import KitTile
    kit = Kit.of("swirl", "arrow")
    kit = kit.replace("arrow", KitTile("arrow", "arrow", 2, 5, "poison"))
    s = State(7, Hero(1, 1, queue=["arrow"], kit=kit), [Enemy("ashigaru", 4, -1, 3, uid=1, shield=True)])
    for _, s2, _ in step(s, "attack"):
        assert s2.enemies[0].poison == 0                          # HYP: the shield negates the whole attack
    assert _queue_damage(s, ["arrow"]) == {}                      # the shield takes the loaded Arrow
    s.enemies[0].shield = False
    assert _queue_damage(s, ["arrow"]) == {1: 2}
    v_off = evaluate(s)[0]
    s.enemies[0].shield = True
    assert evaluate(s)[1]["enemy_shield"] < 0 and evaluate(s)[0] < v_off


# ---------------------------------------------------------------- the simulator: Kowa the Coward
def _kowa(x=4, facing=-1, **kw):
    return Enemy("kowa", x, facing, 30, kw.pop("queue", []), uid=9, **kw)


def _intent(s, e):
    return behaviour.intents(s, e)


def test_kowa_steps_away_while_he_can_then_turns_to_the_hero_then_loads():
    # [rec: collect/20261028-190026] f1277-f1278: cell 4 -> 5 away from the hero on 2, still facing left; f1297: 2 -> 1
    k = _kowa(4, -1, last="hold", fage=None)
    s = State(7, Hero(2, 1), [k], level="stage5-boss")
    assert _intent(s, k) == [(1.0, "move-right")]
    k.x = 6                                                      # the wall: loads, facing the hero
    assert _intent(s, k) == [(1.0, "queue")]
    s = State(7, Hero(4, -1), [_kowa(1, -1, last="move"), Enemy("ashigaru_archer", 0, 1, 2, uid=1)])
    assert _intent(s, s.enemies[0]) == [(1.0, "turn")]          # f1301-f1302: blocked by an ally, facing away
    k = _kowa(5, -1, last="hold")                                # an ally beyond him blocks the retreat (f1279)
    s = State(7, Hero(2, 1), [k, Enemy("ashigaru", 6, -1, 3, uid=1)])
    assert _intent(s, k) == [(1.0, "queue")]


def test_kowa_alternates_shield_and_smoke_bomb_whatever_his_shield():
    # 11 of 11 loads: Shield first (f1279, f1302, f1315, f1332, f1351; rec2 f1479), then Smoke Bomb, also when the hero
    # had just broken the shield (f1286, rec2 f1490)
    k = _kowa(6, -1, intent="queue")
    s = State(7, Hero(2, 1), [k], level="stage5-boss")
    (_, s2, _), = step(s, "wait")
    assert s2.enemies[0].queue == ["shield"] and s2.enemies[0].loaded == "shield"
    k = _kowa(6, -1, intent="queue", loaded="shield")
    (_, s2, _), = step(State(7, Hero(2, 1), [k]), "wait")
    assert s2.enemies[0].queue == ["smoke_bomb"]


def test_kowa_uses_a_tile_from_the_second_phase_and_rests_after():
    k = _kowa(6, -1, queue=["shield"], last="queue", wind=0, loaded="shield")
    s = State(7, Hero(2, 1), [k], level="stage5-boss")
    assert _intent(s, k) == [(1.0, "hold")]                      # R1: the phase after loading (f1280-f1281)
    k.last, k.wind = "hold", 1
    assert _intent(s, k) == [(1.0, "fire")]                      # the Shield always (f1283-f1284)
    k.intent = "fire"
    (_, s2, _), = step(s, "wait")
    k2 = s2.enemies[0]
    assert k2.shield and not k2.queue and k2.fage == 0
    k2.intent = None
    assert _intent(s2, k2) == [(1.0, "hold")]                    # rests after a use (12 of 12)


def test_kowas_smoke_bomb_waits_for_a_clear_lane_then_swaps_with_the_hero():
    # rec2 f1491-f1499: held four phases while a Guardian stood between them, fired once the lane was clear: Kowa and
    # the hero changed places, both kept their facing, the hero lost 1 (also rec1 f1295-f1296, f1342-f1343)
    k = _kowa(6, -1, queue=["smoke_bomb"], last="hold", loaded="smoke_bomb")
    s = State(7, Hero(3, 1), [k, Enemy("guardian", 5, -1, 5, uid=1)], level="stage5-boss")
    assert _intent(s, k) == [(1.0, "hold")]
    s = State(7, Hero(4, 1), [_kowa(6, -1, queue=["smoke_bomb"], last="hold", loaded="smoke_bomb")])
    assert _intent(s, s.enemies[0]) == [(1.0, "fire")]
    s.enemies[0].intent = "fire"
    (_, s2, _), = step(s, "wait")
    assert (s2.hero.x, s2.hero.hp, s2.hero.facing, s2.enemies[0].x, s2.enemies[0].facing) == (6, 9, 1, 4, -1)


def test_kowas_shield_takes_the_first_hit_of_a_queue_and_his_allies_vanish_when_he_dies():
    # rec1 f1311-f1312: two Swirls, the first absorbed, the second -4 (4-damage Swirls); f1353-f1354: he dies, the
    # Archer and the Ashigaru vanish and the fight ends
    from shogunbot.model import KitTile
    kit = Kit((KitTile("swirl", "swirl", 4, 3), KitTile("swirl#2", "swirl", 4, 3)))
    s = State(7, Hero(1, -1, queue=["swirl", "swirl#2"], kit=kit), [_kowa(2, 1, shield=True)], level="stage5-boss")
    for _, s2, _ in step(s, "attack"):
        assert s2.enemies[0].hp == 26 and not s2.enemies[0].shield
    s = State(7, Hero(1, -1, queue=["swirl"], kit=kit),
              [Enemy("kowa", 2, 1, 3, uid=9), Enemy("ashigaru_archer", 4, -1, 2, uid=1),
               Enemy("ashigaru", 6, -1, 3, ["sword"], uid=2)], level="stage5-boss")
    out = step(s, "attack")
    assert all(not s2.alive() and s2.hero.hp == 10 for _, s2, _ in out)


def _boss_frame(hero_x: int, kowa: dict, others=(), frac: float = 0.3):
    from shogunbot.perception.perceive import Perception
    cells = 7
    ens = [dict({"kind": "kowa", "hp": 0, "hp_max": 0}, **kowa)] + list(others)
    return Perception(screen="boss", cells=[{"x": i, "cx": 300 + 140 * i, "kind": "plate"} for i in range(cells)],
                      hero={"x": hero_x, "cx": 300 + 140 * hero_x, "facing": -1, "hp": 4, "hp_max": 10, "queue": [],
                            "swap_red": 0},
                      enemies=[dict({"cx": 300 + 140 * e["x"], "queue": [], "queue_conf": [], "intent": None,
                                     "facing": 0, "alt": None, "elite": False, "spawning": False, "shield": False}, **e)
                               for e in ens],
                      boss_frac=frac)


def test_the_tracker_plays_kowas_rules_where_the_screen_says_nothing():
    # rec1 T34-T37 (f1332-f1335): Kowa on the wall cell 0 facing the hero loads his Shield ('+'), holds it (red), casts
    # it on a red frame (a use turn's bubble blinks yellow and red: f1334 read red), then rests with the cast's leftover
    # bubble read as '+' (f1284, f1311 alike)
    from shogunbot.model import Kit
    tr = Tracker(Kit.of("swirl", "arrow"))
    tr.route = list(ROUTE1)
    tr.stage, tr.level_i, tr.advance, tr.level = 5, 2, True, "stage5-3"
    seq = [({"x": 0, "intent": "queue"}, "queue"),
           ({"x": 0, "queue": ["shield"], "queue_conf": [1.0]}, "hold"),
           ({"x": 0, "queue": ["shield"], "queue_conf": [1.0]}, "fire"),
           ({"x": 0, "intent": "queue", "shield": True}, "hold")]
    for kowa, want in seq:
        s = tr.state(_boss_frame(3, kowa))
        k = next(e for e in s.enemies if e.kind == "kowa")
        assert (k.facing, k.intent) == (1, want), (kowa, k.intent)
        tr.acted(s, "wait")
    assert s.level == "stage5-boss" and k.shield and k.loaded == "shield" and k.hp == 9
    # his next load is the Smoke Bomb
    k.intent = "queue"
    k.fage = None
    (_, s2, _), = [o for o in step(s, "wait")]
    assert next(e for e in s2.enemies if e.kind == "kowa").queue == ["smoke_bomb"]


# ---------------------------------------------------------------- the Shielder
def test_ally_shield_goes_to_the_first_shieldless_ally_ahead_past_the_hero_and_shielded_allies():
    # rec1 f1063-f1064: the Shielder on cell 4 facing left shielded the Charger on cell 1 past the hero on cell 2;
    # f1204-f1205: it passed the shielded Yari Master on cell 6 for the one on cell 4
    sh = Enemy("shielder", 4, -1, 4, ["ally_shield"], uid=1)
    ch = Enemy("charger", 1, 1, 1, uid=2)
    s = State(7, Hero(2, 1), [sh, ch])
    _enemy_fire(s, sh, Events())
    assert ch.shield and not sh.shield
    sh = Enemy("shielder", 7, -1, 4, ["ally_shield"], uid=1)
    y6, y4 = Enemy("yari_master", 6, -1, 4, uid=2, shield=True), Enemy("yari_master", 4, -1, 4, uid=3)
    s = State(9, Hero(1, 1), [sh, y6, y4])
    _enemy_fire(s, sh, Events())
    assert y4.shield and y6.shield and not sh.shield
    sh = Enemy("shielder", 0, 1, 4, ["shield"], uid=1)    # Shield: on itself (rec1 f1118-f1119)
    s = State(7, Hero(3, 1), [sh])
    _enemy_fire(s, sh, Events())
    assert sh.shield and s.hero.hp == 10


def test_the_shielder_loads_holds_fires_rests_and_picks_its_tile():
    # rec2's turn counter: '+', one phase red, then it fires (f1156-f1161, f1292-f1296), then a phase of rest
    a = Enemy("ashigaru", 2, 1, 3, uid=2)
    sh = Enemy("shielder", 5, -1, 4, uid=1, last="turn")
    s = State(7, Hero(0, 1), [sh, a])
    assert behaviour.intents(s, sh) == [(1.0, "queue")]                   # a shieldless ally ahead
    sh.intent = "queue"
    (_, s2, _), = step(s, "wait")
    sh2 = next(e for e in s2.enemies if e.kind == "shielder")
    assert sh2.queue == ["ally_shield"]
    sh2.intent = None
    assert behaviour.intents(s2, sh2) == [(1.0, "hold")]                  # R1
    sh2.last, sh2.wind = "hold", 1
    assert behaviour.intents(s2, sh2) == [(1.0, "fire")]
    sh2.queue, sh2.fage = [], 0
    assert behaviour.intents(s2, sh2) == [(1.0, "hold")]                  # rests after a use
    sh2.fage = 1
    a2 = next(e for e in s2.enemies if e.kind == "ashigaru")
    a2.shield = True                                                      # nothing ahead to shield, unshielded: Shield
    assert behaviour.intents(s2, sh2) == [(1.0, "queue")]
    sh2.intent = "queue"
    sh2.queue = []
    out = step(s2, "wait")
    assert {tuple(e.queue) for _, s3, _ in out for e in s3.enemies if e.kind == "shielder"} == {("shield",)}
    sh2.intent, sh2.shield = None, True                                   # shielded, nothing to shield: it turns
    assert behaviour.intents(s2, sh2) == [(1.0, "turn")]
    a2.shield, a2.x, sh2.facing = False, 6, -1                            # a shieldless ally behind: it turns to it
    assert behaviour.intents(s2, sh2) == [(1.0, "turn")]


def test_hot_springs_icons_shields_and_the_fire_orb_read_on_frames_outside_the_samples():
    def units(rec, n, route):
        return {e["x"]: (e["kind"], e["queue"], e["shield"], e["orb"]) for e in _p(rec, n, route).enemies}
    u = units(REC1, 1052, ROUTE1)
    assert u[4][:2] == ("shielder", ["ally_shield"]) and u[2][2]           # and the Tachi it shielded
    assert units(REC1, 1177, ROUTE1)[0] == ("shielder", ["ally_shield"], False, True)    # yellow, the fire orb
    assert units(REC1, 1217, ROUTE1)[7][:2] == ("shielder", ["shield"])
    assert units(REC2, 1330, ROUTE2)[4][:2] == ("shielder", ["ally_shield"])
    assert units(REC2, 1164, ROUTE2)[6] == ("shielder", ["shield"], False, True)
    assert units(REC1, 1317, ROUTE1)[3][:2] == ("kowa", ["shield"])       # Kowa's Shield, red
    assert units(REC1, 1310, ROUTE1)[1][1:3] == (["smoke_bomb"], True)    # his Smoke Bomb, and his sphere
    assert not units(REC1, 1283, ROUTE1)[5][2]                            # unshielded
    # shields on the hand-labelled frames of the survey: rec1 f1113 cell 2, f1205 cells 4 and 6; rec2 f1319 cell 4
    assert units(REC1, 1113, ROUTE1)[2][2] and not units(REC1, 1110, ROUTE1).get(2, (0, 0, False))[2]
    assert units(REC1, 1205, ROUTE1)[4][2] and units(REC1, 1205, ROUTE1)[6][2]
    assert units(REC2, 1319, ROUTE2)[4][2] and not units(REC2, 1316, ROUTE2)[4][2]


def test_shields_are_read_only_on_a_route_with_a_shield_giver():
    # Spirit Gateway's route has none: the same frame reads no shield and no orb there
    route4 = ROUTE1[:4]
    assert all(not e["shield"] and not e["orb"] for e in _p(REC1, 1113, route4).enemies)


def test_the_tracker_fires_a_shielder_on_its_orb_or_its_rule_and_rests_it_after():
    # rec1 f1176-f1178: the bubble blinks, the orb at its chest shows the fire; rec2's counter: load, red, fire, rest
    from shogunbot.perception.perceive import Perception

    def frame_(sh: dict):
        ens = [dict({"x": 5, "kind": "shielder", "hp": 4, "hp_max": 4}, **sh), {"x": 2, "kind": "ashigaru", "hp": 3,
                                                                                 "hp_max": 3}]
        return Perception(screen="combat", cells=[{"x": i, "cx": 300 + 140 * i, "kind": "plate"} for i in range(7)],
                          hero={"x": 0, "cx": 300, "facing": 1, "hp": 10, "hp_max": 10, "queue": [], "swap_red": 0},
                          enemies=[dict({"cx": 300 + 140 * e["x"], "queue": [], "queue_conf": [], "intent": None,
                                         "facing": -1, "alt": None, "elite": False, "spawning": False, "shield": False,
                                         "orb": False}, **e) for e in ens])
    tr = Tracker()
    tr.route = list(ROUTE1)
    tr.stage, tr.level_i, tr.advance, tr.level = 5, 0, True, "stage4-boss"
    seq = [({"intent": "queue"}, "queue"),
           ({"queue": ["ally_shield"], "queue_conf": [1.0]}, "hold"),               # R1
           ({"queue": ["ally_shield"], "queue_conf": [1.0], "orb": True}, "fire"),  # the orb
           ({"intent": "queue"}, "hold")]                                           # rests; the '+' is a phantom
    for sh, want in seq:
        s = tr.state(frame_(sh))
        e = next(x for x in s.enemies if x.kind == "shielder")
        assert e.intent == want, (sh, e.intent)
        tr.acted(s, "wait")
    # without the orb, a tile loaded earlier fires on the rule
    for sh, want in (({"intent": "queue"}, "queue"), ({"queue": ["ally_shield"], "queue_conf": [1.0]}, "hold"),
                     ({"queue": ["ally_shield"], "queue_conf": [1.0]}, "fire")):
        s = tr.state(frame_(sh))
        assert next(x for x in s.enemies if x.kind == "shielder").intent == want
        tr.acted(s, "wait")


# ---------------------------------------------------------------- 0.46: Hot Springs III's long stall (core-requests 5)
RUN_053431 = "runs/20261101-053431/frames/"
RUN_021522 = "runs/20261101-021522/frames/"
R5 = ("bamboo_grove", "whispering_caves", "moonlit_port", "spirit_gateway", "hot_springs")


def _run_frame(rel: str):
    import os
    import pytest
    if not os.path.exists(f"{DATA}/{rel}"):
        pytest.skip(f"run frame not available: {rel}")
    return frame(rel)


def _units(rel: str, route=R5):
    from shogunbot.perception.perceive import perceive
    return {e["x"]: e for e in perceive(_run_frame(rel), regions=route).enemies}


def test_no_plus_on_the_lava_backdrop_and_the_archers_icons_read_beside_their_centre():
    # run 20261101-053431 Hot Springs III: the Shielder on cell 0 has no bubble, but the lava behind it matched the red
    # frame on one side and its fill read as a '+' (t1062, t1065: 277 'armed' mispredictions in the level); the two
    # Archers' Arrows sit 4 px right of their cells' centres when the board pans, and read as Spears (Tachi) there
    for t in (1062, 1065):
        u = _units(f"{RUN_053431}t{t}_before.png")
        assert u[0]["intent"] is None and u[0]["queue"] == []
        assert [(u[x]["kind"], u[x]["queue"]) for x in (1, 2)] == [("ashigaru_archer", ["arrow"])] * 2
    # real '+' bubbles still read: white (t0672, the Shielder about to load) and yellow (t0985)
    assert _units(f"{RUN_053431}t0672_before.png")[2]["intent"] == "queue"
    assert _units(f"{RUN_053431}t0985_before.png")[5]["intent"] == "queue"
    # the Shielder's Ally Shield 5 px beside its centre read as Dash (run 20261101-021522 t0769)
    assert _units(f"{RUN_021522}t0769_before.png")[5]["queue"] == ["ally_shield"]


def test_ally_shield_is_held_while_no_shieldless_ally_is_ahead():
    # run 20261101-021522 t0767-t0792: the Shielder on cell 5 facing right with Ally Shield, the hero on 8, its one ally
    # (an Archer on 2, behind it) shielded: it held the tile 26 phases, its bubble blinking and its orb lit
    sh = Enemy("shielder", 5, 1, 2, ["ally_shield"], uid=1, last="hold", wind=1, fage=3)
    ar = Enemy("ashigaru_archer", 2, 1, 2, ["arrow"], uid=2, shield=True)
    s = State(9, Hero(8, -1), [sh, ar])
    assert behaviour.intents(s, sh) == [(1.0, "hold")]
    ar.x = 7                                                            # a shieldless ally ahead: it fires
    ar.shield = False
    assert behaviour.intents(s, sh) == [(1.0, "fire")]


def _hs_frame(units: list[dict], hero_x: int, cells: int = 5):
    from shogunbot.perception.perceive import Perception
    return Perception(screen="combat", cells=[{"x": i, "cx": 300 + 140 * i, "kind": "plate"} for i in range(cells)],
                      hero={"x": hero_x, "cx": 300 + 140 * hero_x, "facing": 1, "hp": 6, "hp_max": 12, "queue": [],
                            "swap_red": 0},
                      enemies=[dict({"cx": 300 + 140 * e["x"], "queue": [], "queue_conf": [], "intent": None,
                                     "facing": 0, "alt": None, "elite": False, "spawning": False, "shield": False,
                                     "orb": False}, **e) for e in units])


def test_the_tracker_follows_the_shielder_through_its_name_and_its_turn():
    # run 20261101-053431 t0670-t0675 (Hot Springs II, 5 cells): the hero on 1, a 4-pip unit on 2 (the Shielder, read as
    # a Yari Master until its Ally Shield showed) and a shieldless Yari Master on 4. t0671: a turn telegraph, toward
    # the Yari Master (0.45 took it as turning to the hero); t0672 '+'; t0673 Ally Shield, red, now read as a Shielder
    # (0.45 lost the '+' with the old name and fired it a phase early); t0674 the orb: it fires and the Yari Master on 4
    # is shielded at t0675
    tr = Tracker()
    tr.route = list(R5)
    tr.stage, tr.level_i, tr.advance, tr.level = 5, 0, True, "stage4-boss"
    y4 = {"x": 4, "kind": "yari_master", "hp": 1, "hp_max": 4, "facing": -1}
    seq = [({"kind": "yari_master", "intent": "turn"}, None, "turn"),          # read as a Yari Master: facing unsure
           ({"kind": "yari_master", "intent": "queue"}, None, "queue"),
           ({"kind": "shielder", "queue": ["ally_shield"], "queue_conf": [1.0]}, 1, "hold"),
           ({"kind": "shielder", "queue": ["ally_shield"], "queue_conf": [1.0], "orb": True}, 1, "fire")]
    for sh, facing, intent in seq:
        s = tr.state(_hs_frame([dict({"x": 2, "hp": 4, "hp_max": 4}, **sh), y4], 1))
        e = next(x for x in s.enemies if x.x == 2)
        assert e.intent == intent and facing in (None, e.facing), (sh, e.facing, e.intent)
        if intent != "fire":
            tr.acted(s, "wait")
    (_, s2, _), = step(s, "wait")
    assert next(x for x in s2.enemies if x.x == 4).shield
    # as a Shielder seen turning, it faces away from its one shieldless ally (not from the hero)
    tr = Tracker()
    tr.route = list(R5)
    tr.stage, tr.level_i, tr.advance, tr.level = 5, 0, True, "stage4-boss"
    s = tr.state(_hs_frame([{"x": 2, "kind": "shielder", "hp": 4, "hp_max": 4, "intent": "turn"}, y4], 1))
    assert next(x for x in s.enemies if x.x == 2).facing == -1


def test_killing_the_shielder_first_scores_above_an_equal_kill():
    # core-requests item 5 ("target the Shielder first"): with two 2-HP units left, the line that kills the Shielder
    # scores above the one that kills the Archer
    a = State(7, Hero(3, 1), [Enemy("shielder", 5, -1, 2, uid=1), Enemy("ashigaru_archer", 1, 1, 2, uid=2)])
    b = a.clone()
    a.enemies[0].hp = 0
    b.enemies[1].hp = 0
    assert evaluate(a)[0] > evaluate(b)[0]
    assert evaluate(b)[1]["shield_giver"] < 0 and "shield_giver" not in evaluate(a)[1]


def test_a_shielder_holding_ally_shield_keeps_a_move_marker_as_read():
    # run 20261101-021522 t0839 (Hot Springs III): a loaded Shielder with no shieldless ally ahead showed a move-right
    # arrow and stepped right; the no-target hold applies to its bubble reads only, as markers are kept as read
    tr = Tracker()
    tr.route = list(R5)
    tr.stage, tr.level_i, tr.advance, tr.level = 5, 0, True, "stage4-boss"
    units = [{"x": 6, "kind": "shielder", "hp": 3, "hp_max": 4, "facing": -1, "queue": ["ally_shield"],
              "queue_conf": [1.0], "intent": "move-right"},
             {"x": 8, "kind": "shielder", "hp": 1, "hp_max": 4, "facing": -1, "shield": True}]
    s = tr.state(_hs_frame(units, 3, cells=9))
    e = next(x for x in s.enemies if x.x == 6)
    assert e.intent == "move-right"
