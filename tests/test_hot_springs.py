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
    assert "hot_springs" in B.MAP_REGIONS and B.FULL_LAST_STAGE == 5


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
