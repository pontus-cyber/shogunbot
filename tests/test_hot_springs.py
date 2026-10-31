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
