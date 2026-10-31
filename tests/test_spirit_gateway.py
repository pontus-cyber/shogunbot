"""0.36: --full goes on to region 4, Spirit Gateway and The Statue [user review of run 20261030-213120: "raise
FULL_LAST_STAGE to 4 (Spirit Gateway, The Statue)"]. Both recordings took Spirit Gateway after Moonlit Port's shop
(collect/20261028-190026 f0597, collect/20261029-160032 f0798): three wave levels of 7, 7 and 9 cells, then The
Statue's 7-cell arena. Frame tests skip when the frames are missing."""

from test_bot_flow import DATA, _stub_windows_modules, frame

from shogunbot.bridge import Tracker

_stub_windows_modules()

REC1 = "collect/20261028-190026/"
REC2 = "collect/20261029-160032/"


def test_the_map_names_spirit_gateway_and_the_bot_prefers_it():
    from shogunbot import bot as B
    from shogunbot.perception import map_screen as M
    sel = M.selected(frame(REC1 + "f0597_CLICK.png"))          # the tooltip over the node after the shop
    assert (sel["kind"], sel["region"]) == ("region", "spirit_gateway")
    assert "spirit_gateway" in B.MAP_REGIONS and B.FULL_LAST_STAGE == 4


def test_spirit_gateway_counts_three_wave_levels_before_the_statue():
    tr = Tracker()
    tr.route = ["bamboo_grove", "whispering_caves", "moonlit_port", "spirit_gateway"]
    tr.stage, tr.level_i = 4, -1
    names = []
    for screen, cells in (("combat", 7), ("combat", 7), ("combat", 9), ("boss", 7)):
        if names:
            tr.next_level()
        names.append(tr._level(screen, cells))
    assert names == ["stage4-1", "stage4-2", "stage4-3", "stage4-boss"]


def test_a_quick_unit_may_fire_the_phase_after_it_loads(monkeypatch):
    # Quick [wiki: "can attack immediately after a tile is added to its attack queue"]: R1 does not hold for it
    from shogunbot import behaviour
    from shogunbot.model import Enemy, Hero, State
    s = State(5, Hero(2, 1), [Enemy("ashigaru", 3, -1, 3, ["sword"], uid=1)])
    e = s.enemies[0]
    e.wind, e.last = 0, "queue"                       # loaded in the phase just played
    assert all(it != "fire" for _, it in behaviour.intents(s, e))          # R1
    monkeypatch.setattr(behaviour, "QUICK", frozenset({"ashigaru"}))
    assert behaviour.intents(s, e) == [(1.0, "fire")]                      # R2 at once


# ---------------------------------------------------------------- the simulator
from shogunbot import behaviour
from shogunbot.model import Enemy, Hero, State
from shogunbot.rules import Events, _enemy_fire, legal_hero_actions, step


def _hp_after(s, action="wait"):
    return {s2.hero.hp for _, s2, _ in step(s, action)}


def test_the_grappling_hook_pulls_the_hero_next_to_the_grappler_and_the_sword_follows():
    # [wiki: Tiles, 'Pull the first target to the cell directly ahead'] [rec: The Statue's Hook pulled the hero from
    # cell 1 to cell 5, collect/20261028-190026 f0868-f0869]; the Grappler's Sword then hits the cell ahead
    g = Enemy("grappler", 6, -1, 5, ["grappling_hook", "sword"], uid=1, intent="fire")
    s = State(7, Hero(2, 1), [g])
    _enemy_fire(s, g, Events())
    assert (s.hero.x, s.hero.hp) == (5, 7)                 # pulled to cell 5: 1 + 2
    # an ally in the lane takes the hook instead
    g = Enemy("grappler", 6, -1, 5, ["grappling_hook", "sword"], uid=1)
    s = State(7, Hero(1, 1), [g, Enemy("ashigaru", 3, 1, 3, uid=2)])
    _enemy_fire(s, g, Events())
    a = s.enemies[1]
    assert s.hero.hp == 10 and a.x == 5 and a.hp == 0      # pulled next to it, then the Sword (1 + 2 = 3)


def test_the_grappler_loads_its_sword_right_after_its_hook_then_holds_then_fires():
    s = State(7, Hero(2, 1), [Enemy("grappler", 6, -1, 5, ["grappling_hook"], uid=1, last="queue", wind=0)])
    assert behaviour.intents(s, s.enemies[0]) == [(1.0, "queue")]
    s.enemies[0].queue = ["grappling_hook", "sword"]       # just loaded the Sword: R1
    assert all(it != "fire" for _, it in behaviour.intents(s, s.enemies[0]))
    s.enemies[0].wind, s.enemies[0].last = 1, "hold"       # the hero the first unit ahead: R2
    assert behaviour.intents(s, s.enemies[0]) == [(1.0, "fire")]
    s.enemies.append(Enemy("ashigaru", 4, 1, 3, uid=2))    # an ally in its lane: it holds
    assert all(it != "fire" for _, it in behaviour.intents(s, s.enemies[0]))
    # a queue telegraph adds the next tile of its sequence
    s = State(7, Hero(2, 1), [Enemy("grappler", 6, -1, 5, ["grappling_hook"], uid=1, intent="queue")])
    assert [e.queue for _, s2, _ in step(s, "wait") for e in s2.enemies] == [["grappling_hook", "sword"]]


def test_the_shinobi_is_quick_and_its_smoke_bomb_swaps_it_with_the_hero():
    sh = Enemy("shinobi", 5, -1, 4, ["shuriken"], uid=1, last="queue", wind=0)
    s = State(7, Hero(1, 1), [sh])
    assert behaviour.intents(s, sh) == [(1.0, "fire")]     # Quick: the phase after loading [rec1 f0693-f0695]
    sb = Enemy("shinobi", 5, -1, 4, ["smoke_bomb"], uid=1, intent="fire")
    s = State(7, Hero(1, 1), [sb])
    out = step(s, "wait")
    assert {(s2.hero.x, s2.hero.hp, s2.enemies[0].x, s2.hero.facing, s2.enemies[0].facing) for _, s2, _ in out} == \
        {(5, 9, 1, 1, -1)}                                  # changed places, both kept their facing [rec1 f0708-f0709]


def test_the_statue_keeps_its_nine_turn_cycle():
    seq, q, last, holds = [], 0, None, 0
    for _ in range(18):
        it = behaviour.cycle_intent(q, last, holds)
        seq.append(it)
        if it == "queue":
            q, last, holds = q + 1, "queue", 0
        elif it == "fire":
            q, last, holds = 0, "fire", 0
        else:
            holds, last = (holds + 1 if q >= 3 else 0), "hold"
    cycle = ["hold", "queue", "hold", "queue", "hold", "queue", "hold", "hold", "fire"]
    assert seq == cycle * 2          # [rec: rec1 T1-T41, rec2 T1-T32, 7 fires and 25 loads]


def test_the_statue_fires_its_whole_stack_wherever_the_hero_is_and_never_moves():
    # rec1 T9: Hook, Spear, Earth Impale with the hero on cell 1: pulled to cell 5, -1 then -2
    st = Enemy("the_statue", 6, -1, 30, ["grappling_hook", "spear", "earth_impale"], uid=1, last="hold", wind=2)
    s = State(7, Hero(1, 1), [st], level="stage4-boss")
    out = step(s, "wait")
    assert {(s2.hero.x, s2.hero.hp) for _, s2, _ in out} == {(5, 7)}
    # cycle turns 7 and 8 hold, whatever the hero does; it never moves or turns
    st = Enemy("the_statue", 6, -1, 30, ["arrow", "tetsubo", "tetsubo"], uid=1, last="queue", wind=0)
    s = State(7, Hero(5, 1), [st], level="stage4-boss")
    for _, s2, _ in step(s, "wait"):
        assert s2.hero.hp == 10 and s2.enemies[0].x == 6 and s2.enemies[0].facing == -1


def test_heavy_units_are_not_swapped_and_the_summoner_follows_the_statue():
    st = Enemy("the_statue", 6, -1, 30, [], uid=1, last="hold")
    sm = Enemy("summoner", 0, 1, 15, [], uid=2)
    s = State(7, Hero(5, 1), [sm, st], level="stage4-boss")
    assert "right" not in legal_hero_actions(s)            # the swap with The Statue: Heavy
    assert behaviour.intents(s, sm) == [(1.0, "queue")]    # the bell with the Statue's first load
    st.queue, st.last, sm.queue = ["spear"], "hold", ["summon"]
    assert behaviour.intents(s, sm) == [(1.0, "fire")]     # it summons with the second
    st.last = "queue"
    assert behaviour.intents(s, sm) == [(1.0, "hold")]


# ---------------------------------------------------------------- the tracker on recording 1's Statue fight
ROUTE1 = ("bamboo_grove", "whispering_caves", "moonlit_port", "spirit_gateway")


def _f(rec: str, n: int):
    """Frame n of a recording, whatever input its name carries."""
    import glob
    import pytest
    hit = glob.glob(f"{DATA}/{rec}f{n:04d}_*.png")
    if not hit:
        pytest.skip(f"recorded frame not available: {rec}f{n:04d}")
    return frame(rec + hit[0].split("/")[-1])


def _statue_tracker():
    from shogunbot.model import Kit
    tr = Tracker(Kit.of("swirl", "arrow"))
    tr.route = list(ROUTE1)
    tr.stage, tr.level_i, tr.advance, tr.level = 4, 2, True, "stage4-3"
    return tr


def test_the_tracker_follows_the_statues_cycle_on_recording_1():
    # T2-T13 of rec1's fight (T1, f0859, shows no bar yet: the bot waits for it)
    from shogunbot.perception.perceive import perceive
    turns = [(860, "queue", "queue"), (861, "hold", "hold"), (862, "queue", "fire"), (863, "hold", "hold"),
             (864, "queue", "hold"), (866, "hold", "hold"), (867, "hold", "hold"), (868, "fire", "hold"),
             (872, "hold", "hold"), (873, "queue", "queue"), (874, "hold", "hold"), (875, "queue", "fire")]
    tr, got = _statue_tracker(), []
    for n, *_ in turns:
        p = perceive(_f(REC1, n), ["swirl", "arrow"], regions=ROUTE1, day=1)
        s = tr.state(p)
        st = next(e for e in s.enemies if e.kind == "the_statue")
        sm = next((e for e in s.enemies if e.kind == "summoner"), None)
        got.append((n, st.intent, sm.intent if sm else "hold"))
        tr.acted(s, "wait")
    assert got == [(n, a, b) for n, a, b in turns]
    assert s.level == "stage4-boss"


def test_the_statues_stack_is_kept_through_its_fire_turn_shake():
    # rec1 T33-T36: Spear, Earth Impale, Sword; at T36 (f0913) the stack shook 4 px and read Sword, Impale, Sword
    from shogunbot.perception.perceive import perceive
    tr = _statue_tracker()
    for n in (906, 907, 908, 910, 912, 913):
        p = perceive(_f(REC1, n), ["swirl", "arrow"], regions=ROUTE1, day=1)
        s = tr.state(p)
        st = next(e for e in s.enemies if e.kind == "the_statue")
        tr.acted(s, "wait")
    assert st.queue == ["spear", "earth_impale", "sword"] and st.intent == "fire"


def test_a_tile_read_while_it_glows_is_replaced_by_its_settled_read():
    # rec1 T4-T9: the Statue loads Hook, Spear, Earth Impale; at T5 (f0863) the new Spear glows and the stack reads
    # Arrow 0.08, Sword 0.24; settled frames read each slot at 1.0
    from shogunbot.perception.perceive import perceive
    tr, stacks = _statue_tracker(), {}
    for n in (862, 863, 864, 866, 867, 868):
        s = tr.state(perceive(_f(REC1, n), ["swirl", "arrow"], regions=ROUTE1, day=1))
        st = next(e for e in s.enemies if e.kind == "the_statue")
        stacks[n] = (list(st.queue), st.intent)
        tr.acted(s, "wait")
    assert stacks[863][0] == ["grappling_hook", "sword"]           # the glowing slot, until a settled read
    assert stacks[864][0][:2] == ["grappling_hook", "spear"]
    assert stacks[868] == (["grappling_hook", "spear", "earth_impale"], "fire")


# ---------------------------------------------------------------- reading Spirit Gateway's units
def _p(rec, n, route=ROUTE1):
    from shogunbot.perception.perceive import perceive
    return perceive(_f(rec, n), ["swirl", "arrow"], regions=route, day=1)


def test_spirit_gateways_new_icons_read_on_frames_outside_the_samples():
    def units(rec, n, route=ROUTE1):
        return [(e["x"], e["kind"], e["queue"]) for e in _p(rec, n, route).enemies]
    route2 = ("bamboo_grove", "hibiku_wastelands", "moonlit_port", "spirit_gateway")
    assert (0, "grappler", ["grappling_hook", "sword"]) in units(REC1, 616)
    assert (5, "shinobi", ["shuriken"]) in units(REC1, 694)
    assert (5, "shinobi", ["smoke_bomb"]) in units(REC1, 701)
    assert (5, "grappler", ["grappling_hook", "sword"]) in units(REC2, 820, route2)
    assert (6, "the_statue", ["spear", "earth_impale", "sword"]) in units(REC1, 911)
    assert (6, "the_statue", ["arrow", "sword", "spear"]) in units(REC2, 1096, route2)
    assert (0, "summoner", []) in units(REC1, 877)                 # 15 pips in three rows (one Swirl took none yet)
    p = _p(REC1, 616)                                              # no '+' above the Grappler's two tiles
    assert next(e for e in p.enemies if e["x"] == 0)["intent"] is None
    # routes without Spirit Gateway keep their reads: Iwao's arena, the same icons closed
    from shogunbot.perception.perceive import GATED_ICONS, open_icons, roster_tables
    base, _ = roster_tables(("bamboo_grove", "whispering_caves"))
    assert not ({"grappling_hook", "shuriken", "smoke_bomb", "shadow_kama"} & open_icons(5, base, False))


def test_a_loaded_shinobi_with_the_hero_in_its_lane_fires_whatever_one_frame_shows():
    # rec1 f0694: the Shuriken loaded in the phase before, its bubble red; after the input the hero lost 1 (f0695)
    tr = _statue_tracker()
    tr.advance, tr.level_i, tr.level = False, 1, "stage4-2"
    s = tr.state(_p(REC1, 694))
    sh = next(e for e in s.enemies if e.kind == "shinobi")
    assert sh.queue == ["shuriken"] and sh.intent == "fire"


def test_a_unit_read_as_the_statue_where_another_stood_keeps_its_kind_and_hp():
    # rec1 f0868-f0869: the Hook pulled the hero past the Summoner on cell 0, whose pips did not show, so it read as
    # The Statue; f0879: the Tachi summoned on cell 3 the turn before read as The Statue too. Before 0.36 the extra
    # read became the boss's summoned unit (a Charger, then the Summoner's Ashigaru)
    tr = _statue_tracker()
    seen = {}
    for n in (866, 867, 868, 869, 870, 878, 879):
        s = tr.state(_p(REC1, n))
        seen[n] = sorted((e.x, e.kind, e.hp) for e in s.enemies)
        tr.acted(s, "wait")
    assert seen[868][0] == (0, "summoner", 7) and seen[869][0] == (0, "summoner", 7)
    assert seen[879][0] == (3, "tachi", 2)


# ---------------------------------------------------------------- the 0.36 replay gate: what the two recordings showed
ROUTE2_FULL = ("bamboo_grove", "hibiku_wastelands", "moonlit_port", "spirit_gateway", "hot_springs", "hideyoshis_keep",
               "shoguns_castle")


def test_a_five_pip_unit_with_a_sword_at_the_bottom_is_a_guardian_not_maybe_a_grappler():
    # the Grappler loads its Hook under its Sword, so a Sword at the bottom names a Guardian: rec1 f0656 and f0794
    # were planned as maybe Grapplers, whose Hook would have hit the hero at any distance; he lost nothing
    from shogunbot.perception.perceive import bottom_ok, roster_tables, stack_kind
    base, _ = roster_tables(ROUTE1)
    assert stack_kind("sword", 5, base, False) == "guardian" and stack_kind("grappling_hook", 5, base, False) == "grappler"
    assert stack_kind("spear", 4, base, False) == "yari_master" and stack_kind("shuriken", 4, base, False) == "shinobi"
    assert stack_kind("sword", 3, base, False) == "ashigaru" and not bottom_ok("grappler", "sword")
    for n in (656, 794):
        g = next(e for e in _p(REC1, n).enemies if e["kind"] == "guardian")
        assert g["queue"] == ["sword"] and g["alt"] is None
    # rec2 f1840: a 5-pip unit whose bubble reads a Spear at 1.0 is no Grappler either
    assert next(e for e in _p(REC2, 1840, ROUTE2_FULL).enemies if e["x"] == 1)["alt"] is None


def test_the_summoners_unit_is_not_placed_as_its_cell_is_random():
    # rec1 f0863 (cell 4, hero on 1), f0876 (cell 3, next to the hero on 2), rec2 f1077 (cell 5, hero on 3): never
    # the cell farthest from the hero that the bosses' summons are placed on, so the unit is read when it comes
    sm = Enemy("summoner", 0, 1, 15, ["summon"], uid=1, intent="fire")
    st = Enemy("the_statue", 6, -1, 30, ["spear"], uid=2, intent="hold")
    for _, s2, _ in step(State(7, Hero(1, 1), [sm, st], level="stage4-boss"), "wait"):
        assert sorted(e.kind for e in s2.alive()) == ["summoner", "the_statue"]
    dk = Enemy("daisuke", 4, -1, 15, ["summon"], uid=1, intent="fire")
    assert all(any(e.kind == "charger" for e in s2.alive()) for _, s2, _ in step(State(5, Hero(0, 1), [dk]), "wait"))


def test_the_replay_applies_an_item_the_recording_used():
    # rec1 f0726: Mass Ice clicked (fx 0.058); the Shinobi on cell 0 stayed frozen for three turns (f0727-f0729),
    # which 0.35's replay did not know
    from shogunbot.replay import item_click
    p = _p(REC1, 726)
    assert item_click({"action": "CLICK", "click": [0.0578, 0.8648]}, p) == "ice_card"
    assert item_click({"action": "CLICK", "click": [0.4292, 0.8815]}, p) is None             # a hand tile
    assert item_click({"action": "CLICK", "item": "white_potion", "click": [0.0625, 0.8815]}, p) == "white_potion"
    tr = Tracker()
    tr.item_used("ice_card")
    tr.item_used("blue_potion")
    assert tr.freeze_left == 3 and tr.shield


def _synthetic(hero_x: int, hp: int, enemies: list, cells: int = 5):
    from shogunbot.perception.perceive import Perception
    return Perception(screen="combat", cells=[{"x": i, "cx": 300 + 140 * i, "kind": "plate"} for i in range(cells)],
                      hero={"x": hero_x, "cx": 300 + 140 * hero_x, "facing": 1, "hp": hp, "hp_max": 10, "queue": [],
                            "swap_red": 0},
                      enemies=[dict({"cx": 300 + 140 * e["x"], "queue": [], "queue_conf": [], "intent": None,
                                     "facing": -1, "alt": None, "elite": False, "spawning": False}, **e)
                               for e in enemies])


def test_hp_lost_ends_a_tracked_shield():
    # rec2: the Kami Brew drunk at The Twins (f0758) stayed tracked into Spirit Gateway I, where at f0875 a
    # Charger's 1 was planned as blocked; a shield negates the next attack, so HP lost means it is gone
    tr = Tracker()
    ash = [{"x": 4, "kind": "ashigaru", "hp": 3}]
    tr.state(_synthetic(1, 10, ash))
    tr.shield = True
    assert tr.state(_synthetic(1, 10, ash)).hero.shield                   # no HP lost: kept
    s = tr.state(_synthetic(1, 9, ash))
    assert not tr.shield and not s.hero.shield


def test_a_hero_moved_by_an_enemy_ends_a_tracked_shield():
    # run 20261029-095311 t76: Rei's Dragon Punch pushed the hero from cell 4 to the wall; the shield took the hit (no
    # HP lost) and stayed on in the tracker, so t82's -2 was planned as -1
    tr = Tracker()
    ash = [{"x": 6, "kind": "ashigaru", "hp": 3}]
    tr.shield = True
    s = tr.state(_synthetic(4, 10, ash, cells=7))
    tr.acted(s, "wait")
    s = tr.state(_synthetic(4, 10, ash, cells=7))
    assert s.hero.shield                                                    # where his action left him: kept
    tr.acted(s, "wait")
    s = tr.state(_synthetic(0, 10, ash, cells=7))
    assert not tr.shield and not s.hero.shield


def test_a_quick_unit_fires_on_its_rule_only_with_its_own_tile_read_sure():
    # rec2 f1051: a Shinobi's bubble read 'swap_toss' at 0.06 (no tile of its own, set to its Shuriken); 0.36 had
    # it firing on the hero two cells away, and nothing came
    tr = _statue_tracker()
    tr.route = list(ROUTE2_FULL[:4])
    tr.advance, tr.level_i, tr.level = False, 2, "stage4-3"
    s = tr.state(_p(REC2, 1051, ROUTE2_FULL[:4]))
    sh = next(e for e in s.enemies if e.x == 5)
    assert sh.kind == "shinobi" and sh.intent != "fire"
