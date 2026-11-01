"""The stall rule [user 2026-10-28] and the combo the planner should find first.

Position: self-play seed 9008, Bamboo Grove II, turn 40 (bot 0.17 at the fixed
self-play depth 3 danced here until the 200-turn limit). The hero stands at 0 with
Arrow loaded; a Tachi at 2 blocks a loaded Charger at 6. Killing the Tachi opens the
Charger's lane, so damage-first scoring did not do it at depth 3. (With the behaviour
model rebuilt from 8 runs for 0.18, depth 3 happens to step right first and then finds
the combo; the stall itself depends on the model, so it is not asserted here.)
"""

from shogunbot.model import State
from shogunbot.planner import STALL_TURNS, WEIGHTS, evaluate, plan, progress_key, stall_weights
from shogunbot.rules import step

STALLED = {
    "cells": 7, "level": "bamboo-2", "wave": 5, "waves_total": 5, "turn": 40,
    "hero": {"x": 0, "facing": 1, "hp": 10, "queue": ["arrow"], "charge": {"swirl": 3, "arrow": 5},
             "swap_charge": 4, "shield": False,
             "kit": [{"label": "swirl", "tile": "swirl", "damage": 2, "cooldown": 3, "enchant": None,
                      "levels": 3, "used": 0},
                     {"label": "arrow", "tile": "arrow", "damage": 2, "cooldown": 5, "enchant": None,
                      "levels": 3, "used": 0}]},
    "enemies": [
        {"kind": "tachi", "x": 2, "facing": -1, "hp": 2, "queue": [], "uid": 41, "frozen": 0, "cd": 3,
         "intent": "hold", "primed": False, "wind": 2, "cursed": False, "poison": 0, "last": "fire", "fage": 0},
        {"kind": "charger", "x": 6, "facing": -1, "hp": 1, "queue": ["charge"], "uid": 43, "frozen": 0, "cd": 0,
         "intent": "hold", "primed": False, "wind": 2, "cursed": False, "poison": 0, "last": "hold", "fage": None},
    ],
}


def follow(depth: int, weights, turns: int = 8):
    """Play the planner's choice for a few turns, taking the likeliest outcome each time."""
    s = State.from_dict(STALLED)
    actions = []
    for _ in range(turns):
        if not s.alive():
            break
        d = plan(s, depth, weights)
        actions.append(d.action)
        _, s, _ = max(step(s, d.action, validate=False), key=lambda o: o[0])
    return actions, s


def test_four_plies_find_the_zero_damage_double_kill():
    # [user 2026-10-28] kill the blocker and keep the Arrow for the Charger: Swirl queued
    # behind the Arrow, swap past the Tachi, attack. The Arrow then hits the Charger first
    # in line and the Swirl hits the Tachi behind the hero. Live search is 5 plies here.
    actions, s = follow(4, WEIGHTS)
    assert "queue:swirl" in actions and actions[-1] == "attack" and len(actions) <= 4
    assert not s.alive() and s.hero.hp == 10


def test_stall_weights_trade_one_hp_when_no_free_kill_is_in_sight():
    actions, s = follow(3, stall_weights(WEIGHTS, 10))
    assert actions[0] == "attack" and not s.alive() and s.hero.hp == 9


def test_stall_weights_still_prefer_the_free_double_kill():
    actions, s = follow(4, stall_weights(WEIGHTS, 10))
    assert not s.alive() and s.hero.hp == 10


def test_stall_weights_never_allow_more_than_one_hp():
    s = State.from_dict(STALLED)
    w = stall_weights(WEIGHTS, 10)
    s.hero.hp = 9                                   # the one HP the trade may cost
    assert "below_floor" not in evaluate(s, w)[1]
    s.hero.hp = 8                                   # a second HP lost: penalised
    assert evaluate(s, w)[1]["below_floor"] == w["below_floor"]


def test_progress_key_changes_on_damage_kills_and_spawns_only():
    s = State.from_dict(STALLED)
    k = progress_key(s)
    moved = s.clone()
    moved.hero.x, moved.enemies[0].x = 1, 3
    assert progress_key(moved) == k
    hurt = s.clone()
    hurt.enemies[0].hp = 1
    assert progress_key(hurt) != k
    assert STALL_TURNS == 15


# [user 2026-10-31: HP stays the priority, but the stall lever goes to 3 instead of 1]
from shogunbot.planner import STALL_MIN_HP, STALL_STUCK_TURNS, stall_budget, stall_push, plan_live
import json, os

def test_stall_budget_grows_with_the_quiet_turns():
    assert [stall_budget(q) for q in (0, 14, 15, 39, 40, 59, 60, 500)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert STALL_STUCK_TURNS == 200


def test_stall_floor_never_below_three_hp():
    assert stall_weights(WEIGHTS, 10, 3)["hp_floor"] == 7
    assert stall_weights(WEIGHTS, 5, 3)["hp_floor"] == STALL_MIN_HP == 3
    assert stall_weights(WEIGHTS, 3, 3)["hp_floor"] == 2      # at 3 HP only the 1-HP trade, as before
    assert stall_weights(WEIGHTS, 10)["hp_floor"] == 9          # default budget unchanged


# run 20261031-102644 t627 (Iwao): the hero waited 450 turns at the wall with three tiles loaded
IWAO_T627 = {"cells": 7, "hero": {"x": 0, "facing": 1, "hp": 9, "queue": ["shuriken", "swirl", "charge"], "charge": {"swirl": 3, "arrow": 5, "charge": 4, "shuriken": 3}, "swap_charge": 4, "shield": False, "kit": [{"label": "swirl", "tile": "swirl", "damage": 3, "cooldown": 3, "enchant": None, "levels": 3, "used": 2}, {"label": "arrow", "tile": "arrow", "damage": 2, "cooldown": 5, "enchant": None, "levels": 3, "used": 0}, {"label": "charge", "tile": "charge", "damage": 2, "cooldown": 4, "enchant": None, "levels": 3, "used": 1}, {"label": "shuriken", "tile": "shuriken", "damage": 1, "cooldown": 3, "enchant": None, "levels": 3, "used": 0}]}, "enemies": [{"kind": "yari_master", "x": 3, "facing": -1, "hp": 1, "queue": [], "uid": 1, "frozen": 0, "cd": 0, "intent": "move-right", "primed": False, "wind": 1, "cursed": False, "poison": 0, "last": "fire", "fage": 0, "alt": None, "elite": False, "aim": None}, {"kind": "iwao", "x": 5, "facing": -1, "hp": 8, "queue": ["earth_impale", "swirl"], "uid": 2, "frozen": 0, "cd": 0, "intent": "hold", "primed": False, "wind": 1, "cursed": False, "poison": 0, "last": "hold", "fage": 459, "alt": None, "elite": False, "aim": None}], "level": "stage2-boss", "wave": 0, "waves_total": 0, "turn": 627}


def test_stall_push_ends_the_iwao_wait():
    s = State.from_dict(IWAO_T627)
    d1 = plan_live(s, stall_weights(WEIGHTS, 9, 1))
    assert d1.action != "attack" and stall_push(s, d1, 1) is None          # 1-HP budget: still waits
    s = State.from_dict(IWAO_T627)
    d2 = plan_live(s, stall_weights(WEIGHTS, 9, 2))
    assert stall_push(s, d2, 2) == "attack"                               # 40+ quiet turns: attack, which hits


# run 20261031-180506 t419 (0.41, Moonlit Port II): no tile queued, so no attack line; the hero walked with a 3-HP
# Guardian for 200 turns [0.44]
MOONLIT2_T419 = {"cells": 5, "hero": {"x": 2, "facing": -1, "hp": 8, "queue": [], "charge": {"swirl": 1, "arrow": 5, "swirl#2": 4, "arrow#2": 2}, "swap_charge": 3, "shield": False, "kit": [{"label": "swirl", "tile": "swirl", "damage": 2, "cooldown": 1, "enchant": None, "levels": 3, "used": 1}, {"label": "arrow", "tile": "arrow", "damage": 2, "cooldown": 5, "enchant": None, "levels": 3, "used": 0}, {"label": "swirl#2", "tile": "swirl", "damage": 3, "cooldown": 5, "enchant": "shockwave", "levels": 3, "used": 2}, {"label": "arrow#2", "tile": "arrow", "damage": 3, "cooldown": 2, "enchant": None, "levels": 3, "used": 3}]}, "enemies": [{"kind": "guardian", "x": 4, "facing": -1, "hp": 3, "queue": ["sword"], "uid": 1, "frozen": 0, "cd": 0, "intent": "move-left", "primed": False, "wind": 1, "cursed": False, "poison": 0, "last": "hold", "fage": 3, "alt": None, "elite": False, "aim": None}], "level": "stage3-2", "wave": 3, "waves_total": 5, "turn": 419}


def test_stall_push_loads_a_tile_when_nothing_is_queued():
    s = State.from_dict(MOONLIT2_T419)
    d = plan(s, 5, stall_weights(WEIGHTS, 8, 1))
    assert d.action == "left" and stall_push(s, d, 1) is None             # below a 2-HP budget nothing changes
    d = plan(s, 5, stall_weights(WEIGHTS, 8, 2))
    assert d.action == "left" and stall_push(s, d, 2) == "queue:arrow#2"   # its line hits the Guardian


# ---------------------------------------------------------------- 0.46 [core-requests item 5]
# run 20261101-053431 (0.45) t1066, Hot Springs III: a Shielder on 0 and two Archers on 1-2, all shielded, a shielded
# 1-HP Ashigaru on 6 beside the hero; killing it opens the Archers' line, so the hero stepped left and right
HS3_T1066 = {"cells": 9, "hero": {"x": 7, "facing": -1, "hp": 6, "queue": ["katana", "arrow#2", "spear"], "charge": {"swirl": 6, "arrow": 4, "back_charge": 5, "spear": 7, "arrow#2": 7, "katana": 0}, "swap_charge": 4, "shield": False, "kit": [{"label": "swirl", "tile": "swirl", "damage": 4, "cooldown": 6, "enchant": "perfect_strike", "levels": 5, "used": 3}, {"label": "arrow", "tile": "arrow", "damage": 3, "cooldown": 5, "enchant": None, "levels": 3, "used": 1}, {"label": "back_charge", "tile": "back_charge", "damage": 2, "cooldown": 5, "enchant": "curse", "levels": 2, "used": 2}, {"label": "spear", "tile": "spear", "damage": 4, "cooldown": 7, "enchant": None, "levels": 3, "used": 2}, {"label": "arrow#2", "tile": "arrow", "damage": 2, "cooldown": 7, "enchant": "poison", "levels": 3, "used": 1}, {"label": "katana", "tile": "katana", "damage": 2, "cooldown": 0, "enchant": None, "levels": 2, "used": 0}]}, "enemies": [{"kind": "shielder", "x": 0, "facing": 1, "hp": 4, "queue": [], "uid": 1, "intent": "hold", "wind": 1, "last": "queue", "fage": 265, "shield": True}, {"kind": "ashigaru_archer", "x": 1, "facing": 1, "hp": 2, "queue": ["arrow"], "uid": 2, "intent": "hold", "wind": 1, "shield": True}, {"kind": "ashigaru_archer", "x": 2, "facing": 1, "hp": 2, "queue": ["arrow"], "uid": 3, "intent": "hold", "wind": 1, "shield": True}, {"kind": "ashigaru", "x": 6, "facing": 1, "hp": 1, "queue": ["sword"], "uid": 4, "intent": "fire", "wind": 1, "last": "move", "fage": 5, "shield": True}], "level": "stage5-3", "wave": 4, "waves_total": 6, "turn": 1066}


def test_progress_key_ignores_pip_tie_kind_flips_and_shields():
    s = State.from_dict(HS3_T1066)
    k = progress_key(s)
    flip = s.clone()
    flip.enemies[2].kind = "tachi"              # the Archer's 2 pips tie with the Tachi's: read as one on every other turn
    assert progress_key(flip) == k
    strip = s.clone()
    strip.enemies[3].shield = False             # a shield taken (the Shielder gives it back): no progress yet
    assert progress_key(strip) == k
    hit = s.clone()
    hit.enemies[1].hp = 1
    assert progress_key(hit) != k
    gone = s.clone()
    gone.enemies = gone.enemies[:3]
    assert progress_key(gone) != k


def test_the_stall_budget_would_have_grown_in_hot_springs_iii():
    # the logged states of the level: with the HP key the counter runs from t795 (the last HP change) to t1579 (the
    # Ashigaru's death), so the 3-HP budget comes at t855; with 0.45's kind key it never ran past 46 turns
    import pytest
    from test_bot_flow import DATA
    path = DATA / "runs/20261101-053431/turns.jsonl"
    if not path.exists():
        pytest.skip("run log not available")
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["level"] == "stage5-3"]
    last, prev, budget3 = None, None, None
    for r in rows:
        k = progress_key(State.from_dict(r["state"]))
        if k != prev:
            prev, last = k, r["turn"]
        if budget3 is None and stall_budget(r["turn"] - last) == 3:
            budget3 = r["turn"]
    assert budget3 == 855


# ---------------------------------------------------------------- 0.47 [core-requests item 9]
# run 20261101-164934 (0.46), Hot Springs II t698-t897, stopped as stuck after 200 quiet turns: Grappler (Hook and
# Sword loaded), Archer (Arrow loaded) and Shielder (1 HP) in a row, all shielded, each holding while an ally blocked
# its lane (the simulator predicted all 199 turns); the hero on cell 4 at the right wall. Turns 758 (three tiles
# loaded), 886 (none) and 887 (a Swirl loaded).
HS2_KIT = [{"label": "swirl", "tile": "swirl", "damage": 3, "cooldown": 0, "enchant": None, "levels": 4, "used": 3}, {"label": "arrow", "tile": "arrow", "damage": 3, "cooldown": 3, "enchant": None, "levels": 3, "used": 3}, {"label": "back_charge", "tile": "back_charge", "damage": 1, "cooldown": 4, "enchant": "perfect_strike", "levels": 2, "used": 1}, {"label": "arrow#2", "tile": "arrow", "damage": 2, "cooldown": 1, "enchant": None, "levels": 2, "used": 2}, {"label": "back_charge#2", "tile": "back_charge", "damage": 1, "cooldown": 3, "enchant": None, "levels": 2, "used": 0}, {"label": "swirl#2", "tile": "swirl", "damage": 2, "cooldown": 3, "enchant": None, "levels": 4, "used": 0}, {"skill": "healthy", "level": 1}, {"skill": "quick_recovery", "level": 1}]
HS2_ROW = [{"kind": "grappler", "x": 0, "facing": 1, "hp": 2, "queue": ["grappling_hook", "sword"], "uid": 1, "intent": "hold", "wind": 1, "last": "hold", "fage": None, "shield": True}, {"kind": "ashigaru_archer", "x": 1, "facing": 1, "hp": 2, "queue": ["arrow"], "uid": 2, "intent": "hold", "wind": 1, "last": "hold", "fage": 115, "shield": True}, {"kind": "shielder", "x": 2, "facing": 1, "hp": 1, "queue": [], "uid": 3, "intent": "hold", "wind": 1, "last": "hold", "fage": 57, "alt": "shinobi", "shield": True}]


def hs2(queue: list[str], turn: int) -> State:
    return State.from_dict({"cells": 5, "hero": {"x": 4, "facing": -1, "hp": 12, "queue": queue, "charge": {"swirl": 0, "arrow": 3, "back_charge": 4, "arrow#2": 1, "back_charge#2": 3, "swirl#2": 3}, "swap_charge": 3, "shield": False, "kit": HS2_KIT}, "enemies": [dict(e) for e in HS2_ROW], "level": "stage5-2", "wave": 15, "waves_total": 15, "turn": turn})


def test_stall_push_never_fires_into_the_air():
    # t887: the Swirl hits only the cells beside the hero and cell 3 is empty; 0.46 pushed it every second turn
    # for 70 turns (attack 1735 within the 3-HP window of queue:swirl#2 4457). The planned line (load Swirl #2, two
    # steps left, attack: both kills at once) is already the shortest, so nothing is pushed.
    s = hs2(["swirl"], 887)
    d = plan(s, 4, stall_weights(WEIGHTS, 12, 3))
    assert d.action != "attack" and stall_push(s, d, 3) is None


def test_stall_push_takes_the_planned_kills_sooner():
    # t758 (three tiles loaded): the search's line waits a turn, then steps left twice and kills the Archer and the
    # Shielder on its last ply, where the Grappler's reply falls past it; left, left, attack does it a turn sooner
    s = hs2(["swirl", "arrow", "back_charge#2"], 758)
    d = plan(s, 4, stall_weights(WEIGHTS, 12, 3))
    assert d.action == "wait"
    why = []
    assert stall_push(s, d, 3, notes=why) == "left" and "2 kill(s)" in why[0]
    d = plan(s, 4, stall_weights(WEIGHTS, 12, 2))                   # the Grappler's reply costs 3 HP: not with 2
    assert stall_push(s, d, 2) is None
    # t886 (nothing loaded): load the Arrow, whose line kills the Shielder a turn sooner
    s = hs2([], 886)
    d = plan(s, 4, stall_weights(WEIGHTS, 12, 3))
    assert stall_push(s, d, 3) in ("queue:arrow", "queue:arrow#2")


def test_the_hot_springs_ii_standoff_ends():
    # planner and push on the likeliest outcomes from the three logged turns: an enemy loses HP within 5 turns, and the
    # hero keeps his HP (0.46 stood 200 turns here)
    for queue, turn in ((["swirl", "arrow", "back_charge#2"], 758), ([], 886), (["swirl"], 887)):
        s = hs2(queue, turn)
        k = progress_key(s)
        for _ in range(5):
            d = plan(s, 4, stall_weights(WEIGHTS, s.hero.hp, 3))
            a = stall_push(s, d, 3) or d.action
            s = max(step(s, a, validate=False), key=lambda o: o[0])[1]
            if progress_key(s) != k:
                break
        assert progress_key(s) != k and s.hero.hp == 12, (turn, s.render())


def test_a_pip_read_that_flips_back_does_not_reset_the_stall_counter():
    # [0.47] run 20261101-201012 (0.46), Spirit Gateway III: the loaded Grappler on cell 0 read with 5 pips while the hero
    # stood on cell 7 and as a 4-pip Yari Master when he stepped to cell 8 (the camera pans the 9-cell board); the HP
    # key flipped every fourth turn (220 times in 466 turns) and the stall rule never started
    from shogunbot.planner import progress_mark
    keys = [(1, 3, 4, 5), (1, 3, 4, 5), (1, 3, 4, 4), (1, 3, 4, 4), (1, 3, 4, 5), (1, 3, 4, 5), (1, 3, 4, 4)] * 10
    mark, last, resets = None, 0, []
    for turn, k in enumerate(keys):
        m = progress_mark(k, mark)
        if m is not None:
            mark, last = m, turn
            resets.append(turn)
    assert resets == [0, 2]                                    # the first read, and the first drop to 4 pips
    assert len(keys) - 1 - last >= STALL_TURNS                 # then the counter runs
    assert progress_mark((1, 3, 2, 4), mark) == (4, 10)        # a real hit is progress
    assert progress_mark((3, 4, 4), mark) == (3, 11)           # a death too
    assert progress_mark((1, 1, 3, 4, 4), mark) == (5, 13)     # and a new unit


def test_the_hot_springs_iii_budget_with_the_low_mark():
    # the bot's counter since 0.47 (progress_mark over progress_key) on run 20261101-053431's Hot Springs III: the 3-HP
    # budget at t855, as with the key alone (test_the_stall_budget_would_have_grown_in_hot_springs_iii)
    import pytest
    from shogunbot.planner import progress_mark
    from test_bot_flow import DATA
    path = DATA / "runs/20261101-053431/turns.jsonl"
    if not path.exists():
        pytest.skip("run log not available")
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["level"] == "stage5-3"]
    mark, last, budget3 = None, None, None
    for r in rows:
        m = progress_mark(progress_key(State.from_dict(r["state"])), mark)
        if m is not None:
            mark, last = m, r["turn"]
        if budget3 is None and stall_budget(r["turn"] - last) == 3:
            budget3 = r["turn"]
    assert budget3 == 855
