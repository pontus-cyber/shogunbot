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
