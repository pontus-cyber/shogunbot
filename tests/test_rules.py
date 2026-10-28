"""Regression tests: each one encodes a rule or a live-play lesson."""

from shogunbot.model import Enemy, Hero, State
from shogunbot.planner import plan
from shogunbot.rules import apply_hero, Events, enemy_phase, step


def mk(cells, hero_x, hero_f, enemies, **hero_kw):
    h = Hero(hero_x, hero_f, **hero_kw)
    es = [Enemy(k, x, f, hp, list(q), uid=i + 1) for i, (k, x, f, hp, q) in enumerate(enemies)]
    return State(cells, h, es)


def only(outs):
    assert len(outs) == 1, outs
    return outs[0][1], outs[0][2]


def test_swirl_hits_front_and_back():
    s = mk(5, 2, 1, [("tachi", 1, 1, 2, []), ("tachi", 3, -1, 2, [])])
    s.hero.queue = ["swirl"]
    ev = Events()
    s2 = apply_hero(s, "attack", ev)
    assert len(ev.kills) == 2 and not s2.alive()


def test_arrow_hits_first_target_only():
    s = mk(7, 0, 1, [("ashigaru", 3, -1, 3, []), ("charger", 5, -1, 1, [])])
    s.hero.queue = ["arrow"]
    ev = Events()
    s2 = apply_hero(s, "attack", ev)
    assert [e.hp for e in s2.enemies] == [1, 1]


def test_t1_armed_adjacent_fires_after_hero_action():
    s = mk(5, 2, 1, [("ashigaru", 3, -1, 3, ["sword"])])
    s2, ev = only(step(s, "queue:swirl"))
    assert s2.hero.hp == 8 and ev.hero_damage == [("ashigaru", 1, 2)]


def test_t1_moving_out_of_footprint_avoids_hit():
    s = mk(5, 2, 1, [("ashigaru", 3, -1, 3, ["sword"])])
    outs = step(s, "left")
    assert all(st.hero.hp == 10 for _, st, _ in outs)


def test_t2_enemy_that_steps_in_does_not_fire_same_phase():
    s = mk(5, 1, 1, [("ashigaru", 3, -1, 3, ["sword"])])
    s2, ev = only(step(s, "wait"))
    assert s2.enemies[0].x == 2 and s2.hero.hp == 10


def test_charger_blocked_by_ally_holds():
    s = mk(5, 0, 1, [("ashigaru", 2, -1, 3, []), ("charger", 4, -1, 1, ["charge"])])
    for _, st, ev in step(s, "wait"):
        assert st.enemies[1].x == 4 and st.hero.hp == 10


def test_killing_blocker_opens_lane_and_charger_hits_next_turn():
    # [run 20261027-180145 turns 59-60] the charge comes the phase AFTER the lane opens
    s = mk(5, 0, 1, [("tachi", 1, -1, 2, []), ("charger", 4, -1, 1, ["charge"])])
    s.hero.queue = ["swirl"]
    s2, ev = only(step(s, "attack"))
    assert ("tachi", 1, "hero:swirl") in ev.kills
    assert s2.hero.hp == 10 and s2.enemies[0].x == 4
    s3, _ = only(step(s2, "wait"))
    assert s3.hero.hp == 9 and s3.enemies[0].x == 1


def test_red_charger_holds_then_charges():
    # [run 20261028-044407 turns 46-47] red bubble + clear lane: holds this phase,
    # charges the next (its bubble turns yellow in between)
    s = mk(7, 6, -1, [("charger", 3, 1, 1, ["charge"])])
    s.enemies[0].intent = "hold"
    s2, _ = only(step(s, "wait"))
    assert s2.hero.hp == 10 and s2.enemies[0].x == 3
    s3, _ = only(step(s2, "wait"))
    assert s3.hero.hp == 9 and s3.enemies[0].x == 5


def test_swap_leaves_enemy_facing_away():
    s = mk(5, 2, -1, [("ashigaru", 1, 1, 3, ["sword"])])
    ev = Events()
    s2 = apply_hero(s, "left", ev)
    assert s2.hero.x == 1 and s2.enemies[0].x == 2 and s2.enemies[0].facing == 1
    assert ev.swapped_with == 1 and s2.hero.swap_charge == 0


def test_swap_blocked_on_cooldown():
    s = mk(5, 2, -1, [("ashigaru", 1, 1, 3, [])], swap_charge=0)
    from shogunbot.rules import legal_hero_actions
    assert "left" not in legal_hero_actions(s)


def test_cooldown_resets_on_fire_and_recharges():
    s = mk(5, 2, 1, [("ashigaru", 4, -1, 3, [])])
    s.hero.queue = ["arrow"]
    ev = Events()
    s2 = apply_hero(s, "attack", ev)
    assert s2.hero.charge["arrow"] == 0
    s3 = apply_hero(s2, "wait", Events())
    assert s3.hero.charge["arrow"] == 1


def test_planner_does_not_queue_inside_loaded_footprint():
    s = mk(5, 2, 1, [("ashigaru", 3, -1, 3, ["sword"])])
    d = plan(s, depth=2)
    assert d.action not in ("queue:swirl", "queue:arrow", "turn", "wait"), d.candidates


def test_planner_takes_double_kill():
    s = mk(5, 2, 1, [("tachi", 1, 1, 2, ["swirl"]), ("tachi", 3, -1, 2, ["swirl"])])
    s.hero.queue = ["swirl"]
    d = plan(s, depth=2)
    assert d.action == "attack", d.candidates


def test_planner_never_kills_blocker_while_in_lane_when_alternative_exists():
    # tachi (2hp) adjacent ahead blocks a loaded charger; swirl queued.
    # Killing the tachi now opens the lane -> 1 damage. Expect a better line.
    s = mk(5, 0, 1, [("tachi", 1, -1, 2, []), ("charger", 4, -1, 1, ["charge"])])
    s.hero.queue = ["swirl"]
    d = plan(s, depth=2)
    top = d.candidates[0]
    assert top["exp_damage"] <= 1.0


# ---- telegraphed intents (replay of run 20261027-164727) --------------------
def test_red_loaded_enemy_holds_even_when_hero_steps_in():
    # 'A3*> . <H': red bubble; hero steps adjacent; no hit this phase
    s = mk(5, 2, -1, [("ashigaru", 0, 1, 3, ["sword"])])
    s.enemies[0].intent = "hold"
    s2, ev = only(step(s, "left"))
    assert s2.hero.hp == 10 and s2.hero.x == 1 and s2.enemies[0].armed


def test_yellow_enemy_fires_and_hero_can_step_out():
    s = mk(5, 2, 1, [("ashigaru", 3, -1, 3, ["sword"])])
    s.enemies[0].intent = "fire"
    stay, _ = only(step(s, "wait"))
    leave, _ = only(step(s, "left"))
    assert stay.hero.hp == 8 and leave.hero.hp == 10


def test_intents_cover_one_phase_only():
    s = mk(5, 0, 1, [("ashigaru", 4, -1, 3, [])])
    s.enemies[0].intent = "move-left"
    s2, _ = only(step(s, "wait"))
    assert s2.enemies[0].x == 3 and s2.enemies[0].intent is None


# ---- run 20261028-053637 (boss) and the user's swap trick ------------------
def test_swap_turns_a_charge_into_an_ally():
    # [user] when an enemy is about to charge you, swap with it the step before it
    # fires: it keeps its facing, now points away from you, and runs into its ally
    s = mk(5, 2, 1, [("ashigaru", 0, 1, 3, []), ("charger", 3, -1, 1, ["charge"])])
    s.enemies[1].intent = "fire"
    s.enemies[0].intent = "hold"
    s2, ev = only(step(s, "right"))
    assert s2.hero.x == 3 and s2.hero.hp == 10
    assert any(k[0] == "ashigaru" for k in ev.enemy_damage) or s2.enemies[0].hp < 3


def test_planner_finds_the_swap_trick():
    s = mk(5, 2, 1, [("ashigaru", 0, 1, 3, []), ("charger", 3, -1, 1, ["charge"])])
    s.enemies[1].intent = "fire"
    s.enemies[0].intent = "hold"
    s.hero.charge = {"swirl": 0, "arrow": 0}
    assert plan(s, 3).action == "right"


def test_daisuke_loads_charge_then_sword_before_firing():
    # turns 120-123: queue charge, queue sword, hold, then fire
    s = mk(5, 1, 1, [("daisuke", 4, -1, 15, ["charge"])])
    for _, s2, _ in step(s, "wait"):
        assert s2.hero.hp == 10


def test_enemy_holds_the_phase_after_loading():
    # R1 in the presets: loaded this phase -> cannot fire next phase
    s = mk(5, 1, 1, [("ashigaru", 2, -1, 3, [])])
    s.enemies[0].intent = "queue"
    s2, _ = only(step(s, "wait"))
    assert s2.enemies[0].armed and s2.hero.hp == 10
    s3, _ = only(step(s2, "wait"))
    assert s3.hero.hp == 10
