"""Skills bought in shops (data/skills.json) [wiki: Shop & Skills; recording 2, collect/20261029-160032,
the user's purchases at f0181 (Combo Recharge), f0447-f0460 (Quick Recovery twice, Dynamic Boost,
Two-Way Move) and f0787 (Healthy)]. The kit carries them as (id, level) pairs."""

from shogunbot.catalogue import CAT
from shogunbot.model import DEFAULT_KIT, HERO_MAX_HP, SWAP_COOLDOWN, Hero, Kit, State
from shogunbot.rules import Events, apply_hero, legal_hero_actions


def test_the_catalogue_has_the_wikis_36_skills_with_recorded_prices():
    assert len(CAT.skills) == 36
    # prices on the counter in recording 2 [f0176, f0440, f0459, f0783, f1134]
    seen = {"combo_recharge": 20, "quick_recovery": 15, "dynamic_boost": 15, "two_way_move": 15, "healthy": 15,
            "combo_coin": 15}
    assert {sid: CAT.skill(sid).coins for sid in seen} == seen
    assert CAT.skill_by_name("- QUICK RECOVERY -".strip(" -")).id == "quick_recovery"
    assert CAT.skill_by_name("TWO-WAY MOVE").id == "two_way_move"


def test_levels_scale_the_effect():
    qr = CAT.skill("quick_recovery")
    # 'LEVEL 2: -2 COOLDOWN FOR SPECIAL MOVE (SWAP).' [rec2 f0457]
    assert (qr.amount(1), qr.amount(2)) == (-1, -2)
    assert CAT.skill("dynamic_boost").amount(2) == 2          # 'LEVEL 2: +2 DAMAGE ...' [rec2 f0790]
    assert CAT.skill("combo_recharge").amount(2) == 8         # [wiki] max level 2
    assert CAT.skill("healthy").amount(1) == 2


def test_kit_skills_set_max_hp_and_the_swap_cooldown():
    k = DEFAULT_KIT.with_skill("healthy")
    assert k.max_hp == HERO_MAX_HP + 2                        # the map showed HP 10/12 after Healthy [rec2 f0796]
    assert k.swap_cooldown == SWAP_COOLDOWN
    q2 = DEFAULT_KIT.with_skill("quick_recovery").with_skill("quick_recovery")
    assert q2.level("quick_recovery") == 2 and q2.swap_cooldown == SWAP_COOLDOWN - 2
    assert DEFAULT_KIT.with_skill("combo_recharge").with_skill("combo_recharge").with_skill(
        "combo_recharge").level("combo_recharge") == 2         # max level 2 [wiki]
    assert k != DEFAULT_KIT and hash(k) != hash(DEFAULT_KIT)
    assert Kit.from_list(q2.to_list()) == q2
    assert DEFAULT_KIT.to_list() == [d for d in q2.to_list() if "skill" not in d]


def test_quick_recovery_readies_the_swap_sooner():
    kit = DEFAULT_KIT.with_skill("quick_recovery")
    s = State(5, Hero(1, 1, kit=kit))
    from shogunbot.model import Enemy
    s.enemies = [Enemy("ashigaru", 2, -1, 3)]
    assert "right" in legal_hero_actions(s)                   # a swap with the Ashigaru
    s = apply_hero(s, "right", Events())
    assert s.hero.swap_charge == 0
    for _ in range(kit.swap_cooldown):
        assert not s.hero.swap_ready
        s = apply_hero(s, "wait", Events())
    assert s.hero.swap_ready and kit.swap_cooldown == 3


def test_quick_recovery_level_2_leaves_two_turns_of_swap_cooldown():
    # after Quick Recovery level 2 (f0457) every swap showed 2 red ticks under the hero, not 4
    # [rec2 f0541-f0542, f0574-f0575, f0696-f0697]
    kit = DEFAULT_KIT.with_skill("quick_recovery").with_skill("quick_recovery")
    assert kit.swap_cooldown == 2


def test_two_way_move_swaps_with_a_unit_behind_the_hero():
    # rec2 f0491 -> f0492 (Two-Way Move bought at f0460): the hero on cell 3 facing left pressed D
    # with a Tachi on cell 4 behind him; next frame he stood on cell 4, still facing left, the swap
    # recharging (2 red ticks, Quick Recovery level 2)
    from shogunbot.model import Enemy
    def board(kit):
        return State(7, Hero(3, -1, kit=kit), [Enemy("tachi", 4, -1, 2, uid=1)])
    assert "right" not in legal_hero_actions(board(DEFAULT_KIT))          # without it the move is blocked
    kit = DEFAULT_KIT.with_skill("two_way_move").with_skill("quick_recovery").with_skill("quick_recovery")
    s = board(kit)
    assert "right" in legal_hero_actions(s)
    s2 = apply_hero(s, "right", Events())
    assert (s2.hero.x, s2.hero.facing, s2.enemies[0].x, s2.hero.swap_charge) == (4, -1, 3, 0)


# ---- skills from the wiki's text [wiki: Shop & Skills], not yet seen in play ------------------------

def _fire(kit, hero_x, facing, enemies, tiles):
    """Queue the tiles and fire them; returns the state after the hero's action and its events."""
    from shogunbot.model import Enemy
    s = State(9, Hero(hero_x, facing, kit=kit), [Enemy(k, x, f, hp, uid=i + 1) for i, (k, x, f, hp) in enumerate(enemies)])
    for t in tiles:
        s = apply_hero(s, "queue:" + t, Events())
    ev = Events()
    return apply_hero(s, "attack", ev), ev


def _dealt(ev):
    return sum(d for _, _, d, _ in ev.enemy_damage)


def test_back_stabber_sniper_and_close_combat_add_damage():
    base = Kit.of("swirl", "arrow", "charge")
    # Back Stabber: +1 on an enemy that faces away from the hero
    _, ev = _fire(base.with_skill("back_stabber"), 2, 1, [("ashigaru", 4, 1, 3)], ["arrow"])
    assert _dealt(ev) == 3
    _, ev = _fire(base.with_skill("back_stabber"), 2, 1, [("ashigaru", 4, -1, 3)], ["arrow"])
    assert _dealt(ev) == 2
    # Sniper: +1 from 4 or more cells away; two Snipers +2
    _, ev = _fire(base.with_skill("sniper"), 0, 1, [("ashigaru", 4, -1, 5)], ["arrow"])
    assert _dealt(ev) == 3
    _, ev = _fire(base.with_skill("sniper").with_skill("sniper"), 0, 1, [("ashigaru", 4, -1, 5)], ["arrow"])
    assert _dealt(ev) == 4
    _, ev = _fire(base.with_skill("sniper"), 1, 1, [("ashigaru", 4, -1, 5)], ["arrow"])
    assert _dealt(ev) == 2
    # Close Combat: +1 on an adjacent enemy, not when the tile moved the hero there (Charge)
    _, ev = _fire(base.with_skill("close_combat"), 3, 1, [("ashigaru", 4, -1, 5)], ["swirl"])
    assert _dealt(ev) == 3
    _, ev = _fire(base.with_skill("close_combat"), 1, 1, [("ashigaru", 4, -1, 5)], ["charge"])
    assert _dealt(ev) == 1


def test_damaging_move_hurts_the_unit_swapped_with():
    from shogunbot.model import Enemy
    s = State(5, Hero(1, 1, kit=DEFAULT_KIT.with_skill("damaging_move")), [Enemy("ashigaru", 2, -1, 3, uid=1)])
    ev = Events()
    s2 = apply_hero(s, "right", ev)
    assert s2.hero.x == 2 and s2.enemies[0].x == 1 and s2.enemies[0].hp == 2


def test_iron_skin_and_reactive_shield():
    from shogunbot.rules import _hurt_hero
    from shogunbot.model import Enemy
    e = Enemy("tachi", 3, -1, 2, uid=1)
    h = Hero(2, 1, hp=7, kit=DEFAULT_KIT.with_skill("iron_skin"))
    assert h.kit.max_hp == HERO_MAX_HP - 3
    _hurt_hero(h, 2, e, Events())
    _hurt_hero(h, 1, e, Events())                       # never to 0
    assert h.hp == 5
    h = Hero(2, 1, hp=10, kit=DEFAULT_KIT.with_skill("reactive_shield"))
    _hurt_hero(h, 2, e, Events())
    assert h.hp == 8 and h.shield                        # hit, then a shield
    _hurt_hero(h, 2, e, Events())
    assert h.hp == 8 and not h.shield                    # the shield takes the next hit


def test_unfriendly_fire_and_mindfulness():
    from shogunbot.rules import _enemy_hit
    from shogunbot.model import Enemy
    s = State(5, Hero(0, 1, kit=DEFAULT_KIT.with_skill("unfriendly_fire")),
              [Enemy("tachi", 2, 1, 2, uid=1), Enemy("ashigaru", 3, -1, 3, uid=2)])
    _enemy_hit(s, s.enemies[0], s.enemies[1], 2, "swirl", Events())
    assert s.enemies[1].hp == 0                          # 2 + 1
    kit = DEFAULT_KIT.with_skill("mindfulness")
    s = State(5, Hero(0, 1, kit=kit, charge={"swirl": 0, "arrow": 0}, swap_charge=0))
    s = apply_hero(s, "wait", Events())
    assert s.hero.charge == {"swirl": 2, "arrow": 2} and s.hero.swap_charge == 2


def test_fortress_starts_every_fight_with_a_shield():
    from shogunbot import selfplay
    kit = DEFAULT_KIT.with_skill("fortress")
    assert kit.max_hp == HERO_MAX_HP - 2
    seen = []
    orig = selfplay.play_level
    def spy(seed, level_i, lp, s, *a, **k):
        seen.append(s.hero.shield)
        return s, True
    selfplay.play_level = spy
    try:
        selfplay.run_once(0, 1, kit=kit, start="bamboo-2", scenario="stress")
    except Exception:
        pass
    finally:
        selfplay.play_level = orig
    assert seen and all(seen)


def test_dynamic_boost_after_a_tile_moved_the_hero():
    # rec2 f1158 -> f1159 (Dynamic Boost level 2 since f0790): the hero on cell 3 facing left fired a
    # Charge, which carried him next to the Guardian on cell 0, then a 4-damage Swirl, which showed 6
    kit = Kit.of("charge").add("swirl", damage=4).with_skill("dynamic_boost").with_skill("dynamic_boost")
    s, ev = _fire(kit, 3, -1, [("ashigaru", 0, 1, 20)], ["charge", "swirl"])
    hits = [d for _, _, d, src in ev.enemy_damage]
    assert s.hero.x == 1 and hits[-1] == 6
    assert hits[0] == 3            # the Charge's own strike after its move [hyp]
    # nothing moved: no boost (a Charge with its target next to the hero; rec2 f1275 -> f1276 took 1)
    s, ev = _fire(kit, 1, -1, [("ashigaru", 0, 1, 20)], ["charge"])
    assert [d for _, _, d, _ in ev.enemy_damage] == [1]
