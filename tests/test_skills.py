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
