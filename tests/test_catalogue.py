"""M1: the catalogue (shogunbot/data/*.json) is the single source of game facts.

The first block pins the values the simulator used through 0.13, so moving
them into data changed no number (all 753 planner states and every replay
were also checked identical when 0.14 shipped)."""

from shogunbot.catalogue import CAT
from shogunbot.model import (
    DEFAULT_KIT, ENEMY_KINDS, ENEMY_TILE_COOLDOWN, ENEMY_TILE_DAMAGE, HERO_MAX_HP, QUEUE_MAX,
    SWAP_COOLDOWN, Hero, Kit, State, Enemy,
)


def test_values_match_the_013_constants():
    assert HERO_MAX_HP == 10 and SWAP_COOLDOWN == 4 and QUEUE_MAX == 3
    assert [(t.label, t.damage, t.cooldown) for t in DEFAULT_KIT] == [("swirl", 2, 3), ("arrow", 2, 5)]
    # Stage 1 as in 0.13; the Yari Master of Whispering Caves is simulated since 0.26 (M4)
    stage1 = {"ashigaru", "tachi", "charger", "daisuke"}
    assert {k: v for k, v in ENEMY_KINDS.items() if k in stage1} == {
        "ashigaru": (3, "sword"), "tachi": (2, "swirl"), "charger": (1, "charge"), "daisuke": (15, "sword")}
    # 0.26: the Yari Master; 0.27: Rei the Ruthless (13 HP; Dragon Punch, Back Charge, the red bell);
    # 0.28: Iwao the Impaler (25 HP; Earth Impale, Swirl, Spear, Summon); 0.31: Moonlit Port's Guardian
    # (5 HP, Sword) and Ashigaru Archer (2 HP, Arrow), and The Twins (35 HP between them); 0.35: the Yumi Sniper
    # (3 HP, Volley), a visitor in Moonlit Port
    assert set(ENEMY_KINDS) - stage1 == {"yari_master", "rei", "iwao", "guardian", "ashigaru_archer", "the_twins",
                                         "yumi_sniper"}
    assert ENEMY_KINDS["yari_master"] == (4, "spear") and ENEMY_KINDS["rei"] == (13, "dragon_punch")
    assert ENEMY_KINDS["iwao"] == (25, "earth_impale")
    assert ENEMY_KINDS["guardian"] == (5, "sword") and ENEMY_KINDS["ashigaru_archer"] == (2, "arrow")
    assert ENEMY_KINDS["the_twins"] == (35, "spear")
    assert ENEMY_TILE_DAMAGE == {"sword": 2, "swirl": 2, "charge": 1, "summon": 0, "spear": 2,
                                 "dragon_punch": 1, "back_charge": 1, "summon_double": 0, "earth_impale": 2,
                                 "arrow": 2, "tetsubo": 4, "shadow_dash": 1, "dash": 0, "swap_toss": 0,
                                 "volley": 2}
    assert ENEMY_TILE_COOLDOWN == {"sword": 2, "swirl": 3, "charge": 4, "summon": 6, "spear": 5,
                                   "dragon_punch": 4, "back_charge": 3, "summon_double": 6, "earth_impale": 4,
                                   "arrow": 5, "tetsubo": 7, "shadow_dash": 5, "dash": 6, "swap_toss": 7,
                                   "volley": 0}   # no cooldown on the wiki: the Sniper loaded again one or two turns
                                                  # after firing [rec: collect/20261029-160032 f0201; run 20261030-115221 t176]


def test_ready_values_reproduce_the_013_weights():
    # 0.13 had swirl_ready 8 and arrow_ready 6 in planner.WEIGHTS
    assert CAT.tile("swirl").ready_value == 8 and CAT.tile("arrow").ready_value == 6


def test_base_pool_is_the_tile_choice_pool_seen_live():
    # [bot] tile choices after Bamboo Grove I in runs 20261027-180145 .. 20261028-061420 and
    # collect 20261027-164727 offered only these eight
    assert {t.id for t in CAT.base_pool()} == {
        "katana", "swirl", "spear", "shuriken", "arrow", "charge", "shadow_dash", "back_charge"}
    assert all(t.sim for t in CAT.base_pool())


def test_footprint_directionality():
    assert not CAT.tile("swirl").directional
    assert CAT.tile("sword").directional and CAT.tile("charge").directional
    assert not CAT.tile("nagiboku").directional and not CAT.tile("earth_impale").directional


def test_bamboo_grove_enemies_by_pips_and_bubble():
    assert CAT.kinds_by_hp("bamboo_grove") == {3: "ashigaru", 2: "tachi", 1: "charger"}
    assert CAT.kinds_by_tile("bamboo_grove") == {"sword": "ashigaru", "swirl": "tachi", "charge": "charger"}


def test_consumable_effects():
    assert CAT.consumable_by_screen("green_potion").effect == {"type": "heal", "amount": 3}
    assert CAT.consumable_by_screen("ice_card").effect == {"type": "freeze_all", "turns": 3}
    assert CAT.consumable_by_screen("skull_card").effect["turns"] == 3
    assert CAT.consumable_by_screen("white_potion").id == "cool_up"


def test_upgrades_seen_on_shrines():
    seen = {u.screen_text for u in CAT.upgrades.values() if u.seen}
    assert {"-1 COOLDOWN", "-2 COOLDOWN", "+1 DAMAGE +1 COOLDOWN", "+2 DAMAGE +3 COOLDOWN",
            "+1 UPGRADE SLOT -1 COOLDOWN"} <= seen


def test_levels_from_regions():
    lv = CAT.levels
    assert (lv["bamboo-1"]["cells"], lv["bamboo-1"]["waves"], lv["bamboo-1"]["reward"]) == (5, 4, "tile_choice")
    assert (lv["bamboo-2"]["cells"], lv["bamboo-2"]["waves"], lv["bamboo-2"]["reward"]) == (7, 5, "tile_upgrade")
    assert lv["boss"]["boss"] == "daisuke" and lv["boss"]["reward"] == "tile_upgrade"


def test_kit_labels_and_round_trip():
    kit = DEFAULT_KIT.add("swirl").add("katana", damage=3)
    assert kit.labels == ("swirl", "arrow", "swirl#2", "katana") and kit["katana"].damage == 3
    s = State(5, Hero(2, 1, kit=kit), [Enemy("ashigaru", 3, -1, 3)])
    s2 = State.from_dict(s.to_dict())
    assert s2.hero.kit == kit and s2.hero.charge["swirl#2"] == 3


def test_old_logs_load_with_the_default_kit():
    d = State(5, Hero(2, 1), []).to_dict()
    d["hero"].pop("kit")
    assert State.from_dict(d).hero.kit == DEFAULT_KIT
    assert Kit.of("swirl", "arrow") == DEFAULT_KIT
