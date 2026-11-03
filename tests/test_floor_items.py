"""0.50: consumables on the floor (core-requests item 15c) [user 2026-11-03 09:36: "It goes into item column if there is
space. So depending on situation you want to trade lesser potions for higher tier ones when needed. So important to
keep a look at the floor and include it into strategy as it can be crucial for survival. Any pots on the floor when item
column is full gets discarded and lost"]. Run 20261102-191231 (0.47.1) died at 3 HP in Hot Springs III with three
Cool Ups in the column, two Edamame Brews and a Kami Brew on the floor (t0653_after). Frame tests skip when the frames
are missing."""
import numpy as np
import pytest
from PIL import Image

from test_bot_flow import DATA, _stub_windows_modules

from shogunbot.catalogue import CAT
from shogunbot.model import Enemy, Hero, Kit, State
from shogunbot.planner import plan_live
from shogunbot.rules import item_rank, step

_stub_windows_modules()


def _frame(run: str, name: str) -> np.ndarray:
    path = DATA / f"runs/{run}/frames/{name}.png"
    if not path.exists():
        pytest.skip("frame not available")
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------- perception
def test_the_floor_items_of_run_191231_t653_are_read():
    from shogunbot.perception.perceive import perceive
    p = perceive(_frame("20261102-191231", "t0653_after"), regions=("bamboo_grove", "whispering_caves", "moonlit_port",
                                                                     "spirit_gateway", "hot_springs"))
    assert [(f["x"], f["item"]) for f in p.floor] == [(0, "green_potion"), (2, "green_potion"), (8, "blue_potion")]
    assert [it["item"] for it in p.items] == ["white_potion"] * 3


@pytest.mark.parametrize("run,frame,cell,item", [
    ("20261102-065902", "t0741_before", 1, "green_potion"),     # Edamame Brew under Moonlit Port's night grading
    ("20261101-164934", "t0185_before", 2, "blue_potion"),      # Kami Brew on a dark blue backdrop
    ("20261101-122050", "t0294_before", 3, "ice_card"),         # Mass Ice (t0269: its glint, read at 0.77, no item)
    ("20261031-165514", "t0206_before", 1, "skull_card"),       # Mass Poison
    ("20261031-165514", "t0099_before", 4, "mirror_card"),      # Rain of Mirrors
    ("20261101-021522", "t0769_before", 4, "green_potion"),     # (the frame before its flash, below)
])
def test_floor_items_on_other_backdrops(run, frame, cell, item):
    from shogunbot.perception import board, floor
    from shogunbot.perception.perceive import normalize
    rgb = normalize(_frame(run, frame))
    got = {f["x"]: f["item"] for f in floor.read(rgb, board.cells(rgb))}
    assert got.get(cell) == item


@pytest.mark.parametrize("run,frame", [
    ("20261101-021522", "t0770_before"),    # the Edamame Brew on cell 4 in its pale flash: no item (and not Kami Brew)
    ("20261101-164934", "t0417_before"),    # the Lucky Die on cell 8: no template, no item
    ("20261101-164934", "t0460_before"),    # Mass Curse on cell 0: no template, no item
])
def test_a_flash_or_an_unknown_item_reads_no_item(run, frame):
    from shogunbot.perception import board, floor
    from shogunbot.perception.perceive import normalize
    rgb = normalize(_frame(run, frame))
    assert floor.read(rgb, board.cells(rgb)) == []


# ---------------------------------------------------------------- simulator
def _state(hp: int, floor: list, room: int, hero_x: int = 3) -> State:
    kit = Kit.of("swirl", "arrow")
    hero = Hero(hero_x, -1, hp=hp, kit=kit)
    far = Enemy("ashigaru", 8, -1, 3, [], uid=1)
    return State(9, hero, [far], level="stage5-3", wave=1, waves_total=99, floor=floor, room=room)


def test_a_brew_stepped_on_with_room_is_drunk_when_all_of_its_heal_fits():
    s = _state(3, [[2, "edamame_brew"]], room=1)
    outs = step(s, "left")
    assert all(s2.hero.x == 2 and not s2.floor for _, s2, _ in outs)
    # the turn's end shows the HP the next frame shows; the brew is drunk as the next turn begins (no turn passes)
    assert all(s2.hero.hp == 3 and s2.heal_due == 3 and s2.room == 1 for _, s2, _ in outs)
    for _, s3, _ in step(outs[0][1], "wait"):
        assert s3.hero.hp == 6 and s3.heal_due == 0              # 3 + 3 [catalogue: Edamame Brew heals 3]


def test_the_last_brew_before_the_final_region_is_kept():
    # bot.edamame_reason's reserve [core-requests item 11]: with no other brew held, a brew taken before the final
    # region is kept for a hit that would kill, not drunk at once (State.brew_keep)
    s = _state(3, [[2, "edamame_brew"]], room=1)
    s.brew_keep = True
    for _, s2, _ in step(s, "left"):
        assert s2.heal_due == 0 and s2.room == 0 and s2.picked == item_rank("edamame_brew", 3, 10) == 8


def test_with_the_column_full_the_item_stays_on_the_floor():
    s = _state(3, [[2, "edamame_brew"]], room=0)
    for _, s2, _ in step(s, "left"):
        assert s2.floor == [[2, "edamame_brew"]] and s2.hero.hp == 3 and s2.picked == 0


def test_standing_still_takes_nothing():
    s = _state(3, [[3, "kami_brew"]], room=1)          # (the tracker leaves the hero's own cell out; a wait takes nothing)
    for _, s2, _ in step(s, "wait"):
        assert s2.floor == [[3, "kami_brew"]] and s2.room == 1


def test_an_item_kept_counts_its_rank():
    s = _state(9, [[2, "kami_brew"]], room=1)
    for _, s2, _ in step(s, "left"):
        assert s2.picked == item_rank("kami_brew", 9, 10) == CAT.consumable("kami_brew").sell and s2.room == 0


def test_ranks_follow_the_sell_value_and_hp_items_rise_at_low_hp():
    assert item_rank("edamame_brew", 10, 10) == 5 and item_rank("cool_up", 10, 10) == 2     # [wiki] sell values
    assert item_rank("kami_brew", 10, 10) == 2 < item_rank("kami_brew", 7, 10) == 5          # low_hp_bonus 3
    assert item_rank("mass_ice", 3, 10) == 2


def test_the_planner_walks_to_a_brew_at_low_hp():
    s = _state(3, [[2, "edamame_brew"]], room=1)
    assert plan_live(s).action == "left"


def test_the_planner_does_not_trade_hp_for_a_brew():
    # the step onto the brew lies in a fire lane: a loaded Ashigaru beside the brew faces it [damage-first]
    s = _state(5, [[2, "edamame_brew"]], room=1)
    s.enemies = [Enemy("ashigaru", 1, 1, 3, ["sword"], uid=1, intent="fire")]
    assert plan_live(s).action != "left"


# ---------------------------------------------------------------- the trade
def _items(*ids):
    return [{"item": i, "fx": 0.06, "fy": y, "dim": False} for i, y in zip(ids, (0.70, 0.79, 0.88))]


def test_run_191231_t653_frees_a_slot_for_an_edamame_brew():
    from shogunbot.bot import floor_trade
    s = _state(3, [[0, "edamame_brew"], [2, "edamame_brew"], [8, "kami_brew"]], room=0, hero_x=6)
    seen = {0: ("green_potion", 2), 2: ("green_potion", 2), 8: ("blue_potion", 2)}
    use = floor_trade(s, _items("white_potion", "white_potion", "white_potion"), seen, 10)
    assert use is not None and use[0] == "white_potion" and "Edamame Brew" in use[1]
    # read on one state only: not yet (a misread would waste an item)
    once = {c: (sid, 1) for c, (sid, _) in seen.items()}
    assert floor_trade(s, _items("white_potion", "white_potion", "white_potion"), once, 10) is None
    s.room = 1                                                    # a slot is free: nothing to trade
    assert floor_trade(s, _items("white_potion", "white_potion"), seen, 10) is None


def test_no_trade_between_equal_ranks():
    from shogunbot.bot import floor_trade
    s = _state(9, [[2, "kami_brew"]], room=0)
    seen = {2: ("blue_potion", 3)}
    assert floor_trade(s, _items("white_potion", "ice_card", "skull_card"), seen, 10) is None   # 2 against 2
    s.hero.hp = 6                                                 # low HP: the shield's rank rises to 5
    assert floor_trade(s, _items("white_potion", "ice_card", "skull_card"), seen, 10)[0] == "white_potion"


def test_a_held_brew_is_drunk_for_another_on_the_floor_only_below_the_cap():
    from shogunbot.bot import floor_trade
    seen = {2: ("green_potion", 2)}
    s = _state(9, [[2, "edamame_brew"]], room=0)
    assert floor_trade(s, _items("green_potion", "green_potion", "green_potion"), seen, 10)[0] == "green_potion"
    s.hero.hp = 10
    assert floor_trade(s, _items("green_potion", "green_potion", "green_potion"), seen, 10) is None


def test_kami_brew_goes_first_unless_a_shield_is_up():
    from shogunbot.bot import floor_trade
    s = _state(9, [[2, "edamame_brew"]], room=0)
    seen = {2: ("green_potion", 2)}
    assert floor_trade(s, _items("white_potion", "blue_potion", "dice_card"), seen, 10)[0] == "blue_potion"
    s.hero.shield = True
    assert floor_trade(s, _items("white_potion", "blue_potion", "dice_card"), seen, 10)[0] == "white_potion"
    # the Lucky Die only against a fire telegraph, Rain of Mirrors only while no enemy is armed (their use can turn
    # against the hero)
    s.enemies[0].queue = ["sword"]
    assert floor_trade(s, _items("blue_potion", "dice_card", "mirror_card"), seen, 10) is None
    s.enemies[0].intent = "fire"
    assert floor_trade(s, _items("blue_potion", "dice_card", "mirror_card"), seen, 10)[0] == "dice_card"


# ---------------------------------------------------------------- tracker
def test_the_tracker_keeps_an_item_a_unit_stands_on():
    from types import SimpleNamespace
    from shogunbot.bridge import Tracker
    tr = Tracker()

    def p(kinds, floor):
        return SimpleNamespace(cells=[{"x": i, "kind": k} for i, k in enumerate(kinds)], floor=floor, items=[])
    assert tr._floor(p(["empty", "hero", "empty", "enemy"], [{"x": 2, "item": "green_potion"}]), 4, 1) == \
        [[2, "edamame_brew"]]
    # an enemy steps onto the brew: kept, and counted on
    assert tr._floor(p(["empty", "hero", "enemy", "empty"], []), 4, 1) == [[2, "edamame_brew"]]
    assert tr.floor[2] == ("green_potion", 1, 0)
    # the cell is empty again and nothing is read there: kept through one such read (the sprite's pale flash, run
    # 20261101-021522 t770), gone at the second
    assert tr._floor(p(["empty", "hero", "empty", "enemy"], []), 4, 1) == [[2, "edamame_brew"]]
    assert tr._floor(p(["empty", "hero", "empty", "enemy"], []), 4, 1) == []
    # the hero's own cell is left out
    assert tr._floor(p(["empty", "empty", "hero", "enemy"], [{"x": 0, "item": "blue_potion"}]), 4, 2) == \
        [[0, "kami_brew"]]
    assert tr.room(SimpleNamespace(items=[{"item": "white_potion"}] * 2)) == 1


def test_the_tracker_sees_a_pickup():
    from types import SimpleNamespace
    from shogunbot.bridge import Tracker
    tr = Tracker()

    def p(kinds, floor, held):
        return SimpleNamespace(cells=[{"x": i, "kind": k} for i, k in enumerate(kinds)], floor=floor,
                               items=[{"item": "white_potion"}] * held)
    tr._floor(p(["empty", "hero", "empty", "enemy"], [{"x": 2, "item": "green_potion"}], 2), 4, 1)
    assert tr._floor(p(["empty", "empty", "hero", "enemy"], [], 3), 4, 2) == []
    assert tr.floor == {} and "picked up green_potion on 2" in tr.notes
