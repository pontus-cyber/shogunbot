"""Consumables read from the item column. Frame-based; skipped when frames are missing."""

from test_bot_flow import frame


def slot3(rel):
    from shogunbot.perception.perceive import read_consumables
    items = {round(i["fy"] * 1080): i["item"] for i in read_consumables(frame(rel))}
    return items.get(952)


def test_lucky_die_reads_as_the_die_in_combat_and_under_banners():
    # [user 2026-10-28] the grey card with six red pips is the Lucky Die, which rerolls the
    # enemies' intents. 0.18.1 read it as Mass Curse in combat (and clicked it at turn 69
    # for a doubled hit) and as Rain of Mirrors under the OBLITERATED / CLEARED banners.
    assert slot3("runs/20261028-122652/frames/f0026_CLICK.png") == "dice_card"
    assert slot3("runs/20261028-122652/frames/item_mirror_card.png") == "dice_card"


def test_mass_curse_still_reads_as_the_curse_card():
    assert slot3("collect/20261027-164727/f0083_SPACE.png") == "curse_card"


def test_the_die_is_in_the_catalogue():
    from shogunbot.catalogue import CAT
    die = CAT.consumable_by_screen("dice_card")
    assert die is not None and die.name == "Lucky Die" and die.effect["type"] == "reroll"
