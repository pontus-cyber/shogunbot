"""0.30: Moonlit Port's boards are dark. Its empty plates read 60-70 on the plate rows (summed RGB)
against 66 for the bright-plate reader, so 0.21-0.29 read most of its fight frames as 'unknown' and
the bot would have stalled there; its bright plates alone made boards with holes or of the wrong
width [collect/20261028-190026 f0335-f0582, collect/20261029-160032 f0462-f0778]. The dark-board
reader (board.dark_cells) takes the cell grid from the bright plates and the dark gaps between plates.
On both recordings' Moonlit Port fights it reads 463 of 537 frames at the level's width (7, 5, 9 and
the boss arena 7) and none at a wrong width; the rest stay 'unknown' and the bot looks again.
Frame tests skip when the frames are missing."""

from test_bot_flow import frame

from shogunbot.perception import board
from shogunbot.perception.perceive import normalize, screen_type

REC1 = "collect/20261028-190026/"
REC2 = "collect/20261029-160032/"
MARK = {"hero": "H", "enemy": "E", "empty": "."}


def _read(rel):
    rgb = normalize(frame(rel))
    cs = board.cells(rgb)
    return screen_type(rgb, cs), "".join(MARK[c["kind"]] for c in cs)


def _cell(x0, kind="empty"):
    return {"x0": x0, "x1": x0 + board.PLATE_W, "cx": x0 + board.PLATE_W // 2, "kind": kind, "rgb": [0, 0, 0]}


def test_a_dark_board_of_7_reads_with_its_empty_plates():
    # 0.29 saw only the 4 bright plates (the enemies' and the hero's): the empty ones are dark
    assert _read(REC1 + "f0350_CLICK.png") == ("combat", "EE..H.E")
    assert _read(REC2 + "f0470_S.png") == ("combat", "..EH.E.")


def test_the_hero_plate_reads_under_red_swap_ticks():
    # right after a swap the hero's tick box is red; the teal stone ends still mark his plate
    assert _read(REC1 + "f0343_W.png") == ("combat", "E.E..H.")


def test_dark_boards_of_5_and_9():
    assert _read(REC2 + "f0600_W.png") == ("combat", "EHE.E")
    assert _read(REC1 + "f0440_idle.png") == ("combat", "EE..H.E.E")
    assert _read(REC2 + "f0700_SPACE.png") == ("combat", "....HE...")


def test_the_twins_arena_reads_as_a_boss_fight_of_7():
    st, row = _read(REC2 + "f0750_A.png")
    assert st == "boss" and len(row) == 7 and row.count("H") == 1


def test_bright_plates_next_to_a_dark_plate_are_not_the_whole_board():
    # f0513: 7 bright plates of a 9-cell board; f1407 (Hideyoshi's Keep): 5 of 7. 0.29 read them at
    # those widths; now they stay unread (open_end)
    assert _read(REC1 + "f0513_idle.png")[0] == "unknown"
    assert _read(REC1 + "f1407_idle.png")[0] == "unknown"


def test_a_tooltip_over_the_plate_row_leaves_the_frame_unread():
    assert _read(REC2 + "f0532_CLICK.png")[0] == "unknown"


def test_other_regions_still_read_as_before():
    # the start of Caves II (0.29: 4 of its 7 cells) and a Spirit Gateway board of 9
    assert _read(REC1 + "f0222_idle.png") == ("combat", "...H...")
    assert _read(REC1 + "f0813_D.png") == ("combat", "H...E...E")
    assert _read(REC1 + "f0204_CLICK.png")[0] == "combat"
    # bright boards with one plate's edge moved (board.ROW_TOL): spawn flames in Bamboo Grove II (12 px),
    # a tooltip's bar in Hot Springs (64 px), the item column in the Shogun's Castle (48 px)
    assert _read(REC2 + "f0047_idle.png") == ("combat", "EE.H.E.")
    assert _read(REC2 + "f1370_idle.png") == ("combat", "E....EH..")
    assert _read(REC2 + "f1893_SPACE.png") == ("combat", "EEE..E..H")


def test_a_row_with_a_hole_is_not_a_board():
    row = [_cell(400 + i * board.PITCH) for i in range(5)]
    row[2]["kind"] = "hero"
    assert board.is_board(row)
    holed = row[:3] + [_cell(row[3]["x0"] + board.PITCH), _cell(row[4]["x0"] + board.PITCH)]
    assert not board.is_board(holed)
    assert not board.is_board(row[:4])                      # 4 cells: no level is 4 wide
    assert not board.is_board([dict(c, kind="empty") for c in row])   # no hero
