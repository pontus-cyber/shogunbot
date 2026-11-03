"""Bubble icon templates for tiles first seen on Rei the Ruthless (0.27) and Iwao (0.28).

    python -m shogunbot.perception.build_icons ROOT    # ROOT holds runs/ and collect/

`perception/assets/icons.npz` holds the enemy bubble icons that `perceive.classify_icon` matches
(an icon interior shrunk to 25 x 24 px). The Stage 1 icons (Sword, Swirl, Arrow, Charge, Spear,
Summon) came with 0.10 and stay as they are; this builder owns the labels in SAMPLES and replaces
them on every run, so it can be run again when samples are added.

Rei's icons [bot: runs 20261029-072323 ... -103242, every bubble over her on the boss frames]:
Dragon Punch is a white fist trailing a flame, damage 1 (0.24 read it as Charge 48 times and as
Sword 9 times); Back Charge is Charge's teal arrow pointing down (read as Charge or Sword); the red
bell is her double-strike Summon, which brings two Twin Tachi (read as Arrow, Charge or Summon);
her orange bell is the ordinary Summon (read right: the orange rule). Samples: every Back Charge
and red bell seen, every fourth Dragon Punch, spread over the five runs.

Iwao the Impaler's icons (0.28) [bot: run 20261029-124921 t159-t198; rec: collect/20261028-190026
f0285-f0318]: Earth Impale is brown spikes rising from the ground, damage 2 (read as Spear at
confidence 0.15-0.8); his Swirl, stacked on an Earth Impale, read as Spear 7 times and as Swirl
once; his Spear is the Yari Master's drawing (read as Spear at 0.1-1.0, and on the Yari Master as
Sword 12 of 14 times). 'swirl' and 'spear' are Stage 1 labels too: the builder owns the templates
it added (array B), not whole labels, so the Stage 1 templates stay.

The Twins' icons (0.31) [rec: collect/20261028-190026 f0544-f0580, collect/20261029-160032 f0727-f0775]:
Tetsubo, a '4' and a spiked club (read as Spear or Sword before); Shadow Dash, a '1' and a ghost with an
arrow (read as Charge); Phantom Leap, the Dash tile, a white bird under Swap Toss's two circling arrows
(both read as the red Summon, or Earth Impale). Their Spear and Swirl read right with the existing ones.

The Yumi Sniper's Volley (0.35) [rec: collect/20261029-160032, Hibiku Wastelands f0198-f0332 and Moonlit Port
f0512-f0597]: a '2' and arrows rising and falling. 0.34 read it as Spear (confidence 0-0.13) or, under the
volley's firing flash, as Summon, so the tracker named the Sniper an Ashigaru [user review of run
20261030-115221, t174]. The bot's own frames of that run are left out of the samples and serve as the check.

Spirit Gateway (0.36) [rec: collect/20261028-190026 f0600-f0921, collect/20261029-160032 f0801-f1120; surveys of
2026-10-31]: the Grappler's Grappling Hook (read as Charge before), the Shinobi's Shuriken (Back Charge) and Smoke Bomb
(Shadow Dash), The Statue's Hook, Shuriken and Shadow Kama (Arrow). A label 'tile@kind' holds that tile as one kind
draws it: The Statue's stack bobs in 4-px steps and its slots sit 4 px apart from the 120-px pitch (its Earth Impale
28 px lower than Iwao's, its Sword read as Spear at the stack's highest bob), and the Grappler's upper Sword sits 4 px
high. perceive opens these templates only on a unit that can carry them (GATED_ICONS), so other routes read as before.

Hot Springs (0.45) [rec: collect/20261028-190026 f0945-f1356, collect/20261029-160032 f1150-f1508; surveys of
2026-10-31]: Kowa's and the Shielder's Shield, a big two-tone blue shield with no digit (read as Dash at 0.02-0.15
before), and the Shielder's Ally Shield, a blue wavy line and a small shield (read as Dash, Swap Toss, Back Charge or
Sword at 0.02-0.28, and on yellow frames under the fire flames as Summon at 1.0); Kowa's Smoke Bomb is the Shinobi's and
reads with its templates once open.

Hideyoshi's Keep (0.47) [rec: collect/20261028-190026 f1526-f1592, collect/20261029-160032 f1689-f1735; survey of
2026-11-01]: Hideyoshi the Cunning's tiles as he draws them ('tile@hideyoshi', open only on him): Origin of Symmetry (a
circle with a dot; read as Swap Toss at 0.1-0.4 before), Mirror (two figures; Earth Impale or Spear at 0.0-0.3), Back
Charge (a '1' and a downward arrowhead; Back Charge or Earth Impale at 0.1-0.2), Dash (a white streak; Dash, Swap Toss
or Earth Impale at 0.1), Bomb (a '3' and a black bomb; Earth Impale), and his Swirl, Earth Impale and Spear, which read
right on red frames and as Earth Impale at 0.0 on yellow ones. His Summon is Rei's red bell and reads with its
template (rec1 f1561, 1.0). The Shadow Charger's Charge and Sword read with the Stage 1 templates.

The Shogun's Castle (0.48) [rec: collect/20261028-190026 f1739-f1830, collect/20261029-160032 f1916-f2007; survey of
2026-11-02]: the Shogun's tiles as he draws them ('tile@the_shogun', open on a unit without pips in his arena): Charge (a '1'
and an upward arrowhead; read as Spear at 0.2 before), the Corrupted Barrage (a '3' and a pink bar; Earth Impale at 0.1),
the Shuriken (a '1' and a four-point star; Back Charge or Spear at 0.0-0.2), Mirror on his yellow frames (Summon at 1.0),
and his Sword, Spear, Swirl, Origin and Back Charge. His Summoners' Ally Shield is the Shielder's and reads with its
template once open (1.0).

The Shogun's Dragon Punch (0.51) [bot: the Shogun fights of runs 20261103-020829 ... -113809; core-requests item 19]: a '1'
and a white fist with a tan inside, red or yellow, in his first bar. His bubbles sit higher than the standard slot (the
'1' is cut at the slot's top, as in his other templates), so Rei's Dragon Punch templates miss it: 0.48-0.50 read it as
Charge (at 1.0 on his yellow frames), Back Charge or Shuriken (0.0-0.46), and the punch, its push and the collision
were not simulated (run 20261103-052436 t820, t841; run 20261103-085026 t870, -2). Samples from four runs; the frames of
runs 20261103-085026 and -113809 are the check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .perceive import SLOT_PITCH, SLOT_Y0, SLOT_Y1, normalize

ASSETS = Path(__file__).parent / "assets"
LEGACY_LABELS = ("dragon_punch", "back_charge", "summon_double")   # what the 0.27 builder added
SAMPLES = {   # label: [(frame relative to ROOT, the unit's centre x at 1080p, slot from the bottom)]
    "dragon_punch": [
        ("runs/20261029-072323/frames/t0068_before.png", 1284, 0),
        ("runs/20261029-095311/frames/t0076_before.png", 1192, 0),
        ("runs/20261029-100218/frames/t0068_before.png", 1380, 1),
        ("runs/20261029-100218/frames/t0084_before.png", 1284, 1),
        ("runs/20261029-100218/frames/t0090_before.png", 632, 0),
        ("runs/20261029-100218/frames/t0106_before.png", 444, 0),
        ("runs/20261029-100218/frames/t0114_before.png", 732, 1),
        ("runs/20261029-100218/frames/t0124_before.png", 1240, 0),
        ("runs/20261029-100218/frames/t0134_before.png", 1004, 1),
        ("runs/20261029-101549/frames/t0065_before.png", 1332, 1),
        ("runs/20261029-101549/frames/t0071_before.png", 1096, 0),
        ("runs/20261029-101549/frames/t0085_before.png", 640, 0),
        ("runs/20261029-103242/frames/t0069_before.png", 1284, 0),
        ("runs/20261029-103242/frames/t0079_before.png", 1284, 0),
        ("runs/20261029-103242/frames/t0091_before.png", 1151, 0),
    ],
    "back_charge": [
        ("runs/20261029-095311/frames/t0085_before.png", 1152, 0),
        ("runs/20261029-100218/frames/t0071_before.png", 1340, 0),
        ("runs/20261029-100218/frames/t0072_before.png", 1292, 0),
        ("runs/20261029-100218/frames/t0073_before.png", 1240, 0),
        ("runs/20261029-100218/frames/t0074_before.png", 1192, 0),
        ("runs/20261029-100218/frames/t0094_before.png", 1012, 0),
        ("runs/20261029-100218/frames/t0095_before.png", 1012, 0),
        ("runs/20261029-100218/frames/t0096_before.png", 1012, 0),
        ("runs/20261029-100218/frames/t0097_before.png", 1012, 0),
        ("runs/20261029-101549/frames/t0094_before.png", 676, 0),
    ],
    "summon_double": [
        ("runs/20261029-072323/frames/t0064_before.png", 1332, 0),
        ("runs/20261029-072323/frames/t0065_before.png", 1332, 0),
        ("runs/20261029-095311/frames/t0070_before.png", 1384, 0),
        ("runs/20261029-095311/frames/t0071_before.png", 1384, 0),
        ("runs/20261029-100218/frames/t0061_before.png", 1384, 0),
        ("runs/20261029-100218/frames/t0062_before.png", 1384, 0),
        ("runs/20261029-101549/frames/t0059_before.png", 1332, 0),
        ("runs/20261029-101549/frames/t0060_before.png", 1332, 0),
        ("runs/20261029-103242/frames/t0064_before.png", 1332, 0),
        ("runs/20261029-103242/frames/t0065_before.png", 1332, 0),
    ],
    "earth_impale": [   # Iwao, alone (hero not adjacent) or under his Swirl
        ("runs/20261029-124921/frames/t0165_before.png", 1292, 0),
        ("runs/20261029-124921/frames/t0167_before.png", 1151, 0),
        ("runs/20261029-124921/frames/t0184_before.png", 1284, 0),
        ("runs/20261029-124921/frames/t0186_before.png", 1284, 0),
        ("runs/20261029-124921/frames/t0187_before.png", 1284, 0),
        ("collect/20261028-190026/f0289_idle.png", 1151, 0),
        ("collect/20261028-190026/f0291_CLICK.png", 1100, 0),
        ("collect/20261028-190026/f0293_CLICK.png", 1100, 0),
        ("collect/20261028-190026/f0296_idle.png", 1100, 0),
        ("collect/20261028-190026/f0297_CLICK.png", 1100, 0),
        ("collect/20261028-190026/f0313_idle.png", 1096, 0),
        ("collect/20261028-190026/f0316_CLICK.png", 1096, 0),
    ],
    "swirl": [   # Iwao: the Swirl he stacks on an Earth Impale when the hero is adjacent
        ("runs/20261029-124921/frames/t0186_before.png", 1284, 1),
        ("runs/20261029-124921/frames/t0187_before.png", 1284, 1),
        ("collect/20261028-190026/f0292_idle.png", 1100, 1),
        ("collect/20261028-190026/f0294_SPACE.png", 1100, 1),
        ("collect/20261028-190026/f0297_CLICK.png", 1100, 1),
        ("collect/20261028-190026/f0316_CLICK.png", 1096, 1),
    ],
    "spear": [   # Iwao's Spear (first ten), then the Yari Masters'
        ("runs/20261029-124921/frames/t0161_before.png", 1151, 0),
        ("runs/20261029-124921/frames/t0162_before.png", 1100, 0),
        ("runs/20261029-124921/frames/t0170_before.png", 1240, 0),
        ("runs/20261029-124921/frames/t0172_before.png", 1151, 0),
        ("runs/20261029-124921/frames/t0179_before.png", 1380, 0),
        ("runs/20261029-124921/frames/t0181_before.png", 1240, 0),
        ("runs/20261029-124921/frames/t0190_before.png", 1528, 0),
        ("runs/20261029-124921/frames/t0192_before.png", 1284, 0),
        ("collect/20261028-190026/f0308_D.png", 1332, 0),
        ("collect/20261028-190026/f0309_D.png", 1284, 0),
        ("runs/20261029-124921/frames/t0180_before.png", 820, 0),
        ("runs/20261029-124921/frames/t0181_before.png", 820, 0),
        ("collect/20261028-190026/f0308_D.png", 492, 0),
        ("collect/20261028-190026/f0309_D.png", 444, 0),
        ("collect/20261028-190026/f0310_W.png", 536, 0),
        ("collect/20261028-190026/f0313_idle.png", 536, 0),
        ("collect/20261028-190026/f0315_CLICK.png", 536, 0),
    ],
    "tetsubo": [   # The Twins' Tetsubo, a '4' and a spiked club (0.31) [rec: read as Sword or Spear before]
        ("collect/20261028-190026/f0570_W.png", 920, 0),
        ("collect/20261028-190026/f0571_CLICK.png", 920, 0),
        ("collect/20261029-160032/f0730_SPACE.png", 544, 0),
        ("collect/20261029-160032/f0731_idle.png", 444, 0),
        ("collect/20261029-160032/f0732_idle.png", 444, 0),
        ("collect/20261029-160032/f0733_D.png", 444, 0),
        ("collect/20261029-160032/f0734_W.png", 536, 0),
    ],
    "shadow_dash": [   # The Twins' Shadow Dash, a '1' and a ghost with an arrow [rec: read as Charge before]
        ("collect/20261028-190026/f0562_CLICK.png", 1012, 0),
        ("collect/20261028-190026/f0563_CLICK.png", 1012, 0),
        ("collect/20261028-190026/f0564_CLICK.png", 1012, 0),
        ("collect/20261028-190026/f0579_SPACE.png", 920, 0),
        ("collect/20261029-160032/f0755_CLICK.png", 1100, 0),
        ("collect/20261029-160032/f0757_idle.png", 1100, 0),
        ("collect/20261029-160032/f0758_CLICK.png", 1100, 0),
        ("collect/20261029-160032/f0759_SPACE.png", 1100, 0),
    ],
    "dash": [   # The Twins' Phantom Leap (the Dash tile), a white bird, under Swap Toss [rec: read as the red Summon]
        ("collect/20261029-160032/f0739_CLICK.png", 1096, 0),
        ("collect/20261029-160032/f0740_SPACE.png", 1096, 0),
        ("collect/20261029-160032/f0741_idle.png", 1096, 0),
        ("collect/20261029-160032/f0742_idle.png", 1096, 0),
        ("collect/20261029-160032/f0743_A.png", 1096, 0),
        ("collect/20261029-160032/f0767_idle.png", 1004, 0),
        ("collect/20261029-160032/f0768_CLICK.png", 1004, 0),
        ("collect/20261029-160032/f0769_CLICK.png", 1004, 0),
        ("collect/20261029-160032/f0770_SPACE.png", 1004, 0),
    ],
    "volley": [   # the Yumi Sniper's Volley, red and yellow frames, the bright Hibiku and the dark Moonlit Port backdrops
        ("collect/20261029-160032/f0198_W.png", 444, 0),
        ("collect/20261029-160032/f0203_idle.png", 492, 0),
        ("collect/20261029-160032/f0207_idle.png", 1480, 0),
        ("collect/20261029-160032/f0216_D.png", 1432, 0),
        ("collect/20261029-160032/f0222_D.png", 1284, 0),
        ("collect/20261029-160032/f0226_SPACE.png", 816, 0),
        ("collect/20261029-160032/f0228_D.png", 912, 0),
        ("collect/20261029-160032/f0265_idle.png", 1284, 0),
        ("collect/20261029-160032/f0314_SPACE.png", 1192, 0),
        ("collect/20261029-160032/f0332_CLICK.png", 772, 0),
        ("collect/20261029-160032/f0512_idle.png", 1200, 0),
        ("collect/20261029-160032/f0514_A.png", 1200, 0),
        ("collect/20261029-160032/f0517_CLICK.png", 1388, 0),
        ("collect/20261029-160032/f0587_D.png", 820, 0),
        ("collect/20261029-160032/f0595_CLICK.png", 680, 0),
        ("collect/20261029-160032/f0597_SPACE.png", 732, 0),
    ],
    "swap_toss": [   # The Twins' Swap Toss, two circling arrows, over the Dash [rec: read as the red Summon or Earth Impale]
        ("collect/20261029-160032/f0740_SPACE.png", 1096, 1),
        ("collect/20261029-160032/f0741_idle.png", 1096, 1),
        ("collect/20261029-160032/f0742_idle.png", 1096, 1),
        ("collect/20261029-160032/f0743_A.png", 1096, 1),
        ("collect/20261029-160032/f0769_CLICK.png", 1004, 1),
        ("collect/20261029-160032/f0770_SPACE.png", 1004, 1),
        ("collect/20261029-160032/f0771_idle.png", 1004, 1),
    ],
    "grappling_hook": [   # the Grappler's Grappling Hook (the lower tile, under its Sword) and The Statue's, a '1' and a
                          # grey three-pronged hook over a coiled brown rope, read as Charge before (0.36)
        ("collect/20261028-190026/f0862_CLICK.png", 1480, 0),
        ("collect/20261028-190026/f0864_SPACE.png", 1480, 0),
        ("collect/20261028-190026/f0865_idle.png", 1480, 0),
        ("collect/20261028-190026/f0866_CLICK.png", 1480, 0),
        ("collect/20261028-190026/f0868_CLICK.png", 1480, 0),
        ("collect/20261028-190026/f0917_D.png", 1380, 0),
        ("collect/20261028-190026/f0918_D.png", 1332, 0),
        ("collect/20261028-190026/f0920_idle.png", 1284, 0),
        ("collect/20261028-190026/f0615_idle.png", 544, 0),
        ("collect/20261028-190026/f0620_A.png", 540, 0),
        ("collect/20261028-190026/f0777_CLICK.png", 548, 0),
        ("collect/20261028-190026/f0823_idle.png", 1480, 0),
        ("collect/20261028-190026/f0827_SPACE.png", 1480, 0),
        ("collect/20261028-190026/f0829_CLICK.png", 1484, 0),
        ("collect/20261028-190026/f0831_CLICK.png", 1480, 0),
        ("collect/20261029-160032/f0819_idle.png", 1192, 0),
        ("collect/20261029-160032/f0821_D.png", 1192, 0),
        ("collect/20261029-160032/f0826_idle.png", 1004, 0),
        ("collect/20261029-160032/f0828_CLICK.png", 1004, 0),
        ("collect/20261029-160032/f0868_idle.png", 1292, 0),
        ("collect/20261029-160032/f0871_SPACE.png", 1192, 0),
        ("collect/20261029-160032/f0988_idle.png", 816, 0),
        ("collect/20261029-160032/f1031_CLICK.png", 1284, 0),
        ("collect/20261029-160032/f1032_idle.png", 1284, 0),
    ],
    "shuriken": [   # the Shinobi's and The Statue's Shuriken, a '1' and a grey four-pointed star, read as Back Charge before
        ("collect/20261029-160032/f1075_CLICK.png", 1384, 0),
        ("collect/20261029-160032/f1078_SPACE.png", 1383, 0),
        ("collect/20261029-160032/f1079_idle.png", 1479, 0),
        ("collect/20261029-160032/f1080_CLICK.png", 1479, 0),
        ("collect/20261029-160032/f1082_idle.png", 1479, 0),
        ("collect/20261029-160032/f1083_S.png", 1479, 0),
        ("collect/20261029-160032/f1085_CLICK.png", 1480, 0),
        ("collect/20261029-160032/f1105_D.png", 1332, 0),
        ("collect/20261029-160032/f1107_CLICK.png", 1284, 0),
        ("collect/20261029-160032/f1109_idle.png", 1283, 0),
        ("collect/20261028-190026/f0672_SPACE.png", 732, 0),
        ("collect/20261028-190026/f0711_idle.png", 584, 0),
        ("collect/20261028-190026/f0712_CLICK.png", 584, 0),
        ("collect/20261028-190026/f0736_SPACE.png", 1432, 0),
        ("collect/20261028-190026/f0737_idle.png", 1432, 0),
        ("collect/20261028-190026/f0740_D.png", 1432, 0),
        ("collect/20261028-190026/f0742_CLICK.png", 1380, 0),
        ("collect/20261028-190026/f0743_D.png", 1380, 0),
        ("collect/20261028-190026/f0744_D.png", 1332, 0),
        ("collect/20261028-190026/f0745_SPACE.png", 1284, 0),
        ("collect/20261029-160032/f0953_idle.png", 492, 0),
        ("collect/20261029-160032/f0954_A.png", 492, 0),
        ("collect/20261029-160032/f0955_A.png", 540, 0),
        ("collect/20261029-160032/f0956_CLICK.png", 592, 0),
        ("collect/20261029-160032/f0957_A.png", 592, 0),
        ("collect/20261029-160032/f0958_SPACE.png", 628, 0),
    ],
    "smoke_bomb": [   # the Shinobi's Smoke Bomb, a '1' and a smoke cloud with a black up-arrow and an ember, read as
                      # Shadow Dash before (only recording 1 showed it)
        ("collect/20261028-190026/f0697_idle.png", 1388, 0),
        ("collect/20261028-190026/f0699_CLICK.png", 1388, 0),
        ("collect/20261028-190026/f0703_CLICK.png", 1388, 0),
        ("collect/20261028-190026/f0705_SPACE.png", 1340, 0),
        ("collect/20261028-190026/f0707_idle.png", 1336, 0),
        ("collect/20261028-190026/f0708_CLICK.png", 1344, 0),
        ("collect/20261028-190026/f0718_idle.png", 444, 0),
        ("collect/20261028-190026/f0720_CLICK.png", 444, 0),
        ("collect/20261028-190026/f0722_A.png", 396, 0),
        ("collect/20261028-190026/f0724_SPACE.png", 492, 0),
        ("collect/20261028-190026/f0726_CLICK.png", 488, 0),
        ("collect/20261028-190026/f0728_A.png", 492, 0),
        ("collect/20261028-190026/f0730_A.png", 592, 0),
    ],
    "shadow_kama": [   # The Statue's Shadow Kama, a '3' and a sickle with a purple handle, read as Arrow before
        ("collect/20261028-190026/f0892_CLICK.png", 1332, 0),
        ("collect/20261028-190026/f0893_CLICK.png", 1332, 0),
        ("collect/20261028-190026/f0897_idle.png", 1284, 0),
        ("collect/20261028-190026/f0898_idle.png", 1284, 0),
        ("collect/20261028-190026/f0900_A.png", 1332, 0),
        ("collect/20261028-190026/f0920_idle.png", 1284, 1),
        ("collect/20261029-160032/f1081_SPACE.png", 1479, 2),
        ("collect/20261029-160032/f1082_idle.png", 1479, 2),
        ("collect/20261029-160032/f1083_S.png", 1479, 2),
        ("collect/20261029-160032/f1118_CLICK.png", 1284, 0),
        ("collect/20261029-160032/f1119_CLICK.png", 1284, 0),
    ],
    "earth_impale@the_statue": [   # The Statue's Earth Impale; its bubble sits about 28 px lower than Iwao's
        ("collect/20261028-190026/f0866_CLICK.png", 1480, 2),
        ("collect/20261028-190026/f0867_SPACE.png", 1480, 2),
        ("collect/20261028-190026/f0868_CLICK.png", 1480, 2),
        ("collect/20261028-190026/f0907_D.png", 1332, 1),
        ("collect/20261028-190026/f0908_SPACE.png", 1284, 1),
        ("collect/20261028-190026/f0910_A.png", 1284, 1),
        ("collect/20261028-190026/f0912_CLICK.png", 1332, 1),
        ("collect/20261029-160032/f1107_CLICK.png", 1284, 1),
        ("collect/20261029-160032/f1108_SPACE.png", 1284, 1),
        ("collect/20261029-160032/f1110_CLICK.png", 1284, 1),
        ("collect/20261029-160032/f1111_SPACE.png", 1284, 1),
        ("collect/20261029-160032/f1113_A.png", 1284, 1),
    ],
    "sword@the_statue": [   # The Statue's Sword, slots 1-2, the stack's highest bob included (spear/sword at 0.04 before)
        ("collect/20261028-190026/f0893_CLICK.png", 1332, 1),
        ("collect/20261028-190026/f0894_D.png", 1332, 1),
        ("collect/20261028-190026/f0897_idle.png", 1284, 1),
        ("collect/20261028-190026/f0900_A.png", 1332, 1),
        ("collect/20261028-190026/f0910_A.png", 1284, 2),
        ("collect/20261028-190026/f0911_idle.png", 1332, 2),
        ("collect/20261029-160032/f1077_idle.png", 1383, 1),
        ("collect/20261029-160032/f1079_idle.png", 1479, 1),
        ("collect/20261029-160032/f1080_CLICK.png", 1479, 1),
        ("collect/20261029-160032/f1083_S.png", 1479, 1),
        ("collect/20261029-160032/f1092_idle.png", 1480, 1),
        ("collect/20261029-160032/f1094_CLICK.png", 1480, 1),
        ("collect/20261029-160032/f1097_idle.png", 1284, 1),
        ("collect/20261029-160032/f1120_SPACE.png", 1284, 1),
    ],
    "spear@the_statue": [   # The Statue's Spear, slots 0-2
        ("collect/20261028-190026/f0864_SPACE.png", 1480, 1),
        ("collect/20261028-190026/f0865_idle.png", 1480, 1),
        ("collect/20261028-190026/f0868_CLICK.png", 1480, 1),
        ("collect/20261028-190026/f0906_D.png", 1380, 0),
        ("collect/20261028-190026/f0907_D.png", 1332, 0),
        ("collect/20261028-190026/f0911_idle.png", 1332, 0),
        ("collect/20261029-160032/f1095_idle.png", 1480, 2),
        ("collect/20261029-160032/f1097_idle.png", 1284, 2),
        ("collect/20261029-160032/f1099_CLICK.png", 1284, 2),
    ],
    "arrow@the_statue": [   # The Statue's Arrow, slot 0 (read right before at confidence 0.19-0.56)
        ("collect/20261028-190026/f0875_A.png", 1380, 0),
        ("collect/20261028-190026/f0876_idle.png", 1432, 0),
        ("collect/20261028-190026/f0877_A.png", 1432, 0),
        ("collect/20261028-190026/f0883_CLICK.png", 1480, 0),
        ("collect/20261029-160032/f1090_W.png", 1480, 0),
        ("collect/20261029-160032/f1093_CLICK.png", 1480, 0),
        ("collect/20261029-160032/f1094_CLICK.png", 1480, 0),
        ("collect/20261029-160032/f1092_idle.png", 1480, 0),
        ("collect/20261029-160032/f1099_CLICK.png", 1284, 0),
    ],
    "tetsubo@the_statue": [   # The Statue's Tetsubo, slots 1-2
        ("collect/20261028-190026/f0876_idle.png", 1432, 1),
        ("collect/20261028-190026/f0878_SPACE.png", 1480, 1),
        ("collect/20261028-190026/f0880_idle.png", 1480, 2),
        ("collect/20261028-190026/f0881_idle.png", 1480, 2),
        ("collect/20261028-190026/f0897_idle.png", 1284, 2),
        ("collect/20261028-190026/f0900_A.png", 1332, 2),
        ("collect/20261029-160032/f1110_CLICK.png", 1284, 2),
        ("collect/20261029-160032/f1111_SPACE.png", 1284, 2),
        ("collect/20261029-160032/f1113_A.png", 1284, 2),
    ],
    "shield": [   # Kowa's Shield, red and yellow frames, the board panned to either side
        ("collect/20261028-190026/f1281_idle.png", 1292, 0),
        ("collect/20261028-190026/f1304_A.png", 584, 0),
        ("collect/20261028-190026/f1316_CLICK.png", 1011, 0),
        ("collect/20261028-190026/f1333_A.png", 540, 0),
        ("collect/20261028-190026/f1334_A.png", 592, 0),
        ("collect/20261029-160032/f1481_idle.png", 1192, 0),
        ("collect/20261029-160032/f1483_CLICK.png", 1191, 0),
        ("collect/20261028-190026/f1282_idle.png", 1292, 0),
        ("collect/20261029-160032/f1484_CLICK.png", 1332, 0),
        # the Shielder's Shield, the same drawing (on yellow frames under the fire flames it read as Summon at 1.0 before:
        # rec1 f1183, f1218)
        ("collect/20261028-190026/f1118_A.png", 452, 0),
        ("collect/20261028-190026/f1183_A.png", 304, 0),
        ("collect/20261028-190026/f1218_SPACE.png", 1284, 0),
        ("collect/20261029-160032/f1163_A.png", 1480, 0),
    ],
    "ally_shield": [   # the Shielder's Ally Shield, a blue wavy line and a small shield; on a yellow frame under the fire
                       # flames it read as Summon at 1.0 before (rec1 f1045)
        ("collect/20261028-190026/f0952_CLICK.png", 1292, 0),
        ("collect/20261028-190026/f0969_SPACE.png", 1099, 0),
        ("collect/20261028-190026/f0998_D.png", 1432, 0),
        ("collect/20261028-190026/f1111_CLICK.png", 544, 0),
        ("collect/20261028-190026/f1210_D.png", 1432, 0),
        ("collect/20261028-190026/f0999_SPACE.png", 1380, 0),
        ("collect/20261028-190026/f1045_idle.png", 1292, 0),
        ("collect/20261028-190026/f1063_CLICK.png", 1240, 0),
        ("collect/20261028-190026/f1196_idle.png", 1528, 0),
        ("collect/20261029-160032/f1159_idle.png", 1480, 0),
        ("collect/20261029-160032/f1293_idle.png", 732, 0),
        ("collect/20261029-160032/f1431_A.png", 1240, 0),
        ("collect/20261029-160032/f1439_SPACE.png", 1292, 0),
        ("collect/20261029-160032/f1295_A.png", 732, 0),
        ("collect/20261029-160032/f1332_idle.png", 1191, 0),
    ],
    "sword@grappler": [   # the Grappler's Sword, the upper tile of its stack, 4 px higher than the 120-px pitch (read as
                          # Spear 7 of 35 before)
        ("collect/20261028-190026/f0616_CLICK.png", 544, 1),
        ("collect/20261028-190026/f0617_A.png", 544, 1),
        ("collect/20261028-190026/f0620_A.png", 540, 1),
        ("collect/20261028-190026/f0778_CLICK.png", 548, 1),
        ("collect/20261028-190026/f0824_idle.png", 1480, 1),
        ("collect/20261028-190026/f0827_SPACE.png", 1480, 1),
    ],
    # Hideyoshi the Cunning (0.47): every tile of his combos as he draws it, red and yellow frames, both fights; his bubble
    # sits higher in its slot window than a regular unit's
    "origin_of_symmetry@hideyoshi": [
        ("collect/20261029-160032/f1694_CLICK.png", 1152, 0),
        ("collect/20261029-160032/f1695_SPACE.png", 1152, 0),
        ("collect/20261029-160032/f1697_idle.png", 1099, 0),
        ("collect/20261029-160032/f1698_idle.png", 1099, 0),
        ("collect/20261029-160032/f1700_CLICK.png", 1099, 0),
        ("collect/20261029-160032/f1723_idle.png", 1331, 0),
        ("collect/20261029-160032/f1726_idle.png", 1380, 0),
        ("collect/20261029-160032/f1729_SPACE.png", 1380, 0),
        ("collect/20261028-190026/f1553_D.png", 444, 0),
        ("collect/20261028-190026/f1583_idle.png", 536, 0),
        ("collect/20261028-190026/f1586_CLICK.png", 536, 0),
        ("collect/20261028-190026/f1587_SPACE.png", 536, 0),
    ],
    "swirl@hideyoshi": [
        ("collect/20261029-160032/f1695_SPACE.png", 1152, 1),
        ("collect/20261029-160032/f1697_idle.png", 1099, 1),
        ("collect/20261029-160032/f1698_idle.png", 1099, 1),
        ("collect/20261029-160032/f1700_CLICK.png", 1099, 1),
        ("collect/20261029-160032/f1726_idle.png", 1380, 1),
        ("collect/20261029-160032/f1729_SPACE.png", 1380, 1),
        ("collect/20261028-190026/f1538_idle.png", 1380, 0),
        ("collect/20261028-190026/f1540_idle.png", 1332, 0),
        ("collect/20261028-190026/f1542_idle.png", 1282, 0),
        ("collect/20261028-190026/f1542_idle.png", 1282, 2),
        ("collect/20261028-190026/f1543_A.png", 1282, 0),
        ("collect/20261028-190026/f1543_A.png", 1282, 2),
        ("collect/20261028-190026/f1571_idle.png", 1098, 0),
        ("collect/20261028-190026/f1573_SPACE.png", 1098, 0),
        ("collect/20261028-190026/f1575_CLICK.png", 1098, 0),
        ("collect/20261028-190026/f1575_CLICK.png", 1098, 2),
        ("collect/20261028-190026/f1576_idle.png", 1098, 0),
        ("collect/20261028-190026/f1576_idle.png", 1098, 2),
        ("collect/20261028-190026/f1586_CLICK.png", 536, 1),
        ("collect/20261028-190026/f1587_SPACE.png", 536, 1),
    ],
    "earth_impale@hideyoshi": [
        ("collect/20261029-160032/f1697_idle.png", 1099, 2),
        ("collect/20261029-160032/f1698_idle.png", 1099, 2),
        ("collect/20261029-160032/f1700_CLICK.png", 1099, 2),
        ("collect/20261029-160032/f1729_SPACE.png", 1380, 2),
        ("collect/20261028-190026/f1587_SPACE.png", 536, 2),
    ],
    "back_charge@hideyoshi": [
        ("collect/20261029-160032/f1706_idle.png", 914, 0),
        ("collect/20261029-160032/f1708_SPACE.png", 914, 0),
        ("collect/20261029-160032/f1710_idle.png", 914, 0),
        ("collect/20261029-160032/f1711_CLICK.png", 914, 0),
        ("collect/20261029-160032/f1717_SPACE.png", 1332, 2),
        ("collect/20261029-160032/f1718_idle.png", 1284, 2),
        ("collect/20261028-190026/f1534_idle.png", 1152, 2),
        ("collect/20261028-190026/f1535_CLICK.png", 1152, 2),
        ("collect/20261028-190026/f1549_idle.png", 492, 2),
        ("collect/20261028-190026/f1550_CLICK.png", 492, 2),
        ("collect/20261028-190026/f1564_CLICK.png", 816, 0),
        ("collect/20261028-190026/f1565_CLICK.png", 816, 0),
        ("collect/20261028-190026/f1566_SPACE.png", 816, 0),
    ],
    "mirror@hideyoshi": [
        ("collect/20261029-160032/f1708_SPACE.png", 914, 1),
        ("collect/20261029-160032/f1710_idle.png", 914, 1),
        ("collect/20261029-160032/f1711_CLICK.png", 914, 1),
        ("collect/20261028-190026/f1540_idle.png", 1332, 1),
        ("collect/20261028-190026/f1542_idle.png", 1282, 1),
        ("collect/20261028-190026/f1543_A.png", 1282, 1),
        ("collect/20261028-190026/f1565_CLICK.png", 816, 1),
        ("collect/20261028-190026/f1566_SPACE.png", 816, 1),
        ("collect/20261028-190026/f1573_SPACE.png", 1098, 1),
        ("collect/20261028-190026/f1575_CLICK.png", 1098, 1),
        ("collect/20261028-190026/f1576_idle.png", 1098, 1),
    ],
    "spear@hideyoshi": [
        ("collect/20261029-160032/f1710_idle.png", 914, 2),
        ("collect/20261029-160032/f1711_CLICK.png", 914, 2),
        ("collect/20261028-190026/f1566_SPACE.png", 816, 2),
    ],
    "dash@hideyoshi": [
        ("collect/20261029-160032/f1714_CLICK.png", 1332, 0),
        ("collect/20261029-160032/f1716_CLICK.png", 1332, 0),
        ("collect/20261029-160032/f1717_SPACE.png", 1332, 0),
        ("collect/20261029-160032/f1718_idle.png", 1284, 0),
        ("collect/20261028-190026/f1531_SPACE.png", 1152, 0),
        ("collect/20261028-190026/f1532_idle.png", 1152, 0),
        ("collect/20261028-190026/f1534_idle.png", 1152, 0),
        ("collect/20261028-190026/f1535_CLICK.png", 1152, 0),
        ("collect/20261028-190026/f1546_CLICK.png", 492, 0),
        ("collect/20261028-190026/f1547_SPACE.png", 492, 0),
        ("collect/20261028-190026/f1549_idle.png", 492, 0),
        ("collect/20261028-190026/f1550_CLICK.png", 492, 0),
    ],
    "bomb@hideyoshi": [
        ("collect/20261029-160032/f1716_CLICK.png", 1332, 1),
        ("collect/20261029-160032/f1717_SPACE.png", 1332, 1),
        ("collect/20261029-160032/f1718_idle.png", 1284, 1),
        ("collect/20261028-190026/f1532_idle.png", 1152, 1),
        ("collect/20261028-190026/f1534_idle.png", 1152, 1),
        ("collect/20261028-190026/f1535_CLICK.png", 1152, 1),
        ("collect/20261028-190026/f1547_SPACE.png", 492, 1),
        ("collect/20261028-190026/f1549_idle.png", 492, 1),
        ("collect/20261028-190026/f1550_CLICK.png", 492, 1),
    ],
    # The Shogun (0.48): his tiles as he draws them, red and yellow frames, both fights (his Origin, Mirror and Back Charge
    # are Hideyoshi's drawings, catalogue 'icon_art'; these add his own frames); Charge is a '1' and an upward arrowhead,
    # the Corrupted Barrage a '3' and a pink bar, the Shuriken a '1' and a four-point star [survey of 2026-11-02]
    "mirror@the_shogun": [
        ("collect/20261028-190026/f1743_idle.png", 1200, 0),
        ("collect/20261028-190026/f1744_SPACE.png", 1200, 0),
        ("collect/20261028-190026/f1752_idle.png", 920, 0),
        ("collect/20261028-190026/f1753_SPACE.png", 920, 0),
        ("collect/20261028-190026/f1812_SPACE.png", 920, 1),
        ("collect/20261028-190026/f1814_idle.png", 640, 0),
        ("collect/20261029-160032/f1972_SPACE.png", 1012, 1),
        ("collect/20261029-160032/f1995_SPACE.png", 920, 1),
        ("collect/20261029-160032/f1996_idle.png", 920, 1),
    ],
    "spear@the_shogun": [
        ("collect/20261028-190026/f1757_idle.png", 1200, 0),
        ("collect/20261028-190026/f1778_idle.png", 920, 1),
        ("collect/20261028-190026/f1781_idle.png", 920, 1),
        ("collect/20261029-160032/f1933_idle.png", 1200, 0),
        ("collect/20261029-160032/f1936_idle.png", 1100, 0),
        ("collect/20261029-160032/f1943_idle.png", 1100, 1),
        ("collect/20261029-160032/f1944_D.png", 1100, 1),
        ("collect/20261029-160032/f1949_idle.png", 632, 0),
        ("collect/20261029-160032/f1956_SPACE.png", 680, 0),
    ],
    "swirl@the_shogun": [
        ("collect/20261028-190026/f1763_D.png", 820, 0),
        ("collect/20261028-190026/f1807_idle.png", 1060, 1),
        ("collect/20261028-190026/f1826_idle.png", 1108, 1),
        ("collect/20261029-160032/f1928_SPACE.png", 1152, 0),
        ("collect/20261029-160032/f1929_idle.png", 1012, 0),
        ("collect/20261029-160032/f1990_idle.png", 1060, 1),
        ("collect/20261029-160032/f1991_CLICK.png", 1060, 1),
    ],
    "sword@the_shogun": [
        ("collect/20261028-190026/f1770_A.png", 732, 0),
        ("collect/20261029-160032/f1921_CLICK.png", 1152, 0),
        ("collect/20261029-160032/f1922_idle.png", 1012, 0),
        ("collect/20261029-160032/f1924_idle.png", 1060, 0),
    ],
    "charge@the_shogun": [
        ("collect/20261028-190026/f1775_idle.png", 920, 0),
        ("collect/20261028-190026/f1776_idle.png", 920, 0),
        ("collect/20261028-190026/f1778_idle.png", 920, 0),
        ("collect/20261028-190026/f1781_idle.png", 920, 0),
        ("collect/20261028-190026/f1807_idle.png", 1060, 0),
        ("collect/20261028-190026/f1826_idle.png", 1108, 0),
        ("collect/20261029-160032/f1941_SPACE.png", 1100, 0),
        ("collect/20261029-160032/f1943_idle.png", 1100, 0),
        ("collect/20261029-160032/f1990_idle.png", 1060, 0),
    ],
    "corrupted_barrage@the_shogun": [
        ("collect/20261028-190026/f1801_SPACE.png", 920, 1),
        ("collect/20261028-190026/f1802_idle.png", 920, 1),
        ("collect/20261029-160032/f1964_SPACE.png", 732, 1),
        ("collect/20261029-160032/f1965_idle.png", 732, 1),
        ("collect/20261029-160032/f1984_idle.png", 1012, 0),
        ("collect/20261029-160032/f1985_A.png", 1012, 0),
    ],
    "origin_of_symmetry@the_shogun": [
        ("collect/20261028-190026/f1800_CLICK.png", 920, 0),
        ("collect/20261029-160032/f1963_CLICK.png", 732, 0),
        ("collect/20261029-160032/f1964_SPACE.png", 732, 0),
        ("collect/20261029-160032/f1979_CLICK.png", 592, 0),
    ],
    "back_charge@the_shogun": [
        ("collect/20261028-190026/f1811_CLICK.png", 920, 0),
        ("collect/20261029-160032/f1971_CLICK.png", 1012, 0),
        ("collect/20261029-160032/f1972_SPACE.png", 1012, 0),
        ("collect/20261029-160032/f1994_CLICK.png", 920, 0),
        ("collect/20261029-160032/f1995_SPACE.png", 920, 0),
    ],
    # his Dragon Punch (0.51): every one seen on the bot's frames of four runs (3 yellow, 11 red), all in his first bar
    "dragon_punch@the_shogun": [
        ("runs/20261103-020829/frames/t0877_before.png", 772, 0),
        ("runs/20261103-020829/frames/t0878_before.png", 912, 0),
        ("runs/20261103-020829/frames/t0904_before.png", 724, 0),
        ("runs/20261103-020829/frames/t0906_before.png", 772, 0),
        ("runs/20261103-035113/frames/t0760_before.png", 1200, 0),
        ("runs/20261103-035113/frames/t0762_before.png", 1012, 0),
        ("runs/20261103-035113/frames/t0776_before.png", 1292, 0),
        ("runs/20261103-035113/frames/t0777_before.png", 1192, 0),
        ("runs/20261103-052436/frames/t0818_before.png", 632, 0),
        ("runs/20261103-052436/frames/t0820_before.png", 912, 0),
        ("runs/20261103-052436/frames/t0840_before.png", 1152, 0),
        ("runs/20261103-052436/frames/t0841_before.png", 1100, 0),
        ("runs/20261103-070730/frames/t0842_before.png", 1152, 0),
        ("runs/20261103-070730/frames/t0843_before.png", 1012, 0),
    ],
    "shuriken@the_shogun": [
        ("collect/20261028-190026/f1816_idle.png", 640, 0),
        ("collect/20261028-190026/f1817_idle.png", 640, 0),
        ("collect/20261029-160032/f1976_idle.png", 592, 0),
        ("collect/20261029-160032/f1977_CLICK.png", 592, 0),
        ("collect/20261029-160032/f1999_idle.png", 640, 0),
    ],
}


def interior(rgb: np.ndarray, cx: int, k: int) -> np.ndarray:
    """The icon template of bubble slot k over the unit at cx, as classify_icon sees it."""
    y0, y1 = SLOT_Y0 - SLOT_PITCH * k, SLOT_Y1 - SLOT_PITCH * k
    return cv2.resize(rgb[y0 + 12:y1 - 12, cx - 50:cx + 50], (25, 24), interpolation=cv2.INTER_AREA)


def build(root: str | Path) -> dict:
    root = Path(root)
    d = np.load(ASSETS / "icons.npz")
    # B marks the templates this builder added; files from 0.27 lack it, when it added only these
    built = d["B"] if "B" in d.files else np.isin(d["K"], LEGACY_LABELS)
    keep = ~built
    X, K = list(d["X"][keep]), list(d["K"][keep])
    B = [False] * len(K)
    added = {}
    for label, rows in SAMPLES.items():
        for rel, cx, k in rows:
            rgb = normalize(np.array(Image.open(root / rel).convert("RGB")))
            X.append(interior(rgb, cx, k))
            K.append(label)
            B.append(True)
        added[label] = len(rows)
    np.savez_compressed(ASSETS / "icons.npz", X=np.array(X, dtype=np.uint8), K=np.array(K), B=np.array(B))
    return {"kept": int(keep.sum()), "added": added}


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "."))
