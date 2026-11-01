"""The hero's queue bubbles (0.47, core-requests item 6).

Run 20261101-105146 (0.46) lost 5 HP on three attacks the simulator expected to kill: t387 (Spirit Gateway II, -1),
t639 and t669 (Hot Springs III, -2 each). In all three the queue was misread. The game had [swirl, spear, charge] at
t387, [charge] at t639 and [charge, arrow] at t669; the tracker planned [swirl, charge, spear], [spear] and
[spear, arrow], because perceive read the hero's Charge bubble (a '0' or '1' and a teal arrowhead) as Spear and his
Spear as Arrow with the enemy bubble icons. At t639 the hero turned and fired the 0-damage Charge at a 1-HP Ashigaru
whose Sword then hit; at t669 the Charge carried him to x7 next to the Yari Master (hence x7, predicted x6); at t387 the
Spear fired from x1 at empty cells before the Charge took the hero next to the Shinobi.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from shogunbot.bridge import BUBBLE_SURE, Tracker
from shogunbot.model import Kit
from test_rewards import face

DATA = Path(os.environ.get("SHOGUNBOT_DATA", Path(__file__).resolve().parents[1]))


def test_unsure_bubbles_keep_the_queue_the_bot_built():
    # t638-t639: one hand face unread (the counts disagree) and the Charge read as Spear at 0.05: the bot queued the
    # Charge on t637, so the queue is [charge]
    tr = Tracker(Kit.of("swirl", "arrow", "spear", "charge", "shuriken"))
    tr.queue_track = ["charge"]
    hand = [face("swirl"), face("arrow"), face("shuriken"), dict(face("spear"), tile=None)]   # one face unread
    assert tr._queue({"queue": ["spear"], "queue_conf": [0.05]}, hand) == ["charge"]
    # a sure read that contradicts the built queue still wins (a click the game did not take)
    assert tr._queue({"queue": ["spear"], "queue_conf": [1.0]}, hand) == ["spear"]


def test_unsure_bubble_does_not_reorder_the_queue():
    # t386: bubbles read [swirl 1.0, arrow 0.1, spear 0.24] over [swirl, spear, charge]; 0.46 took [swirl, charge,
    # spear] (two bubbles matched that way); unsure reads now leave the order to the bot's clicks
    tr = Tracker(Kit.of("swirl", "arrow", "spear", "charge"))
    tr.queue_track = ["swirl", "spear", "charge"]
    h = {"queue": ["swirl", "arrow", "spear"], "queue_conf": [1.0, 0.1, 0.24]}
    assert 0.24 < BUBBLE_SURE
    assert tr._queue(h, [face("arrow")]) == ["swirl", "spear", "charge"]
    # reads without confidences (older callers) count as sure, as before
    assert tr._queue({"queue": ["swirl", "arrow", "spear"]}, [face("arrow")]) == ["swirl", "charge", "spear"]


FRAMES = {   # frame: (the kit's tiles, the queue the bot built, bottom first)
    "runs/20261101-105146/frames/t0386_before.png": (["swirl", "arrow", "spear", "charge", "spear"],
                                                     ["swirl", "spear", "charge"]),
    "runs/20261101-105146/frames/t0638_before.png": (["swirl", "arrow", "spear", "charge", "spear", "shuriken"],
                                                     ["charge"]),
    "runs/20261101-105146/frames/t0669_before.png": (["swirl", "arrow", "spear", "charge", "spear", "shuriken"],
                                                     ["charge", "arrow"]),
    # a '3' Spear under a '2' Arrow, read the other way round on the whole bubble (Iwao, run 20261031-214756)
    "runs/20261031-214756/frames/t0137_before.png": (["swirl", "arrow", "spear", "arrow"], ["swirl", "spear", "arrow"]),
}


@pytest.mark.parametrize("frame", sorted(FRAMES))
def test_hero_bubbles_read_on_the_item_6_frames(frame):
    path = DATA / frame
    if not path.exists():
        pytest.skip("frame not available")
    from PIL import Image
    from shogunbot.perception.perceive import perceive
    kit, queue = FRAMES[frame]
    p = perceive(np.array(Image.open(path).convert("RGB")), kit)
    assert p.hero["queue"] == queue and min(p.hero["queue_conf"]) >= BUBBLE_SURE


def test_hero_icon_templates_cover_the_tiles_seen():
    from shogunbot.perception.perceive import HICO_K
    assert {"swirl", "arrow", "spear", "charge", "back_charge", "shuriken", "katana", "shadow_dash"} <= set(HICO_K)
