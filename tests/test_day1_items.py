"""0.49: the loop's Day 1 requests 15 (a death at 3 HP in Hot Springs III: Dynamic Boost, Cool Up, floor items),
16 (upgrade slots) and 17 (the first run to the Shogun's Castle) [claude/core-requests.md]. Run-log tests skip when the
logs are missing."""
import json

import pytest

from test_bot_flow import DATA, _stub_windows_modules

from shogunbot.model import Enemy, Hero, Kit, State
from shogunbot.planner import plan_live

_stub_windows_modules()


def _logged(run: str, turn: int) -> State:
    path = DATA / f"runs/{run}/turns.jsonl"
    if not path.exists():
        pytest.skip("run log not available")
    row = next(json.loads(l) for l in open(path) if json.loads(l)["turn"] == turn)
    return State.from_dict(row["state"])


# ---------------------------------------------------------------- item 15a
def test_the_step_into_the_hooks_line_is_no_longer_planned():
    # run 20261102-191231 (0.47.1) t653, Hot Springs III at 3 HP: the hero stepped left into the line of a loaded
    # Grappler (Hook then Sword) four cells away, planning to kill it at t654 with Charge, Back Charge and Shuriken: 0.47.1
    # gave the Charge's own strike Dynamic Boost (3 planned, 2 real), the Grappler lived, pulled him and cut him: dead.
    # Since 0.48 a dash strike gets no boost (the Shuriken after a dash that moved the hero still does: 4 of 4 clean
    # logged cases), and the step is planned as the death it was
    s = _logged("20261102-191231", 653)
    d = plan_live(s)
    assert d.action != "left"
    assert next(c for c in d.candidates if c["action"] == "left")["value"] < -50000


# ---------------------------------------------------------------- item 15b
def test_cool_up_is_used_when_a_recharged_kit_saves_hp():
    # cornered on cell 0 by an armed Ashigaru facing him, the swap recharging: every move takes 2. Cool Up recharges the
    # swap too [catalogue: recharge_all], and the swap past the Ashigaru leaves it facing away
    from shogunbot.bot import Bot
    kit = Kit.of("swirl", "arrow")
    hero = Hero(0, 1, hp=4, kit=kit)
    hero.swap_charge = 0
    s = State(5, hero, [Enemy("ashigaru", 1, -1, 3, ["sword"], uid=1, intent="fire")], level="stage5-1", wave=1,
              waves_total=5)
    assert Bot._coolup_gain(s) >= 1.0
    hero.swap_charge = kit.swap_cooldown                    # nothing to recharge: no use
    assert Bot._coolup_gain(s) == 0.0
