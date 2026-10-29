"""Replay a collected run through perception + rules and score the model.

For every (frame, input, next frame) triple in combat, the recorded input is
mapped to a model action, the simulator predicts the outcome from the
perceived state (with telegraphed intents as the first enemy ply), and the
prediction is compared with what the next frame shows. Mismatches are the
to-do list for rules and perception.

    python -m shogunbot replay collect/20261027-164727
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .bridge import Tracker
from .perception.perceive import perceive
from .rules import step, legal_hero_actions

KEY_ACTION = {"A": "left", "D": "right", "W": "turn", "S": "wait", "SPACE": "attack"}


def _sig(s) -> dict:
    return {"hero": (s.hero.x, s.hero.hp),
            "enemies": sorted((e.x, e.kind, e.hp) for e in s.alive())}


def to_action(rec: dict, p) -> str | None:
    a = rec.get("action")
    if a in KEY_ACTION:
        return KEY_ACTION[a]
    if a == "CLICK" and rec.get("click") and p.hand:
        fx, fy = rec["click"]
        if fy < 0.8:
            return None
        t = min(p.hand, key=lambda h: abs(h["fx"] - fx))
        if abs(t["fx"] - fx) < 0.04:
            return f"queue:{t['tile']}"
    return None


def replay(folder: str | Path, verbose: bool = True) -> dict:
    folder = Path(folder)
    recs = [json.loads(l) for l in open(folder / "actions.jsonl", encoding="utf-8")]
    percs = {}

    tr = Tracker()

    def P(i):
        if i not in percs:
            percs[i] = perceive(np.array(Image.open(folder / recs[i]["frame"]).convert("RGB")),
                                [t.tile for t in tr.kit])
        return percs[i]

    n = ok_any = ok_top = skipped = 0
    rows = []
    for i in range(len(recs) - 1):
        p = P(i)
        if p.screen == "reward":
            tr.next_level()             # as the bot does: the next fight is the next level (0.24)
        if p.screen not in ("combat", "boss") or not p.enemies:
            continue
        act = to_action(recs[i], p)
        if act is None:
            continue
        s = tr.state(p)
        if s is None:
            continue
        legal = legal_hero_actions(s)
        if act not in legal:
            # e.g. a tile outside the default kit (the recorded run took a Spear)
            skipped += 1
            continue
        outs = step(s, act)
        tr.acted(s, act)
        q = P(i + 1)
        if q.screen not in ("combat", "boss") or q.hero is None:
            continue
        obs_sig = {"hero": (q.hero["x"], q.hero["hp"]),
                   "enemies": sorted((e["x"], e["kind"], e["hp"] if e["kind"] != "daisuke" else None)
                                     for e in q.enemies)}
        preds = []
        for prob, s2, ev in outs:
            sg = _sig(s2)
            sg["enemies"] = sorted((x, k, h if k != "daisuke" else None) for x, k, h in sg["enemies"])
            preds.append((prob, sg))
        n += 1
        def same(sg):
            # new enemies in the observed frame are spawns/summons, not model errors
            return sg["hero"] == obs_sig["hero"] and all(e in obs_sig["enemies"] for e in sg["enemies"])
        hit = [pr for pr, sg in preds if same(sg)]
        top = same(preds[0][1])
        ok_any += bool(hit)
        ok_top += top
        if not top:
            rows.append({"frame": recs[i]["frame"], "action": act, "board": s.render(),
                         "intents": [(e.x, e.intent) for e in s.enemies],
                         "pred_top": preds[0][1], "observed": obs_sig,
                         "matched_other": bool(hit)})
    res = {"turns": n, "top_match": ok_top, "any_match": ok_any, "skipped_off_kit": skipped,
           "mismatches": rows}
    if verbose:
        print(f"turns scored {n}; top prediction matched {ok_top}; some outcome matched {ok_any}; "
              f"skipped (off-kit action) {skipped}")
        for r in rows:
            print(json.dumps(r))
    return res
