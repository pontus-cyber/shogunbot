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
from .model import BOSS_KINDS, ENEMY_KINDS
from .perception.perceive import perceive
from .rules import step, legal_hero_actions

KEY_ACTION = {"A": "left", "D": "right", "W": "turn", "S": "wait", "SPACE": "attack"}


def _sig(s) -> dict:
    return {"hero": (s.hero.x, s.hero.hp),
            "enemies": sorted((e.x, e.kind, e.hp) for e in s.alive())}


def kind_ties(route) -> dict:
    """kind -> the kinds the same unit reads as on the next frame (perceive.pip_ties): a unit named by its tile
    (a Yumi Sniper's Volley, an Archer's Arrow) reads as the roster's first kind with its HP once its bubble is
    empty, and the other way round when its tile shows: a read, not a model error (0.35)."""
    from .perception.perceive import pip_ties
    return pip_ties(tuple(route))


def matches(pred: dict, alts: dict, obs: dict, ties: dict | None = None) -> bool:
    """A predicted next state against the observed one: the hero's cell and HP, and every predicted
    unit seen on its cell with its HP (new units in the observed frame are spawns or summons, not
    model errors). A unit's kind must match unless the simulator lacks the observed kind (it is
    planned as a stand-in, Tracker.state, so it is scored on its cell and HP: the Moonlit Port
    Guardian before 0.31 [M4: collect/20261028-190026 with its route, f0604-f1799]) or the observed
    kind is the unit's alt, the other kind it was planned as: a 2-pip unit without a readable tile is
    planned as Tachi and Archer, and its Arrow showing on the next frame names it (0.31) [collect/
    20261028-190026 f0352, f0372, f0444; -160032 f0471]. alts: (cell, kind) -> alt of predicted units.
    ties: kind -> kinds the same unit may read as on the next frame (kind_ties)."""
    if pred["hero"] != obs["hero"]:
        return False
    ties = ties or {}
    return all(any(ox == x and oh == h and (ok == k or ok == alts.get((x, k)) or ok not in ENEMY_KINDS
                                           or ok in ties.get(k, ()))
                   for ox, ok, oh in obs["enemies"])
               for x, k, h in pred["enemies"])


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


def logged_route(folder: Path) -> tuple[list | None, int | None]:
    """A bot run's route and day from its event log beside the frames folder: Bamboo Grove, then
    each region the map took it to (map_travel), and the day chosen at the camp (day_select)."""
    ev = folder.parent / "events.jsonl"
    if folder.name != "frames" or not ev.exists():
        return None, None
    route, day = ["bamboo_grove"], None
    for line in ev.open(encoding="utf-8"):
        e = json.loads(line)
        if e.get("event") == "map_travel" and e.get("region") and e["region"] not in route:
            route.append(e["region"])
        elif e.get("event") == "day_select" and e.get("day"):
            day = e["day"]
    return route, day


def replay(folder: str | Path, verbose: bool = True, route: list | None = None, day: int | None = None) -> dict:
    """route: the regions of the run in order, and day: its day, as the bot passes them to
    perception (M4). By default from route.json in the folder ({"route": [...], "day": 1}) when
    there is one, else from the run's event log (a bot run: logged_route), else Bamboo Grove only
    and the day unknown."""
    folder = Path(folder)
    recs = [json.loads(l) for l in open(folder / "actions.jsonl", encoding="utf-8")]
    percs = {}
    meta = folder / "route.json"
    if route is None and meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        route, day = info.get("route"), info.get("day", day)
    elif route is None:
        route, logged_day = logged_route(folder)
        day = day if day is not None else logged_day
    route = list(route or ["bamboo_grove"])

    tr = Tracker()
    tr.route = list(route)          # levels per region: Moonlit Port has three before its boss

    def regions():
        # the stage of the next fight: a reward after a boss starts the next stage
        stage = tr.stage + (1 if tr.advance and tr.level_i >= tr.boss_index() else 0)
        return tuple(route[:max(1, stage)])

    def P(i):
        if i not in percs:
            percs[i] = perceive(np.array(Image.open(folder / recs[i]["frame"]).convert("RGB")),
                                [t.tile for t in tr.kit], regions=regions(), day=day)
        return percs[i]

    n = ok_any = ok_top = skipped = 0
    rows, scored = [], []
    for i in range(len(recs) - 1):
        p = P(i)
        if recs[i].get("screen") == "camp" and tr.level_i >= 0:
            # a new run, as the bot logged its camp inputs: the fights before it were a run left over and
            # given up at the start, as the bot does since 0.24 (run 20261030-115221 f0001); its levels count
            # from the camp. The screen read alone is not used: recording 2's f1133 reads as the camp in mid-run
            tr = Tracker()
            tr.route = list(route)
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
                   "enemies": sorted((e["x"], e["kind"], e["hp"] if e["kind"] not in BOSS_KINDS else None)
                                     for e in q.enemies)}
        preds = []
        for prob, s2, ev in outs:
            sg = _sig(s2)
            sg["enemies"] = sorted((x, k, h if k not in BOSS_KINDS else None) for x, k, h in sg["enemies"])
            preds.append((prob, sg, {(e.x, e.kind): e.alt for e in s2.alive() if e.alt}))
        n += 1
        ties = kind_ties(regions())
        hit = [pr for pr, sg, alts in preds if matches(sg, alts, obs_sig, ties)]
        top = matches(preds[0][1], preds[0][2], obs_sig, ties)
        ok_any += bool(hit)
        ok_top += top
        scored.append([recs[i]["frame"], bool(top)])
        if not top:
            rows.append({"frame": recs[i]["frame"], "action": act, "board": s.render(),
                         "intents": [(e.x, e.intent) for e in s.enemies],
                         "pred_top": preds[0][1], "observed": obs_sig,
                         "matched_other": bool(hit)})
    # scored: every scored turn's frame and whether the top prediction matched, so two versions can be
    # compared on the turns both score when perception changes which frames are fights (0.30)
    res = {"turns": n, "top_match": ok_top, "any_match": ok_any, "skipped_off_kit": skipped,
           "mismatches": rows, "scored": scored}
    if verbose:
        print(f"turns scored {n}; top prediction matched {ok_top}; some outcome matched {ok_any}; "
              f"skipped (off-kit action) {skipped}")
        for r in rows:
            print(json.dumps(r))
    return res
