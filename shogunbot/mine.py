"""M3: learn how the game behaves from the bot's own run logs.

Every turn the bot logs the board it read (turns.jsonl), including each
enemy's telegraph: the action the game has already decided for the coming
enemy phase. That makes every logged turn a labelled example of enemy
behaviour, and every wave_spawn event (events.jsonl) an example of the level
generator. This module turns the logs into those examples, scores the
behaviour model against them, and writes data/behaviour.json.

    python -m shogunbot.mine report <folder with runs/>
    python -m shogunbot.mine build  <folder with runs/>    -> shogunbot/data/behaviour.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import behaviour as B
from .model import ENEMY_TILE_DAMAGE, State


@dataclass
class Run:
    id: str
    version: str
    turns: list = field(default_factory=list)
    events: list = field(default_factory=list)


MIN_VERSION = (0, 8)   # older bots misread facing and blinking bubbles; leaving their runs out raised
                       # leave-one-run-out accuracy on 0.8+ runs from 85.8% to 87.2% [2026-10-28]


def _version(v: str) -> tuple:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


# Left out of the behaviour model [2026-10-29]: Day 2 runs have elite enemies (a Quick one fires
# right after loading, breaking R1).
SKIP_RUNS = {"20261029-045856": "Day 2: elite enemies [camp f0003-f0008]"}
SKIP_LEVELS: dict = {}
# Bots before 0.27 read Rei the Ruthless as Daisuke and her icons as his tiles, so her fights would
# be learned as his: a Stage 1 boss level in a 7-cell arena (hers; his has 5 [wiki]) is left out
# for those bots [runs 20261029-072323, -095311, -100218, -101549, -103242]. From 0.27 she is logged
# as herself.
REI_READ_FROM = (0, 27)


def _rei_as_daisuke(t: dict, version: str) -> bool:
    return (_version(version) < REI_READ_FROM and t.get("level") == "boss"
            and (t.get("state") or {}).get("cells") == 7)


# Bots before 0.28 planned a boss past Stage 1 (Iwao) as Daisuke, the stand-in, and rewrote his
# icons as Daisuke's tiles [run 20261029-124921 t159-t198]: those fights are left out too.
IWAO_READ_FROM = (0, 28)


def _stand_in_boss(t: dict, version: str) -> bool:
    lv = t.get("level") or ""
    return _version(version) < IWAO_READ_FROM and lv.startswith("stage") and lv.endswith("boss")


def load_runs(root: str | Path, min_version: tuple = MIN_VERSION) -> list[Run]:
    """Bot runs with turn logs under root/runs (stopped or empty runs, runs of bots older
    than min_version and SKIP_RUNS skipped; the turns of SKIP_LEVELS dropped)."""
    out = []
    for d in sorted((Path(root) / "runs").glob("2*")):
        tf = d / "turns.jsonl"
        if not tf.exists() or d.name in SKIP_RUNS:
            continue
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        turns = [json.loads(l) for l in tf.open(encoding="utf-8") if l.strip()]
        version = str(meta.get("bot_version", "?"))
        turns = [t for t in turns if (d.name, t.get("level")) not in SKIP_LEVELS
                 and not _rei_as_daisuke(t, version) and not _stand_in_boss(t, version)]
        ef = d / "events.jsonl"
        events = [json.loads(l) for l in ef.open(encoding="utf-8") if l.strip()] if ef.exists() else []
        if turns and _version(version) >= min_version:
            out.append(Run(d.name, version, turns, events))
    return out


# ------------------------------------------------------------------ enemy behaviour
def enemy_samples(run: Run) -> list[dict]:
    """One row per enemy per logged turn: the behaviour features, the
    telegraph as label (y), and what the 0.16 preset table predicted."""
    from .rules import _hits_hero, enemy_options
    rows = []
    prev_s, prev_turn, prev_rows, prev_sig, skipped = None, None, {}, None, 0
    for t in run.turns:
        # a turn whose board is identical to the last one is the same game turn read
        # again (the bot's action did not land, e.g. 47 retries in run 20261028-092419):
        # one example, not many, and the enemy's history does not advance
        sig = json.dumps({k: v for k, v in t["state"].items() if k != "turn"}, sort_keys=True)
        if sig == prev_sig:
            skipped += 1
            continue
        prev_sig = sig
        try:
            s = State.from_dict(t["state"])
        except Exception:
            prev_s = None
            continue
        alive = s.alive()
        consecutive = (prev_s is not None and t.get("level") == prev_turn.get("level")
                       and 0 < t.get("turn", 0) - prev_turn.get("turn", -99) <= 1 + skipped)
        pairs = B.match([(p.kind, p.x, p.intent) for p in prev_s.alive()], [(e.kind, e.x) for e in alive]) \
            if consecutive else {}
        cur_rows = {}
        for j, e in enumerate(alive):
            pr = prev_rows.get(pairs[j]) if j in pairs else None
            last, fage = B.carry(pr["intent"], pr["fage_raw"]) if pr else (None, None)
            f = B.features(s, e, last, fage)
            # the 0.16 preset's view: no telegraph, decided on the same board
            q = e.clone()
            q.intent = None
            q.primed = bool(q.queue) and q.wind >= 1 and ENEMY_TILE_DAMAGE.get(q.queue[0], 0) > 0 \
                and _hits_hero(s, q, q.queue[0])
            s2 = s.clone()
            s2.enemies = [q if u.uid == e.uid else u for u in s2.enemies]
            preset = Counter()
            try:
                for p, a in enemy_options(s2, q, presets=True):
                    preset[_action_class(a, q.x, s.hero.x)] += p
            except Exception:
                pass
            row = dict(f, y=B.to_class(e.intent, e.x, s.hero.x), intent=e.intent, fage_raw=fage,
                       preset=dict(preset), run=run.id, version=run.version, turn=t.get("turn"),
                       level=t.get("level"), x=e.x, hero_x=s.hero.x, queue=list(e.queue), next_queue=None)
            if pr is not None and pr["intent"] == "queue":
                pr["next_queue"] = list(e.queue)      # what the queued tile turned out to be
            rows.append(row)
            cur_rows[j] = row
        prev_s, prev_turn, prev_rows, skipped = s, t, cur_rows, 0
    return rows


def _action_class(action: tuple, e_x: int, hero_x: int) -> str:
    k = action[0]
    if k == "move":
        return "toward" if action[1] == (1 if hero_x > e_x else -1) else "away"
    return {"fire": "fire", "queue": "queue", "turn": "turn"}.get(k, "hold")


def score(rows: list[dict], predict) -> dict:
    """top-1: share of rows whose telegraph is the most likely class;
    mean_p: average probability given to the telegraph; nll: mean -log p."""
    n = top = 0
    ps, nll = 0.0, 0.0
    for r in rows:
        if r["y"] is None:
            continue
        p = predict(r)
        if not p:
            continue
        n += 1
        top += max(p, key=p.get) == r["y"]
        pr = max(p.get(r["y"], 0.0), 1e-3)
        ps += pr
        nll -= math.log(pr)
    return {"rows": n, "top1": round(top / max(n, 1), 3), "mean_p": round(ps / max(n, 1), 3),
            "nll": round(nll / max(n, 1), 3)}


def cross_validate(runs: list[Run]) -> dict:
    """Leave one run out: learn from the others, score on it."""
    per_run = {r.id: enemy_samples(r) for r in runs}
    learned, preset, rows_all = [], [], []
    for rid, rows in per_run.items():
        train = [x for k, v in per_run.items() if k != rid for x in v]
        m = B.Model.fit(train)
        for r in rows:
            r["p_learned"] = m.predict(r)
        rows_all += rows
    return {"learned": score(rows_all, lambda r: r["p_learned"]),
            "presets": score(rows_all, lambda r: r["preset"])}


# ------------------------------------------------------------------ spawns
def spawn_samples(run: Run) -> list[dict]:
    """wave_spawn events with the hero's cell at that turn."""
    hero_at = {(t.get("level"), t.get("turn")): (t.get("hero") or {}).get("x") for t in run.turns}
    level, out = None, []
    for e in run.events:
        if e.get("event") == "level_start":
            level = e.get("level")
        if e.get("event") == "wave_spawn":
            out.append({"run": run.id, "level": level, "wave": e.get("wave"), "turn": e.get("turn"),
                        "alive_before": e.get("alive_before"), "hero_x": hero_at.get((level, e.get("turn"))),
                        "spawned": e.get("spawned", [])})
    return out


# ------------------------------------------------------------------ build / report
def build(root: str | Path, out: Path = B.DATA) -> B.Model:
    runs = load_runs(root)
    rows = [x for r in runs for x in enemy_samples(r)]
    cv = cross_validate(runs)
    meta = {"source": f"learned from {len(runs)} bot runs ({runs[0].id} .. {runs[-1].id}), "
                      f"{sum(1 for r in rows if r['y'])} labelled enemy-turns; telegraph = label "
                      f"[bot: shogunbot.mine build, {date.today().isoformat()}]",
            "runs": [r.id for r in runs],
            "cross_validated": cv}
    m = B.Model.fit(rows, meta)
    out.write_text(json.dumps(m.to_json(), indent=1, sort_keys=True), encoding="utf-8")
    B.model.cache_clear()
    return m


def report(root: str | Path) -> str:
    runs = load_runs(root)
    L = []
    rows = [x for r in runs for x in enemy_samples(r)]
    L.append(f"{len(runs)} runs, {sum(len(r.turns) for r in runs)} turns, {len(rows)} enemy-turns "
             f"({sum(1 for r in rows if r['y'])} with a telegraph)")
    cv = cross_validate(runs)
    L.append(f"telegraph prediction, leave-one-run-out: learned {cv['learned']}; 0.16 presets {cv['presets']}")
    sp = [s for r in runs for s in spawn_samples(r)]
    L.append(f"{len(sp)} wave spawns logged")
    return "\n".join(L)


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv and argv[0] in ("report", "build") else "report"
    rest = argv[1:] if argv and argv[0] in ("report", "build") else argv
    root = rest[0] if rest else "."
    if cmd == "build":
        m = build(root)
        print(json.dumps(m.meta, indent=1))
    else:
        print(report(root))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
