"""M3 step 2: the level generator, learned from recorded frames.

Every recording (the bot's runs/*/frames and the collect/* folders, each with its
actions.jsonl) is read again frame by frame the way the bot reads a fight since 0.21:
the HUD wave counter says when a wave arrives and how many the level has, and the
tracker's new units (bridge.Tracker.fresh) say what it brought. From those waves this
module learns, per level:

  waves        how many the level has (the counter's total)
  templates    each wave's units, per wave index (e.g. Bamboo Grove II: AA, AC, CT,
               AAC, ATT, CTT)
  cells        where units appear: a weight per distance from the hero
  facing       the chance a new unit faces the hero, per kind, from its next
               telegraph (turn = it faced away; queue, fire or a move = it faced him)
  next wave    when the next wave comes: at once when no enemy is left; with one
               left, for sure on the 6th turn in a row at one; with two, rarely

and writes data/waves.json, which self-play reads (selfplay.plan_levels, play_level).

    python -m shogunbot.levelgen build  <folder with runs/ and collect/>
    python -m shogunbot.levelgen report <folder with runs/ and collect/>

Stage 1 only for now: a folder is read up to the map after the first boss (the tracker
names levels by their width, which is right for Bamboo Grove only).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data" / "waves.json"
MIN_TEMPLATE_SEEN = 2       # a wave seen once in a level is taken as a misread (4 of 202 waves)
ONE_LEFT_AGE = (3, 5, 8)    # wave-age buckets (turns since the wave came) for the one-left chance


# ------------------------------------------------------------------ reading recordings
# Runs on Day 2, which brings elite enemies: the generator learns Day 1, the day the bot plays
# [user 2026-10-29]. Run 20261029-045856 (0.22) started on the newest day, Day 2 [camp f0003-f0008];
# since 0.23.1 the bot selects Day 1 and logs it (day_select).
DAY2_RUNS = frozenset({"20261029-045856"})


def recordings(root: str | Path) -> list[Path]:
    """Folders with an actions.jsonl: runs/*/frames and collect/*, Day 2 runs left out."""
    root = Path(root)
    out = [d for d in sorted((root / "runs").glob("2*/frames"))
           if (d / "actions.jsonl").exists() and d.parent.name not in DAY2_RUNS]
    out += [d for d in sorted((root / "collect").glob("2*")) if (d / "actions.jsonl").exists()]
    return out


def frame_waves(folder: str | Path) -> dict:
    """Waves and per-turn board counts of one recording, read from its frames.

    Returns {"waves": [...], "turns": [...]}: each wave with its level, number, the
    level's wave count, the new units (x, kind, facing read), survivors and the hero's
    cell; each turn with the level, wave, living enemies (kind, x, intent) and whether a
    wave arrived on it."""
    import numpy as np
    from PIL import Image

    from .bridge import Tracker
    from .perception.perceive import perceive
    from .replay import to_action
    from .rules import Events, apply_hero, legal_hero_actions
    from .waves import Waves

    folder = Path(folder)
    recs = [json.loads(l) for l in open(folder / "actions.jsonl", encoding="utf-8") if l.strip()]
    tr, wv = Tracker(), Waves()
    level, seen_boss, turn = None, False, 0
    waves, turns = [], []
    after = None              # the last state after the hero's action, before the enemies' (HP check)
    for rec in recs:
        f = folder / rec["frame"]
        if not f.exists():
            continue
        p = perceive(np.array(Image.open(f).convert("RGB")), [t.tile for t in tr.kit])
        if p.screen == "map" and seen_boss:
            break                                  # Stage 1 is over
        if p.screen == "reward":
            tr.next_level()                        # as the bot does (0.24)
            if seen_boss:
                break                              # the boss shrine: Stage 1 is over
        if p.screen not in ("combat", "boss") or p.hero is None:
            continue
        s = tr.state(p)
        if s is None:
            continue
        if s.level != level:
            level = s.level
            wv.reset()
        seen_boss = seen_boss or level == "boss"
        spawned = False
        if level != "boss":
            if s.alive():
                for name, fields in wv.update(p.wave, s.alive(), tr.fresh, turn):
                    if name == "wave_spawn":
                        spawned = True
                        more = _replaced(s, after, fields["spawned"])
                        if more:
                            fields = dict(fields, spawned=fields["spawned"] + more,
                                          survivors=max(0, len(s.alive()) - len(fields["spawned"]) - len(more)))
                        waves.append(dict(fields, level=level, cells=s.cells, hero_x=s.hero.x, frame=rec["frame"],
                                          alive_now=[(e.x, e.kind, e.facing) for e in s.alive()]))
            else:
                wv.prev_alive = 0
            sprite = {e["x"]: e["facing"] for e in p.enemies}      # the raw read, 0 under spawn flames
            turns.append({"turn": turn, "level": level, "wave": wv.wave, "spawned": spawned, "frame": rec["frame"],
                          "enemies": [(e.kind, e.x, e.intent, sprite.get(e.x, 0)) for e in s.alive()]})
        act = rec.get("bot_action") or to_action(rec, p)
        if act and act in legal_hero_actions(s):
            tr.acted(s, act)
            after = apply_hero(s, act, Events())
            turn += 1
    return {"folder": str(folder), "waves": waves, "turns": turns}


def _replaced(s, after, spawned: list) -> list[dict]:
    """New units the tracker took for old ones: same kind on a cell where the old unit was left
    with less HP by the hero's action (HP never rises), or where no unit of the kind was left
    within a cell. Run 20261029-100218 f0059: the Swirl left the Ashigaru on cell 1 at 1 HP or
    less and wave 4 put a full Ashigaru there; it was counted as a one-unit wave [0.27]."""
    if after is None:
        return []
    xs = {u["x"] for u in spawned}
    out = []
    for e in s.alive():
        if e.x in xs:
            continue
        near = [o for o in after.enemies if o.kind == e.kind and abs(o.x - e.x) <= 1 and o.hp > 0]
        same = next((o for o in near if o.x == e.x), None)
        if (same is not None and e.hp > same.hp) or not near:
            out.append({"x": e.x, "kind": e.kind, "facing": e.facing})
    return out


def _one(folder):
    return frame_waves(folder)


def read_all(root: str | Path, workers: int = 8) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor
    folders = recordings(root)
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_one, folders))


# ------------------------------------------------------------------ learning
def _template(units) -> tuple:
    return tuple(sorted(u["kind"] for u in units))


FACING_TURNS = 4            # turns after a unit came in which its first telegraph is looked for


def _true_facing(rec: dict, w: dict, u: dict) -> int | None:
    """+1 if the new unit faced the hero when it came, -1 if away, None if not shown.
    Its first telegraph other than hold while it stays on its cell (turn = it faced away;
    queue, fire or a move = it faced the hero) [0.20.1 facing pass]; failing that, for
    the units whose sprite shows a side (not the Spike Charger), the first sprite read
    clear of spawn flames. A single recorded frame misses some telegraphs the bot
    catches by sampling blinking bubbles, so a unit can look as if it held."""
    toward = 1 if w["hero_x"] > u["x"] else -1
    by_turn = {t["turn"]: t for t in rec["turns"] if t["level"] == w["level"]}
    sprite = 0
    for dt in range(0, FACING_TURNS + 1):
        t = by_turn.get(w["turn"] + dt)
        if t is None:
            break
        me = next((e for e in t["enemies"] if e[0] == u["kind"] and e[1] == u["x"]), None)
        if me is None:
            break
        it, sp = me[2], me[3]
        sprite = sprite or sp
        if dt == 0:
            continue
        if it == "turn":
            return -1
        if it in ("queue", "fire", "move-left", "move-right"):
            return 1
    if sprite and u["kind"] != "charger":
        return 1 if sprite == toward else -1
    return None


def _intervals(rec: dict) -> list[dict]:
    """The turns between two waves of a level: the living count after each turn (the
    next turn's count, or the survivors when the next wave came)."""
    out = []
    by_level = defaultdict(list)
    for t in rec["turns"]:
        by_level[t["level"]].append(t)
    spawns = {(w["level"], w["turn"]): w for w in rec["waves"]}
    for lv, ts in by_level.items():
        per = {}
        for t in ts:                               # one entry per turn: the last frame's board, and
            spawned = t["spawned"] or per.get(t["turn"], {}).get("spawned", False)   # any frame's wave
            per[t["turn"]] = dict(t, spawned=spawned)
        order = sorted(per)
        starts = [i for i, k in enumerate(order) if per[k]["spawned"]]
        for a, b in zip(starts, starts[1:]):
            alive = [len(per[k]["enemies"]) for k in order[a:b]]
            nxt = spawns[(lv, order[b])]
            out.append({"level": lv, "after": alive[1:] + [nxt["survivors"]]})
    return out


def learn(recs: list[dict]) -> dict:
    waves = [(r, w) for r in recs for w in r["waves"]]
    levels: dict = {}
    for lv in sorted({w["level"] for _, w in waves}):
        ws = [w for _, w in waves if w["level"] == lv]
        totals = Counter(w["total"] for w in ws if w.get("total"))
        tmpl_all = Counter(_template(w["spawned"]) for w in ws)
        keep = {t for t, n in tmpl_all.items() if n >= MIN_TEMPLATE_SEEN and t}
        per_wave = defaultdict(Counter)
        for w in ws:
            t = _template(w["spawned"])
            if t in keep:
                per_wave[w["wave"]][t] += 1
        n_waves = totals.most_common(1)[0][0] if totals else max(per_wave)
        # spawn cells: how often a free cell at distance d from the hero got a new unit
        got, free = Counter(), Counter()
        for w in ws:
            occupied = {x for x, k, f in w["alive_now"]}
            new = {u["x"] for u in w["spawned"]}
            for x in range(w["cells"]):
                if x == w["hero_x"] or (x in occupied and x not in new):
                    continue
                free[abs(x - w["hero_x"])] += 1
            for u in w["spawned"]:
                got[abs(u["x"] - w["hero_x"])] += 1
        rate = {d: got[d] / free[d] for d in free if free[d] >= 10}
        top = max(rate.values()) if rate else 1.0
        weights = {str(d): round(r / top, 2) for d, r in sorted(rate.items())}
        levels[lv] = {
            "waves": n_waves,
            "waves_seen": dict(sorted((str(k), v) for k, v in totals.items())),
            "templates": {str(i): [{"units": list(t), "seen": n} for t, n in per_wave[i].most_common()]
                          for i in sorted(per_wave)},
            "dropped_as_misread": {"+".join(t) or "-": n for t, n in tmpl_all.items() if t not in keep},
            "cell_weight_by_distance": weights,
            "cell_counts": {"spawned": {str(d): got[d] for d in sorted(got)}, "free": {str(d): free[d] for d in sorted(free)}},
        }
    # facing of new units, from their next telegraph
    toward, away = Counter(), Counter()
    for r, w in waves:
        for u in w["spawned"]:
            f = _true_facing(r, w, u)
            if f == 1:
                toward[u["kind"]] += 1
            elif f == -1:
                away[u["kind"]] += 1
    facing = {k: {"p_toward": round(toward[k] / (toward[k] + away[k]), 2), "toward": toward[k], "away": away[k]}
              for k in sorted(set(toward) | set(away)) if toward[k] + away[k] >= 10}
    # when the next wave comes
    ones = defaultdict(lambda: [0, 0])          # one left: (age bucket) -> [waves, turns], before the 6th turn
    sixth = [0, 0]
    twos = defaultdict(lambda: [0, 0])          # two left: (age >= 12) -> [waves, turns]
    for r in recs:
        for iv in _intervals(r):
            after = iv["after"]
            last = len(after) - 1
            run1 = 0
            for i, a in enumerate(after):
                if a == 0:
                    break
                run1 = run1 + 1 if a <= 1 else 0
                age = i + 1
                came = i == last
                if a == 1:
                    if run1 >= 6:
                        sixth[0] += came
                        sixth[1] += 1
                    else:
                        b = next((str(x) for x in ONE_LEFT_AGE if age <= x), f">{ONE_LEFT_AGE[-1]}")
                        ones[b][0] += came
                        ones[b][1] += 1
                elif a == 2:
                    b = "12+" if age >= 12 else "<12"
                    twos[b][0] += came
                    twos[b][1] += 1
    timing = {
        "none_left": "",
        "one_left_sixth_turn": {"waves": sixth[0], "turns": sixth[1]},
        "one_left_before": {k: {"p": round(v[0] / v[1], 3), "waves": v[0], "turns": v[1]} for k, v in sorted(ones.items())},
        "two_left": {k: {"p": round(v[0] / v[1], 3), "waves": v[0], "turns": v[1]} for k, v in sorted(twos.items())},
    }
    return {"levels": levels, "facing": facing, "next_wave": timing}


def build(root: str | Path, out: Path = DATA) -> dict:
    recs = read_all(root)
    model = learn(recs)
    n_waves = sum(len(r["waves"]) for r in recs)
    model["_meta"] = {
        "source": f"learned from {sum(1 for r in recs if r['waves'])} recordings ({n_waves} waves, every one "
                  f"from the HUD counter) [bot: python -m shogunbot.levelgen build, {date.today().isoformat()}]",
        "recordings": [Path(r["folder"]).parent.name if Path(r["folder"]).name == "frames" else Path(r["folder"]).name
                       for r in recs if r["waves"]],
        "waves_by_source": dict(Counter(w.get("source") for r in recs for w in r["waves"])),
    }
    # the timing note must match the data it summarises
    zero = sum(1 for r in recs for w in r["waves"] if w.get("survivors") == 0 and w["wave"] > 1)
    model["next_wave"]["none_left"] = f"the next wave comes at once [obs: {zero} waves came with no older enemy on the board]"
    out.write_text(json.dumps(model, indent=1, sort_keys=True), encoding="utf-8")
    generator.cache_clear()
    return model


# ------------------------------------------------------------------ using it
@lru_cache(maxsize=1)
def generator() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8")) if DATA.exists() else {}


def wave_templates(level: str, wave: int) -> list[tuple[tuple, int]]:
    """[(units, times seen)] for this level's wave number; the nearest learned wave
    number when this one was never seen (e.g. the stress scenario's 6th and 7th)."""
    lv = generator().get("levels", {}).get(level)
    if not lv:
        return []
    t = lv["templates"]
    key = str(wave) if str(wave) in t else str(min((int(k) for k in t), key=lambda k: abs(k - wave)))
    return [(tuple(x["units"]), x["seen"]) for x in t[key]]


def cell_weight(level: str, distance: int) -> float:
    w = generator().get("levels", {}).get(level, {}).get("cell_weight_by_distance", {})
    if not w:
        return 1.0
    if str(distance) in w:
        return w[str(distance)]
    return w[max(w, key=int)]                     # beyond the learned range: the farthest one's


def p_toward(kind: str, default: float = 0.5) -> float:
    return generator().get("facing", {}).get(kind, {}).get("p_toward", default)


def p_next_wave(alive_after: int, ones_in_row: int, age: int) -> float:
    """Chance the next wave arrives at the end of this turn: alive_after enemies left,
    ones_in_row turns in a row at one or none (this one included), age turns since the
    current wave came."""
    nw = generator().get("next_wave", {})
    if alive_after <= 0:
        return 1.0
    if alive_after == 1:
        if ones_in_row >= 6:
            return 1.0
        before = nw.get("one_left_before", {})
        b = next((str(x) for x in ONE_LEFT_AGE if age <= x), f">{ONE_LEFT_AGE[-1]}")
        return before.get(b, {}).get("p", 0.0)
    if alive_after == 2:
        return nw.get("two_left", {}).get("12+" if age >= 12 else "<12", {}).get("p", 0.0)
    return 0.0


# ------------------------------------------------------------------ report
def report(model: dict) -> str:
    L = [model.get("_meta", {}).get("source", "")]
    for lv, d in model["levels"].items():
        L.append(f"{lv}: {d['waves']} waves (counter totals {d['waves_seen']})")
        for i, ts in d["templates"].items():
            L.append(f"  wave {i}: " + ", ".join(f"{'+'.join(x['units'])} {x['seen']}" for x in ts))
        L.append(f"  cell weight by distance from the hero: {d['cell_weight_by_distance']}")
        if d["dropped_as_misread"]:
            L.append(f"  dropped as misreads: {d['dropped_as_misread']}")
    L.append("facing the hero when new: " + ", ".join(f"{k} {v['toward']}/{v['toward'] + v['away']}"
                                                     for k, v in model["facing"].items()))
    nw = model["next_wave"]
    L.append(f"next wave: {nw['none_left']}; one left: sixth turn in a row {nw['one_left_sixth_turn']['waves']}"
             f"/{nw['one_left_sixth_turn']['turns']}, before that by wave age "
             + ", ".join(f"{k}: {v['waves']}/{v['turns']}" for k, v in nw["one_left_before"].items())
             + "; two left: " + ", ".join(f"{k}: {v['waves']}/{v['turns']}" for k, v in nw["two_left"].items()))
    return "\n".join(L)


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv and argv[0] in ("build", "report") else "report"
    rest = argv[1:] if argv and argv[0] in ("build", "report") else argv
    root = rest[0] if rest else "."
    model = build(root) if cmd == "build" else learn(read_all(root))
    print(report(model))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
