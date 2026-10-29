"""Enemy behaviour learned from the bot's run logs (M3).

The game decides every enemy's next action at the start of the hero's turn
and shows it as a telegraph (domain model section 6). Each logged turn is
therefore a labelled example: the board as the enemy saw it, and the action
it chose. `mine.py` turns the logs into counts per situation; this module
turns the counts into probabilities for the simulator and the planner.

Classes: fire, queue, turn, toward (step toward the hero), away, hold.

A situation is described from specific to general (LEVELS). A prediction
starts from the most general level and refines it with each more specific
level that has data (additive smoothing, ALPHA pseudo-counts from the level
above), so a rare situation leans on its broader pattern and a kind never
seen in the logs falls back on what all enemies do.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .catalogue import CAT
from .model import BOSS_KINDS, ENEMY_KINDS, ENEMY_TILE_DAMAGE, State

CLASSES = ("fire", "queue", "turn", "toward", "away", "hold")
FEATURES = ("kind", "face", "armed", "in_fp", "blocked", "dist", "last", "fage")
LEVELS = (
    ("kind", "face", "armed", "in_fp", "blocked", "dist", "last", "fage"),
    ("kind", "face", "armed", "in_fp", "blocked", "dist", "last"),
    ("kind", "face", "armed", "in_fp", "blocked", "last"),
    ("kind", "face", "armed", "in_fp", "last"),
    ("kind", "face", "armed", "in_fp"),
    ("face", "armed", "in_fp", "last"),
    ("face", "armed", "in_fp"),
    (),
)
ALPHA = 4.0        # pseudo-counts taken from the level above (best log-loss of 0.5-8 in leave-one-run-out)
MIN_P = 0.08       # options rarer than this are dropped (mostly read noise) and the rest renormalised
DATA = Path(__file__).parent / "data" / "behaviour.json"


# ------------------------------------------------------------------ features
def _first_ahead(s: State, x: int, d: int, skip_uid: int):
    nx = x + d
    while s.in_bounds(nx):
        u = s.unit_at(nx)
        if u is not None and getattr(u, "uid", None) != skip_uid:
            return u
        nx += d
    return None


def in_footprint(s: State, e, tile: str) -> bool:
    """The hero is where this tile would hit if fired now."""
    from .rules import _hits_hero
    return tile in ENEMY_TILE_DAMAGE and _hits_hero(s, e, tile)


def features(s: State, e, last: str | None = None, fage: int | None = None) -> dict:
    """What an enemy sees at the start of the hero's turn, plus its own
    recent history: the class of its last action and phases since it fired."""
    hx = s.hero.x
    toward = 1 if hx > e.x else -1
    nxt = e.x + toward
    return {
        "kind": e.kind,
        "face": e.facing == toward,
        "armed": bool(e.queue),
        "in_fp": bool(e.queue) and in_footprint(s, e, e.queue[0]),
        "blocked": not s.in_bounds(nxt) or s.unit_at(nxt) is not None,
        "dist": min(abs(e.x - hx), 3),
        "last": last,
        "fage": None if fage is None else min(fage, 2),
    }


def history_class(intent: str | None) -> str | None:
    """A telegraph as the 'last action' feature: moves are one class."""
    if intent in ("move-left", "move-right"):
        return "move"
    return intent if intent in ("fire", "queue", "turn", "hold") else None


def to_class(intent: str | None, e_x: int, hero_x: int) -> str | None:
    """A telegraph as a label, moves relative to the hero."""
    if intent in ("move-left", "move-right"):
        d = -1 if intent == "move-left" else 1
        return "toward" if d == (1 if hero_x > e_x else -1) else "away"
    return intent if intent in ("fire", "queue", "turn", "hold") else None


def match(prev: list, cur: list) -> dict[int, int]:
    """Pair this turn's enemies [(kind, x)] with last turn's [(kind, x, telegraph, ...)]:
    same kind, the cell its telegraph pointed to first, then the nearest within
    two cells (Daisuke dashes, so any distance for him). {cur index: prev index}"""
    out, used = {}, set()
    for j, (kind, x) in enumerate(cur):
        best, bc = None, None
        for i, p in enumerate(prev):
            if i in used or p[0] != kind:
                continue
            px, it = p[1], p[2]
            if abs(x - px) > 2 and kind not in BOSS_KINDS:
                continue
            exp = px + {"move-left": -1, "move-right": 1}.get(it, 0)
            c = abs(x - exp) + 0.5 * abs(x - px)
            if bc is None or c < bc:
                best, bc = i, c
        if best is not None:
            used.add(best)
            out[j] = best
    return out


def carry(prev_intent: str | None, prev_fage: int | None) -> tuple[str | None, int | None]:
    """An enemy's history one turn on: (last, phases since it fired)."""
    return history_class(prev_intent), 0 if prev_intent == "fire" else (prev_fage + 1 if prev_fage is not None else None)


def key(f: dict, level: tuple) -> str:
    return "|".join(f"{k}={f[k]}" for k in level)


# ------------------------------------------------------------------ model
class Model:
    def __init__(self, counts: list[dict], meta: dict | None = None, boss_tiles: dict | None = None):
        self.counts = counts            # per level: {key: {class: n}}
        self.meta = meta or {}
        # per boss: {kind: {"dist|loaded": {tile: n}}}, what a 'queue' telegraph loaded
        self.boss_tiles = boss_tiles or {}

    @staticmethod
    def fit(rows: list[dict], meta: dict | None = None) -> "Model":
        counts = [dict() for _ in LEVELS]
        boss = {}
        for r in rows:
            y = r.get("y")
            if y not in CLASSES:
                continue
            for i, lv in enumerate(LEVELS):
                c = counts[i].setdefault(key(r, lv), {})
                c[y] = c.get(y, 0) + 1
            nq = r.get("next_queue")
            if r["kind"] in BOSS_KINDS and r.get("intent") == "queue" and nq and nq[:len(r["queue"])] == r["queue"]:
                added = "+".join(nq[len(r["queue"]):])
                if added:
                    c = boss.setdefault(r["kind"], {}).setdefault(f"{r['dist']}|{'+'.join(r['queue']) or '-'}", {})
                    c[added] = c.get(added, 0) + 1
        return Model(counts, meta, boss)

    def boss_tile(self, dist: int, loaded: list, kind: str = "daisuke") -> list[tuple[float, str]] | None:
        """What a boss loads on a 'queue' telegraph (the bubble does not show it)."""
        lk = "+".join(loaded) or "-"
        tiles = self.boss_tiles.get(kind, {})
        pooled = {}
        for k, c in tiles.items():
            if k.split("|", 1)[1] == lk:
                for t, n in c.items():
                    pooled[t] = pooled.get(t, 0) + n
        if not pooled:
            return None
        tot_p = sum(pooled.values())
        base = {t: n / tot_p for t, n in pooled.items()}
        c = tiles.get(f"{min(dist, 3)}|{lk}", {})
        n = sum(c.values())
        p = {t: (c.get(t, 0) + ALPHA * base[t]) / (n + ALPHA) for t in base}
        keep = {t: v for t, v in p.items() if v >= MIN_P} or p
        tot = sum(keep.values())
        return sorted(((v / tot, t) for t, v in keep.items()), reverse=True)

    def predict(self, f: dict) -> dict[str, float]:
        p = {c: 1.0 / len(CLASSES) for c in CLASSES}
        for i in range(len(LEVELS) - 1, -1, -1):
            c = self.counts[i].get(key(f, LEVELS[i]))
            if not c:
                continue
            n = sum(c.values())
            p = {k: (c.get(k, 0) + ALPHA * p[k]) / (n + ALPHA) for k in CLASSES}
        return p

    def to_json(self) -> dict:
        return {"_meta": self.meta, "classes": list(CLASSES), "levels": [list(l) for l in LEVELS],
                "counts": self.counts, "boss_tiles": self.boss_tiles}

    @staticmethod
    def from_json(d: dict) -> "Model":
        assert [tuple(l) for l in d["levels"]] == list(LEVELS), "behaviour.json was built for other levels"
        bt = d.get("boss_tiles") or {}
        if bt and all("|" in k for k in bt):
            bt = {"daisuke": bt}         # built before 0.27: Daisuke's only
        return Model(d["counts"], d.get("_meta"), bt)


@lru_cache(maxsize=1)
def model() -> Model | None:
    if not DATA.exists():
        return None
    return Model.from_json(json.loads(DATA.read_text(encoding="utf-8")))


# ------------------------------------------------------------------ intents
def intents(s: State, e, m: Model | None = None) -> list[tuple[float, str]] | None:
    """[(p, telegraph)] for an enemy without a read telegraph, decided on the
    board at the start of the hero's turn. None when no model is available."""
    m = m or model()
    if m is None:
        return None
    f = features(s, e, getattr(e, "last", None), getattr(e, "fage", None))
    loaded_earlier = bool(e.queue) and e.wind >= 1 and f["last"] != "queue"
    # rules confirmed in play come first; the model fills in everything else
    boss = e.kind in BOSS_KINDS
    kind = CAT.enemy(e.kind) if boss else None
    b = kind.behaviour if boss else {}
    toward = 1 if s.hero.x > e.x else -1

    def step(d: int) -> str:                   # a move telegraph, or hold when the cell is taken
        nx = e.x + d
        if not s.in_bounds(nx) or s.unit_at(nx) is not None:
            return "hold"
        return "move-right" if d > 0 else "move-left"

    # Charge alone is half Daisuke's combo; Earth Impale is half Iwao's only with the hero next to him
    half = boss and kind.combo_next(e.queue, abs(e.x - s.hero.x)) is not None
    if boss and b.get("after_fire") == "away" and not e.queue and e.fage is not None:
        # Iwao: the turn after he fires he steps away from the hero (or holds when blocked), and
        # loads the turn after [run 20261029-124921: after each of his 8 shots, t162-t198]
        return [(1.0, step(-toward) if e.fage == 0 else "queue")]
    if loaded_earlier and boss and CAT.tile(e.queue[0]).kind == "summon":
        return [(1.0, "fire")]                # a loaded Summon holds one phase, then fires [rules.py, 6 of 6
                                              # cases; Rei's bells 10 of 10, runs 20261029-072323 ... -103242]
    if loaded_earlier and boss and len(e.queue) > 1 and e.kind != "daisuke":
        return [(1.0, "fire")]                # Rei fires a finished combo one phase after it (12 of 12)
    if loaded_earlier and half and e.kind != "daisuke":
        return [(1.0, "queue")]               # ... and loads its second half after the wind-up phase (12 of 12)
    if loaded_earlier and f["in_fp"] and ENEMY_TILE_DAMAGE.get(e.queue[0], 0) > 0 and not half:
        return [(1.0, "fire")]                # R2: loaded, hero in the footprint at the start of the turn
    if loaded_earlier and b.get("approach") and not half and ENEMY_TILE_DAMAGE.get(e.queue[0], 0) > 0:
        # Iwao, loaded with the hero out of reach: steps toward him, then fires [run 20261029-124921
        # t166, t171, t180; t191 blocked]; a Spear facing away needs a turn first
        if CAT.tile(e.queue[0]).directional and not f["face"]:
            return [(1.0, "turn")]
        return [(1.0, step(toward))]
    p = m.predict(f)
    # what the rules allow [domain model R1 and section 7]
    if not e.queue:
        p["fire"] = 0.0
    elif not boss:
        p["queue"] = 0.0                      # one tile per regular enemy
    if e.queue and (f["last"] == "queue" or e.wind < 1):
        p["fire"] = 0.0                       # R1: never fires the phase right after loading
    if e.kind != "daisuke" and not f["in_fp"]:
        p["fire"] = 0.0                       # regular enemies fire only at the hero: 1 of 424 telegraphs
                                              # outside the footprint, a misread [runs 0.8+, mine.py]
    if half:
        p["fire"] = 0.0                       # Charge is half his combo: Sword follows before he fires
                                              # [run 20261028-053637 turns 120-123, 130-133]
    for cls, d in (("toward", toward), ("away", -toward)):
        nx = e.x + d
        if not s.in_bounds(nx) or s.unit_at(nx) is not None:
            p["hold"] += p[cls]
            p[cls] = 0.0
    keep = {k: v for k, v in p.items() if v >= MIN_P}
    if not keep:
        keep = {"hold": 1.0}
    tot = sum(keep.values())
    out = []
    for k, v in sorted(keep.items(), key=lambda kv: -kv[1]):
        if k in ("toward", "away"):
            d = toward if k == "toward" else -toward
            out.append((v / tot, "move-right" if d > 0 else "move-left"))
        else:
            out.append((v / tot, k))
    return out
