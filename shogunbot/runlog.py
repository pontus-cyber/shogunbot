"""Run logging built for improvement between runs.

Layout (one folder per run):

    runs/<run_id>/
        meta.json        config, weights, versions, calibration used
        turns.jsonl      one record per turn: perceived state, decision with
                         all candidate scores, predicted outcomes, observed
                         next state, simulator diff, damage attribution
        events.jsonl     level/wave starts and ends, spawns, deaths, rewards,
                         perception failures, exceptions with tracebacks
        frames/          PNGs: before/after frames for every turn with damage,
                         a simulator mismatch or low perception confidence
        summary.json     KPIs for the run (see Summary)
    runs/index.csv       one row per run, for quick comparison

Every turn record is written only after the next state is observed, so it
always contains both the prediction and what really happened.
"""

from __future__ import annotations

import csv
import json
import os
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .model import State

INDEX_FIELDS = [
    "run_id", "started", "result", "reached_level", "waves_cleared", "turns", "hp_end",
    "hp_lost", "hp_lost_predicted", "hp_lost_unpredicted", "kills", "double_kills",
    "sim_mismatch_turns", "low_conf_turns", "avg_plan_ms", "avg_perceive_ms", "minutes", "notes",
]


def state_diff(pred: State, obs: State) -> list[str]:
    """Field-level differences between a predicted and an observed state."""
    out = []
    ph, oh = pred.hero, obs.hero
    for f in ("x", "facing", "hp"):
        if getattr(ph, f) != getattr(oh, f):
            out.append(f"hero.{f}: pred={getattr(ph, f)} obs={getattr(oh, f)}")
    pmap = {e.x: e for e in pred.alive()}
    omap = {e.x: e for e in obs.alive()}
    for x in sorted(set(pmap) | set(omap)):
        p, o = pmap.get(x), omap.get(x)
        if p is None:
            out.append(f"cell{x}: pred=empty obs={o.kind}")
        elif o is None:
            out.append(f"cell{x}: pred={p.kind} obs=empty")
        else:
            if p.kind != o.kind:
                out.append(f"cell{x}.kind: pred={p.kind} obs={o.kind}")
            for f in ("hp", "facing"):
                if getattr(p, f) != getattr(o, f):
                    out.append(f"cell{x}.{f}: pred={getattr(p, f)} obs={getattr(o, f)}")
            if p.armed != o.armed:
                out.append(f"cell{x}.armed: pred={p.armed} obs={o.armed}")
    return out


def diff_category(line: str) -> str:
    """'cell3.facing: ...' -> 'enemy.facing' for aggregation."""
    key = line.split(":", 1)[0]
    if key.startswith("cell"):
        return "enemy." + key.split(".", 1)[1] if "." in key else "enemy.presence"
    return key


@dataclass
class _Stats:
    turns: int = 0
    hp_lost: int = 0
    hp_lost_predicted: int = 0
    hp_lost_unpredicted: int = 0
    hp_lost_by_enemy: Counter = field(default_factory=Counter)
    kills: Counter = field(default_factory=Counter)
    double_kills: int = 0
    sim_mismatch_turns: int = 0
    mismatch_by_field: Counter = field(default_factory=Counter)
    low_conf_turns: int = 0
    plan_ms: list = field(default_factory=list)
    perceive_ms: list = field(default_factory=list)
    actions: Counter = field(default_factory=Counter)
    levels: list = field(default_factory=list)       # [{level, hp_start, hp_end, turns, waves}]
    waves_cleared: int = 0
    exceptions: int = 0


class RunLogger:
    def __init__(self, root: str | os.PathLike = "runs", frame_policy: str = "events"):
        """frame_policy: 'events' (damage / mismatch / low confidence), 'all' or 'none'."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.frame_policy = frame_policy
        self.run_dir: Path | None = None
        self.run_id = ""
        self.stats = _Stats()
        self._pending: dict | None = None
        self._t0 = time.time()
        self._level: dict | None = None
        self.last_turn: dict | None = None   # the last finished turn record (prediction and outcome)
        self.on_event = None                 # callback(kind, data), e.g. the live viewer

    # ---- lifecycle ------------------------------------------------------
    def start_run(self, meta: dict) -> str:
        # one logger serves a whole series (play --runs N): each run starts from zero
        self.stats = _Stats()
        self._pending = None
        self._t0 = time.time()
        self._level = None
        self.last_turn = None
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        n = 2
        while (self.root / run_id).exists():      # two runs started in the same second
            run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{n}"
            n += 1
        self.run_id = run_id
        self.run_dir = self.root / self.run_id
        (self.run_dir / "frames").mkdir(parents=True, exist_ok=True)
        meta = {"run_id": self.run_id, "started": datetime.now().isoformat(timespec="seconds"), **meta}
        self._write_json("meta.json", meta)
        self.event("run_start")
        return self.run_id

    def end_run(self, result: str, notes: str = "") -> dict:
        self._flush_pending(None, None)
        if self._level:
            self.level_end(self._level.get("hp_end", 0), result)
        s = self.stats
        summary = {
            "run_id": self.run_id,
            "result": result,
            "notes": notes,
            "minutes": round((time.time() - self._t0) / 60, 2),
            "turns": s.turns,
            "levels": s.levels,
            "reached_level": s.levels[-1]["level"] if s.levels else "",
            "waves_cleared": s.waves_cleared,
            "hp_end": s.levels[-1]["hp_end"] if s.levels else None,
            "hp_lost": s.hp_lost,
            "hp_lost_predicted": s.hp_lost_predicted,
            "hp_lost_unpredicted": s.hp_lost_unpredicted,
            "hp_lost_by_enemy": dict(s.hp_lost_by_enemy),
            "kills": dict(s.kills),
            "double_kills": s.double_kills,
            "sim_mismatch_turns": s.sim_mismatch_turns,
            "mismatch_by_field": dict(s.mismatch_by_field.most_common()),
            "low_conf_turns": s.low_conf_turns,
            "avg_plan_ms": round(sum(s.plan_ms) / len(s.plan_ms), 1) if s.plan_ms else None,
            "avg_perceive_ms": round(sum(s.perceive_ms) / len(s.perceive_ms), 1) if s.perceive_ms else None,
            "actions": dict(s.actions),
            "exceptions": s.exceptions,
        }
        self._write_json("summary.json", summary)
        self.event("run_end", result=result)
        self._append_index(summary)
        return summary

    # ---- levels / waves -------------------------------------------------
    def level_start(self, level: str, hp: int, waves_total: int = 0) -> None:
        if self._level:
            self.level_end(hp, "next")
        self._level = {"level": level, "hp_start": hp, "hp_end": hp, "turns": 0,
                       "waves_total": waves_total, "waves_seen": 0}
        self.event("level_start", level=level, hp=hp, waves_total=waves_total)

    def level_end(self, hp: int, outcome: str) -> None:
        if not self._level:
            return
        self._level["hp_end"] = hp
        self._level["outcome"] = outcome
        self.stats.levels.append(self._level)
        self.event("level_end", **self._level)
        self._level = None

    def wave_seen(self, wave: int, spawned: list[dict], alive_before: int, turn: int) -> None:
        if self._level:
            if wave > self._level["waves_seen"] and self._level["waves_seen"] > 0:
                self.stats.waves_cleared += 1
            self._level["waves_seen"] = max(wave, self._level["waves_seen"])
        self.event("wave_spawn", wave=wave, turn=turn, alive_before=alive_before, spawned=spawned)

    # ---- turns ----------------------------------------------------------
    def turn(self, *, state: State, confidence: dict, perceive_ms: float, decision, plan_ms: float,
             frame=None, weights: dict | None = None) -> None:
        """Record the decision for this turn. Finalised when the next state arrives."""
        self._flush_pending(state, frame)
        low = {k: v for k, v in confidence.items() if v < 0.8}
        rec = {
            "run_id": self.run_id,
            "level": state.level,
            "wave": state.wave,
            "turn": state.turn,
            "ts": round(time.time() - self._t0, 2),
            "board": state.render(),
            "hero": {"x": state.hero.x, "facing": state.hero.facing, "hp": state.hero.hp,
                     "queue": state.hero.queue, "charge": state.hero.charge,
                     "swap_charge": state.hero.swap_charge},
            "state": state.to_dict(),
            "confidence": confidence,
            "low_confidence": low,
            "perceive_ms": round(perceive_ms, 1),
            "decision": {
                "action": decision.action,
                "value": decision.value,
                "depth": decision.depth,
                "nodes": decision.nodes,
                "plan_ms": round(plan_ms, 1),
                "candidates": decision.candidates[:8],
            },
            "predicted": [{k: v for k, v in p.items() if k != "state"} for p in decision.predicted],
            "_predicted_states": [p["state"] for p in decision.predicted],
            "_predicted_p": [p["p"] for p in decision.predicted],
            "_frame_before": frame,
        }
        self.stats.turns += 1
        self.stats.actions[decision.action] += 1
        self.stats.plan_ms.append(plan_ms)
        self.stats.perceive_ms.append(perceive_ms)
        if low:
            self.stats.low_conf_turns += 1
        if self._level:
            self._level["turns"] += 1
            self._level["hp_end"] = state.hero.hp
        self._pending = rec

    def _flush_pending(self, observed: State | None, frame_after) -> None:
        rec = self._pending
        if rec is None:
            return
        self._pending = None
        before_frame = rec.pop("_frame_before")
        preds = [State.from_dict(d) for d in rec.pop("_predicted_states")]
        probs = rec.pop("_predicted_p")
        if observed is not None and observed.level == rec["level"] and observed.wave == rec["wave"]:
            # pick the predicted outcome closest to what happened
            diffs = [state_diff(p, observed) for p in preds]
            best = min(range(len(diffs)), key=lambda i: len(diffs[i])) if diffs else None
            diff = diffs[best] if best is not None else ["no prediction"]
            hp_before = rec["hero"]["hp"]
            lost = max(0, hp_before - observed.hero.hp)
            pred_events = rec["predicted"][best]["events"] if best is not None else {}
            pred_lost = sum(d for _, _, d in pred_events.get("hero_damage", []))
            rec["observed"] = {"board": observed.render(), "hero_hp": observed.hero.hp}
            rec["matched_outcome"] = best
            rec["matched_p"] = probs[best] if best is not None else 0.0
            rec["sim_diff"] = diff
            rec["hp_lost"] = lost
            rec["hp_lost_predicted"] = min(lost, pred_lost)
            rec["damage_sources_predicted"] = pred_events.get("hero_damage", [])
            if lost:
                self.stats.hp_lost += lost
                self.stats.hp_lost_predicted += min(lost, pred_lost)
                self.stats.hp_lost_unpredicted += max(0, lost - pred_lost)
                for kind, _, d in pred_events.get("hero_damage", []) or [("unknown", 0, lost)]:
                    self.stats.hp_lost_by_enemy[kind] += d
            kills = pred_events.get("kills", []) if not diff else []
            for kind, _, _ in kills:
                self.stats.kills[kind] += 1
            if len(kills) >= 2:
                self.stats.double_kills += 1
            if diff:
                self.stats.sim_mismatch_turns += 1
                for line in diff:
                    self.stats.mismatch_by_field[diff_category(line)] += 1
            save = self.frame_policy == "all" or (
                self.frame_policy == "events" and (lost or diff or rec["low_confidence"]))
            if save:
                rec["frames"] = {
                    "before": self.save_frame(before_frame, f"t{rec['turn']:04d}_before"),
                    "after": self.save_frame(frame_after, f"t{rec['turn']:04d}_after"),
                }
        else:
            rec["observed"] = None
            rec["sim_diff"] = ["level or wave changed; not compared"]
        self.last_turn = rec
        self._append_jsonl("turns.jsonl", rec)

    # ---- misc -----------------------------------------------------------
    def event(self, kind: str, **data) -> None:
        self._append_jsonl("events.jsonl", {"ts": round(time.time() - self._t0, 2), "event": kind, **data})
        if self.on_event is not None:
            try:
                self.on_event(kind, data)
            except Exception:
                pass

    def exception(self, where: str, exc: BaseException, frame=None) -> None:
        self.stats.exceptions += 1
        path = self.save_frame(frame, f"exception_{self.stats.exceptions:03d}") if frame is not None else None
        self.event("exception", where=where, error=repr(exc), traceback=traceback.format_exc(), frame=path)

    def save_frame(self, frame, name: str) -> str | None:
        """frame: numpy RGB/BGRA array or PIL image."""
        if frame is None or self.run_dir is None:
            return None
        try:
            from PIL import Image
            import numpy as np
            img = frame
            if isinstance(frame, np.ndarray):
                arr = frame[..., 2::-1] if frame.shape[-1] == 4 else frame  # BGRA -> RGB
                img = Image.fromarray(np.ascontiguousarray(arr))
            path = self.run_dir / "frames" / f"{name}.png"
            img.save(path)
            return str(path.relative_to(self.run_dir))
        except Exception as e:  # never let logging kill a run
            self.event("frame_save_failed", name=name, error=repr(e))
            return None

    def _write_json(self, name: str, obj) -> None:
        (self.run_dir / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

    def _append_jsonl(self, name: str, obj) -> None:
        if self.run_dir is None:
            return
        with open(self.run_dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")

    def _append_index(self, summary: dict) -> None:
        idx = self.root / "index.csv"
        new = not idx.exists()
        row = {k: summary.get(k, "") for k in INDEX_FIELDS}
        row["started"] = self.run_id
        row["kills"] = sum(summary.get("kills", {}).values())
        with open(idx, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
