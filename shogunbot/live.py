"""Live decision feed for the viewer (`python -m shogunbot watch`): zero tokens, local.

The bot writes runs/<id>/live.json whenever something changes: the board with every
unit's telegraph, all candidate moves with value and expected damage, the principal
variation (the likely line the search found), why the best move beat the second best
in plain words, the last 10 turns with predicted vs observed, the latest reward
decision and recent events. During a reward decision also the valuation as it runs
(0.23): the offer, the seeds played per option with running results, an example
fight, the choice, the rule and the click. The file is replaced atomically, and
nothing here may ever stop the bot: a failed write is skipped and the next one tries
again.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

LIVE_FILE = "live.json"
HISTORY = 10
EVENTS = 10
# events worth showing in the viewer, with a one-line text
_EVENT_TEXT = {
    "level_start": lambda d: f"level {d.get('level')} starts at {d.get('hp')} HP",
    "wave_spawn": lambda d: f"wave {d.get('wave')}" + (f"/{d['total']}" if d.get("total") else "") + ": " + (
        ", ".join(f"{u.get('kind')}@{u.get('x')}" for u in d.get("spawned", [])) or "no new unit read"),
    "item_use": lambda d: f"used {d.get('name')}: {d.get('reason')}",
    "item_result": lambda d: f"{d.get('item')} {'consumed' if d.get('consumed') else 'still in its slot'}",
    "potion_result": lambda d: f"{d.get('item')}: HP {d.get('hp_before')} -> {d.get('hp_after')}",
    "hp_flicker": lambda d: f"HP read {d.get('read')}, second look {d.get('second_read')} "
                            f"({d.get('second_pips')} pips): kept {d.get('kept')}",
    "stall_mode": lambda d: f"stall: {d.get('quiet_turns')} turns without progress, a 1-HP trade allowed",
    "loop_break": lambda d: f"loop breaker: not repeating {', '.join(d.get('banned', []))}",
    "reward_taken": lambda d: f"took {d.get('choice')}",
    "skip": lambda d: f"skipped the reward: {d.get('reason')}",
    "give_up": lambda d: f"giving the run up: {d.get('step')}" + (f" ({d.get('screen')})" if d.get("screen") else ""),
    "give_up_failed": lambda d: f"give-up failed: {d.get('reason')}",
    "menu": lambda d: f"menu: {d.get('screen')}" + (f" ({d.get('how')})" if d.get("how") else ""),
    "day_select": lambda d: f"camp: Day {d.get('day')} selected ({d.get('clicks')} click(s) on the day arrow)",
    "start_elsewhere": lambda d: f"run start on the {d.get('screen')} of an earlier run: giving it up first",
    "shop": lambda d: f"shop: statue {' / '.join(d.get('upgrade_text') or []) or '-'}"
                      + (f" for {d['upgrade_price']}" if d.get("upgrade_price") is not None else "")
                      + f", coins {d.get('coins')}; GO",
    "map_stuck": lambda d: f"map: {d.get('tries')} inputs did not leave the map",
    "map": lambda d: f"map: {d.get('step')}" + (f" ({d.get('node')})" if d.get("node") else ""),
    "map_travel": lambda d: f"map: travel to {d.get('node') or 'the marked node'} ({d.get('why')})",
    "mouse_mode": lambda d: f"{d.get('screen')}: keyboard-only mode off (a click on an empty spot)",
    "cursor": lambda d: f"click on the {d.get('screen')}: cursor at {d.get('cursor')}, target {d.get('target')}"
                        + ("" if d.get("ok") else " (missed)"),
    "level_end": lambda d: f"level {d.get('level')} ends at {d.get('hp_end')} HP ({d.get('outcome')})",
    "exception": lambda d: f"error in {d.get('where')}: {d.get('error')}",
    "run_end": lambda d: f"run ended: {d.get('result')}",
}


def _board(s) -> dict:
    h = s.hero
    tiles = []
    for t in h.kit.tiles:
        ch = h.charge.get(t.label, 0)
        tiles.append({"label": t.label, "tile": t.tile, "damage": t.damage, "cooldown": t.cooldown,
                      "charge": ch, "queued": t.label in h.queue,
                      "ready": t.label not in h.queue and ch >= t.cooldown})
    return {
        "cells": s.cells, "render": s.render(), "level": s.level, "wave": s.wave, "waves_total": s.waves_total,
        "hero": {"x": h.x, "facing": h.facing, "hp": h.hp, "queue": list(h.queue), "tiles": tiles,
                 "swap_ready": h.swap_ready, "shield": h.shield},
        "enemies": [{"x": e.x, "kind": e.kind, "hp": e.hp, "max_hp": e.max_hp, "facing": e.facing,
                     "intent": e.intent, "queue": list(e.queue), "frozen": e.frozen, "cursed": e.cursed,
                     "poison": e.poison} for e in s.alive()],
    }


def _decision(d, plan_ms: float, stalled: bool) -> dict:
    ranked = sorted(d.candidates, key=lambda c: -c["value"])
    top = ranked[0]["value"] if ranked else 0.0
    return {
        "action": d.action, "value": d.value, "depth": d.depth, "nodes": d.nodes,
        "plan_ms": round(plan_ms), "stalled": stalled,
        "candidates": [{"action": c["action"], "value": c["value"], "delta": round(c["value"] - top, 2),
                        "exp_damage": c["exp_damage"], "p_damage": c["p_damage"]} for c in ranked],
        "pv": d.pv, "reason": d.reason,
        "predicted": [{"p": p["p"], "board": p["board"], "hero_hp": p["hero_hp"]} for p in d.predicted],
    }


def _history_row(rec: dict) -> dict:
    preds = rec.get("predicted") or []
    obs = rec.get("observed")
    m = rec.get("matched_outcome")
    return {
        "turn": rec.get("turn"), "level": rec.get("level"), "board": rec.get("board"),
        "action": (rec.get("decision") or {}).get("action"),
        "predicted": preds[0]["board"] if preds else None,
        "predicted_hp": preds[0]["hero_hp"] if preds else None,
        "predicted_p": preds[0]["p"] if preds else None,
        "matched_p": rec.get("matched_p") if obs else None,
        "matched_rank": (m + 1) if (obs and m is not None) else None,
        "observed": obs["board"] if obs else None,
        "observed_hp": obs["hero_hp"] if obs else None,
        "match": (not rec.get("sim_diff")) if obs else None,
        "diff": (rec.get("sim_diff") or [])[:4],
        "hp_lost": rec.get("hp_lost", 0), "hp_lost_predicted": rec.get("hp_lost_predicted", 0),
    }


class LiveView:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.doc: dict = {}
        self.history: deque = deque(maxlen=HISTORY)
        self.events: deque = deque(maxlen=EVENTS)
        self.seq = 0
        self.errors = 0
        self._last_hist_turn = None
        self._t0 = time.time()
        self.val_s, self.val_writes = 0.0, 0

    # ---- lifecycle ------------------------------------------------------
    def start(self, run_dir, run_id: str, version: str) -> None:
        self.path = Path(run_dir) / LIVE_FILE
        self.history.clear()
        self.events.clear()
        self._last_hist_turn = None
        self._t0 = time.time()
        self.doc = {"run_id": run_id, "bot_version": version, "status": "starting", "status_info": {},
                    "turn": None, "level": None, "wave": None, "board": None, "decision": None,
                    "reward": None, "valuation": None, "result": None}
        self._write()

    def status(self, status: str, **info) -> None:
        if self.path is None or (status == self.doc.get("status") and info == self.doc.get("status_info")):
            return
        self.doc["status"], self.doc["status_info"] = status, info
        self._write()

    def turn(self, *, s, d, turn: int, plan_ms: float, stalled: bool = False, last_record: dict | None = None) -> None:
        if self.path is None:
            return
        try:
            if last_record is not None and last_record.get("turn") != self._last_hist_turn:
                self._last_hist_turn = last_record.get("turn")
                self.history.append(_history_row(last_record))
            self.doc.update(status="playing", status_info={}, turn=turn, level=s.level, wave=s.wave,
                            board=_board(s), decision=_decision(d, plan_ms, stalled),
                            turn_at=round(time.time() - self._t0, 1))   # the viewer leaves the valuation
            self._write()
        except Exception as e:      # the viewer must never stop the bot
            self._fail(e)

    def reward(self, info: dict) -> None:
        if self.path is None:
            return
        self.doc["reward"] = {**info, "at": round(time.time() - self._t0, 1)}
        self.doc["status"], self.doc["status_info"] = "reward", {"layout": info.get("layout")}
        self._write()

    def valuation(self, info: dict, reset: bool = False) -> None:
        """The reward valuation while it runs [user review of run 20261029-045856]: the
        offer, then rewards.evaluate's running state (at least every rewards.FEED_S
        seconds), the choice and rule, the click and its outcome. Each call merges into
        doc['valuation']; reset starts a new decision. The viewer shows it until combat
        resumes. val_s: time spent here, the feed's whole cost to the valuation."""
        if self.path is None:
            return
        t1 = time.perf_counter()
        try:
            v = self.doc.get("valuation")
            if reset or not isinstance(v, dict):
                v = {}
            v.update(info)
            v["at"] = round(time.time() - self._t0, 1)
            self.doc["valuation"] = v
            self.doc["status"] = "reward"
            self.doc["status_info"] = {"layout": v.get("layout"), "phase": v.get("phase")}
            self._write()
        except Exception as e:        # the viewer must never stop the bot
            self._fail(e)
        self.val_s += time.perf_counter() - t1
        self.val_writes += 1

    def event(self, kind: str, data: dict) -> None:
        if self.path is None or kind not in _EVENT_TEXT:
            return
        try:
            text = _EVENT_TEXT[kind](data)
        except Exception:
            text = kind
        self.events.append({"t": round(time.time() - self._t0, 1), "kind": kind, "text": text})
        if kind in ("give_up", "give_up_failed", "stall_mode", "item_use", "level_start", "wave_spawn",
                    "start_elsewhere", "shop", "map_stuck", "map_travel"):
            self._write()

    def end(self, result: str, summary: dict | None = None) -> None:
        if self.path is None:
            return
        self.doc["status"], self.doc["result"] = "ended", result
        if summary:
            self.doc["summary"] = {k: summary.get(k) for k in ("turns", "hp_end", "hp_lost", "hp_lost_unpredicted",
                                                               "minutes", "reached_level")}
        self._write()

    # ---- file -----------------------------------------------------------
    def _write(self) -> None:
        if self.path is None:
            return
        try:
            self.seq += 1
            self.doc["seq"] = self.seq
            self.doc["updated"] = time.time()
            self.doc["history"] = list(self.history)
            self.doc["events"] = list(self.events)
            tmp = self.path.with_name(LIVE_FILE + ".tmp")
            tmp.write_text(json.dumps(self.doc, default=str), encoding="utf-8")
            for _ in range(5):
                try:
                    os.replace(tmp, self.path)
                    return
                except PermissionError:   # Windows: the viewer is reading the file right now
                    time.sleep(0.02)
        except Exception as e:
            self._fail(e)

    def _fail(self, e: Exception) -> None:
        self.errors += 1
        if self.errors <= 3:
            print(f"   live view: {e!r} (the bot carries on)", flush=True)
