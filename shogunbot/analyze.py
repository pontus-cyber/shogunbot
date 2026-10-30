"""Compact text analysis of one bot run (standard library only).

    python -m shogunbot analyze            # latest run
    python -m shogunbot analyze RUN_ID

Writes runs/<id>/analysis.txt and prints it. Built so a review needs a few KB
of text instead of screenshots: frames are only opened for turns this report
cannot explain (it names those files).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _intents(rec: dict) -> str:
    es = rec.get("state", {}).get("enemies", [])
    return " ".join(f"{e['x']}:{(e.get('intent') or '-')[:4]}" for e in es)


def analyze(run_dir: Path) -> str:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").exists() else {}
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
    turns = _load(run_dir / "turns.jsonl")
    events = _load(run_dir / "events.jsonl")
    L = []
    L.append(f"RUN {run_dir.name}  bot {meta.get('bot_version', '?')}  result {summ.get('result', 'running?')}  "
             f"{summ.get('minutes', '?')} min  {len(turns)} turns")
    for lv in summ.get("levels", []):
        L.append(f"  {lv['level']:<9} HP {lv['hp_start']}->{lv['hp_end']}  turns {lv['turns']}  "
                 f"waves seen {lv.get('waves_seen')}/{lv.get('waves_total')}  {lv.get('outcome')}")

    # damage
    L.append("")
    L.append("HP LOST")
    lost = [t for t in turns if t.get("hp_lost")]
    if not lost:
        L.append("  none")
    for t in lost:
        pred = t.get("hp_lost_predicted", 0)
        tag = "predicted" if pred >= t["hp_lost"] else "UNPREDICTED"
        fr = (t.get("frames") or {}).get("before", "")
        L.append(f"  t{t['turn']:<3} {t['level']:<8} -{t['hp_lost']} {tag:<11} {t['board']:<28} intents[{_intents(t)}] "
                 f"-> {t['decision']['action']}")
        L.append(f"        observed {t['observed']['board']}  diff: {'; '.join(t.get('sim_diff', [])) or '-'}  {fr}")

    # model errors
    cat = Counter()
    for t in turns:
        for d in t.get("sim_diff", []):
            if "not compared" in d:
                continue
            k = d.split(":", 1)[0]
            cat["enemy." + k.split(".", 1)[1] if k.startswith("cell") and "." in k else ("enemy.presence" if k.startswith("cell") else k)] += 1
    n_cmp = sum(1 for t in turns if t.get("observed"))
    n_bad = sum(1 for t in turns if t.get("observed") and t.get("sim_diff"))
    L.append("")
    L.append(f"PREDICTION  {n_cmp - n_bad}/{n_cmp} turns matched fully;  mismatch fields: "
             + (", ".join(f"{k} {v}" for k, v in cat.most_common()) or "none"))

    # events
    ev = Counter(e["event"] for e in events)
    L.append("")
    L.append("EVENTS  " + ", ".join(f"{k} {v}" for k, v in ev.most_common()))
    for e in events:
        if e["event"] in ("exception",):
            tb = (e.get("traceback") or "").strip().splitlines()
            L.append(f"  EXCEPTION at {e.get('where')}: {e.get('error')}  last: {tb[-3].strip() if len(tb) >= 3 else ''}  {e.get('frame') or ''}")
        elif e["event"] in ("loop_break", "potion_use", "potion_result", "item_use", "item_result", "item_seen", "act_failed", "unknown_screen"):
            L.append("  " + json.dumps({k: v for k, v in e.items() if k != "ts"}))
    # rewards (M2): what was offered, how each option scored, what was taken
    for e in events:
        if e["event"] == "reward_screen":
            offers = ", ".join(f"{o.get('tile')} {o.get('damage')}/{o.get('cooldown')} L{o.get('levels')}"
                               for o in e.get("offers") or [])
            what = offers if e.get("layout") == "tile_choice" else (e.get("upgrade") or f"unread {e.get('upgrade_lines')}")
            L.append(f"  REWARD {e.get('level')} {e.get('layout')}: {what}  (frame {e.get('frame')})"
                     + (f"  notes {e['notes']}" if e.get("notes") else ""))
        elif e["event"] == "reward_decision":
            scored = [o for o in e.get("options", []) if o.get("scored", True)]
            opts = ", ".join(f"{o['name']} {o['value']:+.2f}"
                             + (f" ({o['diff']:+.2f}±{o['se']:.2f} vs skip{', net loss' if o.get('net_loss') else ''})"
                                if o.get("diff") is not None else "")
                             + (f" [{o['seeds']} seeds]" if o.get("seeds") else "")
                             for o in sorted(scored, key=lambda o: -o["value"]))
            fights = f", {e['fights']} fights" if e.get("fights") else ""     # 0.23: sequential valuation
            fights += f", {e['cached']} from the cache" if e.get("cached") else ""   # 0.29: fight cache
            L.append(f"    values ({e.get('seconds')}s{fights}, HP {e.get('hp')}): {opts or 'not simulated'} -> {e.get('choice')}"
                     + (f"  [{e['rule']}]" if e.get("rule") else ""))
        elif e["event"] == "shop":                                       # 0.24: read; 0.29: the statue bought from
            named = {g.get("x"): g.get("title") or "unread" for g in e.get("goods") or []}   # 0.32: the counter's tooltips
            L.append(f"  SHOP {e.get('n')}: statue {' / '.join(e.get('upgrade_text') or []) or '-'} ({e.get('upgrade')}) "
                     f"for {e.get('upgrade_price')}; coins {e.get('coins')}; counter "
                     + (", ".join((f"{named[o.get('x')]} " if o.get("x") in named else "") + f"{o.get('price')} {o.get('currency')}s"
                                  for o in e.get("counter") or []) or "-"))
        elif e["event"] == "shop_decision":
            scored = [o for o in e.get("options", []) if o.get("scored", True)]
            opts = ", ".join(f"{o['name']} {o['value']:+.2f}" + (f" [{o['seeds']} seeds]" if o.get("seeds") else "")
                             for o in sorted(scored, key=lambda o: -o["value"]))
            what = f"skill {e['skill']}, " if e.get("skill") else ""                    # 0.32: the counter
            # no option scored: nothing to simulate (a pure gain, or an offer never bought) [user review of
            # run 20261030-041652 read 'not simulated' as 'not modelled']
            L.append(f"    shop decision ({what}{e.get('seconds')}s, HP {e.get('hp')}): {opts or 'no simulation needed'} -> "
                     f"{'keep the coins' if e.get('choice') == 'skip' else e.get('choice')}  [{e.get('rule')}]")
        elif e["event"] in ("shop_bought", "shop_buy_failed"):
            L.append(f"    {e['event']}: {e.get('choice')} {e.get('reason', '')}"
                     + (f" coins {e.get('coins_before')} -> {e.get('coins_after')}" if e.get("coins_before") is not None else ""))
        elif e["event"] in ("window_moved", "window_move_failed", "window_back"):   # 0.29: checked every capture
            L.append("  " + json.dumps({k: v for k, v in e.items() if k != "ts"}))
        elif e["event"] in ("reward_taken", "reward_failed"):
            L.append(f"    {e['event']}: {e.get('choice')} {e.get('reason', '')}"
                     + (" (unverified)" if e.get("unverified") else ""))
        elif e["event"] == "reward_retry":
            L.append(f"    reward_retry {e.get('attempt')}: {e.get('choice')} not in the statue "
                     f"(statue {(e.get('statue') or {}).get('tile')}, UPGRADE lit {e.get('lit')})")
        elif e["event"] == "skip":
            L.append(f"    SKIP: {e.get('reason', '')}")
        elif e["event"] == "kit":
            L.append(f"  KIT at {e.get('level')}: " + ", ".join(
                f"{t['skill']} level {t['level']}" if "skill" in t else
                f"{t['label']} {t['damage']}/{t['cooldown']}" + (f"+{t['enchant']}" if t.get("enchant") else "")
                for t in e.get("kit", [])))
    notes = Counter()
    for e in events:
        if e["event"] in ("tracker_notes", "blink"):
            for n in e.get("notes", []):
                notes[" ".join(n.split()[:4]) if e["event"] == "tracker_notes" else "blink: " + n.split("->")[-1].strip()] += 1
    if notes:
        L.append("  notes: " + "; ".join(f"{k} x{v}" for k, v in notes.most_common(10)))

    # repeated identical boards (stuck loops)
    boards = Counter((t["board"], t["decision"]["action"]) for t in turns)
    loops = [(b, a, n) for (b, a), n in boards.items() if n >= 4]
    if loops:
        L.append("")
        L.append("REPEATS (same board and action 4+ times)")
        for b, a, n in sorted(loops, key=lambda x: -x[2])[:6]:
            L.append(f"  x{n} {b} -> {a}")

    # ending
    L.append("")
    L.append("LAST 8 TURNS")
    for t in turns[-8:]:
        o = (t.get("observed") or {}).get("board", "-")
        L.append(f"  t{t['turn']:<3} HP{t['hero']['hp']:<2} {t['board']:<28} [{_intents(t)}] -> {t['decision']['action']:<12} => {o}")
    return "\n".join(L)


def main(argv: list[str]) -> int:
    root = Path("runs")
    if argv:
        run_dir = root / argv[0] if not Path(argv[0]).exists() else Path(argv[0])
    else:
        run_dir = sorted(p for p in root.iterdir() if p.is_dir())[-1]
    text = analyze(run_dir)
    (run_dir / "analysis.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
