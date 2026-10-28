"""Cross-run report and single-run review.

    python -m shogunbot report            -> runs/report.html + console table
    python -m shogunbot review <run_id>   -> turn-by-turn text, loss review
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _turns(run_dir: Path) -> list[dict]:
    p = run_dir / "turns.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def summaries(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.iterdir()):
        s = d / "summary.json"
        if d.is_dir() and s.exists():
            out.append(_load(s))
    return out


def turning_point(turns: list[dict]) -> int | None:
    """Turn the run was lost: the last turn before death where HP was > 2x the
    damage the next three turns dealt (i.e. where a different action mattered)."""
    if not turns:
        return None
    hp = [t["hero"]["hp"] for t in turns]
    lost = [t.get("hp_lost", 0) or 0 for t in turns]
    for i in range(len(turns) - 1, -1, -1):
        ahead = sum(lost[i:i + 3])
        if ahead and hp[i] > ahead:
            return turns[i]["turn"]
    return turns[0]["turn"]


def review(root: Path, run_id: str, last: int | None = None) -> str:
    rd = root / run_id
    turns = _turns(rd)
    summ = _load(rd / "summary.json") if (rd / "summary.json").exists() else {}
    lines = [f"RUN {run_id}  result={summ.get('result')}  turns={len(turns)}  hp_lost={summ.get('hp_lost')}"]
    tp = turning_point(turns) if summ.get("result") == "death" else None
    if tp is not None:
        lines.append(f"TURNING POINT: turn {tp} (review alternatives there)")
    sel = turns[-last:] if last else turns
    for t in sel:
        c = t["decision"]["candidates"][:3]
        alts = "  ".join(f"{x['action']}={x['value']:.0f}/d{x['exp_damage']}" for x in c)
        mark = ""
        if t.get("hp_lost"):
            mark += f"  !!-{t['hp_lost']}HP"
        if t.get("sim_diff"):
            mark += f"  SIM:{'; '.join(t['sim_diff'][:3])}"
        if t.get("low_confidence"):
            mark += f"  LOWCONF:{list(t['low_confidence'])}"
        tp_mark = " <== TURNING POINT" if tp is not None and t["turn"] == tp else ""
        lines.append(
            f"{t['level']} w{t['wave']} t{t['turn']:>3} hp{t['hero']['hp']:>2} | {t['board']:<40} | "
            f"{t['decision']['action']:<12} [{alts}]{mark}{tp_mark}"
        )
        if t.get("observed"):
            lines.append(f"{'':>22}-> {t['observed']['board']}")
    return "\n".join(lines)


def report(root: Path) -> str:
    runs = summaries(root)
    if not runs:
        return "no finished runs yet"
    wins = sum(1 for r in runs if r["result"] == "win")
    mism = Counter()
    lost_by = Counter()
    for r in runs:
        mism.update(r.get("mismatch_by_field", {}))
        lost_by.update(r.get("hp_lost_by_enemy", {}))
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r.get(k, '')))}</td>" for k in (
            "run_id", "result", "reached_level", "waves_cleared", "turns", "hp_lost",
            "hp_lost_unpredicted", "double_kills", "sim_mismatch_turns", "low_conf_turns", "minutes")) + "</tr>"
        for r in runs)
    page = f"""<!doctype html><meta charset=utf-8><title>Shogunbot runs</title>
<style>body{{font:14px system-ui;margin:24px;background:#111;color:#eee}}table{{border-collapse:collapse}}
td,th{{border:1px solid #444;padding:4px 8px}}th{{background:#222}}</style>
<h1>Shogunbot: Stage 1</h1>
<p>Runs: {len(runs)} · Wins: {wins} ({wins / len(runs):.0%})</p>
<h2>HP lost by enemy (all runs)</h2><p>{html.escape(str(dict(lost_by.most_common())))}</p>
<h2>Simulator mismatches by field (all runs)</h2><p>{html.escape(str(dict(mism.most_common())))}</p>
<h2>Runs</h2><table><tr><th>run</th><th>result</th><th>level</th><th>waves</th><th>turns</th>
<th>HP lost</th><th>unpredicted</th><th>double kills</th><th>sim mismatch turns</th><th>low conf</th><th>min</th></tr>{rows}</table>"""
    (root / "report.html").write_text(page, encoding="utf-8")
    return (f"runs={len(runs)} wins={wins}\nHP lost by enemy: {dict(lost_by.most_common())}\n"
            f"mismatches: {dict(mism.most_common(8))}\nreport: {root / 'report.html'}")
