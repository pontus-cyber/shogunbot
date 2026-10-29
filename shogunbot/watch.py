"""Live decision viewer: a local page that follows the bot turn by turn. Zero tokens.

    python -m shogunbot watch                 # follows the newest run (a series too)
    python -m shogunbot watch --run 20261028-130456
    python -m shogunbot watch --port 8766 --no-browser

Serves http://127.0.0.1:8765 with the standard library only. The page polls
/live.json twice a second; the bot writes runs/<id>/live.json (shogunbot/live.py).
During a reward decision the 'Reward valuation' panel replaces the board until
combat resumes (0.23); the last decision stays in the 'Reward decision' box.
"""

from __future__ import annotations

import json
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .live import LIVE_FILE

WAITING = {"status": "waiting", "seq": 0, "note": "no live.json yet: start the bot (play.bat)"}


def find_live(root: Path, run: str | None = None) -> Path | None:
    """runs/<run>/live.json, or the most recently written live.json under root."""
    if run:
        p = root / run / LIVE_FILE
        return p if p.exists() else None
    best, best_t = None, -1.0
    for p in root.glob(f"*/{LIVE_FILE}"):
        try:
            t = p.stat().st_mtime
        except OSError:
            continue
        if t > best_t:
            best, best_t = p, t
    return best


class _Reader:
    """Reads live.json, keeping the last good copy (the bot may be replacing it)."""

    def __init__(self, root: Path, run: str | None):
        self.root, self.run, self.last = root, run, json.dumps(WAITING).encode()

    def read(self) -> bytes:
        p = find_live(self.root, self.run)
        if p is None:
            return json.dumps(WAITING).encode()
        for _ in range(3):
            try:
                data = p.read_bytes()
                json.loads(data)               # only hand out complete documents
                self.last = data
                return data
            except (OSError, ValueError):
                time.sleep(0.02)
        return self.last


def make_server(root: str | Path = "runs", run: str | None = None, port: int = 8765,
                host: str = "127.0.0.1") -> ThreadingHTTPServer:
    reader = _Reader(Path(root), run)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body, ctype = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            elif path == "/live.json":
                body, ctype = reader.read(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):   # quiet: the page polls twice a second
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(root: str | Path = "runs", run: str | None = None, port: int = 8765, open_browser: bool = True) -> None:
    srv = make_server(root, run, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print(f"live viewer on {url} (following {'run ' + run if run else 'the newest run'}); Ctrl+C to stop", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shogunbot live</title>
<style>
  :root {
    --bg: #121317; --panel: #1b1d23; --line: #2b2e37; --text: #e7e7ea; --dim: #9a9ca6;
    --red: #ef5a5f; --green: #4fb873; --amber: #f2a93b; --blue: #5aa2e8; --violet: #a98bf0;
    --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.45 system-ui, "Segoe UI", Roboto, sans-serif; }
  header { display: flex; flex-wrap: wrap; gap: 8px 18px; align-items: baseline;
           padding: 12px 18px; border-bottom: 1px solid var(--line); position: sticky; top: 0; background: var(--bg); }
  header h1 { font-size: 16px; margin: 0 8px 0 0; }
  .chip { padding: 2px 8px; border-radius: 10px; background: var(--line); font-size: 12px; }
  .chip.playing { background: #23402e; color: #a7e3bb; }
  .chip.ended { background: #3a2a2a; color: #f0b3b3; }
  .chip.reward, .chip.give_up { background: #3d3220; color: #f4d39a; }
  .stat { color: var(--dim); } .stat b { color: var(--text); font-weight: 600; }
  main { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 14px; padding: 14px 18px; }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
  section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin-bottom: 14px; }
  section h2 { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--dim); margin: 0 0 10px; }
  .board { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px; }
  .cell { min-width: 92px; flex: 1; border: 1px solid var(--line); border-radius: 6px; padding: 8px 6px;
          text-align: center; background: #16181d; }
  .cell .x { color: var(--dim); font-size: 11px; }
  .cell .who { font-weight: 600; margin: 2px 0; }
  .cell.hero { border-color: var(--blue); background: #172231; }
  .cell.enemy { border-color: #4a3434; }
  .facing { font-size: 18px; line-height: 1; }
  .hp { font-family: var(--mono); font-size: 12px; }
  .intent { display: inline-block; margin-top: 4px; padding: 1px 6px; border-radius: 8px; font-size: 11px; background: var(--line); }
  .intent.fire { background: var(--red); color: #fff; } .intent.queue { background: var(--amber); color: #201400; }
  .intent.move { background: var(--blue); color: #08131f; } .intent.turn { background: var(--violet); color: #12082a; }
  .q { font-family: var(--mono); font-size: 11px; color: var(--amber); margin-top: 3px; }
  .tiles { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
  .tile { font-family: var(--mono); font-size: 12px; padding: 3px 7px; border-radius: 5px; border: 1px solid var(--line); }
  .tile.ready { border-color: var(--green); color: #b9ecc9; } .tile.queued { border-color: var(--amber); color: var(--amber); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: var(--dim); font-weight: 500; font-size: 12px; padding: 4px 6px; border-bottom: 1px solid var(--line); }
  td { padding: 4px 6px; border-bottom: 1px solid #23262e; vertical-align: top; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono); }
  tr.chosen td { background: #1d2a3a; }
  .bar { height: 6px; background: var(--blue); border-radius: 3px; min-width: 2px; }
  .dmg { color: var(--red); } .ok { color: var(--green); } .bad { color: var(--red); } .dim { color: var(--dim); }
  .mono { font-family: var(--mono); font-size: 12px; white-space: pre; }
  ol.plan { margin: 0; padding-left: 20px; } ol.plan li { margin-bottom: 8px; }
  ul.reason { margin: 0; padding-left: 18px; } ul.reason li { margin-bottom: 4px; }
  .muted { color: var(--dim); font-size: 12px; }
  .stale { color: var(--amber); }
  .wide { grid-column: 1 / -1; }
  .seedbar { display: inline-block; vertical-align: middle; width: 70px; height: 6px; margin-left: 6px;
             background: var(--line); border-radius: 3px; overflow: hidden; }
  .seedbar i { display: block; height: 100%; background: var(--blue); }
  .seedbar i.done { background: var(--dim); }
  .tag { display: inline-block; padding: 0 6px; border-radius: 8px; font-size: 11px; background: var(--line); }
  .tag.lead { background: #23402e; color: #a7e3bb; } .tag.loss { background: var(--red); color: #fff; }
  .tag.stop { background: #2b2e37; color: var(--dim); }
  .valhead { font-size: 15px; } .valline { margin-top: 6px; }
  .rpctl { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin: 6px 0 8px; }
  .rpctl input[type=range] { flex: 1; min-width: 160px; }
  .rpctl button, .rpctl select { background: var(--line); color: var(--text); border: 1px solid #3a3e4a;
                                 border-radius: 5px; padding: 3px 10px; font: inherit; cursor: pointer; }
  .rpgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }
  .rpcol { border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; background: #16181d; }
  .rpboard { font-family: var(--mono); font-size: 17px; white-space: pre; margin: 6px 0 4px; overflow-x: auto; }
</style>
</head>
<body>
<header>
  <h1>Shogunbot live</h1>
  <span id="status" class="chip">connecting</span>
  <span class="stat">run <b id="run">–</b></span>
  <span class="stat">bot <b id="ver">–</b></span>
  <span class="stat">level <b id="level">–</b></span>
  <span class="stat">wave <b id="wave">–</b></span>
  <span class="stat">turn <b id="turn">–</b></span>
  <span class="stat">HP <b id="hp">–</b></span>
  <span class="stat" id="age"></span>
</header>
<main>
  <section id="valbox" class="wide" hidden><h2>Reward valuation</h2><div id="valuation"></div><div id="replay"></div></section>
  <div id="combat">
    <section><h2>Board</h2><div id="board" class="board"></div><div id="tiles" class="tiles"></div>
      <div id="boardnote" class="muted"></div></section>
    <section><h2>Options, best first</h2><div id="options"></div></section>
    <section><h2>Why this move</h2><ul id="reason" class="reason"></ul></section>
  </div>
  <div id="side">
    <section id="plansec"><h2>Plan: the likely line</h2><ol id="plan" class="plan"></ol></section>
    <section><h2>Last 10 turns: predicted vs actual</h2><div id="history"></div></section>
    <section id="rewardbox" hidden><h2>Reward decision</h2><div id="reward"></div></section>
    <section><h2>Events</h2><div id="events" class="muted"></div></section>
  </div>
</main>
<script>
const NAMES = {ashigaru: "Ashigaru", tachi: "Twin Tachi", charger: "Spike Charger", daisuke: "Daisuke"};
const INTENT = {fire: ["FIRE", "fire"], queue: ["LOAD", "queue"], "move-left": ["← STEP", "move"],
                "move-right": ["STEP →", "move"], turn: ["TURN", "turn"], hold: ["hold", ""]};
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const arrow = f => f > 0 ? "→" : "←";
let lastSeq = -1, lastUpdated = 0;

function board(b) {
  const el = document.getElementById("board"), tiles = document.getElementById("tiles");
  if (!b) { el.innerHTML = '<span class="muted">no board yet</span>'; tiles.innerHTML = ""; return; }
  const byX = {}; b.enemies.forEach(e => byX[e.x] = e);
  let html = "";
  for (let x = 0; x < b.cells; x++) {
    if (b.hero.x === x) {
      html += `<div class="cell hero"><div class="x">${x}</div><div class="who">YOU</div>
        <div class="facing">${arrow(b.hero.facing)}</div><div class="hp">${b.hero.hp} HP${b.hero.shield ? " · shield" : ""}</div>
        ${b.hero.queue.length ? `<div class="q">${esc(b.hero.queue.join(" › "))}</div>` : ""}
        <div class="muted">${b.hero.swap_ready ? "swap ready" : "swap recharging"}</div></div>`;
    } else if (byX[x]) {
      const e = byX[x], it = INTENT[e.intent] || ["?", ""];
      const pips = "●".repeat(Math.max(e.hp, 0)) + "○".repeat(Math.max((e.max_hp || e.hp) - e.hp, 0));
      const flags = [e.frozen ? "frozen " + e.frozen : "", e.cursed ? "cursed" : "", e.poison ? "poison " + e.poison : ""].filter(Boolean).join(" · ");
      html += `<div class="cell enemy"><div class="x">${x}</div><div class="who">${esc(NAMES[e.kind] || e.kind)}</div>
        <div class="facing">${arrow(e.facing)}</div><div class="hp">${pips}</div>
        <div><span class="intent ${it[1]}">${it[0]}</span></div>
        ${e.queue.length ? `<div class="q">loaded: ${esc(e.queue.join(", "))}</div>` : ""}
        ${flags ? `<div class="muted">${esc(flags)}</div>` : ""}</div>`;
    } else {
      html += `<div class="cell"><div class="x">${x}</div><div class="who dim">·</div></div>`;
    }
  }
  el.innerHTML = html;
  tiles.innerHTML = b.hero.tiles.map(t =>
    `<span class="tile ${t.queued ? "queued" : t.ready ? "ready" : ""}">${esc(t.label)} ${t.damage}/${t.cooldown}` +
    ` · ${t.queued ? "loaded" : t.ready ? "ready" : "charge " + t.charge + "/" + t.cooldown}</span>`).join("");
  document.getElementById("boardnote").textContent = "Telegraphs are what each enemy does after your next move.";
}

function options(d) {
  const el = document.getElementById("options");
  if (!d) { el.innerHTML = '<span class="muted">waiting for the first decision</span>'; return; }
  const vals = d.candidates.map(c => c.value), lo = Math.min(...vals), hi = Math.max(...vals);
  el.innerHTML = `<table><tr><th>#</th><th>Move</th><th class="num">Value</th><th class="num">vs best</th>
    <th class="num">Dmg now</th><th class="num">P(hit)</th><th style="width:22%"></th></tr>` +
    d.candidates.map((c, i) => `<tr class="${c.action === d.action ? "chosen" : ""}"><td>${i + 1}</td>
      <td>${esc(c.action)}${c.action === d.action ? " ✓" : ""}</td><td class="num">${c.value.toFixed(0)}</td>
      <td class="num">${c.delta ? c.delta.toFixed(0) : "–"}</td>
      <td class="num ${c.exp_damage > 0 ? "dmg" : ""}">${c.exp_damage.toFixed(1)}</td>
      <td class="num">${Math.round(c.p_damage * 100)}%</td>
      <td><div class="bar" style="width:${hi > lo ? Math.max(2, 100 * (c.value - lo) / (hi - lo)) : 100}%"></div></td></tr>`).join("") +
    `</table><div class="muted" style="margin-top:6px">search depth ${d.depth} · ${d.nodes} nodes · ${d.plan_ms} ms` +
    `${d.stalled ? ' · <span class="stale">stall rule on</span>' : ""}</div>`;
}

function plan(d) {
  const el = document.getElementById("plan");
  if (!d || !d.pv || !d.pv.length) { el.innerHTML = '<li class="muted">no line yet</li>'; return; }
  el.innerHTML = d.pv.map((p, i) => {
    const does = p.enemies.map(e => `${NAMES[e.kind] || e.kind}@${e.x} ${e.does}`).join(", ") || "none left";
    const extra = (p.kills.length ? ` · <span class="ok">kills ${esc(p.kills.join(", "))}</span>` : "") +
                  (p.hero_damage ? ` · <span class="dmg">you lose ${p.hero_damage} HP</span>` : "");
    return `<li><b>${i === 0 ? "now" : "turn +" + i}:</b> you <b>${esc(p.action)}</b>
      <div class="muted">enemies ${i === 0 ? "(their telegraphs)" : "(predicted)"}: ${esc(does)}${extra} · line p ${(p.p * 100).toFixed(0)}%</div>
      <div class="mono">${esc(p.board)}   HP ${p.hero_hp}</div></li>`;
  }).join("");
}

function reason(d) {
  const el = document.getElementById("reason");
  el.innerHTML = (d && d.reason && d.reason.length ? d.reason : ["no decision yet"]).map((r, i) =>
    `<li${i === 0 ? ' style="list-style:none;margin-left:-18px"' : ""}>${esc(r)}</li>`).join("");
}

function history(h) {
  const el = document.getElementById("history");
  if (!h || !h.length) { el.innerHTML = '<span class="muted">no finished turns yet</span>'; return; }
  el.innerHTML = `<table><tr><th>Turn</th><th>Move</th><th></th><th>Predicted</th><th>Actual</th></tr>` +
    h.slice().reverse().map(r => {
      const mark = r.match === null ? '<span class="dim">–</span>' : r.match
        ? '<span class="ok">✓</span>' + (r.matched_rank > 1 ? ` <span class="muted">#${r.matched_rank}</span>` : "")
        : '<span class="bad">✗</span>';
      const lost = r.hp_lost ? ` · <span class="dmg">lost ${r.hp_lost}${r.hp_lost > r.hp_lost_predicted ? " (unpredicted " + (r.hp_lost - r.hp_lost_predicted) + ")" : ""}</span>` : "";
      const diff = r.match === false && r.diff && r.diff.length
        ? `<div class="muted" style="white-space:normal">${esc(r.diff.slice(0, 3).join("; "))}</div>` : "";
      return `<tr><td>${r.turn}</td><td>${esc(r.action)}</td><td>${mark}</td>
        <td class="mono">${esc(r.predicted ?? "–")}<div class="muted">HP ${r.predicted_hp ?? "–"} · p ${r.predicted_p != null ? Math.round(r.predicted_p * 100) + "%" : "–"}</div></td>
        <td class="mono">${esc(r.observed ?? "not compared")}<div class="muted">HP ${r.observed_hp ?? "–"}${lost}</div>${diff}</td></tr>`;
    }).join("") +
    `</table><div class="muted" style="margin-top:6px">✓ one of the predicted outcomes happened (#n: not the likeliest) · ✗ none did, the differences are listed · – not compared (level or wave changed)</div>`;
}

function reward(r) {
  const box = document.getElementById("rewardbox"), el = document.getElementById("reward");
  if (!r) { box.hidden = true; return; }
  box.hidden = false;
  const opts = (r.options || []).slice().sort((a, b) => (b.value ?? -1e9) - (a.value ?? -1e9));
  el.innerHTML = `<div>after <b>${esc(r.level)}</b> · ${esc(r.layout)}${r.upgrade ? " · " + esc(r.upgrade) : ""} · HP ${r.hp ?? "–"}${r.seconds ? " · " + r.seconds + " s" : ""}${r.fights ? " · " + r.fights + " fights" : ""}</div>` +
    (opts.length ? `<table style="margin-top:6px"><tr><th>Option</th><th class="num">Seeds</th><th class="num">Score</th><th class="num">vs skip</th><th class="num">Wins</th><th></th></tr>` +
      opts.map(o => `<tr class="${o.name === r.choice ? "chosen" : ""}"><td>${esc(o.name)}${o.name === r.choice ? " ✓" : ""}</td>
        <td class="num">${o.seeds ?? "–"}</td>
        <td class="num">${o.scored ? o.value.toFixed(2) : "–"}</td>
        <td class="num">${o.diff != null ? (o.diff >= 0 ? "+" : "") + o.diff.toFixed(2) + " ± " + (o.se ?? 0).toFixed(2) : "–"}</td>
        <td class="num">${o.scored ? o.wins : "–"}</td><td>${o.net_loss ? '<span class="bad">net loss</span>' : o.stopped ? '<span class="muted">stopped</span>' : ""}</td></tr>`).join("") + `</table>` : "") +
    `<div style="margin-top:6px">choice <b>${esc(r.choice)}</b> · rule: ${esc(r.rule)}</div>`;
}

const PHASE = {starting: "starting the fights", running: "playing the fights", valued: "valued",
               decided: "decided", done: "done"};
const pm = (x, s) => x == null ? "–" : (x >= 0 ? "+" : "") + x.toFixed(2) + (s != null ? " ± " + s.toFixed(2) : "");

function fight(f, sc) {
  if (!f) return '<span class="dim">–</span>';
  const how = f.win ? `<span class="ok">won</span>, ${f.hp} HP` : f.timeout ? `<span class="stale">stalled</span>, ${f.hp} HP`
    : `<span class="bad">died</span> ${f.died === "stress-boss" ? "against " + esc((sc && sc.boss) || "the boss") : "on the board"}`;
  return `${how}, ${f.turns} turns`;
}

function valuation(v, show) {
  const box = document.getElementById("valbox"), el = document.getElementById("valuation");
  if (!v || !show) { box.hidden = true; return; }
  box.hidden = false;
  const tc = v.layout === "tile_choice";
  const what = tc ? (v.offers || []).map(o => `${esc(o.tile ?? "unread")} ${o.damage ?? "?"}/${o.cooldown ?? "?"}${o.levels ? " · " + o.levels + " slots" : ""}`).join("  |  ")
                  : esc(v.upgrade_text || v.upgrade || "upgrade not read");
  const sc = v.scenario || {};
  const rows = v.options || [];
  const stats = rows.some(o => o.seeds != null);
  let html = `<div class="valhead"><b>${tc ? "Tile choice" : v.layout === "shrine" ? "Shrine upgrade" : esc(v.layout)}</b>: ${what}</div>
    <div class="muted">after ${esc(v.level)} · HP ${v.hp ?? "–"} · kit ${esc((v.kit || []).join(", "))} · options: ${esc(rows.map(o => o.name).join(", ") || "none")}</div>`;
  if (stats || v.phase === "starting") {
    html += `<div class="valline">${PHASE[v.phase] || esc(v.phase)} · <b>${(v.elapsed ?? 0).toFixed(1)} s</b> · ${v.workers ?? "?"} worker${v.workers === 1 ? "" : "s"} · ${v.fights ?? 0} fights played` +
      `${v.fights_max ? " (at most " + v.fights_max + ")" : ""}${v.looked ? " · compared after " + v.looked + " seeds" : ""}</div>`;
  }
  if (stats) {
    html += `<table style="margin-top:6px"><tr><th>Option</th><th>Seeds</th><th class="num">Mean ± SE</th><th class="num">vs skip ± SE</th><th class="num">Wins</th><th>Status</th><th>Example fight${v.example_seed ? " (seed " + v.example_seed + ")" : ""}</th></tr>` +
      rows.map(o => {
        const chosen = v.choice && o.name === v.choice;
        const tags = (o.name === v.leader && !v.choice ? '<span class="tag lead">leader</span> ' : "") +
          (o.net_loss ? '<span class="tag loss">net loss</span> ' : "") +
          (o.stopped ? `<span class="tag stop">stopped</span> <span class="muted">${esc(o.stopped)}</span>` : o.playing && !v.choice ? '<span class="muted">playing</span>' : "");
        return `<tr class="${chosen ? "chosen" : ""}"><td>${esc(o.name)}${chosen ? " ✓" : ""}</td>
          <td class="mono">${o.seeds}/${o.of}<span class="seedbar"><i class="${o.playing ? "" : "done"}" style="width:${o.of ? 100 * o.seeds / o.of : 0}%"></i></span></td>
          <td class="num">${o.mean == null ? "–" : o.mean.toFixed(2) + (o.se != null ? " ± " + o.se.toFixed(2) : "")}</td>
          <td class="num">${o.name === "skip" ? "" : pm(o.diff, o.diff_se)}</td>
          <td class="num">${o.seeds ? o.wins + "/" + o.seeds : "–"}</td>
          <td style="white-space:normal">${tags}</td><td>${fight(o.example, sc)}</td></tr>`;
      }).join("") + `</table>`;
  }
  if (sc.name) {
    html += `<div class="valline muted">Scenario: ${esc(sc.name)}${sc.board ? " · " + esc(sc.board) : ""}${sc.waves ? " · " + sc.waves + " waves at " + esc(sc.enemy_hp) + " enemy HP" : ""}` +
      `${sc.boss ? " · then " + esc(sc.boss) + " at " + sc.boss_hp + " HP" : ""} · planner depth ${sc.depth}${sc.score ? " · score: " + esc(sc.score) : ""}</div>`;
    if (v.example_waves) html += `<div class="muted">Example seed ${v.example_seed} (where the options differ most so far): waves ${esc(v.example_waves.join(", "))} (A Ashigaru, T Twin Tachi, C Spike Charger)</div>`;
    if (v.stop_z) html += `<div class="muted">Stops an option once it is ${v.stop_z} SE behind the leader or below skipping, and skipping once every option left is ${v.stop_z} SE above it; compared after ${esc((v.looks || []).join(", "))} seeds. Net loss: more than 2 SE below skipping.</div>`;
  }
  if (v.choice) {
    html += `<div class="valline">choice <b>${esc(v.choice)}</b> · rule: ${esc(v.rule)}</div>` +
      `<div class="valline">click: <b>${esc(v.click || "–")}</b>${v.outcome ? " · " + esc(v.outcome) : v.phase === "decided" ? " · clicking now" : ""}${v.seconds != null ? " · decided in " + v.seconds + " s" : ""}</div>`;
  }
  el.innerHTML = html;
}

// replay of the example seed's fight for every option, turn by turn [user 2026-10-29]. The player
// keeps its place across the twice-a-second updates and waits for the end of a playback before it
// moves to a newer example seed.
const rp = {key: null, seed: null, cols: [], sc: {}, i: 0, n: 0, playing: true, ended: false, speed: 1, timer: null};

function rpCols(v) {
  return (v.options || []).filter(o => o.example && o.example.trace && o.example.seed === v.example_seed)
    .map(o => ({name: o.name, ex: o.example, tr: o.example.trace}));
}

function replay(v, show) {
  const el = document.getElementById("replay");
  if (!show || !v) { el.innerHTML = ""; rp.key = rp.seed = null; rp.playing = true; rp.ended = false; rpSchedule(); return; }
  const cols = rpCols(v);
  if (!cols.length) return;
  const key = v.example_seed + "|" + cols.map(c => c.name + ":" + c.tr.length).join(",");
  if (key === rp.key) return;
  const newSeed = v.example_seed !== rp.seed;
  if (newSeed && rp.seed !== null && !rp.ended) return;   // the fight being watched first (playing or paused)
  if (newSeed) { rp.i = 0; rp.playing = true; rp.ended = false; }
  rp.key = key; rp.seed = v.example_seed; rp.cols = cols; rp.sc = v.scenario || {};
  rp.n = Math.max(...cols.map(c => c.tr.length));
  el.innerHTML = `<div class="valline"><b>Replay, seed ${v.example_seed}</b> <span class="muted">the same fight for every option, turn by turn ·
      H you, A Ashigaru, T Twin Tachi, C Spike Charger, D Daisuke; the digit is HP, * loaded, &lt; &gt; facing</span></div>
    <div class="rpctl"><button id="rpplay"></button><input id="rpslider" type="range" min="0" max="${rp.n - 1}">
      <span id="rpturn" class="mono"></span>
      <select id="rpspeed"><option value="1">4 turns/s</option><option value="2">8 turns/s</option><option value="4">16 turns/s</option></select></div>
    <div class="rpgrid">${cols.map((c, k) => `<div class="rpcol"><div><b>${esc(c.name)}</b> · ${fight(c.ex, rp.sc)}</div><div id="rpcell${k}"></div></div>`).join("")}</div>`;
  document.getElementById("rpplay").onclick = () => {
    rp.playing = !rp.playing;
    if (rp.playing && rp.i >= rp.n - 1) rp.i = 0;
    rp.ended = false;
    rpSchedule(); rpDraw();
  };
  document.getElementById("rpslider").oninput = e => { rp.i = +e.target.value; rp.ended = false; rpDraw(); };
  const sp = document.getElementById("rpspeed");
  sp.value = String(rp.speed);
  sp.onchange = e => { rp.speed = +e.target.value; rpSchedule(); };
  rpSchedule(); rpDraw();
}

function rpSchedule() {
  if (rp.timer) { clearInterval(rp.timer); rp.timer = null; }
  if (rp.playing && rp.key) rp.timer = setInterval(() => {
    if (rp.i >= rp.n - 1) { rp.playing = false; rp.ended = true; rpSchedule(); rpDraw(); return; }  // a newer example may follow
    rp.i++; rpDraw();
  }, 250 / rp.speed);
}

function rpDraw() {
  const btn = document.getElementById("rpplay");
  if (!btn) return;
  btn.textContent = rp.playing ? "❚❚ pause" : "▶ play";
  document.getElementById("rpslider").value = rp.i;
  document.getElementById("rpturn").textContent = `turn ${rp.i + 1} / ${rp.n}`;
  rp.cols.forEach((c, k) => {
    const j = Math.min(rp.i, c.tr.length - 1), r = c.tr[j], prev = j > 0 ? c.tr[j - 1] : r;
    const [lv, wave, board, hp, act, tele, queue] = r;
    const where = lv === "boss" ? (rp.sc.boss || "boss") : "stress board" + (wave ? ", wave " + wave : "");
    const lost = hp < prev[3] ? ` <span class="dmg">−${prev[3] - hp}</span>` : "";
    const over = rp.i > c.tr.length - 1 ? ' · <span class="muted">fight over</span>' : "";
    document.getElementById("rpcell" + k).innerHTML =
      `<div class="muted">${esc(where)} · HP <b>${hp}</b>${lost}${over}</div>
       <div class="rpboard">${esc(board)}</div>
       <div class="muted">${["cleared", "died", "stalled"].includes(act) ? "level " + esc(act) : "move: <b>" + esc(act) + "</b>"}` +
      `${tele ? " · telegraphs " + esc(tele) : ""}${queue ? " · queue " + esc(queue) : ""}</div>`;
  });
}

function render(doc) {
  const st = document.getElementById("status");
  st.textContent = doc.status + (doc.status_info && Object.keys(doc.status_info).length ? " · " + Object.values(doc.status_info).join(" ") : "") + (doc.result ? " · " + doc.result : "");
  st.className = "chip " + (doc.status || "");
  document.getElementById("run").textContent = doc.run_id || "–";
  document.getElementById("ver").textContent = doc.bot_version || "–";
  document.getElementById("level").textContent = doc.level || "–";
  const wt = doc.board ? doc.board.waves_total : 0;
  document.getElementById("wave").textContent = doc.board ? (wt && wt < 20 ? `${doc.board.wave}/${wt}` : (doc.board.wave || "\u2013")) : "\u2013";
  document.getElementById("turn").textContent = doc.turn ?? "–";
  document.getElementById("hp").textContent = doc.board ? doc.board.hero.hp : "–";
  // during a reward decision the valuation replaces the board, until combat resumes
  const v = doc.valuation, showVal = !!(v && (v.at ?? 0) >= (doc.turn_at ?? -1));
  valuation(v, showVal);
  replay(v, showVal);
  document.getElementById("combat").hidden = showVal;
  document.getElementById("plansec").hidden = showVal;
  document.getElementById("side").classList.toggle("wide", showVal);
  board(doc.board); options(doc.decision); plan(doc.decision); reason(doc.decision);
  history(doc.history); reward(doc.reward);
  document.getElementById("events").innerHTML = (doc.events || []).slice().reverse()
    .map(e => `<div>${e.t.toFixed(0)} s · ${esc(e.text)}</div>`).join("") || "none yet";
}

async function poll() {
  try {
    const r = await fetch("/live.json", {cache: "no-store"});
    const doc = await r.json();
    lastUpdated = doc.updated || 0;
    if (doc.seq !== lastSeq || doc.status === "waiting") { lastSeq = doc.seq; render(doc); }
  } catch (e) {
    document.getElementById("status").textContent = "viewer offline";
  }
  const age = lastUpdated ? Date.now() / 1000 - lastUpdated : null;
  const el = document.getElementById("age");
  el.textContent = age == null ? "" : `updated ${age < 60 ? age.toFixed(0) + " s" : (age / 60).toFixed(0) + " min"} ago`;
  el.className = "stat" + (age != null && age > 15 ? " stale" : "");
}
poll(); setInterval(poll, 500);
</script>
</body>
</html>
"""
