"""Reward choices: take every free reward, pick the target by simulation.

Rules [user, after run 20261028-082756, where two shrine upgrades were
skipped on noise]:
  1. A pure-gain upgrade (+damage, -cooldown, +upgrade slot, nothing worse)
     is always taken. The simulation only picks which tile gets it.
  2. A trade-off (e.g. +2 damage +3 cooldown, an enchantment that costs
     cooldown) and a tile choice are scored against skipping on the same
     seeded fights, in a scenario harder than the rest of Stage 1 so the
     options separate.
  3. Skip only when every option is a net loss: worse than skipping by more
     than the noise (paired difference + 2 standard errors below zero).
Never offered for a decision: Sacrifice a Tile (gives coins, loses a tile)
and upgrades whose effect is unknown.

Scoring: each option plays the 'stress' scenario (selfplay.py) on the same
48 seeds with common random numbers, so a seed is the same fight for every
option. A run scores the HP left minus 0.02 per turn (faster wins count
when HP ends equal), -10 for a death, HP - 3 for a stalled run.

Sequential valuation (0.23) [user review of run 20261029-045856: a decision
took 45-73 s, about 2.5 of the 8 minutes of a Stage 1 run]. The options play
the seeds in order, every option the same seed. At LOOKS (after 24, 32 and 40
seeds) an option stops once it is settled: more than STOP_Z (3)
standard errors behind the leader, so it cannot be the best, or that far
below skipping, a net loss. Skipping stops once every option left is that far
above it. The decision ends when one option is left and settled against
skipping, or after the 48th seed, where the rules above decide on the seeds
played. An option that stopped behind the leader is not chosen. Pure gains
use the same test: a cap at 16 seeds changed 1 of the 46 logged choices (the
boss shrine of run 20261028-155955: at 16 seeds Arrow, Swirl and Spear were
tied; at 48 Spear won, Swirl and Arrow each died once), and a faster decision
is not worth a worse one [user 2026-10-29]. The worker processes start once
and stay for the bot's lifetime (pool()); each decision used to start seven
new ones.

Fight cache (0.29) [user review of run 20261029-154200: reward decisions took
4.6 of 16 minutes]. A simulated fight is fixed by the kit (each tile's label,
tile, damage, cooldown and enchantment, in kit order), the HP, the seed, the
depth, the scenario and the simulator itself, so a fight already played is not
played again: a shrine's 'skip' kit is the kit the last decision chose, at the
same HP, and every run's first tile choice starts from the same kit. Of the
1,008 fights of that run, 144 repeated a fight of the same run and 144 more one
of an earlier run (with the same simulator). Results are kept for the bot's
lifetime and on disk (CACHE_FILE, runs/fight_cache.jsonl on the PC) under a
fingerprint of the simulator's code and data, so a new version starts afresh;
the decisions are the same as without the cache.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .catalogue import CAT
from .model import HERO_MAX_HP, Kit, KitTile
from .selfplay import STRESS_WAVES, LEVELS, enemy_hp, plan_levels, run_once

SEEDS = tuple(range(9000, 9048))  # fixed: the same 48 simulated fights score every option, in this order
DEPTH = 3                         # planner depth inside the valuation (live play uses 4-6). Depth 2
                                  # undervalued the dash tiles against depth 3 [valuation study 2026-10-28]
SCENARIO = "stress"               # harder than the rest of Stage 1, where every option won at ~full HP
WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
TURN_COST = 0.02                  # per simulated turn
STALL_PENALTY = 3.0               # a simulated run that hit the turn limit alive: HP kept, minus this
NOISE_Z = 2.0                     # a net loss is worse than skipping by more than 2 standard errors
LOOKS = (24, 32, 40)              # seeds after which the options are compared (and after the last seed).
                                  # From 8 seeds, 3 of 950 random seed orders changed a logged choice;
                                  # from 24, 1 of 3,000 [0.23 re-decision of the logged decisions]
STOP_Z = 3.0                      # an option stops once its gap to the leader or to skip exceeds this many SE
SD_FLOOR = 0.5                    # HP: the spread of one seed's difference is taken as at least this, so a
                                  # few seeds that happen to agree exactly cannot stop an option
FEED_S = 0.5                      # while fights run, the progress feed (live viewer) at least this often


@dataclass
class Option:
    name: str                   # 'skip', 'take:<tile>', 'upgrade:<label>'
    kit: Kit
    value: float = 0.0          # mean score
    wins: int = 0
    scores: list = field(default_factory=list, repr=False)
    diff: float | None = None   # mean paired difference to skipping
    se: float | None = None     # its standard error
    detail: dict = field(default_factory=dict)
    fights: list = field(default_factory=list, repr=False)   # per seed played: _job's result
    stopped: str | None = None  # why it stopped before the last seed
    out: bool = False           # stopped as not the best option: never chosen
    above: bool = False         # settled above skipping

    @property
    def net_loss(self) -> bool:
        return self.diff is not None and self.diff + NOISE_Z * (self.se or 0.0) < 0

    def summary(self) -> dict:
        d = {"name": self.name, "scored": bool(self.scores), "value": self.value, "wins": self.wins}
        if self.scores:
            d["seeds"] = len(self.scores)
        if self.stopped:
            d["stopped"] = self.stopped
        if self.diff is not None:
            d.update(diff=self.diff, se=self.se, net_loss=self.net_loss)
        return d


@dataclass
class Decision:
    best: Option
    options: list
    rule: str
    seconds: float = 0.0
    stats: dict = field(default_factory=dict)   # the valuation: fights, seeds, feed time


def score(res) -> float:
    if res.win:
        return float(res.hp) - TURN_COST * res.turns
    if res.timeout:
        return float(res.hp) - STALL_PENALTY - TURN_COST * res.turns
    return -float(HERO_MAX_HP)


# ---------------------------------------------------------------- upgrades
def pure_gain(up) -> bool:
    """Nothing gets worse: no damage lost, no cooldown added, no tile removed."""
    return (up.kind == "stat" and not up.removes_tile and not up.unknown_effect
            and up.damage >= 0 and up.cooldown <= 0 and up.slots >= 0
            and (up.damage > 0 or up.cooldown < 0 or up.slots > 0))


def offerable(up) -> bool:
    """Upgrades the bot ever takes: not Sacrifice (coins for a tile), not unknown effects."""
    return not up.removes_tile and not up.unknown_effect


def upgraded(kt: KitTile, up) -> KitTile | None:
    """The tile after a shrine upgrade, or None when it cannot take it or it
    would change nothing (e.g. -1 cooldown on a cooldown-0 tile)."""
    if not offerable(up):
        return None
    if up.uses_level and kt.used >= kt.levels:
        return None
    if up.kind == "enchant":
        if kt.enchant is not None:
            return None   # one enchantment per tile [wiki]
        return KitTile(kt.label, kt.tile, kt.damage, max(0, kt.cooldown + up.cooldown), up.id,
                       kt.levels, kt.used + 1)
    new = KitTile(kt.label, kt.tile, max(0, kt.damage + up.damage), max(0, kt.cooldown + up.cooldown),
                  kt.enchant, kt.levels + up.slots, kt.used + (1 if up.uses_level else 0))
    if (new.damage, new.cooldown, new.levels) == (kt.damage, kt.cooldown, kt.levels):
        return None
    return new


def tile_choice_options(kit: Kit, offers: list[dict]) -> list[Option]:
    opts = [Option("skip", kit)]
    for i, o in enumerate(offers):
        t = o.get("tile")
        if t is None or t not in CAT.tiles or not CAT.tile(t).sim:
            continue      # unknown or not simulated: never taken blind
        cat = CAT.tile(t)
        dmg = o["damage"] if o.get("damage") is not None else cat.damage
        cd = o["cooldown"] if o.get("cooldown") is not None else cat.cooldown
        new = kit.add(t, damage=dmg, cooldown=cd, levels=o.get("levels") or 3, used=o.get("used") or 0)
        opts.append(Option(f"take:{t}", new, detail={"offer": i}))
    return opts


def upgrade_options(kit: Kit, upgrade_id: str) -> list[Option]:
    opts = [Option("skip", kit)]
    up = CAT.upgrade(upgrade_id)
    for kt in kit:
        new = upgraded(kt, up)
        if new is not None:
            opts.append(Option(f"upgrade:{kt.label}", kit.replace(kt.label, new), detail={"label": kt.label}))
    return opts


# ---------------------------------------------------------------- worker processes
_POOL: ProcessPoolExecutor | None = None
_POOL_WORKERS = 0


def _warm() -> int:
    """Runs in a new worker: importing this module loads the simulator once."""
    return os.getpid()


def pool(workers: int = WORKERS) -> ProcessPoolExecutor | None:
    """The valuation's worker processes, started on first use and kept for the bot's
    lifetime (a series too). spawn: the same on Windows (the PC) and in tests."""
    global _POOL, _POOL_WORKERS
    if workers <= 1:
        return None
    if _POOL is None or _POOL_WORKERS != workers:
        close_pool()
        try:
            _POOL, _POOL_WORKERS = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")), workers
        except Exception:
            _POOL = None
    return _POOL


def warm_pool(workers: int = WORKERS) -> None:
    """Start the workers now, so the first reward decision does not wait for them (play())."""
    p = pool(workers)
    if p is not None:
        try:
            for _ in range(workers):
                p.submit(_warm)
        except Exception:
            close_pool()


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        try:
            _POOL.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _POOL = None


atexit.register(close_pool)


# ---------------------------------------------------------------- fight cache
CACHE_FILE: Path | None = None    # set by the bot (runs/fight_cache.jsonl); None: memory only
_CACHE: dict = {}                 # key -> (score, win, hp, turns, stalled, level reached, [])
_CACHE_LOADED: Path | None = None
_FP: str | None = None


def fingerprint() -> str:
    """The simulator's code and data: every module of the package's top level and every data file."""
    global _FP
    if _FP is None:
        root = Path(__file__).parent
        h = hashlib.sha1()
        for f in sorted(root.glob("*.py")) + sorted((root / "data").glob("*.json")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
        _FP = h.hexdigest()[:16]
    return _FP


def fight_key(kit_list: list, hp: int, seed: int, depth: int, scenario: str) -> str:
    """What fixes a simulated fight: the tiles and the skills (Kit.to_list puts one {'skill', 'level'} dict per
    skill after the tiles; 0.31-0.32 read every entry as a tile and failed on a build with a skill). A tile's
    levels and used levels change no fight."""
    tiles = [[t["label"], t["tile"], t["damage"], t["cooldown"], t.get("enchant")] for t in kit_list if "skill" not in t]
    skills = [[t["skill"], t["level"]] for t in kit_list if "skill" in t]
    return json.dumps([fingerprint(), tiles, hp, seed, depth, scenario] + ([skills] if skills else []),
                      separators=(",", ":"))


def _cache_load() -> None:
    """Read CACHE_FILE's fights of this fingerprint once; lines of other versions are dropped."""
    global _CACHE_LOADED
    if CACHE_FILE is None or _CACHE_LOADED == CACHE_FILE:
        return
    _CACHE_LOADED = CACHE_FILE
    fp, keep = fingerprint(), []
    try:
        if Path(CACHE_FILE).exists():
            for line in Path(CACHE_FILE).read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if json.loads(d["k"])[0] == fp:
                    _CACHE[d["k"]] = tuple(d["f"]) + ([],)
                    keep.append(line)
            Path(CACHE_FILE).write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    except Exception:
        pass                              # the cache must never stop a decision


def _cache_store(new: dict) -> None:
    for k, f in new.items():
        _CACHE[k] = tuple(f[:6]) + ([],)  # without the viewer's replay of the fight
    if CACHE_FILE is None or not new:
        return
    try:
        with open(CACHE_FILE, "a", encoding="utf-8") as fh:
            for k, f in new.items():
                fh.write(json.dumps({"k": k, "f": list(f[:6])}, separators=(",", ":")) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- simulation
def _job(args):
    """One simulated fight: (score, win, hp left, turns, stalled, level reached, its turns for the
    viewer's replay)."""
    kit_list, hp, seed, depth, scenario = args
    trace: list = []
    kit = Kit.from_list(kit_list)
    # a build with less max HP (Iron Skin, Fortress) starts at most at its max [wiki: Shop & Skills]
    r = run_once(seed, depth, kit=kit, hp=min(hp, kit.max_hp), scenario=scenario, start="bamboo-2", trace=trace)
    return score(r), r.win, r.hp, r.turns, r.timeout, r.reached, trace


def _pair(a: list, b: list) -> tuple[float, float]:
    """Mean and standard error of a[i] - b[i] over the seeds both played."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0
    d = [a[i] - b[i] for i in range(n)]
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / max(1, n - 1)
    return m, math.sqrt(var / n)


def paired(opts: list[Option]) -> None:
    """Each option's mean difference to skipping, over the seeds both played."""
    base = next((o for o in opts if o.name == "skip" and o.scores), None)
    if base is None:
        return
    for o in opts:
        if o is base or not o.scores:
            continue
        m, se = _pair(o.scores, base.scores)
        o.diff, o.se = round(m, 3), round(se, 3)


def _best(cands: list[Option]) -> Option:
    """Highest mean; on a tie the first (kit order, or the first offer). Options that
    stopped behind the leader are never chosen."""
    left = [o for o in cands if not o.out] or cands
    return max(left, key=lambda o: (o.value, -left.index(o)))


@lru_cache(maxsize=64)
def _waves(seed: int, scenario: str) -> tuple:
    """The seed's waves on the scenario's board, e.g. ('AAC', 'TT', ...)."""
    lp = plan_levels(seed, scenario=scenario)[0]
    return tuple("".join(CAT.enemy(k).tag for k, _, _ in w) for w in lp.waves)


def scenario_info(scenario: str = SCENARIO) -> dict:
    """The simulated fights in words, for the live viewer."""
    if scenario != "stress":
        return {"name": scenario, "board": "the rest of Stage 1", "depth": DEPTH}
    name, cells = LEVELS[-1][0], LEVELS[-1][1]
    lv = CAT.levels[name]
    title = f"{CAT.regions[lv['region']]['name']} {'I' * lv['n']}"
    boss = CAT.levels["boss"]["boss"]
    return {"name": "stress", "board": f"{title} ({cells} cells)", "waves": STRESS_WAVES, "enemy_hp": "Day 5+",
            "boss": CAT.enemy(boss).name, "boss_hp": enemy_hp(boss, True), "depth": DEPTH,
            "score": f"HP left - {TURN_COST} per turn; a death -{HERO_MAX_HP}; a stalled run HP - {STALL_PENALTY:g}"}


def _fight(seed: int, f, trace: bool = False) -> dict:
    sc, win, hp, turns, timeout, reached = f[:6]
    d = {"seed": seed, "score": round(sc, 2), "win": bool(win), "hp": hp, "turns": turns,
         "timeout": bool(timeout), "died": None if (win or timeout) else reached}
    if trace and len(f) > 6 and f[6]:
        d["trace"] = f[6]
    return d


class _Valuation:
    """One sequential valuation (module docstring): which fights to play next, the looks,
    and the running state for the progress feed."""

    def __init__(self, opts, seeds, hp, depth, scenario, marks, stop_z, workers, progress, job):
        self.opts, self.seeds, self.hp, self.depth, self.scenario = opts, seeds, hp, depth, scenario
        self.marks, self.stop_z, self.workers, self.progress, self.job = marks, stop_z, workers, progress, job
        self.skip = next((o for o in opts if o.name == "skip"), None)
        self.got = {id(o): {} for o in opts}       # option -> {seed index: fight}
        self.playing = list(opts)                 # options still being played, in option order
        self.k = 0                                 # the next look
        self.looked = 0                            # seeds at the last look
        self.fights = 0
        self.t0 = time.time()
        self.feed_s, self.feeds, self.last_feed = 0.0, 0, 0.0
        for o in opts:
            o.scores, o.fights, o.stopped, o.out, o.above = [], [], None, False, False
        # fights played before, by this or an earlier decision (the fight cache): taken as played
        self.cached, self.new = 0, {}
        self.caching = self.job is _job
        if self.caching:
            _cache_load()
            for o in opts:
                kl = o.kit.to_list()
                for i, seed in enumerate(seeds):
                    f = _CACHE.get(fight_key(kl, hp, seed, depth, scenario))
                    if f is not None:
                        self.got[id(o)][i] = f
                        self.cached += 1

    def _got(self, o: Option, i: int, f) -> None:
        self.got[id(o)][i] = f
        self.fights += 1
        if self.caching:
            self.new[fight_key(o.kit.to_list(), self.hp, self.seeds[i], self.depth, self.scenario)] = f

    # ---- order and looks
    def next_job(self, sent: set):
        for i in range(len(self.seeds)):
            for o in self.playing:
                if (id(o), i) not in sent and i not in self.got[id(o)]:
                    return o, i
        return None

    def ready(self) -> bool:
        n = self.marks[self.k]
        return all(all(i in self.got[id(o)] for i in range(n)) for o in self.playing)

    def _take(self, o: Option, n: int) -> None:
        g = self.got[id(o)]
        o.fights = [g[i] for i in range(n)]
        o.scores = [f[0] for f in o.fights]
        o.value = round(sum(o.scores) / n, 3) if n else 0.0
        o.wins = sum(1 for f in o.fights if f[1])

    def look(self) -> bool:
        """Compare the options on the first n seeds; True when the decision is settled."""
        n = self.marks[self.k]
        self.k += 1
        self.looked = n
        for o in self.playing:
            self._take(o, n)
        if self.k == len(self.marks) or self.stop_z is None:
            return True
        z, floor = self.stop_z, SD_FLOOR / math.sqrt(n)
        skip = self.skip if self._playing(self.skip) else None
        cands = [o for o in self.playing if o is not self.skip]
        if skip is not None:
            for o in cands:
                m, se = _pair(o.scores, skip.scores)
                if m + z * max(se, floor) < 0:
                    o.out, o.stopped = True, f"a net loss, {m:+.2f} +- {se:.2f} HP against skipping after {n} seeds"
                elif m - z * max(se, floor) > 0:
                    o.above = True
        left = [o for o in cands if not o.out]
        if left:
            lead = _best(left)
            for o in left:
                if o is not lead:
                    m, se = _pair(lead.scores, o.scores)
                    if m > z * max(se, floor):
                        o.out, o.stopped = True, f"{m:.2f} +- {se:.2f} HP behind {lead.name} after {n} seeds"
        left = [o for o in cands if not o.out]
        stop = [o for o in cands if o.out]
        need_skip = skip is not None and bool(left) and not all(o.above for o in left)
        if skip is not None and left and not need_skip:
            skip.stopped = f"settled after {n} seeds: every option left beats it by more than {z:g} SE"
            stop.append(skip)
        self.playing = [o for o in self.playing if not any(o is x for x in stop)]
        return not left or (len(left) == 1 and not need_skip)

    def _playing(self, o) -> bool:
        return o is not None and any(o is x for x in self.playing)

    # ---- playing the fights
    def run(self) -> None:
        self.feed(force=True)
        if self.workers > 1 and len(self.opts) > 1:
            for _ in range(2):            # a broken pool is replaced once, then this process plays
                p = pool(self.workers)
                if p is None:
                    break
                try:
                    if self._run_pool(p):
                        return
                    break
                except Exception:
                    close_pool()
        self._run_here()

    def _settle(self) -> bool:
        while self.k < len(self.marks) and self.ready():
            if self.look():
                return True
        return False

    def _run_pool(self, p) -> bool:
        inflight: dict = {}
        sent: set = set()
        try:
            while True:
                if self._settle():
                    return True
                while len(inflight) < self.workers:
                    nj = self.next_job(sent)
                    if nj is None:
                        break
                    o, i = nj
                    sent.add((id(o), i))
                    f = p.submit(self.job, (o.kit.to_list(), self.hp, self.seeds[i], self.depth, self.scenario))
                    inflight[f] = (o, i)
                if not inflight:
                    return False
                done, _ = wait(list(inflight), timeout=FEED_S, return_when=FIRST_COMPLETED)
                for f in done:
                    o, i = inflight.pop(f)
                    self._got(o, i, f.result())
                self.feed()
        finally:
            for f in inflight:            # fights of options that stopped: not waited for
                f.cancel()

    def _run_here(self) -> None:
        while not self._settle():
            nj = self.next_job(set())
            if nj is None:
                return
            o, i = nj
            self._got(o, i, self.job((o.kit.to_list(), self.hp, self.seeds[i], self.depth, self.scenario)))
            self.feed()

    # ---- progress feed
    def feed(self, force: bool = False, phase: str = "running") -> None:
        if self.progress is None:
            return
        t = time.time()
        if not force and t - self.last_feed < FEED_S:
            return
        t1 = time.perf_counter()
        try:
            self.progress(self.state(phase))
        except Exception:
            pass                          # the viewer must never stop the valuation
        self.feed_s += time.perf_counter() - t1
        self.feeds += 1
        self.last_feed = t

    def state(self, phase: str = "running") -> dict:
        """Running results: per option the seeds played in order so far, mean +- SE, the
        difference to skipping +- SE, whether it stopped and why, and one example fight:
        the seed where the options differ most so far (the same fight for every option)."""
        pre = {}
        for o in self.opts:
            if self._playing(o):
                g, k = self.got[id(o)], 0
                while k < len(self.seeds) and k in g:
                    k += 1
                pre[id(o)] = [g[i] for i in range(k)]
            else:
                pre[id(o)] = o.fights
        common = min(len(v) for v in pre.values())
        ex = None
        if common:
            ex = max(range(common), key=lambda i: (max(pre[id(o)][i][0] for o in self.opts)
                                                   - min(pre[id(o)][i][0] for o in self.opts), -i))
        sk = [f[0] for f in pre[id(self.skip)]] if self.skip is not None else []
        rows, lead, lead_v = [], None, None
        for o in self.opts:
            f = pre[id(o)]
            sc = [x[0] for x in f]
            n = len(sc)
            mean = sum(sc) / n if n else None
            se = math.sqrt(sum((x - mean) ** 2 for x in sc) / (n - 1) / n) if n > 1 else None
            row = {"name": o.name, "seeds": n, "of": len(self.seeds), "mean": None if mean is None else round(mean, 2),
                   "se": None if se is None else round(se, 2), "wins": sum(1 for x in f if x[1]),
                   "playing": self._playing(o), "stopped": o.stopped, "out": o.out}
            if o is not self.skip and sk and n:
                m, s = _pair(sc, sk)
                if min(n, len(sk)) > 1:
                    row.update(diff=round(m, 2), diff_se=round(s, 2), net_loss=m + NOISE_Z * s < 0)
            if f:
                i = ex if ex is not None and ex < n else n - 1
                row["example"] = _fight(self.seeds[i], f[i], trace=i == ex)   # the replay: that seed only
            if o is not self.skip and not o.out and mean is not None and (lead_v is None or mean > lead_v):
                lead, lead_v = o.name, mean
            rows.append(row)
        st = {"phase": phase, "elapsed": round(time.time() - self.t0, 1), "workers": self.workers,
              "seeds": len(self.seeds), "looked": self.looked, "fights": self.fights,
              "fights_max": len(self.opts) * len(self.seeds), "leader": lead, "options": rows,
              "looks": list(self.marks), "stop_z": self.stop_z, "scenario": scenario_info(self.scenario)}
        if ex is not None:
            st["example_seed"] = self.seeds[ex]
            try:
                st["example_waves"] = list(_waves(self.seeds[ex], self.scenario))
            except Exception:
                pass
        return st


def progress_line(st: dict) -> str:
    """The running state in one console line (the bot prints it every few seconds)."""
    parts = []
    for r in st.get("options", []):
        t = f"{r['name']} {r['seeds']}/{r['of']}"
        if r.get("mean") is not None:
            t += f" {r['mean']:.2f}"
        if r.get("diff") is not None:
            t += f" ({r['diff']:+.2f}+-{r['diff_se']:.2f}{', net loss' if r.get('net_loss') else ''})"
        if r.get("stopped"):
            t += " stopped"
        elif r["name"] == st.get("leader"):
            t += " leader"
        parts.append(t)
    return f"{st.get('elapsed', 0):.0f}s, {st.get('fights', 0)} fights: " + " | ".join(parts)


def evaluate(opts: list[Option], hp: int, seeds=SEEDS, depth: int = DEPTH, workers: int = WORKERS,
             scenario: str = SCENARIO, *, looks=LOOKS, stop_z: float | None = STOP_Z, progress=None,
             job=None) -> dict:
    """Fill each option's per-seed scores, mean, wins and paired difference to 'skip' (when
    skip is among the options), playing the seeds in order and stopping options that are
    settled (module docstring). stop_z=None plays every option on every seed (0.22).
    progress(state) gets the running results at least every FEED_S seconds and at the end."""
    seeds = tuple(seeds)
    hp = max(1, min(max([HERO_MAX_HP] + [o.kit.max_hp for o in opts]), hp))     # Healthy: up to 12 [rec2 f0802]
    marks = sorted({n for n in looks if 0 < n < len(seeds)} | {len(seeds)}) if stop_z else [len(seeds)]
    v = _Valuation(opts, seeds, hp, depth, scenario, marks, stop_z, workers, progress, job or _job)
    v.run()
    _cache_store(v.new)
    paired(opts)
    v.feed(force=True, phase="valued")
    return {"fights": v.fights, "cached": v.cached, "seeds": max((len(o.scores) for o in opts), default=0),
            "of": len(seeds), "feed_s": round(v.feed_s, 3), "feeds": v.feeds, "seconds": round(time.time() - v.t0, 1)}


def _stops(opts: list[Option], stats) -> str:
    """What the sequential test settled early, for the rule text."""
    stopped = [o for o in opts if o.stopped]
    if not stopped or not isinstance(stats, dict) or not stats.get("of"):
        return ""
    head = (f"settled after {stats['seeds']} of {stats['of']} seeds" if stats["seeds"] < stats["of"]
            else f"all {stats['of']} seeds")
    return f" [{head}; " + "; ".join(f"{o.name} stopped: {o.stopped}" for o in stopped) + "]"


# ---------------------------------------------------------------- decisions
def decide_upgrade(kit: Kit, upgrade_id: str, hp: int, *, seeds=SEEDS, depth: int = DEPTH,
                   workers: int = WORKERS, scenario: str = SCENARIO, evaluator=None, progress=None) -> Decision:
    t0 = time.time()
    evaluator = evaluator or evaluate
    up = CAT.upgrade(upgrade_id)
    opts = upgrade_options(kit, upgrade_id)
    skip, cands = opts[0], opts[1:]
    if not offerable(up):
        return Decision(skip, opts, "not a free reward (sacrifice or unknown effect)", 0.0)
    if not cands:
        return Decision(skip, opts, "no tile can take it", 0.0)
    if pure_gain(up):
        if len(cands) == 1:
            return Decision(cands[0], opts, "pure gain: always taken; one tile can take it", 0.0)
        if up.damage == 0 and up.cooldown == 0:
            # only adds an upgrade slot: no fight changes, so no simulation can tell
            # tiles apart; give it to the strongest tile, the one most worth upgrading again
            best = max(cands, key=lambda o: (CAT.tile(kit[o.detail["label"]].tile).ready_value,
                                             -kit[o.detail["label"]].free_levels, -cands.index(o)))
            return Decision(best, opts, "pure gain: always taken; slot goes to the strongest tile", 0.0)
        stats = evaluator(cands, hp, seeds, depth, workers, scenario, progress=progress)
        n = max((len(o.scores) for o in cands), default=0)
        return Decision(_best(cands), opts, f"pure gain: always taken; simulation picked the tile on {n} seeds"
                        + _stops(cands, stats), round(time.time() - t0, 1), stats or {})
    stats = evaluator(opts, hp, seeds, depth, workers, scenario, progress=progress)
    return _tradeoff(opts, "trade-off", t0, stats)


def decide_tile(kit: Kit, offers: list[dict], hp: int, *, seeds=SEEDS, depth: int = DEPTH,
                workers: int = WORKERS, scenario: str = SCENARIO, evaluator=None, progress=None) -> Decision:
    t0 = time.time()
    evaluator = evaluator or evaluate
    opts = tile_choice_options(kit, offers)
    if len(opts) == 1:
        return Decision(opts[0], opts, "no offered tile the simulator can play", 0.0)
    stats = evaluator(opts, hp, seeds, depth, workers, scenario, progress=progress)
    return _tradeoff(opts, "tile choice", t0, stats)


# The coins' worth when kept, in the valuation's HP: none yet. Coins buy only in shops; since 0.31 a
# --full run passes two (after Bamboo Grove and after the second region), and what the second sells is
# not known at the first, so a pure gain is bought whenever affordable [runs 20261029-124921, -154200].
COIN_HP = 0.0


# ---------------------------------------------------------------- the shop: order and the counter
def skill_class(sid: str | None) -> str | None:
    """How a skill on the counter is bought: 'pure' (nothing gets worse), 'trade' (a cost: Iron Skin's
    and Fortress's max HP), None (not simulated, or unknown: never bought) [data/skills.json]."""
    sk = CAT.skills.get(sid) if sid else None
    if sk is None or not sk.sim:
        return None
    e = sk.effect
    if e.get("max_hp", 0) < 0 or e.get("special_cooldown", 0) > 0:
        return "trade"
    return "pure"


def shop_priority(upgrade_id: str | None = None, good: dict | None = None) -> int | None:
    """The order of the shop's purchases [user 2026-10-29, review of recording 2: "your priorities
    first: -2 cooldown and +1 damage upgrades, Healthy (+2 max HP), +1 upgrade slot"]: 1 a statue
    upgrade that adds damage or takes cooldown off, 2 Healthy, 3 a statue upgrade that only adds a slot,
    4 the other simulated skills, 5 trade-offs (the statue's or a skill's); None: never bought (an
    unread, unknown or unsimulated offer, Sacrifice, items and services)."""
    if upgrade_id is not None:
        up = CAT.upgrades.get(upgrade_id)
        if up is None or not offerable(up):
            return None
        if pure_gain(up):
            return 1 if (up.damage > 0 or up.cooldown < 0) else 3
        return 5
    if not good or good.get("kind") != "skill" or good.get("currency") != "coin" or good.get("price") is None:
        return None
    cls = skill_class(good.get("id"))
    if cls is None:
        return None
    if cls == "trade":
        return 5
    return 2 if CAT.skill(good["id"]).effect.get("type") == "max_hp" else 4


def decide_skill(kit: Kit, sid: str | None, price: int | None, coins: int | None, hp: int, *,
                 seeds=SEEDS, depth: int = DEPTH, workers: int = WORKERS, scenario: str = SCENARIO,
                 evaluator=None, progress=None) -> Decision:
    """A skill on the shop's counter against keeping the coins, as the statue's upgrade (decide_shop):
    never unread, unaffordable or unsimulated; a pure gain is bought (no simulation: the build only gains,
    and kept coins have no value yet); a trade-off only when it scores above keeping the coins by more
    than the noise (difference minus 2 standard errors above price x COIN_HP). A skill already owned is
    bought to its next level [rec2 f0790: 'LEVEL 2: +2 DAMAGE ...']."""
    t0 = time.time()
    skip = Option("skip", kit)
    sk = CAT.skills.get(sid) if sid else None
    if sk is None:
        return Decision(skip, [skip], "shop: the counter's offer was not read as a skill; coins kept", 0.0)
    if kit.level(sid) >= sk.max_level:
        return Decision(skip, [skip], f"shop: {sk.name} is at its last level; coins kept", 0.0)
    opt = Option(f"skill:{sid}", kit.with_skill(sid), detail={"skill": sid, "level": kit.level(sid) + 1})
    opts = [skip, opt]
    if price is None or coins is None:
        return Decision(skip, opts, f"shop: {'price' if price is None else 'coins'} not read; coins kept", 0.0)
    if price > coins:
        return Decision(skip, opts, f"shop: {sk.name} costs {price}, {coins} coins held; coins kept", 0.0)
    cls = skill_class(sid)
    if cls is None:
        return Decision(skip, opts, f"shop: {sk.name} is not simulated yet; coins kept", 0.0)
    if cls == "pure":
        return Decision(opt, opts, f"shop: {sk.name} for {price} of {coins} coins, a simulated skill that only gains: "
                                   f"bought without a valuation", 0.0)
    stats = (evaluator or evaluate)(opts, hp, seeds, depth, workers, scenario, progress=progress)
    secs = round(time.time() - t0, 1)
    stats = stats if isinstance(stats, dict) else {}
    if opt.diff is not None and not opt.out and opt.diff - NOISE_Z * (opt.se or 0.0) > price * COIN_HP:
        return Decision(opt, opts, f"shop: {sk.name} for {price} of {coins} coins scores {opt.diff:+.2f} +- "
                                   f"{opt.se or 0.0:.2f} HP against keeping the coins", secs, stats)
    return Decision(skip, opts, f"shop: {sk.name} for {price}: it does not score above keeping the coins by more "
                                f"than the noise; coins kept", secs, stats)


def decide_shop(kit: Kit, upgrade_id: str | None, price: int | None, coins: int | None, hp: int, *,
                seeds=SEEDS, depth: int = DEPTH, workers: int = WORKERS, scenario: str = SCENARIO,
                evaluator=None, progress=None) -> Decision:
    """The shop statue's upgrade against keeping the coins [user review of run 20261029-154200:
    '+1 UPGRADE SLOT' for 10 coins with 26 held, not bought]. The same options as at a shrine
    ('skip' = keep the coins) and the same simulation, with two differences, as a purchase costs
    coins: nothing is bought unread or unaffordable, and a trade-off is bought only when its best
    option scores above keeping the coins by more than the noise (difference minus 2 standard
    errors above price x COIN_HP), not merely when it is no net loss. A pure gain is bought, its
    tile picked as at a shrine; Sacrifice and unknown effects never. An enchantment is bought as a
    shrine takes one, unless every option is a net loss [user 2026-10-30: "rather more use than less,
    and we learn from that"]."""
    t0 = time.time()
    up = CAT.upgrades.get(upgrade_id) if upgrade_id else None
    opts = upgrade_options(kit, upgrade_id) if up is not None else [Option("skip", kit)]
    skip = opts[0]
    if up is None:
        return Decision(skip, opts, "shop: the statue's upgrade was not read; coins kept", 0.0)
    if not offerable(up):
        # before the price: Sacrifice has none (it pays 40 coins), and 0.29-0.32.1 logged 'price not read'
        # [run 20261030-041652, shop 2]
        return Decision(skip, opts, f"shop: {up.name} is never bought (sacrifice or unknown effect)", 0.0)
    if price is None or coins is None:
        return Decision(skip, opts, f"shop: {'price' if price is None else 'coins'} not read; coins kept", 0.0)
    if price > coins:
        return Decision(skip, opts, f"shop: {up.name} costs {price}, {coins} coins held; coins kept", 0.0)
    if len(opts) == 1:
        return Decision(skip, opts, f"shop: no tile can take {up.name}; coins kept", 0.0)
    if pure_gain(up):
        d = decide_upgrade(kit, upgrade_id, hp, seeds=seeds, depth=depth, workers=workers, scenario=scenario,
                           evaluator=evaluator, progress=progress)
        d.rule = f"shop: {up.name} for {price} of {coins} coins, a " + d.rule
        return d
    stats = (evaluator or evaluate)(opts, hp, seeds, depth, workers, scenario, progress=progress)
    if up.kind == "enchant":
        # enchantments: more use rather than less [user 2026-10-30, after the comparison with recording 2's
        # shrines, where you took 1 of 5: "dont trust my playstyle ... rather more use than less, and we learn
        # from that"]. Run 20261030-041652's shop 1 kept 20 coins over Shockwave on Swirl at +0.13 HP
        return _tradeoff(opts, f"shop: {up.name} for {price} of {coins} coins, an enchantment", t0, stats)
    secs = round(time.time() - t0, 1)
    stats = stats if isinstance(stats, dict) else {}
    note = _stops(opts, stats)
    cost = price * COIN_HP
    above = [o for o in opts[1:] if not o.out and o.diff is not None and o.diff - NOISE_Z * (o.se or 0.0) > cost]
    if not above:
        return Decision(skip, opts, f"shop: {up.name} for {price}: no option scores above keeping the coins by "
                                    f"more than the noise; coins kept" + note, secs, stats)
    best = _best(above)
    return Decision(best, opts, f"shop: {up.name} for {price} of {coins} coins: {best.name} scores "
                                f"{best.diff:+.2f} +- {best.se or 0.0:.2f} HP against keeping the coins" + note,
                    secs, stats)


def _tradeoff(opts: list[Option], what: str, t0: float, stats=None) -> Decision:
    """Skip only when every option is a net loss; otherwise take the best option that is
    not one, also when it scored below skipping but within the noise [user, after run
    20261028-082756]. The rule text says which case decided: run 20261028-154317 took
    -1 damage -4 cooldown on Arrow at -0.20 +- 0.29 below skipping under the line 'best
    option that is not a net loss', which read as if it had beaten skipping [user review
    of the 0.20 series]. Options that stopped behind the leader are not chosen."""
    skip, cands = opts[0], opts[1:]
    secs = round(time.time() - t0, 1)
    stats = stats if isinstance(stats, dict) else {}
    note = _stops(opts, stats)
    keep = [o for o in cands if not o.net_loss and not o.out]
    if not keep:
        return Decision(skip, opts, f"{what}: every option a net loss, skipped" + note, secs, stats)
    best = _best(keep)
    if best.diff is None:
        return Decision(best, opts, f"{what}: best option that is not a net loss" + note, secs, stats)
    margin = f"{best.diff:+.2f} +- {best.se or 0.0:.2f} HP against skipping"
    if best.diff >= 0:
        return Decision(best, opts, f"{what}: {best.name} scores {margin}, the best option" + note, secs, stats)
    below = "every option scored below skipping; " if all(o.diff is not None and o.diff < 0 for o in cands) else ""
    return Decision(best, opts, f"{what}: {below}{best.name} scores {margin}, a loss within the noise, "
                                f"so it is taken (skip only when every option is a net loss)" + note, secs, stats)
