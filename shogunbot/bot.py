"""Live bot: capture -> perceive -> plan -> act -> log.

    python -m shogunbot play                 # one run of Stage 1, frames saved on events
    python -m shogunbot play --runs 10       # ten runs in a row, unattended
    python -m shogunbot play --full          # go on after Stage 1's boss (M4, in progress)
    python -m shogunbot play --frames all    # save every turn's frames
    python -m shogunbot play --no-autostart  # start the run yourself; bot takes over in combat

Hold F8 to stop the bot at the next turn boundary (Ctrl+C also works).

Screens:
  run start -> anywhere but the camp or the title: the run in progress is given up first
             (pause menu); the death screen -> BACK TO CAMP (autostart) [user 2026-10-29]
  title   -> click NEW RUN                    (autostart)
  camp    -> step the day selector down to Day 1, walk right, click GO (autostart)
  map     -> Stage 1: click the first node; after a boss (--full): keyboard-only mode, read the
             marked node, Enter on a shop or a MAP_REGIONS region (perception/map_screen.py);
             stops after MAP_TRIES_MAX inputs (autostart)
  shop    -> read the offers, click GO (perception/shop_screen.py)
  combat  -> plan one action and send it
  reward  -> shrine: read the upgrade; a pure gain is always taken (simulation picks the tile),
             a trade-off is taken unless every tile option is a net loss against skipping;
             tile choice: take the best tile unless both are net losses;
             anything unreadable -> SKIP (rewards.py, perception/reward_screen.py)
  unknown -> wait (banners, transitions); a frame is logged if it lasts
The run ends on: the final boss beaten (Stage 1's with --stage1, the default; else the boss of
FULL_LAST_STAGE; the reward after it is taken first), hero HP 0 or a death screen, or a screen
that stays unknown too long. In a series, a won run is given up from the pause menu, which
leads back to the camp; the next run walks to GO there.
"""

from __future__ import annotations

import ctypes
import json
import queue
import threading
import time
import types

import numpy as np

from . import rewards
from .bridge import Tracker, kit_from_faces, level_name, match_faces
from .catalogue import CAT
from .model import BOSS_KINDS, DEFAULT_KIT, ENEMY_TILE_DAMAGE, HERO_MAX_HP
from .perception import map_screen, reward_screen, screens, shop_screen
from .perception.perceive import BASE_POOL, merge_blink, normalize, perceive
from .perception.tiles import FACE_H, FACE_W
from .planner import STALL_TURNS, WEIGHTS, plan, plan_live, progress_key, stall_weights
from .rules import ICE_TURNS, POISON_TICK, POISON_TURNS, free_action, legal_hero_actions, step
from .live import LiveView
from .runlog import RunLogger
from .waves import Waves
from .winio import GameWindow

VERSION = "0.34"
ACTION_KEY = {"left": "a", "right": "d", "turn": "w", "wait": "s", "attack": "space"}
LEVEL_WAVES = {lv["id"]: lv["waves"] for lv in CAT.regions["bamboo_grove"]["level_list"]}  # 4, 5, boss 0
VK_F8 = 0x77
# Edamame Brew restores up to 3 HP [catalogue]. It is drunk [user 2026-10-29] (a) when a hit this
# turn or the next would kill the hero or leave HP within one enemy hit of 0, and its HP prevent
# that; or (b) when all of its HP heal (HP <= max - 3), threat or not; or (c) to free its slot for a
# better item or another potion on the ground (ground items are not read yet, so (c) never
# applies). 0.24-0.27.1 required (a) and (b) together, so run 20261029-124921 held two brews from
# HP 7 down to 3 [user review]. It is not part of the last level's 'use it or lose it' pass.
HEAL = CAT.consumable_by_screen("green_potion").effect["amount"]
RISK_P = 0.1              # an outcome this likely or more counts as a hit that can land
POISON_MIN_VALUE = 4      # use Mass Poison when it deals at least this much in total
# The run's last level, the final boss: items still held there are lost, so each is spent at its
# first useful moment [user 2026-10-28: "use every consumable"; series 20261028-132238..135635
# ended five runs holding Mass Ice, Edamame, Cool Up and Kami Brew]. With --stage1 that is Stage
# 1's boss; a --full run is won, and in a series given up, after the boss of FULL_LAST_STAGE
# [m4-plan: the bot clears the region 2 boss]. Items carry over between stages [user 2026-10-29].
FULL_LAST_STAGE = 3       # Moonlit Port, The Twins [user 2026-10-29: extend --full to region 3]
MAP_TRIES_MAX = 12        # map inputs that leave the map unchanged before the run stops as 'stuck'
                          # [run 20261029-072042: 18 inputs on a Stage 2 map, none applied]
START_GIVE_UPS_MAX = 2    # give-ups at a run start that still do not reach the camp: 'stuck'
SHOP_GO_TRIES = 3         # GO clicks before the run stops as 'stuck' in a shop
HOVER_S = 0.5             # the cursor on a counter slot this long before its tooltip is read
HOVER_TRIES = 2           # looks per slot
NO_CHANGE_STOP = 20       # inputs in a row that change nothing on screen before the run stops as 'stuck'
                          # [user review of run 20261030-041652: the refused Arrow click was sent on every turn
                          # from t237, over 800 in a row; the longest streak in the 37 earlier run logs was 5]
# The map between regions (M4 step 1) is played in keyboard-only mode [user 2026-10-29]: its button
# turns it on, the up/down arrows move the mark among the reachable nodes, Enter travels there, and
# any mouse click turns it off again (perception/map_screen.py). Regions in order of preference;
# a shop is always taken (both shops after Bamboo Grove led to both regions).
MAP_REGIONS = ("whispering_caves", "moonlit_port")   # [user 2026-10-28: Whispering Caves first; 2026-10-29:
                                                     # region 3, the one region after the Caves]
MAP_LOOKS_MAX = 4         # arrow presses spent looking for a preferred region before taking the mark
MOUSE_MODE_SPOT = (0.97, 0.27)   # an empty spot: a click there turns keyboard-only mode off
BLINK_SAMPLES = 6         # extra frames per turn to catch the yellow blink
BLINK_GAP_S = 0.15
UNKNOWN_FRAME_S = 6.0     # save a frame when a screen stays unknown this long
UNKNOWN_GIVE_UP_S = 90.0  # end the run as 'stuck'
GIVE_UP_WAIT_S = 20.0     # after a win: wait up to this long for the map before pausing
DEATH_SCREEN_MAX_S = 30.0 # a death screen without a readable RESTART ends the new run as 'stuck'
HP_CONFIRM_S = 0.35       # a second look before an HP drop counts [run 20261028-132952 t77, t82]
SPAWN_WAIT_S = 2.5        # wait at most this long for spawn flames to clear before reading facing
SPAWN_POLL_S = 0.25
CONSOLE_FEED_S = 5.0      # during a reward valuation, a progress line in the console this often
LOWEST_DAY = True         # at the camp, step the day selector down to Day 1 [user 2026-10-29: until the
                          # elite traits of Day 2 are simulated]
DAY_CLICKS_MAX = 4        # left-arrow clicks before the run stops as 'stuck' (the selector did not react)


def _stop_requested() -> bool:
    try:
        return bool(ctypes.WinDLL("user32").GetAsyncKeyState(VK_F8) & 0x8000)
    except Exception:
        return False


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def one_hit(s) -> int:
    """The most damage one enemy tile on the board deals [catalogue]: HP at or below this is
    at real risk (Stage 1: Sword and Arrow, 2). Unknown kinds count every enemy tile."""
    dmg = []
    for e in s.alive():
        kind = CAT.enemies.get(e.kind)
        tiles = list(kind.tiles) if kind is not None and kind.tiles else list(ENEMY_TILE_DAMAGE)
        dmg += [ENEMY_TILE_DAMAGE.get(t, 0) for t in tiles + list(e.queue)]
    return max(dmg, default=0) or 1


def worst_loss(s, action: str, plies: int = 2, free_left: int = 1) -> int:
    """HP the hero can lose with `action` this turn and, from each outcome, the least damaging
    reply on the next turn: the worst outcome that has at least RISK_P probability (the
    likeliest always counts). Queueing a Free-Play tile does not end the turn (one is followed)."""
    free = free_left > 0 and free_action(s, action)
    outs = sorted(step(s, action), key=lambda t: -t[0])
    worst = 0
    for i, (p, s2, ev) in enumerate(outs):
        if i and p < RISK_P:
            break
        loss = sum(d for _, _, d in ev.hero_damage)
        left = plies if free else plies - 1
        if left > 0 and s2.hero.hp > 0 and s2.alive():
            loss += min(worst_loss(s2, a, left, free_left - free) for a in legal_hero_actions(s2))
        worst = max(worst, loss)
    return worst


def edamame_reason(s, action: str, hp_cap: int) -> str | None:
    """Why to drink Edamame Brew now, or None [user 2026-10-29]: (b) all of its HP heal, or (a) a
    hit this turn (with the chosen action) or the next (with the least damaging reply) would kill
    the hero or leave HP within one enemy hit of 0, and the brew prevents that."""
    hp = s.hero.hp
    if hp <= 0 or hp >= hp_cap:
        return None
    if hp <= hp_cap - HEAL:
        return f"all {HEAL} of its HP heal (HP {hp} of {hp_cap})"
    if not s.alive():
        return None
    healed = min(hp_cap, hp + HEAL)
    m = s.clone()
    m.hero.hp = healed               # the hits do not depend on HP; counted without an early death
    loss = worst_loss(m, action)
    hit = one_hit(s)
    without, with_brew = hp - loss, healed - loss
    if loss <= 0 or without > hit:
        return None                                  # no hit coming, or no real risk
    if with_brew <= (hit if without > 0 else 0):
        return None                                  # the brew would not prevent it
    risk = "kill the hero" if without <= 0 else f"leave {without} HP (one hit: {hit})"
    return f"a hit this turn or the next could {risk}; with the brew {with_brew} HP"


KEY_LABEL = {"a": "A", "d": "D", "w": "W", "s": "S", "space": "SPACE", "enter": "ENTER", "esc": "ESC",
             "up": "UP", "down": "DOWN"}


class Recorder:
    """One frame per input, same layout as the collector, so
    `python -m shogunbot replay runs/<id>` scores the bot's own runs."""

    def __init__(self) -> None:
        self.dir = None
        self.n = 0
        self.t0 = time.time()
        self.jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._writer, daemon=True).start()

    def _writer(self) -> None:
        from PIL import Image
        while True:
            arr, path = self.jobs.get()
            try:
                Image.fromarray(np.ascontiguousarray(arr[..., :3])).save(path, compress_level=1)
            except Exception:
                pass

    def start(self, run_dir) -> None:
        self.dir, self.n, self.t0 = run_dir, 0, time.time()

    def record(self, frame, label: str, click=None, **extra) -> None:
        if self.dir is None or frame is None:
            return
        self.n += 1
        name = f"f{self.n:04d}_{label}.png"
        self.jobs.put((frame, self.dir / name))
        rec = {"frame": name, "t": round(time.time() - self.t0, 2), "action": label, **extra}
        if click is not None:
            rec["click"] = [round(click[0], 4), round(click[1], 4)]
        with open(self.dir / "actions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def flush(self, timeout: float = 5.0) -> None:
        t_end = time.time() + timeout
        while not self.jobs.empty() and time.time() < t_end:
            time.sleep(0.1)
        time.sleep(0.3)


class Bot:
    live = LiveView()   # never started unless __init__ runs: a no-op for tests that build a bare Bot
    stage1 = True       # the run ends after Stage 1's boss (--stage1); False: it goes on (M4)
    start_at_camp = False   # every run starts at the camp, giving a leftover run up first (__init__ sets it)

    def __init__(self, frame_policy: str = "events", autostart: bool = True, verbose: bool = True,
                 stage1: bool = True):
        self.gw = GameWindow()
        self.log = RunLogger("runs", frame_policy)
        rewards.CACHE_FILE = self.log.root / "fight_cache.jsonl"   # simulated fights kept between runs
        self.live = LiveView()            # runs/<id>/live.json for `python -m shogunbot watch`
        self.log.on_event = self.live.event
        self.rec = Recorder()
        self.autostart = autostart
        self.stage1 = stage1
        self.start_at_camp = autostart   # [user 2026-10-29: run 20261029-072042 began on a Stage 2 map]
        self.verbose = verbose
        self.restart_after = False   # play() sets it when another run follows this one
        self.gave_up = None          # after a won run: whether the pause-menu give-up worked

    regions = ("bamboo_grove",)   # the run's route so far: whose enemies can appear (perceive)
    day = None                    # the day the run is on, once the camp's selector was read

    def _see(self, frame, tr=None):
        """perceive() with the run's kit, route and day."""
        return perceive(frame, [t.tile for t in tr.kit] if tr is not None else None,
                        regions=tuple(self.regions), day=self.day)

    # every input goes through these three, so every input is recorded
    def key(self, name: str, frame, **extra) -> None:
        self.rec.record(frame, KEY_LABEL.get(name, name.upper()), **extra)
        self.gw.key(name)

    def click(self, fx: float, fy: float, frame, away: bool = True, **extra) -> None:
        self.rec.record(frame, "CLICK", click=(fx, fy), **extra)
        self.gw.click(fx, fy)
        self._check_cursor(extra.get("screen"))
        if away:
            self.gw.move_mouse_away()

    def hover(self, fx: float, fy: float, frame, **extra) -> None:
        """The cursor onto a spot and left there, so its tooltip shows (the shop's counter, 0.32)."""
        self.rec.record(frame, "HOVER", click=(fx, fy), **extra)
        self.gw.hover(fx, fy)

    CURSOR_TOL = 3     # px between the click's target and the cursor when the button went down

    def _check_cursor(self, screen) -> None:
        """Log where the cursor was when the button went down: on every map click, and on any
        click that missed its target by more than CURSOR_TOL px [user review of run
        20261029-072042: map clicks where the cursor never seemed to move]."""
        lc = getattr(self.gw, "last_click", None)
        if not lc or lc[2] is None:
            return
        off = max(abs(lc[0] - lc[2]), abs(lc[1] - lc[3]))
        if off > self.CURSOR_TOL or screen == "map":
            self.log.event("cursor", screen=screen, target=list(lc[:2]), cursor=list(lc[2:]), ok=off <= self.CURSOR_TOL)

    def _park_mouse(self) -> None:
        """Focus the game and park the cursor again. A window that takes the focus (a
        TeamViewer time-out dialog in run 20261028-133837) leaves the game showing the
        hover tooltip of whatever tile last sat under the cursor."""
        try:
            if hasattr(self.gw, "focus") and hasattr(self.gw, "is_foreground") and not self.gw.is_foreground():
                self.gw.focus()
                time.sleep(0.1)
            self.gw.move_mouse_away()
        except Exception as e:                    # never stop a reward over the cursor
            self.log.event("park_mouse_failed", error=repr(e))

    def say(self, msg: str) -> None:
        if self.verbose:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    # ------------------------------------------------------------------ act
    def _wait_change(self, before: np.ndarray, timeout: float = 2.5) -> bool:
        t_end = time.time() + timeout
        while time.time() < t_end:
            time.sleep(0.08)
            if _diff(self.gw.grab(), before) > 3.0:
                return True
        return False

    def do(self, action: str, p, frame: np.ndarray, kit=DEFAULT_KIT, charge=None) -> bool | None:
        """Send one hero action. Returns True if the screen reacted, False if it
        did not visibly change, None if nothing could be sent."""
        if action in ACTION_KEY:
            self.key(ACTION_KEY[action], frame, bot_action=action)
        elif action.startswith("queue:"):
            label = action.split(":", 1)[1]
            tile = kit[label].tile if label in kit else label
            # the face the bridge gave this label; with two copies of a tile, any ready copy
            slot = match_faces(kit, p.hand, charge).get(label) if label in kit else None
            if slot is None or not slot.get("ready", True):
                faces = [h for h in p.hand if h["tile"] == tile]
                slot = next((h for h in faces if h.get("ready")), slot or (faces[0] if faces else None))
            if slot is None:
                self.log.event("act_failed", action=action, reason="tile not found in hand",
                               hand=p.hand)
                return None
            if not slot.get("ready", True):
                # a dimmed face is recharging, whatever its marks say: the game ignores a click on it
                # [run 20261030-041652: the 4/8 Arrow's eighth mark fell outside the marks band, so it
                # read 7 of 7 charged while dimmed, and its click was refused on every turn from t237]
                self.log.event("act_failed", action=action, reason="the tile is dimmed (recharging)")
                return None
            self.click(slot["fx"], slot["fy"], frame, bot_action=action)
        else:
            self.log.event("act_failed", action=action, reason="unknown action")
            return None
        return self._wait_change(frame)

    # ------------------------------------------------------------------ rewards
    def _read_reward(self, tr):
        frame = self.gw.grab_stable(settle_s=0.25, timeout_s=3.0)
        # the reward readers measure 1080p pixels: other window sizes are scaled first.
        # Clicks are window fractions, so they fit any size [review of 0.15]
        return frame, reward_screen.read(normalize(frame), tuple(set(t.tile for t in tr.kit) | BASE_POOL))

    def _reward(self, frame, tr, level: str, hp: int) -> str:
        """Take every free reward (rules in rewards.py): pure-gain upgrades always,
        trade-offs and tiles unless every option is a net loss in simulation.
        Returns 'taken', 'skipped' or 'failed' (fell back to SKIP)."""
        time.sleep(0.8)                        # let the screen finish sliding in
        frame, rs = self._read_reward(tr)
        self.reward_n += 1
        tag = f"reward_{self.reward_n}_{rs.kind}"
        self.log.save_frame(frame, tag)
        notes: list = []
        if rs.hand:
            tr.kit = kit_from_faces(tr.kit, rs.hand, notes)
        # RunLogger.event's first parameter is called 'kind': the layout goes in as 'layout'
        self.log.event("reward_screen", level=level, hp=hp, kit=tr.kit.to_list(), notes=notes, frame=tag,
                       layout=rs.kind, **{k: v for k, v in rs.to_dict().items() if k not in ("hand", "kind")},
                       hand=[{k: h.get(k) for k in ("tile", "damage", "cooldown", "levels", "used", "conf")}
                             for h in rs.hand])
        offer = self._offer(rs, tr, level, hp)
        if rs.kind not in ("tile_choice", "shrine") or (rs.kind == "shrine" and not rs.upgrade):
            why = "upgrade text not read" if rs.kind == "shrine" else f"layout {rs.kind}"
            self.say(f"reward ({rs.kind}): {why}; SKIP")
            self.live.reward({"layout": rs.kind, "upgrade": None, "level": level, "hp": hp, "choice": "skip",
                              "rule": why, "options": []})
            self.live.valuation({**offer, "phase": "decided", "options": [], "choice": "skip", "rule": why,
                                 "click": "SKIP"}, reset=True)
            out = self._skip(frame, rs, why)
            self.live.valuation({"phase": "done", "outcome": out})
            return out
        self.say(f"reward ({rs.kind}{': ' + rs.upgrade if rs.upgrade else ''}): valuing the options "
                 f"on up to {len(rewards.SEEDS)} simulated fights if needed ...")
        self.live.valuation({**offer, "phase": "starting"}, reset=True)
        printed = [time.time()]

        def feed(st):
            """rewards.evaluate's running state: to the viewer at every call, and a line in
            this window every CONSOLE_FEED_S seconds [user 2026-10-29]."""
            self.live.valuation(st)
            if st.get("phase") == "running" and time.time() - printed[0] >= CONSOLE_FEED_S:
                printed[0] = time.time()
                self.say("   " + rewards.progress_line(st))
        if rs.kind == "tile_choice":
            dec = rewards.decide_tile(tr.kit, rs.offers, hp, progress=feed)
        else:
            dec = rewards.decide_upgrade(tr.kit, rs.upgrade, hp, progress=feed)
        best, opts, secs, st = dec.best, dec.options, dec.seconds, dec.stats or {}
        self.log.event("reward_decision", layout=rs.kind, upgrade=rs.upgrade, level=level, hp=hp, seconds=secs,
                       workers=rewards.WORKERS, choice=best.name, rule=dec.rule, options=[o.summary() for o in opts],
                       fights=st.get("fights"), cached=st.get("cached"), seeds=st.get("seeds"), seeds_max=st.get("of"),
                       feed_ms=round(1000 * st.get("feed_s", 0.0)), feeds=st.get("feeds"))
        self.live.reward({"layout": rs.kind, "upgrade": rs.upgrade, "level": level, "hp": hp, "seconds": round(secs, 1),
                          "choice": best.name, "rule": dec.rule, "options": [o.summary() for o in opts],
                          "fights": st.get("fights"), "seeds_max": st.get("of")})
        click = self._click_text(rs, best)
        self.live.valuation({"phase": "decided", "choice": best.name, "rule": dec.rule, "seconds": round(secs, 1),
                             "click": click, **({} if st else {"options": [{"name": o.name} for o in opts]})})
        scored = [o for o in opts if o.scores]
        self.say("   " + (", ".join(f"{o.name} {o.value:+.2f} ({len(o.scores)})"
                                    for o in sorted(scored, key=lambda o: -o.value)) or "no simulation needed")
                 + f"  ({secs:.0f}s{', ' + str(st['fights']) + ' fights' if st.get('fights') else ''}"
                   f"{', ' + str(st['cached']) + ' from the cache' if st.get('cached') else ''})"
                 + f" -> {best.name}  [{dec.rule}]")
        out = self._take(frame, rs, best, tr, dec)
        self.live.valuation({"phase": "done", "outcome": out})
        return out

    def _take(self, frame, rs, best, tr, dec) -> str:
        if best.name == "skip":
            return self._skip(frame, rs, dec.rule)
        if rs.kind == "tile_choice":
            ok = self._take_tile(frame, rs, best, tr)
        else:
            ok = self._take_upgrade(frame, rs, best, tr)
        if ok:
            tr.set_kit(best.kit)
            self.log.event("reward_taken", choice=best.name, kit=best.kit.to_list())
            self.say(f"   took {best.name}; kit {best.kit}")
            return "taken"
        frame, rs2 = self._read_reward(tr)
        if rs2.kind == "unknown" and rs2.skip is None:
            # the screen moved on although the check missed it: assume it was taken
            tr.set_kit(best.kit)
            self.log.event("reward_taken", choice=best.name, kit=best.kit.to_list(), unverified=True)
            return "taken"
        self._skip(frame, rs2, f"{best.name} did not register")
        return "failed"

    @staticmethod
    def _offer(rs, tr, level: str, hp: int) -> dict:
        """The reward screen for the live viewer: layout, offered tiles or the upgrade text, kit."""
        up = CAT.upgrades.get(rs.upgrade) if rs.kind == "shrine" and rs.upgrade else None
        return {"layout": rs.kind, "level": level, "hp": hp, "upgrade": rs.upgrade,
                "upgrade_text": up.name if up else None,
                "offers": [{"tile": o.get("tile"), "damage": o.get("damage"), "cooldown": o.get("cooldown"),
                            "levels": o.get("levels")} for o in rs.offers] if rs.kind == "tile_choice" else [],
                "kit": [f"{t.label} {t.damage}/{t.cooldown}" for t in tr.kit]}

    @staticmethod
    def _click_text(rs, best) -> str:
        """The click that follows the decision, in words."""
        if best.name == "skip":
            return "SKIP"
        if rs.kind == "tile_choice":
            o = rs.offers[best.detail["offer"]]
            name = CAT.tile(o["tile"]).name if o.get("tile") in CAT.tiles else o.get("tile")
            return f"the {name} pedestal ({'left' if o.get('fx', 0.5) < 0.5 else 'right'})"
        return f"{best.detail['label']} from the hand into the statue, then UPGRADE"

    def _skip(self, frame, rs, why: str) -> str:
        pos = rs.skip or screens.find_skip(normalize(frame))
        self.log.event("skip", reason=why, pos=pos)
        if pos is not None:
            self.click(*pos, frame, screen="reward")
            time.sleep(1.0)
        return "skipped"

    def _gone(self, before_kind: str, tr) -> bool:
        """True once the reward screen has moved on."""
        for _ in range(6):
            time.sleep(0.5)
            _, rs = self._read_reward(tr)
            if rs.kind != before_kind or rs.skip is None:
                return True
        return False

    def _take_tile(self, frame, rs, best, tr) -> bool:
        offer = rs.offers[best.detail["offer"]]
        for attempt in (1, 2):
            self.click(offer["fx"], offer["fy"], frame, screen="reward", reward=best.name, attempt=attempt)
            if self._gone("tile_choice", tr):
                return True
            frame = self.gw.grab()
        return False

    @staticmethod
    def _in_statue(rs, tr, label: str) -> bool:
        """The chosen copy sits in the statue's mouth. The statue face decides: the tile,
        and with two copies of it in the kit also the cooldown marks, which tell the copies
        apart. UPGRADE need not read as lit: the hover tooltip of the hand tile that slid
        under the cursor can cover it. In run 20261028-133837 (boss shrine, f0123) the
        chosen Swirl sat in the statue with UPGRADE hidden, the bot looked for that Swirl
        in the hand again, found only the other copy and skipped the upgrade. With the
        statue unread, a lit UPGRADE and the tile gone from the hand count."""
        kt = tr.kit[label]
        st = rs.statue
        if st is not None:
            if st.get("tile") != kt.tile:
                return False
            copies = [t for t in tr.kit if t.tile == kt.tile]
            return len(copies) == 1 or st.get("cooldown") == kt.cooldown
        return rs.upgrade_lit and label not in match_faces(tr.kit, rs.hand)

    def _take_upgrade(self, frame, rs, best, tr) -> bool:
        """Click the chosen tile into the statue, check that it is the one in the
        statue, then click UPGRADE. The hand closes up when a tile leaves it, so
        the tile is found again before every click; clicking a hand tile while
        another sits in the statue swaps them [run 20261028-092419: UPGRADE lit
        but unrecognised, and a second click at the old spot put Katana and then
        Arrow in the statue instead of the chosen tile].
        """
        label = best.detail["label"]
        cur = rs
        for attempt in (1, 2, 3):
            if self._in_statue(cur, tr, label):
                break                              # already there: a click landed but the check missed it
            face = match_faces(tr.kit, cur.hand).get(label)
            if face is None:
                self.log.event("reward_failed", choice=best.name, reason="tile not found in hand",
                               statue=cur.statue, attempt=attempt)
                return False
            self.click(face["fx"], face["fy"], frame, screen="reward", reward=best.name, attempt=attempt)
            time.sleep(0.8)
            self._park_mouse()
            frame, cur = self._read_reward(tr)
            if self._in_statue(cur, tr, label):
                break
            self.log.event("reward_retry", choice=best.name, attempt=attempt, lit=cur.upgrade_lit,
                           statue=cur.statue)
        else:
            self.log.event("reward_failed", choice=best.name, reason="the tile did not go into the statue",
                           statue=cur.statue, lit=cur.upgrade_lit)
            return False
        for attempt in (1, 2):
            self.click(*reward_screen.UPGRADE_BUTTON, frame, screen="reward", reward="UPGRADE", attempt=attempt)
            if self._gone("shrine", tr):
                return True
            frame = self.gw.grab()
        return False

    # ------------------------------------------------------------------ items
    @property
    def last_level(self) -> str:
        """The run's final level: Stage 1's boss with --stage1, else FULL_LAST_STAGE's boss."""
        return level_name(1, 2) if self.stage1 else level_name(FULL_LAST_STAGE, 99)

    def _pre_plan_item(self, p, s, tries: dict):
        """Items whose value does not depend on this turn's plan (Mass Poison). Edamame Brew
        depends on the plan since 0.24: see edamame_reason."""
        have = {it["item"] for it in p.items if not it.get("dim")}     # a dimmed item cannot be used
        if "skull_card" in have and tries.get("skull_card", 0) < 2:
            # Mass Poison: every enemy, 1 damage per turn for 3 turns [catalogue]
            value = sum(min(e.hp, POISON_TICK * POISON_TURNS) for e in s.alive())
            if value >= POISON_MIN_VALUE or any(e.kind in BOSS_KINDS for e in s.alive()):
                return "skull_card", f"poison worth {value} damage"
        return None

    def _confirm_hp_drop(self, read: int, last_hp: int, hp_cap: int, tr, turn: int) -> int:
        """An HP drop counts only when a second frame shows it too, with the whole pip row
        visible [user 2026-10-28]. In run 20261028-132952 (turns 77 and 82) a sword animation
        over the hero hid two pips for one frame: read 8 of 8 pips, then 10 of 10. A false
        drop could trigger the heal, the stall rule or a wrong plan."""
        time.sleep(HP_CONFIRM_S)
        p2 = self._see(self.gw.grab(), tr)
        h2 = p2.hero or {}
        second = h2.get("hp") if h2.get("hp_max") else None
        if second is not None and second < last_hp and h2.get("hp_max") >= hp_cap:
            return second
        self.log.event("hp_flicker", turn=turn, read=read, second_read=second,
                       second_pips=h2.get("hp_max"), kept=last_hp)
        return last_hp

    def _wait_spawns(self, frame, p, tr, turn: int):
        """A unit in its spawn flames hides its sprite, so its facing reads as unknown
        (52 of the 86 wrong facings of new units in 21 runs) [user review of the 0.20
        series]. Read again every SPAWN_POLL_S until the flames clear, at most
        SPAWN_WAIT_S; the game waits for the hero's move meanwhile."""
        t0 = time.time()
        while time.time() - t0 < SPAWN_WAIT_S:
            time.sleep(SPAWN_POLL_S)
            f2 = self.gw.grab()
            p2 = self._see(f2, tr)
            if p2.screen not in ("combat", "boss"):
                break
            frame, p = f2, p2
            if not any(e.get("spawning") for e in p.enemies):
                break
        self.log.event("spawn_wait", turn=turn, seconds=round(time.time() - t0, 2),
                       cleared=not any(e.get("spawning") for e in p.enemies))
        return frame, p

    def _spend_item(self, s, d, tr, have: set) -> tuple[str, str] | None:
        """The run's last level: spend each held item at its first useful moment
        [user 2026-10-28]. One item per pass; the board is read again after each. Edamame
        Brew is not spent here, only by edamame_reason [user 2026-10-29]. Kami Brew is not spent
        here either: it is kept for a turn where every move takes damage (_damage_item) [user review
        of run 20261029-124921: spent at t161 on an armed Iwao, the unavoidable hits came at t172
        and t192; the same in the 0.24 series]."""
        hx = s.hero.x
        active = [e for e in s.alive() if e.frozen <= 0]
        armed = [e for e in active if e.armed]
        if "ice_card" in have and any(e.armed or abs(e.x - hx) == 1 for e in active):
            return "ice_card", "last level: an enemy is armed or adjacent"
        if "dice_card" in have and any(e.intent == "fire" for e in active):
            return "dice_card", "last level: an enemy telegraphs fire"
        if "mirror_card" in have and any(e.facing == (1 if hx > e.x else -1) for e in armed):
            return "mirror_card", "last level: an armed enemy faces you"
        if "white_potion" in have and self._best_tile_recharging(s):
            return "white_potion", "last level: the best tile is recharging"
        if "curse_card" in have and d.action == "attack" and self._attack_hits_boss(s):
            return "curse_card", "last level: the attack hits the boss"
        return None

    def _damage_item(self, s, have: set) -> str | None:
        """Every move takes damage this turn: freeze everyone (Mass Ice), turn everyone around
        (Rain of Mirrors) if that removes the damage, reroll every enemy's intent (Lucky Die; the
        new telegraphs are read off the screen), or shield (Kami Brew, not while a shield is up);
        the board is then read again and re-planned with the effect in the model."""
        if "ice_card" in have:
            return "ice_card"
        if "mirror_card" in have and self._mirror_saves(s):
            return "mirror_card"
        if "dice_card" in have:
            return "dice_card"
        if "blue_potion" in have and not s.hero.shield:
            return "blue_potion"
        return None

    @staticmethod
    def _best_tile_recharging(s) -> bool:
        """Cool Up is worth most when the strongest tile (catalogue ready value) is empty."""
        h = s.hero
        if not h.kit.ready_values:
            return False
        label, cd, _ = max(h.kit.ready_values, key=lambda t: t[2])
        return label not in h.queue and h.charge.get(label, 0) < cd

    @staticmethod
    def _attack_hits_boss(s) -> bool:
        from .rules import step
        out = max(step(s, "attack"), key=lambda t: t[0])[2]
        return any(k in BOSS_KINDS and dmg > 0 for k, _, dmg, _ in out.enemy_damage)

    @staticmethod
    def _mirror_saves(s) -> bool:
        """Would turning every enemy around leave a zero-damage move?"""
        m = s.clone()
        for e in m.enemies:
            e.facing = -e.facing
        d = plan(m, 2)
        return min(c["exp_damage"] for c in d.candidates) < 0.5

    @staticmethod
    def _curse_gain(s) -> bool:
        """Does doubling the coming hit kill more, or hit the boss harder?"""
        from .rules import step
        def result(cursed: bool):
            m = s.clone()
            for e in m.enemies:
                e.cursed = cursed
            out = max(step(m, "attack"), key=lambda t: t[0])[2]
            return len(out.kills), sum(d for k, _, d, _ in out.enemy_damage if k in BOSS_KINDS)
        k0, b0 = result(False)
        k1, b1 = result(True)
        return k1 > k0 or b1 >= b0 + 2

    def _use_item(self, item: str, p, s, frame, turn: int, reason: str, tr, tries: dict) -> None:
        slot = next(it for it in p.items if it["item"] == item and not it.get("dim"))
        tries[item] = tries.get(item, 0) + 1
        cons = CAT.consumable_by_screen(item)
        name = cons.name if cons else item
        self.say(f"   using {name}: {reason}")
        self.log.event("item_use", item=item, name=name, reason=reason, hp=s.hero.hp, turn=turn, attempt=tries[item])
        self.click(slot["fx"], slot["fy"], frame, item=item)
        time.sleep(1.0)
        after = self._see(self.gw.grab_stable(settle_s=0.2, timeout_s=2.0))
        still = sum(1 for it in after.items if it["item"] == item) >= sum(1 for it in p.items if it["item"] == item)
        self.log.event("item_result", item=item, consumed=not still)
        if not still:
            tries[item] = 0
            # the effect goes into the model only once the item is really gone
            if item == "ice_card":
                tr.freeze(ICE_TURNS)  # Mass Ice: all enemies frozen for 3 turns [catalogue]
            elif item == "blue_potion":
                tr.shield = True  # Kami Brew: negates the next attack [wiki]
            elif item == "curse_card":
                tr.cursed = True  # Mass Curse: next hit on each enemy is doubled [wiki]
            elif item == "mirror_card":
                tr.mirror()       # Rain of Mirrors: every enemy turned around [wiki]
            # Lucky Die (dice_card) rerolls every enemy's intent [wiki] [user 2026-10-28]: nothing
            # to model, the next frame shows the new telegraphs
        else:
            self.say(f"   {item} still in its slot; the click may not have registered")

    # ------------------------------------------------------------------ series
    def _give_up(self, wait_map: bool = True) -> bool:
        """End a won run so the next one can start [user 2026-10-28], or a run found in
        progress when a run starts (0.24) [user 2026-10-29]: after the boss the
        game goes on to Stage 2, so the bot opens PAUSE MENU (the bottom of the three
        round buttons, bottom right), picks GIVE UP, then YES: GIVE UP. From the Stage 2
        map that leads straight to the camp [run 20261028-130456, give_up_failed.png];
        given up in a fight it leads to the death screen [captures 2026-10-28]. The next
        run starts from either (camp: walk to GO; death screen: RESTART). Each click waits
        until its screen is recognised; on any doubt the bot stops and leaves the game as
        it is. True when the camp, the title or the death screen shows."""
        gw, log = self.gw, self.log
        t0 = time.time()
        log.event("give_up", step="start")
        self.live.status("give_up", step="start")
        self.say("giving this run up from the pause menu")
        # 1. a settled screen with the pause button: the map, or any after half the wait (a run
        #    start does not wait: wait_map False). The pause menu or its confirmation may already
        #    be open (a run start that finds the game paused): the pause button is hidden then
        #    [collect/20261028-121900-giveup]
        m = c = None
        esc_sent = False
        while True:
            if _stop_requested():
                return self._give_up_failed("stopped (F8)", None)
            frame = gw.grab_stable(settle_s=0.3, timeout_s=3.0)
            f = normalize(frame)
            if not esc_sent and map_screen.keyboard_mode(f):
                # keyboard-only mode hides the pause button; Esc opens the menu [obs 2026-10-29]
                self.key("esc", frame, screen="give_up", step="esc")
                esc_sent = True
                time.sleep(1.0)
                continue
            if screens.pause_button(f):
                if not wait_map or screens.classify(f) == "map" or time.time() - t0 > GIVE_UP_WAIT_S / 2:
                    break
            else:
                open_menu = screens.menu(f)
                if open_menu is not None:
                    m, c = (open_menu, None) if open_menu[0] == "pause" else (None, open_menu)
                    log.event("give_up", step="menu already open", menu=open_menu[0])
                    break
            if time.time() - t0 > GIVE_UP_WAIT_S:
                return self._give_up_failed("pause button not found", frame)
            time.sleep(0.5)
        # 2. open the pause menu (the click is repeated if the menu does not show)
        if m is None and c is None:
            for attempt in range(3):
                self.click(*screens.PAUSE_BUTTON, frame, away=False, screen="give_up", step="pause")
                time.sleep(1.0)
                frame = gw.grab_stable(settle_s=0.3, timeout_s=3.0)
                m = screens.menu(normalize(frame))
                if m is not None:
                    break
            if m is None or m[0] != "pause":
                return self._give_up_failed(f"pause menu not recognised ({m[0] if m else 'nothing'})", frame)
        # 3. GIVE UP, the third of five rows
        if c is None:
            for attempt in range(2):
                self.click(*m[1][2], frame, away=False, screen="give_up", step="give_up")
                time.sleep(1.0)
                frame = gw.grab_stable(settle_s=0.3, timeout_s=3.0)
                c = screens.menu(normalize(frame))
                if c is None or c[0] != "pause":
                    break
                m = c
            if c is None or c[0] != "confirm":
                return self._give_up_failed(f"confirmation not recognised ({c[0] if c else 'nothing'})", frame)
        # 4. YES: GIVE UP, the second of two rows; then wait for where the game goes:
        #    the camp, the title, or the death screen
        self.click(*c[1][1], frame, away=True, screen="give_up", step="yes")
        clicked_again = False
        while time.time() - t0 < GIVE_UP_WAIT_S + 30:
            time.sleep(0.5)
            frame = gw.grab()
            f = normalize(frame)
            where = "death" if screens.find_restart(f) is not None else screens.classify(f)
            if where in ("camp", "title", "death"):
                log.event("give_up", step="done", screen=where, seconds=round(time.time() - t0, 1))
                log.save_frame(frame, "gave_up")
                return True
            c = screens.menu(f)
            if c is not None and c[0] == "confirm" and not clicked_again:
                time.sleep(1.0)
                self.click(*c[1][1], frame, away=True, screen="give_up", step="yes_again")
                clicked_again = True
        return self._give_up_failed("no camp, title or death screen after YES: GIVE UP", frame)

    def _give_up_failed(self, reason: str, frame) -> bool:
        self.log.event("give_up_failed", reason=reason)
        if frame is not None:
            self.log.save_frame(frame, "give_up_failed")
        self.say(f"   could not give the run up: {reason}; stopping the series here")
        return False

    # ------------------------------------------------------------------ shop
    def _shop(self, frame, tr=None, hp: int | None = None) -> bool:
        """The shop between regions [user 2026-10-29: read the offers and prices]: what
        perception/shop_screen.py reads (the statue's upgrade and price, the counter's prices, the
        coins) is logged, with each coin offer on the counter named by its tooltip (0.32); the purchases
        follow your order (_shop_buy) [user review of run 20261029-154200; of run -184142, note 2]; then
        GO. Without a tracker (tests of 0.25-0.28) it only reads and leaves. True once the shop has gone."""
        self.shop_n += 1
        self._park_mouse()                    # a hover tooltip can cover a price [f0175]
        time.sleep(0.3)
        frame = self.gw.grab_stable(settle_s=0.25, timeout_s=3.0)
        hand: list = []
        try:
            f = normalize(frame)
            offer = shop_screen.read(f)
            if tr is not None:
                hand = reward_screen.read_hand(f, tuple(set(t.tile for t in tr.kit) | BASE_POOL), dots=True)
        except Exception as e:                # reading must never keep the bot in the shop
            self.log.exception("shop_read", e, frame=frame)
            offer = {}
        self.log.save_frame(frame, f"shop_{self.shop_n}")
        if tr is not None and offer:
            try:
                offer["goods"] = self._shop_goods(frame, offer)
            except Exception as e:            # an unread counter is left alone
                self.log.exception("shop_goods", e, frame=frame)
                offer["goods"] = []
        self.log.event("shop", n=self.shop_n, **offer)
        self.live.status("shop", **{k: offer.get(k) for k in ("upgrade", "coins")})
        self.say(f"shop: {shop_screen.describe(offer)}")
        if tr is not None and offer:
            try:
                frame = self._shop_buy(frame, offer, hand, tr, hp if hp is not None else HERO_MAX_HP)
            except Exception as e:            # buying must never keep the bot in the shop
                self.log.exception("shop_buy", e, frame=frame)
        self.say("shop: GO")
        for attempt in range(1, SHOP_GO_TRIES + 1):
            self.click(*screens.SHOP_GO, frame, screen="shop", attempt=attempt)
            time.sleep(1.5)
            frame = self.gw.grab_stable(settle_s=0.25, timeout_s=3.0)
            if not screens.shop(normalize(frame)):
                return True
        self.log.save_frame(frame, "shop_stuck")
        return False

    def _shop_goods(self, frame, offer: dict) -> list[dict]:
        """Each coin offer on the counter with what its tooltip names: the cursor rests on the slot
        until the tooltip shows [rec2: the user's cursor on the counter, f0177-f1753], then is parked
        again. {'x', 'price', 'currency', 'title', 'kind', 'id'}; kind 'skill', 'consumable' or 'other'
        (None when no tooltip was read). Each tooltip frame is saved, so new titles can be learned."""
        goods = []
        for i, o in enumerate(offer.get("counter") or []):
            if o.get("currency") != "coin" or o.get("price") is None:
                continue                      # padlocks (skulls) and blood services (hearts) are not bought
            tip, fr = None, frame
            for attempt in range(HOVER_TRIES):
                self.hover(o["x"] / 1920, shop_screen.SLOT_Y / 1080, fr, screen="shop", slot=i + 1, attempt=attempt + 1)
                time.sleep(HOVER_S)
                fr = self.gw.grab()
                tip = shop_screen.tooltip(normalize(fr))
                if tip is not None and not shop_screen.tip_over(tip, o["x"]):
                    tip = None                # another slot's tooltip, still fading
                if tip is not None:
                    break
            self.log.save_frame(fr, f"shop_{self.shop_n}_offer_{i + 1}")
            g = dict(o, title=tip and tip["title"], kind=tip and tip["kind"], id=tip and tip["id"])
            goods.append(g)
        if goods:
            self._park_mouse()
            time.sleep(0.3)
        return goods

    def _shop_look(self, tr):
        """The shop again after a click: the frame, and the hand, the statue's mouth, UPGRADE and
        the coins as a reward-screen-like record for _in_statue."""
        frame = self.gw.grab_stable(settle_s=0.25, timeout_s=3.0)
        f = normalize(frame)
        tiles = tuple(set(t.tile for t in tr.kit) | BASE_POOL)
        look = types.SimpleNamespace(hand=reward_screen.read_hand(f, tiles, dots=True),
                                     statue=shop_screen.statue_tile(f, tiles), upgrade_lit=shop_screen.upgrade_lit(f),
                                     coins=shop_screen.number_in(f, shop_screen.COINS_BOX))
        return frame, look

    def _shop_buy(self, frame, offer: dict, hand: list, tr, hp: int):
        """The shop's purchases in your order [user 2026-10-29: -2 cooldown and +1 damage upgrades,
        Healthy, +1 upgrade slot]: the statue's upgrade when it adds damage or takes cooldown off, then
        Healthy, a slot-only statue upgrade, the other simulated skills, then trade-offs
        (rewards.shop_priority); each is valued against keeping the coins when its turn comes
        (rewards.decide_shop, decide_skill), with the coins read again after every purchase.
        Returns the last frame."""
        notes: list = []
        if hand:
            tr.kit = kit_from_faces(tr.kit, hand, notes)
        steps = []
        pri = rewards.shop_priority(upgrade_id=offer.get("upgrade")) if offer.get("upgrade") else None
        steps.append((pri if pri is not None else 9, 0, None))     # the statue: logged even when never bought
        for i, g in enumerate(offer.get("goods") or []):
            pri = rewards.shop_priority(good=g)
            if pri is not None:
                steps.append((pri, i + 1, g))
            elif g.get("title"):
                notes.append(f"counter: {g['title']} for {g['price']} {g['currency']}s is not bought "
                             f"({g.get('kind') or 'unread'}{', not simulated' if g.get('kind') == 'skill' else ''})")
        coins = offer.get("coins")
        for _, _, g in sorted(steps, key=lambda t: (t[0], t[1])):
            if g is None:
                frame, coins = self._shop_statue(frame, offer, hand, tr, hp, coins, notes)
            else:
                frame, coins = self._shop_skill(frame, g, tr, hp, coins)
            hand = []                         # read again by the statue's flow when it comes later
        return frame

    def _valuation_feed(self):
        printed = [time.time()]

        def feed(st):
            self.live.valuation(st)
            if st.get("phase") == "running" and time.time() - printed[0] >= CONSOLE_FEED_S:
                printed[0] = time.time()
                self.say("   " + rewards.progress_line(st))
        return feed

    def _shop_statue(self, frame, offer: dict, hand: list, tr, hp: int, coins, notes: list):
        """Value the statue's upgrade against keeping the coins (rewards.decide_shop) and buy it when
        that wins: the tile into the statue's mouth (checked as at a shrine), then UPGRADE, checked by
        the coins dropping [collect/20261028-190026 f0172-f0178]. A tile left in the mouth after a
        failed purchase is clicked back into the hand. Returns the last frame and the coins."""
        up, price = offer.get("upgrade"), offer.get("upgrade_price")
        name = CAT.upgrades[up].name if up in CAT.upgrades else (up or "statue upgrade not read")
        self.live.valuation({"layout": "shop", "upgrade": up, "price": price, "coins": coins, "hp": hp,
                             "upgrade_text": f"{name} for {price if price is not None else '?'} coins "
                                             f"({coins if coins is not None else '?'} held)",
                             "kit": [f"{t.label} {t.damage}/{t.cooldown}" for t in tr.kit], "phase": "starting"},
                            reset=True)
        dec = rewards.decide_shop(tr.kit, up, price, coins, hp, progress=self._valuation_feed())
        best, opts, st = dec.best, dec.options, dec.stats or {}
        self.log.event("shop_decision", upgrade=up, price=price, coins=coins, hp=hp, kit=tr.kit.to_list(), notes=notes,
                       seconds=dec.seconds, choice=best.name, rule=dec.rule, options=[o.summary() for o in opts],
                       fights=st.get("fights"), cached=st.get("cached"), seeds=st.get("seeds"), seeds_max=st.get("of"))
        scored = [o for o in opts if o.scores]
        self.say("   " + (", ".join(f"{o.name} {o.value:+.2f} ({len(o.scores)})"
                                    for o in sorted(scored, key=lambda o: -o.value)) or "no simulation needed")
                 + f" -> {'keep the coins' if best.name == 'skip' else 'buy ' + best.name}  [{dec.rule}]")
        self.live.valuation({"phase": "decided", "choice": best.name, "rule": dec.rule, "seconds": dec.seconds,
                             "click": "GO" if best.name == "skip" else f"{best.detail['label']} into the statue, "
                                                                        f"then UPGRADE ({price} coins)"})
        if best.name == "skip":
            self.live.valuation({"phase": "done", "outcome": "kept the coins"})
            return frame, coins
        label = best.detail["label"]
        if not hand:
            frame, cur = self._shop_look(tr)
            hand = cur.hand
        cur = types.SimpleNamespace(hand=hand, statue=None, upgrade_lit=False, coins=coins)
        for attempt in (1, 2, 3):
            face = match_faces(tr.kit, cur.hand).get(label)
            if face is None:
                self.log.event("shop_buy_failed", choice=best.name, reason="tile not found in hand", attempt=attempt)
                return self._shop_clear(frame, tr), coins
            self.click(face["fx"], face["fy"], frame, screen="shop", shop=best.name, attempt=attempt)
            time.sleep(0.8)
            self._park_mouse()
            frame, cur = self._shop_look(tr)
            if self._in_statue(cur, tr, label):
                break
            self.log.event("shop_retry", choice=best.name, attempt=attempt, lit=cur.upgrade_lit, statue=cur.statue)
            frame = self._shop_clear(frame, tr)      # a wrong tile in the mouth goes back first
            frame, cur = self._shop_look(tr)
        else:
            self.log.event("shop_buy_failed", choice=best.name, reason="the tile did not go into the statue")
            return self._shop_clear(frame, tr), coins
        for attempt in (1, 2):
            self.click(*shop_screen.UPGRADE_BUTTON, frame, screen="shop", shop="UPGRADE", attempt=attempt)
            time.sleep(1.2)
            self._park_mouse()
            frame, cur = self._shop_look(tr)
            if cur.coins is not None and coins is not None and cur.coins < coins:
                tr.set_kit(best.kit)
                self.log.event("shop_bought", choice=best.name, upgrade=up, price=price, coins_before=coins,
                               coins_after=cur.coins, kit=best.kit.to_list())
                self.say(f"   bought {CAT.upgrades[up].name} on {label} for {coins - cur.coins} coins; kit {best.kit}")
                self.live.valuation({"phase": "done", "outcome": f"bought {best.name}"})
                return frame, cur.coins
        self.log.event("shop_buy_failed", choice=best.name, reason="the coins did not drop after UPGRADE",
                       coins_before=coins, coins_after=cur.coins)
        self.live.valuation({"phase": "done", "outcome": "purchase failed"})
        return self._shop_clear(frame, tr), coins

    def _shop_skill(self, frame, g: dict, tr, hp: int, coins):
        """A skill on the counter, valued against keeping the coins (rewards.decide_skill) and bought
        with one click on its slot, checked by the coins dropping [rec2: f0181 -> f0182 Combo Recharge,
        31 -> 11 coins; f0447, f0449, f0458, f0460, f0786, f0791 the same]. Returns the last frame and
        the coins."""
        sid, price = g.get("id"), g.get("price")
        name = CAT.skill(sid).name if sid in CAT.skills else (g.get("title") or "?")
        self.live.valuation({"layout": "shop", "upgrade": sid, "price": price, "coins": coins, "hp": hp,
                             "upgrade_text": f"{name} (skill) for {price} coins ({coins if coins is not None else '?'} held)",
                             "kit": [f"{t.label} {t.damage}/{t.cooldown}" for t in tr.kit], "phase": "starting"},
                            reset=True)
        dec = rewards.decide_skill(tr.kit, sid, price, coins, hp, progress=self._valuation_feed())
        best, opts, st = dec.best, dec.options, dec.stats or {}
        self.log.event("shop_decision", skill=sid, price=price, coins=coins, hp=hp, kit=tr.kit.to_list(),
                       seconds=dec.seconds, choice=best.name, rule=dec.rule, options=[o.summary() for o in opts],
                       fights=st.get("fights"), cached=st.get("cached"), seeds=st.get("seeds"), seeds_max=st.get("of"))
        self.say(f"   {name}: {'keep the coins' if best.name == 'skip' else 'buy it'}  [{dec.rule}]")
        self.live.valuation({"phase": "decided", "choice": best.name, "rule": dec.rule, "seconds": dec.seconds,
                             "click": "nothing" if best.name == "skip" else f"the counter's slot ({price} coins)"})
        if best.name == "skip":
            self.live.valuation({"phase": "done", "outcome": "kept the coins"})
            return frame, coins
        for attempt in (1, 2):
            self.click(g["x"] / 1920, shop_screen.SLOT_Y / 1080, frame, screen="shop", shop=best.name, attempt=attempt)
            time.sleep(1.0)
            self._park_mouse()
            frame = self.gw.grab_stable(settle_s=0.25, timeout_s=3.0)
            now = shop_screen.number_in(normalize(frame), shop_screen.COINS_BOX)
            if now is not None and coins is not None and now < coins:
                tr.set_kit(best.kit)
                self.log.event("shop_bought", choice=best.name, skill=sid, price=price, coins_before=coins,
                               coins_after=now, kit=best.kit.to_list())
                self.say(f"   bought {name} for {coins - now} coins; skills {tr.kit.skills}")
                self.live.valuation({"phase": "done", "outcome": f"bought {name}"})
                return frame, now
        self.log.event("shop_buy_failed", choice=best.name, reason="the coins did not drop after the click",
                       coins_before=coins, coins_after=now)
        self.live.valuation({"phase": "done", "outcome": "purchase failed"})
        return frame, coins if now is None else now

    def _shop_clear(self, frame, tr):
        """A tile still in the statue's mouth (UPGRADE lit) goes back to the hand: click it [f0174]."""
        frame, cur = self._shop_look(tr)
        if cur.upgrade_lit:
            x, y = shop_screen.STATUE_FACE
            self.click((x + FACE_W / 2) / 1920, (y + FACE_H / 2) / 1080, frame,
                       screen="shop", shop="tile back to the hand")
            time.sleep(0.8)
            self._park_mouse()
            frame = self.gw.grab_stable(settle_s=0.25, timeout_s=3.0)
        return frame

    def _start_elsewhere(self, scr: str, p, frame) -> str | None:
        """A run start that finds the game anywhere but the camp or the title [user 2026-10-29]:
        'wait' (the last run's hero dying; its death screen follows), 'gave_up' (the run in
        progress was given up), or None when that failed."""
        if scr in ("combat", "boss") and p.hero and p.hero.get("hp_max") and p.hero.get("hp", 1) <= 0:
            time.sleep(0.5)
            return "wait"
        self.say(f"run start: the game shows the {scr} of an earlier run; giving it up to reach the camp")
        self.log.event("start_elsewhere", screen=scr)
        self.log.save_frame(frame, f"start_{scr}")
        return "gave_up" if self._give_up(wait_map=False) else None

    # ------------------------------------------------------------------ map (M4)
    def _map_step(self, frame, look: dict) -> None:
        """One input on the map between regions [user 2026-10-29]: turn keyboard-only mode on,
        then read the marked node (perception/map_screen.py). A shop or a region in MAP_REGIONS is
        entered with Enter; another region makes the bot look at the other reachable nodes first
        (down, then up when down did not move the mark), at most MAP_LOOKS_MAX presses, after which
        it takes the mark. look keeps that state for one map visit."""
        f = normalize(frame)
        if not map_screen.keyboard_mode(f):
            self.say("map: keyboard-only mode on")
            self.log.event("map", step="keyboard on")
            self.click(*map_screen.KEYBOARD_BUTTON, frame, screen="map", step="keyboard")
            time.sleep(0.8)
            return
        sel = map_screen.selected(f)
        if sel is None:
            look["unread"] = look.get("unread", 0) + 1
            self.log.event("map", step="no node read", tries=look["unread"])
            self.log.save_frame(frame, f"map_unread_{look['unread']}")
            if look["unread"] >= 3:      # a mark the reader cannot name: take it
                self._map_travel(frame, None, look, "the marked node could not be read")
            else:
                time.sleep(0.6)
            return
        seen = look.setdefault("seen", [])
        if sel["text"] not in seen:
            seen.append(sel["text"])
        if sel["kind"] == "shop":
            self._map_travel(frame, sel, look, "a shop (every shop led on to both regions)")
            return
        if sel["region"] in MAP_REGIONS:
            self._map_travel(frame, sel, look, f"region {sel['region']}, the preferred one")
            return
        if look.get("looks", 0) >= MAP_LOOKS_MAX:
            self._map_travel(frame, sel, look, f"no region of {list(MAP_REGIONS)} reachable")
            return
        key = "up" if look.get("bottom") else "down"
        self.log.event("map", step="look", key=key, node=sel["text"], region=sel["region"])
        self.key(key, frame, screen="map", step="look")
        look["looks"] = look.get("looks", 0) + 1
        time.sleep(0.6)
        after = map_screen.selected(normalize(self.gw.grab()))
        if key == "down" and after is not None and after["text"] == sel["text"]:
            look["bottom"] = True        # down did not move the mark: look upward next

    def _map_travel(self, frame, sel, look: dict, why: str) -> None:
        name = sel["text"] if sel else None
        self.say(f"map: travel to {name or 'the marked node'} ({why})")
        # RunLogger.event's first parameter is called 'kind': the node's goes in as 'node_kind'
        self.log.event("map_travel", node=name, node_kind=sel and sel["kind"], region=sel and sel["region"],
                       why=why, seen=look.get("seen", []))
        self.log.save_frame(frame, "map_travel")
        if sel and sel["kind"] == "region" and sel["region"] not in self.regions:
            if not isinstance(self.regions, list):
                self.regions = list(self.regions)
            self.regions.append(sel["region"])   # its enemies can appear from now on; the tracker counts its levels
        self.key("enter", frame, screen="map", step="travel")
        look.clear()
        time.sleep(2.5)

    def _mouse_mode(self, frame, scr: str) -> None:
        """Keyboard-only mode left on by the map: a click on an empty spot turns it off, so the
        clicks of the other screens work as before [obs 2026-10-29: any click turns it off]."""
        self.log.event("mouse_mode", screen=scr)
        self.click(*MOUSE_MODE_SPOT, frame, screen=scr, step="mouse mode")
        time.sleep(0.5)

    # ------------------------------------------------------------------ run
    def run(self) -> str:
        gw, log = self.gw, self.log
        # capture reads the primary display only: a game window on another monitor is moved there
        # first [user review of run 20261029-124921: run 124858 stopped with a capture error]
        moved, move_error = None, None
        if hasattr(gw, "ensure_on_primary"):
            try:
                moved = gw.ensure_on_primary()
            except Exception as e:          # the capture says it plainly if the window is still off
                move_error = repr(e)
        gw.focus()
        run_id = log.start_run({"bot_version": VERSION, "weights": WEIGHTS,
                                "autostart": self.autostart, "kit": DEFAULT_KIT.to_list()})
        if moved:
            log.event("window_moved", before=list(moved[0]), after=list(moved[1]))
            self.say(f"moved the game window onto the primary display: {moved[0]} -> {moved[1]}")
        if move_error:
            log.event("window_move_failed", error=move_error)
            self.say(f"could not move the game window onto the primary display: {move_error}")
        if hasattr(gw, "notify"):
            # during the run every capture checks the window [run 20261029-153952: it left the primary
            # display at turn 14 of Bamboo Grove I and the capture stopped the run]: winio puts it back
            # (or waits for it) and the moves are logged here
            def on_window(event: str, **fields) -> None:
                log.event(event, **fields)
                self.say(f"{event.replace('_', ' ')}: {fields}")
            gw.notify, gw.should_stop = on_window, _stop_requested
        self.rec.start(log.run_dir / "frames")
        if "live" not in self.__dict__:      # a Bot built without __init__ (tests): its own viewer feed
            self.live = LiveView()
        self.live.start(log.run_dir, run_id, VERSION)
        self.say(f"run {run_id} started; recording to runs/{run_id}/frames; hold F8 to stop")
        tr = Tracker()
        self.reward_n = self.shop_n = 0
        self.regions, self.day = ["bamboo_grove"], None
        tr.route = self.regions               # the same list: the map appends each region travelled to
        self.gave_up = None
        death_since = None
        progress, last_progress = None, 0     # stall rule: see planner.stall_weights
        hp_cap = HERO_MAX_HP                  # most HP pips seen in one read this run
        turn = 0
        level = None
        final = self.last_level
        seen_final = False                    # the run's final boss level was reached
        started = not self.start_at_camp      # the camp (or the title) was seen [user 2026-10-29]
        start_give_ups = 0
        waves = Waves()
        last_hp = 10
        unknown_since = None
        unknown_saved = False
        camp_steps = 0
        day_clicks, day_done = 0, not LOWEST_DAY
        map_tries, map_look = 0, {}
        zero_hp = 0
        last_key, last_action = None, None
        potion_check = None
        item_tries: dict = {}
        seen_items: set = set()
        potion_failed: set = set()
        banned: dict = {}
        no_change = 0                         # inputs in a row that changed nothing on screen
        result = "stopped"

        try:
            while True:
                if _stop_requested():
                    result = "stopped"
                    break
                frame = gw.grab_stable(settle_s=0.2, timeout_s=3.0)
                t0 = time.time()
                p = self._see(frame, tr)
                perceive_ms = (time.time() - t0) * 1000
                scr = p.screen

                if scr != "unknown":
                    unknown_since, unknown_saved = None, False
                if scr not in ("combat", "boss"):
                    self.live.status({"unknown": "waiting"}.get(scr, scr))
                if scr not in ("map", "black") and map_screen.keyboard_mode(normalize(frame)):
                    self._mouse_mode(frame, scr)      # the map's keyboard mode is still on
                    continue

                # ---------------- run start [user 2026-10-29]: every run starts at the camp, where
                # Day 1 is chosen. Run 20261029-072042 began on the Stage 2 map a single run had
                # left (a lone run is not given up), repeated its Stage 1 map inputs 18 times and
                # read the shop as a fight. Anywhere else the run in progress is given up first;
                # the death screen goes BACK TO CAMP (RESTART would keep the last run's day).
                if not started:
                    if scr in ("camp", "title"):
                        started = True
                    elif scr == "death":
                        death_since = death_since or time.time()
                        if time.time() - death_since > DEATH_SCREEN_MAX_S:
                            result = "stuck"           # BACK TO CAMP unreadable or not reacting
                            log.save_frame(frame, "stuck_death_screen")
                            break
                        btn = screens.death_buttons(normalize(frame))
                        if btn is not None:
                            self.say("death screen: BACK TO CAMP")
                            log.event("menu", screen="death", back_to_camp=[round(v, 4) for v in btn[1]])
                            self.click(*btn[1], frame, screen="death")
                            time.sleep(3.0)
                        else:
                            time.sleep(0.5)            # the buttons fade in after the kanji
                        continue
                    elif scr in ("map", "shop", "reward", "combat", "boss"):
                        if start_give_ups >= START_GIVE_UPS_MAX:
                            result = "stuck"
                            log.save_frame(frame, "stuck_start")
                            break
                        how = self._start_elsewhere(scr, p, frame)
                        if how is None:
                            result = "stuck"
                            break
                        start_give_ups += how == "gave_up"
                        continue
                    # 'unknown' and 'black' wait below

                # ---------------- rewards (M2: taken when the planner values them above skipping)
                if scr == "reward":
                    log.save_frame(frame, "win" if seen_final else f"reward_seen_{self.reward_n + 1}")
                    log.level_end(last_hp, "cleared")      # a reward screen ends the level
                    try:
                        self._reward(frame, tr, level or "bamboo-1", last_hp)
                    except Exception as e:   # a reward must never end the run: fall back to SKIP
                        log.exception("reward", e, frame=frame)
                        self.say(f"   reward handling failed ({e!r}); SKIP")
                        frame = gw.grab()
                        pos = screens.find_skip(normalize(frame))
                        if pos is not None:
                            self.click(*pos, frame, screen="reward")
                            time.sleep(1.0)
                    # the next fight is the next level [user review of run 20261029-072323: the
                    # boss shrine was logged as a new level 'bamboo-2' after Rei's 7-cell arena]
                    tr.next_level()
                    if seen_final:
                        result = "win"   # the reward after the final boss ends the run
                        break
                    continue

                # ---------------- end conditions
                if seen_final and scr == "map":
                    result = "win"
                    log.save_frame(frame, "win")
                    break
                if scr == "shop":
                    if not self._shop(frame, tr, last_hp):
                        self.say("shop: GO did not leave the shop; stopping")
                        result = "stuck"
                        break
                    map_tries, map_look = 0, {}       # the next map is a new one
                    continue
                if scr == "death" and turn == 0:
                    # the last run's death screen (a death, or a run given up in a fight): RESTART
                    # starts this one [run 20261027-182223 ended after 1 s on that screen]
                    if not self.autostart:           # the player restarts
                        time.sleep(0.5)
                        continue
                    death_since = death_since or time.time()
                    if time.time() - death_since > DEATH_SCREEN_MAX_S:
                        # still here after this long: RESTART unreadable or not reacting
                        result = "stuck"
                        log.save_frame(frame, "stuck_death_screen")
                        break
                    pos = screens.find_restart(normalize(frame))
                    if pos is not None:
                        self.say("death screen: RESTART")
                        log.event("menu", screen="death", restart=[round(v, 4) for v in pos])
                        self.click(*pos, frame, screen="death")
                        time.sleep(3.0)
                    else:
                        time.sleep(0.5)          # the buttons fade in after the kanji
                    continue
                if scr == "death":
                    result = "death"
                    log.save_frame(frame, "death")
                    break
                if scr in ("title", "camp", "map", "black"):
                    if not self.autostart or scr == "black":
                        time.sleep(0.5)
                        continue
                    if scr == "title":
                        self.say("title: NEW RUN")
                        log.event("menu", screen="title")
                        # 'PRESS ANY BUTTON' splash: click the logo (harmless on the menu,
                        # where Space could pick the highlighted CONTINUE)
                        self.click(0.5, 0.3, frame, away=False, screen="title-splash")
                        time.sleep(1.0)
                        frame = gw.grab()
                        self.click(*screens.TITLE_NEW_RUN, frame, screen="title")
                        time.sleep(2.0)
                    elif scr == "camp":
                        sel = None if day_done else screens.day_selector(normalize(frame))
                        if sel is not None and sel["left"] and sel["elite"]:
                            # an earlier day exists and this one has elites: step down before GO
                            # [user 2026-10-29]. Day 1 is the first day with the left arrow dark or
                            # without the red elite rule
                            if day_clicks >= DAY_CLICKS_MAX:
                                self.say("camp: the day selector does not go down to Day 1; stopping")
                                log.event("menu", screen="camp", day="not selected", **sel)
                                log.save_frame(frame, "camp_day_not_selected")
                                result = "stuck"
                                break
                            self.say("camp: one day earlier")
                            log.event("menu", screen="camp", day="left", **sel)
                            self.click(*screens.DAY_LEFT, frame, screen="camp", day="left")
                            day_clicks += 1
                            time.sleep(0.8)
                            continue
                        if sel is not None:
                            day_done = True
                            # perception reads without elites only when the selector shows none
                            self.day = None if sel["elite"] else 1
                            self.say("camp: Day 1" + (" (the elite rule still shows)" if sel["elite"] else ""))
                            log.event("day_select", day=1, clicks=day_clicks, **sel)
                            log.save_frame(frame, "camp_day1")
                        go = screens.find_go(frame)
                        if go and camp_steps >= 2:
                            self.say("camp: GO")
                            log.event("menu", screen="camp", go=go)
                            self.click(*go, frame, screen="camp")
                            time.sleep(2.0)
                        elif camp_steps < 6:
                            self.key("d", frame, screen="camp")
                            camp_steps += 1
                            time.sleep(0.6)
                        else:
                            self.say("camp: GO not found, clicking recorded spot")
                            log.event("menu", screen="camp", go="fallback")
                            self.click(*screens.CAMP_GO, frame, screen="camp")
                            time.sleep(2.0)
                    elif scr == "map":
                        if map_tries >= MAP_TRIES_MAX:
                            # [run 20261029-072042] no input left this map: stop instead of looping
                            self.say(f"map: {map_tries} inputs did not leave the map; stopping")
                            log.event("map_stuck", tries=map_tries)
                            log.save_frame(frame, "map_stuck")
                            result = "stuck"
                            break
                        if tr.level_i >= 2 or tr.stage > 1:
                            # the map after a boss: keyboard-only mode (M4) [user 2026-10-29]
                            map_tries += 1
                            self._map_step(frame, map_look)
                            continue
                        # [run 20261027-175511] a click followed by an instant mouse move did not
                        # register; keep the cursor on the node, then try other inputs in turn
                        how = screens.MAP_TRIES[map_tries % len(screens.MAP_TRIES)]
                        self.say(f"map: {how[0]} (try {map_tries + 1})")
                        log.event("menu", screen="map", how=how[0], attempt=map_tries + 1)
                        if map_tries == 0 or map_tries % 4 == 3:
                            log.save_frame(frame, f"map_{map_tries + 1}")
                        if how[0].startswith("key"):
                            self.key(how[1], frame, screen="map")
                        else:
                            self.click(*how[1], frame, away=False, screen="map")
                        map_tries += 1
                        time.sleep(3.0)
                    continue

                if scr == "unknown":
                    now = time.time()
                    if unknown_since is None:
                        unknown_since = now
                    elif not unknown_saved and now - unknown_since > UNKNOWN_FRAME_S:
                        log.save_frame(frame, f"unknown_{int(now)}")
                        log.event("unknown_screen", seconds=round(now - unknown_since, 1), last_hp=last_hp)
                        unknown_saved = True
                    elif now - unknown_since > UNKNOWN_GIVE_UP_S:
                        result = "death?" if last_hp <= 3 else "stuck"
                        log.save_frame(frame, "stuck")
                        break
                    time.sleep(0.4)
                    continue

                # ---------------- combat / boss
                map_tries, map_look = 0, {}
                if p.screen in ("combat", "boss") and any(e.get("spawning") for e in p.enemies):
                    frame, p = self._wait_spawns(frame, p, tr, turn)
                if p.screen in ("combat", "boss") and any(e["queue"] or e["intent"] is None for e in p.enemies):
                    # catch the yellow phase of blinking bubbles: ~1 s of extra frames
                    extra = []
                    for _ in range(BLINK_SAMPLES):
                        time.sleep(BLINK_GAP_S)
                        extra.append(gw.grab())
                    blink_notes = merge_blink(p, extra)
                    if blink_notes:
                        log.event("blink", turn=turn, notes=blink_notes)
                s = tr.state(p)
                if s is None:
                    time.sleep(0.3)
                    continue
                if p.hero["hp_max"]:
                    hp_cap = max(hp_cap, p.hero["hp_max"])
                    if s.hero.hp < last_hp:
                        s.hero.hp = self._confirm_hp_drop(s.hero.hp, last_hp, hp_cap, tr, turn)
                if turn == 0 and p.hero["hp_max"] and s.hero.hp <= 0:
                    # the last run's hero dying before its death screen shows: not this run
                    time.sleep(0.5)
                    continue
                if s.level != level:
                    level = s.level
                    waves.reset()
                    log.level_start(level, s.hero.hp, p.wave[1] if p.wave else LEVEL_WAVES.get(level, 0))
                    log.event("kit", level=level, kit=tr.kit.to_list())
                    self.say(f"level {level}, HP {s.hero.hp}")
                if level == final:
                    seen_final = True
                if level.endswith("boss"):
                    s.wave = 0
                elif s.alive():
                    # the HUD counter, else a rise in living enemies (shogunbot.waves); before
                    # anything that may skip this turn's planning, so no new unit goes unlogged
                    for name, fields in waves.update(p.wave, s.alive(), tr.fresh, turn):
                        if name == "wave_spawn":
                            log.wave_seen(fields.pop("wave"), fields.pop("spawned"), fields.pop("alive_before"),
                                          fields.pop("turn"), **fields)
                        else:
                            log.event(name, **fields)
                    s.wave = waves.wave
                    if waves.total:
                        s.waves_total = waves.total
                else:
                    waves.prev_alive = 0
                if p.hero["hp_max"] and s.hero.hp <= 0:
                    zero_hp += 1
                    if zero_hp >= 2:  # two stable reads, so a hit flash is not a death
                        result = "death"
                        log.save_frame(frame, "death")
                        break
                    time.sleep(0.5)
                    continue
                zero_hp = 0
                if potion_check is not None:
                    hp_before, used = potion_check
                    log.event("potion_result", item=used, hp_before=hp_before, hp_after=s.hero.hp,
                              worked=s.hero.hp > hp_before)
                    self.say(f"   {used}: HP {hp_before} -> {s.hero.hp}")
                    if s.hero.hp <= hp_before:
                        potion_failed.add(used)
                    potion_check = None
                for it in p.items:
                    if it["item"] not in seen_items:
                        # first sighting of each item type: keep a frame to verify the read
                        seen_items.add(it["item"])
                        log.save_frame(frame, f"item_{it['item']}")
                        log.event("item_seen", item=it["item"], fy=round(float(it["fy"]), 3), dim=bool(it.get("dim")))
                # ---- items before planning (all cost no turn [obs]; effects from the wiki)
                spend = level == final
                use = self._pre_plan_item(p, s, item_tries)
                if use:
                    item, reason = use
                    self._use_item(item, p, s, frame, turn, reason, tr, item_tries)
                    continue
                last_hp = s.hero.hp if p.hero["hp_max"] else last_hp
                if not s.alive():
                    time.sleep(0.3)
                    continue
                s.turn = turn
                # stall rule [user 2026-10-28]: after STALL_TURNS turns in which no enemy lost
                # HP, died or appeared, a 1-HP trade that makes progress is allowed
                pk = progress_key(s)
                if pk != progress:
                    progress, last_progress = pk, turn
                stalled = turn - last_progress >= STALL_TURNS

                t1 = time.time()
                try:
                    d = plan_live(s, stall_weights(WEIGHTS, s.hero.hp) if stalled else WEIGHTS, explain=True)
                except Exception as e:  # a bad read must not end the run
                    log.exception("plan", e, frame=frame)
                    self.say(f"   planner error ({e!r}); re-reading the screen")
                    time.sleep(0.5)
                    continue
                plan_ms = (time.time() - t1) * 1000
                # loop breaker [run 20261027-182237: a misread facing made 'right' a
                # blocked move; the board never changed and the bot repeated it ~40 times]
                key = s.key()
                if last_key is not None and key == last_key and (last_action in ("left", "right", "turn")
                                                                 or last_action.startswith("queue:")):
                    banned.setdefault(key, set()).add(last_action)
                    why = "the board did not change"
                    if last_action.startswith("queue:"):
                        # a queued tile leaves the hand: on the same board it is still there, so the game
                        # refused it and the tile is recharging whatever was read [user review of run
                        # 20261030-041652, fix 1: the 4/8 Arrow read 7 of 7 charged, its click was refused
                        # on every turn from t237, and 54 times at Iwao from t143]
                        tr.recharging(last_action.split(":", 1)[1])
                        why = "the tile did not leave the hand: planned as recharging"
                    log.event("loop_break", turn=turn, banned=sorted(banned[key]), board=s.render(), reason=why)
                    self.say(f"   '{last_action}' changed nothing; not repeating it here")
                if d.action in banned.get(key, ()):
                    alt = next((c["action"] for c in d.candidates if c["action"] not in banned[key]), d.action)
                    d.action = alt
                last_key, last_action = key, d.action
                have = {it["item"] for it in p.items if item_tries.get(it["item"], 0) < 2 and not it.get("dim")}
                # every move takes damage: the least damaging one (in stall mode the chosen move
                # may take a 1-HP trade while a safe move exists; that is not 'every move')
                every = min((c["exp_damage"] for c in d.candidates), default=0.0)
                if every >= 1.0:
                    item = self._damage_item(s, have)
                    if item:
                        self._use_item(item, p, s, frame, turn,
                                       f"expected {every:.1f} damage on every move", tr, item_tries)
                        continue
                if "green_potion" in have:
                    # [user 2026-10-29] when all of its HP heal, or against a hit that would kill or
                    # leave HP within one enemy hit of 0 (edamame_reason)
                    why = edamame_reason(s, d.action, hp_cap)
                    if why:
                        self._use_item("green_potion", p, s, frame, turn, why, tr, item_tries)
                        potion_check = (s.hero.hp, "green_potion")
                        continue
                if spend:
                    use = self._spend_item(s, d, tr, have)
                    if use:
                        self._use_item(use[0], p, s, frame, turn, use[1], tr, item_tries)
                        continue
                if d.action == "attack" and "curse_card" in have and self._curse_gain(s):
                    self._use_item("curse_card", p, s, frame, turn, "doubles the hit about to land", tr, item_tries)
                    continue
                if stalled:
                    log.event("stall_mode", turn=turn, quiet_turns=turn - last_progress, action=d.action,
                              value=round(d.value, 1), board=s.render())
                    self.say(f"   stall: {turn - last_progress} turns without progress; a 1-HP trade is allowed")
                conf = dict(p.confidence)
                log.turn(state=s, confidence=conf, perceive_ms=perceive_ms, decision=d,
                         plan_ms=plan_ms, frame=frame)
                self.live.turn(s=s, d=d, turn=turn, plan_ms=plan_ms, stalled=stalled, last_record=log.last_turn)
                if tr.notes:
                    log.event("tracker_notes", turn=turn, notes=tr.notes)
                self.say(f"t{turn:03d} {level} HP{s.hero.hp} {s.render():<28} -> {d.action:<12} "
                         f"(v={d.value:.0f}, d{d.depth}, {plan_ms:.0f}ms)")
                reacted = self.do(d.action, p, frame, tr.kit, tr.charge)
                if reacted:
                    no_change = 0
                else:
                    # hard stop [user review of run 20261030-041652, fix 2: over 800 inputs in a row changed
                    # nothing, for hours]. The longest such streak in the 37 earlier run logs was 5 inputs,
                    # apart from the 36 refused Katana clicks of run 20261028-092419 (t105-t140), which the
                    # loop breaker above now ends after two. Inputs that could not be sent count too.
                    no_change += 1
                    log.event("no_visible_change", turn=turn, action=d.action, in_a_row=no_change,
                              sent=reacted is not None)
                    if no_change >= NO_CHANGE_STOP:
                        self.say(f"   {no_change} inputs in a row changed nothing; stopping the run as stuck")
                        log.event("no_change_stop", turn=turn, inputs=no_change, action=d.action, board=s.render())
                        log.save_frame(frame, "stuck_no_change")
                        result = "stuck"
                        break
                if reacted is None:
                    # nothing was sent (e.g. the tile is not in the hand): never plan it again
                    # on this board, and do not advance the tracker [review of 0.15: the same
                    # unsendable queue action repeated forever]
                    banned.setdefault(key, set()).add(d.action)
                    log.event("loop_break", turn=turn, banned=sorted(banned[key]), board=s.render(),
                              reason="action could not be sent")
                    time.sleep(0.25)
                    continue
                # not reacted: [run 20261027-180145] small changes (an enemy queuing) fell under the
                # threshold; retrying sent the same key again and cost real turns. The input is counted
                # as done and the next frame is read fresh. Queued tiles too: 201 queue inputs in the 37 earlier
                # run logs showed no change within the wait and still reached the queue. A refused one leaves
                # the board as it was, and the loop breaker above bans it.
                tr.acted(s, d.action)
                turn += 1
                time.sleep(0.25)
        except KeyboardInterrupt:
            result = "stopped"
        except Exception as e:  # log and end the run cleanly
            log.exception("main_loop", e, frame=locals().get("frame"))
            result = "error"
            raise
        finally:
            if result == "win" and self.restart_after:
                try:
                    self.gave_up = self._give_up()
                except BaseException as e:   # the give-up must never turn a win into an error
                    log.exception("give_up", e)
                    self.gave_up = False
            self.rec.flush()
            summary = log.end_run(result)
            self.live.end(result, summary)
            self.say(f"run {run_id} ended: {result}; turns {summary['turns']}, "
                     f"HP lost {summary['hp_lost']} (unpredicted {summary['hp_lost_unpredicted']}); "
                     f"logs in runs/{run_id}")
        return result


def play(runs: int = 1, frame_policy: str = "events", autostart: bool = True, stage1: bool = True) -> None:
    bot = Bot(frame_policy, autostart, stage1=stage1)
    rewards.warm_pool()    # the valuation's workers start now, not at the first reward screen
    results = []
    for i in range(runs):
        # each run starts where the last one ended: a won run is given up from the pause
        # menu, which leads back to the camp; after a death the next run clicks RESTART on
        # the death screen [user 2026-10-28]
        bot.restart_after = autostart and i < runs - 1
        res = bot.run()
        results.append(res)
        if i == runs - 1:
            break
        if res == "win" and not bot.gave_up:
            print("stopping the series: the won run could not be given up from the pause menu")
            break
        if res not in ("win", "death"):
            print(f"stopping the series: the run ended '{res}'")
            break
    if runs > 1:
        print(f"series: {len(results)} runs, {results.count('win')} won, {results.count('death')} lost")
