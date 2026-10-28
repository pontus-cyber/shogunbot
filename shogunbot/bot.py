"""Live bot: capture -> perceive -> plan -> act -> log, for Stage 1.

    python -m shogunbot play                 # one run, frames saved on events
    python -m shogunbot play --frames all    # save every turn's frames
    python -m shogunbot play --no-autostart  # start the run yourself; bot takes over in combat

Hold F8 to stop the bot at the next turn boundary (Ctrl+C also works).

Screens:
  title   -> click NEW RUN                    (autostart)
  camp    -> walk right, click GO             (autostart)
  map     -> click the first node             (autostart)
  combat  -> plan one action and send it
  reward  -> click SKIP (tile rewards, shrines, the free-play shrine)
  unknown -> wait (banners, transitions); a frame is logged if it lasts
The run ends on: boss beaten (first reward/map screen after the boss fight),
hero HP 0 or a death screen, or a screen that stays unknown too long.
"""

from __future__ import annotations

import ctypes
import json
import queue
import threading
import time

import numpy as np

from .bridge import Tracker
from .catalogue import CAT
from .model import DEFAULT_KIT, HERO_MAX_HP
from .perception import screens
from .perception.perceive import merge_blink, perceive
from .planner import WEIGHTS, auto_depth, plan
from .rules import ICE_TURNS, POISON_TICK, POISON_TURNS
from .runlog import RunLogger
from .winio import GameWindow

VERSION = "0.14"
ACTION_KEY = {"left": "a", "right": "d", "turn": "w", "wait": "s", "attack": "space"}
LEVEL_WAVES = {lv["id"]: lv["waves"] for lv in CAT.regions["bamboo_grove"]["level_list"]}  # 4, 5, boss 0
VK_F8 = 0x77
# Edamame Brew restores up to 3 HP [catalogue]: drink at 7 or lower, so no healing is wasted
HEAL_AT = HERO_MAX_HP - CAT.consumable_by_screen("green_potion").effect["amount"]
POISON_MIN_VALUE = 4      # use Mass Poison when it deals at least this much in total
BLINK_SAMPLES = 6         # extra frames per turn to catch the yellow blink
BLINK_GAP_S = 0.15
UNKNOWN_FRAME_S = 6.0     # save a frame when a screen stays unknown this long
UNKNOWN_GIVE_UP_S = 90.0  # end the run as 'stuck'


def _stop_requested() -> bool:
    try:
        return bool(ctypes.WinDLL("user32").GetAsyncKeyState(VK_F8) & 0x8000)
    except Exception:
        return False


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


KEY_LABEL = {"a": "A", "d": "D", "w": "W", "s": "S", "space": "SPACE", "enter": "ENTER", "esc": "ESC"}


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
    def __init__(self, frame_policy: str = "events", autostart: bool = True, verbose: bool = True):
        self.gw = GameWindow()
        self.log = RunLogger("runs", frame_policy)
        self.rec = Recorder()
        self.autostart = autostart
        self.verbose = verbose

    # every input goes through these two, so every input is recorded
    def key(self, name: str, frame, **extra) -> None:
        self.rec.record(frame, KEY_LABEL.get(name, name.upper()), **extra)
        self.gw.key(name)

    def click(self, fx: float, fy: float, frame, away: bool = True, **extra) -> None:
        self.rec.record(frame, "CLICK", click=(fx, fy), **extra)
        self.gw.click(fx, fy)
        if away:
            self.gw.move_mouse_away()

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

    def do(self, action: str, p, frame: np.ndarray) -> bool:
        """Send one hero action. Returns True if the screen reacted."""
        if action in ACTION_KEY:
            self.key(ACTION_KEY[action], frame, bot_action=action)
        elif action.startswith("queue:"):
            tile = action.split(":", 1)[1]
            slot = next((h for h in p.hand if h["tile"] == tile), None)
            if slot is None:
                self.log.event("act_failed", action=action, reason="tile not found in hand",
                               hand=p.hand)
                return False
            self.click(slot["fx"], slot["fy"], frame, bot_action=action)
        else:
            self.log.event("act_failed", action=action, reason="unknown action")
            return False
        return self._wait_change(frame)

    # ------------------------------------------------------------------ items
    def _pre_plan_item(self, p, s, tries: dict):
        """Items whose value does not depend on this turn's plan."""
        have = {it["item"] for it in p.items}
        if "green_potion" in have and tries.get("green_potion", 0) < 2 and 0 < s.hero.hp <= HEAL_AT:
            return "green_potion", f"HP {s.hero.hp}"
        if "skull_card" in have and tries.get("skull_card", 0) < 2:
            # Mass Poison: every enemy, 1 damage per turn for 3 turns [catalogue]
            value = sum(min(e.hp, POISON_TICK * POISON_TURNS) for e in s.alive())
            if value >= POISON_MIN_VALUE or any(e.kind == "daisuke" for e in s.alive()):
                return "skull_card", f"poison worth {value} damage"
        return None

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
            return len(out.kills), sum(d for k, _, d, _ in out.enemy_damage if k == "daisuke")
        k0, b0 = result(False)
        k1, b1 = result(True)
        return k1 > k0 or b1 >= b0 + 2

    def _use_item(self, item: str, p, s, frame, turn: int, reason: str, tr, tries: dict) -> None:
        slot = next(it for it in p.items if it["item"] == item)
        tries[item] = tries.get(item, 0) + 1
        cons = CAT.consumable_by_screen(item)
        name = cons.name if cons else item
        self.say(f"   using {name}: {reason}")
        self.log.event("item_use", item=item, name=name, reason=reason, hp=s.hero.hp, turn=turn, attempt=tries[item])
        self.click(slot["fx"], slot["fy"], frame, item=item)
        time.sleep(1.0)
        after = perceive(self.gw.grab_stable(settle_s=0.2, timeout_s=2.0))
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
        else:
            self.say(f"   {item} still in its slot; the click may not have registered")

    # ------------------------------------------------------------------ run
    def run(self) -> str:
        gw, log = self.gw, self.log
        gw.focus()
        run_id = log.start_run({"bot_version": VERSION, "weights": WEIGHTS,
                                "autostart": self.autostart, "kit": DEFAULT_KIT.to_list()})
        self.rec.start(log.run_dir / "frames")
        self.say(f"run {run_id} started; recording to runs/{run_id}/frames; hold F8 to stop")
        tr = Tracker()
        turn = 0
        level = None
        seen_boss = False
        prev_alive = 0
        wave = 0
        last_hp = 10
        unknown_since = None
        unknown_saved = False
        camp_steps = 0
        map_tries = 0
        zero_hp = 0
        last_key, last_action = None, None
        potion_check = None
        item_tries: dict = {}
        seen_items: set = set()
        potion_failed: set = set()
        banned: dict = {}
        result = "stopped"

        try:
            while True:
                if _stop_requested():
                    result = "stopped"
                    break
                frame = gw.grab_stable(settle_s=0.2, timeout_s=3.0)
                t0 = time.time()
                p = perceive(frame)
                perceive_ms = (time.time() - t0) * 1000
                scr = p.screen

                if scr != "unknown":
                    unknown_since, unknown_saved = None, False

                # ---------------- end conditions
                if seen_boss and scr in ("reward", "map"):
                    result = "win"
                    log.save_frame(frame, "win")
                    break
                if scr == "death":
                    result = "death"
                    log.save_frame(frame, "death")
                    break

                # ---------------- menus
                if scr == "reward":
                    pos = screens.find_skip(frame)
                    self.say("reward/shrine: SKIP")
                    log.event("skip", level=level, pos=pos)
                    self.click(*pos, frame, screen="reward")
                    time.sleep(1.0)
                    continue
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
                map_tries = 0
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
                if s.level != level:
                    level = s.level
                    wave = 0
                    prev_alive = 0
                    log.level_start(level, s.hero.hp, LEVEL_WAVES.get(level, 0))
                    self.say(f"level {level}, HP {s.hero.hp}")
                if level == "boss":
                    seen_boss = True
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
                        log.event("item_seen", item=it["item"], fy=round(float(it["fy"]), 3))
                # ---- items before planning (all cost no turn [obs]; effects from the wiki)
                use = self._pre_plan_item(p, s, item_tries)
                if use:
                    item, reason = use
                    self._use_item(item, p, s, frame, turn, reason, tr, item_tries)
                    if item == "green_potion":
                        potion_check = (s.hero.hp, "green_potion")
                    continue
                last_hp = s.hero.hp if p.hero["hp_max"] else last_hp
                alive = s.alive()
                if not alive:
                    prev_alive = 0
                    time.sleep(0.3)
                    continue
                if len(alive) > prev_alive and level != "boss":
                    wave += 1
                    log.wave_seen(wave, [{"x": e.x, "kind": e.kind, "facing": e.facing} for e in alive],
                                  prev_alive, turn)
                prev_alive = len(alive)
                s.wave = wave if level != "boss" else 0
                s.turn = turn

                t1 = time.time()
                try:
                    d = plan(s, auto_depth(s))
                except Exception as e:  # a bad read must not end the run
                    log.exception("plan", e, frame=frame)
                    self.say(f"   planner error ({e!r}); re-reading the screen")
                    time.sleep(0.5)
                    continue
                plan_ms = (time.time() - t1) * 1000
                # loop breaker [run 20261027-182237: a misread facing made 'right' a
                # blocked move; the board never changed and the bot repeated it ~40 times]
                key = s.key()
                if last_key is not None and key == last_key and last_action in ("left", "right", "turn"):
                    banned.setdefault(key, set()).add(last_action)
                    log.event("loop_break", turn=turn, banned=sorted(banned[key]), board=s.render())
                    self.say(f"   '{last_action}' changed nothing; not repeating it here")
                if d.action in banned.get(key, ()):
                    alt = next((c["action"] for c in d.candidates if c["action"] not in banned[key]), d.action)
                    d.action = alt
                last_key, last_action = key, d.action
                best = next((c for c in d.candidates if c["action"] == d.action), {})
                have = {it["item"] for it in p.items if item_tries.get(it["item"], 0) < 2}
                if best.get("exp_damage", 0) >= 1.0:
                    # every move takes damage: freeze everyone (Mass Ice), turn everyone
                    # around (Rain of Mirrors) if that removes the damage, or shield
                    # (Kami Brew); then re-read and re-plan with the effect in the model
                    item = None
                    if "ice_card" in have:
                        item = "ice_card"
                    elif "mirror_card" in have and self._mirror_saves(s):
                        item = "mirror_card"
                    elif "blue_potion" in have:
                        item = "blue_potion"
                    if item:
                        self._use_item(item, p, s, frame, turn,
                                       f"expected {best['exp_damage']:.1f} damage on every move", tr, item_tries)
                        continue
                if d.action == "attack" and "curse_card" in have and self._curse_gain(s):
                    self._use_item("curse_card", p, s, frame, turn, "doubles the hit about to land", tr, item_tries)
                    continue
                conf = dict(p.confidence)
                log.turn(state=s, confidence=conf, perceive_ms=perceive_ms, decision=d,
                         plan_ms=plan_ms, frame=frame)
                if tr.notes:
                    log.event("tracker_notes", turn=turn, notes=tr.notes)
                self.say(f"t{turn:03d} {level} HP{s.hero.hp} {s.render():<28} -> {d.action:<12} "
                         f"(v={d.value:.0f}, d{d.depth}, {plan_ms:.0f}ms)")
                reacted = self.do(d.action, p, frame)
                if not reacted:
                    # [run 20261027-180145] small changes (an enemy queuing) fell under the
                    # threshold; retrying sent the same key again and cost real turns.
                    # The input is counted as done; the next frame is read fresh anyway.
                    log.event("no_visible_change", turn=turn, action=d.action)
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
            self.rec.flush()
            summary = log.end_run(result)
            self.say(f"run {run_id} ended: {result}; turns {summary['turns']}, "
                     f"HP lost {summary['hp_lost']} (unpredicted {summary['hp_lost_unpredicted']}); "
                     f"logs in runs/{run_id}")
        return result


def play(runs: int = 1, frame_policy: str = "events", autostart: bool = True) -> None:
    bot = Bot(frame_policy, autostart)
    for i in range(runs):
        res = bot.run()
        if res in ("stopped", "error", "stuck", "win") and i < runs - 1:
            # a win leaves the game in Stage 2; restarting needs a manual abandon for now
            print("stopping the series: restart from the title screen is not automated yet")
            break
