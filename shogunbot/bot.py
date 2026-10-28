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
  reward  -> shrine: read the upgrade; a pure gain is always taken (simulation picks the tile),
             a trade-off is taken unless every tile option is a net loss against skipping;
             tile choice: take the best tile unless both are net losses;
             anything unreadable -> SKIP (rewards.py, perception/reward_screen.py)
  unknown -> wait (banners, transitions); a frame is logged if it lasts
The run ends on: boss beaten (the reward after the boss fight is taken first),
hero HP 0 or a death screen, or a screen that stays unknown too long.
"""

from __future__ import annotations

import ctypes
import json
import queue
import threading
import time

import numpy as np

from . import rewards
from .bridge import Tracker, kit_from_faces, match_faces
from .catalogue import CAT
from .model import DEFAULT_KIT, HERO_MAX_HP
from .perception import reward_screen, screens
from .perception.perceive import BASE_POOL, merge_blink, normalize, perceive
from .planner import WEIGHTS, plan, plan_live
from .rules import ICE_TURNS, POISON_TICK, POISON_TURNS
from .runlog import RunLogger
from .winio import GameWindow

VERSION = "0.16"
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
        if rs.kind not in ("tile_choice", "shrine") or (rs.kind == "shrine" and not rs.upgrade):
            why = "upgrade text not read" if rs.kind == "shrine" else f"layout {rs.kind}"
            self.say(f"reward ({rs.kind}): {why}; SKIP")
            return self._skip(frame, rs, why)
        self.say(f"reward ({rs.kind}{': ' + rs.upgrade if rs.upgrade else ''}): valuing the options "
                 f"on {len(rewards.SEEDS)} simulated fights if needed ...")
        if rs.kind == "tile_choice":
            dec = rewards.decide_tile(tr.kit, rs.offers, hp)
        else:
            dec = rewards.decide_upgrade(tr.kit, rs.upgrade, hp)
        best, opts, secs = dec.best, dec.options, dec.seconds
        self.log.event("reward_decision", layout=rs.kind, upgrade=rs.upgrade, level=level, hp=hp, seconds=secs,
                       workers=rewards.WORKERS, choice=best.name, rule=dec.rule, options=[o.summary() for o in opts])
        scored = [o for o in opts if o.scores]
        self.say("   " + (", ".join(f"{o.name} {o.value:+.2f}" for o in sorted(scored, key=lambda o: -o.value))
                          or "no simulation needed")
                 + f"  ({secs:.0f}s) -> {best.name}  [{dec.rule}]")
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
        """The chosen tile sits in the statue and UPGRADE is lit."""
        if not rs.upgrade_lit:
            return False
        if rs.statue is not None:
            return rs.statue.get("tile") == tr.kit[label].tile
        return label not in match_faces(tr.kit, rs.hand)   # statue unread: the hand says it left

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
            face = match_faces(tr.kit, cur.hand).get(label)
            if face is None:
                self.log.event("reward_failed", choice=best.name, reason="tile not found in hand",
                               statue=cur.statue, attempt=attempt)
                return False
            self.click(face["fx"], face["fy"], frame, screen="reward", reward=best.name, attempt=attempt)
            time.sleep(0.8)
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
        self.reward_n = 0
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
                p = perceive(frame, [t.tile for t in tr.kit])
                perceive_ms = (time.time() - t0) * 1000
                scr = p.screen

                if scr != "unknown":
                    unknown_since, unknown_saved = None, False

                # ---------------- rewards (M2: taken when the planner values them above skipping)
                if scr == "reward":
                    log.save_frame(frame, "win" if seen_boss else f"reward_seen_{self.reward_n + 1}")
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
                    if seen_boss:
                        result = "win"   # the reward after the boss fight ends Stage 1
                        break
                    continue

                # ---------------- end conditions
                if seen_boss and scr == "map":
                    result = "win"
                    log.save_frame(frame, "win")
                    break
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
                    log.event("kit", level=level, kit=tr.kit.to_list())
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
                    d = plan_live(s)
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
                reacted = self.do(d.action, p, frame, tr.kit, tr.charge)
                if reacted is None:
                    # nothing was sent (e.g. the tile is not in the hand): never plan it again
                    # on this board, and do not advance the tracker [review of 0.15: the same
                    # unsendable queue action repeated forever]
                    banned.setdefault(key, set()).add(d.action)
                    log.event("loop_break", turn=turn, banned=sorted(banned[key]), board=s.render(),
                              reason="action could not be sent")
                    time.sleep(0.25)
                    continue
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
