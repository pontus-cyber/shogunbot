"""The level's waves, for the bot's logs and the planner (M3 step 2).

The HUD counter "WAVE x/y" (perception.hud) gives the wave and the level's wave count. It
was read on 185 of the 185 bot turn frames that show it in the Stage 1 runs, and it
changes on the frame the wave's units appear (203 of 203 changes on readable boards in
the recordings). When it cannot be read (a wave popping in, a banner), the 0.20.1 rule
stands in: a wave arrives when the number of living enemies rises. That rule alone logged
2-3 of Bamboo Grove I's 4 waves: a wave that arrives as the last enemies die does not
raise the count [m3-plan finding 4].

A wave's `spawned` list holds the units the tracker could not match to an earlier unit
(bridge.Tracker.fresh) in this turn or the one before; 0.20.1 listed every living enemy,
survivors included. `survivors` counts the older enemies on the board when it arrived.
"""

from __future__ import annotations


class Waves:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """A new level."""
        self.wave = 0              # the last wave seen in this level
        self.total = 0             # waves in the level from the HUD; 0 = not read yet
        self.prev_alive = 0
        self._fresh: list = []     # (turn, unit) new units not yet logged with a wave

    def update(self, hud, alive: list, fresh: list, turn: int) -> list[tuple[str, dict]]:
        """One fight state with living enemies. hud: (wave, waves in the level) or None;
        alive: the living enemies; fresh: (x, kind, facing) of the units new in it.
        Returns the events to log: ("wave_spawn", fields) or ("wave_corrected", fields)."""
        self._fresh = [(t, u) for t, u in self._fresh if t >= turn - 1]
        self._fresh += [(turn, {"x": x, "kind": k, "facing": f}) for x, k, f in fresh]
        out = []
        if hud:
            cur, total = hud
            self.total = total
            if cur > self.wave:
                out.append(self._spawn(cur, len(alive), turn, "hud"))
            elif cur < self.wave:
                # the fallback counted a wave that the HUD does not show (a unit missed on
                # one frame and seen again); the HUD wins
                out.append(("wave_corrected", {"turn": turn, "counted": self.wave, "hud": cur}))
            self.wave = cur
        elif len(alive) > self.prev_alive:
            out.append(self._spawn(self.wave + 1, len(alive), turn, "count"))
            self.wave += 1
        self.prev_alive = len(alive)
        return out

    def _spawn(self, wave: int, n_alive: int, turn: int, source: str) -> tuple[str, dict]:
        spawned = [u for _, u in self._fresh]
        self._fresh = []
        return ("wave_spawn", {"wave": wave, "total": self.total or None, "source": source, "turn": turn,
                               "alive_before": self.prev_alive, "survivors": max(0, n_alive - len(spawned)),
                               "spawned": spawned})
