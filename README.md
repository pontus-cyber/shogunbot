# shogunbot

A Python bot that plays Shogun Showdown from the screen. It reads the game window, plans each turn in a simulator of the game's rules, and sends the keys and clicks
itself. It plays a full Day 1 run unattended with the Wanderer, from the camp through seven regions to the Shogun.

**Status:** wrapped up on 2026-11-04 at version 0.51. The bot was built for one PC and is kept here as a record of the
project. It is not packaged for other machines, and no installer is planned.

The code was written by Claude (Anthropic) as the coding agent. The author played the game, ran the bot and reviewed
its runs.

## Result

The bot won Day 1 six times. The first win was run 20261103-020829 on 2026-11-03 (version 0.48).

| Measure | Value |
|---|---|
| First full Day 1 win | Run 20261103-020829, version 0.48, 2026-11-03 03:41, 92.8 min, 6 HP left |
| Day 1 wins | 6: runs 20261103-020829, -035113, -052436, -070730 (0.48), -113809 (0.49), -131803 (0.50) |
| Full runs finished on 0.48–0.50 | 9: 6 wins, 2 deaths (one in the Shogun's Castle's wave level, one to the Shogun), 1 stuck (fixed in 0.49) |
| Best runs | 20261103-113809: 1 HP lost all run; 20261103-131803: 4 HP lost, none of it unpredicted |
| Run length | 80–107 minutes, 780–970 turns |
| Final version | 0.51 |

The route it plays: Bamboo Grove (Daisuke or Rei the Ruthless), Whispering Caves (Iwao the Impaler), Moonlit Port
(The Twins), Spirit Gateway (The Statue), Hot Springs (Kowa the Coward), Hideyoshi's Keep (Hideyoshi the Cunning) and
the Shogun's Castle (the Shogun), with a shop between regions.

## How it works

Screen capture of the game window → perception reads the board, units, telegraphs, hand, items, rewards, shop and map
→ a tracker keeps what the screen hides → a planner searches 4–6 turns ahead in a rules simulator (enemy behaviour
learned from the bot's own logs) and plays damage-first → reward and shop choices are valued by simulating 48 seeded
fights per option. Every game fact lives in `shogunbot/data/*.json`.

| Part | File(s) | What it does |
|---|---|---|
| Catalogue | `data/*.json`, `catalogue.py` | Every game fact (tiles with their footprints, enemies, traits, consumables, upgrades, skills, regions, heroes), each with its source. The code reads these and never hard-codes them |
| Rules engine (simulator) | `model.py`, `rules.py` | The game state (including the hero's tiles and skills) and the turn rules: the hero's action, then the enemy phase |
| Enemy behaviour | `behaviour.py`, `mine.py`, `data/behaviour.json` | What each enemy does next, learned from the bot's own run logs; the bosses' patterns as rules. Rebuilt with `python -m shogunbot.mine build` |
| Level generator | `levelgen.py`, `data/waves.json` | Each level's waves (units, spawn cells, facing, when the next wave comes), learned from recorded frames; used by the simulated fights. Rebuilt with `python -m shogunbot.levelgen build` |
| Planner | `planner.py` | Looks 4–6 turns ahead and scores positions: HP first, then kills, threats and cooldowns |
| Perception | `perception/board.py`, `perceive.py`, `screens.py`, `hud.py`, `tiles.py`, `text.py`, `reward_screen.py`, `shop_screen.py`, `map_screen.py`, `floor.py` | Reads the cells, units, HP, facing, telegraphs (bubbles that blink, '+' loads, move and turn markers), the wave counter, the hand with each tile's charge, the item column and items on the floor, the menus, the camp's day selector, reward screens, the shop and the map. Icon templates are built from recorded frames (`perception/build_*.py`) |
| Bridge | `bridge.py` | Turns a screen read into a simulator state and keeps what the screen hides (the kit, the queue order, facing, shields, bosses' combos) |
| Rewards | `rewards.py`, `selfplay.py` | Takes pure-gain upgrades; values trade-offs, tile choices and shop purchases against skipping (or keeping the coins) on the same seeded simulated fights |
| Bot loop | `bot.py` | Capture → perceive → plan → act → log. Every run starts at the camp on Day 1; menus, rewards, the shop, the map, items, the stall rule, and stops when inputs change nothing |
| Logs and review | `runlog.py`, `analyze.py`, `replay.py`, `report.py` | Run folders, a text report per run, and replays of recorded runs through the simulator |
| Live viewer | `live.py`, `watch.py` | Writes `runs/<id>/live.json` each turn; `watch` serves it as a local page |
| Tests | `tests/` | 586 tests; each rule learned in play is a test. Tests that read recorded frames skip when the frames are missing |

`CLAUDE.md` holds the working rules the coding agent followed (data before code, a regression test for every rule,
replays of every recording before a change shipped).

## How it was run

On the author's PC:

- Windows, with the game windowed at about 1920×1080 on the primary monitor.
- The Python environment of the sibling `local-game-MCP` folder (`..\local-game-MCP\.venv`), which supplies screen
  capture (dxcam) and input (SendInput); `install.bat` added OpenCV (`requirements.txt`) to it.
- Day 1 only: the bot steps the camp's day selector down to Day 1.

Commands, from this folder:

| To | Run |
|---|---|
| Play a full Day 1 run | `.\play.bat --full` (hold **F8** to stop at the next turn) |
| Play several runs in a row | `.\play.bat --full --runs 5` |
| Watch the decisions live | `.\watch.bat` in a second window (a local page at http://127.0.0.1:8765) |
| Text report of the last run | `..\local-game-MCP\.venv\Scripts\python.exe -m shogunbot analyze` |
| Run the tests | `..\local-game-MCP\.venv\Scripts\python.exe -m pytest -q tests` |

Without `--full` a run ends after Stage 1's boss. Run logs and frames go to `runs/` and recordings to `collect/`; they
stay on the PC and are not in this repository.

## Known limits

- Day 1 only, with the default hero. Tiles outside the base eight read as unknown and are never taken.
- Perception is tuned on one PC's game window (about 1920×1080 on the primary monitor) and was not tested elsewhere.
- The simulator predicts 89–90% of the recorded turns exactly (the target was 97%); the remaining misses rarely change
  the outcome of a Day 1 run.
- Not fixed: buying healing in a shop while the item column is full; a run ends when Windows refuses input (an input
  PermissionError, WinError 5) instead of waiting.
- Using the PC while the bot runs can stop it: input is refused, or a click pauses the console it runs in (Enter
  resumes it).
