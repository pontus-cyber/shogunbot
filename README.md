# shogunbot

Standalone Python bot for Shogun Showdown (Stage 1: Bamboo Grove I, II, boss Daisuke), Wanderer default kit.
Zero token cost per run: screen capture -> perception -> expectimax planner over a rules simulator -> keys/clicks, with full run logs.

Project docs (claude.ai project "autogame"): `claude/00-start-here.md`, `claude/status.md`, `claude/run-review-procedure.md`, `claude/stage1-domain-model.md`.

Game catalogue: `shogunbot/data/*.json` holds every tile, enemy, trait, consumable, upgrade, region and hero (from the wiki, with source tags); `shogunbot/catalogue.py` loads it.

Enemy behaviour (0.17, M3): learned from the bot's own run logs. `python -m shogunbot.mine build <folder with runs/>` rebuilds `shogunbot/data/behaviour.json`; `python -m shogunbot.mine report <folder>` scores it against the logs.

Rewards (0.16, `rewards.py`): pure-gain upgrades (+damage, −cooldown, +slot) are always taken and simulation picks the tile; trade-offs and tile choices are simulated against skipping on the same seeded fights in a harder-than-Stage-1 scenario, and skipped only when every option is a net loss. It never rerolls.

## Run (Windows, from this folder)

    .\play.bat                                                   # one run, hold F8 to stop
    .\play.bat --runs 10                                         # ten runs in a row, unattended (0.18)
    .\watch.bat                                                  # live decision viewer in the browser (0.19)
    ..\local-game-MCP\.venv\Scripts\python.exe -m shogunbot analyze   # text report of the last run
    ..\local-game-MCP\.venv\Scripts\python.exe -m pytest -q tests     # tests

In a series, a won run is given up from the pause menu (PAUSE MENU, GIVE UP, YES: GIVE UP), which
leads back to the camp, and the next run walks to GO; after a death the next run clicks RESTART.
Start a series from the title screen, the camp, a Stage 1 map, a Stage 1 fight or the death screen,
not from the Stage 2 map a won run ends on.

Live decision viewer (0.19): while the bot plays, `watch.bat` (or `python -m shogunbot watch`) serves
http://127.0.0.1:8765 from the standard library, which follows `runs/<id>/live.json`: the board with every
telegraph, all candidate moves with value and expected damage, the likely line the search found, why the
best move beat the second best, the last 10 turns predicted vs actual, and the latest reward decision.

Requires the sibling `local-game-MCP` venv (capture + input) and `opencv-python-headless`.
