# shogunbot

Standalone Python bot for Shogun Showdown (Stage 1: Bamboo Grove I, II, boss Daisuke), Wanderer default kit.
Zero token cost per run: screen capture -> perception -> expectimax planner over a rules simulator -> keys/clicks, with full run logs.

Project docs (claude.ai project "autogame"): `claude/00-start-here.md`, `claude/status.md`, `claude/run-review-procedure.md`, `claude/stage1-domain-model.md`.

Game catalogue: `shogunbot/data/*.json` holds every tile, enemy, trait, consumable, upgrade, region and hero (from the wiki, with source tags); `shogunbot/catalogue.py` loads it.

Rewards (0.16, `rewards.py`): pure-gain upgrades (+damage, −cooldown, +slot) are always taken and simulation picks the tile; trade-offs and tile choices are simulated against skipping on the same seeded fights in a harder-than-Stage-1 scenario, and skipped only when every option is a net loss. It never rerolls.

## Run (Windows, from this folder)

    .\play.bat                                                   # one run, hold F8 to stop
    ..\local-game-MCP\.venv\Scripts\python.exe -m shogunbot analyze   # text report of the last run
    ..\local-game-MCP\.venv\Scripts\python.exe -m pytest -q tests     # tests

Requires the sibling `local-game-MCP` venv (capture + input) and `opencv-python-headless`.
