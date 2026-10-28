# shogunbot

Standalone Python bot for Shogun Showdown (Stage 1: Bamboo Grove I, II, boss Daisuke), Wanderer default kit.
Zero token cost per run: screen capture -> perception -> expectimax planner over a rules simulator -> keys/clicks, with full run logs.

Project docs (claude.ai project "autogame"): `claude/00-start-here.md`, `claude/status.md`, `claude/run-review-procedure.md`, `claude/stage1-domain-model.md`.

Game catalogue: `shogunbot/data/*.json` holds every tile, enemy, trait, consumable, upgrade, region and hero (from the wiki, with source tags); `shogunbot/catalogue.py` loads it.

Rewards (0.15): on a tile choice or shrine the bot simulates the rest of the stage with each option (`rewards.py`) and takes the best; it never rerolls.

## Run (Windows, from this folder)

    .\play.bat                                                   # one run, hold F8 to stop
    ..\local-game-MCP\.venv\Scripts\python.exe -m shogunbot analyze   # text report of the last run
    ..\local-game-MCP\.venv\Scripts\python.exe -m pytest -q tests     # tests

Requires the sibling `local-game-MCP` venv (capture + input) and `opencv-python-headless`.
