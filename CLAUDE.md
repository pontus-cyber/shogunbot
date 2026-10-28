# CLAUDE.md: working rules for this repo

Read the claude.ai project docs first: `claude/00-start-here.md`, then `claude/status.md`.
Game rules with sources are in `claude/stage1-domain-model.md`.

## Where the code runs
- The bot runs on the user's Windows PC: `C:\Users\Pontu\claudegamedevwagents\shogunbot\`.
- It uses the sibling venv `..\local-game-MCP\.venv` (dxcam capture and SendInput come from `local_game_mcp`).
- The user has little programming experience. He runs `.\play.bat` and reports turn numbers. Claude does all code work.
- Run data (`runs/`, `collect/`) stays on the PC and is never committed. Review it with `python3 shogunbot/analyze.py <run_id>` through the device shell (plain Python; works in the Linux VM).

## Ship procedure (every change)
1. `python -m pytest -q tests` must pass.
2. Replay every recorded run you can reach (`python -m shogunbot replay <folder>`); the match rate must not drop.
3. Bump `VERSION` in `shogunbot/bot.py`.
4. Copy changed files to the PC:
   - Stage copies under new file names in `/mnt/user-data/outputs/<new dir>/` (reusing a staged name can deliver a stale file).
   - Run `device_commit_files` with `force: true`.
   - Verify sizes on the PC with `wc -c` in the device shell.
5. Commit here and push to `main`, with a message that names the version and the reason.
6. Update `claude/status.md` in the project (run history, version log, next actions).

## Code rules
- Game facts live in `shogunbot/data/*.json` (tiles, enemies, traits, consumables, upgrades, regions, heroes), read through `catalogue.py`. Never hard-code a tile's damage or footprint, an enemy's HP or tiles, or an item's effect in code; add or fix it in the data with its source tag.
- `rules.py` and `model.py` are the simulator. A rule change needs a regression test that cites the run and turn that showed it.
- Damage-first scoring: never trade HP for kills while a zero-damage move exists (`planner.WEIGHTS`).
- Perception thresholds are measured on recorded frames. Note the frame or run in a comment when you change one.
- Only one chat edits code at a time.
