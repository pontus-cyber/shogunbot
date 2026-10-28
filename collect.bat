@echo off
cd /d "%~dp0"
set PY=%USERPROFILE%\claudegamedevwagents\local-game-MCP\.venv\Scripts\python.exe
title shogunbot-collect
"%PY%" -m shogunbot collect > collect_log.txt 2>&1
