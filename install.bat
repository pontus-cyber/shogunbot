@echo off
cd /d "%~dp0"
set PY=%USERPROFILE%\claudegamedevwagents\local-game-MCP\.venv\Scripts\python.exe
"%PY%" -m pip install -r requirements.txt > install_log.txt 2>&1
"%PY%" -m pytest -q tests >> install_log.txt 2>&1
echo DONE >> install_log.txt
