@echo off
cd /d "%~dp0"
"..\local-game-MCP\.venv\Scripts\python.exe" -m shogunbot play %*
pause
