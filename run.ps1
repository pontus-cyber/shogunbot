# Usage:  .\run.ps1 collect | play | report | review <run_id> | selfplay
$venv = "$HOME\claudegamedevwagents\local-game-MCP\.venv\Scripts\python.exe"
Set-Location $PSScriptRoot
& $venv -m shogunbot @args
