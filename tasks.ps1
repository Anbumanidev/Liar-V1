# Windows equivalent of the Makefile.  Usage:  .\tasks.ps1 <target> [-Start 2026-10-01] [-End 2026-10-31] [-Modes INTRADAY,SWING] [-Profile low] [-Llm rules]
param(
    [Parameter(Position = 0)][string]$Target = "help",
    [string]$Start = "",
    [string]$End = "",
    [string]$Modes = "INTRADAY,SWING",
    [string]$Symbols = "",
    [string]$Profile = "low",
    [string]$Llm = "rules"
)

$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

switch ($Target) {
    "setup" {
        python -m venv .venv
        & $Py -m pip install --upgrade pip
        & $Py -m pip install -r requirements.txt
        & $Py main.py init-db
    }
    "pull-models" { & $Py scripts/pull_models.py --profile $Profile }
    "init-db" { & $Py main.py init-db }
    "set-password" { & $Py scripts/set_password.py }
    "run" { & $Py main.py engine }
    "dashboard" { & $Py -m streamlit run dashboard/app.py --server.address 127.0.0.1 }
    "test" { & $Py -m pytest tests -q }
    "replay" {
        $argsList = @("main.py", "replay", "--start", $Start, "--end", $End, "--modes", $Modes, "--llm", $Llm)
        if ($Symbols) { $argsList += @("--symbols", $Symbols) }
        & $Py @argsList
    }
    "walk-forward" { & $Py main.py walk-forward --start $Start --end $End --modes $Modes --llm $Llm }
    "backup" { & $Py scripts/backup_db.py }
    "health" { & $Py scripts/health_check.py }
    default {
        Write-Output "Targets: setup, pull-models, init-db, set-password, run, dashboard, test, replay, walk-forward, backup, health"
    }
}
