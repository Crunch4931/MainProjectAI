$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
python -m src.pipeline --config configs/baseline.json --output results/summary.json
python -m unittest discover -s tests -v
