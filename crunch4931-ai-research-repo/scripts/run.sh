#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.pipeline --config configs/baseline.json --output results/summary.json
python3 -m unittest discover -s tests -v
