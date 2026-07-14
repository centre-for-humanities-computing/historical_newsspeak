#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv-analysis
source .venv-analysis/bin/activate
uv pip install -r requirements-analysis.txt
uv pip check
echo "Activate with: source .venv-analysis/bin/activate"
