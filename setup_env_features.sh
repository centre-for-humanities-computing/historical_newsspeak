#!/usr/bin/env bash
# setup_env_features.sh
#
# Rebuilds the feature-extraction environment (feature_pipeline.py,
# finalize_sentiment.py) on a B200 (Blackwell, sm_100) GPU.
#
# This is a SEPARATE venv from setup_env.sh's .venv, because:
#   - run_pipeline.py needs da_dacy_large_trf, which pins
#     spacy>=3.5.2,<3.6.0 -> old thinc -> Python <=3.10, numpy<2.0.
#   - feature_pipeline.py needs SemanticProjection, which requires
#     Python >=3.12.
# These two requirements can't coexist in one venv. feature_pipeline.py
# doesn't need spaCy/DaCy at all -- it only reads the parquet files
# run_pipeline.py already wrote -- so the split is clean, not a hack.
#
# Usage:
#   cd /work/<your-project-folder>
#   bash setup_env_features.sh

set -euo pipefail

echo "==> Setting up PATH for uv"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "==> Creating virtual environment (Python 3.12)"
uv venv --python 3.12 .venv-features
source .venv-features/bin/activate

echo "==> Checking GPU visibility"
nvidia-smi || echo "WARNING: nvidia-smi failed — confirm this job has a GPU attached"

echo "==> Installing torch (cu128 — required for B200/sm_100 support)"
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "==> Verifying torch can actually use the GPU (not just detect it)"
python -c "
import torch
print('torch version:', torch.__version__)
print('device:', torch.cuda.get_device_name(0))
print(torch.tensor([1.0]).cuda())
"

echo "==> Installing feature-extraction dependencies"
uv pip install pandas pyarrow tqdm lexical-diversity langdetect

echo "==> Installing SemanticProjector (requires Python >=3.12)"
uv pip install "git+https://github.com/lauritswl/SemanticProjection"

echo "==> Checking for dependency conflicts"
uv pip check

echo "==> Verifying SemanticProjector loads and runs on GPU"
python -c "
from semanticprojection.SemanticProjecter import SemanticProjector
projector = SemanticProjector()
projector.project_texts(texts=['Dette er en dansk sætning.'], concept_vector='Sentiment')
print('results keys:', list(projector.results.keys()))
print(projector.results)
"

echo "==> Setup complete. Activate with: source .venv-features/bin/activate"