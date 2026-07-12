#!/usr/bin/env bash
# setup_env.sh
#
# Rebuilds the DaCy (da_dacy_large_trf) pipeline environment on a B200
# (Blackwell, sm_100) GPU. Verified working July 2026 on UCloud.
#
# Usage:
#   cd /work/<your-project-folder>
#   bash setup_env.sh
#
# Why this specific sequence:
#   - da_dacy_large_trf requires spacy>=3.5.2,<3.6.0, which pulls in an old
#     thinc build (8.1.12) compiled against the NumPy 1.x C ABI.
#   - torch needs CUDA 12.8 wheels (cu128) — B200 (sm_100) support was only
#     added starting in PyTorch 2.7; cu124 and earlier do NOT have sm_100
#     kernels and will fail with "no kernel image is available".
#   - cupy 14.x requires numpy>=2.0, which conflicts with thinc's numpy<2.0
#     requirement — must pin cupy-cuda12x<14 (cupy v13.x supports numpy 1.22+).
#   - Install torch and cupy BEFORE spacy/dacy-model to avoid pip/uv pulling
#     in a CPU-only or mismatched build as a transitive dependency.

set -euo pipefail

echo "==> Setting up PATH for uv"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "==> Creating virtual environment (Python 3.10)"
uv venv --python 3.10 .venv
source .venv/bin/activate

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

echo "==> Installing spaCy (pinned to what da_dacy_large_trf requires)"
uv pip install "spacy[transformers]>=3.5.2,<3.6.0"

echo "==> Installing NumPy 1.x (required by thinc's compiled extensions)"
uv pip install "numpy<2.0"

echo "==> Installing CuPy (pinned <14 for NumPy 1.x compatibility)"
uv pip install "cupy-cuda12x<14"

echo "==> Downloading and installing the DaCy large transformer model wheel"
curl -L -o /tmp/da_dacy_large_trf-0.2.0-py3-none-any.whl \
  https://huggingface.co/chcaa/da_dacy_large_trf/resolve/main/da_dacy_large_trf-any-py3-none-any.whl
uv pip install /tmp/da_dacy_large_trf-0.2.0-py3-none-any.whl

echo "==> Installing remaining pipeline dependencies"
uv pip install pandas datasets tqdm pyarrow

echo "==> Checking for dependency conflicts"
uv pip check

echo "==> Verifying the full pipeline loads on GPU"
python -c "
import spacy
spacy.require_gpu()
nlp = spacy.load(
    'da_dacy_large_trf',
    disable=['coref', 'span_resolver', 'span_cleaner', 'entity_linker'],
)
doc = nlp('Dette er en dansk sætning.')
print([(t.text, t.pos_, t.ent_type_) for t in doc])
"

echo "==> Setup complete. Activate with: source .venv/bin/activate"