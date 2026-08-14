# historical_newsspeak pipeline

SRC contains some helper functions for the analysis (`factor_sweep.py` and config files `config.py` / `config.R`), but MAINLY the feature extraction pipeline, which is run from the command line. The pipeline is:

Danish historical newspaper OCR → spaCy/DaCy tokenization → stylistic
feature extraction → sentiment standardization.

Runs on UCloud (SDU eScience), GPU job (B200), split across **two
separate Python venvs** — see below for why.

## Pipeline overview

```
raw text (newspapers)
        │
        ▼
run_pipeline.py            (.venv, Python 3.10)
        │  spaCy/DaCy tokenization, POS, dependency parse
        │  → data/spacy_books/run_<timestamp>/shard_*.parquet
        ▼
feature_pipeline.py        (.venv-features, Python 3.12)
        │  per-article stylistic features + semantic sentiment (raw)
        │  → data/spacy_books/run_<timestamp>/features/features_shard_*.parquet
        ▼
finalize_sentiment.py      (.venv-features, Python 3.12)
           one-time global standardization of sentiment scores
           → features_combined_standardized.parquet
```

## Why two venvs

- `run_pipeline.py` needs `da_dacy_large_trf`, which pins
  `spacy>=3.5.2,<3.6.0` → pulls in an old `thinc` build that requires
  Python ≤3.10 and `numpy<2.0`.
- `feature_pipeline.py` needs `SemanticProjection`, which requires
  Python **≥3.12**.

These two requirements cannot coexist in one venv. `feature_pipeline.py`
doesn't need spaCy/DaCy at all — it only reads the parquet files
`run_pipeline.py` already wrote — so this split is clean.

## Setup

### `.venv` (Python 3.10) — for run_pipeline.py

Run `bash setup_env.sh` (included), or manually:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 .venv
source .venv/bin/activate

uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "spacy[transformers]>=3.5.2,<3.6.0"
uv pip install "numpy<2.0"
uv pip install "cupy-cuda12x<14"

curl -L -o /tmp/da_dacy_large_trf-0.2.0-py3-none-any.whl \
  https://huggingface.co/chcaa/da_dacy_large_trf/resolve/main/da_dacy_large_trf-any-py3-none-any.whl
uv pip install /tmp/da_dacy_large_trf-0.2.0-py3-none-any.whl

uv pip install pandas datasets tqdm pyarrow
uv pip check
```

### `.venv-features` (Python 3.12) — for feature_pipeline.py / finalize_sentiment.py

Run bash setup_env_features.sh (included), or manually:

```bash
uv venv --python 3.12 .venv-features
source .venv-features/bin/activate

uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install pandas pyarrow tqdm lexical-diversity langdetect
uv pip install "git+https://github.com/lauritswl/SemanticProjection"
uv pip check
```

## Running

### 1. Tokenization (`.venv`)

```bash
source .venv/bin/activate
export HF_HOME=/work/<your-folder>/.hf_cache   # persist HF dataset cache across jobs

nohup python -u run_pipeline.py --batch-size 256 < /dev/null > run.log 2>&1 &
echo $! > pipeline.pid
disown
tail -f run.log
```

- Resumable: reads `processed_ids.txt` in the run dir and skips
  already-done articles. Use `--run-dir <path>` to resume into a
  specific existing run.
- `--limit N` for pilot/timing runs before committing to the full corpus.
- Output: `data/spacy_books/run_<timestamp>/shard_*.parquet` (5000
  articles per shard).

### 2. Feature extraction (`.venv-features`)

```bash
source .venv-features/bin/activate

nohup python -u feature_pipeline.py --run-dir data/spacy_books/run_<timestamp> \
  < /dev/null > features.log 2>&1 &
echo $! > features.pid
disown
tail -f features.log
```

- Resumable at the **shard level**: skips a shard entirely if
  `features_shard_XXXXX.parquet` already exists. No partial-shard
  resume — if a shard was killed mid-way, delete that one shard's
  output file and it'll be redone in full.
- To restart everything from scratch:
  ```bash
  rm -rf data/spacy_books/run_<timestamp>/features
  ```
- Output: `data/spacy_books/run_<timestamp>/features/features_shard_*.parquet`.

### 3. Sentiment finalization (`.venv-features`)

Run **once**, only after every shard from step 2 is done:

```bash
python finalize_sentiment.py --features-dir data/spacy_books/run_<timestamp>/features
```

Computes an exact pooled mean/std across every sentence-level raw
sentiment score in the whole corpus, then applies one consistent
z-score to every article. Output: `features_combined_standardized.parquet`.

**Do not** treat any single shard's sentiment scores as standardized on
their own — `SemanticProjector.standardize()` z-scores relative to
whatever batch it's given, so per-shard standardization would make the
same absolute sentiment value get different scores depending on what
else was in that shard. Raw scores are stored as (sum, sum-of-squares,
n) per article in each shard specifically so this final pass can pool
them exactly.

## Known gotchas (in case they resurface)

- **B200 (Blackwell/sm_100) needs `torch` built for `cu128`** — `cu121`/
  `cu124` wheels lack sm_100 kernels and fail at runtime with
  `CUDA error: no kernel image is available`, even though
  `torch.cuda.is_available()` still returns `True`. Always verify with:
  ```bash
  python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.tensor([1.0]).cuda())"
  ```
- **numpy ABI conflicts**: `thinc==8.1.12` needs `numpy<2.0`;
  `cupy-cuda12x>=14` needs `numpy>=2.0`. Fix is `cupy-cuda12x<14` (v13.x
  supports numpy 1.x). Always run `uv pip check` after installing new
  packages — it doesn't catch every ABI issue, but catches declared
  version-range conflicts.
- **CPU-bound steps should be parallelized** — both the OCR cleaning
  step in `run_pipeline.py` and the per-article feature loop in
  `feature_pipeline.py` run 12-18x faster under `multiprocessing.Pool`
  than single-threaded. 
- **`tail -f` is a blocking foreground command** — `Ctrl+C` to exit it
  and get your shell back; it does not kill the background job.