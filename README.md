# historical_newsspeak

Code for *Genre-Specific Language Change in Danish Newspapers, 1666–1849* (CHR 2027). Tracks genre-specific linguistic change in Danish newspapers, 1740–1848 (ENO corpus, 3.66M articles): DaCy parsing → stylistic features → factor analysis, category separability, and GAMMs of change over time for National, International, Advertisement and Fiction.

## Running the analysis

The feature table is too large for the repo (see Data). With it in `data/`, run in `notebooks/`:

1. `01_prep_data.py` – cleaning, length residualisation
2. `02_fa.py` – factor analysis
3. `03_coherence.py` – category separability (classifiers, rolling windows)
4. `04_correlations.py` – feature–year correlations
5. `05_export_for_GAM.py` – modelling frame for R
6. `06_gam.R` – GAMMs on newspaper × year × category means: trajectories, periods of change, change table (`data/gam_cells/gam_change.tex`), figures `figs/gamcell_*.pdf`

For a quick test, point `DATA_FILE` in `src/config.py` to `data/usage_features_w_pwa_test.parquet`.

`notebooks/sample_ad_sentences.py` draws ads for a manual check of sentence segmentation (needs the token shards). `notebooks/archive/` holds the earlier article-level GAMMs, kept for reference.

## Repository structure

```
src/                      Data pipeline (tokenization -> features -> sentiment)
|-- config.py           Config file for paths, constants, and feature definitions (Python)
|-- config.R           Config file for paths, constants, and feature definitions (R)
|-- run_pipeline.py            spaCy/DaCy tokenization, POS, dependency parsing
|-- feature_pipeline.py        Per-article stylistic features + semantic sentiment
|-- finalize_sentiment.py      Corpus-wide sentiment standardization
|-- recompute_nominal_verb_ratio.py   Fix for a nominality-measure bug (see Known Issues)
`-- factor_sweep.py             Factor-count sweep utility (used by 02_fa.py)

notebooks/                Analysis, run in order (see above)
|-- 01_prep_data.py ... 05_export_for_GAM.py
|-- 06_gam.R                    GAMMs on newspaper-year means
|-- sample_ad_sentences.py      Sample for manual segmentation check
`-- archive/                    Earlier article-level GAMMs (superseded)

setup_env.sh               Python 3.10 env for src/run_pipeline.py (spaCy/DaCy)
setup_env_features.sh      Python 3.12 env for src/feature_pipeline.py (GPU sentiment)
setup_env_analysis.sh      CPU-only env for the notebooks
requirements.txt
requirements-analysis.txt  Requirements for the notebooks
requirements.R             R packages for 06_gam.R

data/                      (not tracked -- see Data section below for download)
figs/                      (not tracked) generated figures
```

## Environments

This project uses **four separate virtual environments**, because the
tokenization stage, the feature-extraction stage, and the analysis
stage have non-overlapping (and partly conflicting) dependency
requirements: spaCy/DaCy needs Python ≤3.10 and an old `thinc`/
`numpy<2.0` stack; the sentiment model requires Python ≥3.12 and GPU
libraries; the analysis stage needs neither spaCy/DaCy nor
GPU/torch at all, just stats/ML libraries.

```bash
# Tokenization (spaCy/DaCy, GPU)
bash setup_env.sh
source .venv/bin/activate

# Feature extraction + sentiment (GPU)
bash setup_env_features.sh
source .venv-features/bin/activate

# Sentiment finalization only (CPU, lightweight)
bash setup_env_finalize.sh
source .venv-finalize/bin/activate

# Analysis notebooks (CPU, no GPU/spaCy)
bash setup_env_analysis.sh
source .venv-analysis/bin/activate
```

## Running the pipeline

Runs in order; each stage reads the previous stage's output from `data/`.

```bash
# 1. Tokenization (.venv)
python src/run_pipeline.py --batch-size 256
#    -> data/spacy_books/run_<timestamp>/shard_*.parquet

# 2. Feature extraction + sentiment (.venv-features)
python src/feature_pipeline.py --run-dir data/spacy_books/run_<timestamp>
#    -> data/spacy_books/run_<timestamp>/features/features_shard_*.parquet

# 3. Sentiment finalization, once ALL shards from step 2 are done (.venv-finalize)
python src/finalize_sentiment.py --features-dir data/spacy_books/run_<timestamp>/features
#    -> features_combined_standardized.parquet

# 4. Nominal-verb-ratio fix (should be fixed in run_pipeline, so no need to rerun.) (.venv-features)
python src/recompute_nominal_verb_ratio.py --run-dir data/spacy_books/run_<timestamp>
```

Both `run_pipeline.py` and `feature_pipeline.py` are resumable: the
former skips already-processed articles via `processed_ids.txt`, the
latter skips a shard entirely if its output file already exists.


## Known issues / fixed bugs worth knowing about

- **`nominal_verb_ratio`**: the original feature computation omitted
  `NOUN` from the nominality set and used an unbounded count-ratio
  (unstable for verb-sparse short articles). Fixed in
  `recompute_nominal_verb_ratio.py`; ran this after `feature_pipeline.py`
  and merged its output in, replacing the original column. this bug is fixed.
- **`lix`/`rix`**: excluded from factor analysis (though kept in
  descriptive tables) -- both are collinear with `avg_sentlen` by
  formula construction (they contain sentence length as a direct term),
  not just empirically.
- **`cttr`, `noun_ttr`, `verb_ttr`**: residualized against log article
  length (OLS) before use -- raw type-token ratios are severely
  length-confounded in this corpus (length explained 88.9% of raw CTTR
  variance).
- **`compressrat`, `mtld`**: excluded from factor analysis due to
  length-threshold-gated missingness that otherwise collapses the
  usable sample to <1% of the corpus; reported as descriptive stats on
  their length-eligible subsets only.
- **GAMMs** are fitted to newspaper × year × category means (at least 20
  articles each), not to articles: articles from one paper-year share an
  editor and news cycle, and treating them as independent let the basis
  dimension, not the data, set the shape of the curves.

## Data

Corpus: ENO project (Aalborg University), Danish-language newspapers,
1666-1849, ~5M articles across four genre categories (National,
International, Advertisement, Fiction).

The final processed feature table
(`usage_features_13-07-26_with_nominal.parquet`, ~648MB) is too large
to store directly in this GitHub repo and is hosted externally
(ON DEMAND - email me at: pascale.feldkamp@cas.au.dk)

## Citation

If you use this pipeline or findings, please cite:

```
[great BibTeX entry]
```