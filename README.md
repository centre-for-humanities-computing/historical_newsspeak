# historical_newsspeak

Tracking diachronic, genre-specific linguistic change in ~5 million Danish newspaper articles (1666–1849). Pipeline: spaCy/DaCy tokenization → stylistic feature extraction → sentiment scoring → factor analysis and mixed-effects modeling of change over time, across four genres (National, International, Advertisement, Fiction).

Accompanying poster: *"Language in Expansion: Genre and Diachronic Variation in Danish Newspapers"* (EADH 2026).


## Repository structure

```
src/                      Data pipeline (tokenization -> features -> sentiment)
|-- config.py           Config file for paths, constants, and feature definitions (Python)
|-- config.R           Config file for paths, constants, and feature definitions (R)
|-- run_pipeline.py            spaCy/DaCy tokenization, POS, dependency parsing
|-- feature_pipeline.py        Per-article stylistic features + semantic sentiment
|-- finalize_sentiment.py      Corpus-wide sentiment standardization
|-- recompute_nominal_verb_ratio.py   Fix for a nominality-measure bug (see Known Issues)
`-- factor_sweep.py             Systematic factor-count sweep utility (imported by notebooks/main.py)

notebooks/                Analysis pipeline, run in order
|-- 01_prep_data.py            Loads processed features, applies cleaning and
                                length residualisation (TTRs ~ log tokens,
                                within-article SDs ~ sentence count)
|-- 02_fa.py                    Factor analysis: VIF, factorability (KMO,
                                Bartlett), factor-count sweep, promax solution,
                                and genre-invariance checks (leave-one-out
                                Tucker congruence + subsampling null)
|-- 03_coherence.py             Genre separability: multinomial logistic and
                                gradient-boosted classifiers under grouped CV,
                                rolling 30-year windows, coefficient profiles
                                (drift, cosine similarity, ||beta||)
|-- 05_correlations.py          Spearman correlations between features and
                                publication year, per genre
|-- 06_export_for_GAM.py        Writes the modelling frame for R
|-- 06_gam.R                    GAMMs per indicator and genre (bam, factor
                                smooth by newspaper); trajectories, first
                                derivatives, periods of supported change
`-- 06b_sensitivity_check.R     Refits across k in {8,12,16} and gamma in
                                {1,1.5,2}; agreement across configurations

setup_env.sh               Python 3.10 env for src/run_pipeline.py (spaCy/DaCy)
setup_env_features.sh      Python 3.12 env for src/feature_pipeline.py (GPU sentiment)
setup_env_analysis.sh      Lightweight CPU-only env for notebooks/main.py (no GPU/spaCy needed)
requirements.txt
requirements-analysis.txt  Requirements for notebooks/main.py specifically

data/                      (not tracked -- see Data section below for download)
figs/                      (not tracked) generated figures
methods_checkpoint.md      Running log of methodological decisions and diagnostics
```

## Environments

This project uses **four separate virtual environments**, because the
tokenization stage, the feature-extraction stage, and the analysis
stage have non-overlapping (and partly conflicting) dependency
requirements: spaCy/DaCy needs Python ≤3.10 and an old `thinc`/
`numpy<2.0` stack; the sentiment model requires Python ≥3.12 and GPU
libraries; the analysis stage (`main.py`) needs neither spaCy/DaCy nor
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

# Analysis / notebooks/main.py (CPU, lightweight, no GPU/spaCy)
bash setup_env_analysis.sh
source .venv-analysis/bin/activate
```

See `methods_checkpoint.md` for the full list of environment-specific
gotchas encountered (B200/Blackwell GPU CUDA build requirements, numpy
ABI conflicts, etc.) if you're setting this up on similar hardware.

## Running the pipeline

Run in order; each stage reads the previous stage's output from `data/`.

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

## Running the analysis

```bash
source .venv-features/bin/activate
python notebooks/main.py
```

This loads the merged, cleaned feature table and runs, in sequence:

1. Descriptive stats + article-volume figure
2. PCA (baseline: no dominant single "complexity" component)
3. VIF check + factor analysis (3-factor solution: syntactic complexity,
   lexical diversity, involved-vs-informational register)
4. Genre-specific measurement invariance (leave-one-out factor
   congruence -- structure breaks down for Fiction specifically)
5. Correlation-with-time analysis, per genre
6. Changepoint detection on yearly trends
7. Mixed-effects models on six representative features, by genre
8. Effect-size translation (SD-relative slopes, rho^2)

Each numbered section is labeled with which table/figure in the
paper/poster it produces.

## Known issues / fixed bugs worth knowing about

- **`nominal_verb_ratio`**: the original feature computation omitted
  `NOUN` from the nominality set and used an unbounded count-ratio
  (unstable for verb-sparse short articles). Fixed in
  `recompute_nominal_verb_ratio.py`; ran this after `feature_pipeline.py`
  and merged its output in, replacing the original column. this bug is fixed.
- **`lix`/`rix`**: excluded from factor analysis (though kept in
  descriptive tables) -- both are collinear with `avg_sentlen` by
  formula construction (they contain sentence length as a direct term),
  not just empirically. See `methods_checkpoint.md` for the full VIF
  diagnostic.
- **`cttr`, `noun_ttr`, `verb_ttr`**: residualized against log article
  length (OLS) before use -- raw type-token ratios are severely
  length-confounded in this corpus (length explained 88.9% of raw CTTR
  variance).
- **`compressrat`, `mtld`**: excluded from factor analysis due to
  length-threshold-gated missingness that otherwise collapses the
  usable sample to <1% of the corpus; reported as descriptive stats on
  their length-eligible subsets only.
- Mixed-effects models use a **random-intercept-only** specification
  (`(1|newspaper)`), not a random slope on year -- the random-slope
  specification is empirically unidentified at this sample size (see
  `methods_checkpoint.md`).

## Data

Corpus: ENO project (Aalborg University), Danish-language newspapers,
1666-1849, ~5M articles across four genre categories (National,
International, Advertisement, Fiction).

The final processed feature table
(`usage_features_13-07-26_with_nominal.parquet`, ~648MB) is too large
to store directly in this GitHub repo and is hosted externally
(ON DEMAND - email me: pascale.feldkamp@cas.au.dk)

## Citation

If you use this pipeline or findings, please cite:

```
[BibTeX entry for the EADH 2026 poster -- add once available]
```