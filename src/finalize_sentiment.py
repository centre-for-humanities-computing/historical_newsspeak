# %%
"""
Finalizes semantic-projection sentiment scores across the whole corpus.

feature_pipeline.py stores RAW sentiment projection scores per article
as (sum, sum_of_squares, n) rather than a per-shard standardized value,
because SemanticProjector.standardize() z-scores relative to whatever
batch of texts it's given — doing that once per shard would make the
same absolute sentiment value get a different standardized score
depending on what else happened to be in that shard.

This script:
  1. Reads every features_shard_*.parquet
  2. Computes the EXACT pooled mean and std of the raw per-sentence
     scores across the entire corpus (using sum/sumsq/n — no need to
     hold every individual sentence score in memory at once)
  3. Applies one consistent z-score to every article's mean raw score
  4. Writes a single combined output parquet with the standardized
     column added

Run this once, after all shards from feature_pipeline.py are done.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--features-dir", type=str, required=True, help="Directory containing features_shard_*.parquet from feature_pipeline.py")
parser.add_argument("--out", type=str, default=None, help="Output path for the combined, standardized feature table")
args = parser.parse_args()

features_dir = Path(args.features_dir)
out_path = Path(args.out) if args.out else features_dir / "features_combined_standardized.parquet"

# %%
shard_paths = sorted(features_dir.glob("features_shard_*.parquet"))
print(f"Found {len(shard_paths)} feature shards in {features_dir}")

dfs = []
total_sum = 0.0
total_sumsq = 0.0
total_n = 0

for p in shard_paths:
    df = pd.read_parquet(p)
    dfs.append(df)
    total_sum += df["semantic_sentiment_sum"].sum()
    total_sumsq += df["semantic_sentiment_sumsq"].sum()
    total_n += df["semantic_sentiment_n"].sum()

if total_n == 0:
    raise ValueError("No sentiment scores found across any shard — check feature_pipeline.py output")

# Exact pooled mean/std across every individual sentence-level raw score
# in the corpus, computed from the per-article sums without ever holding
# every raw sentence score in memory simultaneously.
global_mean = total_sum / total_n
global_var = (total_sumsq / total_n) - (global_mean**2)
global_std = np.sqrt(max(global_var, 0.0))  # guard against tiny negative float noise

print(f"Global pooled stats across {total_n} sentence-level scores:")
print(f"  mean = {global_mean:.6f}")
print(f"  std  = {global_std:.6f}")

# %%
combined = pd.concat(dfs, ignore_index=True)

if global_std == 0:
    print("WARNING: global std is 0 — all sentiment scores identical? Standardized column will be all zeros/NaN.")
    combined["semantic_sentiment_standardized"] = np.nan
else:
    combined["semantic_sentiment_standardized"] = (
        combined["semantic_sentiment_mean_raw"] - global_mean
    ) / global_std

# Per-article dispersion: how much does sentiment vary sentence-to-sentence
# WITHIN an article? Derived directly from the already-stored sum/sumsq/n —
# no need to touch raw sentence-level scores again.
# Var(X) = E[X^2] - (E[X])^2
combined["semantic_sentiment_var_raw"] = (
    combined["semantic_sentiment_sumsq"] / combined["semantic_sentiment_n"]
) - (combined["semantic_sentiment_mean_raw"] ** 2)
# guard against tiny negative float noise from floating-point subtraction
combined["semantic_sentiment_var_raw"] = combined["semantic_sentiment_var_raw"].clip(lower=0)
combined["semantic_sentiment_std_raw"] = np.sqrt(combined["semantic_sentiment_var_raw"])

# Articles with only 1 sentence have exactly 0 variance by construction
# (nothing to vary against) — not missing data, but not meaningful either.
# Flag rather than silently mixing them in with genuine low-variance articles.
combined["semantic_sentiment_std_defined"] = combined["semantic_sentiment_n"] >= 2

# Rigorous "standardized" variance, matching the mean's treatment in spirit
# but using the statistically correct transform for a variance quantity.
# Variances are ratio-scale (always positive, right-skewed) — unlike the
# mean, simple subtraction/z-scoring isn't appropriate. Instead: compare
# each article's variance to the pooled corpus-wide sentence-level variance
# via a RATIO, then log-transform (same logic behind an F-test comparing
# group variances). log_ratio ≈ 0 means "as variable as the corpus average";
# positive means more variable within this article than typical; negative
# means less variable (more tonally consistent) than typical.
#
# Undefined (NaN) when var_raw == 0 — covers both single-sentence articles
# (n=1, trivially zero, flagged separately via std_defined) and the rare
# genuine case of identical scores across multiple sentences; log(0) is
# undefined either way, and "zero variance" isn't meaningfully comparable
# on a ratio scale.
#
# Caveat: articles with few sentences have noisier variance ESTIMATES
# (sampling variability of a sample variance shrinks with n) — this
# standardization doesn't correct for that; treat log_var_ratio for very
# low semantic_sentiment_n articles with appropriate caution.
if global_var > 0:
    combined["semantic_sentiment_var_ratio"] = combined["semantic_sentiment_var_raw"] / global_var
    with np.errstate(divide="ignore"):
        combined["semantic_sentiment_log_var_ratio"] = np.log(combined["semantic_sentiment_var_ratio"])
    combined.loc[combined["semantic_sentiment_var_raw"] == 0, "semantic_sentiment_log_var_ratio"] = np.nan
else:
    print("WARNING: global_var is 0 — cannot compute variance ratio.")
    combined["semantic_sentiment_var_ratio"] = np.nan
    combined["semantic_sentiment_log_var_ratio"] = np.nan

combined.to_parquet(out_path, index=False)
print(f"\nWrote {len(combined)} rows to {out_path}")