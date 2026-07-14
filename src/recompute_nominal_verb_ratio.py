"""
Recomputes nominal_verb_ratio with the bug fix (includes NOUN; bounded
proportion instead of unbounded count-ratio) from the already-stored
token shards. Two speedups vs. the original version:

  1. VECTORIZED per-shard aggregation — the original looped over each
     article in Python (~5000 iterations/shard) building dict rows one
     at a time. This version builds boolean columns once and uses a
     single groupby().sum() per shard — vectorized, C-level, no
     per-article Python loop at all.
  2. PARALLELIZED across shards via multiprocessing — same pattern as
     the other CPU-bound steps in this pipeline (run_pipeline.py's OCR
     cleaning, feature_pipeline.py's per-article feature loop).

Also only reads the 4 columns actually needed (article_id, pos,
is_punct, is_space) instead of the full shard — lighter I/O per shard.

Usage:
    python recompute_nominal_verb_ratio.py --run-dir data/spacy_books/run_<timestamp>
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=str, required=True)
args = parser.parse_args()

run_dir = Path(args.run_dir)
features_dir = run_dir / "features"


def recompute_shard(shard_path):
    token_df = pd.read_parquet(
        shard_path, columns=["article_id", "pos", "is_punct", "is_space"]
    )

    mask = ~token_df["is_punct"] & ~token_df["is_space"] & (token_df["pos"] != "NUM")
    d = token_df.loc[mask, ["article_id", "pos"]].copy()

    # Boolean columns computed once, vectorized — no per-article Python loop
    d["is_nominal"] = d["pos"].isin(["NOUN", "PROPN", "ADJ"])
    d["is_verb"] = d["pos"] == "VERB"

    counts = d.groupby("article_id", sort=False)[["is_nominal", "is_verb"]].sum()
    denom = counts["is_nominal"] + counts["is_verb"]

    counts["nominal_verb_ratio_fixed"] = (counts["is_nominal"] / denom).where(denom > 0, other=float("nan"))

    return counts.reset_index()[["article_id", "nominal_verb_ratio_fixed"]]


if __name__ == "__main__":
    shard_paths = sorted(run_dir.glob("shard_*.parquet"))
    print(f"Recomputing nominal_verb_ratio across {len(shard_paths)} shards...")

    n_workers = max(1, mp.cpu_count() - 1)
    fixed_dfs = []

    with mp.Pool(n_workers) as pool:
        for result in tqdm(pool.imap(recompute_shard, shard_paths, chunksize=5), total=len(shard_paths)):
            fixed_dfs.append(result)

    fixed_df = pd.concat(fixed_dfs, ignore_index=True)
    out_path = features_dir / "nominal_verb_ratio_fixed.parquet"
    fixed_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(fixed_df)} rows to {out_path}")