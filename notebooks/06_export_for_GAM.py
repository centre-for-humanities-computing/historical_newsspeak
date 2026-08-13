# %%
"""
Export the columns the GAM needs. mgcv (R) rather than pyGAM: we need
factor-smooth interactions (s(year, by = category)), factor-smooth random
trajectories (bs = "fs") and simultaneous-interval derivatives, none of
which pyGAM supports.

Fixes over the previous version:
  - `.dropna()` ran silently over every exported column. passive_ratio is
    undefined in 29.1% of articles and present_tense_ratio in 12.9%; if
    either had been in `cols`, a third of the corpus would have vanished
    non-randomly and genre-correlated. We now export only what the model
    needs and report the loss before dropping.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path.cwd().parent / "src"))
from config import CATEGORIES, DATA_FILE, DATA_PATH, MIN_YEAR, REPRESENTATIVE

MAX_ACCEPTABLE_LOSS = 0.02  # abort above this; investigate rather than drop

df = pd.read_parquet(DATA_FILE)

required = ["year", "category", "newspaper", "german_probability",
            "n_tokens_for_diversity", "num_sents"]
missing = [c for c in required + REPRESENTATIVE if c not in df.columns]
if missing:
    raise KeyError(f"missing columns: {missing}")

cols = required + list(REPRESENTATIVE)
if "pwa" in df.columns:
    cols.append("pwa")
else:
    print("NOTE: no `pwa` column -- merge OCR quality from the ENO dataset "
          "before fitting, or drop s(pwa) from 06_gam.R")

out = df[cols]

# Look before dropping.
na_counts = out.isna().sum()
na_counts = na_counts[na_counts > 0]
if len(na_counts):
    print("\nNAs per column:")
    print((na_counts.to_frame("n")
           .assign(pct=lambda t: (100 * t.n / len(out)).round(2))
           .to_string()))

before = len(out)
out = out.dropna().copy()
loss = 1 - len(out) / before
print(f"\ndropna: {before - len(out):,} rows ({loss:.2%})")
if loss > MAX_ACCEPTABLE_LOSS:
    raise ValueError(
        f"dropna removed {loss:.1%} of rows -- check the NA table above. "
        "A column that is undefined by construction (e.g. passive_ratio "
        "where no verb carries Voice) should be excluded, not row-dropped.")

# Distribution of the loss across genres: an even loss is tolerable, a
# concentrated one is a confound.
lost = (df.loc[~df.index.isin(out.index), "category"].value_counts()
        / df["category"].value_counts()).dropna()
if len(lost):
    print("\nshare of each genre lost:")
    print((100 * lost).round(2).to_string())

if MIN_YEAR:
    n = len(out)
    out = out[out["year"] >= MIN_YEAR]
    print(f"\nyear >= {MIN_YEAR}: dropped {n - len(out):,} rows")

out["category"] = pd.Categorical(out["category"], categories=CATEGORIES)
if out["category"].isna().any():
    raise ValueError("category values outside config.CATEGORIES")

print(f"\n{len(out):,} rows, {out.newspaper.nunique()} newspapers, "
      f"{out.year.min()}-{out.year.max()}")
print(out.category.value_counts().to_string())

# Per-genre-per-decade counts: the GAM will happily smooth through a decade
# holding 30 fiction articles, so check the density before believing a curve.
print("\narticles per genre per decade:")
print(pd.crosstab(out["year"] // 10 * 10, out["category"]).to_string())

dest = DATA_PATH / "gam_input.parquet"
out.to_parquet(dest, index=False)
print(f"\nwrote {dest}")
# %%
