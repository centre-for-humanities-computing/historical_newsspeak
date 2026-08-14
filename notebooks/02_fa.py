# %%
"""
Check (1): how well do the features cover the intended stylistic space?

Are some redundant or overly collinear, and do they resolve into
distinguishable latent constructs? Factor analysis with promax (oblique)
rotation, since the constructs we expect -- syntactic complexity, lexical
diversity, register -- are not independent, and forcing orthogonal factors
would distort the structure.

Ends with the loose end from earlier drafts: is Fiction's factor-structure
breakdown a property of Fiction, or just of its sample size?
"""
import sys
from pathlib import Path
# surpress future warnings from factor_analyzer, which are not relevant to our use
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (calculate_bartlett_sphericity,
                                             calculate_kmo)

sys.path.append(str(Path.cwd().parent / "src"))
from factor_sweep import run_factor_sweep
from config import DATA_PATH, FIGS_PATH, FEATURES, FA_FEATURES, CATEGORIES, DATA_FILE, DISPLAY_NAMES, CATEGORY_COLORS

N_FACTORS = 3
FACTOR_NAMES = [f"F{i+1}" for i in range(N_FACTORS)]

df = pd.read_parquet(DATA_FILE)
df["category"] = pd.Categorical(df["category"], categories=CATEGORIES)


# drop any rows with missing values
X = df[FA_FEATURES].dropna()
print(f"{len(X):,} / {len(df):,} rows retained ({len(X)/len(df):.1%})")


# %%
# --- collinearity: does excluding lix/rix resolve it? ----------------------
# Both take sentence length as a direct term, so they are collinear with
# avg_sentlen by formula construction rather than merely empirically.

# check all features, i.e., features
X_all = df[FEATURES].dropna()
vif_X_all = sm.add_constant(X_all)
vif_all = pd.DataFrame({
    "feature": vif_X_all.columns,
    "VIF": [variance_inflation_factor(vif_X_all.values, i)
            for i in range(vif_X_all.shape[1])]}).sort_values("VIF", ascending=False)
print(vif_all[vif_all.feature != "const"].to_string(index=False))

vif_X = sm.add_constant(X)
vif = pd.DataFrame({
    "feature": vif_X.columns,
    "VIF": [variance_inflation_factor(vif_X.values, i)
            for i in range(vif_X.shape[1])]}).sort_values("VIF", ascending=False)
print(vif[vif.feature != "const"].to_string(index=False))


# %%
# --- factorability + sweep -------------------------------------------------

chi_sq, p = calculate_bartlett_sphericity(X)
_, kmo = calculate_kmo(X)
print(f"Bartlett's chi2={chi_sq:.1f}, p={p:.4g}   KMO={kmo:.3f}")

# Heywood cases (communality >= 1 or negative uniqueness) mark improper
# solutions; the sweep reports these alongside cumulative variance.
sweep, all_loadings, all_uniq, best_k = run_factor_sweep(
    X, FA_FEATURES, factor_range=range(2, 9))
sweep["marginal_var_gain"] = sweep["cumulative_var"].diff()
print(sweep.to_string(index=False))

# yes, avg_sentlen goes from 19.2% to 3.1% variance explained when we drop lix/rix, 
# so the collinearity is resolved.

# sweep shows 3 or 4 factors, but we choose 3
# The fourth factor is marked by \textsc{ndd} (0.59) and function word ratio (0.53),
# with sentence length cross-loading at 0.49 --- it absorbs variance the 
# length-normalisation in \textsc{ndd} introduces rather than identifying a
# further construct, and it degrades the three-factor solution's clean
# structure (sentence length falls from 0.96 to 0.84). 

# %%
# --- final --------------------------------------------------------

# use FA_FEATURES, not FEATURES, because we dropped the collinear ones
# use 3 factors, because the sweep shows that 3 is the elbow point and 4+ are Heywood cases

fa = FactorAnalyzer(n_factors=N_FACTORS, rotation="promax").fit(X)
loadings = pd.DataFrame(fa.loadings_, index=FA_FEATURES, columns=FACTOR_NAMES)
uniq = pd.Series(fa.get_uniquenesses(), index=FA_FEATURES)

var = pd.DataFrame(fa.get_factor_variance(),
                   index=["SS loadings", "Proportion var", "Cumulative var"],
                   columns=FACTOR_NAMES)
print(loadings.round(2).to_string())
print("\n", var.round(3).to_string())
print("\nstrongest marker per factor:")
for f in FACTOR_NAMES:
    print(f"  {f}: {loadings[f].abs().idxmax()} ({loadings[f].abs().max():.2f})")


# %%
# --- loadings heatmap ------------------------------------------------------

PRETTY = {f: DISPLAY_NAMES.get(f, f) for f in FA_FEATURES}

sns.set_theme(style="whitegrid")

dominant = loadings.abs().idxmax(axis=1)
disp = loadings.loc[dominant.sort_values().index].copy()
disp[disp.abs() < 0.1] = 0

plt.figure(figsize=(4, 8))
sns.heatmap(disp, cmap="RdGy_r", center=0, annot=True, fmt=".2f", cbar=False)
plt.yticks(np.arange(len(disp)) + 0.5,
           disp.index.to_series().replace(PRETTY).str.replace("_", " ").str.title(),
           rotation=0)
plt.tight_layout()
plt.savefig(FIGS_PATH / "fa_loadings_heatmap.pdf", dpi=300)
plt.show()


# %%
# --- does the structure hold within each genre? ----------------------------
# Leave-one-out: fit the reference on everything EXCEPT the target genre, so
# the pooled solution is not dominated by the genre being tested.

def tucker(a, b):
    return float(np.sum(a * b) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))


def fit_loadings(data, feats=FA_FEATURES, k=N_FACTORS):
    fa_ = FactorAnalyzer(n_factors=k, rotation="promax").fit(data[feats].dropna())
    return pd.DataFrame(fa_.loadings_, index=feats,
                        columns=[f"F{i+1}" for i in range(k)])


from itertools import permutations

def best_match(target, ref):
    """One-to-one factor matching maximizing total absolute congruence."""
    scores = np.array([
        [abs(tucker(target[t].values, ref[r].values))
         for r in ref.columns]
        for t in target.columns
    ])

    best_perm = max(
        permutations(range(scores.shape[1])),
        key=lambda p: sum(scores[i, p[i]] for i in range(scores.shape[0]))
    )

    return np.array([
        scores[i, best_perm[i]]
        for i in range(scores.shape[0])
    ])


refs, actual = {}, {}
for genre in CATEGORIES:
    refs[genre] = fit_loadings(df[df.category != genre])
    own = fit_loadings(df[df.category == genre])
    actual[genre] = best_match(own, refs[genre])
    print(f"{genre:15s} best-match congruence per factor: "
          f"{[round(v, 3) for v in actual[genre]]}")


# %%
# --- is Fiction's breakdown real, or just its n? ---------------------------
# For each larger genre, draw subsamples at Fiction's size, refit, and record
# congruence against that genre's own leave-one-out reference. The procedure
# is identical to Fiction's actual test, so the distribution answers: what
# does congruence look like for a genre that DOES match the pooled solution,
# measured with only this much data?

N_ITER = 30
fiction_n = len(df[df.category == "fiction"][FA_FEATURES].dropna())
print(f"Fiction n = {fiction_n:,}\n")

nulls, rows = {}, []
for genre in [c for c in CATEGORIES if c != "fiction"]:
    pool = df[df.category == genre]
    null = np.array([
        best_match(fit_loadings(pool.sample(fiction_n, random_state=i)), refs[genre])
        for i in range(N_ITER)])
    nulls[genre] = null
    for fi in range(N_FACTORS):
        a = actual["fiction"][fi]
        rows.append({
            "comparison": genre, "factor": FACTOR_NAMES[fi],
            "fiction": round(a, 3),
            "null_mean": round(null[:, fi].mean(), 3),
            "null_min": round(null[:, fi].min(), 3),
            "pct_below_fiction": round(100 * (null[:, fi] < a).mean(), 1),
        })

print(pd.DataFrame(rows).to_string(index=False))
print(f"\nn_iter={N_ITER}, each subsample n={fiction_n:,}")




# %%
order = ["Advertisement", "National", "International"]
rng = np.random.default_rng(0)

lo = min(min(nulls[g].min() for g in order), min(actual["fiction"]))
xlim = (lo - 0.03, 1.02)

fig, axes = plt.subplots(1, N_FACTORS, figsize=(9, 2.6), sharex=True)
for fi, (ax, fname) in enumerate(zip(axes, FACTOR_NAMES)):
    fic = actual["fiction"][fi]
    for gi, genre in enumerate(order):
        v = nulls[genre][:, fi]
        ax.scatter(v, np.full_like(v, gi) + rng.uniform(-.14, .14, len(v)),
                   s=16, color="#555555", alpha=.4, lw=0, zorder=2)
    ax.axvline(fic, color=CATEGORY_COLORS["fiction"], lw=1.8, zorder=3)
    ax.set_title(fname, fontsize=10)
    ax.set_yticks(range(len(order)))
    ax.set_ylim(-.6, len(order) - .35)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color="0.88", lw=.4, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_yticklabels(order, fontsize=8)
for ax in axes[1:]:
    ax.set_yticklabels([])
axes[0].text(actual["fiction"][0] - 0.05, len(order) - .04, " Fiction",
             color="#055f11", fontsize=8, va="top")
fig.supxlabel(f"Tucker's congruence".replace(",", "{,}"),
              fontsize=9)
plt.tight_layout()
plt.savefig(FIGS_PATH / "fa_subsample_null.pdf", bbox_inches="tight", dpi=300)
plt.show()
# %%
