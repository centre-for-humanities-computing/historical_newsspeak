# %%
"""
05_correlations.py -- correlations between linguistic features and time.

Computes Spearman correlations between publication year and each feature,
separately by genre, for:

    1. the full corpus
    2. the corpus restricted to 1740+

Only correlations with |rho| >= RHO_THRESHOLD are highlighted in the
heatmap. P-values are not reported because the corpus contains millions
of observations, making statistical significance largely uninformative.

Outputs:
    data/correlations_full.csv
    data/correlations_post1715.csv
    figs/corr_matrix_post_1715.pdf
"""

# %%
# ============================================================
# SETUP
# ============================================================

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

sys.path.append(str(Path.cwd().parent / "src"))

from config import (
    CATEGORIES,
    DATA_FILE,
    DATA_PATH,
    DISPLAY_NAMES,
    FIGS_PATH,
    FEATURES,
    MIN_YEAR,
    RHO_THRESHOLD,
)


# %%
# ============================================================
# 1. LOAD
# ============================================================

df = pd.read_parquet(DATA_FILE)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")


df["category"] = pd.Categorical(df["category"], categories=CATEGORIES)

print("\nCategory counts:")
print(df["category"].value_counts().to_string())

na = df[FEATURES].isna().sum()
if na.any():
    print("\nNaNs:")
    print(na[na > 0].to_string())


# %%
# ============================================================
# 2. SPEARMAN CORRELATIONS WITH TIME
# ============================================================

def corr_with_time(
    data,
    features=FEATURES,
    time_col="year",
):
    """
    Spearman correlation between publication year and each feature,
    calculated separately for each genre.
    """
    out = {}

    for category in CATEGORIES:
        cat_df = data.loc[data["category"] == category]
        out[category] = {}
        for feature in features:
            pair = cat_df[[time_col, feature]].dropna()
            if len(pair) < 2:
                out[category][feature] = float("nan")
                continue
            rho, _ = spearmanr(pair[time_col], pair[feature])
            out[category][feature] = rho
    return pd.DataFrame(out)


corr_tables = {}

for label, subset in [("full", df), (f"post{MIN_YEAR}", df.loc[df["year"] >= MIN_YEAR])]:

    tab = corr_with_time(subset)
    corr_tables[label] = tab

    # Save the complete correlation matrix.
    output = DATA_PATH / f"correlations_{label}.csv"
    tab.to_csv(output)

    over = tab.abs() >= RHO_THRESHOLD

    print(f"\n{'=' * 60}")
    print(f"{label}: n = {len(subset):,}")
    print(f"{'=' * 60}")

    print(f"\nFeatures with |rho| >= {RHO_THRESHOLD}:")
    print(over.sum().to_string())

    print(f"\nSum of |rho| over threshold:")
    print(
        (tab.abs() * over)
        .sum()
        .round(2)
        .to_string())


# %%
# ============================================================
# 3. POOLED CORRELATIONS
# ============================================================

pooled = {}

for feature in FEATURES:
    pair = df[["year", feature]].dropna()

    rho, _ = spearmanr(
        pair["year"],
        pair[feature])

    pooled[feature] = rho

pooled = (
    pd.Series(pooled)
    .sort_values(key=abs, ascending=False))

print("\n" + "=" * 60)
print("POOLED CORRELATIONS — ALL GENRES")
print("=" * 60)
print(pooled.round(3).to_string())


# %%
# ============================================================
# 4. HEATMAP 
# ============================================================

tab = corr_tables["full"].copy()

# Order features by their strongest absolute correlation
# across the four genres.
feature_order = (
    tab.abs()
    .max(axis=1)
    .sort_values(ascending=False)
    .index
)

tab = tab.loc[feature_order]

# Display names from config.py.
names = [
    DISPLAY_NAMES.get(
        feature,
        feature.replace("_", " ").title(),
    )
    for feature in tab.index
]

# Highlight only correlations above the reporting threshold.
weak = tab.abs() < RHO_THRESHOLD

sns.set_style("whitegrid")

fig, ax = plt.subplots(
    figsize=(4.5, max(5, len(tab) * 0.32)),
    dpi=500,
)

# Strong correlations: show value and colour.
sns.heatmap(
    tab,
    cmap="RdGy_r",
    center=0,
    annot=True,
    fmt=".2f",
    mask=weak,
    cbar=False,
    ax=ax,
)

# Weak correlations: show as grey cells without annotation.
sns.heatmap(
    tab,
    cmap=["lightgrey"],
    mask=~weak,
    cbar=False,
    alpha=0.8,
    ax=ax,
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
)

ax.set_yticklabels(
    names,
    rotation=0,
)

ax.set_xlabel("")
ax.set_ylabel("")

fig.tight_layout()

output = FIGS_PATH / "corr_matrix.pdf"
fig.savefig(
    output,
    dpi=500,
    bbox_inches="tight",
)

plt.show()


# %%
# ============================================================
# 5. RANKED CORRELATIONS
# ============================================================

# Useful for inspecting the strongest temporal associations
# without relying on the heatmap.

ranked = (
    corr_tables[f"post{MIN_YEAR}"]
    .stack()
    .rename("rho")
    .reset_index()
    .rename(columns={
        "level_0": "feature",
        "level_1": "category",
    })
)

ranked["abs_rho"] = ranked["rho"].abs()

ranked = (
    ranked
    .sort_values("abs_rho", ascending=False)
    .reset_index(drop=True)
)

ranked["feature"] = ranked["feature"].map(
    lambda x: DISPLAY_NAMES.get(
        x,
        x.replace("_", " ").title(),
    )
)

# ranked.to_csv(
#     DATA_PATH / "correlations_post1715_ranked.csv",
#     index=False,
# )

print("\n" + "=" * 60)
print(f"TOP TEMPORAL CORRELATIONS — POST-{MIN_YEAR}")
print("=" * 60)
print(
    ranked.head(30).to_string(index=False)
)


# %%
# ============================================================
# 6. VARIANCE EXPLAINED
# ============================================================

# rho² is the squared rank correlation. It gives a compact
# measure of the strength of the monotonic association with year,
# without imposing a linear functional form.

rho_squared = (
    corr_tables[f"post{MIN_YEAR}"] ** 2 * 100
)

rho_squared.index = [
    DISPLAY_NAMES.get(
        feature,
        feature.replace("_", " ").title(),
    )
    for feature in rho_squared.index
]

print("\n" + "=" * 60)
print("RHO² — %")
print("=" * 60)
print(rho_squared.round(1).to_string())
# %%

# %%
# %%
# ============================================================
# 7. LATEX TABLE ROWS
# ============================================================

# Column order:
#   Fiction Full | Fiction 1715+
#   National Full | National 1715+
#   International Full | International 1715+
#   Advertisement Full | Advertisement 1715+
#   All
#
# Full-corpus values are black.
# 1715+ values are grey.
# Values with |rho| < RHO_THRESHOLD are left blank.

TABLE_FEATURES = [
    ("avg_sentlen", "Avg sent. length"),
    ("avg_mdd", "Avg MDD"),
    ("rix", "RIX"),
    ("avg_ndd", "Avg NDD"),
    ("std_ndd_resid", "Std NDD"),
    ("avg_wordlen", "Avg word length"),
    ("lix", "LIX"),
    ("passive_ratio", "Passive Ratio"),
    ("nominal_verb_ratio", "Nominal--verb Ratio"),
    ("that_ratio", "`That' Ratio"),
    ("of_ratio", "`Of' Ratio"),
    ("function_word_ratio", "Functionword Ratio"),
    ("adjective_adverb_ratio", "Adj--adv Ratio"),
    ("std_mdd_resid", "Std MDD"),
    ("verb_ttr_resid", "Verb TTR"),
    ("cttr_resid", "CTTR"),
    ("noun_ttr_resid", "Noun TTR"),
    ("personal_pronoun_ratio", "Personal Pron. Ratio"),
    (
        "semantic_sentiment_log_var_ratio_resid",
        "Sentiment SD",
    ),
    ("semantic_sentiment_standardized", "Sentiment"),
    ("present_tense_ratio", "Present Tense Ratio"),
]


def latex_rho(value, gray=False):
    """Format a correlation for the LaTeX table."""
    if pd.isna(value) or abs(value) < RHO_THRESHOLD:
        return r"{\color{gray}}" if gray else ""

    value = f"{value:.2f}"

    if gray:
        return rf"{{\color{{gray}}{value}}}"

    return value


full = corr_tables["full"]
post = corr_tables[f"post{MIN_YEAR}"]


# ------------------------------------------------------------
# Pooled correlations across all genres
# ------------------------------------------------------------

pooled = {}

for feature in FEATURES:
    pair = df[["year", feature]].dropna()

    if len(pair) < 2:
        pooled[feature] = float("nan")
    else:
        pooled[feature] = spearmanr(
            pair["year"],
            pair[feature],
        )[0]

pooled = pd.Series(pooled)


# %%
# ------------------------------------------------------------
# Feature rows
# ------------------------------------------------------------

for feature, label in TABLE_FEATURES:

    values = []

    for category in CATEGORIES:

        # Full corpus: first, black.
        values.append(
            latex_rho(
                full.loc[feature, category],
                gray=False,
            )
        )

        # 1715+: second, grey.
        values.append(
            latex_rho(
                post.loc[feature, category],
                gray=True,
            )
        )

    # All: pooled full-corpus correlation.
    values.append(
        latex_rho(
            pooled.get(feature, float("nan")),
            gray=False,
        )
    )

    print(
        f"{label:<30} & "
        + " & ".join(values)
        + r" \\"
    )


# %%
# ------------------------------------------------------------
# Summary rows
# ------------------------------------------------------------

n_full = (full.abs() >= RHO_THRESHOLD).sum()
n_post = (post.abs() >= RHO_THRESHOLD).sum()

sum_full = (
    full.abs()
    .where(full.abs() >= RHO_THRESHOLD)
    .sum()
)

sum_post = (
    post.abs()
    .where(post.abs() >= RHO_THRESHOLD)
    .sum()
)

print(r"\midrule")

print(
    r"\textbf{N ($|\rho|>.05$)} & "
    + " & ".join(
        f"{n_full[category]:.0f} & "
        rf"{{\color{{gray}}{n_post[category]:.0f}}}"
        for category in CATEGORIES
    )
    + r" \\"
)

print(
    r"\textbf{Sum ($|\rho|>.05$)} & "
    + " & ".join(
        f"{sum_full[category]:.2f} & "
        rf"{{\color{{gray}}{sum_post[category]:.2f}}}"
        for category in CATEGORIES
    )
    + r" \\"
)
# %%
