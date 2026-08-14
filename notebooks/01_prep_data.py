# %%
"""
01_assemble.py -- build the analysis table.

features parquet + ENO metadata + pwa + fixed nominal-verb ratio
  -> filter -> parse dates -> drop implausible -> residualise -> save

Order matters: the TTR residualisation is fitted on the analysis sample, so
implausible rows must be dropped BEFORE it, or their extreme values pull the
regression coefficients.
"""
import sys
from datetime import date as py_date
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import spearmanr

sys.path.append(str(Path.cwd().parent / "src"))
from config import DATA_PATH, FEATURES, FIGS_PATH

ENO = DATA_PATH / "eno"
MIN_SENTS = 2
MIN_PWA = 0.5


# %%
# --- 0. one-off download (skipped if meta.parquet exists) -----------------
# The repo stores the same 4.9M articles twice: 68 `train-*` shards and 80
# `decade_*` files. Reading both gives exact duplicates of every article, so
# take the train shards only.

META = ENO / "pwa.parquet"
COLS = ["id", "pwa", "date", "newspaper", "predicted_category",
        "fictionality_tag", "fiction_prob", "non_fiction_prob"]

if not META.exists():
    from huggingface_hub import hf_hub_download, list_repo_files
    REPO = "chcaa/eno-newspapers-enriched"

    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("data/train-") and f.endswith(".parquet"))
    print(f"{len(files)} train shards")

    parts = []
    for i, f in enumerate(files, 1):
        p = hf_hub_download(REPO, f, repo_type="dataset")
        parts.append(pd.read_parquet(p, columns=COLS))
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {sum(map(len, parts)):,} rows")

    meta = pd.concat(parts, ignore_index=True)
    assert meta.id.nunique() == len(meta), \
        f"{len(meta) - meta.id.nunique():,} duplicate ids within the train split"
    meta.to_parquet(META, index=False)
    print(f"wrote {META}: {len(meta):,} rows")
else:
    print(f"{META} exists -- delete it to re-download")

# %%
# --- 1. load and merge -----------------------------------------------------
df = pd.read_parquet(ENO / "features_combined_standardized_13-07-26.parquet")
print(f"features: {len(df):,} rows")

meta = pd.read_parquet(ENO / "pwa.parquet").rename(columns={"id": "article_id"})
dupes = meta.article_id.duplicated().sum()
if dupes:
    print(f"WARNING: {dupes} duplicate ids in metadata -- keeping first")
    meta = meta.drop_duplicates("article_id")

df["article_id"] = df["article_id"].astype(str)
meta["article_id"] = meta["article_id"].astype(str)
# merge on article_id, left join to keep all features rows
df = df.merge(meta, on="article_id", how="left")
print(f"merged; unmatched: {df.predicted_category.isna().sum():,}")

nominal = pd.read_parquet(DATA_PATH / "nominal_verb_ratio_fixed.parquet")
# merge on article_id, left join to keep all features rows
df = df.merge(nominal, on="article_id", how="left")
print(f"nominal_verb_ratio_fixed NaN: {df.nominal_verb_ratio_fixed.isna().sum():,}")
df = (df.drop(columns=["nominal_verb_ratio"], errors="ignore").rename(columns={"nominal_verb_ratio_fixed": "nominal_verb_ratio"}))


# %%
# --- 2. category and length filters ---------------------------------------
# Fiction is relabelled from the fictionality tag and Paratext dropped AFTER
# the merge: filtering the metadata first left those article_ids with no
# matching row at all, producing ~62k "unknown" categories.

df = df[df.num_sents >= MIN_SENTS]
print(f"{len(df):,} rows with >= {MIN_SENTS} sentences")

df["predicted_category"] = np.where(df.fictionality_tag == "fiction",
                                    "fiction", df.predicted_category)
df = df[df.predicted_category != "Paratext"]
df["category"] = df.predicted_category.fillna("unknown").str.split().str[0]


# %%
# --- 3. dates --------------------------------------------------------------
# Dates before ~1677-09-21 overflow pandas' ns-precision datetime64 and become
# NaT under errors="coerce" -- a dtype limitation, not a formatting problem.
# Recover those with Python's native date, which covers the full proleptic
# Gregorian calendar.

df["dt"] = pd.to_datetime(df["date"], errors="coerce")
early = df.dt.isna() & df.date.notna()
print(f"pre-1677 dates needing fallback: {early.sum()}")


def parse_early(s):
    try:
        y, m, d = map(int, str(s).split("-")[:3])
        o = py_date(y, m, d)
        return o.toordinal(), o.year
    except (ValueError, TypeError):
        return None, None


df.loc[~early, "date_ordinal"] = df.loc[~early, "dt"].map(
    lambda x: x.toordinal() if pd.notnull(x) else None)
df.loc[~early, "year"] = df.loc[~early, "dt"].dt.year

fb = df.loc[early, "date"].apply(parse_early)
df.loc[early, "date_ordinal"] = fb.map(lambda t: t[0])
df.loc[early, "year"] = fb.map(lambda t: t[1])

df = df.dropna(subset=["article_id", "year"])
print(f"{len(df):,} rows, {int(df.year.min())}-{int(df.year.max())}")


# %%
# --- 4. drop implausible ---------------------------------------------------
# Segmentation failures (sentence lengths above 200 words, dependency
# distances above 20) and articles whose OCR word accuracy is below 0.5, i.e.
# more than half the words wrong. Both make every derived feature unreliable.

bad = ((df.avg_sentlen > 200) | (df.avg_mdd > 20) |
       (df.std_mdd > 20) | (df.pwa < MIN_PWA) |
       (df.avg_mdd <= 0))
print(f"dropping {bad.sum():,} rows ({bad.mean():.4%})")
print(df[bad].groupby("category").size().to_string())
df = df[~bad].copy()

print(df.category.value_counts().to_string())

# %%
# --- 5A. check confounding factors for diversity features ----------------------

# how many articles do we have shorter than 50 or 100 words?

n = df.n_tokens_for_diversity
for w in (50, 100):
    print(f"window {w}: {(n < w).mean():.1%} of articles too short")
    print(df[n < w].groupby("category").size().to_string(), "\n")

# --- TABLE TTR_residulaization in paper ----------------------------------------
# Note: we are not resiudalizing features with the OCR quality (pwa) 
# If denser typesetting makes text both harder to OCR and genuinely more nominal, 
# residualising strips the linguistic variation along with the error.
# so only what is mathematically dependent on TOKEN COUNT

def rho(a, b):
    m = df[[a, b]].dropna()
    return spearmanr(m[a], m[b])[0]

print(f"year vs n_tokens: {rho('year', 'n_tokens_for_diversity'):+.3f}\n")
print(f"year vs avg_sentlen: {rho('year', 'avg_sentlen'):+.3f}\n")
print(f"{'feature':12s} {'rho(year) raw':>14s} {'rho(sent_len)':>10s} {'rho(n_tokens)':>10s}")
for col in ["cttr", "noun_ttr", "verb_ttr"]:
    print(f"{col:12s} {rho('year', col):+14.3f} "
          f"{rho(col, 'avg_sentlen'):+10.3f} "
          f"{rho(col, 'n_tokens_for_diversity'):+10.3f}")

# %%
# --- 5B. residualise the TTR measures --------------------------------------
# Fitted after the exclusions above, so extreme values do not pull the
# coefficients.

def residualise(y, length):
    m = y.notna() & length.notna()
    fit = sm.OLS(y[m], sm.add_constant(np.log1p(length[m]))).fit()
    out = pd.Series(index=y.index, dtype=float)
    out[m] = fit.resid
    return out, fit.rsquared


for col in ["cttr", "noun_ttr", "verb_ttr"]:
    df[col + "_resid"], r2 = residualise(df[col], df.n_tokens_for_diversity)
    print(f"{col}: length explains {r2:.1%} of variance")


# %%
# --- 5a. which features depend on length? ----------------------------------
# Run on RAW columns: this diagnostic is what decides what gets residualised,
# so it cannot reference the _resid names that step will create.

RAW_FEATURES = [
    "avg_sentlen", "avg_wordlen", "avg_ndd", "avg_mdd", "std_ndd", "std_mdd",
    "cttr", "noun_ttr", "verb_ttr",
    "nominal_verb_ratio", "of_ratio", "that_ratio", "present_tense_ratio",
    "passive_ratio", "adjective_adverb_ratio", "personal_pronoun_ratio",
    "function_word_ratio",
    "semantic_sentiment_standardized", "semantic_sentiment_log_var_ratio",
    "lix", "rix",
]

print(f"{'feature':34s} {'rho(tokens)':>12s} {'rho(sents)':>11s} {'% zero':>8s}")
for f in RAW_FEATURES:
    if f not in df.columns:
        print(f"{f:34s} {'MISSING':>12s}")
        continue
    s = df[[f, "n_tokens_for_diversity", "num_sents"]].dropna()
    print(f"{f:34s} {spearmanr(s[f], s.n_tokens_for_diversity)[0]:+12.3f} "
          f"{spearmanr(s[f], s.num_sents)[0]:+11.3f} "
          f"{(df[f] == 0).mean():8.1%}")
    
# --- 5D. residualise the SD measures --------------------------------------

for col in ["std_ndd", "std_mdd", "semantic_sentiment_log_var_ratio"]:
    df[col + "_resid"], r2 = residualise(df[col], df.num_sents)
    print(f"{col}: sentence count explains {r2:.1%}")

# %%
# --- 6. save ---------------------------------------------------------------

out = DATA_PATH / "usage_features_w_pwa.parquet"
df.to_parquet(out, index=False)
print(f"wrote {out}: {len(df):,} rows, {len(df.columns)} columns")
print(f"\n{df.category.value_counts().to_string()}")
print(f"\nmean pwa by category:\n{df.groupby('category').pwa.mean().round(4).to_string()}")


# %%
# --- 7. corpus descriptives ------------------------------------------------

print(f"{len(df):,} articles | {df.newspaper.nunique()} newspapers | "
      f"{int(df.year.min())}-{int(df.year.max())}")

cats = df.category.value_counts()

# LaTeX for Table 1 -- generated, so it cannot drift from the data
print("\n% --- category sizes ---")
for c in cats.index:
    print(f"{c:15s} & {cats[c]:>9,} \\\\")
print(f"\\midrule\nTotal & {cats.sum():,} \\\\")



# %%
# --- figure: newspaper coverage over time ---------------------------------
# Bar height encodes article volume on a log scale: the corpus spans three
# orders of magnitude (891 to 1.38M articles per title), so linear heights
# would make everything but the two Copenhagen giants invisible.

from matplotlib import pyplot as plt
import matplotlib as mpl

sns.set_style("whitegrid")

papers = (df.groupby("newspaper")
.agg(year_min=("year", "min"), year_max=("year", "max"),
articles=("article_id", "size"),
words=("n_tokens_for_diversity", "sum"),
pwa=("pwa", "mean")).sort_values(["year_min", "year_max"], ascending=[False, False]))

y = np.arange(len(papers))

# Map total words to a truncated Greys palette
v = np.log10(papers["words"].to_numpy())
norm = mpl.colors.Normalize(v.min(), v.max())

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "Greys_no_white",
    mpl.cm.Greys(np.linspace(0.15, 1.0, 256)))

fig, ax = plt.subplots(figsize=(11, 4.2), dpi=300)

ax.barh(
    y,
    papers["year_max"] - papers["year_min"],
    left=papers["year_min"],
    height=0.65,
    color=cmap(norm(v)),
    lw=0,
)

# Word-count labels
for i, (_, r) in enumerate(papers.iterrows()):
    w = r.words
    lab = f"{w/1e6:.0f}M" if w >= 1e6 else f"{w/1e3:.0f}k"
    ax.text(
        r.year_max + 1.5, i, lab,
        va="center", fontsize=9, color="0.35"
    )

# Colorbar
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)

ticks = np.log10([1e5, 1e6, 1e7, 1e8])
ticks = ticks[(ticks >= v.min()) & (ticks <= v.max())]

cb.set_ticks(ticks)
cb.set_ticklabels([
    f"{10**t/1e6:.0f}M" if t >= 6 else f"{10**t/1e3:.0f}k"
    for t in ticks
])
cb.set_label("Words", fontsize=10)

ax.set_yticks(y)
ax.set_yticklabels(papers.index, fontsize=10)
ax.set_xlim(papers.year_min.min() - 5, papers.year_max.max() + 5)
ax.set_ylim(-0.8, len(papers) - 0.2)

ax.grid(axis="x", color="0.85", lw=0.4, zorder=0)
ax.set_axisbelow(True)

for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

ax.tick_params(axis="y", length=0)
ax.set_xlabel("Year", fontsize=10)

plt.tight_layout()
plt.savefig(FIGS_PATH / "newspaper_coverage.pdf", bbox_inches="tight")
plt.show()

# full numeric table for the appendix
print(papers.assign(
    period=papers.year_min.astype(int).astype(str) + "--"
           + papers.year_max.astype(int).astype(str)
)[["period", "articles", "words", "pwa"]].round(3).to_string())

for name, r in papers.iterrows():
    print(f"{name} & {int(r.year_min)}--{int(r.year_max)} & "
          f"{int(r.articles):,} & {int(r.words):,} \\\\")
print(f"\\midrule\nTotal & & {len(df):,} & {int(papers.words.sum()):,} \\\\")

# %%
# --- 7. OCR quality diagnostics -------------------------------------------
# The flat overall trend is reassuring, but genre-specific trends are the
# real risk: if OCR quality changes differently by genre, it confounds
# exactly the comparisons this paper makes.

print("mean pwa by decade:\n", df.groupby(df.year // 10 * 10).pwa.mean().round(4).to_string())
print("\nmean pwa by category:\n", df.groupby("category").pwa.mean().round(4).to_string())

print("\nrho(year, pwa) within genre:")
for c in df.category.cat.categories if hasattr(df.category, "cat") else df.category.unique():
    s = df[df.category == c][["year", "pwa"]].dropna()
    print(f"  {c:15s} {spearmanr(s.year, s.pwa)[0]:+.3f}")

print("\nrho(pwa, feature) -- OCR sensitivity per feature:")
for f in FEATURES:
    if f in df.columns:
        s = df[[f, "pwa"]].dropna()
        r = spearmanr(s.pwa, s[f])[0]
        flag = "  <-- check" if abs(r) > 0.1 else ""
        print(f"  {f:32s} {r:+.3f}{flag}")

print("\nlowest-pwa newspapers:\n", papers.pwa.nsmallest(5).round(4).to_string())


# %%
# --- 8. figure: OCR quality over time, by genre ---------------------------


df["year"] = df["dt"].dt.year

colors = CATEGORY_COLORS.copy()

fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
for c, g in df.groupby("category", observed=True):
    m = g.groupby("year").pwa.mean().sort_index()
    smoothed = m.rolling(window=5, center=True, min_periods=1).mean()
    ax.scatter(m.index, m.values, s=8, alpha=0.3, color=colors.get(c), zorder=1)
    ax.plot(smoothed.index, smoothed.values, lw=1.5, label=c, alpha=0.9, color=colors.get(c), zorder=2)

# add mean line for all categories combined
m_all = df.groupby("year").pwa.mean().sort_index()
smoothed_all = m_all.rolling(window=5, center=True, min_periods=1).mean()
ax.plot(smoothed_all.index, smoothed_all.values, lw=1.5, label="Overall", alpha=0.8, color="black", zorder=2)

ax.set_ylim(0.5, 1.0)
ax.set_ylabel("mean pwa")
ax.set_xlabel("Year")
ax.legend(frameon=False, fontsize=8, ncol=5)
plt.tight_layout()
plt.savefig(FIGS_PATH / "pwa_over_time.pdf")
plt.show()

# overall trend
fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
m = df.groupby("year").pwa.mean().sort_index()
smoothed = m.rolling(window=3, center=True, min_periods=1).mean()
ax.scatter(m.index, m.values, s=3, alpha=0.3, color="grey", zorder=1)
ax.plot(smoothed.index, smoothed.values, lw=1.5, label="Overall", alpha=0.9, color="black", zorder=2)
ax.set_ylabel("mean pwa")
ax.set_xlabel("Year")
ax.legend(frameon=False, fontsize=8, ncol=4)
ax.set_title("Overall OCR word accuracy over time", fontsize=10)
plt.tight_layout()
plt.savefig(FIGS_PATH / "pwa_over_time_overall.pdf")
plt.show()


# %%
# --- 9. figure: article volume over time (fig:dists) ----------------------

sns.set_style("whitegrid")
palette = list(sns.color_palette("hsv_r", n_colors=4))
palette[-1] = "#00c853"
palette[0] = "#ff0090"
palette[1] = "#ffb066"
palette[2] = "#b98cff"

cat_colors = CATEGORY_COLORS.copy()

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 4), sharex=True,
    gridspec_kw={"height_ratios": [1, 3], "hspace": 0.2}, dpi=500)

sns.histplot(data=df, x="year", discrete=True, color="grey", ax=ax1,
             kde=True, alpha=0.6)
ax1.set_ylabel("Total articles")

sns.histplot(data=df, x="year", hue="category", multiple="fill",
             discrete=True, legend=True, ax=ax2, palette=palette, alpha=0.8)
ax2.set_ylabel("Proportion")
ax2.set_xlabel("Year")
sns.move_legend(ax2, "upper right", bbox_to_anchor=(.92, .3), frameon=False, ncol=4)
plt.tight_layout()
plt.savefig(FIGS_PATH / "article_volume_over_time.pdf", dpi=500)
plt.show()

# %%
