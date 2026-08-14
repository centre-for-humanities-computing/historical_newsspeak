# %%

"""
To answer "is the object we are studying real?"

The categories come from a classifier over document *embeddings* (semantic,
topical). They have never been validated as groupings in the surface-feature
space we actually analyse. Every claim of the form "Advertisement declines in
sentence length" presupposes that Advertisement is a coherent stylistic object
rather than an arbitrary partition.

Because the labels were assigned from embeddings and never saw these features,
separability here is an independent check rather than a circular one.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append("../src")
from config import DATA_PATH, DATA_FILE, FIGS_PATH, FA_FEATURES, CATEGORIES, DISPLAY_NAMES, COMPLEXITY, DIVERSITY, REGISTER, AFFECT, MIN_YEAR, CATEGORY_COLORS

MODEL = "logit"
year_cutoff = MIN_YEAR
N_SEEDS = 30

df = pd.read_parquet(DATA_FILE)
df["category"] = pd.Categorical(df["category"], categories=CATEGORIES)
print(df.category.value_counts().to_string())
df.head()

# %%

# FUNCTIONS

def _balance(d: pd.DataFrame, label: str, n_per_class: int | None, seed: int) -> pd.DataFrame:
      """Downsample every class to the same size."""
      n = n_per_class or d[label].value_counts().min()
      rng = np.random.default_rng(seed)
      idx = []
      for _, g in d.groupby(label, observed=True):
            take = min(len(g), n)
            idx.extend(rng.choice(g.index.to_numpy(), size=take, replace=False))
      return d.loc[idx]


def separability(df: pd.DataFrame, features: list[str], label: str = "category", model: str = "logit", n_per_class: int | None = None, n_splits: int = 5, seed: int = 0, shuffled: bool = False, group_col: str | None = None):
      cols = features + [label]
      if group_col is not None: cols.append(group_col)
      d = df[cols].dropna()
      d = _balance(d, label, n_per_class, seed)
      X = d[features].to_numpy()
      y = d[label].astype(str).to_numpy()
      if shuffled: y = np.random.default_rng(seed).permutation(y)
      groups = d[group_col].to_numpy() if group_col is not None else None
      clf = (HistGradientBoostingClassifier(random_state=seed) if model == "gb" else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
      if group_col is not None:
           cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
           splits = cv.split(X, y, groups=groups)
      else:
           cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
           splits = cv.split(X, y)
      scores, y_true, y_pred = [], [], []
      for tr, te in splits:
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[te])
            scores.append(f1_score(y[te], p, average="macro"))
            y_true.extend(y[te]); y_pred.extend(p)
      labels = sorted(set(y))
      cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels, normalize="true"), index=labels, columns=labels)
      return float(np.mean(scores)), float(np.std(scores)), cm, labels


def separability_per_genre(df, features, label="category", model="logit", n_per_class=None, n_splits=5, seed=0, group_col=None):
      cols = features + [label] + ([group_col] if group_col else [])
      d = _balance(df[cols].dropna(), label, n_per_class, seed)
      X, y = d[features].to_numpy(), d[label].astype(str).to_numpy()
      labels = sorted(set(y))
      clf = (HistGradientBoostingClassifier(random_state=seed) if model == "gb" else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
      yt, yp = [], []
      if group_col is not None:
            groups = d[group_col].to_numpy()
            for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups=groups):
                  clf.fit(X[tr], y[tr]); yp.extend(clf.predict(X[te])); yt.extend(y[te])
      else:
            for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
                  clf.fit(X[tr], y[tr]); yp.extend(clf.predict(X[te])); yt.extend(y[te])
      return pd.Series(f1_score(yt, yp, average=None, labels=labels), index=labels)


def separability_rolling(df, features, window=30, step=5, min_per_class=400, label="category", year_col="year", model="logit", seeds=range(5), n_per_class=None, group_col=None):
      cols = features + [label, year_col] + ([group_col] if group_col else [])
      d = df[cols].dropna()
      y0, y1 = int(d[year_col].min()), int(d[year_col].max())
      ok = {}
      for s in range(y0, y1 - window + 2, step):
            c = d[d[year_col].between(s, s + window - 1)][label].value_counts()
            if len(c) == d[label].nunique() and c.min() >= min_per_class: ok[s] = int(c.min())
      if not ok: raise ValueError("no window qualifies; widen window, lower min_per_class, or restrict the year range")
      n_fixed = n_per_class or min(ok.values())
      rows = []
      for s in sorted(ok):
            g = d[d[year_col].between(s, s + window - 1)]
            f1s = [separability(g, features, label, model=model, n_per_class=n_fixed, seed=sd, group_col=group_col)[0] for sd in seeds]
            rows.append({"start": s, "mid": s + window // 2, "n_per_class": n_fixed, "n_articles": len(g), "mean": float(np.mean(f1s)), "lo": float(np.quantile(f1s, .05)), "hi": float(np.quantile(f1s, .95))})
      return pd.DataFrame(rows)


def saturation_curve(df, features, n_values=(50, 100, 200, 350, 500, 1000, 2000, 4000, 8000, 16000), label="category", model="logit", seeds=range(5), group_col=None):
      cols = features + [label] + ([group_col] if group_col else [])
      d = df[cols].dropna()
      avail = d[label].value_counts().min()
      rows = []
      for n in [v for v in n_values if v <= avail]:
            f1s = [separability(d, features, label, model=model, n_per_class=n, seed=s, group_col=group_col)[0] for s in seeds]
            rows.append({"n_per_class": n, "mean": float(np.mean(f1s)), "lo": float(np.quantile(f1s, .05)), "hi": float(np.quantile(f1s, .95))})
      return pd.DataFrame(rows)


def per_genre_rolling(data, features, window=30, step=5, min_per_class=300, model="logit", seeds=range(5), year_col="year", label="category", group_col=None):
      d = data.dropna(subset=features + [label, year_col] + ([group_col] if group_col else []))
      y0, y1 = int(d[year_col].min()), int(d[year_col].max())
      ok = {}
      for s in range(y0, y1 - window + 2, step):
            c = d[d[year_col].between(s, s + window - 1)][label].value_counts()
            if len(c) == d[label].nunique() and c.min() >= min_per_class: ok[s] = int(c.min())
      if not ok: raise ValueError("no window qualifies")
      n_fixed = min(ok.values())
      means, los, his = [], [], []
      for s in sorted(ok):
            g = d[d[year_col].between(s, s + window - 1)]
            runs = pd.concat([separability_per_genre(g, features, model=model, n_per_class=n_fixed, seed=sd, group_col=group_col) for sd in seeds], axis=1)
            mid = s + window // 2
            means.append(runs.mean(axis=1).rename(mid)); los.append(runs.quantile(.05, axis=1).rename(mid)); his.append(runs.quantile(.95, axis=1).rename(mid))
      out = tuple(pd.DataFrame(x) for x in (means, los, his))
      for f in out: f.attrs["n_per_class"] = n_fixed
      return out


def chance_rolling(df, features, window=30, step=5, min_per_class=300, seeds=range(30), label="category", year_col="year", group_col=None):
      """Permutation baseline separately for each temporal window."""
      cols = features + [label, year_col] + ([group_col] if group_col else [])
      d = df[cols].dropna()
      y0, y1 = int(d[year_col].min()), int(d[year_col].max())
      ok = {}
      for s in range(y0, y1 - window + 2, step):
            c = d[d[year_col].between(s, s + window - 1)][label].value_counts()
            if len(c) == d[label].nunique() and c.min() >= min_per_class: ok[s] = int(c.min())
      n_fixed = min(ok.values())
      rows = []
      for s in sorted(ok):
            g = d[d[year_col].between(s, s + window - 1)]
            vals = [separability(g, features, label, model=MODEL, n_per_class=n_fixed, seed=sd, shuffled=True, group_col=group_col)[0] for sd in seeds]
            rows.append({"start": s, "mid": s + window // 2, "mean": np.mean(vals), "lo": np.quantile(vals, .05), "hi": np.quantile(vals, .95)})
      return pd.DataFrame(rows)

# %%

# --- 1. are the categories coherent at all? --------------------------------

f1_articles, f1_newspapers, chances, cms, gb_ceilings = [], [], [], [], []

for seed in range(N_SEEDS):
     f1_newpaper, _, cm, _ = separability(df, FA_FEATURES, model=MODEL, seed=seed, group_col="newspaper")
     f1_article = separability(df, FA_FEATURES, model=MODEL, seed=seed)[0]
     chance = separability(df, FA_FEATURES, model=MODEL, shuffled=True, seed=seed, group_col="newspaper")[0]
     gb_ceiling = separability(df, FA_FEATURES, model="gb", seed=seed, group_col="newspaper")[0]
     f1_articles.append(f1_article); f1_newspapers.append(f1_newpaper); chances.append(chance); cms.append(cm); gb_ceilings.append(gb_ceiling)

print(f"article-level:  {np.mean(f1_articles):.3f} [{min(f1_articles):.3f}-{max(f1_articles):.3f}]")
print(f"newspaper-held:  {np.mean(f1_newspapers):.3f} [{min(f1_newspapers):.3f}-{max(f1_newspapers):.3f}]")
print(f"gb     ceiling:  {np.mean(gb_ceilings):.3f} [{min(gb_ceilings):.3f}-{max(gb_ceilings):.3f}]")
chance_mean, chance_lo, chance_hi = np.mean(chances), np.quantile(chances, .05), np.quantile(chances, .95)
print(f"chance:  {chance_mean:.3f} [{chance_lo:.3f}-{chance_hi:.3f}]")
print("\nmean confusion (true class in rows):\n", pd.concat(cms).groupby(level=0).mean().round(3).to_string())

f1_per_genre = [separability_per_genre(df, FA_FEATURES, model=MODEL, n_per_class=None, seed=seed, group_col="newspaper") for seed in range(N_SEEDS)]
print("\nper-genre F1:\n", pd.concat(f1_per_genre, axis=1).mean(axis=1).round(3).to_string())

# %%

# --- 1b. which features do the separating? ---------------------------------

N_SPLITS = 5
cols = FA_FEATURES + ["category", "newspaper"]
bal = _balance(df[cols].dropna(), "category", None, 0)
X, yv = bal[FA_FEATURES].to_numpy(), bal["category"].astype(str).to_numpy()
groups = bal["newspaper"].to_numpy()
runs = []
for seed in range(N_SEEDS):
      for tr, _ in StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, yv, groups=groups):
            if len(tr) < 0.6 * len(X): continue
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], yv[tr])
            runs.append(pd.DataFrame(pipe.named_steps["logisticregression"].coef_.T, index=FA_FEATURES, columns=pipe.classes_))

print(f"{len(runs)} folds retained of {5 * N_SPLITS}")
coefs = sum(runs) / len(runs)
coef_sd = pd.concat([r.stack() for r in runs], axis=1).std(axis=1).unstack()
coef_sd = coef_sd.reindex(index=coefs.index, columns=coefs.columns)
print(f"max SD: {coef_sd.values.max():.4f}")
print(coefs.round(2).to_string())
print(f"\nmax SD across folds: {coef_sd.values.max():.4f}")
print("\ntop 3 discriminative features per genre:")
for c in coefs.columns: print(f"  {c:15s}", coefs[c].abs().nlargest(3).index.tolist())

# %%

GROUPS = {"Register/style": REGISTER + AFFECT, "Syntax": COMPLEXITY, "Diversity": DIVERSITY}
PRETTY = DISPLAY_NAMES.copy()

feats = [f for grp in GROUPS.values() for f in grp]
m_all, s_all = coefs.loc[feats], coef_sd.loc[feats]
classes = [c for c in CATEGORIES if c in coefs.columns]
y = np.arange(len(feats))[::-1]

palette = [CATEGORY_COLORS[c] for c in classes]
fig, axes = plt.subplots(1, len(classes), figsize=(2.2 * len(classes), 5.5), sharey=True, dpi=300)
lim = np.abs(m_all.values).max() * 1.35

for ax, cls, color in zip(axes, classes, palette):
    m, s = m_all[cls].to_numpy(), s_all[cls].to_numpy()
    ok = np.abs(m) >= s
    ax.errorbar(m, y, xerr=s, fmt="none", ecolor="black", elinewidth=0.5, capsize=8, capthick=0.5, zorder=1)
    ax.scatter(m[ok], y[ok], color=color, edgecolor="black", linewidth=0.8, s=45, zorder=2)
    ax.scatter(m[~ok], y[~ok], color="white", edgecolor="black", linewidth=0.8, s=45, zorder=2)
    ax.axvline(0, color="0.2", ls="--", lw=0.8, zorder=0.5)
    cum = 0
    for grp in list(GROUPS)[:-1]:
        cum += len(GROUPS[grp]); ax.axhline(len(feats) - cum - 0.5, color="0.3", lw=0.7)
    ax.set_title(cls.title(), fontsize=10); ax.set_xlim(-lim, lim)
    ax.grid(True, color="0.6", lw=0.3, alpha=0.5, zorder=0); ax.set_axisbelow(True)
axes[0].set_yticks(y); axes[0].set_yticklabels([PRETTY.get(f, f) for f in feats], fontsize=8)
plt.tight_layout()
plt.savefig(FIGS_PATH / "logit_coefs.pdf", bbox_inches="tight")
plt.show()
print(f"\nhollow markers (|mean| < SD): {(np.abs(m_all.values) < s_all.values).sum()} of {m_all.size}")

# %%

# how many of the 150 splits were retained? (and SD among them)
print(f"{len(runs)} folds retained of {N_SEEDS * N_SPLITS}")   # was 5 * N_SPLITS
print(f"max SD: {coef_sd.values.max():.4f}")

# %%

GROUP_ORDER = ["Register/style", "Syntax", "Diversity"]
COL_ORDER = ["Advertisement", "International", "National", "fiction"]

def coefs_to_latex(coefs, sd, top_n=3):
    top = {c: set(coefs[c].abs().nlargest(top_n).index) for c in COL_ORDER}

    def fmt(v, bold):
        s = ("$-$" if v < 0 else "") + f"{abs(v):.2f}"
        return f"\\textbf{{{s}}}" if bold else s

    lines = []
    for grp in GROUP_ORDER:
        lines.append(f"\\multicolumn{{{len(COL_ORDER)+1}}}{{l}}"
                     f"{{\\textit{{{grp}}}}} \\\\")
        for f in sorted(GROUPS[grp], key=lambda x: -coefs.loc[x].abs().max()):
            cells = " & ".join(fmt(coefs.loc[f, c], f in top[c]) for c in COL_ORDER)
            lines.append(f"\\hspace{{1em}} {PRETTY[f]:<24} & {cells} \\\\")
        lines.append("\\midrule")

    lines.append("\\textbf{Max $|\\beta|$} & "
                 + " & ".join(f"{coefs[c].abs().max():.2f}" for c in COL_ORDER) + " \\\\")
    return "\n".join(lines)

print(coefs_to_latex(coefs, coef_sd))
print(f"\n% max SD across folds: {coef_sd.values.max():.3f}")


# %%

# --- 1c. where does macro-F1 saturate? -------------------------------------

sat = saturation_curve(df[df.year >= year_cutoff], FA_FEATURES, model=MODEL, seeds=range(N_SEEDS), group_col="newspaper")
print(sat.round(3).to_string(index=False))
print("\ngain per doubling:")
print((sat["mean"].diff() / np.log2(sat["n_per_class"]).diff()).round(4).to_string())
plt.figure(figsize=(5, 3))
plt.fill_between(sat["n_per_class"], sat["lo"], sat["hi"], alpha=0.25)
plt.plot(sat["n_per_class"], sat["mean"], marker="o", ms=3)
plt.xscale("log"); plt.xlabel("n per class"); plt.ylabel("Macro-F1"); plt.tight_layout(); plt.show()

# %%

# --- 2. separability over time ---------------------------------------------

window = 30
step = 5
min_per_class = 300
seeds = range(N_SEEDS)

roll = separability_rolling(df[df.year >= year_cutoff], FA_FEATURES, window=window, step=step, model=MODEL, seeds=seeds, min_per_class=min_per_class, group_col="newspaper")
print(f"n per class: {roll.n_per_class.iloc[0]}")
print(roll.round(3).to_string(index=False))

roll_all = separability_rolling(df, FA_FEATURES, window=window, step=step, model=MODEL, seeds=seeds, min_per_class=min_per_class, group_col="newspaper")
print(f"n per class: {roll_all.n_per_class.iloc[0]}")
print(roll_all.round(3).to_string(index=False))

chance_roll = chance_rolling(df[df.year >= year_cutoff], FA_FEATURES, window=window, step=step, min_per_class=min_per_class, seeds=seeds, group_col="newspaper")

dd = df.dropna(subset=FA_FEATURES + ["category", "year"])
span = dd[dd.year.between(1780, 1839)].groupby("newspaper").year.agg(["min", "max"])
panel = span[(span["min"] <= 1785) & (span["max"] >= 1835)].index.tolist()
panel = [n for n in panel if n != "Københavns Adresseavis"]
print(f"{len(panel)} continuous newspapers")
print(panel)

roll_panel = separability_rolling(dd[dd.newspaper.isin(panel)], FA_FEATURES, window=window, step=step, model=MODEL, seeds=seeds, min_per_class=min_per_class)
print(roll_panel.round(3).to_string(index=False))

# %%

sns.set_style("whitegrid")
color_full = "grey"
color_panel = "#d4499a"
color_subset = "black"

plt.figure(figsize=(9, 3.5))
plt.fill_between(roll["mid"], roll["lo"], roll["hi"], alpha=0.2, color=color_subset)
plt.plot(roll["mid"], roll["mean"], marker="o", ms=3, label=f"1740+ (n={roll.n_per_class.iloc[0]:,}/class)", color=color_subset)
plt.fill_between(roll_all["mid"], roll_all["lo"], roll_all["hi"], alpha=0.2, color=color_full)
plt.plot(roll_all["mid"], roll_all["mean"], marker="o", ms=3, label=f"full corpus (n={roll_all.n_per_class.iloc[0]:,}/class)", color=color_full)
plt.fill_between(roll_panel["mid"], roll_panel["lo"], roll_panel["hi"], alpha=0.2, color=color_panel)
plt.plot(roll_panel["mid"], roll_panel["mean"], marker="o", ms=3, label=f"panel (n={roll_panel.n_per_class.iloc[0]:,}/class)", color=color_panel)

plt.fill_between(chance_roll["mid"], chance_roll["lo"], chance_roll["hi"], color="grey", alpha=0.1, lw=0)
plt.plot(chance_roll["mid"], chance_roll["mean"], color="grey", ls=":", lw=1.5, label="permuted labels")

plt.legend(frameon=False, fontsize=12, loc="lower left")
plt.ylim(bottom=min(chance_roll["lo"].min(), roll["lo"].min()) - 0.01)
plt.xticks(np.arange(1730, 1841, 10))
plt.xlabel(f"Window midpoint ({window}-year window, {step}-year step)")
plt.ylabel("Macro-F1")
plt.tight_layout()
plt.savefig(FIGS_PATH / "separability_rolling.pdf", dpi=300)
plt.show()

# %%

# zoomed version
plt.figure(figsize=(7, 3), dpi=500)
for r, lab, c in [(roll, "1740+", "#888888"), (roll_panel, f"{len(panel)} continuous newspapers", color_panel)]:
    plt.fill_between(r["mid"], r["lo"], r["hi"], alpha=0.2, color=c, lw=0)
    plt.plot(r["mid"], r["mean"], color=c, lw=1, label=lab)

plt.plot(chance_roll["mid"], chance_roll["mean"], color="grey", ls=":", lw=1.5)
plt.fill_between(chance_roll["mid"], chance_roll["lo"], chance_roll["hi"], color="grey", alpha=0.1, lw=0)
plt.text(chance_roll["mid"].iloc[-1], chance_roll["mean"].iloc[-1] + 0.006, "permuted labels", ha="right", va="bottom", fontsize=10, color="grey")
plt.legend(frameon=False, fontsize=10, loc="center right")
plt.xlabel(f"Window midpoint ({window}-year window, {step}-year step)")
plt.ylabel("Macro-F1")
plt.ylim(0.22, 0.68)
plt.tight_layout()
plt.savefig(FIGS_PATH / "separability_rolling_panel.pdf", dpi=300)
plt.show()

# %%

# why do we have a dip around 1800?
comp = dd.groupby(dd.year // 10 * 10).newspaper.agg(["nunique", "size"])
print(comp)
for dec in range(1780, 1840, 10):
    print(dec, sorted(dd[dd.year // 10 * 10 == dec].newspaper.unique()))

# %%

# --- 3. which genre drives the change? -------------------------------------

panel_df = df[(df.year >= year_cutoff) & (df.newspaper != "Københavns Adresseavis")]
pg, pg_lo, pg_hi = per_genre_rolling(panel_df, FA_FEATURES, window=window, step=step, model=MODEL, min_per_class=min_per_class, group_col="newspaper")
panel3 = [p for p in panel if p != "Københavns Adresseavis"]
pgp, pgp_lo, pgp_hi = per_genre_rolling(df[df.newspaper.isin(panel3)], FA_FEATURES, window=window, step=step, model=MODEL, min_per_class=min_per_class)

print(f"n per class: {pg.attrs['n_per_class']} / {pgp.attrs['n_per_class']}")
print("\nall newspapers:\n", pg.round(3).to_string())
print("\ncontinuous panel:\n", pgp.round(3).to_string())

# %%

fig, axes = plt.subplots(1, 4, figsize=(13, 2.2), sharey=True)
for ax, g in zip(axes, pg.columns):
    ax.fill_between(pg.index, pg_lo[g], pg_hi[g], color="#888888", alpha=0.25, lw=0)
    ax.plot(pg.index, pg[g], color="#888888", lw=1.2, label="all")
    ax.fill_between(pgp.index, pgp_lo[g], pgp_hi[g], color=color_panel, alpha=0.25, lw=0)
    ax.plot(pgp.index, pgp[g], color=color_panel, lw=1.2, label=f"{len(panel)} continuous newspapers")
    ax.set_title(g.title(), fontsize=11); ax.set_xticks(np.arange(1760, 1831, 10))
axes[0].set_ylabel("F1"); axes[0].set_ylim(0.35, 0.75); axes[0].legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(FIGS_PATH / "per_genre_rolling.pdf", dpi=500); plt.show()

# %%

# --- 3b. do the discriminating features drift, or just weaken? -------------

df["category"] = pd.Categorical(df["category"], categories=CATEGORIES)
WINDOW, STEP, SEEDS, N_PER_CLASS = 30, 10, seeds, 2500
PAIRS = [(1740, 1820), (1760, 1800)]
starts = list(range(1740, 1850 - WINDOW + 1, STEP))

rows = {}
for s in starts:
    g = df[df.year.between(s, s + WINDOW - 1)]
    for sd in SEEDS:
        bal = _balance(g[FA_FEATURES + ["category"]].dropna(), "category", N_PER_CLASS, sd)
        Xs = StandardScaler().fit_transform(bal[FA_FEATURES])
        lr = LogisticRegression(max_iter=2000).fit(Xs, bal["category"].astype(str))
        c = pd.DataFrame(lr.coef_.T, index=FA_FEATURES, columns=lr.classes_)
        for genre in c.columns: rows.setdefault((genre, s), []).append(c[genre])

def _sim(genre, a, b):
    return cosine_similarity(np.array(rows[(genre, a)]), np.array(rows[(genre, b)])).mean()

def _ceiling(genre, s):
    m = cosine_similarity(np.array(rows[(genre, s)]))
    return m[np.triu_indices_from(m, k=1)].mean()

recs = []
for genre in df["category"].cat.categories:
    r = {"genre": genre}
    for a, b in PAIRS:
        ceil = np.mean([_ceiling(genre, a), _ceiling(genre, b)])
        sim = _sim(genre, a, b)
        r[f"ceil {a}v{b}"] = round(ceil, 3); r[f"cos {a}v{b}"] = round(sim, 3); r[f"drift {a}v{b}"] = round(ceil - sim, 3)
    recs.append(r)

res = pd.DataFrame(recs).set_index("genre")
print(res.to_string())
for genre, r in res.iterrows():
    cells = " & ".join(f"{r[f'{k} {a}v{b}']:.2f}" for a, b in PAIRS for k in ("ceil", "cos", "drift"))
    print(f"{genre} & {cells} \\\\")

print("\nper-window ceilings:")
print(pd.DataFrame({g: {s: round(_ceiling(g, s), 3) for s in starts} for g in df["category"].cat.categories}).to_string())

# %%

# seed-pair distributions together rather than one panel per pair
genre = "National"
a, b = 1760, 1800
A, B = np.array(rows[(genre, a)]), np.array(rows[(genre, b)])
within = np.concatenate([cosine_similarity(A)[np.triu_indices(len(A), 1)], cosine_similarity(B)[np.triu_indices(len(B), 1)]])
across = cosine_similarity(A, B).ravel()

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(5.5, 2.5), dpi=300)
ax.scatter(within, np.full_like(within, 1) + rng.uniform(-.08, .08, len(within)), s=10, alpha=.1, color="#888888", label="within window")
ax.scatter(across, np.full_like(across, 0) + rng.uniform(-.08, .08, len(across)), s=10, alpha=.1, color="#1f4fd8", label="across windows")


for v, y in [(within.mean(), 1), (across.mean(), 0)]:
    ax.plot([v, v], [y - .14, y + .14], color="black", lw=1.8, zorder=3)
ax.annotate("", xy=(within.mean(), .5), xytext=(across.mean(), .5), arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
ax.text((within.mean() + across.mean()) / 2, .59, f"drift = {within.mean() - across.mean():.3f}", ha="center", fontsize=9)
ax.set_yticks([0, 1]); ax.set_yticklabels(["across\nwindows", "within\nwindow"], fontsize=8)
ax.set_ylim(-.35, 1.35); ax.grid(axis="x", color="0.9", lw=.4); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
plt.tight_layout(); plt.savefig(FIGS_PATH / "drift_illustration.pdf", bbox_inches="tight"); plt.show()

# %%

# --- 3c. which genres get mistaken for which? ------------------------------

def confusion_over_time(data, features, window=30, step=5, min_per_class=300, model="logit", seeds=seeds, n_per_class=None, label="category", year_col="year", group_col=None):
    cols = features + [label, year_col] + ([group_col] if group_col else [])
    d = data[cols].dropna()
    y0, y1 = int(d[year_col].min()), int(d[year_col].max())
    ok = {}
    for s in range(y0, y1 - window + 2, step):
        c = d[d[year_col].between(s, s + window - 1)][label].value_counts()
        if len(c) == d[label].nunique() and c.min() >= min_per_class: ok[s] = int(c.min())
    n_fixed = n_per_class or min(ok.values())
    mean, lo, hi = {}, {}, {}
    for s in sorted(ok):
        g = d[d[year_col].between(s, s + window - 1)]
        cs = [separability(g, features, label, model=model, n_per_class=n_fixed, seed=sd, group_col=group_col)[2] for sd in seeds]
        stack = np.stack([c.to_numpy() for c in cs])
        m = s + window // 2
        idx, col = cs[0].index, cs[0].columns
        mean[m] = pd.DataFrame(stack.mean(0), idx, col)
        lo[m] = pd.DataFrame(np.quantile(stack, .05, axis=0), idx, col)
        hi[m] = pd.DataFrame(np.quantile(stack, .95, axis=0), idx, col)
    return mean, lo, hi

# %%

cms, cm_lo, cm_hi = confusion_over_time(df[df.year >= 1740], FA_FEATURES, model=MODEL, min_per_class=min_per_class, group_col="newspaper")
pairs = [("National", "International"), ("International", "National"), ("National", "Advertisement"), ("National", "fiction")]
tr = pd.DataFrame({f"{a[:4]}->{b[:4]}": {m: cm.loc[a, b] for m, cm in cms.items()} for a, b in pairs})
print(tr.round(3).to_string())

order = list(cms[min(cms)].index)
mids = sorted(cms)
n = len(order)

# %%

fig, axes = plt.subplots(n, n, figsize=(7.5, 6), sharex=True, sharey=True, dpi=300)
for i, a in enumerate(order):
    for j, b in enumerate(order):
        ax = axes[i, j]
        col = "#555555" if i == j else "#1f4fd8"
        ax.fill_between(mids, [cm_lo[m].loc[a, b] for m in mids], [cm_hi[m].loc[a, b] for m in mids], color=col, alpha=.2, lw=0)
        ax.plot(mids, [cms[m].loc[a, b] for m in mids], lw=1.3, color=col)
        if i == j: ax.set_facecolor("#f6f6f6")
        else: ax.grid(color="0.9", lw=.4); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=6)
        if i == 0: 
            ax.set_title(b, fontsize=8)
        if j == 0: ax.set_ylabel(a, fontsize=8)
fig.supxlabel("Window midpoint", fontsize=9); fig.supylabel("Fraction of true class predicted as column", fontsize=9)
fig.suptitle("predicted →", fontsize=8, x=.55, y=.99)
plt.tight_layout(rect=[.02, .02, 1, .97]); plt.savefig(FIGS_PATH / "confusion_grid.pdf", bbox_inches="tight"); plt.show()

# %%

print("\nNational -> International, with seed range:")
for m in sorted(cms):
    print(f"  {m}  {cms[m].loc['National','International']:.3f} [{cm_lo[m].loc['National','International']:.3f}-{cm_hi[m].loc['National','International']:.3f}]")

# %%

ax = pd.DataFrame({
    "National → International": [cms[m].loc["National", "International"] for m in mids],
    "International → National": [cms[m].loc["International", "National"] for m in mids],
    "National → Advertisement": [cms[m].loc["National", "Advertisement"] for m in mids],
    "National → fiction": [cms[m].loc["National", "fiction"] for m in mids],
}, index=mids).plot(figsize=(6, 3), lw=1.3)
ax.set_ylabel("Fraction misclassified"); ax.set_xlabel("Window midpoint"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(FIGS_PATH / "confusion_news_pair.pdf", dpi=300); plt.show()

# %%

# --- 3d. do the news genres' profiles converge? ----------------------------

coef_vectors = rows
#del rows

def between_genre_sim(g1, g2, windows, coef_dict):
    out = {}
    for s in windows:
        A = np.array(coef_dict[(g1, s)]); B = np.array(coef_dict[(g2, s)])
        out[s] = cosine_similarity(A, B).mean()
    return out

# def between_genre_sim(g1, g2, windows, coef_dict):
#     out = {}
#     for s in windows:
#         a = np.mean(coef_dict[(g1, s)], axis=0)
#         b = np.mean(coef_dict[(g2, s)], axis=0)
#         out[s] = float(cosine_similarity([a], [b])[0, 0])
#     return out

windows = sorted({s for _, s in coef_vectors})
pairs = [("National", "International"), ("National", "Advertisement"), ("National", "fiction"), ("International", "Advertisement"), ("International", "fiction"), ("Advertisement", "fiction")]
conv = pd.DataFrame({f"{a}-{b}": between_genre_sim(a, b, windows, coef_vectors)
                     for a, b in pairs})
print(conv.round(3).to_string())

ax = conv.plot(figsize=(6.5, 3.5), lw=1.8)
ax.axhline(0, color="0.7", lw=2, ls="dashed", zorder=0)
ax.set_ylabel("Cosine similarity between profiles"); ax.set_xlabel("Window start")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

plt.tight_layout(); plt.savefig(FIGS_PATH / "between_genre_convergence.pdf", dpi=300); plt.show()



fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True, dpi=300)

nat_pairs = [c for c in conv.columns if c.startswith("National")]
oth_pairs = [c for c in conv.columns if not c.startswith("National")]
palette = sns.color_palette("colorblind", 3)

for ax, cols, title in [
    (axes[0], nat_pairs, "National with"),
    (axes[1], oth_pairs, "Pairs without National"),
]:
    ax.axhline(0, color="0.75", lw=1, ls="--", zorder=0)
    for c, col in zip(cols, palette):
        lab = c.split("-")[1] if cols is nat_pairs else c.replace("-", "–")
        ax.plot(conv.index, conv[c], lw=1.6, color=col, label=lab)
    ax.set_xlabel("Window start")
    ax.grid(color="0.92", lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), handlelength=1.2,
              columnspacing=1.0, title=title, title_fontsize=9)

axes[0].set_ylabel("Cosine similarity between profiles")
plt.tight_layout()
plt.savefig(FIGS_PATH / "between_genre_convergence.pdf", bbox_inches="tight")
plt.show()

# %%

# another way to plot rolling cosim

def between_genre_sim(g1, g2, windows, coef_dict):
    mean, lo, hi = {}, {}, {}
    for s in windows:
        A = np.array(coef_dict[(g1, s)]); B = np.array(coef_dict[(g2, s)])
        sims = cosine_similarity(A, B).ravel()
        mean[s] = sims.mean()
        lo[s] = np.percentile(sims, 5)
        hi[s] = np.percentile(sims, 95)
    return pd.DataFrame({"mean": mean, "lo": lo, "hi": hi})

windows = sorted({s for _, s in coef_vectors})
pairs = [("National", "International"), ("National", "Advertisement"), ("National", "fiction"),
         ("International", "Advertisement"), ("International", "fiction"), ("Advertisement", "fiction")]


conv = {f"{a}-{b}": between_genre_sim(a, b, windows, coef_vectors) for a, b in pairs}

print(pd.DataFrame({k: v["mean"] for k, v in conv.items()}).round(3).to_string())

# %%

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True, dpi=300)

nat_pairs = [c for c in conv if c.startswith("National")]
oth_pairs = [c for c in conv if not c.startswith("National")]

STYLES = {
    "International-Advertisement": (":", "0.25"),
    "International-fiction":       ("--", "0.45"),
    "Advertisement-fiction":       ("-.", "0.6"),
}

for ax, cols, title, split in [
    (axes[0], nat_pairs, "National with", True),
    (axes[1], oth_pairs, "Pairs without National", False),
]:
    ax.axhline(0, color="0.75", lw=1, ls="--", zorder=0)
    for c in cols:
        r = conv[c]
        if split:
            other = c.split("-")[1]
            col, ls, lab = CATEGORY_COLORS[other], "-", DISPLAY_NAMES.get(other, other)
            ax.fill_between(r.index, r["lo"], r["hi"], color=col, alpha=.15, lw=0)
        else:
            ls, col = STYLES[c]
            lab = c.replace("-", "–")
        ax.plot(r.index, r["mean"], lw=1.6, color=col, ls=ls, label=lab)
    ax.set_xlabel("Window start")
    ax.grid(color="0.92", lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), handlelength=1.8,
              title=title, title_fontsize=9)

axes[0].set_ylabel("Cosine similarity between profiles")
plt.tight_layout()
plt.savefig(FIGS_PATH / "between_genre_convergence.pdf", bbox_inches="tight")
plt.show()


# %%

# as latex

means = pd.DataFrame({k: v["mean"] for k, v in conv.items()})
order = ["National-International", "National-Advertisement", "National-fiction",
         "International-Advertisement", "International-fiction",
         "Advertisement-fiction"]
means = means[order]

delta = (means.loc[1820] - means.loc[1770]).to_frame("1770--1820").T
tab = pd.concat([means, delta])

def fmt(v):
    return ("$-$" if v < 0 else "") + f"{abs(v):.2f}"

for idx, row in tab.iterrows():
    cells = " & ".join(fmt(v) for v in row)
    label = f"\\textbf{{{idx}}}" if isinstance(idx, str) else str(idx)
    print(f"{label} & {cells} \\\\")

# %%

norms = pd.DataFrame({g: {s: np.linalg.norm(np.array(coef_vectors[(g, s)]).mean(0)) for s in starts} for g in ["National", "International", "Advertisement", "fiction"]})
print(norms.round(3).to_string())

# %%

panel3 = [p for p in panel if p != "Københavns Adresseavis"]

def coef_norms(data, starts=starts, window=WINDOW, seeds=SEEDS, n_per_class=N_PER_CLASS):
    mean, lo, hi = {}, {}, {}
    for s in starts:
        g = data[data.year.between(s, s + window - 1)]
        if g.category.value_counts().min() < n_per_class: continue
        vs = {}
        for sd in seeds:
            bal = _balance(g[FA_FEATURES + ["category"]].dropna(), "category", n_per_class, sd)
            Xs = StandardScaler().fit_transform(bal[FA_FEATURES])
            lr = LogisticRegression(max_iter=2000).fit(Xs, bal["category"].astype(str))
            for k, genre in enumerate(lr.classes_): vs.setdefault(genre, []).append(np.linalg.norm(lr.coef_[k]))
        mean[s] = {g_: np.mean(v) for g_, v in vs.items()}
        lo[s] = {g_: np.quantile(v, .05) for g_, v in vs.items()}
        hi[s] = {g_: np.quantile(v, .95) for g_, v in vs.items()}
    return pd.DataFrame(mean).T, pd.DataFrame(lo).T, pd.DataFrame(hi).T

n_panel = int(df[df.newspaper.isin(panel3)].groupby(df.year // 10 * 10).category.apply(lambda s: s.value_counts().min()).min())
print(f"n per class: all {N_PER_CLASS}, panel {n_panel}")

na, na_lo, na_hi = coef_norms(df)
np_, np_lo, np_hi = coef_norms(df[df.newspaper.isin(panel3)], n_per_class=n_panel)

# %%

fig, axes = plt.subplots(1, 4, figsize=(10, 2.4), sharey=True, dpi=300)
for ax, g in zip(axes, na.columns):
    ax.fill_between(na.index, na_lo[g], na_hi[g], color="gray", alpha=.2, lw=0)
    ax.plot(na.index, na[g], color="gray", lw=1.4)
    ax.set_title(g.title(), fontsize=9); ax.grid(color="0.92", lw=.4); ax.set_axisbelow(True); ax.set_ylim(0, 1.7)
axes[0].set_ylabel(r"$\|\beta\|$")
ymin, ymax = axes[0].get_ylim()
ymin = np.floor(ymin * 2) / 2; ymax = np.ceil(ymax * 2) / 2
axes[0].set_yticks(np.arange(ymin, ymax + 0.5, 0.5))
fig.supxlabel("Window start", fontsize=9)
plt.tight_layout(); plt.savefig(FIGS_PATH / "coef_norms.pdf", bbox_inches="tight"); plt.show()
# %%