# %%
# ============================================================
# SETUP
# ============================================================
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys
sys.path.append(str(Path.cwd().parent / "src"))  # Path to src folder
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

from factor_sweep import run_factor_sweep

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


CWD = Path.cwd().parent
DATA_PATH = CWD / "data"
FIGS_PATH = CWD / "figs"

# NOTES
# - for our next iteration, we want to try and drop pre-1740 data

# %%
# ============================================================
# 1. LOAD DATA & DEFINE FEATURE SETS
# ============================================================
data_sheet_path = DATA_PATH / "usage_features_13-07-26_with_nominal.parquet"
df = pd.read_parquet(data_sheet_path)
print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {data_sheet_path}")

df['year_centered'] = df['year'] - df['year'].mean()
df['category'] = pd.Categorical(
    df['category'], categories=['National', 'International', 'Advertisement', 'fiction']
)

# Full descriptive feature set (Table: linguistic_features in the paper).
# lix/rix are included here for descriptive/PCA purposes but EXCLUDED from
# the factor analysis (see Section 4 -- both are collinear with
# avg_sentlen by formula construction, not just empirically).
features = [
    'nominal_verb_ratio', 'of_ratio', 'that_ratio',
    'present_tense_ratio', 'passive_ratio',
    'adjective_adverb_ratio', 'personal_pronoun_ratio',
    'avg_wordlen', 'avg_sentlen', 'lix', 'rix',
    'cttr_resid', 'noun_ttr_resid', 'verb_ttr_resid',
    'avg_ndd', 'std_ndd', 'avg_mdd', 'std_mdd',
    'function_word_ratio',
    'semantic_sentiment_standardized', 'semantic_sentiment_log_var_ratio',
]

# Feature set actually used for factor analysis (lix/rix excluded)
fa_features = [f for f in features if f not in ('lix', 'rix')]

norm_features = [f for f in features if f.endswith('_norm') and f + '_norm' in features]

print(f"NaN counts:\n{df[features].isna().sum()}")

# %%
# ============================================================
# 2. FIGURE: article volume over time (fig:dists in the paper)
# ============================================================
palette = sns.color_palette("hsv_r", n_colors=5)
palette_list = list(palette)
palette_list[1] = (0.5, 0.9, 0.7)  # light green

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 4), sharex=True,
    gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.2}, dpi=500
)
sns.set_style("whitegrid")

sns.histplot(data=df, x='year', discrete=True, color="grey", ax=ax1, kde=True, alpha=0.6)
ax1.set_ylabel("Total Articles")

sns.histplot(
    data=df, x='year', hue='category', multiple='fill', discrete=True,
    legend=True, ax=ax2, palette=palette_list, alpha=0.8
)
ax2.set_ylabel("Proportion")
ax2.set_xlabel("Year")
sns.move_legend(ax2, "upper right", bbox_to_anchor=(.92, .3), frameon=False, title="Category", ncol=4)
plt.tight_layout()
plt.savefig(FIGS_PATH / "article_volume_over_time.pdf", dpi=500)
plt.show()

print(df['category'].value_counts())

#%%
# ============================================================
# 3. Correlation matrix for the full feature set (Table: corr_matrix in the paper)
# ============================================================

threshold = 0.05 # |rho| threshold for reporting "significant" correlations

corr_by_category = {}
n_by_category = {}
n_over_threshold = {}
sum_over_threshold = {}

corr_df_storage = {}

df_post_1715 = df[df['year'] >= 1715]  # optional: exclude thin early years if counts are low (check first!)

for df_tmp in [df, df_post_1715]:
    for cat in df_tmp['category'].cat.categories:
        cat_df = df_tmp[df_tmp['category'] == cat]
        n_by_category[cat] = len(cat_df)

        corrs = {}
        for feat in features:
            slice_ = cat_df[[feat, 'date_ordinal']].dropna()
            rho, _ = spearmanr(slice_['date_ordinal'], slice_[feat])
            corrs[feat] = rho

        corr_by_category[cat] = corrs
        n_over_threshold[cat] = sum(1 for rho in corrs.values() if abs(rho) >= threshold)
        sum_over_threshold[cat] = sum(abs(rho) for rho in corrs.values() if abs(rho) >= threshold)

        #print(f"\n=== {cat} (n={n_by_category[cat]}) ===")
        #print(pd.Series(corrs).sort_values(key=abs, ascending=False).round(3))
        #print(f"N features with |rho| > 0.1: {n_over_threshold[cat]} / {len(features)}")

    # Combine into one table, ranked by |rho| in Fiction (matches original table's logic)
    corr_table = pd.DataFrame(corr_by_category)
    corr_table['abs_fiction'] = corr_table['fiction'].abs()
    corr_table = corr_table.sort_values('abs_fiction', ascending=False).drop(columns='abs_fiction')

    print("\n=== Combined table, ranked by |rho| in Fiction ===")
    print(corr_table.round(2))
    print(f"\n=== N features with |rho| {threshold} vs. time, by category ===")
    print(pd.Series(n_over_threshold))
    print(f"\n=== Sum of |rho| {threshold} vs. time, by category ===")
    print(pd.Series(sum_over_threshold).round(2))

    corr_df_storage['full' if df_tmp is df else 'post_1715'] = corr_table

# overall correlation matrix (full feature set, all categories combined) with date ordinal
n_over, sum_over = 0, 0
for feat in features:
    slice_ = df[[feat, 'date_ordinal']].dropna()
    rho, _ = spearmanr(slice_['date_ordinal'], slice_[feat])
    print(f"{feat}: rho={rho:.3f}")
    if abs(rho) >= threshold:
        n_over += 1
        sum_over += abs(rho)
print("Rho>threshold, sum of |rho|:", n_over, sum_over)

# %%

# full and post-1715 correlation tables
corr_post = corr_df_storage['post_1715']

cutoff = 15

# order features by strongest absolute correlation
order = corr_post.abs().max(axis=1).sort_values(ascending=False).index
corr_post = corr_post.loc[order][:cutoff]

# feature names
replacements = {"semantic_sentiment_standardized": "Sentiment", "semantic_sentiment_log_var_ratio": "Sentiment SD"}

feat_names = (
    corr_post.index
    .str.replace("_", " ")
    .str.title()
    .to_series()
    .replace(replacements)
    .values)

# mask weak correlations
threshold = 0.05
weak_mask = corr_post.abs() < threshold

plt.figure(figsize=(4.5, 5))
# main heatmap
sns.heatmap(corr_post, cmap="RdGy_r", center=0, annot=True, fmt=".2f", mask=weak_mask, cbar_kws={"label": r"$\rho$"}, cbar=False)
# grey background for weak correlations
sns.heatmap(corr_post, cmap=["lightgrey"], mask=~weak_mask, cbar=False, annot=False, alpha=0.8)
plt.xticks(rotation=90, ha="right")
plt.yticks(ticks=np.arange(len(feat_names)) + 0.5, labels=feat_names, rotation=0)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig(FIGS_PATH / "corr_matrix_post_1715.pdf", dpi=500)
plt.show()

# %%
# how linear are the correlations?
import ruptures as rpt
from statsmodels.nonparametric.smoothers_lowess import lowess

feature = 'avg_sentlen'
categories_to_plot = ['National', 'International', 'Advertisement', 'fiction']

smooth_frac = 0.1     # LOWESS smoothing fraction -- visualization only
pelt_penalty = 5     # ruptures Pelt penalty -- tune this per feature (see note below)
min_year = 1715       # optional: exclude thin early years if counts are low (check first!)

for cat in categories_to_plot:
    cat_df = df[df['category'] == cat]
    yearly_mean = cat_df.groupby('year')[feature].mean().dropna()

    # Optional: drop sparse early years so a handful of noisy articles
    # doesn't get treated as a real structural break. Check article
    # counts per year before deciding whether/where to cut.
    yearly_mean = yearly_mean[yearly_mean.index >= min_year]

    # --- Changepoint detection on the RAW (unsmoothed) series ---
    algo = rpt.Pelt(model="rbf").fit(yearly_mean.values)
    breakpoints = algo.predict(pen=pelt_penalty)
    breakpoint_years = [int(yearly_mean.index[i]) for i in breakpoints[:-1]]

    # --- Smoothed line for the visual only ---
    smoothed = lowess(yearly_mean.values, yearly_mean.index, frac=smooth_frac)

    plt.figure(figsize=(8, 4))
    plt.plot(yearly_mean.index, yearly_mean.values, color='grey', alpha=0.3, label='Yearly mean (raw)')
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='C0', linewidth=2, label='LOWESS smoothed')
    for bp_year in breakpoint_years:
        plt.axvline(bp_year, color='red', linestyle='--', alpha=0.7)
    plt.title(f"{cat}: {feature} with detected breakpoints")
    plt.xlabel("Year")
    plt.ylabel(feature)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS_PATH / f"changepoints_{feature}_{cat}.pdf", dpi=300)
    plt.show()

    print(f"{cat}: breakpoint years = {breakpoint_years}")

# %%
# ============================================================
# 4. PCA (paper text: "no single Complexity component -- 11 of 17
#    components needed for 90% variance")
# ============================================================
X_pca = StandardScaler().fit_transform(df[features].fillna(df[features].median()))
pca = PCA()
pca.fit(X_pca)

# how many component do we need to get to 90% variance explained?
print("Cumulative variance explained:")
for i, cum_var in enumerate(pca.explained_variance_ratio_.cumsum()):
    print(f"Comp. {i+1}: {cum_var:.4f}")
    if cum_var >= 0.9:
        print(f"Reached 90% variance explained at component {i+1}")
        break

pca_loadings = pd.DataFrame(pca.components_.T, index=features, columns=[f"PC{i+1}" for i in range(len(features))])
print(pca_loadings.iloc[:, :3].round(3))

# %%
# ============================================================
# 5. VIF check justifying lix/rix exclusion from factor analysis
#    (both are collinear with avg_sentlen BY FORMULA CONSTRUCTION --
#    LIX/RIX both use sentence length as a direct term)
# ============================================================
X_vif_check = df[fa_features].dropna()
X_vif_const = sm.add_constant(X_vif_check)
vif_df = pd.DataFrame({
    "feature": X_vif_const.columns,
    "VIF": [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1])]
})
print(vif_df.sort_values("VIF", ascending=False).to_string(index=False))
# Expect all VIFs comfortably under ~3 (aside from the intercept, which is
# not diagnostic) -- confirms lix/rix removal fully resolved the
# collinearity, no longer requiring further variable removal.

# %%
# ============================================================
# 6. FACTOR ANALYSIS: sweep + final 3-factor solution
#    (paper: Bartlett/KMO, promax rotation, 3-factor structure)
# ============================================================
X_final = df[fa_features].dropna()
print(f"Rows retained: {len(X_final)} / {len(df)} ({len(X_final)/len(df):.1%})")

chi_sq, p = calculate_bartlett_sphericity(X_final)
kmo_all, kmo_model = calculate_kmo(X_final)
print(f"Bartlett's p={p:.4f}, KMO={kmo_model:.3f}")

# Systematic sweep 2-8 factors: confirms 3 is the largest solution that is
# BOTH statistically proper (no Heywood cases) AND substantively coherent
# (factors 4-5 are technically proper but fragment avg_sentlen's variance
# across multiple factors rather than revealing new constructs -- see
# methods_checkpoint.md for the full diagnostic).
sweep_df, all_loadings, all_uniquenesses, best_k = run_factor_sweep(
    X_final, fa_features, factor_range=range(2, 9)
)

# cumulative gain
sweep_df['marginal_var_gain'] = sweep_df['cumulative_var'].diff()
print(sweep_df.to_string(index=False))

# %%
# Final reported model
n_factors = 3
fa_final = FactorAnalyzer(n_factors=n_factors, rotation='promax')
fa_final.fit(X_final)

final_loadings = pd.DataFrame(
    fa_final.loadings_, index=fa_features, columns=[f"F{i+1}" for i in range(n_factors)]
)
print(final_loadings.round(2))

variance_df = pd.DataFrame(
    fa_final.get_factor_variance(),
    index=["SS Loadings", "Proportion Var", "Cumulative Var"],
    columns=[f"F{i+1}" for i in range(n_factors)]
)
print(variance_df)

uniquenesses = pd.Series(fa_final.get_uniquenesses(), index=fa_features)
print(uniquenesses.sort_values())

# %%
# ============================================================
# 7. DIAGNOSTIC PLOTS for the FA results section
# ============================================================

# --- Parallel analysis (justifies factor count statistically) ---
def parallel_analysis(X, fa_fitted, n_iter=20):
    n_obs, n_vars = X.shape
    real_ev, _ = fa_fitted.get_eigenvalues()
    random_evs = np.zeros((n_iter, n_vars))
    for i in range(n_iter):
        random_data = np.random.normal(size=(n_obs, n_vars))
        fa_random = FactorAnalyzer(n_factors=n_vars, rotation=None)
        fa_random.fit(random_data)
        random_evs[i], _ = fa_random.get_eigenvalues()
    return real_ev, random_evs.mean(axis=0)

# real_ev, random_ev = parallel_analysis(X_final, fa_final)

# plt.figure(figsize=(7, 5))
# plt.plot(range(1, len(real_ev) + 1), real_ev, marker='o', label='Actual data')
# plt.plot(range(1, len(random_ev) + 1), random_ev, marker='o', linestyle='--', label='Random data (mean)')
# plt.axhline(1, color='grey', linestyle=':', alpha=0.5)
# plt.xlabel("Factor number")
# plt.ylabel("Eigenvalue")
# plt.legend()
# plt.title("Parallel Analysis")
# plt.tight_layout()
# plt.savefig(FIGS_PATH / "parallel_analysis.pdf", dpi=300)
# plt.show()

# --- Loadings heatmap, grouped by dominant factor ---
dominant_factor = final_loadings.abs().idxmax(axis=1)
sort_order = final_loadings.loc[dominant_factor.sort_values().index]

replacements = {"semantic_sentiment_standardized": "Sentiment", "semantic_sentiment_log_var_ratio": "Sentiment SD"}

plt.figure(figsize=(4, 8))
display = sort_order.copy()
display[display.abs() < 0.1] = 0
sns.heatmap(display, cmap='RdGy_r', center=0, annot=True, fmt='.2f', cbar=False)
# change feat names
f_names = (
    sort_order.index
    .to_series()
    .replace(replacements)
    .str.replace("_", " ")
    .str.title()
    .values
)
plt.yticks(ticks=np.arange(len(f_names)) + 0.5, labels=f_names, rotation=0)
plt.tight_layout()
plt.savefig(FIGS_PATH / "fa_loadings_heatmap.pdf", dpi=300)
plt.show()

# --- Communality bar chart ---
communalities = 1 - uniquenesses
plt.figure(figsize=(6, 8))
communalities.sort_values().plot(kind='barh')
plt.xlabel(f"Communality (variance explained by {n_factors} factors)")
plt.tight_layout()
plt.savefig(FIGS_PATH / "fa_communalities.pdf", dpi=300)
plt.show()

# %%
# ============================================================
# 8. GENRE-SPECIFIC MEASUREMENT INVARIANCE (leave-one-out congruence)
#    -- the standalone FA finding: structure is domain-specific
# ============================================================

def tucker_congruence(loadings_a, loadings_b):
    numerator = np.sum(loadings_a * loadings_b)
    denominator = np.sqrt(np.sum(loadings_a**2) * np.sum(loadings_b**2))
    return numerator / denominator


def leave_one_out_congruence(data, genre_col, feats, target_genre, n_factors=3):
    other_data = data[data[genre_col] != target_genre][feats].dropna()
    fa_loo = FactorAnalyzer(n_factors=n_factors, rotation='promax')
    fa_loo.fit(other_data)
    loo_loadings = pd.DataFrame(fa_loo.loadings_, index=feats, columns=[f"F{i+1}" for i in range(n_factors)])

    target_data = data[data[genre_col] == target_genre][feats].dropna()
    fa_target = FactorAnalyzer(n_factors=n_factors, rotation='promax')
    fa_target.fit(target_data)
    target_loadings = pd.DataFrame(fa_target.loadings_, index=feats, columns=[f"F{i+1}" for i in range(n_factors)])

    return loo_loadings, target_loadings


MIN_GENRE_N = 500
pooled_factor_names = [f"F{i+1}" for i in range(n_factors)]

for genre in df['category'].cat.categories:
    subset_n = len(df[df['category'] == genre][fa_features].dropna())
    if subset_n < MIN_GENRE_N:
        print(f"Skipping {genre} (n={subset_n}, too small)")
        continue

    loo_loadings, target_loadings = leave_one_out_congruence(df, 'category', fa_features, genre, n_factors)

    print(f"\n=== {genre}: leave-one-out congruence matrix ===")
    cong_matrix = pd.DataFrame(index=[f"genre_{f}" for f in pooled_factor_names], columns=pooled_factor_names, dtype=float)
    for g_factor in pooled_factor_names:
        for p_factor in pooled_factor_names:
            cc = tucker_congruence(target_loadings[g_factor].values, loo_loadings[p_factor].values)
            cong_matrix.loc[f"genre_{g_factor}", p_factor] = abs(cc)
    print(cong_matrix.round(3))

    best_match = cong_matrix.idxmax(axis=1)
    best_score = cong_matrix.max(axis=1)
    for g_factor in cong_matrix.index:
        print(f"  {g_factor} best matches pooled {best_match[g_factor]} (congruence={best_score[g_factor]:.3f})")

    # Print the genre's own loadings too, for inspecting WHICH features
    # diverge (e.g. Fiction's diversity/register collision)
    #print(f"\n{genre} own loadings (n={subset_n}):")
    #print(target_loadings.round(2))


# %%

def subsample_congruence_test(df, genre_col, feats, large_genre, target_n, n_iter=30, n_factors=3):
    """
    Draws n_iter random subsamples of size target_n from `large_genre`,
    refits the 3-factor model on each, and computes congruence against
    the SAME leave-one-out reference used for Fiction's actual test.
    If Fiction's real congruence falls within this distribution, sample
    size alone plausibly explains the breakdown. If Fiction's congruence
    is systematically lower than this distribution, the breakdown is a
    genuine property of Fiction's content, not just its size.
    """
    other_data = df[df[genre_col] != large_genre][feats].dropna()
    fa_loo = FactorAnalyzer(n_factors=n_factors, rotation='promax')
    fa_loo.fit(other_data)
    loo_loadings = pd.DataFrame(fa_loo.loadings_, index=feats, columns=[f"F{i+1}" for i in range(n_factors)])

    full_genre_data = df[df[genre_col] == large_genre][feats].dropna()
    congruences = []

    for i in range(n_iter):
        sample = full_genre_data.sample(n=target_n, random_state=i)
        fa_sub = FactorAnalyzer(n_factors=n_factors, rotation='promax')
        fa_sub.fit(sample)
        sub_loadings = pd.DataFrame(fa_sub.loadings_, index=feats, columns=[f"F{i2+1}" for i2 in range(n_factors)])

        # best-match congruence per factor, same logic as your existing code
        best_matches = []
        for g_factor in [f"F{i2+1}" for i2 in range(n_factors)]:
            best = max(
                abs(tucker_congruence(sub_loadings[g_factor].values, loo_loadings[p_factor].values))
                for p_factor in [f"F{i2+1}" for i2 in range(n_factors)]
            )
            best_matches.append(best)
        congruences.append(best_matches)

    return np.array(congruences)  # shape: (n_iter, n_factors)


fiction_n = len(df[df['category'] == 'fiction'][fa_features].dropna())

for large_genre in ['National', 'Advertisement', 'International']:
    subsample_results = subsample_congruence_test(df, 'category', fa_features, large_genre, target_n=fiction_n, n_iter=30)
    print(f"\n=== {large_genre} subsampled to n={fiction_n} (30 iterations) ===")
    print(f"Mean congruence per factor: {subsample_results.mean(axis=0).round(3)}")
    print(f"Min congruence observed: {subsample_results.min(axis=0).round(3)}")
    print(f"Fiction's actual congruence: [0.90, 0.72 or 0.56 (worst two)]")  # from your earlier output


# %%
# ============================================================
# 9. MIXED-EFFECTS MODELS on representative features
#    (paper: Table with per-decade estimates by genre)
# ============================================================
representative_features = [
    'avg_sentlen', 'avg_mdd', 'nominal_verb_ratio',
    'personal_pronoun_ratio', 'cttr_resid', 'noun_ttr_resid',
]
control_vars = ['year_centered', 'category', 'n_tokens_for_diversity', 'german_probability', 'newspaper']


def fit_one_model(feature, data, verbose=True, re_formula="~1"):
    """
    re_formula defaults to "~1" (random intercept only). Random-slope
    spec (re_formula="~year_centered") produces a genuinely degenerate
    fit -- see methods_checkpoint.md for the diagnostic. Intercept-only
    still emits a benign Hessian warning at this row count but shows no
    pathological parameter signs.
    """
    needed_cols = [feature] + control_vars
    model_data = data[needed_cols].dropna()
    if verbose:
        print(f"{feature}: {len(model_data)} / {len(data)} rows retained "
              f"({len(model_data)/len(data):.1%})")

    formula = f"{feature} ~ year_centered * category + np.log1p(n_tokens_for_diversity) + german_probability"
    model = smf.mixedlm(formula, data=model_data, groups=model_data["newspaper"], re_formula=re_formula)
    result = model.fit(method="lbfgs")

    if verbose:
        print(f"Converged: {result.converged}")
    return result, model_data


results_summary = []
fitted_models = {}

for feature in representative_features:
    print(f"\n=== Fitting: {feature} ===")
    result, model_data = fit_one_model(feature, df)
    fitted_models[feature] = result

    for term in result.params.index:
        results_summary.append({
            "feature": feature, "term": term,
            "coef": result.params[term], "std_err": result.bse[term],
            "p_value": result.pvalues[term], "converged": result.converged,
            "n_obs": len(model_data),
        })

results_df = pd.DataFrame(results_summary)
results_df.to_csv(DATA_PATH / "mixed_effects_summary.csv", index=False)

# --- Automated degeneracy check across all six random-slope fits ---
# The single-feature pilot showed a specific failure signature: the
# random-intercept variance and random-slope variance collapse to a
# near-identical value, with their covariance landing near 0 -- worth
# checking systematically across all six rather than eyeballing each
# printed summary individually.

print("\n=== Degeneracy check (only applicable to random-slope spec) ===")
for feature, result in fitted_models.items():
    cov_re = result.cov_re
    if cov_re.shape != (2, 2):
        print(f"{feature}: intercept-only model (cov_re shape={cov_re.shape}) -- check not applicable")
        continue
    intercept_var = cov_re.iloc[0, 0]
    slope_var = cov_re.iloc[1, 1]
    cov = cov_re.iloc[0, 1]
    var_ratio = slope_var / intercept_var if intercept_var != 0 else float('nan')
    likely_degenerate = abs(var_ratio - 1) < 0.01 and abs(cov) < 1e-6
    print(f"{feature}: var_ratio={var_ratio:.4f} {'*** LIKELY DEGENERATE ***' if likely_degenerate else ''}")

# --- Per-genre, per-decade slopes table (for the paper's results table) ---
categories = ['National', 'International', 'Advertisement', 'fiction']

def compute_genre_slopes(res_df, feature):
    base = res_df[(res_df['feature'] == feature) & (res_df['term'] == 'year_centered')]['coef'].values[0]
    slopes = {'National': base}
    for cat in categories[1:]:
        term = f"year_centered:category[T.{cat}]"
        interaction = res_df[(res_df['feature'] == feature) & (res_df['term'] == term)]['coef'].values[0]
        slopes[cat] = base + interaction
    return slopes

slope_table = {}
for feature in representative_features:
    slopes = compute_genre_slopes(results_df, feature)
    slope_table[feature] = {cat: slope * 10 for cat, slope in slopes.items()}  # per-decade

slope_df = pd.DataFrame(slope_table).T
print("\nPer-decade slopes by genre:")
print(slope_df.round(4))
slope_df.to_csv(DATA_PATH / "per_decade_slopes.csv")





# %%
 # rerun with post-1715 data only (to check robustness of the main results)

results_summary = []
fitted_models = {}

post_1715_df = df[df['year'] >= 1715]

for feature in representative_features:
    print(f"\n=== Fitting: {feature} ===")
    result, model_data = fit_one_model(feature, post_1715_df)
    fitted_models[feature] = result

    for term in result.params.index:
        results_summary.append({
            "feature": feature, "term": term,
            "coef": result.params[term], "std_err": result.bse[term],
            "p_value": result.pvalues[term], "converged": result.converged,
            "n_obs": len(model_data),
        })

results_df = pd.DataFrame(results_summary)
results_df.to_csv(DATA_PATH / "mixed_effects_summary.csv", index=False)

# --- Automated degeneracy check across all six random-slope fits ---
# The single-feature pilot showed a specific failure signature: the
# random-intercept variance and random-slope variance collapse to a
# near-identical value, with their covariance landing near 0 -- worth
# checking systematically across all six rather than eyeballing each
# printed summary individually.

print("\n=== Degeneracy check (only applicable to random-slope spec) ===")
for feature, result in fitted_models.items():
    cov_re = result.cov_re
    if cov_re.shape != (2, 2):
        print(f"{feature}: intercept-only model (cov_re shape={cov_re.shape}) -- check not applicable")
        continue
    intercept_var = cov_re.iloc[0, 0]
    slope_var = cov_re.iloc[1, 1]
    cov = cov_re.iloc[0, 1]
    var_ratio = slope_var / intercept_var if intercept_var != 0 else float('nan')
    likely_degenerate = abs(var_ratio - 1) < 0.01 and abs(cov) < 1e-6
    print(f"{feature}: var_ratio={var_ratio:.4f} {'*** LIKELY DEGENERATE ***' if likely_degenerate else ''}")

# --- Per-genre, per-decade slopes table (for the paper's results table) ---
categories = ['National', 'International', 'Advertisement', 'fiction']

slope_table = {}
for feature in representative_features:
    slopes = compute_genre_slopes(results_df, feature)
    slope_table[feature] = {cat: slope * 10 for cat, slope in slopes.items()}  # per-decade

slope_df = pd.DataFrame(slope_table).T
print("\nPer-decade slopes by genre:")
print(slope_df.round(4))
slope_df.to_csv(DATA_PATH / "post_1715_per_decade_slopes.csv")
# %%




"""
Effect-size translation for the mixed-effects and correlation results.

Two deliberately DIFFERENT approaches, matched to what's actually valid:

1. Mixed-effects slopes -> expressed as a fraction of the feature's own
   SD, per decade. NOT extrapolated over the full 1666-1849 span, since
   the changepoint-detection work already showed several of these
   features are non-monotonic (rise to a peak ~1720-1750, then decline)
   -- a linear per-decade slope integrated over ~18 decades would
   misrepresent a real curved trend as a straight line, producing
   numbers that don't correspond to anything that actually happened.

2. Correlation table -> rho^2 (variance explained), which is valid
   regardless of the underlying functional form (monotonic or not),
   unlike a naive slope-times-years extrapolation would be.
"""

# %%
# --- 1. SD-relative effect sizes for the six mixed-effects features ---

representative_features = [
    'avg_sentlen', 'avg_mdd', 'nominal_verb_ratio',
    'personal_pronoun_ratio', 'cttr_resid', 'noun_ttr_resid',
]

# Per-decade slopes by genre, from your existing slope_df / mixed_effects_summary.csv
# (reload here if not already in memory)
slope_df = pd.read_csv(DATA_PATH / "per_decade_slopes.csv", index_col=0)

# Compute each feature's own SD across the analysis sample (same filtered
# df used for the mixed-effects models)
feature_sds = {feat: df[feat].std() for feat in representative_features}

effect_size_rows = []
for feat in representative_features:
    sd = feature_sds[feat]
    for genre in slope_df.columns:
        slope = slope_df.loc[feat, genre]
        effect_size_rows.append({
            "feature": feat,
            "genre": genre,
            "slope_per_decade": slope,
            "feature_sd": sd,
            "pct_sd_per_decade": 100 * slope / sd,
        })

effect_size_df = pd.DataFrame(effect_size_rows)
print("=== Effect sizes: % of feature's own SD, per decade ===")
print(effect_size_df.pivot(index="feature", columns="genre", values="pct_sd_per_decade").round(2))

# %%
# --- 2. rho^2 (variance explained) for the correlation table ---
# Valid regardless of monotonicity/linearity -- unlike a slope-times-years
# extrapolation, which would misrepresent the known non-monotonic
# (rise-then-decline) pattern in several features as if it were linear.

# Reuse the corr_table already built earlier (features x genre, Spearman rho)
rho_squared_table = (corr_table ** 2) * 100  # as a percentage
print("\n=== Variance explained (rho^2, %) by feature and genre ===")
print(rho_squared_table.round(1))

print("\nInterpretation: even the single strongest correlations in the "
      "table correspond to roughly this much variance in the feature "
      "explained by publication year alone -- useful for arguing effects")
# %%
