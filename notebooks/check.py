# %%
import pandas as pd
from datasets import load_dataset
import umap
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

import numpy as np
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# %%

CWD = Path.cwd().parent
DATA_PATH = CWD / "data"
FIGS_PATH = CWD / "figs"

# %%

ds = load_dataset("chcaa/eno-newspapers-enriched", split="train", streaming=True, 
                       columns=['id', 'date', 'newspaper', 'predicted_category','fiction_prob','non_fiction_prob','fictionality_tag'])
df_meta = pd.DataFrame(list(ds))
# add fiction to df_meta category based on fictionality_tag
df_meta['predicted_category'] = df_meta.apply(lambda row: 'fiction' if row['fictionality_tag'] == 'fiction' else row['predicted_category'], axis=1)
# remove anything with cat == paratext
df_meta = df_meta[df_meta['predicted_category'] != 'Paratext']
print(f"Total rows after merging and filtering: {len(df_meta)}")

df_meta.head()

# %%
df_feats = pd.read_parquet(DATA_PATH / "features_combined_standardized.parquet")

# # check in on our sentiment feature
# print(df_feats.nsmallest(10, "semantic_sentiment_standardized")[["article_id", "num_sents", "semantic_sentiment_standardized"]])
# print(df_feats.nlargest(10, "semantic_sentiment_standardized")[["article_id", "num_sents", "semantic_sentiment_standardized"]])

# set a num_sents filter
print(f"Total rows before filtering: {len(df_feats)}")
threshold = 2
df_feats = df_feats[df_feats["num_sents"] >= threshold]
print(f"Filtered out articles with fewer than {threshold} sentences.")

# stats
print(f"Total rows: {len(df_feats)}")
print(df_feats.columns)
print(df_feats.describe())

df_feats.head()
# %%

# %%
# rename id to article_id in df_meta to match df_feats
df_meta = df_meta.rename(columns={"id": "article_id"})
df = df_feats.merge(df_meta, on="article_id", how="left")

# see how many have nan in date
nan_date = df['date'].isna().sum()
print(f"Number of rows with NaN in 'date': {nan_date} out of {len(df)} total rows.")
df.head()
# %%

#### defining features ####

features = [ 
       'nominal_verb_ratio', 'of_ratio', 'that_ratio', # perplexity/information related
       'present_tense_ratio','passive_ratio',              # register/style related
       'adjective_adverb_ratio', 'personal_pronoun_ratio', # register/style related
       'avg_wordlen', 'avg_sentlen', 'lix', 'rix',     # readability related
       #'mtld', 
       'cttr', 'noun_ttr', 'verb_ttr',         # lexical diversity
       'avg_ndd', 'std_ndd', 'avg_mdd', 'std_mdd',     # dependency distance
       #'compressrat', 
       'function_word_ratio',
       #'german_probability', 'german_sentence_share',
       'semantic_sentiment_standardized'
      #'n_tokens_for_diversity',
       ]


# normalize fetaures to space 0-1 for better visualization
scaler = MinMaxScaler()

norm_feats = []
for feat in features:
    df[feat + '_norm'] = scaler.fit_transform(df[[feat]].fillna(0))
    norm_feats.append(feat + '_norm')

# show nans per feature and set
nancounts = {}
for feat in features:
    nancounts[feat] = df[feat].isna().sum()
nan_df = pd.DataFrame.from_dict(nancounts, orient='index', columns=['nans'])
nan_df.head(20)

# Direct test: do features correlate with article length (tokens) / sentlen?
corr_df = pd.DataFrame(columns=['feature', 'token_corr', 'sentlen_corr'])
for feature in features:
    slice = df[[feature, 'n_tokens_for_diversity']].dropna()
    token_corr, _ = spearmanr(slice[feature], slice['n_tokens_for_diversity'])
    slice = df[[feature, 'avg_sentlen']].dropna()
    sentlen_corr, _ = spearmanr(slice[feature], slice['avg_sentlen'])
    corr_df = pd.concat([corr_df, pd.DataFrame({'feature': [feature], 'token_corr': [token_corr], 'sentlen_corr': [sentlen_corr]})], ignore_index=True)
corr_df['token_corr'] = corr_df['token_corr'].round(2)
corr_df['sentlen_corr'] = corr_df['sentlen_corr'].round(2)
corr_df['feature'] = corr_df['feature'].str.replace('_', ' ').str.title()
corr_df.head(40)

# Residualization
import statsmodels.api as sm

def residualize(y, length_var):
    mask = y.notna() & length_var.notna()
    X = sm.add_constant(np.log1p(length_var[mask]))
    model = sm.OLS(y[mask], X).fit()
    resid = pd.Series(index=y.index, dtype=float)
    resid[mask] = model.resid
    return resid, model.rsquared

for col in ['cttr', 'noun_ttr', 'verb_ttr']:
    df[col + '_resid'], r2 = residualize(df[col], df['n_tokens_for_diversity'])
    print(f"{col}: length explains {r2:.1%} of variance")

features = [f.replace('cttr','cttr_resid').replace('noun_ttr','noun_ttr_resid').replace('verb_ttr','verb_ttr_resid') for f in features]
# %%

# try clustering features by correlation
corr = df[features].corr(method='spearman')
sns.clustermap(corr, cmap='vlag', center=0, figsize=(10, 10), 
                dendrogram_ratio=0.15, annot=False)
plt.show()

# %%


X = StandardScaler().fit_transform(df[features].fillna(df[features].median()))
pca = PCA() 
pca.fit(X)
print(pca.explained_variance_ratio_.cumsum())  # how many components needed for e.g. 80%?
loadings = pd.DataFrame(pca.components_.T, index=features, columns=[f"PC{i+1}" for i in range(len(features))])
print(loadings.iloc[:, :3])


# %%
#### Factor analysis ####

X = df[features].dropna() 
print(f"Factor analysis on {len(X)} rows with {len(features)} features.")

# Check factorability first
chi_sq, p = calculate_bartlett_sphericity(X)
kmo_all, kmo_model = calculate_kmo(X)
print(f"Bartlett's p={p:.4f}, KMO={kmo_model:.3f}")  # KMO > 0.6 is usually considered adequate

fa = FactorAnalyzer(n_factors=3, rotation='promax')
fa.fit(X)
loadings = pd.DataFrame(fa.loadings_, index=features, columns=[f"F{i+1}" for i in range(3)])
print(loadings.round(2))


variance_df = pd.DataFrame(
    fa.get_factor_variance(),
    index=["SS Loadings", "Proportion Var", "Cumulative Var"],
    columns=[f"F{i+1}" for i in range(3)]
)
print(variance_df)

# %%

uniquenesses = pd.Series(fa.get_uniquenesses(), index=features)
print(uniquenesses.sort_values().head(10))

df[['personal_pronoun_ratio', 'adjective_adverb_ratio', 'that_ratio']].corr()

# %%
def parallel_analysis(X, n_iter=50):
    n_obs, n_vars = X.shape
    real_ev, _ = fa.get_eigenvalues()
    random_evs = np.zeros((n_iter, n_vars))
    for i in range(n_iter):
        random_data = np.random.normal(size=(n_obs, n_vars))
        fa_random = FactorAnalyzer(n_factors=n_vars, rotation=None)
        fa_random.fit(random_data)
        random_evs[i], _ = fa_random.get_eigenvalues()
    return real_ev, random_evs.mean(axis=0)

real_ev, random_ev = parallel_analysis(X)

plt.figure(figsize=(7,5))
plt.plot(range(1, len(real_ev)+1), real_ev, marker='o', label='Actual data')
plt.plot(range(1, len(random_ev)+1), random_ev, marker='o', linestyle='--', label='Random data (mean)')
plt.axhline(1, color='grey', linestyle=':', alpha=0.5)
plt.xlabel("Factor number")
plt.ylabel("Eigenvalue")
plt.legend()
plt.title("Parallel Analysis")
plt.tight_layout()
plt.show()

# %%

dominant_factor = loadings.abs().idxmax(axis=1)
sort_order = loadings.loc[dominant_factor.sort_values().index]

plt.figure(figsize=(5, 8))
display = sort_order.copy()
display[display.abs() < 0.3] = 0
sns.heatmap(display, cmap='vlag', center=0, annot=True, fmt='.2f', cbar=False)
plt.title("Rotated Loadings (grouped by dominant factor)")
plt.tight_layout()
plt.show()

# %%
communalities = 1 - pd.Series(fa.get_uniquenesses(), index=features)
communalities.sort_values().plot(kind='barh', figsize=(6,8))
plt.xlabel("Communality (variance explained by 3 factors)")
plt.tight_layout()
plt.show()

# %%
print(df[['of_ratio', 'nominal_verb_ratio']].describe())
print("of_ratio == 0:", (df['of_ratio'] == 0).mean())
print("nominal_verb_ratio == 0:", (df['nominal_verb_ratio'] == 0).mean())

# Do these two even correlate with EACH OTHER? If they're jointly measuring
# "nominality," they should — if they don't, that undermines treating them
# as one shared construct.
print(df[['of_ratio', 'nominal_verb_ratio']].corr())




# %%
# histogram over time colored by category

# Defining palette for categories
palette = sns.color_palette("hsv_r", n_colors=5)
# Convert to a list of RGB tuples
palette_list = list(palette)
# change second to light green
palette_list[2] = (0.5, 0.9, 0.7)  # light green
# change 4th to light purple
palette_list[3] = (0.7, 0.5, 0.9)  # light purple

df['dt'] = pd.to_datetime(df['date'], errors='coerce')
df['date_ordinal'] = df['dt'].apply(lambda x: x.toordinal() if pd.notnull(x) else None)
df['year'] = df['dt'].dt.year
df['category'] = [x.split(' ')[0] for x in df['predicted_category']] # take first word as category

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True, gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.2}, dpi=500)
sns.set_style("whitegrid")

# ---- TOP: total volume per year ----
sns.histplot(data=df, x='year', discrete=True, color="grey", ax=ax1, kde=True, alpha=0.6)
ax1.set_ylabel("Total Articles")
# change ytick label number to 5k if 5,000
labels1 = [item.get_text() for item in ax1.get_yticklabels()]
labels1[1] = ['5k' if labels1[1] == '5000' else labels1[1]][0]
ax1.set_yticklabels(labels1)

# ---- BOTTOM: proportional stacked histogram ----
sns.histplot(data=df, x='year', hue='category', multiple='fill', discrete=True, legend=True, ax=ax2,palette=palette_list, alpha=0.8)
ax2.set_ylabel("Proportion")
ax2.set_xlabel("Year")
sns.move_legend(ax2,"upper right",  bbox_to_anchor=(.92, .6), frameon=False, title="Category",ncol=4)
plt.tight_layout()
plt.show()

df['category'].value_counts()

# %%

plt.figure(figsize=(12, 6))
for feat in norm_feats:
    year_std = df.groupby('year')[feat].std().reset_index()
    sns.lineplot(data=year_std, x='year', y=feat, label=feat.replace('_norm', ''))
plt.title('Standard Deviation of Normalized Features Over Time')
plt.xlabel('Year')
plt.ylabel('Standard Deviation (0-1 normalized)')
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# do correlation of feature std with year
corr_dict = {}
for feat in norm_feats:
    year_std = df.groupby('year')[feat].std().reset_index()
    stat, p_value = spearmanr(year_std['year'], year_std[feat])
    start_end_diff = year_std[feat].iloc[-1] - year_std[feat].iloc[0]
    range_feat = (round(year_std[feat].min(),2), round(year_std[feat].max(),2))
    corr_dict[feat] = (stat, p_value, start_end_diff, range_feat)

df_corr = pd.DataFrame.from_dict(corr_dict, orient='index', columns=['spearman_corr', 'p_value', 'start-end_diff', 'range'])
df_corr.head(20)

# %%
# same thing, but per genre instead of overall
for category in df['predicted_category'].unique():
    plt.figure(figsize=(12, 6))
    cat_data = df[df['predicted_category'] == category]
    for feat in norm_feats:
        year_std = cat_data.groupby('year')[feat].std().reset_index()
        sns.lineplot(data=year_std, x='year', y=feat, label=feat.replace('_norm', ''))
    plt.title(f'SD {category.upper()}')
    plt.xlabel('Year')
    plt.ylabel('Standard Deviation (0-1 normalized)')
    plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

corr_dict_cat = {}
for category in df['predicted_category'].unique():
    cat_data = df[df['predicted_category'] == category]
    for feat in norm_feats:
        year_std = cat_data.groupby('year')[feat].std().reset_index()
        stat, p_value = spearmanr(year_std['year'], year_std[feat])
        start_end_diff = year_std[feat].iloc[-1] - year_std[feat].iloc[0]
        range_feat = (round(year_std[feat].min(),2), round(year_std[feat].max(),2))
        corr_dict_cat[(category.upper(), feat)] = (stat, p_value, start_end_diff, range_feat)

df_corr_cat = pd.DataFrame.from_dict(corr_dict_cat, orient='index', columns=['spearman_corr', 'p_value', 'start-end_diff', 'range'])
df_corr_cat.head(60)

# %%

def scatter_per_group(data, x_col, y_col, group_col):
    plt.figure(figsize=(5, 5), dpi=500)
    sns.set_style("whitegrid")
    sns.scatterplot(data=data, x=x_col, y=y_col, color='grey', alpha=0.05)
    plt.xticks(rotation=45)
    # reduce ticks to ~10 evenly spaced points
    x_min, x_max = data[x_col].min(), data[x_col].max()
    ticks = np.linspace(x_min, x_max, 10)
    # convert ordinal → datetime properly
    tick_labels = [pd.Timestamp.fromordinal(int(t)).strftime('%Y') for t in ticks]
    plt.xticks(ticks=ticks, labels=tick_labels, rotation=45)

    # add lines for each group
    groups = data[group_col].unique()
    for group in groups:
        group_data = data[data[group_col] == group]
        sns.regplot(x=group_data[x_col], y=group_data[y_col], scatter=False, label=group, line_kws={'linewidth': 2, 'alpha': 0.7, 'color': palette_list[groups.tolist().index(group)]})
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title="Category", loc='upper right')
    plt.tight_layout()
    plt.show()

#### plot feats side by side #####
def plot_features_side_by_side(data, x_col, y_cols,          # list of feature names
    group_col, palette_list, figsize_per_plot=(4, 4), dpi=500):
    sns.set_style("whitegrid")
    
    n_plots = len(y_cols)
    fig, axes = plt.subplots(1, n_plots, figsize=(figsize_per_plot[0] * n_plots, figsize_per_plot[1]), dpi=dpi,sharex=True)
    
    # If only one feature is passed, axes is not iterable
    if n_plots == 1:
        axes = [axes]
    
    groups = data[group_col].unique().tolist()
    
    # Precompute x ticks
    x_min, x_max = data[x_col].min(), data[x_col].max()
    ticks = np.linspace(x_min, x_max, 10)
    tick_labels = [pd.Timestamp.fromordinal(int(t)).strftime('%Y') for t in ticks]
    
    for ax, y_col in zip(axes, y_cols):
        if y_col == 'avg_sentlen':
            # remove outliers for avg_sentlen (e.g. articles with avg_sentlen > 100)
            plot_data = data[data[y_col] < 200]
        else:
            plot_data = data
        
        # Scatter (background cloud)
        sns.scatterplot(data=plot_data, x=x_col,y=y_col,color='grey',alpha=0.05,ax=ax)
        
        # Group regression lines
        for i, group in enumerate(groups):
            group_data = plot_data[plot_data[group_col] == group]
            
            sns.regplot(x=group_data[x_col], y=group_data[y_col], scatter=False, ax=ax,
                label=group,line_kws={
                    'linewidth': 2,
                    'alpha': 0.7,
                    'color': palette_list[i]})
        
        ax.set_xlabel(x_col.split('_')[0].capitalize())  # e.g. "date_ordinal" → "Date"
        ax.set_ylabel(y_col.replace('_', ' ').capitalize())  # e.g. "avg_wordlen" → "Avg Wordlen"
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45)
        # set title to spearman correlation
        stat, p_value = spearmanr(plot_data[x_col], plot_data[y_col])
        ax.set_title(fr"$\bar{{\rho}}$: {stat:.2f}")
    
    # Single legend (from first axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Category", loc='center right')
    plt.tight_layout()
    plt.show()

# %%
# check features

def corr_with_date(data, measure):
    stats_df = data[[measure, 'date_ordinal', 'date']].dropna().copy()
    stat, p_value = spearmanr(stats_df['date_ordinal'], stats_df[measure])
    print(f"Rho btw {measure} vs date: {stat:.4f} (p-value: {p_value:.4e})")

    for category in data['predicted_category'].unique():
        cat_data = data[data['predicted_category'] == category]
        stats_df_cat = cat_data[[measure, 'date_ordinal', 'date']].dropna().copy()
        stat_cat, p_value_cat = spearmanr(stats_df_cat['date_ordinal'], stats_df_cat[measure])
        print(f"  Rho btw {measure} vs date for category {category}: {stat_cat:.4f} (p-value: {p_value_cat:.4e})")
    print("\n")
    scatter_per_group(df, 'date_ordinal', measure, 'predicted_category')
    return measure

# for feat in ['avg_wordlen', 'avg_sentlen']:
#     corr_with_date(df, feat)

plot_features_side_by_side(df, 'date_ordinal', ['avg_wordlen', 'avg_sentlen'], 'predicted_category', palette_list)

# %%
# print top 10 correlation of all features with date for all categories
categories = df['predicted_category'].unique()
for category in categories:
    dat = df.loc[df['predicted_category'] == category]
    print(f"\n Category: {category}, n={len(dat)} ")
    results = []
    for col in features:
        slice = dat[[col, 'date_ordinal', 'date']].dropna()
        stat, p_value = spearmanr(slice['date_ordinal'], slice[col])
        results.append((col, stat, p_value))

    results_df = pd.DataFrame(results, columns=['feature', 'spearman_corr', 'p_value'])
    results_df['abs_spearman_corr'] = results_df['spearman_corr'].abs()
    results_df = results_df.sort_values(by='abs_spearman_corr', ascending=False)
    print(results_df.head(10).reset_index(drop=True))

# %%

# see top german_prob articles over time
top_german = dat.sort_values(by='german_prob', ascending=False).head(10)
for i, r in top_german.head(10).iterrows():
    print(f"Date: {r['date']}, German Prob: {r['german_prob']:.4f}, Category: {r['predicted_category']}")
    print(f"Text: {r['text'][:1000]}...\n")


# %%
# lets see how a linear model would perform on predicting date_ordinal from features
from sklearn.model_selection import cross_val_score
selected_features = features

linreg_df = df.copy()
# standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
linreg_df[selected_features] = scaler.fit_transform(linreg_df[selected_features].fillna(0))


# %%

print(f"\nAll categories together, n={len(df)}")

X = linreg_df[selected_features].fillna(0)
y = linreg_df['date_ordinal'].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Core metrics ---
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f} days (~{rmse/365:.2f} years)")
print(f"MAE: {mae:.2f} days (~{mae/365:.2f} years)")
print(f"R2: {r2:.4f}")

# --- Baseline: predict mean date ---
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(baseline_mse)

print(f"\nBaseline RMSE: {baseline_rmse:.2f} days (~{baseline_rmse/365:.2f} years)")
print(f"Improvement over baseline (RMSE reduction): {baseline_rmse - rmse:.2f} days")

# --- Cross-validated R2 ---
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"\nCV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Standardized coefficients ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_std = LinearRegression()
model_std.fit(X_scaled, y)

coefs = pd.Series(model_std.coef_, index=selected_features)
coefs = coefs.sort_values(key=np.abs, ascending=False)

print("\nTop 10 standardized coefficients:")
print(coefs.head(10))

# visualize pred/actual
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.2, color='grey')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel('Actual Date (ordinal)')
plt.ylabel('Predicted Date (ordinal)')
plt.title('Predicted vs Actual Date (All Categories)')
plt.tight_layout()
plt.show()


# %%

for cat in linreg_df['predicted_category'].unique():

    lin_df = linreg_df.loc[linreg_df['predicted_category'] == cat].copy()
    n = len(lin_df)
    print(f"\nCategory: {cat}, n={n}")

    X = lin_df[selected_features].fillna(0)
    y = lin_df['date_ordinal']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- metrics ---
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.2f} days (~{rmse/365:.2f} years)")
    print(f"MAE: {mae:.2f} days (~{mae/365:.2f} years)")

    # --- baseline (predict mean year) ---
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    print(f"Baseline RMSE: {baseline_rmse/365:.2f} years")
    print(f"RMSE improvement: {(baseline_rmse - rmse)/365:.2f} years")

    # --- predicted vs actual ---
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.2)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )
    plt.xlabel('Actual Date (ordinal)')
    plt.ylabel('Predicted Date (ordinal)')
    plt.title(f'Predicted vs Actual Date for {cat}')
    plt.tight_layout()
    #plt.show()

    # # print top 5 features by absolute standardized coefficient
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # model_std = LinearRegression()
    # model_std.fit(X_scaled, y)
    # coefs = pd.Series(model_std.coef_, index=selected_features)
    # coefs = coefs.sort_values(key=np.abs, ascending=False)
    # print("\nTop 5 standardized coefficients:")
    # print(coefs.head(5))

    # # plot residuals
    # residuals = y_test - y_pred
    # plt.figure(figsize=(4, 4))
    # sns.scatterplot(x=y_pred, y=residuals, alpha=0.2)
    # plt.axhline(0, color='r', linestyle='--')
    # plt.xlabel('Predicted Date (ordinal)')
    # plt.ylabel('Residuals (Actual - Predicted)')
    # plt.tight_layout()
    # plt.show()

# %%
# plot newspapers and categories over time
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='date', hue='newspaper', multiple='fill', discrete=True, palette='tab10', alpha=0.8)
plt.title('Article Count by Newspaper Over Time')
plt.xlabel('Year')
plt.ylabel('Article Count')
plt.legend(title='Newspaper', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %%
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(df[features].fillna(0))
print("features: ", features)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=df['predicted_category'], alpha=0.5, palette=palette_list)
plt.title('UMAP Projection of Articles Colored by Predicted Category')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Predicted Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%

# plot internal heterogeneity of categories over time by looking at the standard deviation of features per year and category
heterogeneity = df.groupby(['year', 'predicted_category'])[features].std().reset_index()
# plot heterogeneity over time for top 3 categories
top_categories = df['predicted_category'].value_counts().head(3).index.tolist()
plt.figure(figsize=(12, 6))
for category in top_categories:
    cat_data = heterogeneity[heterogeneity['predicted_category'] == category]
    plt.plot(cat_data['year'], cat_data[features].mean(axis=1), label=category)
plt.title('Average Standard Deviation of Features Over Time by Category')
plt.xlabel('Year')
plt.ylabel('Average Std Dev of Features')
plt.legend()
plt.tight_layout()
plt.show()


# %%

# check any corr between wordcount and features
measure = "wordcount"

stats_df = df[[measure] + [col for col in df.columns if col not in ['id', 'text', 'predicted_category', 'article_id', 'date', 'year', measure]]].dropna().copy()
results = []
for col in stats_df.columns:
    if col == measure:
        continue

    slice = stats_df[[col, measure]].dropna()
    print(f"Processing column: {col}, n={len(stats_df)}")

    stat, p_value = spearmanr(slice[measure], slice[col])
    results.append((col, stat, p_value))
# create df
results_df = pd.DataFrame(results, columns=['feature', 'spearman_corr', 'p_value'])
# sort by absolute value of spearman_corr
results_df['abs_spearman_corr'] = results_df['spearman_corr'].abs()
results_df = results_df.sort_values(by='abs_spearman_corr', ascending=False)
results_df['actual_corr'] = results_df['spearman_corr']
results_df.head(20)
# %%
# visualize top 5 correlations with wordcount
top5 = results_df.head(10)['feature'].tolist()
for feature in top5:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=stats_df, x=measure, y=feature, marker='o', alpha=0.6)
    plt.title(f'{feature} vs {measure}')
    plt.xlabel(measure)
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()
# %%
