# %%
import json
import pandas as pd
import glob
from datasets import load_dataset

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# %%
# get all files ending in stylstics_all.jsonl in data folder
file_paths = glob.glob("../data/*_stylistics_all.jsonl")
print(f"Found {len(file_paths)} files.")

# load and merge all files
data = []
for file_path in file_paths:
    with open(file_path) as f:
        lines = f.readlines()
        data.extend([json.loads(line) for line in lines])

# list of dicts to df
data = pd.DataFrame(data)
# drop duplicates based on article_id
data = data.drop_duplicates(subset=['article_id'])
# print number of records
print(f"Total records loaded: {len(data)}")
# see amount of nans in each column
print(data.isna().sum())
# and zeroes
print("--------------")
print((data == 0).sum())
print(data.columns)
data.head()
# %%

# load org data
dataset = load_dataset(
    "chcaa/eno-embs-old-news",
    split="train",
    columns=["id", "text", "predicted_category", "date", "newspaper"]
)
df = dataset.to_pandas()
print(f"Total rows: {len(df)}")


# %%
# merge on id
merged = pd.merge(df, data, left_on="id", right_on="article_id", how="inner")
print(f"Merged rows: {len(merged)}")


# remove paratext
merged = merged[merged['predicted_category'] != 'Paratext']
# fix date to ordinal
merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
merged['date_ordinal'] = merged['date'].apply(lambda x: x.toordinal() if pd.notnull(x) else None)

# add ficiton tags
fiction_tags = pd.read_csv("../data/20251015_all_predictions_w_id.csv")
merged = pd.merge(merged, fiction_tags[['id', 'predicted_label']], left_on='id', right_on='id', how='left')
print(f"Merged with fiction tags, total rows: {len(merged)}")
# replace "predicted_category" with fiction if predicted_label == "fiction" or "non-fiction", else leave as is
merged['predicted_category'] = merged.apply(lambda row: row['predicted_label'] if row['predicted_label'] in ['fiction'] else row['predicted_category'], axis=1)
# see amount of national news vs other
print(merged['predicted_category'].value_counts())

merged.head()


# %%
def scatter_per_group(data, x_col, y_col, group_col):
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=group_col, alpha=0.2)
    plt.xticks(rotation=45)
    plt.title(f'{y_col} Over Time by {group_col}')
    # add lines for each group
    groups = data[group_col].unique()
    for group in groups:
        group_data = data[data[group_col] == group]
        sns.regplot(x=group_data[x_col], y=group_data[y_col], scatter=False, label=f'{group} Trend')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()

# %%
# check 1 feature
measure = "avg_wordlen"

for category in merged['predicted_category'].unique():
    stats_df = merged[merged['predicted_category'] == category][[measure, 'date_ordinal', 'date']].dropna().copy()
    stat, p_value = spearmanr(stats_df['date_ordinal'], stats_df[measure])
    print(f"Rho btw {measure} vs date for category {category}: {stat:.4f} (p-value: {p_value:.4e})")

scatter_per_group(merged, 'date_ordinal', measure, 'predicted_category')

stats_df = merged[[measure, 'date_ordinal', 'date']].dropna().copy()
stat, p_value = spearmanr(stats_df['date_ordinal'], stats_df[measure])
print(f"Rho btw {measure} vs date (for all): {stat:.4f} (p-value: {p_value:.4e})")

# see german_prob over time
plt.figure(figsize=(12, 6))
sns.scatterplot(data=stats_df, x='date', y=measure, marker='o', alpha=0.1)
plt.xticks(rotation=45)
plt.title(f'{measure} Over Time')
plt.xlabel('Date')
plt.ylabel('German Probability')
plt.tight_layout()
plt.show()

# %%
# check all features
dat = merged.loc[merged['predicted_category'] == 'International news']

# for all cols, do the spearman correlation with date
results = []
for col in dat.columns:
    if col in ['id', 'text', 'predicted_category', 'article_id', 'date', 'year', "newspaper", "date_ordinal"]:
        continue

    slice = dat[[col, 'date_ordinal', 'date']].dropna()
    print(f"Processing column: {col}, n={len(dat)}")

    stat, p_value = spearmanr(slice['date_ordinal'], slice[col])
    results.append((col, stat, p_value))

# create df
results_df = pd.DataFrame(results, columns=['feature', 'spearman_corr', 'p_value'])
# sort by absolute value of spearman_corr
results_df['abs_spearman_corr'] = results_df['spearman_corr'].abs()
results_df = results_df.sort_values(by='abs_spearman_corr', ascending=False)

# show me the top corr in scatterplots (4)
top5 = results_df.head(5)['feature'].tolist()
for feature in top5:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dat, x='date_ordinal', y=feature, marker='o', alpha=0.1)
    plt.title(f'{feature} Over Time for Fiction')
    plt.xlabel('Date (ordinal)')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

results_df.head(30)

# %%

# see top german_prob articles over time
top_german = dat.sort_values(by='german_prob', ascending=False).head(1000)
top_german.sort_values(by='date').tail(10)[['id', 'date', 'german_prob', 'text']]
# %%
# lets see how a linear model would perform on predicting date_ordinal from features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# prepare data
feature_cols = [col for col in dat.columns if col not in ['id', 'text', 'predicted_category', 'article_id', 'date', 'year', "newspaper", "date_ordinal", 'predicted_label']]
X = dat[feature_cols].fillna(0)
y = dat['date_ordinal'].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train model
model = LinearRegression()
model.fit(X_train, y_train)
# predict
y_pred = model.predict(X_test)
# evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression MSE: {mse:.2f}, R2: {r2:.4f}")

# scatter plot of predicted vs actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Date Ordinal')
plt.ylabel('Predicted Date Ordinal')
plt.title('Predicted vs Actual Date Ordinal for Fiction Articles')
plt.tight_layout()



# %%
# see year range
merged['year'] = merged['date'].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) and '-' in x else None)
print(f"Year range: {merged['year'].min()} - {merged['year'].max()}")

# see newspaper distribution
print(merged['newspaper'].value_counts())

# %%
# check any corr between wordcount and features
measure = "wordcount"

stats_df = merged[[measure] + [col for col in merged.columns if col not in ['id', 'text', 'predicted_category', 'article_id', 'date', 'year', measure]]].dropna().copy()
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
