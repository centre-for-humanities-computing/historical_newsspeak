import pandas as pd
from datasets import load_dataset
import numpy as np

import statsmodels.api as sm

import numpy as np
from pathlib import Path
from datetime import date as py_date

# %%

CWD = Path.cwd().parent
DATA_PATH = CWD / "data"
FIGS_PATH = CWD / "figs"

# %%

# Load dataset
ds = load_dataset(
    "chcaa/eno-newspapers-enriched",
    split="train",
    columns=['id', 'date', 'newspaper', 'predicted_category', 'fiction_prob', 'non_fiction_prob', 'fictionality_tag']
)

df_meta = ds.to_pandas()
df_meta.head()

# %%

# Load features parquet
df_feats = pd.read_parquet(DATA_PATH / "features_combined_standardized_13-07-26.parquet")
df_feats.head()

# %%
# rename id to article_id in df_meta to match df_feats
df_meta = df_meta.rename(columns={"id": "article_id"})

# merge
df = df_feats.merge(df_meta, on="article_id", how="left")

# set a num_sents filter
print(f"Total rows before filtering: {len(df)}")
threshold = 2
df = df[df["num_sents"] >= threshold]
print(f"Filtered out articles with fewer than {threshold} sentences.")
print(f"Number of articles with False in 'semantic_sentiment_std_defined': "
      f"{df['semantic_sentiment_std_defined'].value_counts().get(False, 0)}")

# fix categories: add fiction based on fictionality_tag, drop Paratext AFTER
# the merge (not before, on df_meta alone) -- filtering df_meta pre-merge
# was what produced ~61,820 "unknown" rows earlier, since those article_ids
# then had no metadata row to match at all.
df['predicted_category'] = df.apply(
    lambda row: 'fiction' if row['fictionality_tag'] == 'fiction' else row['predicted_category'], axis=1
)
df = df[df['predicted_category'] != 'Paratext']

print(f"Total rows: {len(df)}")
print(df.columns)
print(df.describe())

# %%
nan_date = df['date'].isna().sum()
print(f"Number of rows with NaN in 'date': {nan_date} out of {len(df)} total rows.")
print(f"N rows with nan in 'article_id': {df['article_id'].isna().sum()} out of {len(df)} total rows.")

# fill nan in date with the date segment embedded in article_id
# (e.g. "danske_mercurius_1667-02-01_1000052")
df['dt'] = pd.to_datetime(df['date'], errors='coerce')

# Dates before ~1677-09-21 overflow pandas' ns-precision datetime64 and
# silently become NaT via errors='coerce' -- not a formatting issue, a real
# limitation of the dtype. Recover just these using Python's native date
# type, which supports the full proleptic Gregorian calendar.
still_missing_mask = df['dt'].isna() & df['date'].notna()
print(f"Rows failing pd.to_datetime (likely pre-1677 dates): {still_missing_mask.sum()}")


def safe_parse_ordinal_and_year(date_str):
    try:
        y, m, d = map(int, str(date_str).split('-')[:3])
        d_obj = py_date(y, m, d)
        return d_obj.toordinal(), d_obj.year
    except (ValueError, TypeError):
        return None, None


fallback = df.loc[still_missing_mask, 'date'].apply(safe_parse_ordinal_and_year)
df.loc[still_missing_mask, 'date_ordinal'] = fallback.apply(lambda t: t[0])
df.loc[still_missing_mask, 'year'] = fallback.apply(lambda t: t[1])

# Vectorized path for everything else -- this is the ONLY place
# date_ordinal/year get set for the bulk of rows. (A duplicate,
# non-masked re-assignment used to follow this and silently overwrite
# the pre-1677 fallback values with NaT again -- removed.)
bulk_mask = ~still_missing_mask
df.loc[bulk_mask, 'date_ordinal'] = df.loc[bulk_mask, 'dt'].apply(lambda x: x.toordinal() if pd.notnull(x) else None)
df.loc[bulk_mask, 'year'] = df.loc[bulk_mask, 'dt'].dt.year

# make sure article id is str
df['article_id'] = df['article_id'].astype(str)
df = df.dropna(subset='article_id')

n_nan_category = df['predicted_category'].isna().sum()
print(f"Rows with NaN predicted_category (no metadata match): {n_nan_category} out of {len(df)}")

# take first word as category -- vectorized, NaN-safe. (A duplicate,
# non-vectorized, non-NaN-safe list-comprehension version used to follow
# this and silently redo the work with the fragile old logic -- removed.)
df['category'] = df['predicted_category'].fillna('unknown').str.split(' ').str[0]
print("VC for category:", df['category'].value_counts())
df.head()

# %%

nominal = pd.read_parquet(DATA_PATH / "nominal_verb_ratio_fixed.parquet")
print(f"Total rows in nominal_verb_ratio dataset: {len(nominal)}")
nominal.head()

df = df.merge(nominal, on="article_id", how="left")

# check the NEWLY MERGED fixed column -- checking the old 'nominal_verb_ratio'
# here would validate the stale, buggy column instead (rename happens below)
nan_nominal_fixed = df['nominal_verb_ratio_fixed'].isna().sum()
print(f"Number of rows with NaN in 'nominal_verb_ratio_fixed': {nan_nominal_fixed} out of {len(df)} total rows.")

df = df.drop(columns=['nominal_verb_ratio'], errors='ignore')
df = df.rename(columns={"nominal_verb_ratio_fixed": "nominal_verb_ratio"})
df.head()

# %%

### deal with features ###

# Residualization -- fit on this filtered (num_sents>=2) sample, consistent
# with how residualization has been done throughout this analysis.

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

# %%

# save this data to parquet for future use
df.to_parquet(DATA_PATH / "usage_features_13-07-26_with_nominal.parquet", index=False)
print(f"Saved {len(df)} rows to usage_features_13-07-26_with_nominal.parquet")

