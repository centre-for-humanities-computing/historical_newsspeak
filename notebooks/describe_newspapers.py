# %%

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr, pearsonr

CWD = Path.cwd().parent
DATA_PATH = CWD / "data"
FIGS_PATH = CWD / "figs"
RESOURCES_PATH = CWD / "resources"

data = pd.read_parquet(DATA_PATH / "eno.parquet")

# %%

# metadata
meta_newspapers = pd.read_json(RESOURCES_PATH / "newspapers.json")

# %%

# let's get min max dates for each newspaper
for newspaper, group in data.groupby("newspaper"):
    min_date = group.date.min()
    max_date = group.date.max()
    print(f"{newspaper}: {min_date} to {max_date}")
    n_articles = len(group)
    print(f"n articles: {n_articles}")
    print(f"n per year: {n_articles / (max_date.year - min_date.year + 1):.2f}")
    n_tokens = group.n_tokens.sum()
    print(f"n tokens: {n_tokens}")
    mean_pwa = group.pwa.mean()
    print(f"mean pwa: {mean_pwa:.3f}")
    print(f"SD pwa: {group.pwa.std():.3f}")
    print(f"mean fictionality: {group.fictionality_tag.mean():.3f}")



# 

