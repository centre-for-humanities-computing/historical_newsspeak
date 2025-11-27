# %%
!pip install --only-binary :all: blis
!pip install -r requirements.txt
# %%
import pandas as pd
import numpy as np
from datasets import load_dataset
import logging
from tqdm import tqdm
import time
import json

from src.feature_utils import (
    process_text, compressrat, get_pos_derived_features,
    avg_sentlen, avg_wordlen, calculate_dependency_distances
    # , project_sentiment
)

# %%
from joblib import Parallel, delayed
import dacy

# --- Worker model setup ---
worker_model = None

def init_worker():
    """Load dacy transformer model once per worker."""
    global worker_model
    worker_model = dacy.load("da_dacy_large_trf-0.2.0")

def process_row(row, dacy_model):
    try:
        text_id = row.article_id
        text = row.text
        features = {}

        process_text(text, text_id, dacy_model=dacy_model)
        features.update(get_pos_derived_features(text_id))
        features["avg_wordlen"] = avg_wordlen(text_id)
        features["avg_sentlen"], features["num_sents"] = avg_sentlen(text_id)
        features.update(calculate_dependency_distances(text_id))
        features["compression_ratio"] = compressrat(text_id)

        # sentiment (optional, comment out if SemanticProjector not installed)
        # features["sentiment"] = project_sentiment(text_id)
        # if isinstance(features.get("sentiment"), (list, np.ndarray)):
        #     features["sentiment_mean"] = np.mean(features["sentiment"])
        #     features["sentiment_std"] = np.std(features["sentiment"])

        features["article_id"] = text_id
        features["label"] = row.label
        return features
    except Exception as e:
        print(f"Error processing article_id {row.article_id}: {e}")
        return None

def process_row_wrapper(row):
    global worker_model
    return process_row(row, dacy_model=worker_model)

# %%
# CONFIGURE
ts = time.strftime("%Y%m%d-%H%M%S")
logging.basicConfig(
    filename='logs/get_feats_report.txt',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    force=True
)

# load dataset
dataset = load_dataset("chcaa/eno-embs-old-news", split="train", columns=["id", "text", "predicted_category"])
df = dataset["train"].to_pandas()

# %%
# apply replacements
with open("data/replacements.json") as f:
    replacements = json.load(f)

def apply_replacements(text, replacements):
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

df["text"] = df["text"].astype(str).apply(lambda x: apply_replacements(x, replacements))

# %%
# Filter short or empty texts
df = df[df['text'].str.strip().str.len() > 0]
df = df[df['text'].str.split().str.len() >= 5]

# %%
# Chunked parallel processing
OUTPUT_FILE = f"data/{ts}_stylistics_all.jsonl"
CHUNK_SIZE = 50000  # adjust as needed

for start in tqdm(range(0, len(df), CHUNK_SIZE), desc="Processing chunks"):
    end = start + CHUNK_SIZE
    chunk = df.iloc[start:end]

    results = Parallel(
        n_jobs=-1,
        backend="loky",
        initializer=init_worker)
        (delayed(process_row_wrapper)(row) for row in tqdm(
            chunk.itertuples(index=False),
            total=len(chunk),
            desc="Processing texts"))

    results = [r for r in results if r is not None]

    with open(OUTPUT_FILE, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

# %%
# Combine results into a DataFrame
records = []
with open(OUTPUT_FILE) as f:
    for line in f:
        records.append(json.loads(line))

stylistics_df = pd.DataFrame(records)
stylistics_df.to_csv(f"data/{ts}_stylistics.csv", sep="\t", index=False, encoding="utf-8")

# %%

# %%
