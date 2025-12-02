

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

# %%
# CONFIGURE
ts = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_JSONL = f"data/{ts}_stylistics_all.jsonl"
OUTPUT_CSV = f"data/{ts}_stylistics.csv"

logging.basicConfig(
    filename='logs/get_feats_report.txt',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    force=True
)

# %%
# Load dataset
dataset = load_dataset(
    "chcaa/eno-embs-old-news",
    split="train",
    columns=["id", "text", "predicted_category"]
)
df = dataset.to_pandas()
print(f"Total rows: {len(df)}")


# %%
# Apply replacements
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
df = df[df['text'].str.split(" ").str.len() >= 10] # filtering below 10 words
print(f"Rows after filtering: {len(df)}")

# %%
# save to file for record
df.to_csv(f"data/{ts}_filtered_texts.csv", index=False, encoding="utf-8")
# %%

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import importlib
import src.feature_utils as feature_utils

# Reload the module
importlib.reload(feature_utils)

# Re-import the functions from the reloaded module
process_text = feature_utils.process_text
compressrat = feature_utils.compressrat
get_pos_derived_features = feature_utils.get_pos_derived_features
compute_basic_lengths = feature_utils.compute_basic_lengths
calculate_dependency_distances = feature_utils.calculate_dependency_distances
german_probability = feature_utils.german_probability

# %%
# Sequential processing
records = []
with open(OUTPUT_JSONL, "w") as f_out:
    for row in tqdm(df[:20].itertuples(index=False), total=20, desc="Processing texts"):
        try:
            text_id = row.id
            print(text_id)
            text = row.text

            if not text:
                logging.warning(f"No features in text: {text_id}")
                continue  # skip empty texts

            features = {}

            process_text(text, text_id)
            features.update(get_pos_derived_features(text_id))
            features['avg_wordlen'], features['avg_sentlen'], features['num_sents'] = compute_basic_lengths(text_id)

            features.update(calculate_dependency_distances(text_id))
            features["compression_ratio"] = compressrat(text_id)
            features['german_prob'] = german_probability(text_id)

            features["article_id"] = text_id

            # Save each row immediately
            f_out.write(json.dumps(features) + "\n")

        except Exception as e:
            logging.error(f"Error processing article_id {row.id}: {e}")

# Combine into CSV
records = []
with open(OUTPUT_JSONL) as f_in:
    for line in f_in:
        records.append(json.loads(line))

stylistics_df = pd.DataFrame(records)
stylistics_df.to_csv(OUTPUT_CSV, sep="\t", index=False, encoding="utf-8")

print("Done. Outputs saved.")


