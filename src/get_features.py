
# %%
import pandas as pd
import numpy as np

from datasets import load_dataset, load_from_disk
import logging
from tqdm import tqdm

from src.feature_utils import process_text, compressrat, get_pos_derived_features, avg_sentlen, avg_wordlen
from src.feature_utils import calculate_dependency_distances, project_sentiment
from joblib import Parallel, delayed
import time
import json

# %%

# CONFIGURE
ts = time.strftime("%Y%m%d-%H%M%S")

# Configure logging
logging.basicConfig(
    filename='logs/get_feats_report.txt',           # Output file
    filemode='w',                    # 'w' to overwrite each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO,              # Minimum level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    force=True,                # Force logging even if already configured
)

# get data
# load it from HF
dataset = load_dataset("chcaa/")
# get the train split
df = dataset["train"].to_pandas()
df.head()

# %%
# also, we want to do the replacements (replacements.json) for all texts before processing
with open("data/replacements.json") as f:
    replacements = json.load(f)

def apply_replacements(text, replacements):
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# apply replacements to all texts
df["text"] = df["text"].astype(str).apply(lambda x: apply_replacements(x, replacements))
df.head()

# %%
# Get stylistics

def process_row(row):
    try:
        text_id = row.article_id
        text = row.text
        features = {}

        process_text(text, text_id)
        features.update(get_pos_derived_features(text_id))
        features["avg_wordlen"] = avg_wordlen(text_id)
        features["avg_sentlen"], features["num_sents"] = avg_sentlen(text_id)
        features.update(calculate_dependency_distances(text_id))
        features["compression_ratio"] = compressrat(text_id)
        features["sentiment"] = project_sentiment(text_id)

        if isinstance(features["sentiment"], (list, np.ndarray)):
            features["sentiment_mean"] = np.mean(features["sentiment"])
            features["sentiment_std"] = np.std(features["sentiment"])

        features["article_id"] = text_id
        features["label"] = row.label
        return features
    except Exception as e:
        # log the error and return None
        print(f"Error processing article_id {row.article_id}: {e}")
        return None

# filtering
df = df[df['text'].astype(str).str.strip().str.len() > 0]
df = df[df['text'].str.split().str.len() >= 5]

# apply
import json

OUTPUT_FILE = f"data/{ts}_stylistics_all.jsonl"
CHUNK_SIZE = 50000  # adjust as needed

for start in tqdm(range(0, len(df), CHUNK_SIZE), desc="Processing chunks"):
    end = start + CHUNK_SIZE
    chunk = df.iloc[start:end]

    # Process rows in parallel with a progress bar per chunk
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_row)(row) for row in tqdm(chunk.itertuples(index=False), 
                                                total=len(chunk), desc="Processing texts")
    )
    results = [r for r in results if r is not None]

    # Append results to JSON Lines
    with open(OUTPUT_FILE, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

# Later, read JSONL back into a DataFrame
records = []
with open(OUTPUT_FILE) as f:
    for line in f:
        records.append(json.loads(line))

stylistics_df = pd.DataFrame(records)
stylistics_df.to_csv(f"data/{ts}_stylistics.csv", sep="\t", index=False, encoding="utf-8")

# %%

# # make progress bar
# tqdm.pandas(desc="Processing texts")

# stylistics_features = []

# for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):

#     # we use IDs to read spacy-parsed files
#     text_id = row['article_id']
#     # we use the text directly for some features
#     text = row["text"]

#     process_text(text, text_id)

#     features = {}

#     # POS and morph features
#     features.update(get_pos_derived_features(text_id))

#     # Average word length
#     features["avg_wordlen"] = avg_wordlen(text_id)
#     # Average sentence length and number of sentences
#     features["avg_sentlen"], features["num_sents"] = avg_sentlen(text_id)

#     # Dependency distance metrics
#     features.update(calculate_dependency_distances(text_id))

#     # Compression ratio
#     features["compression_ratio"] = compressrat(text_id)

#     # Sentiment analysis with our projection tool
#     features["sentiment"] = project_sentiment(text_id)
#     # add mean and sd of sentiment if it's a list
#     if isinstance(features["sentiment"], (list, np.ndarray)):
#         features["sentiment_mean"] = np.mean(features["sentiment"])
#         features["sentiment_std"] = np.std(features["sentiment"])

#     # Add article IDs
#     features["article_id"] = row["article_id"]
#     # Add the label
#     features["label"] = row["label"]

#     # Add the features to the list
#     stylistics_features.append(features)

# # just print colnames to check
# print("stylistics_features columns:")
# print(stylistics_features[0].keys())
# # Create a DataFrame from the list of dictionaries
# stylistics_df = pd.DataFrame(stylistics_features)
# # Save the DataFrame to a CSV file
# stylistics_df.to_csv("data/stylistics.csv", sep="\t", index=False)

# # log it
# logging.info(f"Created stylistic features. Colnames: {stylistics_features[0].keys()}")
# logging.info(f"stylistics_df shape: {stylistics_df.shape}")
# for col in stylistics_features[0].keys():
#     # log the number of missing values
#     logging.info(f"stylistics_df {col} missing values: {stylistics_df[col].isnull().sum()}")
#     # print the distribution of the column
#     logging.info(f"stylistics_df {col} distribution: {stylistics_df[col].describe()}")
#     logging.info("\n")


# %%
