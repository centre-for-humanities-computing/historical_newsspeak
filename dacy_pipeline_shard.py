# %%

import torch
import pandas as pd
import dacy
from pathlib import Path
import time
from datasets import load_dataset
import re
import json
from tqdm import tqdm


# %%

print(torch.cuda.is_available())

# %%

dacy_model = dacy.load(
    "da_dacy_large_trf-0.2.0",
    disable=[])

dacy_model.to("cuda")


# %%

ts = time.strftime("%Y%m%d-%H%M")

DATA_PATH = Path("data")
SPACY_DIR = DATA_PATH / "spacy_books"

run_dir = SPACY_DIR / f"run_{ts}"
run_dir.mkdir(parents=True, exist_ok=True)

# Get replacements from JSON file
replacements_path = DATA_PATH / "replacements.json"
with open(replacements_path) as f:
    replacements = json.load(f)

# function: replacements
def apply_replacements(text, replacements):
    def replace_match(match):
        word = match.group(0)
        return replacements.get(word.lower(), word)

    return re.sub(
        r"[A-Za-zÆØØÅæøå]+",
        replace_match,
        text)

# function: clean_ocr_text
def clean_ocr_text(text, replacements):
    """
    Minimal OCR cleanup before linguistic annotation.
    Keeps historical spelling and punctuation.
    """

    if not isinstance(text, str):
        return ""

    # Normalize unicode spaces
    text = text.replace("\xa0", " ")

    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Common quote normalization
    text = text.replace("„", '"').replace("“", '"').replace("”", '"')

    # apply replacements
    text = apply_replacements(text, replacements)

    # remove =, replace with hyphen if between letters
    text = re.sub(r"(\w)\s*=\s*(\w)", r"\1-\2", text)

    # big letters in word
    text = re.sub(r"\b(?=\w*[A-Z]\w*[A-Z])\w+\b", lambda m: m.group(0).lower() if not m.group(0).isupper() else m.group(0), text)

    return text.strip()


SHARD_SIZE = 5000  # number of articles per parquet file

# function: save_shard
def save_shard(rows, shard_id):
    shard_df = pd.DataFrame(rows)
    shard_df.to_parquet(
        run_dir / f"shard_{shard_id:05d}.parquet",
        index=False
    )
    # Save processed article ids
    processed_ids = shard_df["article_id"].unique()

    with open(run_dir / "processed_ids.txt", "a") as f:
        for article_id in processed_ids:
            f.write(f"{article_id}\n")

# function: save_spacy_doc
def doc_to_rows(doc, text_id):

    rows = []

    for sent_id, sent in enumerate(doc.sents):
        for token in sent:
            rows.append({
                "token_i": token.i,
                "token_text": token.text,
                "lemma": token.lemma_,
                "is_punct": token.is_punct,
                "is_stop": token.is_stop,
                "is_space": token.is_space,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "morph": str(token.morph),

                "ent_type": token.ent_type_,
                "ent_iob": token.ent_iob_,

                "head_i": token.head.i,
                "sent_id": sent_id,
                "start_char": token.idx,
                "end_char": token.idx + len(token.text),

                "article_id": text_id,
            })

    return rows

# %%

# Load dataset
dataset = load_dataset(
    "chcaa/eno-newspapers-enriched",
    split="train",
    columns=["id", "text"])

df = dataset.to_pandas()
print(f"Total rows: {len(df)}")

# Filter short or empty texts
df = df[df['text'].str.strip().str.len() > 0]
df = df[df['text'].str.count(r'\S+') >= 10]  # counts sequences of non-whitespace
print(f"Rows after filtering: {len(df)}")

# shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# %%

texts = [clean_ocr_text(t, replacements=replacements) for t in df["text"].tolist()]
ids = df["id"].tolist()


buffer = []
shard_id = 0

for i, (text_id, doc) in enumerate(
    tqdm(
        zip(ids, dacy_model.pipe(texts, batch_size=16)),
        total=len(ids),
        desc="Parsing texts"
    )):

    buffer.extend(doc_to_rows(doc, text_id))

    # every N articles
    if (i + 1) % SHARD_SIZE == 0:
        save_shard(buffer, shard_id)

        buffer = []
        shard_id += 1


# save remaining documents
if buffer:
    save_shard(buffer, shard_id)