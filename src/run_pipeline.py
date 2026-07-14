# %%
import argparse
import json
import re
import time
import traceback
from pathlib import Path

import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="Process only first N articles (for pilot timing runs)")
parser.add_argument("--batch-size", type=int, default=64, help="nlp.pipe batch_size")
parser.add_argument("--run-dir", type=str, default=None, help="Resume into an existing run dir instead of creating a new one")
args = parser.parse_args()

# %%
print("CUDA available:", __import__("torch").cuda.is_available())

# %%
spacy.require_gpu()
nlp = spacy.load(
    "da_dacy_large_trf",
    disable=["coref", "span_resolver", "span_cleaner", "entity_linker"],
)

# %%
DATA_PATH = Path("data")
SPACY_DIR = DATA_PATH / "spacy_books"

if args.run_dir:
    run_dir = Path(args.run_dir)
    assert run_dir.exists(), f"--run-dir {run_dir} does not exist"
    print(f"Resuming into existing run dir: {run_dir}")
else:
    ts = time.strftime("%Y%m%d-%H%M")
    run_dir = SPACY_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting new run dir: {run_dir}")

# Get replacements from JSON file, detecting silently-duplicated keys
replacements_path = DATA_PATH / "replacements.json"
with open(replacements_path) as f:
    raw_json = f.read()

_dupe_keys = []


def _check_dupes(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            _dupe_keys.append(k)
        seen[k] = v
    return seen


replacements = json.loads(raw_json, object_pairs_hook=_check_dupes)
if _dupe_keys:
    print(
        f"WARNING: duplicate keys in replacements.json (only the LAST value "
        f"for each is kept): {_dupe_keys}"
    )

# Case-insensitive lookup: match text -> replacement value
_replacements_lower = {k.lower(): v for k, v in replacements.items()}


def _make_boundary_pattern(key):
    # Only anchor a \b where the key actually starts/ends on a word
    # character — abbreviations ending in "." or phrases with spaces
    # don't get a boundary there, since \b requires a word/non-word
    # transition that a trailing "." or space already provides naturally.
    prefix = r"\b" if key[0].isalpha() else ""
    suffix = r"\b" if key[-1].isalpha() else ""
    return prefix + re.escape(key) + suffix


# Sort longest-first so multi-word phrases match before their component words
_keys_by_length = sorted(replacements.keys(), key=len, reverse=True)
_replacement_pattern = re.compile(
    "|".join(_make_boundary_pattern(k) for k in _keys_by_length),
    flags=re.IGNORECASE,
)


# function: replacements
def apply_replacements(text, _pattern=_replacement_pattern, _lookup=_replacements_lower):
    def replace_match(match):
        matched = match.group(0)
        return _lookup.get(matched.lower(), matched)

    return _pattern.sub(replace_match, text)


# function: clean_ocr_text
def clean_ocr_text(text):
    """
    Minimal OCR cleanup before linguistic annotation.
    Keeps historical spelling and punctuation.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.replace("\u201e", '"').replace("\u201c", '"').replace("\u201d", '"')
    # Rejoin words split by a trailing hyphen (line-wrap hyphenation),
    # e.g. "arbej-\nde" or "arbej- de" -> "arbejde". Must run before
    # apply_replacements so the dictionary sees whole words, not fragments.
    #
    # EXCEPTION: Danish shared-compound elision, e.g. "Broelægger- og
    # Vognmandslauget" (short for "Broelæggerlauget og Vognmandslauget").
    # Here the hyphen is grammatically meaningful and "og"/"eller" is a
    # real following word, not a continuation of the same word — must NOT
    # be joined. First fix any OCR-dropped space before these conjunctions,
    # then exclude them from the join via negative lookahead.
    _elision_conjunctions = r"(?:og|eller)"
    text = re.sub(rf"-(?={_elision_conjunctions}\b)", "- ", text)
    text = re.sub(rf"(\w)-\s+(?!{_elision_conjunctions}\b)(\w)", r"\1\2", text)
    text = apply_replacements(text)
    text = re.sub(r"(\w)\s*=\s*(\w)", r"\1-\2", text)
    text = re.sub(
        r"\b(?=\w*[A-Z]\w*[A-Z])\w+\b",
        lambda m: m.group(0).lower() if not m.group(0).isupper() else m.group(0),
        text,
    )
    return text.strip()


SHARD_SIZE = 5000  # number of articles per parquet file


# function: save_shard
def save_shard(rows, shard_id):
    shard_df = pd.DataFrame(rows)
    out_path = run_dir / f"shard_{shard_id:05d}.parquet"
    shard_df.to_parquet(out_path, index=False)

    processed_ids = shard_df["article_id"].unique()
    with open(run_dir / "processed_ids.txt", "a") as f:
        for article_id in processed_ids:
            f.write(f"{article_id}\n")


# function: doc_to_rows
def doc_to_rows(doc, text_id):
    rows = []
    for sent_id, sent in enumerate(doc.sents):
        for token in sent:
            rows.append(
                {
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
                }
            )
    return rows


# %%
# Load already-processed ids so we can resume without redoing work
processed_ids_path = run_dir / "processed_ids.txt"
already_processed = set()
if processed_ids_path.exists():
    with open(processed_ids_path) as f:
        already_processed = {line.strip() for line in f if line.strip()}
    print(f"Found {len(already_processed)} already-processed article ids — will skip them")

# %%
# Load dataset
dataset = load_dataset("chcaa/eno-newspapers-enriched", split="train", columns=["id", "text"])
df = dataset.to_pandas()
print(f"Total rows: {len(df)}")

# Filter short or empty texts
df = df[df["text"].str.strip().str.len() > 0]
df = df[df["text"].str.count(r"\S+") >= 10]
print(f"Rows after length filtering: {len(df)}")

# Skip already-processed articles (resume support)
df["id"] = df["id"].astype(str)
df = df[~df["id"].isin(already_processed)]
print(f"Rows remaining after skipping already-processed: {len(df)}")

# shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

if args.limit:
    df = df.head(args.limit)
    print(f"--limit set: processing only {len(df)} rows")

# %%
import multiprocessing as mp

clean_start = time.time()
raw_texts = df["text"].tolist()

# Cleaning is pure-CPU regex work over independent strings so 
# parallelizable, and the GPU sits idle during this phase anyway
_n_workers = max(1, mp.cpu_count() - 1)
print(f"Cleaning {len(raw_texts)} texts using {_n_workers} worker processes...")

with mp.Pool(_n_workers) as pool:
    texts = list(
        tqdm(
            pool.imap(clean_ocr_text, raw_texts, chunksize=500),
            total=len(raw_texts),
            desc="Cleaning OCR text",
        )
    )
print(f"Cleaning took {time.time() - clean_start:.1f}s for {len(texts)} texts")
ids = df["id"].tolist()

buffer = []
# Start shard numbering after any existing shards, so resumed runs don't overwrite
existing_shards = sorted(run_dir.glob("shard_*.parquet"))
shard_id = len(existing_shards)

n_errors = 0
start_time = time.time()

for i, (text_id, doc) in enumerate(
    tqdm(
        zip(ids, nlp.pipe(texts, batch_size=args.batch_size)),
        total=len(ids),
        desc="Parsing texts",
    )
):
    try:
        buffer.extend(doc_to_rows(doc, text_id))
    except Exception:
        n_errors += 1
        print(f"\nError processing article_id={text_id}, skipping. Traceback:")
        traceback.print_exc()
        continue

    if (i + 1) % SHARD_SIZE == 0:
        save_shard(buffer, shard_id)
        buffer = []
        shard_id += 1

# save remaining documents
if buffer:
    save_shard(buffer, shard_id)

elapsed = time.time() - start_time
print(f"\nDone. Processed {len(ids)} articles in {elapsed:.1f}s ({len(ids)/elapsed:.1f} docs/sec)")
print(f"Errors skipped: {n_errors}")