# %%
"""
Sample ENO articles by genre, pull their text from the HF dataset, and POS-tag
each one with three models:

    da_core_news_sm     -- spaCy, modern Danish
    da_dacy_medium_trf  -- DaCy, modern Danish
    da_dacy_large_trf   -- DaCy, modern Danish

Output is long format, one row per token per model, with character offsets into
the original text so the three tokenizations can be aligned afterwards.
"""

import gc
import sys
from pathlib import Path

import pandas as pd
import spacy

sys.path.append(str(Path.cwd().parent / "src"))
from config import DATA_PATH, FIGS_PATH  # noqa: E402

ENO = DATA_PATH / "eno"
OUT_DIR = DATA_PATH / "tagged_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SAMPLES = 3
GENRES = ["Advertisement", "National", "International", "fiction"]
COLS = ["article_id", "pwa", "date", "newspaper", "category"]

REPO = "chcaa/eno-newspapers-enriched"
CACHE = DATA_PATH / "sampled_articles_with_text.parquet"

# Check exact strings with dacy.models() -- they are pinned per DaCy release.
MODELS = {
    "spacy_sm":        ("spacy", "da_core_news_sm",         32),
    "dacy_medium_trf": ("dacy",  "da_dacy_medium_trf-0.2.0", 4),
    "dacy_large_trf":  ("dacy",  "da_dacy_large_trf-0.2.0",  4),
}

# Everything needed for token.pos_ / token.tag_; the rest is dead weight.
KEEP_PIPES = {"transformer", "tok2vec", "tagger", "morphologizer", "attribute_ruler"}
SCHEMA = ["article_id", "model", "tok_i", "word", "pos", "tag", "start", "end"]

# %%
print(df.category.value_counts(dropna=False))

# %%
# ---------------------------------------------------------------- sample + fetch

def build_sample() -> pd.DataFrame:
    if CACHE.exists():
        print(f"reusing {CACHE.name}")
        return pd.read_parquet(CACHE)

    import pyarrow.dataset as ds
    from huggingface_hub import HfFileSystem, list_repo_files

    df = pd.read_parquet(DATA_PATH / "usage_features_w_pwa.parquet")
    collected = pd.concat(
        [df[df.category == g].sample(n=N_SAMPLES, random_state=SEED) for g in GENRES],
        ignore_index=True,
    )[COLS]
    collected["article_id"] = collected["article_id"].astype(str)

    files = sorted(
        f for f in list_repo_files(REPO, repo_type="dataset")
        if f.startswith("data/train-") and f.endswith(".parquet")
    )
    dset = ds.dataset(
        [f"datasets/{REPO}/{f}" for f in files], filesystem=HfFileSystem(), format="parquet"
    )
    tbl = dset.to_table(
        columns=["id", "text"],
        filter=ds.field("id").isin(collected["article_id"].tolist()),
    )
    texts = tbl.to_pandas().rename(columns={"id": "article_id"})
    texts["article_id"] = texts["article_id"].astype(str)

    collected = collected.merge(texts, on="article_id", how="left")
    missing = collected["text"].isna().sum()
    if missing:
        print(f"warning: {missing} articles had no text in the HF repo, dropping")
        collected = collected.dropna(subset=["text"]).reset_index(drop=True)

    collected.to_parquet(CACHE, index=False)
    return collected


collected = build_sample()
print(f"{len(collected)} articles, {collected.text.str.len().sum():,} characters")
collected.head()


# %%
# ---------------------------------------------------------------- tagging




def tag(nlp, collected: pd.DataFrame, model_key: str, batch_size: int) -> pd.DataFrame:
    rows = []
    pairs = zip(collected["text"].tolist(), collected["article_id"].tolist())
    for doc, article_id in nlp.pipe(pairs, as_tuples=True, batch_size=batch_size):
        i = 0
        for tok in doc:
            if tok.is_space:
                continue
            rows.append(
                {
                    "article_id": article_id,
                    "model": model_key,
                    "tok_i": i,
                    "word": tok.text,
                    "pos": tok.pos_,          # coarse UPOS
                    "tag": tok.tag_,          # fine-grained
                    "start": tok.idx,         # char offset into the raw text
                    "end": tok.idx + len(tok.text),
                }
            )
            i += 1
    return pd.DataFrame(rows, columns=SCHEMA)


def configure_transformer(nlp, window=96, stride=72, max_len=512):
    from spacy_transformers.span_getters import configure_strided_spans

    if "transformer" not in nlp.pipe_names:
        return

    trf = nlp.get_pipe("transformer")
    touched = []
    for node in trf.model.walk():
        if "tokenizer" in node.attrs:
            node.attrs["tokenizer"].model_max_length = max_len
            touched.append(f"{node.name}.tokenizer")
        if "get_spans" in node.attrs:
            node.attrs["get_spans"] = configure_strided_spans(window, stride)
            touched.append(f"{node.name}.get_spans")

    if not touched:
        raise RuntimeError(f"found no tokenizer/get_spans in {[n.name for n in trf.model.walk()]}")
    print(f"  patched: {', '.join(touched)}")

def load_model(kind: str, name: str):
    if kind == "spacy":
        nlp = spacy.load(name)
    else:
        import dacy
        nlp = dacy.load(name)

    for pipe in list(nlp.pipe_names):
        if pipe not in KEEP_PIPES:
            nlp.disable_pipe(pipe)
    nlp.max_length = 2_000_000

    configure_transformer(nlp)
    return nlp
# %%
# ---------------------------------------------------------------- run everything

frames = []
for key, (kind, name, batch_size) in MODELS.items():
    print(f"running {key} ...", flush=True)
    nlp = load_model(kind, name)
    print(f"  active pipes: {nlp.pipe_names}")

    frame = tag(nlp, collected, key, batch_size)
    frame.to_parquet(OUT_DIR / f"tagged_{key}.parquet", index=False)
    frames.append(frame)
    print(f"  {len(frame):,} tokens")

    del nlp
    gc.collect()

tagged = pd.concat(frames, ignore_index=True)
tagged = tagged.merge(
    collected[["article_id", "category", "newspaper", "date", "pwa"]],
    on="article_id", how="left",
)
tagged.to_parquet(OUT_DIR / "tagged_all_models.parquet", index=False)
print(f"\ntotal: {len(tagged):,} rows")


# %%
# ---------------------------------------------------------------- compare

wide = tagged.pivot_table(
    index=["article_id", "start", "end", "word"],
    columns="model",
    values="pos",
    aggfunc="first",
).reset_index()

shared = wide.dropna(subset=list(MODELS))
print(f"{len(shared):,} of {len(wide):,} token slots share boundaries across all three")

for a, b in [("spacy_sm", "dacy_medium_trf"),
             ("spacy_sm", "dacy_large_trf"),
             ("dacy_medium_trf", "dacy_large_trf")]:
    print(f"  {a} vs {b}: {(shared[a] == shared[b]).mean():.1%} agreement")

wide.to_csv(OUT_DIR / "pos_comparison.csv", index=False, encoding="utf-8-sig")

# Where do the two DaCy sizes diverge?
disagreements = shared[shared.dacy_medium_trf != shared.dacy_large_trf]
print(f"\nmedium/large disagreements: {len(disagreements):,}")
print(disagreements.groupby(["dacy_medium_trf", "dacy_large_trf"]).size()
      .sort_values(ascending=False).head(15))

# %%

