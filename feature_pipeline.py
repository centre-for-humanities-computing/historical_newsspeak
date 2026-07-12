# %%
"""
Stylistic feature extraction pipeline.

Reads the token-level parquet shards produced by run_pipeline.py
(each row = one token, grouped by article_id) and computes one
feature row per article:

  - POS-derived ratios (nominal/verb ratio, function word ratio, etc.)
  - Basic length stats (avg word/sentence length, sentence count)
  - Dependency distance metrics (MDD, normalized DD)
  - Compression ratio (bz2, as a rough repetitiveness/complexity proxy)
  - Readability: LIX and RIX (Scandinavian-standard, syllable-free)
  - Lexical diversity: MTLD and CTTR (replacing MSTTR, which requires
    >=100 tokens and is therefore NaN for most short newspaper articles)
  - Semantic-projection sentiment (SemanticProjector: contextual sentence
    embeddings projected onto a "Sentiment" concept vector — continuous
    score, not a classifier; see github.com/lauritswl/SemanticProjection)

Does NOT re-run spaCy/DaCy — assumes tokenization/POS/dependency parsing
already happened in run_pipeline.py and is sitting in shard_*.parquet.

IMPORTANT — standardization across shards:
SemanticProjector.standardize() z-scores relative to whatever batch of
texts you feed it. Calling it once per shard would make the same
absolute sentiment value get a DIFFERENT standardized score depending
on what else happened to be in that shard — breaking comparability
across the corpus. So this script stores only the RAW projection per
article (as sum / sum-of-squares / n, enabling exact pooled statistics
later) and defers standardization to finalize_sentiment.py, which
computes one global mean/std across all shards and applies it once.
"""

import argparse
import bz2
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from lexical_diversity import lex_div as ld
from langdetect import DetectorFactory, detect_langs
from semanticprojection.SemanticProjecter import SemanticProjector
from tqdm import tqdm

DetectorFactory.seed = 0  # reproducible langdetect results

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=str, required=True, help="Directory containing shard_*.parquet from run_pipeline.py")
parser.add_argument("--out-dir", type=str, default=None, help="Where to write feature shards (defaults to <run-dir>/features)")
parser.add_argument("--long-word-chars", type=int, default=6, help="Threshold for LIX/RIX 'long word' definition")
parser.add_argument("--german-min-words", type=int, default=4, help="Minimum sentence word count before trusting langdetect for German detection")
args = parser.parse_args()

run_dir = Path(args.run_dir)
out_dir = Path(args.out_dir) if args.out_dir else run_dir / "features"
out_dir.mkdir(parents=True, exist_ok=True)

# %%
# --- SEMANTIC PROJECTION SENTIMENT (loaded once, embeddings batched internally) ---

import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print(
        "WARNING: no GPU detected. SemanticProjector encodes every sentence "
        "through a full transformer model — on CPU, at 5M-article scale, "
        "this step alone could take days rather than hours."
    )

print("Loading SemanticProjector (paraphrase-multilingual-mpnet-base-v2)...")
_projector = SemanticProjector()
_printed_result_keys = False  # print once, so we can confirm the raw-score key name on first use


def score_sentences_semantic(sentences):
    """
    Project a list of sentences onto the "Sentiment" concept vector.
    Returns the RAW (non-standardized) projection scores, one float per
    input sentence, in input order. Standardization happens later, once,
    across the full corpus — see module docstring.
    """
    global _printed_result_keys
    if not sentences:
        return []

    _projector.project_texts(texts=sentences, concept_vector="Sentiment")

    if not _printed_result_keys:
        print(f"SemanticProjector.results keys: {list(_projector.results.keys())}")
        _printed_result_keys = True

    # Raw projection is expected under a key like "projection" (pre-
    # standardize()). Fall back defensively and surface what's actually
    # available if the assumed key name is wrong.
    results = _projector.results
    if "projection" in results:
        raw = results["projection"]
    elif "raw_projection" in results:
        raw = results["raw_projection"]
    else:
        raise KeyError(
            f"Couldn't find a raw projection key in projector.results. "
            f"Available keys: {list(results.keys())}. Update score_sentences_semantic() "
            f"to use the correct key."
        )

    return [float(x) for x in raw]


# %%
# --- helpers to reconstruct sentences from token-level shard data ---


def get_sentences(df, exclude_pos=None):
    """
    Reconstruct sentence strings from a per-article token dataframe.
    exclude_pos: optional list of POS tags to drop before joining
    (e.g. ["PROPN"] for language detection, to avoid proper nouns
    skewing the classifier).

    Note: joining tokens with a plain space loses original spacing
    around punctuation (e.g. "Hej!" -> "Hej !"). This is a known
    simplification consistent with the original pipeline's approach.
    """
    d = df if exclude_pos is None else df[~df["pos"].isin(exclude_pos)]
    sentences = (
        d.groupby("sent_id")["token_text"]
        .apply(lambda x: " ".join(str(t) for t in x if isinstance(t, str)))
    )
    return [s.strip() for s in sentences if s.strip()]


# %%
# --- SYNTACTIC & STYLISTIC FEATURES (per-article, from token dataframe) ---


def get_pos_derived_features(df):
    if "pos" not in df.columns or "morph" not in df.columns:
        raise ValueError("Expected 'pos' and 'morph' columns in token dataframe")

    d = df[~df["is_punct"] & ~df["is_space"] & (df["pos"] != "NUM")].copy()

    if d.empty:
        return {
            k: np.nan
            for k in [
                "nominal_verb_ratio", "noun_ttr", "verb_ttr",
                "personal_pronoun_ratio", "function_word_ratio",
                "of_ratio", "that_ratio", "past_tense_ratio",
                "present_tense_ratio", "passive_ratio", "adjective_adverb_ratio",
            ]
        }

    pos = d["pos"]
    morph = d["morph"].fillna("").astype(str)

    nouns = d[pos == "NOUN"]
    verbs = d[pos == "VERB"]
    adjectives = d[pos == "ADJ"]
    adverbs = d[pos == "ADV"]
    nominals = d[pos.isin(["PROPN", "ADJ"])]
    function_words = d[pos.isin(["ADP", "CCONJ", "SCONJ", "AUX", "PART"])]
    personal_pronouns = d[(pos == "PRON") & morph.str.contains("PronType=Prs")]
    of_like = d[(pos == "ADP") & d["token_text"].str.lower().eq("af")]
    that_like = d[(pos == "SCONJ") & d["token_text"].str.lower().eq("at")]

    verb_morph = verbs["morph"].fillna("").astype(str)
    passive = verbs[verb_morph.str.contains("Voice=Pass")]
    active_or_passive = verbs[verb_morph.str.contains("Voice=")]
    past = verbs[verb_morph.str.contains("Tense=Past")]
    present = verbs[verb_morph.str.contains("Tense=Pres")]

    total_words = len(d)
    total_verbs = len(verbs)
    total_nominals = len(nominals)

    return {
        "nominal_verb_ratio": total_nominals / total_verbs if total_verbs else np.nan,
        "noun_ttr": nouns["lemma"].str.lower().nunique() / len(nouns) if len(nouns) else np.nan,
        "verb_ttr": verbs["lemma"].str.lower().nunique() / total_verbs if total_verbs else np.nan,
        "personal_pronoun_ratio": len(personal_pronouns) / total_words,
        "function_word_ratio": len(function_words) / total_words,
        "of_ratio": len(of_like) / total_words,
        "that_ratio": len(that_like) / total_words,
        "past_tense_ratio": len(past) / total_verbs if total_verbs else np.nan,
        "present_tense_ratio": len(present) / total_verbs if total_verbs else np.nan,
        "passive_ratio": len(passive) / len(active_or_passive) if len(active_or_passive) else np.nan,
        "adjective_adverb_ratio": (len(adjectives) + len(adverbs)) / total_words,
    }


def compute_basic_lengths(df):
    word_lengths = [len(w) for w in df["token_text"] if isinstance(w, str)]
    avg_wordlen = np.mean(word_lengths) if word_lengths else np.nan

    sent_lens = df.groupby("sent_id").size()
    avg_sentlen = sent_lens.mean() if not sent_lens.empty else np.nan
    num_sents = len(sent_lens)

    return avg_wordlen, avg_sentlen, num_sents


def compute_readability(df, long_word_chars=6):
    """
    LIX (Läsbarhetsindex) and RIX — the standard Scandinavian-language
    readability formulas. Syllable-free (unlike Flesch), which matters
    here since Danish syllabification is unreliable, especially on
    OCR'd historical spelling.

        LIX = (words / sentences) + (100 * long_words / words)
        RIX = long_words / sentences
    """
    words_df = df[~df["is_punct"] & ~df["is_space"]]
    words = [w for w in words_df["token_text"] if isinstance(w, str)]

    num_sents = df["sent_id"].nunique()
    num_words = len(words)

    if num_words == 0 or num_sents == 0:
        return {"lix": np.nan, "rix": np.nan}

    long_words = sum(1 for w in words if len(w) > long_word_chars)

    lix = (num_words / num_sents) + (100 * long_words / num_words)
    rix = long_words / num_sents

    return {"lix": lix, "rix": rix}


def compute_lexical_diversity(df, min_tokens_for_mtld=50):
    """
    MTLD and CTTR, replacing MSTTR (which requires >=100 tokens and is
    therefore NaN for most short newspaper articles).

    - MTLD: window-free, generally more robust to text length than
      MSTTR, though still noisy under ~min_tokens_for_mtld tokens —
      no TTR-family metric is reliable on very short texts, this is
      a property of the statistic, not the implementation.
    - CTTR: always computable (types / sqrt(2N)), used as a stable
      fallback signal even on very short articles.
    """
    words_df = df[~df["is_punct"] & ~df["is_space"]]
    tokens = [str(w).lower() for w in words_df["token_text"] if isinstance(w, str)]

    n = len(tokens)
    if n == 0:
        return {"mtld": np.nan, "cttr": np.nan, "n_tokens_for_diversity": 0}

    types = len(set(tokens))
    cttr = types / np.sqrt(2 * n) if n else np.nan

    mtld_val = np.nan
    if n >= min_tokens_for_mtld:
        try:
            mtld_val = ld.mtld(tokens)
        except Exception:
            mtld_val = np.nan

    return {"mtld": mtld_val, "cttr": cttr, "n_tokens_for_diversity": n}


def calculate_dependency_distances(df):
    dds, ndds = [], []

    for _, sent in df.groupby("sent_id"):
        sent = sent[~sent["is_punct"] & ~sent["is_space"]].copy()
        if sent.empty:
            continue
        root = sent[sent["dep"] == "ROOT"]
        if root.empty:
            continue
        root_i = root["token_i"].iloc[0]
        start_i = sent["token_i"].min()
        root_dist = root_i - start_i

        sent["dd"] = abs(sent["token_i"] - sent["head_i"])
        mdd = sent["dd"].mean()
        dds.append(mdd)

        slen = len(sent)
        if mdd > 0 and slen > 0 and root_dist >= 0:
            prod = (root_dist + 1) * slen
            ndd = abs(np.log(mdd / np.sqrt(prod))) if prod else np.nan
            ndds.append(ndd)

    return {
        "avg_ndd": np.mean(ndds) if ndds else np.nan,
        "std_ndd": np.std(ndds) if ndds else np.nan,
        "avg_mdd": np.mean(dds) if dds else np.nan,
        "std_mdd": np.std(dds) if dds else np.nan,
    }


def compressrat_from_sentences(sents):
    if len(sents) < 40:
        return np.nan
    selection = sents[10:50] if len(sents) > 50 else sents[:40]

    as_str = " ".join(selection)
    encoded = as_str.encode("utf-8")
    compressed = bz2.compress(encoded, compresslevel=9)
    return len(encoded) / len(compressed) if len(compressed) else np.nan


def german_detection_metrics(df, threshold=0.9, min_words=4):
    """
    Runs detect_langs ONCE per sentence and derives both German-detection
    metrics from that single pass (previously german_probability and
    german_sentence_share each independently called detect_langs on the
    same sentences — doubling this slow, unbatched, CPU-only cost for
    no reason).

    Three fixes vs. the original version:

    1. LOWERCASED before detection. Danish, like German, capitalized all
       common nouns until the 1948 spelling reform — so older Danish
       articles superficially resemble German just from capitalization
       patterns, regardless of actual language content. Lowercasing
       removes this confound (which otherwise risks making
       german_probability track article AGE via orthography, not
       actual German-language content).
    2. Sentences under `min_words` tokens are SKIPPED entirely.
       langdetect's confidence is unreliable on short fragments — better
       to exclude them than let noise into the average. Skipped count
       is returned so you can see how much data a given article's
       estimate is actually based on.
    3. german_probability is now a LENGTH-WEIGHTED average (weighted by
       word count), not a flat mean — a 3-word sentence and a 30-word
       sentence previously counted equally, even though the longer one
       gives the detector far more signal to work with.

    Returns NaN (not 0.0) when no sentence has enough signal to use —
    a flat 0.0 would misleadingly read as "confirmed not German" rather
    than "no reliable data available".
    """
    sents = get_sentences(df, exclude_pos=["PROPN"])

    probs = []
    weights = []
    n_skipped_short = 0

    for sentence in sents:
        n_words = len(sentence.split())
        if n_words < min_words:
            n_skipped_short += 1
            continue

        try:
            langs = detect_langs(sentence.lower())
            de_prob = next((l.prob for l in langs if l.lang == "de"), 0.0)
        except Exception:
            de_prob = 0.0

        probs.append(de_prob)
        weights.append(n_words)

    if not probs:
        return {
            "german_probability": np.nan,
            "german_sentence_share": np.nan,
            "german_n_sentences_used": 0,
            "german_n_sentences_skipped_short": n_skipped_short,
        }

    probs_arr = np.array(probs, dtype=float)
    weights_arr = np.array(weights, dtype=float)

    return {
        "german_probability": float(np.average(probs_arr, weights=weights_arr)),
        "german_sentence_share": float(np.sum(probs_arr >= threshold) / len(probs_arr)),
        "german_n_sentences_used": len(probs_arr),
        "german_n_sentences_skipped_short": n_skipped_short,
    }


# %%
# --- SHARD PROCESSING ---


# %%
# --- PER-ARTICLE WORKER (module-level so it's picklable across processes) ---


def process_one_article(article_id_and_df, long_word_chars=6, german_min_words=4):
    """
    Computes every CPU-bound feature for a single article. Runs in a
    worker process — must not touch the GPU-based SemanticProjector,
    which stays in the main process (see process_shard).
    """
    article_id, df = article_id_and_df
    df = df.sort_values("token_i")

    row = {"article_id": article_id}
    row.update(get_pos_derived_features(df))

    avg_wordlen, avg_sentlen, num_sents = compute_basic_lengths(df)
    row["avg_wordlen"] = avg_wordlen
    row["avg_sentlen"] = avg_sentlen
    row["num_sents"] = num_sents

    row.update(compute_readability(df, long_word_chars=long_word_chars))
    row.update(compute_lexical_diversity(df))
    row.update(calculate_dependency_distances(df))

    sents = get_sentences(df)
    row["compressrat"] = compressrat_from_sentences(sents)
    row.update(german_detection_metrics(df, min_words=german_min_words))

    return row, article_id, sents


def process_shard(shard_path, pool):
    print(f"\nProcessing {shard_path.name}...")
    token_df = pd.read_parquet(shard_path)

    article_groups = list(token_df.groupby("article_id", sort=False))

    # First pass: compute everything except sentiment (cheap per-article,
    # but 4.67M articles' worth adds up — parallelize across CPU cores,
    # same lesson as the OCR cleaning step in run_pipeline.py. The GPU
    # sentiment step below stays single-process/main-process only.
    # NOTE: `pool` is created ONCE in __main__ and reused across every
    # shard — spawning/tearing down ~380 worker processes per shard
    # (935 times total) was adding real overhead that didn't show up
    # inside the tqdm timer, since fork/teardown happens outside the
    # timed .imap() call.
    worker = partial(
        process_one_article,
        long_word_chars=args.long_word_chars,
        german_min_words=args.german_min_words,
    )

    rows = []
    article_sentences = {}  # article_id -> list[str], collected for batched sentiment
    for row, article_id, sents in tqdm(
        pool.imap(worker, article_groups, chunksize=20),
        total=len(article_groups),
        desc=f"Features ({shard_path.name})",
    ):
        rows.append(row)
        article_sentences[article_id] = sents

    # Second pass: batched semantic-projection sentiment across ALL
    # sentences in this shard at once (SemanticProjector batches the
    # underlying sentence-transformer embedding calls internally).
    # Flatten (article_id, sentence) pairs, project once, then aggregate
    # back per article — avoids one-call-per-sentence overhead.
    flat_article_ids = []
    flat_sentences = []
    for article_id, sents in article_sentences.items():
        for s in sents:
            flat_article_ids.append(article_id)
            flat_sentences.append(s)

    print(f"Scoring semantic-projection sentiment for {len(flat_sentences)} sentences...")
    raw_scores = score_sentences_semantic(flat_sentences)

    sentiment_df = pd.DataFrame({"article_id": flat_article_ids, "raw_score": raw_scores})

    # Store sum / sum-of-squares / n per article rather than a per-shard
    # mean — this lets finalize_sentiment.py compute an EXACT pooled
    # mean/std across the whole corpus afterward, then apply a single
    # consistent standardization, rather than each shard silently having
    # its own incompatible z-score scale.
    agg = sentiment_df.groupby("article_id")["raw_score"].agg(
        semantic_sentiment_sum="sum",
        semantic_sentiment_sumsq=lambda x: (x**2).sum(),
        semantic_sentiment_n="count",
    ).reset_index()
    agg["semantic_sentiment_mean_raw"] = agg["semantic_sentiment_sum"] / agg["semantic_sentiment_n"]

    features_df = pd.DataFrame(rows).merge(agg, on="article_id", how="left")

    out_path = out_dir / f"features_{shard_path.stem}.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(features_df)} article feature rows to {out_path}")


# %%
if __name__ == "__main__":
    shard_paths = sorted(run_dir.glob("shard_*.parquet"))
    print(f"Found {len(shard_paths)} shards in {run_dir}")

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"Starting a pool of {n_workers} worker processes (created once, reused across all shards)...")

    with mp.Pool(n_workers) as pool:
        for shard_path in shard_paths:
            out_path = out_dir / f"features_{shard_path.stem}.parquet"
            if out_path.exists():
                print(f"Skipping {shard_path.name} — features already exist at {out_path}")
                continue
            process_shard(shard_path, pool)

    print("\nDone.")