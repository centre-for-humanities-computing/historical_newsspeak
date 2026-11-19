"""Sample ads for a manual check of sentence segmentation: is the drop in ad
sentence length (34 -> 20 tokens) shorter prose, or punctuation/layout?

Draws 30 ads per window (1740-69, 1820-49), spread over newspapers, and shows
each sentence as DaCy split it. Needs the ENO token shards from
src/run_pipeline.py (on the pipeline machine). Fill in the empty columns of
data/ad_check/ad_articles.csv.

  python notebooks/sample_ad_sentences.py --shards data/spacy_books/run_*/shard_*.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "eno" / "pwa.parquet"
FEATS = ROOT / "data" / "usage_features_w_pwa.parquet"
OUT = ROOT / "data" / "ad_check"

WINDOWS = {"1740-69": (1740, 1769), "1820-49": (1820, 1849)}
N_PER_WINDOW = 30
MIN_PWA = 0.5
SEED = 20260811

parser = argparse.ArgumentParser()
parser.add_argument("--shards", nargs="+", required=True)
parser.add_argument("--n", type=int, default=N_PER_WINDOW)
args = parser.parse_args()
shards = [Path(s).expanduser() for s in args.shards]

# --- which advertisements are in the shards? --------------------------------
ids_in_shards = set()
for s in shards:
    ids_in_shards |= set(pc.unique(pq.read_table(s, columns=["article_id"])["article_id"]).to_pylist())
print(f"{len(ids_in_shards):,} articles in {len(shards)} shard(s)")

meta = pd.read_parquet(META, columns=["id", "pwa", "date", "newspaper",
                                      "predicted_category", "fictionality_tag"])
meta = meta.rename(columns={"id": "article_id"})
meta["year"] = pd.to_datetime(meta["date"], errors="coerce").dt.year
is_fiction = meta["fictionality_tag"].astype(str).str.lower().str.startswith("fic")
ads = meta[(meta["predicted_category"] == "Advertisement") & ~is_fiction
           & (meta["pwa"] >= MIN_PWA) & meta["article_id"].isin(ids_in_shards)]

feats = pd.read_parquet(FEATS, columns=["article_id", "avg_sentlen", "num_sents",
                                        "n_tokens_for_diversity"])
ads = ads.merge(feats, on="article_id", how="inner")


# --- round-robin over newspapers within each window -------------------------
def sample_window(df, n, seed):
    df = df.sample(frac=1, random_state=seed)            # shuffle once
    groups = [g for _, g in df.groupby("newspaper")]
    picks, i = [], 0
    while len(picks) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(picks) < n:
                picks.append(g.iloc[i])
        i += 1
    return pd.DataFrame(picks)


samples = []
for label, (lo, hi) in WINDOWS.items():
    w = ads[(ads["year"] >= lo) & (ads["year"] <= hi)]
    s = sample_window(w, args.n, SEED)
    s["window"] = label
    print(f"{label}: {len(w):,} ads available in shards, "
          f"{w['newspaper'].nunique()} titles -> sampled {len(s)}")
    samples.append(s)
sample = pd.concat(samples, ignore_index=True)
if sample.empty:
    raise SystemExit("No advertisements from these windows in the given shards.")

# --- rebuild the pipeline's sentences for the sampled articles --------------
want = set(sample["article_id"])
tok = pd.concat(
    pq.read_table(s, columns=["article_id", "token_i", "token_text", "is_space",
                              "sent_id", "start_char", "end_char"],
                  filters=[("article_id", "in", list(want))]).to_pandas()
    for s in shards)
tok = tok.sort_values(["article_id", "token_i"])


def article_text(t):
    """Rebuild the text from token offsets, keeping line breaks."""
    out, pos = [], 0
    for _, r in t.iterrows():
        if r.start_char > pos:
            out.append(" ")
        out.append(str(r.token_text))
        pos = r.end_char
    return "".join(out)


sent_rows, texts = [], {}
for aid, t in tok.groupby("article_id", sort=False):
    texts[aid] = article_text(t)
    for k, (sid, s) in enumerate(t.groupby("sent_id", sort=True), start=1):
        words = s[~s["is_space"].astype(bool)]
        text = " ".join(str(x) for x in words["token_text"])
        last = str(words["token_text"].iloc[-1]) if len(words) else ""
        sent_rows.append({
            "article_id": aid, "sent_no": k, "n_tokens": len(words),
            "ends_with": last if not last.isalnum() else "(word)",
            "line_break_inside": bool(s["token_text"].astype(str).str.contains("\n").any()),
            "sentence": text})

sents = pd.DataFrame(sent_rows)
meta_cols = sample[["article_id", "window", "year", "newspaper"]]
sents = meta_cols.merge(sents, on="article_id").sort_values(
    ["window", "newspaper", "article_id", "sent_no"])

arts = sample[["article_id", "window", "year", "newspaper", "pwa",
               "num_sents", "avg_sentlen"]].copy()
arts["text"] = arts["article_id"].map(texts)
for c in ["boundaries_ok", "boundary_kind", "list_like", "comment"]:
    arts[c] = ""
arts = arts.sort_values(["window", "newspaper", "year"])

OUT.mkdir(parents=True, exist_ok=True)
arts.to_csv(OUT / "ad_articles.csv", index=False)
sents.to_csv(OUT / "ad_sentences.csv", index=False)

print("\nwhat ends the pipeline's sentences, by window (share):")
print(pd.crosstab(sents["window"], sents["ends_with"], normalize="index")
        .round(2).T.sort_values(sents["window"].iloc[0], ascending=False).head(8))
print("\nshare of sentences containing a line break:",
      sents.groupby("window")["line_break_inside"].mean().round(2).to_dict())
print(f"\nwrote {OUT / 'ad_articles.csv'} and {OUT / 'ad_sentences.csv'}")
