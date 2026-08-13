"""
reprints.py -- reprint detection and diffusion-lag measurement for ENO newspapers.

  1. Candidates come from the dataset's `pooled` embeddings, blocked by a time
     window. Poems can be compared all-pairs within a language; 4.9M articles
     cannot, but reprints appear within weeks, so a sliding date window cuts
     the comparison space by orders of magnitude before any similarity is
     computed.

  2. Embedding similarity alone is not enough. It encodes topic (mainly), not textual
     identity: two independently written reports of the same event, days apart,
     in different towns, are close in embedding space AND look exactly like a
     diffusion event. That false positive is perfectly correlated with the
     signal, so it does not average out -- it biases the lag estimate toward
     the news cycle. Every candidate is therefore verified by n-gram overlap
     on the raw text.

Here nothing is discarded. The earliest member of a component is simply the point every other
member's date is measured against, and its town is the diffusion origin.

Two lag views come out, and the difference matters:

  pairs['lag_days']   days between the two dates of every matched pair.
  lags['lag_days']    days from a component's origin to each member.

Use the pair view to inspect matches. Do NOT aggregate it into town-to-town
routes: if an item appears in Copenhagen (d0), Odense (d+4) and Aalborg (d+14),
the pair table contains an Odense->Aalborg edge of 10 days, and nothing
travelled that route. Every cluster fabricates edges between all its members.
`lag_table` therefore works on the origin view and conditions on one source
town.

    df  = pd.read_parquet('eno/light/1790s.parquet')
    E   = np.load('eno/emb/1790s.npy')
    res = find_reprints(df, E, towns)
"""
from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd

from config import DATA_PATH, DATA_FILE, FIGS_PATH, CATEGORIES #, FA_FEATURES, DISPLAY_NAMES, COMPLEXITY, DIVERSITY, REGISTER, AFFECT

MODEL = "logit"
year_cutoff = 1740


df = pd.read_parquet(DATA_FILE)
df["category"] = pd.Categorical(df["category"], categories=CATEGORIES)

# print VC
print(df.category.value_counts().to_string())

df.head()


# --------------------------------------------------------------- normalising

_FOLD = str.maketrans({
    "å": "aa", "Å": "aa", "æ": "ae", "Æ": "ae", "ø": "oe", "Ø": "oe",
    "ä": "ae", "ö": "oe", "ü": "ue", "é": "e", "è": "e", "ſ": "s",
})
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    t = text.lower().translate(_FOLD).replace("ß", "ss")
    return _WS.sub(" ", _NONWORD.sub(" ", t)).strip()


def ngrams(text: str, n: int = 5, word_level: bool = False) -> np.ndarray:
    """
    n-gram hashes. Character-level by default: it degrades gracefully under
    OCR error, where a single mangled character breaks only the n grams
    containing it, whereas word n-grams lose the whole word.
    """
    t = normalise(text)
    if word_level:
        w = t.split()
        grams = {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    else:
        grams = {t[i:i + n] for i in range(len(t) - n + 1)}
    if not grams:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.fromiter(
        (zlib.crc32(g.encode()) for g in grams), dtype=np.int64, count=len(grams)))


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """
    |A n B| / min(|A|,|B|).

    Not Jaccard: provincial papers abridged what they reprinted, and Jaccard
    between a text and a 40% excerpt caps near 0.4 by construction, so abridged
    reprints are unreachable at any threshold. Containment stays ~1.0 for a
    true excerpt.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    return np.intersect1d(a, b, assume_unique=True).size / min(a.size, b.size)


# ------------------------------------------------------- candidate retrieval

def time_blocks(dates: np.ndarray, window_days: int = 90, stride: int = 45):
    """Overlapping index blocks so every within-window pair lands together."""
    order = np.argsort(dates)
    d = dates[order]
    day0 = d[0]
    offs = ((d - day0) / np.timedelta64(1, "D")).astype(int)
    out, start = [], 0
    while start <= offs[-1]:
        lo = np.searchsorted(offs, start)
        hi = np.searchsorted(offs, start + window_days, side="right")
        if hi - lo > 1:
            out.append(order[lo:hi])
        start += stride
    return out


def block_candidates(E: np.ndarray, idx: np.ndarray, papers: np.ndarray,
                     min_cos: float, top_k: int):
    """Cosine kNN inside one time block. Cross-paper pairs only."""
    x = E[idx]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    sims = x @ x.T
    np.fill_diagonal(sims, -1.0)
    k = min(top_k, len(idx) - 1)
    nn = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    out = []
    for a in range(len(idx)):
        for b in nn[a]:
            if b <= a or sims[a, b] < min_cos:
                continue
            i, j = idx[a], idx[b]
            if papers[i] != papers[j]:
                out.append((int(i), int(j), float(sims[a, b])))
    return out


# ------------------------------------------------------------------ clusters

class _UF:
    def __init__(self): self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[rx] = ry


# ------------------------------------------------------------------ pipeline

def find_reprints(df: pd.DataFrame, E: np.ndarray, towns: pd.DataFrame,
                min_cos: float = 0.80, min_containment: float = 0.60,
                window_days: int = 90, top_k: int = 30, n: int = 5,
                word_level: bool = False, min_chars: int = 200,
                town_priority: list[str] | None = None, verbose: bool = True):
    """
    df: id, newspaper, date, text (row-aligned with E).
    towns: newspaper, town, dist_cph_km.
    town_priority: tie-break order for the primary variant, e.g. ['København'].

    Returns dict with:
      pairs     verified reprint pairs
      primary   {article_id: origin_id} -- optional, nothing is removed
      lags      per-article lag from its component's origin
    """
    d = df.reset_index(drop=True)
    keep = d.text.str.len().fillna(0) >= min_chars
    d, E = d[keep].reset_index(drop=True), E[keep.values]
    dates = pd.to_datetime(d.date).values
    papers = d.newspaper.values
    if verbose:
        print(f"  {len(d):,} articles after min_chars={min_chars}")

    blocks = time_blocks(dates, window_days, window_days // 2)
    cands = {}
    for bl in blocks:
        for i, j, c in block_candidates(E, bl, papers, min_cos, top_k):
            key = (i, j) if i < j else (j, i)
            cands[key] = c
    if verbose:
        print(f"  {len(blocks)} time blocks -> {len(cands):,} embedding candidates")

    # verify only the texts that appear in some candidate pair
    need = sorted({i for p in cands for i in p})
    gram = {i: ngrams(d.text.iloc[i], n, word_level) for i in need}

    uf, rows = _UF(), []
    for (i, j), cos in cands.items():
        c = containment(gram[i], gram[j])
        if c < min_containment:
            continue
        src, tgt = (i, j) if dates[i] <= dates[j] else (j, i)
        rows.append({"src_id": d.id.iloc[src], "tgt_id": d.id.iloc[tgt],
                     "src_paper": papers[src], "tgt_paper": papers[tgt],
                     "src_date": dates[src], "tgt_date": dates[tgt],
                     "cosine": round(cos, 3), "containment": round(c, 3),
                     "lag_days": int((dates[tgt] - dates[src])
                                     / np.timedelta64(1, "D"))})
        uf.union(i, j)
    pairs = pd.DataFrame(rows)
    if verbose:
        print(f"  {len(pairs):,} verified pairs "
              f"(containment >= {min_containment})")
    if pairs.empty:
        return {"pairs": pairs, "primary": {}, "lags": pd.DataFrame()}

    # components -> primary variant = earliest, tie-broken by town priority
    tmap = towns.set_index("newspaper")
    comp = defaultdict(list)
    for i in {i for p in cands for i in p if uf.find(i) is not None}:
        if i in uf.p:
            comp[uf.find(i)].append(i)

    prio = {t: k for k, t in enumerate(town_priority or [])}
    primary, lag_rows = {}, []
    for members in comp.values():
        members.sort(key=lambda i: (
            dates[i], prio.get(tmap.town.get(papers[i]), 99)))
        origin = members[0]
        otown = tmap.town.get(papers[origin])
        for i in members:
            primary[d.id.iloc[i]] = d.id.iloc[origin]
            if i == origin:
                continue
            lag_rows.append({
                "article_id": d.id.iloc[i],
                "src_town": otown, "tgt_town": tmap.town.get(papers[i]),
                "dist_km": tmap.dist_cph_km.get(papers[i]),
                "src_date": dates[origin],
                "lag_days": int((dates[i] - dates[origin])
                                / np.timedelta64(1, "D")),
                "component_size": len(members)})
    lags = pd.DataFrame(lag_rows)
    if verbose:
        print(f"  {len(comp):,} components, {len(lags):,} non-origin members")
    return {"pairs": pairs, "primary": primary, "lags": lags}


def lag_table(lags: pd.DataFrame, source_town: str = "København",
              by_decade: bool = False, min_n: int = 5) -> pd.DataFrame:
    """Median transit time per town-pair. Restricted to one origin town by
    default: mixed origins let transitive structure fabricate routes."""
    m = lags[(lags.src_town == source_town) & (lags.tgt_town != source_town)]
    if m.empty:
        return pd.DataFrame()
    keys = ["src_town", "tgt_town", "dist_km"]
    if by_decade:
        m = m.assign(decade=(pd.to_datetime(m.src_date).dt.year // 10) * 10)
        keys.append("decade")
    g = (m.groupby(keys)
           .agg(n=("lag_days", "size"), median_lag=("lag_days", "median"),
                q25=("lag_days", lambda s: s.quantile(.25)),
                q75=("lag_days", lambda s: s.quantile(.75)))
           .reset_index())
    return g[g.n >= min_n].sort_values("dist_km").reset_index(drop=True)
