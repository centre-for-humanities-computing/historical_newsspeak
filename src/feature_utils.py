
# %%
#
from pathlib import Path
import pandas as pd
import dacy
import numpy as np
import bz2
from lexical_diversity import lex_div as ld
#from semanticprojection.SemanticProjecter import SemanticProjector

# %%

SPACY_DIR = Path("data/spacy_books/")
SPACY_DIR.mkdir(parents=True, exist_ok=True)

# Load model once
nlp = dacy.load("da_dacy_large_trf-0.2.0")
if not nlp.has_pipe("senter"):
    nlp.add_pipe("senter")
nlp.initialize()

def process_text(text, text_id, nlp=nlp):
    spacy_file = SPACY_DIR / f"{text_id}_spacy.csv"
    if spacy_file.exists():
        return pd.read_csv(spacy_file)

    # Process the full text
    doc = nlp(text)
    print(doc.sents)
    all_attrs = []
    for sent_id, sent in enumerate(doc.sents):
        for token in sent:
            all_attrs.append([
                token.i,              # global token index
                token.text,
                token.lemma_,
                token.is_punct,
                token.is_stop,
                token.morph,
                token.pos_,
                token.tag_,
                token.dep_,
                token.head.i,         # global head index
                sent_id
            ])

    # Build dataframe
    df = pd.DataFrame(all_attrs, columns=[
        "token_i", "token_text", "token_lemma_", "token_is_punct",
        "token_is_stop", "token_morph", "token_pos_", "token_tag_",
        "token_dep_", "token_head", "sent_id"
    ])
    df.to_csv(spacy_file, index=False)
    return df

def read_spacy_df(text_id):
    return pd.read_csv(SPACY_DIR / f"{text_id}_spacy.csv")


# --- SYNTACTICS & STYLISTICS ---

def read_spacy_df(text_id, base_dir="data"):
    path = f"{base_dir}/spacy_books/{text_id}_spacy.csv"
    return pd.read_csv(path)


def get_pos_derived_features(text_id):
    data = read_spacy_df(text_id)

    # sanity checks
    if "token_pos_" not in data.columns or "token_morph" not in data.columns:
        raise ValueError("Expected token_pos_ and token_morph columns in spacy csv")

    # filter to actual lexical tokens
    df = data[~data["token_pos_"].isin(["PUNCT", "SPACE", "NUM"])].copy()

    # if nothing left → no features
    if df.empty:
        print(f"Df is empty for id: {text_id}")
        print(f"Original token counts: {data['token_pos_'].value_counts().to_dict()}")

        return {k: np.nan for k in [
            "nominal_verb_ratio","msttr","noun_ttr","verb_ttr",
            "personal_pronoun_ratio","function_word_ratio",
            "of_ratio","that_ratio","past_tense_ratio",
            "present_tense_ratio","passive_ratio","adjective_adverb_ratio"]}

    pos = df["token_pos_"]
    morph = df["token_morph"].fillna("").astype(str)

    # POS subsets
    nouns = df[pos == "NOUN"]
    verbs = df[pos == "VERB"]
    adjectives = df[pos == "ADJ"]
    adverbs = df[pos == "ADV"]
    nominals = df[pos.isin(["PROPN", "ADJ"])]

    # function words
    function_words = df[pos.isin(["ADP", "CCONJ", "SCONJ", "AUX", "PART"])]

    # pronouns
    personal_pronouns = df[(pos == "PRON") & morph.str.contains("PronType=Prs")]

    # lexicalized items
    of_like = df[(pos == "ADP") & df["token_text"].str.lower().eq("af")]
    that_like = df[(pos == "SCONJ") & df["token_text"].str.lower().eq("at")]

    # verb morphology
    passive = verbs[verbs["token_morph"].str.contains("Voice=Pass")]
    active_or_passive = verbs[verbs["token_morph"].str.contains("Voice=")]
    past = verbs[verbs["token_morph"].str.contains("Tense=Past")]
    present = verbs[verbs["token_morph"].str.contains("Tense=Pres")]

    # convenience counts
    total_words = len(df)
    total_verbs = len(verbs)
    total_nominals = len(nominals)

    # ratios (automatic nan if denominator == 0)
    nominal_verb_ratio = total_nominals / total_verbs if total_verbs else np.nan
    adjective_adverb_ratio = (len(adjectives) + len(adverbs)) / total_words
    personal_pronoun_ratio = len(personal_pronouns) / total_words
    function_word_ratio = len(function_words) / total_words
    of_ratio = len(of_like) / total_words
    that_ratio = len(that_like) / total_words
    past_tense_ratio = len(past) / total_verbs if total_verbs else np.nan
    present_tense_ratio = len(present) / total_verbs if total_verbs else np.nan
    passive_ratio = len(passive) / len(active_or_passive) if len(active_or_passive) else np.nan

    # lexical diversity
    msttr_val = (
        ld.msttr(df["token_text"].tolist(), window_length=100)
        if total_words >= 100 else np.nan
    )

    return {
        "nominal_verb_ratio": nominal_verb_ratio,
        "msttr": msttr_val,
        "noun_ttr": nouns["token_lemma_"].nunique() / len(nouns) if len(nouns) else np.nan,
        "verb_ttr": verbs["token_lemma_"].nunique() / len(verbs) if total_verbs else np.nan,
        "personal_pronoun_ratio": personal_pronoun_ratio,
        "function_word_ratio": function_word_ratio,
        "of_ratio": of_ratio,
        "that_ratio": that_ratio,
        "past_tense_ratio": past_tense_ratio,
        "present_tense_ratio": present_tense_ratio,
        "passive_ratio": passive_ratio,
        "adjective_adverb_ratio": adjective_adverb_ratio,
    }


def compute_basic_lengths(text_id):
    """
    Compute average word length and sentence length metrics
    from a pre-loaded spacy dataframe.
    Returns:
        avg_wordlen: float
        avg_sentlen: float
        num_sents: int
    """
    df = read_spacy_df(text_id)
    
    # Average word length
    word_lengths = [len(w) for w in df["token_text"] if isinstance(w, str)]
    avg_wordlen = np.mean(word_lengths) if word_lengths else np.nan

    # Sentence lengths
    sent_lens = df.groupby("sent_id").size()
    avg_sentlen = sent_lens.mean() if not sent_lens.empty else np.nan
    num_sents = len(sent_lens)

    return avg_wordlen, avg_sentlen, num_sents


def calculate_dependency_distances(text_id):
    df = read_spacy_df(text_id)
    dds, ndds = [], []

    for _, sent in df.groupby("sent_id"):
        sent = sent[~sent['token_is_punct'] & (sent["token_pos_"] != "SPACE")].copy()
        if sent.empty:
            continue
        root = sent[sent["token_dep_"] == "ROOT"]
        if root.empty:
            continue
        root_i = root["token_i"].iloc[0]
        start_i = sent["token_i"].min()
        root_dist = root_i - start_i

        sent["dd"] = abs(sent["token_i"] - sent["token_head_i"])
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

def compressrat(text_id):
    df = read_spacy_df(text_id)
    sentences_df = df.groupby("sent_id")
    sents = list(sentences_df["token_text"].apply(lambda x: " ".join([str(t) for t in x if isinstance(t, str)])))

    if len(sents) < 40:
        return np.nan
    if len(sents) > 50:
        selection = sents[10:50]
    else:
        selection = sents[:40]

    as_str = " ".join(selection)
    encoded = as_str.encode("utf-8")
    compressed = bz2.compress(encoded, compresslevel=9)
    return len(encoded) / len(compressed) if len(compressed) else np.nan


# --- GERMAN DETECTION ---- #

from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0 # for reproducibility

def german_probability(text_id):
    df = read_spacy_df(text_id)

    # Filter out proper nouns before grouping sentences
    df_filtered = df[df["token_pos_"] != "PROPN"]
    sentences_df = df_filtered.groupby("sent_id")

    # reconstruct sentences without PROPN tokens
    sents = [" ".join([str(t) for t in group["token_text"] if isinstance(t, str)]) for _, group in sentences_df]

    # remove empty sentences created after filtering (rare but possible)
    sents = [s.strip() for s in sents if s.strip()]

    probs = []
    for sentence in sents:
        try:
            langs = detect_langs(sentence)
        except:
            probs.append(0.0)
            continue

        de_prob = 0.0
        for l in langs:
            if l.lang == 'de':
                de_prob = l.prob
                break
        probs.append(de_prob)

    print(f"German probabilities per sentence: {probs}")
    print([(prob, sent) for prob, sent in zip(probs, sents) if prob > 0.1])
    
    return sum(probs) / len(probs) if probs else 0.0



# # --- SENTIMENT ANALYSIS ---

# def project_sentiment(text_id):
#     df = read_spacy_df(text_id)
#     sentences_df = df.groupby("sent_id")
#     sents = list(sentences_df["token_text"].apply(lambda x: " ".join([str(t) for t in x if isinstance(t, str)])))

#     # Initialize projector
#     projector = SemanticProjector()
    
#     # Project texts onto Sentiment vector
#     projector.project_texts(texts=sents, concept_vector="Sentiment")
    
#     # Standardize results
#     projector.standardize()
    
#     # Get results
#     standardized = projector.results['standardized_projection']

#     return standardized

# ###


