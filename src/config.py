"""
config.py -- paths, feature sets, constants.

Single source of truth. Every notebook imports from here rather than
redefining FEATURES, so a change to the feature set cannot silently apply to
the factor analysis but not the models.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
FIGS_PATH = ROOT / "figs"
LOGS_PATH = ROOT / "logs"
RESOURCES = ROOT / "resources"

# Hi there.
# set this to `data/usage_features_w_pwa_test.parquet` to run a small random sample of the data
DATA_FILE = DATA_PATH / "usage_features_w_pwa.parquet"

CATEGORIES = ["National", "International", "Advertisement", "fiction"]

COMPLEXITY = ["avg_sentlen", "avg_wordlen", "avg_ndd", "avg_mdd",
              "std_ndd_resid", "std_mdd_resid"]
DIVERSITY  = ["cttr_resid", "noun_ttr_resid", "verb_ttr_resid"]
REGISTER   = ["nominal_verb_ratio", "of_ratio", "that_ratio",
              "present_tense_ratio", "passive_ratio",
              "adjective_adverb_ratio", "personal_pronoun_ratio",
              "function_word_ratio"]
AFFECT     = ["semantic_sentiment_standardized",
              "semantic_sentiment_log_var_ratio_resid"]

# readability indices, reported descriptively but excluded from the FA:
# both take sentence length as a direct term, so they are collinear with
# avg_sentlen by construction rather than empirically
READABILITY = ["lix", "rix"]

FA_FEATURES = COMPLEXITY + DIVERSITY + REGISTER + AFFECT
FEATURES = FA_FEATURES + READABILITY

# adjusted at the measurement stage (01_assemble.py), so length must NOT
# also enter the models as a covariate for these
RESIDUALISED = {
    "cttr_resid": "log_tokens", "noun_ttr_resid": "log_tokens",
    "verb_ttr_resid": "log_tokens",
    "std_ndd_resid": "num_sents", "std_mdd_resid": "num_sents",
    "semantic_sentiment_log_var_resid": "num_sents",
}

CONTROLS = ["year_centered", "category", "german_probability",
            "newspaper", "pwa"]

REPRESENTATIVE = ["avg_sentlen", "avg_mdd", "cttr_resid",
                  "noun_ttr_resid", "nominal_verb_ratio",
                  "personal_pronoun_ratio"]

DISPLAY_NAMES = {
    "semantic_sentiment_standardized": "Sentiment",
    "semantic_sentiment_log_var_ratio_resid": "Sentiment SD",
    # plus all _resid features, remove suffix
    "cttr_resid": "CTTR", "noun_ttr_resid": "Noun TTR",
    "verb_ttr_resid": "Verb TTR",
    "std_ndd_resid": "SD NDD", "std_mdd_resid": "SD MDD",
    "avg_sentlen": "Avg. sent. len.", "avg_wordlen": "Avg. word len.",
    "avg_ndd": "Avg. NDD", "avg_mdd": "Avg. MDD",
    "nominal_verb_ratio": "Nominal/verb ratio", "of_ratio": "'of' ratio",
    "that_ratio": "'that' ratio",
    "present_tense_ratio": "Present tense ratio",
    "passive_ratio": "Passive ratio",
    "adjective_adverb_ratio": "Adj./adv. ratio",
    "personal_pronoun_ratio": "Personal pronoun ratio",
    "function_word_ratio": "Function word ratio",
    "lix": "LIX", "rix": "RIX",
    "year_centered": "Year (centered)",
    "category": "Category", "german_probability": "German probability",
}

MIN_YEAR = 1740
RHO_THRESHOLD = 0.05
N_FACTORS = 3


# Genre colours. Single source of truth; mirrored in config.R.
CATEGORY_COLORS = {
    "Advertisement": "#ff6f00",   # vivid orange
    "National":      "#ff0090",   # hot magenta-pink
    "International": "#7b2ff7",   # electric purple
    "fiction":       "#00c853",   # bright green
}

# Ordered list matching CATEGORIES, for anything that wants a sequence
PALETTE = [CATEGORY_COLORS[c] for c in CATEGORIES]