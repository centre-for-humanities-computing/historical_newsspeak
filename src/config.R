# config.R -- shared constants and helpers for the R scripts.
# Note: CATS has Advertisement first (reference level), unlike config.py.

library(mgcv)
library(gratia)
library(dplyr)
library(tidyr)
library(arrow)
library(ggplot2)
library(patchwork)
library(tibble)
library(here)

SEED <- 20260811   # default seed; re-set immediately before any simulation
set.seed(SEED)     # whose result is reported (see the derivative calls)

# --- paths ------------------------------------------------------------------
# here() resolves from the project root regardless of working directory, so
# these work whether a script is run from src/, notebooks/ or the root.

DATA   <- here("data")
FIGS   <- here("figs")
MODELS <- here("data", "models")
dir.create(MODELS, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGS,   showWarnings = FALSE, recursive = TRUE)

# --- window -----------------------------------------------------------------
# 1849 has a single title, so it's out.
MIN_YEAR       <- 1740
MAX_YEAR       <- 1848
INTERPRET_FROM <- 1780

# --- model ------------------------------------------------------------------
# K_YEAR / GAMMA_PENALTY / INTERPRET_FROM are only used by notebooks/archive/
# (the old article-level fits); 06_gam.R sets its own k.
K_YEAR        <- 12
K_CONTROL     <- 30
GAMMA_PENALTY <- 1.5
N_THREADS     <- 4

# The six indicators modelled: the strongest and second-strongest marker of
# each factor-analytic construct.
REPRESENTATIVE <- c("avg_sentlen", "avg_mdd",
                    "cttr_resid", "noun_ttr_resid",
                    "nominal_verb_ratio", "personal_pronoun_ratio")

CATS <- c("Advertisement", "National", "International", "fiction")
REF  <- CATS[1]   # reference level for the difference smooths

# --- length control -----------------------------------------------------------
# sentence length, MDD: no length term (length is part of the change)
# TTRs: already residualised on length upstream
# ratios: s(log_tokens) as a nuisance term
RESIDUALISED <- c("cttr_resid", "noun_ttr_resid")
NO_LENGTH    <- c("avg_sentlen", "avg_mdd", RESIDUALISED)

# positive and right-skewed; Gamma beats Gaussian on AIC even for the means
FAMILIES <- list(avg_sentlen = Gamma(link = "log"),
                 avg_mdd     = Gamma(link = "log"))

# --- plotting ---------------------------------------------------------------

PAL <- c(Advertisement = "#d95f02", National = "#e7298a",
         International = "#7570b3", fiction = "#1b9e77")

# early-window shading, only used by the archive scripts
UNIDENTIFIED <- annotate("rect", xmin = -Inf, xmax = INTERPRET_FROM,
                         ymin = -Inf, ymax = Inf, fill = "grey50", alpha = .12)

# --- helpers ----------------------------------------------------------------

`%||%` <- function(a, b) if (is.null(a)) b else a

# gratia's .by column is the variable name, so parse the level from the label
genre_of <- function(smooth_label) {
  sub("^s\\(year\\):(category|cat_ord)", "", smooth_label)
}

load_model   <- function(f) readRDS(file.path(MODELS, paste0("gam_", f, ".rds")))
year_smooths <- function(m) grep("^s\\(year\\):", smooths(m), value = TRUE)

load_gam_data <- function() {
  read_parquet(file.path(DATA, "gam_input.parquet")) |>
    filter(year >= MIN_YEAR, year <= MAX_YEAR) |>
    mutate(category      = factor(category, levels = CATS),
           cat_ord       = ordered(category, levels = CATS),
           newspaper     = factor(newspaper),
           log_tokens    = log1p(n_tokens_for_diversity),
           sents_per_100 = 100 * num_sents / n_tokens_for_diversity,
           has_pron      = as.integer(personal_pronoun_ratio > 0))
}

# article-level spec, used by notebooks/archive/ only
fit_gam <- function(feature, data, k_year = K_YEAR, gamma = GAMMA_PENALTY) {
  terms <- c("category",
             sprintf("s(year, by = category, k = %d)", k_year),
             "s(year, newspaper, bs = 'fs', m = 1, k = 10)",
             "german_probability")
  if (!feature %in% NO_LENGTH)
    terms <- c(terms, sprintf("s(log_tokens, k = %d)", K_CONTROL))

  fam <- FAMILIES[[feature]] %||% gaussian()
  if (!is.null(FAMILIES[[feature]]) && min(data[[feature]], na.rm = TRUE) <= 0) {
    warning(feature, ": non-positive values present, falling back to gaussian(). ",
            "Check that 01_assemble.py drops avg_mdd <= 0.")
    fam <- gaussian()
  }

  bam(as.formula(paste(feature, "~", paste(terms, collapse = " + "))),
      data = data, family = fam, gamma = gamma,
      method = "fREML", discrete = TRUE, nthreads = N_THREADS)
}

# difference smooths vs Advertisement (archive only)
fit_diff <- function(feature, data, k_year = K_YEAR, gamma = GAMMA_PENALTY) {
  terms <- c("category",
             sprintf("s(year, k = %d)", k_year),
             sprintf("s(year, by = cat_ord, k = %d)", k_year),
             "s(year, newspaper, bs = 'fs', m = 1, k = 10)",
             "german_probability")
  if (!feature %in% NO_LENGTH)
    terms <- c(terms, sprintf("s(log_tokens, k = %d)", K_CONTROL))

  fam <- FAMILIES[[feature]] %||% gaussian()
  if (!is.null(FAMILIES[[feature]]) && min(data[[feature]], na.rm = TRUE) <= 0)
    fam <- gaussian()

  bam(as.formula(paste(feature, "~", paste(terms, collapse = " + "))),
      data = data, family = fam, gamma = gamma,
      method = "fREML", discrete = TRUE, nthreads = N_THREADS)
}