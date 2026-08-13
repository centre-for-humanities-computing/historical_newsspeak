# config.R --------------------------------------------------------------------
# Shared constants and helpers for the GAM scripts. Sourced by 06_gam.R,
# 06b_gam_sensitivity.R, 06c_boundary_uncertainty.R and 06d_export_for_python.R:
#
#   library(here); source(here("src", "config.R"))
#
# Side effects on sourcing: creates the output directories and sets a default
# seed. It defines load_gam_data() and fit_gam() but does not call them.
#
# Relation to src/config.py: REPRESENTATIVE here corresponds to REPRESENTATIVE
# there, NOT to config.py's FEATURES (which is the full 21-feature set used for
# the factor analysis). CATS corresponds to CATEGORIES but is in a DIFFERENT
# ORDER: Advertisement first, since it is the reference for the difference
# smooths.

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
# We FIT from 1740 to 1848 but INTERPRET only from 1780.
#
# The corpus holds one or two titles in most years before 1760 and fewer than
# eight before 1773, so in the early decades a change over time and a change in
# which newspaper is represented are not separable in principle. The two
# plausible specifications differ by up to 34 tokens per sentence before 1760
# and agree to within 12% from 1780 (robustness section (c) of 06_gam.R).
#
# Coverage thins again at the end -- 7 titles in 1845, 6 in 1847, 4 in 1848, 1
# in 1849 -- and we exclude only the final single-title year. Note that a
# spline endpoint is weakly constrained regardless of coverage, since there is
# data on one side only, so periods terminating at MAX_YEAR are read with that
# in mind.
MIN_YEAR       <- 1740
MAX_YEAR       <- 1848
INTERPRET_FROM <- 1780

# --- model ------------------------------------------------------------------
# The basis dimension, NOT the penalty, sets the timescale the model can
# resolve: edf sits near the k ceiling at every k tried. K_YEAR = 12 is
# therefore an explicit choice of roughly decadal resolution; at k = 30 the
# model resolved 4-year oscillations and reported them as significant change.
# GAMMA_PENALTY does secondary work. 06b_gam_sensitivity.R checks that the
# reported periods survive k in {8,12,16} and gamma in {1,1.5,2}.
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

# --- length-control policy --------------------------------------------------
# avg_sentlen, avg_mdd: no length term. These are partly constitutive of
#   article length (rho = 0.64 with token count), and articles growing shorter
#   is part of how the prose changed. Conditioning on token count removes the
#   diachronic signal and reverses the sign.
# cttr_resid, noun_ttr_resid: already residualised at the measurement stage
#   (01_assemble.py), where the dependence is a construction artefact of
#   type-token ratios. A further term would adjust twice.
# nominal_verb_ratio, personal_pronoun_ratio: keep s(log_tokens). These are
#   rates over words with no mechanical length dependence, so length is a
#   genuine nuisance covariate.
RESIDUALISED <- c("cttr_resid", "noun_ttr_resid")
NO_LENGTH    <- c("avg_sentlen", "avg_mdd", RESIDUALISED)

# Strictly positive and right-skewed; Gaussian at this n lets the upper tail
# steer the fit. Check figs/gam_resid_*.pdf.
FAMILIES <- list(avg_sentlen = Gamma(link = "log"),
                 avg_mdd     = Gamma(link = "log"))

# --- plotting ---------------------------------------------------------------

PAL <- c(Advertisement = "#d95f02", National = "#e7298a",
         International = "#7570b3", fiction = "#1b9e77")

# Shading for the unidentified early window, added to every MODEL figure so a
# reader can see which part of a curve is being claimed. NOT added to the
# raw-data checks, where the point is to show the unadjusted aggregate.
UNIDENTIFIED <- annotate("rect", xmin = -Inf, xmax = INTERPRET_FROM,
                         ymin = -Inf, ymax = Inf, fill = "grey50", alpha = .12)

# --- helpers ----------------------------------------------------------------

`%||%` <- function(a, b) if (is.null(a)) b else a

# gratia's .by column holds the by-VARIABLE name ("category"), constant across
# rows, so grouping on it silently collapses all four genres into one series.
# Parse the level out of the smooth label instead.
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

# The model specification, defined once so that 06_gam.R and the sensitivity
# sweep in 06b cannot drift apart. s(year, newspaper, bs = "fs") rather than
# s(newspaper, bs = "re"): Berlingske and Kbh. Adresseavis account for roughly
# a third of the corpus and both end in the 1830s, so a random INTERCEPT
# (level only) would leave title-specific TRAJECTORIES free to masquerade as
# corpus trend.
# NOTE: bs = "ad" (adaptive) is NOT supported under discrete = TRUE.
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

# As above but with an ordered factor in the by =, giving each genre's
# DEPARTURE from REF rather than its own trajectory. The parametric term stays
# the UNORDERED factor: an ordered factor there gives polynomial contrasts,
# which are meaningless for nominal genres.
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