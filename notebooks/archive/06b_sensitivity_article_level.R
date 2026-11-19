# 06b_sensitivity.R -----------------------------------------------------------
# The reported periods come from one choice of basis dimension (k) and penalty
# (gamma). How much does that choice matter?
#
# Two questions of each genre-year, across nine configurations:
#   - do they agree on the DIRECTION of change?
#   - do they agree on whether change is DETECTABLE at all?
#
# A coarser basis has less power, so disagreement about detectability is
# expected; disagreement about direction is not. Both are reported.
#
# 54 fits, ~45 min on first run. Results are cached to CSV: rerunning skips
# straight to the analysis. Delete the CSVs in data/gam/ to force a refit.

library(here)
source(here("src", "config.R"))
library(purrr)

d   <- load_gam_data()
OUT <- file.path(DATA, "gam"); dir.create(OUT, showWarnings = FALSE)

K_GRID     <- c(8, 12, 16)
GAMMA_GRID <- c(1, 1.5, 2)
N_CONFIG   <- length(K_GRID) * length(GAMMA_GRID)
MIN_RUN    <- 9    # grid points, ~5 years: shorter than the basis can resolve

STATES_CSV  <- file.path(OUT, "sensitivity_states.csv")
SMOOTHS_CSV <- file.path(OUT, "sensitivity_smooths_avg_sentlen.csv")


# --- 1. the sweep ------------------------------------------------------------

if (file.exists(STATES_CSV)) {
  cat("loading cached sweep:", STATES_CSV, "\n")
  sens <- read.csv(STATES_CSV)
} else {
  sens <- expand.grid(feature = REPRESENTATIVE, k = K_GRID, gamma = GAMMA_GRID,
                      stringsAsFactors = FALSE) |>
    pmap(function(feature, k, gamma) {
      cat(sprintf("%-24s k=%2d g=%.1f ... ", feature, k, gamma))
      t0 <- Sys.time()
      m <- fit_gam(feature, d, k_year = k, gamma = gamma)
      set.seed(SEED)   # same posterior draws for every configuration, so
                       # differences reflect the model, not the simulation
      out <- derivatives(m, select = year_smooths(m), type = "central",
                         interval = "simultaneous", n = 200, n_sim = 10000) |>
        transmute(feature, k, gamma, genre = genre_of(.smooth), year,
                  state = ifelse((.lower_ci > 0) | (.upper_ci < 0),
                                 sign(.derivative), 0))
      rm(m); gc()
      cat(sprintf("%.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
      out
    }) |> list_rbind()
  write.csv(sens, STATES_CSV, row.names = FALSE)
}

stopifnot(all(count(sens, feature, genre, year)$n == N_CONFIG))


# --- 2. how far do the configurations agree? --------------------------------

agreement <- sens |>
  group_by(feature, genre, year) |>
  summarise(n_pos         = sum(state == 1),
            n_neg         = sum(state == -1),
            unanimous     = n_distinct(state) == 1,
            contradiction = any(state == 1) & any(state == -1),
            majority      = case_when(n_pos >= 6 ~ 1, n_neg >= 6 ~ -1, TRUE ~ 0),
            .groups = "drop")

cat("\nacross", N_CONFIG, "configurations:\n")
agreement |>
  group_by(feature) |>
  summarise(same_classification = mean(unanimous),
            contradictory       = mean(contradiction), .groups = "drop") |>
  as.data.frame() |> print(digits = 2)

cat(sprintf("\noverall: same classification %.0f%% | contradictory %.0f%%\n",
            100 * mean(agreement$unanimous), 100 * mean(agreement$contradiction)))

write.csv(agreement, file.path(OUT, "sensitivity_agreement.csv"), row.names = FALSE)


# --- 3. periods, strict and loose -------------------------------------------
# strict: all nine configurations classify the year identically
# loose : at least six find the same direction and none finds the opposite

runs_from <- function(df, state_col) {
  df |>
    group_by(feature, genre) |>
    arrange(year, .by_group = TRUE) |>
    mutate(s = .data[[state_col]],
           run = cumsum(c(1, diff(year)) > 1.5 |
                        s != lag(s, default = first(s)))) |>
    group_by(feature, genre, run, s) |>
    summarise(from = min(year), to = max(year), n = n(), .groups = "drop") |>
    filter(n >= MIN_RUN) |>
    arrange(feature, genre, from)
}

strict <- agreement |>
  mutate(state = ifelse(unanimous & n_pos == N_CONFIG, 1,
                 ifelse(unanimous & n_neg == N_CONFIG, -1, 0))) |>
  filter(state != 0) |>
  runs_from("state")

loose <- agreement |>
  filter(!contradiction, majority != 0) |>
  runs_from("majority")

cat("\nstrict:", nrow(strict), "periods | loose:", nrow(loose), "\n")


# --- 4. the reported table --------------------------------------------------
# Every strict period is a subspan of a loose one, so we report the loose set
# and mark which are unanimous rather than printing two tables: the
# configurations disagree about the EXTENT of periods, not their existence.

FEAT_LAB <- c(avg_sentlen            = "Sentence length",
              avg_mdd                = "Dependency distance",
              cttr_resid             = "CTTR",
              noun_ttr_resid         = "Noun TTR",
              nominal_verb_ratio     = "Nominal/verb ratio",
              personal_pronoun_ratio = "Personal pronoun ratio")

prep <- function(df) {
  df |> mutate(direction = ifelse(s > 0, "increase", "decrease"),
               from_yr = floor(from), to_yr = ceiling(to),
               panel = factor(FEAT_LAB[feature], levels = unname(FEAT_LAB)),
               genre = factor(as.character(genre), levels = rev(CATS)))
}

loose_bars  <- prep(loose)
strict_bars <- prep(strict)

periods_out <- loose_bars |>
  rowwise() |>
  mutate(unanimous = any(strict_bars$feature == feature &
                         as.character(strict_bars$genre) == as.character(genre) &
                         strict_bars$s == s &
                         strict_bars$from <= to & strict_bars$to >= from)) |>
  ungroup() |>
  select(feature, panel, genre, direction, from_yr, to_yr, unanimous) |>
  arrange(panel, genre, from_yr)

cat("\n", nrow(periods_out), "periods,", sum(periods_out$unanimous), "unanimous\n\n")
print(as.data.frame(periods_out))
write.csv(periods_out, file.path(OUT, "periods_reported.csv"), row.names = FALSE)


# --- 5. figure: when does change happen? ------------------------------------
# 3 rows x 2 columns, one row per construct, so the two markers of a construct
# sit side by side. Pale bar = detected by >=6 of 9; solid core = all 9. Grey
# backdrop spans the window, so an empty row reads as "no period detected"
# rather than as missing data.

backdrop <- tidyr::expand_grid(
  genre = factor(CATS, levels = rev(CATS)),
  panel = factor(unname(FEAT_LAB), levels = unname(FEAT_LAB)))

p <- ggplot() +
  geom_linerange(data = backdrop,
                 aes(y = genre, xmin = MIN_YEAR, xmax = MAX_YEAR),
                 linewidth = 3.5, colour = "grey94") +
  geom_linerange(data = periods_out,
                 aes(y = genre, xmin = from_yr, xmax = to_yr, colour = direction),
                 linewidth = 3.5, alpha = .35) +
  geom_linerange(data = strict_bars,
                 aes(y = genre, xmin = from_yr, xmax = to_yr, colour = direction),
                 linewidth = 3.5) +
  facet_wrap(~ panel, ncol = 2, dir = "h") +
  scale_colour_manual(values = c(increase = "#0072B2", decrease = "#D55E00")) +
  scale_x_continuous(breaks = seq(1750, 1840, 30)) +
  coord_cartesian(xlim = c(MIN_YEAR, MAX_YEAR)) +
  labs(x = NULL, y = NULL, colour = NULL) +
  theme_minimal(base_size = 8) +
  theme(legend.position    = "top",
        panel.grid.minor   = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.spacing      = unit(0.8, "lines"),
        strip.text         = element_text(size = 8, hjust = 0))

ggsave(file.path(FIGS, "gam_periods.pdf"), p, width = 7, height = 5)


# --- 6. do the CURVES agree, or only the classifications? -------------------
# The agreement table says configurations disagree about dated periods. This
# asks whether they disagree about the trajectory itself, or only about where
# its slope is steep enough to detect. One feature is enough to answer it.

SENS_F <- "avg_sentlen"

if (file.exists(SMOOTHS_CSV)) {
  cat("\nloading cached smooths:", SMOOTHS_CSV, "\n")
  sens_smooths <- read.csv(SMOOTHS_CSV)
} else {
  sens_smooths <- expand.grid(k = K_GRID, gamma = GAMMA_GRID) |>
    pmap(function(k, gamma) {
      cat(sprintf("%s k=%2d g=%.1f ... ", SENS_F, k, gamma))
      m <- fit_gam(SENS_F, d, k_year = k, gamma = gamma)
      out <- smooth_estimates(m, select = year_smooths(m), n = 200) |>
        transmute(k, gamma, config = sprintf("k=%d, g=%.1f", k, gamma),
                  genre = genre_of(.smooth), year, .estimate)
      rm(m); gc(); cat("done\n")
      out
    }) |> list_rbind()
  write.csv(sens_smooths, SMOOTHS_CSV, row.names = FALSE)
}

p <- sens_smooths |>
  mutate(genre = factor(genre, levels = CATS)) |>
  ggplot(aes(year, .estimate, colour = factor(k),
             linetype = factor(gamma), group = config)) +
  geom_hline(yintercept = 0, colour = "grey80", linewidth = .3) +
  geom_line(linewidth = .5) +
  facet_wrap(~ genre) +
  scale_colour_manual(values = c("8" = "#74a9cf", "12" = "#3690c0",
                                 "16" = "#023858")) +
  scale_linetype_manual(values = c("1" = "dotted", "1.5" = "solid",
                                   "2" = "dashed")) +
  guides(colour   = guide_legend(order = 1),
         linetype = guide_legend(order = 2,
                                 override.aes = list(colour = "grey20"))) +
  labs(y = "Partial effect (log scale)", x = NULL,
       colour = "k", linetype = expression(gamma)) +
  theme_minimal(base_size = 9)

ggsave(file.path(FIGS, "gam_sensitivity_curves.pdf"), p, width = 7, height = 5)

# How far apart are the configurations at each year, relative to the spread of
# the reported trajectory? Small values mean the curves agree and the
# disagreement above is about detection, not about shape.
sens_smooths |>
  group_by(genre, year) |>
  summarise(spread = max(.estimate) - min(.estimate), .groups = "drop") |>
  group_by(genre) |>
  summarise(mean_spread = mean(spread), max_spread = max(spread),
            .groups = "drop") |>
  as.data.frame() |> print(digits = 3)


# Spread relative to the curves own range

sens_smooths |>
  group_by(genre) |>
  summarise(curve_range = max(.estimate) - min(.estimate),
            mean_spread = mean(tapply(.estimate, year, \(x) max(x) - min(x))),
            ratio = mean_spread / curve_range, .groups = "drop")

# and where the spread sits

sens_smooths |>
  group_by(genre, year) |>
  summarise(spread = max(.estimate) - min(.estimate), .groups = "drop") |>
  mutate(era = cut(year, c(-Inf, 1780, 1820, Inf))) |>
  group_by(era) |> summarise(mean_spread = mean(spread))
  
cat("\ndone. tables in", OUT, "\n")


