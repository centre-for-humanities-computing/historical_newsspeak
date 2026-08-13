# 06_gam.R --------------------------------------------------------------------
# Diachronic trajectories of six representative indicators, by genre.
#
# A linear model returns the average rate of change over the period, which
# exists for any data including data where nothing changed at a constant rate.
# A GAM estimates the shape instead: a penalised curve per genre, from which we
# read (a) the trajectory and (b) the years where its slope is reliably
# non-zero.
#
# Run after 05_export_gam_input.py. Model settings live in src/config.R.
# The periods table and figure come from 06b_sensitivity.R, which needs the
# models fitted here.
#
# Outputs, in data/gam/ and figs/:
#   titles_per_year.{csv,tex}        coverage, the caveat behind section 6
#   within_title_slopes.csv          do titles move together?
#   edf_per_title.csv                is the fs term earning its keep?
#   edf.csv                          k.check diagnostics
#   smooths.csv                      fitted trajectories with intervals
#   trajectory_precision.{csv,tex}   range relative to band width
#   identification.csv               fs vs random-intercept divergence by year
#   gam_trajectories.pdf             main figure
#   gam_identification_check.pdf     appendix figure

library(here)
source(here("src", "config.R"))
library(purrr)

FEATURES <- REPRESENTATIVE
OUT <- file.path(DATA, "gam"); dir.create(OUT, showWarnings = FALSE)

# Tables that go into the paper are written as LaTeX as well as CSV, so a
# rerun cannot leave the manuscript reporting stale numbers.
write_tex <- function(df, file, caption, label, digits = 2) {
  if (!requireNamespace("xtable", quietly = TRUE)) {
    message("xtable not installed; skipping ", file); return(invisible(NULL))
  }
  print(xtable::xtable(as.data.frame(df), caption = caption,
                       label = label, digits = digits),
        file = file.path(OUT, file), booktabs = TRUE,
        include.rownames = FALSE, caption.placement = "top")
}


# --- 1. DATA -----------------------------------------------------------------

d <- load_gam_data()

cat(nrow(d), "rows,", nlevels(d$newspaper), "newspapers,",
    min(d$year), "-", max(d$year), "\n")

# Titles per year. Coverage is thin at both ends, which is the caveat the
# identification check in section 6 quantifies.
title_counts <- d |>
  group_by(year) |>
  summarise(titles = n_distinct(newspaper), articles = n(), .groups = "drop")

write.csv(title_counts, file.path(OUT, "titles_per_year.csv"), row.names = FALSE)
write_tex(title_counts, "titles_per_year.tex",
          "Distinct newspaper titles and articles per year.",
          "tab:titles_per_year", digits = 0)

cat("years with <8 titles:", sum(title_counts$titles < 8),
    "of", nrow(title_counts),
    "| last:", max(title_counts$year[title_counts$titles < 8]), "\n")

# personal_pronoun_ratio is zero in ~43% of articles, so its "rate" is partly a
# presence/absence indicator. Flagged here, caveated in the results.
for (f in FEATURES) {
  z <- mean(d[[f]] == 0, na.rm = TRUE)
  if (z > 0.10) cat(sprintf("NOTE %-24s %.0f%% exact zeros\n", f, 100 * z))
}


# --- 2. FIT ------------------------------------------------------------------
# feature ~ category                        each genre its own level
#         + s(year, by = category)          each genre its own trajectory
#         + s(year, newspaper, bs = "fs")   each newspaper its own trajectory
#         + s(log_tokens)                   article length, where a confound
#         + german_probability
#
# fit_gam() is in config.R. Fitted one at a time and cached: several bam
# objects on 3.6M rows will not co-exist in memory. DELETE data/models/*.rds
# after changing the specification or the window, or the loop will skip and you
# will diagnose stale models.

for (f in FEATURES) {
  path <- file.path(MODELS, paste0("gam_", f, ".rds"))
  if (file.exists(path)) { cat("skip (exists):", f, "\n"); next }
  cat("fitting:", f, "... ")
  t0 <- Sys.time()
  saveRDS(fit_gam(f, d), path)
  cat(sprintf("%.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  gc()
}


# ---- 3A. Model diagnostics on residuals. 
# the residual check is the one thing that would tell you whether Gamma(log) 
# was the right call for sentence length and MDD. Worth checking:

set.seed(1)
for (f in c("avg_sentlen", "avg_mdd")) {
  m <- load_model(f)
  res <- residuals(m, type = "deviance")
  i <- sample(length(res), 5e4)
  # plot res[i] against fitted(m)[i], look for a fan shape
  rm(m); gc()
}


# --- 3B. WHY A FACTOR SMOOTH FOR NEWSPAPER? -----------------------------------
# s(newspaper, bs = "re") gives each title a LEVEL; s(year, newspaper,
# bs = "fs") gives it a TRAJECTORY. Two diagnostics, because a slope cannot see
# shape: titles turn out to agree in DIRECTION on four of six indicators while
# the fs term still spends ~5 edf per title, i.e. they differ in SHAPE even
# where they agree in direction.

W_LO <- 1790; W_HI <- 1835   # window where several large titles overlap
#W_LO <- 1780; W_HI <- 1840  # window where several large titles overlap

MIN_YEARS <- 15              # years a title needs in the window to count

# for (w in list(c(1780, 1840), c(1790, 1835), c(1785, 1845))) {
#   cat("\n", w[1], "-", w[2], "\n")
#   # rerun the slopes block with W_LO <- w[1]; W_HI <- w[2]
# }

slopes <- map(FEATURES, \(f) {
  d |>
    filter(year >= W_LO, year <= W_HI) |>
    group_by(newspaper, year) |>
    summarise(m = mean(.data[[f]]), n = n(), .groups = "drop") |>
    filter(n >= 30) |>
    group_by(newspaper) |>
    filter(n() >= MIN_YEARS) |>
    summarise(slope_per_decade = coef(lm(m ~ year))[2] * 10, .groups = "drop") |>
    mutate(feature = f)
}) |> list_rbind()

write.csv(slopes, file.path(OUT, "within_title_slopes.csv"), row.names = FALSE)

cat("\nwithin-title slopes per decade,", W_LO, "-", W_HI,
    "\n(agree = share of titles on the majority side of zero)\n")
slopes |>
  group_by(feature) |>
  summarise(n_titles = n(),
            n_neg    = sum(slope_per_decade < 0),
            agree    = max(n_neg, n_titles - n_neg) / n_titles,
            median   = median(slope_per_decade), .groups = "drop") |>
  arrange(desc(agree)) |> as.data.frame() |> print(digits = 3)

p <- ggplot(slopes, aes(slope_per_decade, feature)) +
  geom_vline(xintercept = 0, colour = "firebrick", linewidth = .3) +
  geom_point(size = 2, alpha = .5, colour = "grey20") +
  labs(x = paste0("Within-title slope per decade, ", W_LO, "-", W_HI), y = NULL) +
  theme_minimal(base_size = 9)
ggsave(file.path(FIGS, "gam_within_title_slopes.pdf"), p, width = 8, height = 3.5)

p <- slopes |>
  group_by(feature) |>
  mutate(slope_std = slope_per_decade / sd(slope_per_decade),
         agree = max(sum(slope_per_decade < 0), sum(slope_per_decade > 0)) / n(),
         lab = sprintf("%s  (%.0f%%)", feature, 100 * agree)) |>
  ungroup() |>
  ggplot(aes(slope_std, reorder(lab, agree))) +
  geom_vline(xintercept = 0, colour = "firebrick", linewidth = .3) +
  geom_point(size = 2, alpha = .5, colour = "grey20") +
  labs(x = paste0("Within-title slope, ", W_LO, "-", W_HI,
                  " (SD units within feature)"), y = NULL) +
  theme_minimal(base_size = 9)
ggsave(file.path(FIGS, "gam_within_title_slopes.pdf"), p, width = 8, height = 3.5)


# --- 4. DIAGNOSTICS ----------------------------------------------------------
# k.check compares fitted residuals against what a larger basis would capture.
# Read the two columns together: a low k-index WITH edf near k' means the basis
# ran out of room. Here edf sits near k' whatever k we choose (see 06b), so the
# basis rather than the penalty sets the resolution -- k = 12 is a choice of
# timescale we make and state, and a borderline k-index follows from it: there
# is structure below the decadal scale and we are not modelling it.

edf_all <- map(FEATURES, \(f) {
  m <- load_model(f)
  cat("\n===", f, "===\n"); print(k.check(m))
  out <- gratia::edf(m) |> mutate(feature = f)
  rm(m); gc(); out
}) |> list_rbind()

write.csv(edf_all, file.path(OUT, "edf.csv"), row.names = FALSE)

cat("\nedf per year smooth (1 = a straight line, 11 = basis saturated):\n")
edf_all |>
  filter(grepl("^s\\(year\\):", .smooth)) |>
  mutate(genre = genre_of(.smooth)) |>
  select(feature, genre, .edf) |>
  pivot_wider(names_from = feature, values_from = .edf) |>
  as.data.frame() |> print(digits = 3)

# The second half of the specification argument: ~1 edf per title means the
# term is behaving like an intercept; ~5 means it is fitting title-specific
# curvature that a random intercept cannot hold, which would otherwise land in
# the year term.
edf_per_title <- edf_all |>
  filter(grepl("newspaper", .smooth)) |>
  transmute(feature, edf_per_title = .edf / nlevels(d$newspaper))

cat("\nedf per title on the newspaper term:\n")
print(as.data.frame(edf_per_title), digits = 3)
write.csv(edf_per_title, file.path(OUT, "edf_per_title.csv"), row.names = FALSE)


# --- 5. TRAJECTORIES ---------------------------------------------------------
# Partial effects of year, centred at zero within genre: these show CHANGE, not
# the level differences between genres, which live in the parametric category
# term. For sentence length and dependency distance the link is logarithmic, so
# the curve is proportional change.
#
# Each curve is divided by its own SD so the six panels share an axis. Without
# it, a feature moving by 0.005 in raw units and one moving by 0.5 need
# different scales, and free scales magnify the small ones until noise looks
# like signal. Raw rates are in the periods table (06b).

smooth_all <- map(FEATURES, \(f) {
  m <- load_model(f)
  out <- smooth_estimates(m, select = year_smooths(m), n = 200) |>
    add_confint() |>
    mutate(genre = factor(genre_of(.smooth), levels = CATS), feature = f)
  rm(m); gc(); out
}) |> list_rbind()

write.csv(smooth_all, file.path(OUT, "smooths.csv"), row.names = FALSE)

# Same ordering as the periods figure in 06b: one construct per row, the two
# markers of that construct side by side, stronger factor marker on the left.
# facet_wrap with ncol = 2 fills row-wise, so this order gives complexity /
# diversity / register down the page.
FEAT_NAME <- c(avg_sentlen            = "Avg. sentence length",
               avg_mdd                = "Avg. dependency distance",
               cttr_resid             = "CTTR",
               noun_ttr_resid         = "Noun TTR",
               nominal_verb_ratio     = "Nominal/verb ratio",
               personal_pronoun_ratio = "Personal pronoun ratio")

# Trajectories whose range is small relative to their own uncertainty are drawn
# dotted and grey: a band that accommodates a flat line does not support
# describing a shape. The threshold is arbitrary but stated, and the ratios go
# to the appendix so a reader can apply their own.
RATIO_CUT <- 1.0

traj <- smooth_all |>
  group_by(feature) |>
  mutate(across(c(.estimate, .lower_ci, .upper_ci), \(x) x / sd(.estimate))) |>
  group_by(feature, genre) |>
  mutate(ratio = (max(.estimate) - min(.estimate)) /
                 mean(.upper_ci - .lower_ci),
         determined = ratio >= RATIO_CUT) |>
  ungroup() |>
  mutate(panel = factor(FEAT_NAME[feature], levels = unname(FEAT_NAME)))

p <- ggplot(traj, aes(year, .estimate, group = genre)) +
  geom_hline(yintercept = 0, colour = "grey80", linewidth = .3) +
  geom_ribbon(data = filter(traj, determined),
              aes(ymin = .lower_ci, ymax = .upper_ci, fill = genre),
              alpha = .12, colour = NA) +
  geom_line(data = filter(traj, !determined),
            colour = "grey65", linetype = "dotted", linewidth = .5) +
  geom_line(data = filter(traj, determined),
            aes(colour = genre), linewidth = .8) +
  facet_wrap(~ panel, ncol = 2, dir = "h") +
  scale_colour_manual(values = PAL) + scale_fill_manual(values = PAL) +
  scale_x_continuous(breaks = seq(1750, 1840, 30)) +
  guides(fill = "none") +
  labs(y = "Standardised partial effect", x = NULL, colour = NULL) +
  theme_minimal(base_size = 8) +
  theme(legend.position  = "top",
        panel.grid.minor = element_blank(),
        panel.spacing    = unit(0.8, "lines"),
        strip.text       = element_text(size = 8, hjust = 0))

ggsave(file.path(FIGS, "gam_trajectories.pdf"), p, width = 7, height = 6)

ratios <- traj |>
  group_by(panel, genre) |>
  summarise(range = max(.estimate) - min(.estimate),
            band  = mean(.upper_ci - .lower_ci),
            ratio = range / band, .groups = "drop") |>
  arrange(panel, desc(ratio))

cat("\ntrajectory range relative to mean band width",
    sprintf("(dotted below %.1f):\n", RATIO_CUT))
print(as.data.frame(ratios), digits = 2)
write.csv(ratios, file.path(OUT, "trajectory_precision.csv"), row.names = FALSE)
write_tex(ratios, "trajectory_precision.tex",
          paste("Range of each fitted trajectory relative to the mean width of",
                "its own confidence band. Below 1 the band accommodates a flat",
                "line and the shape should not be described."),
          "tab:trajectory_precision")


# --- 6. IDENTIFICATION -------------------------------------------------------
# The newspaper term absorbs title-specific trajectories. Where the corpus
# holds one or two titles, that is indistinguishable from absorbing the corpus
# trend: "the 1750s changed" and "this paper changed" are the same statement.
#
# Fitting the same model with newspaper INTERCEPTS instead makes the ambiguity
# visible. The two specifications resolve it in opposite directions -- the
# factor smooth toward the title, the intercept toward the year -- so the gap
# between them measures how much the early estimates depend on that choice.

m_ri <- bam(avg_sentlen ~ category + s(year, by = category, k = K_YEAR) +
              s(newspaper, bs = "re") + german_probability,
            data = d, family = Gamma(link = "log"), gamma = GAMMA_PENALTY,
            method = "fREML", discrete = TRUE, nthreads = N_THREADS)

nd <- expand.grid(year = MIN_YEAR:MAX_YEAR,
                  category = factor(CATS, levels = CATS))
nd$newspaper          <- levels(d$newspaper)[1]
nd$log_tokens         <- mean(d$log_tokens)
nd$german_probability <- mean(d$german_probability)

m_fs <- load_model("avg_sentlen")
nd$fs <- predict(m_fs, nd, type = "response", exclude = "s(year,newspaper)")
nd$ri <- predict(m_ri, nd, type = "response", exclude = "s(newspaper)")
rm(m_fs, m_ri); gc()

raw <- d |>
  group_by(year, category) |>
  summarise(m = mean(avg_sentlen), n = n(), .groups = "drop") |>
  filter(n >= 50)

p <- ggplot(nd, aes(year, colour = category)) +
  geom_line(data = raw, aes(year, m), linewidth = .3, alpha = .4) +
  geom_line(aes(y = fs), linewidth = .8) +
  geom_line(aes(y = ri), linewidth = .8, linetype = "22") +
  scale_colour_manual(values = PAL) +
  labs(y = "Fitted mean sentence length (tokens)", x = NULL, colour = NULL) +
  theme_minimal(base_size = 9) + theme(legend.position = "top")

ggsave(file.path(FIGS, "gam_identification_check.pdf"), p, width = 7, height = 5)

ident <- nd |>
  group_by(year) |>
  summarise(maxdiff = max(abs(fs - ri)),
            reldiff = max(abs(fs - ri) / ri), .groups = "drop") |>
  left_join(title_counts, by = "year")
write.csv(ident, file.path(OUT, "identification.csv"), row.names = FALSE)

cat("\nspecification divergence:\n")
ident |>
  filter(year %in% c(1740, 1760, 1780, 1800, 1820, 1840)) |>
  as.data.frame() |> print(digits = 3)
cat("max reldiff from 1780:",
    round(max(ident$reldiff[ident$year >= 1780]), 3), "\n")

# The same story from the intervals: how wide is the band early vs late?
cat("\nmean band width on the sentence-length smooth, by era:\n")
smooth_all |>
  filter(feature == "avg_sentlen") |>
  mutate(width = .upper_ci - .lower_ci,
         era = cut(year, c(-Inf, 1760, 1780, 1820, Inf))) |>
  group_by(era) |>
  summarise(mean_width = mean(width), .groups = "drop") |>
  as.data.frame() |> print(digits = 3)

cat("\ndone. tables in", OUT, "| figures in", FIGS, "\n")

