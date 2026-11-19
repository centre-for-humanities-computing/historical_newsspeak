# 06_gam.R -- GAMMs per indicator and category, fitted to newspaper x year x
# category means (articles in one paper-year aren't independent).
# Run after 05_export_for_GAM.py. Writes data/gam_cells/, data/models_cells/
# and figs/gamcell_*.pdf. Delete data/models_cells/*.rds after changing the spec.

library(here)
source(here("src", "config.R"))
library(purrr)

FEATURES <- REPRESENTATIVE
K_CELL     <- 20     # just a ceiling, REML picks the smoothness
MIN_N      <- 20     # articles per mean; above this they're about equally precise,
                     # so equal weights (see variance_check.csv)
MIN_PERIOD <- 2      # a period has to span more than a year
BASE_SIZE  <- 11

# specs: main = with newspaper trajectories, no_fs = without ("press as
# printed"), k30 = main with a bigger basis. Markers keep the dotted lines readable.
SPEC_LAB   <- c(main  = "with newspaper trajectories",
                no_fs = "without (press as printed)",
                k30   = "k = 30")
SPEC_LTY   <- c(main = "solid", no_fs = "11", k30 = "42")
SPEC_SHAPE <- c(main = NA, no_fs = 1, k30 = 2)
SPEC_MARK  <- c(no_fs = 0, k30 = 5)  # marker years: year %% 10 == this

OUT  <- file.path(DATA, "gam_cells");    dir.create(OUT,  showWarnings = FALSE)
MODS <- file.path(DATA, "models_cells"); dir.create(MODS, showWarnings = FALSE)

FEAT_NAME <- c(avg_sentlen = "Sentence length (tokens)",
               avg_mdd = "Dependency distance (tokens)",
               cttr_resid = "CTTR (residualised)",
               noun_ttr_resid = "Noun TTR (residualised)",
               nominal_verb_ratio = "Nominal/verb ratio",
               personal_pronoun_ratio = "Personal pronoun ratio")


# --- 1. means per newspaper x year x category ---------------------------------

d <- load_gam_data()    # 1740-1848, see config.R

cells <- d |>
  group_by(newspaper, year, category) |>
  summarise(n = n(),
            # variances first, the mean across() overwrites the names
            across(all_of(FEATURES), \(x) var(x, na.rm = TRUE),
                   .names = "{.col}__s2"),
            across(all_of(FEATURES), \(x) mean(x, na.rm = TRUE)),
            german_probability = mean(german_probability),
            log_tokens = mean(log_tokens),
            .groups = "drop") |>
  filter(n >= MIN_N) |>
  mutate(np_year  = factor(paste(newspaper, year, sep = "_")),
         category = factor(category, levels = CATS))

cat(nrow(d), "articles ->", nrow(cells), "cells,",
    nlevels(cells$np_year), "newspaper-years,",
    n_distinct(cells$newspaper), "titles,", min(cells$year), "-", max(cells$year), "\n")
print(count(cells, category), n = Inf)
write.csv(select(cells, -ends_with("__s2")), file.path(OUT, "cells.csv"),
          row.names = FALSE)


# --- 2. models ----------------------------------------------------------------
# no newspaper-year random intercept: tried it, variance came out at zero

fit_cell <- function(f, k = K_CELL, traj = TRUE, data = cells,
                     family = FAMILIES[[f]] %||% gaussian()) {
  terms <- c("category",
             sprintf("s(year, by = category, k = %d)", k),
             if (traj) "s(year, newspaper, bs = 'fs', m = 1, k = 10)",
             "german_probability")
  if (!f %in% NO_LENGTH) terms <- c(terms, "s(log_tokens, k = 10)")
  bam(as.formula(paste(f, "~", paste(terms, collapse = " + "))),
      data = data, family = family,
      method = "fREML", nthreads = N_THREADS)
}

specs <- tribble(~spec,   ~k,     ~traj,
                 "main",  K_CELL, TRUE,
                 "no_fs", K_CELL, FALSE,
                 "k30",   30,     TRUE)
spec_names <- specs$spec

models <- list()
for (f in FEATURES) for (i in seq_len(nrow(specs))) {
  s <- specs[i, ]
  path <- file.path(MODS, sprintf("cell_%s_%s.rds", f, s$spec))
  if (!file.exists(path)) {
    cat("fitting", f, s$spec, "... "); t0 <- Sys.time()
    saveRDS(fit_cell(f, s$k, s$traj), path)
    cat(sprintf("%.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }
  models[[paste(f, s$spec)]] <- readRDS(path)
}


# --- 3. edf vs ceiling (should sit well below k - 1) ---------------------------

edf_tab <- imap(models, \(m, nm) {
  gratia::edf(m) |>
    filter(grepl("^s\\(year\\):", .smooth)) |>
    transmute(feature = sub(" .*", "", nm), spec = sub("^\\S+ ", "", nm),
              genre = genre_of(.smooth), edf = .edf,
              ceiling = m$smooth[[1]]$bs.dim - 1)
}) |> list_rbind()
write.csv(edf_tab, file.path(OUT, "edf.csv"), row.names = FALSE)

cat("\nedf of the year smooths (of ceiling):\n")
edf_tab |>
  mutate(v = sprintf("%.1f/%d", edf, ceiling)) |>
  select(feature, genre, spec, v) |>
  pivot_wider(names_from = spec, values_from = v) |>
  as.data.frame() |> print()


# --- 3b. diagnostics: k.check, sampling vs between-mean variance, residuals ----
# (Gamma vs Gaussian check is at the end, it needs grid_for())

kc <- map(FEATURES, \(f) {
  k <- k.check(models[[paste(f, "main")]])
  as_tibble(k, rownames = "term") |> mutate(feature = f)
}) |> list_rbind() |> filter(grepl("^s\\(year\\):", term))
write.csv(kc, file.path(OUT, "kcheck.csv"), row.names = FALSE)
cat("\nk.check, year smooths of the main fits (lowest k-index first):\n")
kc |> arrange(`k-index`) |> head(8) |> as.data.frame() |> print(digits = 3)

vchk <- map(FEATURES, \(f) {
  m <- models[[paste(f, "main")]]
  samp <- median(cells[[paste0(f, "__s2")]] / cells$n, na.rm = TRUE)
  tot  <- var(residuals(m, type = "response"))
  tibble(feature = f, resid_var = tot, sampling_var = samp,
         n_star = median(cells[[paste0(f, "__s2")]], na.rm = TRUE) / (tot - samp))
}) |> list_rbind()
write.csv(vchk, file.path(OUT, "variance_check.csv"), row.names = FALSE)
cat("\nn* (article count where sampling variance = between-mean variance):\n")
print(as.data.frame(vchk), digits = 3)

res <- map(FEATURES, \(f) {
  m <- models[[paste(f, "main")]]
  tibble(feature = f, fitted = fitted(m), res = residuals(m, type = "deviance"))
}) |> list_rbind() |>
  mutate(panel = factor(FEAT_NAME[feature], levels = FEAT_NAME))
p <- ggplot(res, aes(fitted, res)) +
  geom_point(size = .6, alpha = .3) +
  geom_hline(yintercept = 0, colour = "firebrick", linewidth = .3) +
  geom_smooth(method = "loess", se = FALSE, colour = "steelblue", linewidth = .6) +
  facet_wrap(~ panel, ncol = 2, scales = "free") +
  labs(x = "Fitted", y = "Deviance residual") +
  theme_minimal(base_size = BASE_SIZE)
ggsave(file.path(FIGS, "gamcell_residuals.pdf"), p, width = 8.5, height = 9)

# --- 3c. do titles move together? per title and category, 1790-1835 -----------
# plain OLS slope on the yearly means, no model involved. Disagreeing slopes
# (or same sign, different shape) are why each title gets its own curve.
W_SLOPE <- c(1790, 1835); MIN_YEARS <- 15
slopes <- map(FEATURES, \(f) {
  cells |>
    filter(year >= W_SLOPE[1], year <= W_SLOPE[2]) |>
    group_by(category, newspaper) |>
    filter(n() >= MIN_YEARS) |>
    summarise(slope = 10 * coef(lm(.data[[f]] ~ year))[2], .groups = "drop") |>
    mutate(feature = f)
}) |> list_rbind()
agree_tab <- slopes |>
  group_by(feature, category) |>
  summarise(titles = n(), falling = sum(slope < 0), rising = sum(slope > 0),
            agree = max(falling, rising) / titles, .groups = "drop")
write.csv(slopes, file.path(OUT, "within_title_slopes.csv"), row.names = FALSE)
write.csv(agree_tab, file.path(OUT, "within_title_agreement.csv"), row.names = FALSE)
cat("\nwithin-title slopes per decade,", W_SLOPE, "(agree = share on majority side):\n")
print(as.data.frame(agree_tab), digits = 2)

# diverging bars: titles falling (left) vs rising (right)
p <- agree_tab |>
  mutate(panel = factor(FEAT_NAME[feature], levels = FEAT_NAME),
         category = factor(sub("^fiction$", "Fiction", category),
                           levels = rev(c("Advertisement", "National",
                                          "International", "Fiction")))) |>
  pivot_longer(c(falling, rising), names_to = "dir", values_to = "k") |>
  mutate(x = ifelse(dir == "falling", -k, k)) |>
  ggplot(aes(x, category, fill = dir)) +
  geom_col(width = .7) +
  geom_vline(xintercept = 0, colour = "grey30", linewidth = .4) +
  facet_wrap(~ panel, ncol = 2) +
  scale_fill_manual(values = c(falling = "#d95f02", rising = "#1f78b4"),
                    labels = c(falling = "titles falling", rising = "titles rising")) +
  scale_x_continuous(labels = abs, breaks = seq(-12, 12, 4)) +
  labs(x = NULL, y = NULL, fill = NULL) +
  theme_minimal(base_size = 20) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank())
ggsave(file.path(FIGS, "gamcell_within_title_slopes.pdf"), p, width = 11, height = 10)


# --- 4. trajectories in the features' own units --------------------------------
# category level + year smooth, newspaper term dropped, covariates at the
# category mean. Simultaneous bands from posterior draws.

EXCL  <- "s(year,newspaper)"
YEARS <- seq(min(cells$year), max(cells$year), by = 0.5)

grid_for <- function() {
  covs <- cells |> group_by(category) |>
    summarise(german_probability = mean(german_probability),
              log_tokens = mean(log_tokens), .groups = "drop")
  expand_grid(category = factor(CATS, levels = CATS), year = YEARS) |>
    left_join(covs, by = "category") |>
    mutate(newspaper = cells$newspaper[1])
}

lp_for <- function(m, nd) {
  excl <- intersect(EXCL, sapply(m$smooth, \(s) s$label))
  predict(m, nd, type = "lpmatrix", exclude = excl)
}

draws_for <- function(m, nd, nsim = 2000) {
  X <- lp_for(m, nd)
  set.seed(SEED)
  B <- mgcv::rmvn(nsim, coef(m), vcov(m, unconditional = TRUE))
  X %*% t(B)          # grid points x draws, link scale
}

traj_one <- function(f, spec) {
  m <- models[[paste(f, spec)]]; nd <- grid_for()
  inv <- m$family$linkinv
  X <- lp_for(m, nd)
  fit <- drop(X %*% coef(m))
  se  <- sqrt(rowSums((X %*% vcov(m, unconditional = TRUE)) * X))
  sims <- draws_for(m, nd)
  crit <- tapply(seq_len(nrow(nd)), nd$category, \(ix) {
    t <- abs(sweep(sims[ix, ], 1, fit[ix])) / se[ix]
    quantile(apply(t, 2, max), .95)
  })
  nd |> mutate(feature = f, spec = spec,
               est = inv(fit),
               lo = inv(fit - 1.96 * se), hi = inv(fit + 1.96 * se),
               lo_sim = inv(fit - crit[as.character(category)] * se),
               hi_sim = inv(fit + crit[as.character(category)] * se))
}

traj <- map(FEATURES, \(f) map(spec_names, \(s) traj_one(f, s)) |> list_rbind()) |>
  list_rbind()
write.csv(traj, file.path(OUT, "trajectories.csv"), row.names = FALSE)

# change between window averages (same windows as the classifier drift),
# per posterior draw, for main and no_fs
W0 <- c(1740, 1769); W1 <- c(1820, 1849)
change_tab <- function(spec = "main", w0 = W0, w1 = W1) {
  map(FEATURES, \(f) {
    m <- models[[paste(f, spec)]]
    nd <- grid_for() |> filter(year == round(year))   # whole years
    sims <- m$family$linkinv(draws_for(m, nd))
    win <- map(set_names(CATS), \(g) {
      i0 <- which(nd$category == g & nd$year >= w0[1] & nd$year <= w0[2])
      i1 <- which(nd$category == g & nd$year >= w1[1] & nd$year <= w1[2])
      list(a0 = colMeans(sims[i0, , drop = FALSE]),
           a1 = colMeans(sims[i1, , drop = FALSE]))
    })
    # equal weights, so Advertisement's article count doesn't dominate
    win$Average <- list(a0 = Reduce(`+`, map(win[CATS], "a0")) / length(CATS),
                        a1 = Reduce(`+`, map(win[CATS], "a1")) / length(CATS))
    ED <- setdiff(CATS, "Advertisement")
    win$Editorial <- list(a0 = Reduce(`+`, map(win[ED], "a0")) / length(ED),
                          a1 = Reduce(`+`, map(win[ED], "a1")) / length(ED))
    imap(win, \(w, g) {
      dlt <- w$a1 - w$a0
      tibble(feature = f, genre = g, spec = spec,
             from = paste(w0, collapse = "-"), to = paste(w1, collapse = "-"),
             level_from = mean(w$a0), level_to = mean(w$a1),
             change = mean(dlt), lo = quantile(dlt, .025), hi = quantile(dlt, .975))
    }) |> list_rbind()
  }) |> list_rbind()
}
chg <- bind_rows(change_tab("main"), change_tab("no_fs"))
write.csv(chg, file.path(OUT, "change.csv"), row.names = FALSE)

# LaTeX table: bold = interval excludes zero, dagger = no_fs says the opposite
TAB_DIGITS <- c(avg_sentlen = 1, avg_mdd = 2, cttr_resid = 2,
                noun_ttr_resid = 3, nominal_verb_ratio = 3,
                personal_pronoun_ratio = 3)
TAB_NAME <- c(avg_sentlen = "Sentence length",
              avg_mdd = "Dependency distance",
              cttr_resid = "\\textsc{cttr} (resid.)",
              noun_ttr_resid = "Noun \\textsc{ttr} (resid.)",
              nominal_verb_ratio = "Nominal/verb ratio",
              personal_pronoun_ratio = "Pronoun ratio")
TAB_CATS <- c(Advertisement = "Advertisement", National = "National",
              International = "International", fiction = "Fiction",
              Average = "Average (without Adv.)")
fmt <- function(x, k) {
  out <- formatC(x, format = "f", digits = k)
  out <- sub("^-(0\\.0+)$", "\\1", out)     # no "-0.000"
  sub("^-", "$-$", out)
}

cell_txt <- function(f, g) {
  k  <- TAB_DIGITS[[f]]
  x  <- chg |> filter(feature == f, spec == "main", genre == g)
  x2 <- chg |> filter(feature == f, spec == "no_fs", genre == g)
  sig <- sign(x$lo) == sign(x$hi); sig2 <- sign(x2$lo) == sign(x2$hi)
  est <- fmt(x$change, k)
  if (sig) est <- paste0("\\textbf{", est, "}")
  if (sig && sig2 && sign(x2$change) != sign(x$change))
    est <- paste0(est, "$^\\dagger$")
  est
}
wide_rows <- map_chr(FEATURES, \(f)
  paste0(TAB_NAME[[f]], " & ",
         paste(map_chr(setdiff(names(TAB_CATS), "Average"), \(g) cell_txt(f, g)),
               collapse = " & "),
         " & ", cell_txt(f, "Average"), " (", cell_txt(f, "Editorial"), ")",
         " \\\\"))
tex_wide <- c(
  "% generated by notebooks/06_gam.R -- do not edit by hand",
  "% needs \\usepackage{booktabs}",
  "\\begin{table}[t]",
  "\\centering",
  "\\caption{Change in each indicator's average fitted level between 1740--69 and 1820--49 (the windows of \\autoref{sec:coherence}), in the indicator's own units (tokens for sentence length and dependency distance), from the main model. Average: the mean change of the four categories, each weighted equally regardless of its number of articles; in parentheses, the same for the three editorial categories, without Advertisement. Bold: the 95\\% interval excludes zero. $^\\dagger$: the specification without newspaper trajectories finds a clear change in the opposite direction. A change can be clear here without any dated period in \\autoref{fig:gam_periods} if it was gradual.}",
  "\\label{tab:gam_change}",
  "\\small",
  "\\begin{tabular}{lrrrr@{\\hspace{1.5em}}r}",
  "\\toprule",
  paste0(" & ", paste(TAB_CATS, collapse = " & "), " \\\\"),
  "\\midrule",
  wide_rows,
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}")
writeLines(tex_wide, file.path(OUT, "gam_change.tex"))
cat("\nchange in natural units (main model, 95% interval):\n")
chg |> mutate(across(where(is.double), \(x) signif(x, 3))) |>
  as.data.frame() |> print()

# colour = category, line type + marker = spec, band only for main
spec_plot <- function(which_specs) {
  x <- traj |>
    filter(spec %in% which_specs) |>
    mutate(panel = factor(FEAT_NAME[feature], levels = FEAT_NAME),
           spec  = factor(spec, levels = which_specs))
  marks <- x |> filter(spec != "main",
                       year %% 10 == SPEC_MARK[as.character(spec)])
  ggplot(x, aes(year, est, colour = category)) +
    geom_ribbon(data = filter(x, spec == "main"),
                aes(ymin = lo_sim, ymax = hi_sim, fill = category),
                alpha = .13, colour = NA) +
    geom_line(aes(linetype = spec), linewidth = .8) +
    geom_point(data = marks, aes(shape = spec), size = 2.2, stroke = .9,
               fill = "white") +
    facet_wrap(~ panel, ncol = 2, scales = "free_y") +
    scale_colour_manual(values = PAL) + scale_fill_manual(values = PAL) +
    scale_linetype_manual(values = SPEC_LTY[which_specs],
                          labels = SPEC_LAB[which_specs], drop = FALSE) +
    scale_shape_manual(values = SPEC_SHAPE[which_specs],
                       labels = SPEC_LAB[which_specs], drop = FALSE) +
    scale_x_continuous(breaks = seq(1740, 1840, 20)) +
    guides(fill = "none",
           colour = guide_legend(order = 1),
           linetype = guide_legend(order = 2, override.aes = list(linewidth = .8)),
           shape = guide_legend(order = 2)) +
    labs(x = NULL, y = NULL, colour = NULL, linetype = NULL, shape = NULL) +
    theme_minimal(base_size = BASE_SIZE) +
    theme(legend.position = "top", legend.box = "vertical",
          legend.key.width = unit(2.2, "lines"),
          panel.grid.minor = element_blank())
}

ggsave(file.path(FIGS, "gamcell_trajectories.pdf"),
       spec_plot(c("main", "no_fs")), width = 8.5, height = 10)
ggsave(file.path(FIGS, "gamcell_sensitivity.pdf"),
       spec_plot(c("main", "no_fs", "k30")), width = 8.5, height = 10)

# how far each spec sits from main, in main band half-widths
cat("\nmax |spec - main| / main simultaneous half-width, before 1760 and from:\n")
traj |>
  select(feature, category, year, spec, est) |>
  pivot_wider(names_from = spec, values_from = est) |>
  left_join(traj |> filter(spec == "main") |>
              transmute(feature, category, year, hw = (hi_sim - lo_sim) / 2),
            by = c("feature", "category", "year")) |>
  pivot_longer(setdiff(spec_names, "main"), names_to = "spec") |>
  mutate(era = ifelse(year < 1760, "pre1760", "from1760")) |>
  group_by(feature, spec, era) |>
  summarise(rel = max(abs(value - main) / hw), .groups = "drop") |>
  pivot_wider(names_from = c(spec, era), values_from = rel) |>
  as.data.frame() |> print(digits = 2)


# --- 5. periods of change ----------------------------------------------------
# slope interval excludes zero; reported = main, minus years where no_fs
# finds the opposite direction

states_one <- function(f, spec) {
  m <- models[[paste(f, spec)]]
  derivatives(m, select = year_smooths(m), type = "central",
              interval = "simultaneous", n = 400, n_sim = 5000, seed = SEED) |>
    mutate(genre = genre_of(.smooth),
           state = case_when(.lower_ci > 0 ~ 1L, .upper_ci < 0 ~ -1L, TRUE ~ 0L)) |>
    transmute(feature = f, spec, genre, year, state)
}
states <- map(FEATURES, \(f) map(spec_names, \(s) states_one(f, s)) |> list_rbind()) |>
  list_rbind()
write.csv(states, file.path(OUT, "derivative_states.csv"), row.names = FALSE)

# same derivative grid for every spec, so rows line up
wide <- states |>
  group_by(feature, spec, genre) |> mutate(i = row_number()) |> ungroup() |>
  pivot_wider(id_cols = c(feature, genre, i), names_from = spec,
              values_from = c(state, year)) |>
  transmute(feature, genre, year = year_main,
            main = state_main, no_fs = state_no_fs, k30 = state_k30,
            contradicted = main != 0 & no_fs == -main,
            reported  = ifelse(main != 0 & !contradicted, main, 0L))

runs <- function(s, col) {
  if (all(s[[col]] == 0)) return(tibble())
  s |> arrange(year) |>
    mutate(st = .data[[col]], run = cumsum(st != lag(st, default = 0L))) |>
    filter(st != 0) |>
    group_by(run, st) |>
    # length on unrounded years, same as the figure
    summarise(span = max(year) - min(year),
              from_yr = round(min(year)), to_yr = round(max(year)),
              k30_agree = mean(k30 == st), .groups = "drop")
}

periods_for <- function(col) {
  wide |> group_by(feature, genre) |>
    group_modify(\(s, ...) runs(s, col)) |> ungroup() |>
    filter(!is.na(st)) |>
    filter(span >= MIN_PERIOD) |>
    transmute(feature, genre, direction = ifelse(st > 0, "increase", "decrease"),
              from_yr, to_yr, k30_agree)
}
periods      <- periods_for("reported")
periods_main <- periods_for("main")

write.csv(periods, file.path(OUT, "periods.csv"), row.names = FALSE)
write.csv(periods_main, file.path(OUT, "periods_main_only.csv"), row.names = FALSE)
contra <- wide |> filter(contradicted) |> group_by(feature, genre) |>
  summarise(from_yr = round(min(year)), to_yr = round(max(year)), .groups = "drop")
write.csv(contra, file.path(OUT, "contradicted.csv"), row.names = FALSE)
cat("\nyears where no_fs contradicts main (omitted):\n"); print(as.data.frame(contra))
cat("\nREPORTED periods (> ", MIN_PERIOD - 1, "yr):\n")
print(as.data.frame(periods), digits = 2)
cat(nrow(periods), "reported;", nrow(periods_main), "in main alone\n")

# one rectangle per run (no hairline seams); solid = no_fs agrees, pale = main only
step <- diff(sort(unique(wide$year)))[1]
segs <- wide |>
  group_by(feature, genre) |> arrange(year, .by_group = TRUE) |>
  mutate(period = cumsum(reported != lag(reported, default = 99L))) |>
  filter(reported != 0) |>
  group_by(feature, genre, period) |>
  filter(max(year) - min(year) >= MIN_PERIOD) |>
  mutate(confirmed = no_fs == reported,
         piece = cumsum(confirmed != lag(confirmed, default = !first(confirmed)))) |>
  group_by(feature, genre, period, piece, reported, confirmed) |>
  summarise(xmin = min(year) - step / 2, xmax = max(year) + step / 2,
            .groups = "drop") |>
  mutate(panel = factor(FEAT_NAME[feature], levels = FEAT_NAME),
         genre = factor(genre, levels = rev(CATS)),
         y = as.integer(genre),
         state = factor(reported, levels = c(-1, 1),
                        labels = c("decrease", "increase")))
rows <- expand_grid(panel = factor(FEAT_NAME, levels = FEAT_NAME),
                    genre = factor(CATS, levels = rev(CATS))) |>
  mutate(y = as.integer(genre))

p <- ggplot() +
  geom_rect(data = rows, aes(ymin = y - .4, ymax = y + .4),
            xmin = min(wide$year), xmax = max(wide$year), fill = "grey94") +
  geom_rect(data = segs, aes(xmin = xmin, xmax = xmax, ymin = y - .4,
                             ymax = y + .4, fill = state, alpha = confirmed)) +
  scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = .35), guide = "none") +
  facet_wrap(~ panel, ncol = 2) +
  scale_fill_manual(values = c(decrease = "#d95f02", increase = "#1f78b4")) +
  scale_y_continuous(breaks = seq_along(CATS), labels = rev(CATS)) +
  scale_x_continuous(breaks = seq(1740, 1840, 20)) +
  labs(x = NULL, y = NULL, fill = NULL) +
  theme_minimal(base_size = BASE_SIZE) +
  theme(legend.position = "top", panel.grid = element_blank())
ggsave(file.path(FIGS, "gamcell_periods.pdf"), p, width = 8.5, height = 6)

# --- 6. Gamma vs Gaussian for sentence length / MDD ---------------------------

fam_cmp <- map(names(FAMILIES), \(f) {
  path <- file.path(MODS, sprintf("cell_%s_gaussian.rds", f))
  if (!file.exists(path)) saveRDS(fit_cell(f, family = gaussian()), path)
  g <- readRDS(path); m <- models[[paste(f, "main")]]
  nd <- grid_for()
  excl <- intersect(EXCL, sapply(m$smooth, \(s) s$label))
  pm <- m$family$linkinv(predict(m, nd, exclude = excl))
  pg <- predict(g, nd, exclude = excl)
  tibble(feature = f, aic_gamma = AIC(m), aic_gaussian = AIC(g),
         max_traj_diff = max(abs(pm - pg)), mean_level = mean(pm))
}) |> list_rbind()
write.csv(fam_cmp, file.path(OUT, "family_check.csv"), row.names = FALSE)
cat("\nGamma(log) vs Gaussian on the means (lower AIC is better):\n")
print(as.data.frame(fam_cmp), digits = 4)

cat("\ndone. tables in", OUT, "| figures figs/gamcell_*.pdf\n")
