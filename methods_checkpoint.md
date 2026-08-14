# Methods notes

Why the analysis looks the way it does. Short version of decisions made along the way, mostly so we don't relitigate them.

## Features

**No single "complexity" dimension.** An early PCA needed 11 of 17 components for 90% of the variance. That's a result, not a problem: the features are largely non-redundant, so we report them separately rather than collapsing them into an index.

**Factor analysis, not PCA.** PCA forces orthogonal components and uses every variable in each, so nothing pushes toward interpretable groupings — early PC1–3 cross-loadings were a mess. FA with promax rotation lets factors correlate (register and complexity plausibly do) and optimises for simple structure.

**The theory-driven complexity/register split was half wrong.** `function_word_ratio` and `nominal_verb_ratio` empirically sit in the other group. We stopped defending the a priori split and let the FA decide.

**LIX and RIX dropped from the FA.** Both contain `avg_sentlen` as an additive term, so keeping them alongside it double-counts the same information. Dropping just one shifted the high VIF onto the other; dropping both took `avg_sentlen` from VIF ~15 to 2.92. `avg_sentlen`, `avg_wordlen` and `avg_mdd` carry the readability signal anyway.

**MDD and NDD are not redundant.** VIFs all under 3 — the log transform in NDD's formula decorrelates it enough.

**TTRs residualised against length.** CTTR correlated 0.94 with token count; it was measuring article length via a ceiling effect on short texts. `cttr`, `noun_ttr`, `verb_ttr` are OLS residuals against `log(tokens)` (`_resid` suffix); length was 88.9% / 25.9% / 18.7% of their raw variance. LIX/RIX/MDD's correlation with sentence length is definitional, not confounding, so it stays.

**Article length is a covariate, not residualised out**, where it may reflect a real genre effect rather than noise.

**`compressrat` and `mtld` excluded.** Both are NaN below a length threshold (99.4% and ~30% missing). With `dropna()` this quietly cut the analysis sample to 18,311 rows — 0.5% of the corpus, and badly length-biased. Excluding them restores ~3.3M.

**German-detection features are a covariate.** They load cleanly (.97–1.00) on their own factor, but that's a language-ID signal, not a stylistic construct. Removing them moves total variance explained by <0.1pp.

**`of_ratio` has low communality and that's expected.** Even above 100 tokens, 13.3% of articles score exactly zero — sampling noise on a low-base-rate word, not a missing dimension. More factors won't fix it.

**Bug fix: `nominal_verb_ratio`.** Originally omitted NOUN from the nominal set and used an unbounded count ratio, which is unstable in verb-sparse short articles. Now includes NOUN and is bounded 0–1 as `nominals/(nominals+verbs)`. Communality went from ~0.10 to 0.63 and it loads where nominalisation-as-informational-register theory says it should.

## Sentiment

**Standardised globally, not per shard.** `SemanticProjector.standardize()` z-scores against whatever batch it gets, so per-shard standardisation would score the same absolute value differently depending on shard composition. We pool sum/sumsq/n across all shards and z-score once.

**Variability as a log ratio.** Variance is ratio-scale, not additive like the mean, so within-article variability is `log(article variance / global pooled variance)` — symmetric around 0. Gated by `semantic_sentiment_std_defined` (False at n=1, where variance is trivially zero). It doesn't correct for small-n noise in the variance estimate itself, so treat short articles carefully.

Use `semantic_sentiment_standardized`, `semantic_sentiment_log_var_ratio`, `semantic_sentiment_std_defined`. The rest (`sum`, `sumsq`, `n`, `mean_raw`, `var_raw`, `std_raw`) is intermediate.

## Three factors

A sweep from 2 to 8 gives proper solutions up to 5, but F4/F5 fragment `avg_sentlen` across multiple factors rather than surfacing a new construct, and at k=5 its uniqueness sits at 0.0165, right on the instability boundary. F1–F3 are essentially identical at k=3, 4 and 5 — the strongest validity evidence we have.

- **F1 — Involved vs informational register.** `nominal_verb_ratio` (.71–.80), `personal_pronoun_ratio` (−.62 to −.70), `passive_ratio` (~.5), `present_tense_ratio` (~.45), `that_ratio` (~−.40), `function_word_ratio` (~−.3). Sentiment sits here too: mean ~.3–.35, log-var-ratio ~−.32 to −.39. Matches Biber's Dimension 1.
- **F2 — Syntactic complexity.** `avg_sentlen` (.79–.89), `avg_mdd` (.86), `std_mdd` (.42–.53).
- **F3 — Lexical diversity.** `cttr_resid` (.81–.85), `noun_ttr_resid` (.63–.72), `verb_ttr_resid` (.40–.42).

Loadings have been stable across every variant tested — with and without the German features, before and after the `nominal_verb_ratio` fix, with and without RIX. That cross-run stability is better evidence than the fit diagnostics, which are middling (KMO ~0.62).

## The structure isn't the same in every genre

Tucker's congruence, genre-specific fits against a pooled model:

- **Clean**: National (0.96/0.92/0.83), Advertisement (0.99/0.95/0.93).
- **International**: two factors map (0.94, 0.88), the third matches nothing (max 0.42, spread evenly across all three). Not a sample-size artefact at n=596K.
- **Fiction**: one factor maps (0.91); the other two both collide onto pooled F1 and neither matches F3. Diversity and register aren't separable in fiction the way the pooled model assumes.

**So we don't use pooled factor scores.** A score computed with pooled weights wouldn't measure the same thing in each genre, which is exactly the comparison the paper rests on. The invariance breakdown is reported as a finding in its own right — style's internal organisation is domain-specific, not just its level or trend — and the trend analysis models six representative features directly:

- Complexity: `avg_sentlen`, `avg_mdd`
- Diversity: `cttr_resid`, `noun_ttr_resid`
- Register: `nominal_verb_ratio`, `personal_pronoun_ratio`

## Trends over time: what we tried and why we stopped

Three things we ran, in order, each because the previous one couldn't answer the question.

**Correlation with year, per genre.** Kept, as a first pass. It assumes no functional form beyond monotonicity and no model of the corpus structure, so it shows where temporal signal sits before any modelling choice can shape the answer, and it runs on all features rather than the six we go on to model. Reported as ρ without p-values: at n > 3.6M, effects of no interest clear p < 0.001 on sample size alone, so we use a triviality threshold of |ρ| = 0.05 instead.

*Its limit is what pushed us on.* Spearman and Pearson agreed closely per feature per genre, which looked like evidence of linearity until we plotted `avg_sentlen` by year and found a sharp rise-then-decline (peaking ~1720 in Fiction, ~1750 in Advertisement). Neither coefficient detects a non-monotonic shape, so a small gap between them means "both wrong the same way," not "the relationship is linear." A feature that rises for fifty years and falls for the next fifty returns ρ ≈ 0 — and our hypotheses are specifically about trajectories that reverse and differ in timing between genres. **Lesson: plot the shape before trusting either coefficient on a diachronic series.**

**Linear mixed-effects models.** Fit and dropped. All six converged, and they gave a genuinely interesting result — Advertisement moving opposite the other three genres on two independent constructs. But a linear model returns the average rate of change over the period, a quantity that exists for any data, including data where nothing changed at a constant rate. Given that we already saw the series were non-monotonic, that average is a summary of a shape the model can't see. The Advertisement finding survives in the GAMMs anyway, with the trajectory attached.

Two smaller things also argued against keeping them: `nominal_verb_ratio` and `personal_pronoun_ratio` hit a boundary warning with between-newspaper variance near zero, and a random intercept can only give each title its own level — but titles differ in the *shape* of their trajectories, and with two Copenhagen papers ending in the 1830s, that shape would get absorbed into the corpus-level year term as composition changes.

**Changepoint detection (`ruptures`, Pelt/rbf).** Tried and abandoned. The idea was to date the rise/peak/decline structure with breakpoint years rather than by eye. Two problems. The penalty parameter is a free choice with no principled setting at this scale, and we never got past a placeholder of 10. More fundamentally, at n > 3.6M *any* segmentation finds breaks and all of will show significance, so the method can't tell us whether a break is really there — only where it would be if we assumed one. Imposing discrete breakpoints is also the wrong model for change that the literature describes as gradual and continuous.

**GAMMs. Kept.** They estimate the shape rather than imposing one, and periods of change come from the first derivative — the years where its interval excludes zero — which dates change with an uncertainty statement and without pre-specifying breakpoints. Because we scan the whole range for exclusions rather than testing one year, the intervals are simultaneous rather than pointwise.

Specification notes:

- **Newspaper as a factor smooth** (`bs = "fs"`), for the reason above: titles differ in trajectory shape, not just level.
- **Resolution is our choice, so we report sensitivity.** Effective degrees of freedom sit near the basis ceiling at every setting tried, so the penalty isn't binding and k alone fixes the timescale. Refit across k ∈ {8,12,16} and γ ∈ {1,1.5,2}; a period is reported only where six of nine configurations agree on direction and none disagrees.
- **Article length enters only where it's a confound.** Covariate for the ratio measures; already removed at measurement for the diversity ones; omitted for sentence length and dependency distance, since those are partly constitutive of article length and articles getting shorter is part of the phenomenon.
- **OCR quality doesn't enter.** A smooth of predicted word accuracy saturates any basis we tried, which indicates a threshold rather than a graded relationship. It's near-uncorrelated with year (ρ = 0.07) and articles below 0.5 are already excluded, so we leave it out rather than model it with a basis that doesn't fit it.

# Things that could have been done but weren't

- **Spread of articles in DK / reprints.** How do reprints travel — do they, how many/much, and how far? This could help establish whether differences in newspaper trajectories are partly geographical diffusion lags. Probably a useful addition, but essentially a separate paper and we don't have the time or space here.
- **Dependency parsing validation.** We could have tested the DaCy dependency parser on the historical Danish data. Since it already performs highly on the benchmark, we expect it to perform comparably to the POS tagger, but we don't have an independent historical dependency test set to confirm this.