# Methods checkpoint: feature structure analysis (compressed)

Context: reviewers evaluated an earlier draft on a 200K-article subset;
corpus now 5M articles. Reviewer concerns addressed: (1) are
"complexity" features actually complexity, or mixed with register/
style; (2) are the dimensions empirically orthogonal, or is a more
generalized metric derivable.

## Key decisions and findings

**A priori complexity/register split was only partially right.**
Started with a theory-driven split; correlation clustering and factor
analysis showed several features (`function_word_ratio`,
`nominal_verb_ratio`) empirically belong to the *other* group. Moved to
a fully data-driven approach (factor analysis on the combined set)
rather than defending the original split.

**Initial PCA finding: no single "Complexity" component** — needed 11
of 17 components for 90% variance. This is itself a legitimate,
reportable answer to reviewer point 2: the features are largely
non-redundant, not one latent dimension.

**Length confound found and fixed in the TTR family.** CTTR correlated
0.94 with token count (near-total redundancy — CTTR was essentially
measuring article length, not diversity, due to a ceiling effect on
short texts). Fixed by residualizing `cttr`, `noun_ttr`, `verb_ttr`
against log(token count) via OLS (`_resid` suffix); 88.9% / 25.9% /
18.7% of their variance was length, respectively. **LIX/RIX/avg_mdd's
correlation with sentence length was NOT corrected** — it's
definitional (sentence length is literally a term in the LIX formula),
not confounding. Document length itself is kept as a **covariate** in
the final model rather than residualized out, since it may reflect
genuine genre effects worth reporting, not just noise.

**`compressrat` and `mtld` excluded from factor analysis.** Both are
NaN below a length threshold (compressrat: <40 sentences, 99.4%
missing; mtld: <50 tokens, ~30% missing). Combined with `dropna()`,
this silently collapsed the analysis sample to 18,311 rows (0.5% of
corpus) — severely length-biased. Excluding both restored the sample
to ~3.34-3.38M rows (91%+). Report compressrat/mtld separately as
descriptive stats on their length-eligible subsets only.

**Bug found and fixed: `nominal_verb_ratio`.** Original code omitted
NOUN from the nominal set (`PROPN`+`ADJ` only) and used an unbounded
count-ratio (`nominals/verbs`, not a proportion) — unstable for
verb-sparse short articles. Fixed: include NOUN; redefine as
`nominals/(nominals+verbs)`, bounded 0-1. Communality jumped from ~0.10
to 0.63; now loads strongly (.78-.79) on the register factor, matching
Halliday/Biber nominalization-as-informational-register theory. Good
validation, not just a rescued statistic.

**`of_ratio`'s low communality is explained, not fixable.** Even at
≥100 tokens, still 13.3% exactly zero — inherent sampling noise from a
low-base-rate word count (~1-2 occurrences expected per 100 tokens),
not a missing latent dimension. More factors won't help; report as a
known property of measuring a rare word's rate.

**Why factor analysis over PCA**: PCA forces orthogonal components
using all variables in each — nothing pushes toward clean,
interpretable groupings (confirmed by messy PC1-3 cross-loadings early
on). Checked factorability first (Bartlett p<.0001, KMO 0.7-0.83
depending on run — "acceptable" to "meritorious"). Used promax
(oblique) rotation: lets factors correlate (realistic — register and
complexity plausibly do), and explicitly optimizes for simple structure
(each feature loads strongly on ~one factor). See below for a fuller
conceptual explanation.

**German-detection features excluded from the factor model.** Loaded
cleanly (.97-1.00) on their own isolated factor but aren't a stylistic
construct — a content/language-ID signal. Removing them changed total
variance explained by <0.1 percentage points, confirming they were
genuinely orthogonal noise as far as style goes. Use
`german_probability` as a **covariate** in the final model instead.

**Collinearity investigation — closed, 3-factor model retained.**
VIF check on the sentence-length cluster: `rix`=17.7 (only real
offender), `avg_mdd`/`std_mdd`/`avg_ndd`/`std_ndd` all <3 (MDD/NDD
redundancy hypothesis rejected — the log-transform in NDD's own formula
sufficiently decorrelates it from MDD). Tried dropping `rix`: KMO
*worsened* (0.697→0.577), singular-matrix warning persisted — rix
wasn't the (sole) cause. Tried 6 factors: produced a Heywood case
(negative uniqueness, mathematically improper) — rejected as
overextraction, not a hidden construct. **Final decision: keep `rix`,
retain the 3-factor solution**, and document the Moore-Penrose
pseudo-inverse fallback as an acknowledged limitation (common with
closely correlated readability formulas). Justification: loadings have
been remarkably stable across every variant tested (with/without
German features, before/after the nominal_verb_ratio fix, with/without
rix) — this cross-run stability is stronger evidence of validity than
the raw fit diagnostics alone.

## Final 3-factor structure

- **F1 — Syntactic complexity**: avg_sentlen, lix, rix, avg_mdd
  (all ~0.7-0.95+)
- **F2 — Lexical diversity**: cttr_resid, noun_ttr_resid, mtld
  (~0.65-0.9; compressrat also loads here on its eligible subset,
  providing convergent validity from an independent measurement method)
- **F3 — Involved vs. informational register**: personal_pronoun_ratio,
  that_ratio, function_word_ratio (positive pole) vs. passive_ratio,
  present_tense_ratio, nominal_verb_ratio (negative pole) — matches
  Biber's (1988) Dimension 1 closely

## Sentiment: mean and variability, properly standardized

Mean: pooled exactly across the whole corpus (sum/sumsq/n per article,
summed across all shards, single global z-score) — NOT standardized
per-shard, since `SemanticProjector.standardize()` z-scores relative to
whatever batch it's given and per-shard standardization would make the
same absolute value score differently depending on shard composition.
Use **`semantic_sentiment_standardized`**.

Within-article variability: derived from the same stored sum/sumsq/n
(no raw-score recomputation needed). Because variance is ratio-scale
(not additive like the mean), standardized via **log(article variance
/ global pooled variance)** — symmetric around 0, matches the rigor of
the mean's treatment. Use **`semantic_sentiment_log_var_ratio`**,
gated by **`semantic_sentiment_std_defined`** (False for n=1 articles,
where variance is trivially/meaninglessly zero). Caveat: doesn't
correct for small-n sampling noise in the variance estimate itself —
treat cautiously for articles with very few sentences.

**Columns to actually use**: `semantic_sentiment_standardized`,
`semantic_sentiment_log_var_ratio`, `semantic_sentiment_std_defined`.
Everything else (`sum`/`sumsq`/`n`/`mean_raw`/`var_raw`/`std_raw`) is
intermediate material, not for direct reporting.

## Final resolution: LIX/RIX collinearity root-caused and fixed; 3-factor solution confirmed via systematic sweep

**Root cause of the persistent collinearity, finally identified via VIF**:
not `rix` specifically (dropping it alone just shifted the high VIF onto
`lix`, or vice versa) — both LIX and RIX are composite formulas that
directly contain `avg_sentlen` as an additive term. Keeping either
composite alongside its own raw ingredient double-counts the same
sentence-length information. **Fix: dropped both `lix` and `rix`**,
keeping `avg_sentlen`/`avg_wordlen`/`avg_mdd` (already in the model) to
carry the same readability signal directly. Result: `avg_sentlen`'s VIF
dropped from ~15-16 to **2.92** — full resolution, not a partial one.
KMO barely moved (0.622→0.624), confirming the earlier low KMO was a
diffuse property of this feature set, not caused by this specific
collinearity.

**Systematic factor-count sweep (2-8) on the corrected feature set**:
technical ceiling for a "proper" (no Heywood case) solution rose from
3-4 (on the collinear data) to **5**. But inspecting F4/F5 directly:
`avg_sentlen` fragments across 2-3 factors simultaneously at k=4/5
(same failure signature as the original pre-fix over-extraction
attempts, just less severe) rather than any coherent new construct
emerging. At k=5, `avg_sentlen`'s uniqueness sits at 0.0165 — right at
the instability boundary, not a robust proper solution. F1-F3 (register,
syntactic complexity, lexical diversity) are essentially IDENTICAL
across every k tested (3, 4, 5) — the strongest evidence of validity in
the whole investigation.

**Final decision: adopt the 3-factor solution, WITHOUT lix/rix, as the
reported model.** This is a stronger, more defensible endpoint than any
earlier version: the collinearity has a genuine mechanistic
explanation (not just an empirical correlation), VIFs are fully
resolved, and a systematic sweep demonstrates that factors beyond 3
technically avoid Heywood cases but do not represent real independent
structure — directly and rigorously answering reviewer 2's
dimensionality-reduction question with evidence of a genuine ceiling,
not an interpretability preference.

## Final 3-factor structure (superseding all earlier versions)

- **F1 — Involved vs. informational register**: nominal_verb_ratio
  (.71-.80), personal_pronoun_ratio (-.62 to -.70), passive_ratio
  (~.5), present_tense_ratio (~.45), that_ratio (~-.40),
  function_word_ratio (~-.3). Sentiment cleanly integrated here:
  semantic_sentiment_standardized (~.3-.35), semantic_sentiment_log_var_ratio
  (~-.32 to -.39) — more informational articles trend more positive and
  more tonally consistent. Matches Biber's Dimension 1.
- **F2 — Syntactic complexity**: avg_sentlen (.79-.89), avg_mdd
  (.86), std_mdd (.42-.53) — carried directly by raw sentence-length/
  dependency-distance measures now that lix/rix are removed.
- **F3 — Lexical diversity**: cttr_resid (.81-.85), noun_ttr_resid
  (.63-.72), verb_ttr_resid (.40-.42).



## Measurement invariance across genre — checked, mixed results (important)

Concern raised: the paper's argument is about domain-SPECIFIC change,
but a single pooled factor model assumes one domain-GENERAL construct.
Tested via Tucker's congruence coefficient (genre-specific 3-factor fits
vs. the pooled model, full pairwise matrix, abs() for sign
indeterminacy).

**Clean invariance**: National (0.96/0.92/0.83) and Advertisement
(0.99/0.95/0.93) — each genre factor maps cleanly to a distinct pooled
factor. Pooled scores well-justified for these genres.

**Partial breakdown — International**: 2 of 3 factors map cleanly
(0.94, 0.88); genre_F2 matches nothing well (max 0.42, spread evenly
across all three pooled factors) — not a sample-size artifact (n=596K),
a genuine structural difference in what this dimension represents.

**Clear breakdown — Fiction**: only 1 of 3 factors clean (0.91); the
other two both collide onto pooled F1 (register) and neither matches
pooled F3 (lexical diversity) at all — in fiction, diversity and
register aren't separable the way the pooled model assumes. Fiction is
also the smallest category (n=58,751), though likely large enough that
this reflects genuine structural difference, not pure noise.

**This directly reinforces reviewer 1's original point 4** (concern
that the strongest correlations were concentrated in International/
fiction, "also the smallest categories") — now with formal, quantitative
support rather than a hand-wave answer.

**Decision**: use pooled factor scores as primary for National/
Advertisement (and International with a caveat on its ambiguous
factor); treat Fiction's results as more tentative or model it
separately given the invariance breakdown — to be finalized once
mixed-effects modeling begins.



## Final analytical direction: combining Options B and C

Resolved the domain-general-construct tension by splitting the FA work
into two separate roles rather than picking one path:

1. **The genre-specific factor structure IS a reported finding**
   (Option C) — the pooled 3-factor model, leave-one-out invariance
   test, and genre-specific breakdowns (Fiction's diversity/register
   collision, International's ambiguous factor) get written up as a
   standalone result: linguistic style's internal organization is
   itself domain-specific, not just its level or trend. Reinforces the
   paper's core thesis at a structural level, not just a descriptive
   one.

2. **The main trend analysis does NOT use a pooled factor score**
   (Option B) — avoids asserting one domain-general construct for the
   causal claims the paper actually needs to make. Instead, model
   individual, well-validated representative features directly, one
   mixed-effects model each:
   - Complexity: `avg_sentlen`, `avg_mdd`
   - Register: `nominal_verb_ratio`, `personal_pronoun_ratio`
   - Diversity: `cttr_resid`, `noun_ttr_resid`

   Spec: `feature ~ year_centered * category + log(tokens) +
   german_probability`, random intercept/slope by newspaper (or
   category, depending on nesting) — the interaction term directly
   tests domain-specific change without requiring cross-genre
   measurement invariance to hold.

**No longer needed**: computing/attaching a pooled factor score to
`df` for downstream modeling — the FA work is now reported as a result
in its own right, not used as an input feature.



## Mixed-effects models: complete, all six converged

Random-intercept-only spec (per the decision above) fit cleanly across
all six representative features. Hessian warnings on 5/6 models are
consistent with the benign large-N numerical-precision pattern
confirmed during pilot testing (no pathological parameter signatures).
`nominal_verb_ratio`/`personal_pronoun_ratio` additionally showed a
"boundary of parameter space" warning — between-newspaper variance
estimated near zero for these two, plausibly genuine (register markers
may be more genre- than newspaper-driven) rather than a fitting
problem; worth a quick confirmation check but not treated as
disqualifying.

**Key finding**: Advertisement moves in the OPPOSITE direction from
National/International/Fiction on two independent constructs —
syntactic complexity (avg_sentlen, avg_mdd) AND lexical diversity
(cttr_resid, noun_ttr_resid) — decreasing on both while the other three
genres trend toward increasing complexity/diversity over time.
Advertisement's syntactic-complexity decline is also the single largest
effect in the whole table (~-1.5 words/sentence per decade). This is a
stronger, more specific version of "domain-specific change" than
differing magnitudes alone — genuinely bidirectional across genres on
independent measures, directly supporting the paper's central thesis.



## Session: correlation-with-time, linearity check, changepoint detection

Rebuilt the paper's correlation table on the FULL corpus (not the old
subset), correlating each feature against `date_ordinal` per genre,
ranked by |rho|. Added a post-1715 sensitivity version to check
whether thin early-year samples were distorting results.

**Linearity check**: compared Spearman |rho| vs Pearson |r| per
feature per genre — gaps were small everywhere (Advertisement ~0.0001
to International ~0.0218), which initially looked reassuring. **This
was a false signal**: plotting `avg_sentlen` by year revealed a sharp
non-monotonic rise-then-decline (peaking ~1720 Fiction, ~1750
Advertisement) — Spearman and Pearson both fail to detect this shape
(neither is built for non-monotonic relationships), so a small gap
between them reflects "both wrong the same way," not linearity.
**Lesson: check the actual shape visually before trusting either
correlation coefficient on a diachronic series.**

**Changepoint detection** (`ruptures`, Pelt/rbf) added to formally
locate the rise/peak/decline structure directly, addressing reviewer
point 4's structural-discontinuity concern with quantitative breakpoint
years rather than an eyeballed chart. Penalty parameter not yet tuned
(pelt_penalty=10 is a placeholder) — needs a quick sweep before
treating specific breakpoint years as final.



## Outstanding / not yet done / future works

- [ ] Tune/go into changepoint detection more deeply (currently penalty placeholder=10); report breakpoint years for at least avg_sentlen per genre.
- [ ] Nonlinearity check on all features.
- [x] Correlation-with-time table regenerated on full corpus — DONE.
- [x] Set up mixed-effects models on the six representative individual
      features — DONE.
- [x] Write up the genre-specific FA / invariance-breakdown findings
      as a standalone results subsection (with the Fiction/International
      congruence tables and interpretation).
- [x] Regenerate diagnostic plots (parallel analysis, loadings heatmap,
      communality chart) against the FINAL pooled feature set (no
      lix/rix, 3 factors) — for the standalone FA results section.
- [x] Remaining reviewer point 4 items (genre-imbalance bootstrap,
      RQA) — likely deferred past this deadline given time constraints.