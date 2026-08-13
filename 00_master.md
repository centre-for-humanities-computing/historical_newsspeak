
# June 19
- Need to rerun pipeline & include readability as well
- predicting stylistics vectors from embeddings? I can do that after Dacy pipeline
- cliffhangers might also be preicted for local changes in novel timeline? like, inconsistencies in the embedding-chunks?


## embeddings testing:
- to test embeddings, we might want to look into the raw chunks as well? like how easily can one be confused for another (from a different time?)
- what is better than taking the mean embedding? e.g., for example fiction/nonfiction classification? Maybe better to take the max/min, median, or mean of n sampled chunk-embeddings? (e.g., 10 random chunks per novel, then take the mean of those 10 chunk-embeddings)
- is plot part of the signal? e.g., is narrative arcs part of the signal?

Okay, so what we could do is:
- fiction/nonfiction classification
- test embedding settings (mean vs. max vs. median, centroid, etc.)
- ablate stylistic features + sentiment features (like the ccls paper)

# Juli 7
- rerunning DACY pipeline (only parquet spacy book extraction, no stylistics)

# July 12
- Success, all data run + improved stylistics
- TODO:
  - rerun the diachronic analysis
  - update github structure
  - update abstract incorp. feedback from reviewers
  - change to EADH template!
- Consider adding:
  - Recurrence Quantification (RQ) analysis
  - Novelty detection (e.g., using stylistics vectors)

# July 13
- computed nominal ratio wrongly, corrected now.



# superviison 

- think about what i have learned from EADH and JCLS abstract
- think of what is the next iunteresting q  uestion based on my overall phd project
- then think of measures i want to throw in


- look at how features change over time, whether they change in the same way (could even do arc clustering)
- HP? if literacy levels go down as more articles are produced, then sentence length should change linearly
- fiction might have different inner pools


ALSO:
we could predict memo from fiction in newspapers

# KNL supervision
- some mean-changing method should be enough (we have an expectation about the direction)
- ev. something like arc clustering
- but i dont know if 0 at date of expected change is good enough


# Suggestions

Robustness checks to actually run

OCR-error sensitivity. Since OCR quality plausibly varies by period (older typefaces) and genre (dense ad typesetting vs. prose columns), check whether your diachronic trends survive when you (a) restrict to articles above some CER/confidence threshold, or (b) model OCR-confidence as a covariate. If early-period trends attenuate once you filter for OCR quality, that's important to know and report either way.

Genre-classifier validation, in this paper. Even if the classifier is described elsewhere, report its accuracy/confusion matrix here, and ideally show your key results are stable when you restrict to high-confidence classifications only. Domain divergence is your whole thesis — misclassification directly attenuates or fabricates it.

Non-linear time trends. Refit key features with GAMs or splines instead of (or alongside) linear year terms. Report where linearity holds and where it doesn't — breakpoints (wars, print-tech shifts, orthographic reforms) are a more historically interesting story than a slope anyway.

Threshold sensitivity for |ρ| > 0.05. Show the domain-ranking (Intl > National > Advert > Fiction in # of "non-trivial" correlations) is stable across a couple of alternative thresholds (e.g., 0.03, 0.08), so it doesn't look like an artifact of the cutoff you picked.
langdetect validation. Some spot-check or small hand-labeled sample against the German-probability control, since langdetect on short historical Danish (cognate-heavy with German) is a fairly blunt instrument doing real work in your models.

Fiction's small-n fragility. Either bootstrap the factor-congruence estimates for Fiction specifically, or otherwise quantify how much the "structurally distinct" claim depends on the smaller sample — right now it's flagged in a footnote but carrying a lot of narrative weight.

Effect sizes and interpretability

Translate every headline coefficient into something concrete: "X fewer words per sentence over the full period," "Y% of a between-article SD per decade," etc. At n=3M+, statistical detectability is cheap; the paper needs to argue magnitude, not just direction.

Consider standardized coefficients (or effect size in SD units) in the mixed-effects table so readers can compare across features with different scales (sentence length vs. TTR vs. ratios).
A time-series panel plot (feature × genre, smoothed trend with CI band) would do more work than the correlation heatmap for showing H2/H3 side by side.

Alternative explanations to address head-on

Advertisement reversal — economic/formulaic alternative. Shorter, less diverse ad copy over time could reflect pricing-by-line/word or increasing formulaic standardization of ad genres, not "complexity" in the same sense as prose. Worth discussing (and maybe testing — e.g., does ad length itself shrink over time, independent of style?) rather than leaving it as a pure style-evolution story.

Print-expansion mechanism vs. proxy. Right now "year" stands in for the whole literacy/mass-readership expansion story. If you can get any independent readership/circulation/literacy proxy (even coarse, decade-level), regressing against that directly instead of (or alongside) year would make the framing's causal claim much better supported.

Newspaper-level heterogeneity. Random intercepts control for newspaper, but it's worth checking whether trends are driven by a handful of long-running papers vs. broadly shared across the 28 titles — a supplementary look at trend heterogeneity by newspaper (or a random slope model) would strengthen "genre-wide" claims.

Framing / theoretical grounding

Engage a bit more with why these particular six features were chosen as representative of the three factors beyond loading magnitude — a paragraph connecting each construct back to the theoretical literature (Biber, readability studies) would help non-specialist readers see the constructs as more than statistical artifacts.
Since H1 (shared drift) is essentially rejected, it's worth being explicit early on about what would have counted as evidence for it, so the eventual rejection reads as a real test rather than a foregone conclusion.
Address seasonality/production cycles if relevant (e.g., does article length/style vary by time of year, war years, etc., independent of the decade trend) — probably minor, but worth a sentence ruling it out if you have the resolution.

Presentation/expansion opportunities

A full paper has room for the PC/factor loading heatmap (Fig. currently in appendix) to move into the main text with a fuller discussion of what's not separable in Fiction — that's one of your more interesting findings and it's currently compressed into a dense paragraph.
Consider adding a short qualitative pass — a handful of close-read examples of Advertisement text from early vs. late period — since a purely quantitative paper about a genre reversal benefits a lot from a "here's what this actually looks like on the page" anchor for readers.
If space allows, briefly situate the Danish findings against the specific English-language studies you cite in the intro (Feldkamp et al., Algee-Hewitt et al.) — do you replicate, contradict, or complicate their domain-specific patterns? Right now the comparison is implicit; a full paper can make it explicit in the discussion.