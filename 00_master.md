
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
