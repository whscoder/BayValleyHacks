# ML experiments and model decision

The application retains the existing production TF-IDF/Logistic Regression pipeline.
It was not silently replaced by the experimental hybrid LR or specialist router.
The exact supplied artifact is hashed in `SUBMISSION_MANIFEST.json`.

## Experimental candidate comparison

On the same 6,770-row stored test split:

| Candidate | Accuracy | Macro-F1 |
|---|---:|---:|
| Lexical LR | 83.75% | 0.8331 |
| Hybrid LR (lexical + frozen sentence embeddings) | 85.33% | 0.8487 |
| LR/SVM/NB-SVM/MLP stack | 85.89% | 0.8536 |

These are exploratory source/AI-label agreement results, not independently verified
news accuracy. **85.33% describes the experimental hybrid candidate, not a certified
score for the production pickle in this repository.** PERSPECTRUM had historical
test exposure before a user-approved grouped resplit. The reserved 400-example
human-news reference is still unlabelled. No candidate met every promotion safeguard.
The compact raw comparison is `results/model_comparison.csv`.

## Why keep LR instead of specialists?

Five frozen downloaded neural models and six saved LR/SVM/MLP/stack candidates
were considered. A category router selected NB-SVM for debate language and SVM
for short passages. On 1,925 validation passages from groups excluded from router
fitting, LR scored 88.83% and the router 88.78%: eight corrections, nine new errors.
The apparent small-sample gain did not hold in the broader stress replay. These
validation numbers are not a new full-test model upgrade. See `results/router_summary.json`.

No full transformer fine-tuning was completed. The downloaded transformer weights
and intermediate experiment artifacts are deliberately omitted from this repository.

## Corpus and review

The complete local ledger contained 52,102 rows, including held/mixed/uncertain
records. The latest repaired eligible candidate had 35,292 rows: 22,201 train,
6,321 validation, 6,770 test. Unsupported claimworthiness mappings and full-editorial
units were held rather than automatically relabeled. Eight exact AI-reviewed
excerpts were added with parent groups and offsets. They are not human gold.
See `results/data_manifest.json` and `../data/README.md` for data availability.

Cleanlab marked 2,793/23,354 earlier training rows in at least one model; 589 had
matching flags from all three. These were review priorities, not confirmed wrong
labels. Error weighting and extra ensembles did not establish a reliable improvement.
The grouped learning curve showed no general decline from adding data.

## API cost control

An offline replay through the production LR and updated review policy reduced
reviewed validation passages from 4,807/6,321 (76.0%) to 3,651/6,321 (57.8%).
That is about 24% fewer review passages, not a live billing measurement or an
accuracy guarantee. Requests are batched, so passage counts differ from HTTP
request counts. The aggregate is `results/fallback_summary.json`.
