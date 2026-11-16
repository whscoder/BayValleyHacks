# Data and model availability

Only the production LR artifact is bundled. Raw corpora, intermediate CSVs,
review passages and neural model downloads are omitted to keep the judging
repository small and avoid redistributing the research workspace.

The app and offline runtime tests do not need the training corpora. The optional
`back-end/Ml_datapreprocessing.py` and `back-end/mltraintest.py` show data preparation
and training code; they are not a one-command reconstruction of every reported
research experiment. Re-training requires separately obtaining the source corpora
and reproducing the documented review/group/split decisions. Do not blindly train
an old combined CSV or present training-set accuracy as held-out performance.

The latest local repaired candidate used text,label,source,group_id as its first
columns, plus IDs, lineage, review authority, stored splits and component grouping.
The aggregate manifest and evaluation limitations are in docs/ML_RESULTS.md.
