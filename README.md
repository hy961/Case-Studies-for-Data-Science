# Case Studies for Data Science for Individual Tasks 1 (part 1.3) and 2
Hansen Yonatan (s4178876), RMIT University

Comparing two classifiers across two content moderation datasets, as part of an
application-oriented look at what a Machine Learning Engineer role in Trust and
Safety might involve. Task 2 revisits that analysis: it re-evaluates the models
with cross-validation, tests how far performance depends on training set size,
and audits error rates by targeted community.

## Notebooks

| File | Dataset | Task |
|---|---|---|
| `hatexplain_analysis.ipynb` | HateXplain | Multi-class (normal / hatespeech / offensive), plus all Task 2 experiments |
| `jigsaw_analysis.ipynb` | Jigsaw Toxic Comment | Multi-label (6 binary labels) |

## Datasets (not included in this repo)

- **Jigsaw Toxic Comment Classification**: https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge (CC0). Download `train.csv` and upload it when the notebook asks for it.
- **HateXplain**: https://github.com/hate-alert/HateXplain (MIT). Pulled automatically by the notebook.

## Method

TF-IDF with 10,000 features, unigrams and bigrams, English stop words removed.
80/20 train/test split, `random_state=999`. Models are LinearSVC and Logistic
Regression. For Jigsaw, One-vs-Rest handles the multi-label setup, and it's run
with both balanced and default class weights to compare the effect on
precision/recall.

For HateXplain, posts where all three annotators disagreed (919 of 20,148) are
dropped, leaving 19,229 posts for modelling.

## Task 2

Task 2 extends the same pipeline rather than starting a new one: identical
preprocessing, the same TF-IDF configuration and the same seed, so the Task 1
numbers reproduce inside the new output. Three things were added.

**Cross-validation.** 5-fold stratified CV with the vectoriser refitted inside
each fold, so no test fold leaks into the vocabulary. Macro F1 comes out at
0.613 +/- 0.007 for SVM and 0.643 +/- 0.007 for Logistic Regression, which puts
the single-split Task 1 scores inside the fold range.

**Learning curves.** F1 against training set size on a fixed test set, averaged
over three random subsamples per size, from 500 up to the full 15,383. Both
models flatten after roughly 2,000 examples. Saved as `learning_curves.png`.

**Fairness audit.** Error rates broken down by the target community that
HateXplain's annotators declared, using Fairlearn's `MetricFrame` with bootstrap
confidence intervals. Twelve communities were audited; those with fewer than 20
non-toxic test posts are excluded from the reported table, since their intervals
are too wide to read.

A final cell reproduces the figures added while writing the report: the Jigsaw
stratification check, the worst-case accuracy bound for the dropped posts, the
reporting threshold for the fairness table, the base-rate correlations, and the
token weights.

## Results

Full results and discussion are in the accompanying reports. Notebook outputs
are committed so everything's reproducible without needing to re-run.
