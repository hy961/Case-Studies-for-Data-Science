# Case Studies for Data Science - Individual Task 1, Part 1.3
Hansen Yonatan (s4178876), RMIT University

Comparing two classifiers across two content moderation datasets, as part of an
application-oriented look at what a Machine Learning Engineer role in Trust and
Safety might involve.

## Notebooks

| File | Dataset | Task |
|---|---|---|
| `hatexplain_analysis.ipynb` | HateXplain | Multi-class (normal / hatespeech / offensive) |
| `jigsaw_analysis.ipynb` | Jigsaw Toxic Comment | Multi-label (6 binary labels) |

## Datasets (not included in this repo)

- **Jigsaw Toxic Comment Classification**: https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge (CC0). Download `train.csv` and upload it when the notebook asks for it.
- **HateXplain**: https://github.com/hate-alert/HateXplain (MIT). Pulled automatically by the notebook.

## Method

TF-IDF with 10,000 features, unigrams and bigrams, English stop words removed.
80/20 train/test split, `random_state=42`. Models are LinearSVC and Logistic
Regression. For Jigsaw, One-vs-Rest handles the multi-label setup, and it's run
with both balanced and default class weights to compare the effect on
precision/recall.

For HateXplain, posts where all three annotators disagreed (919 of 20,148) are
dropped, leaving 19,229 posts for modelling.

## Results

Full results and discussion are in the accompanying report. Notebook outputs
are committed so everything's reproducible without needing to re-run.
