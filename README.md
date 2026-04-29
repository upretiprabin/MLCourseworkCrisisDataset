# Disaster Tweet Classifier

A multi-class classifier that labels disaster-related tweets into humanitarian
categories (e.g. `requests_or_needs`, `infrastructure_and_utilities_damage`,
`injured_or_dead_people`). Trained on the
[CrisisBench](https://crisisnlp.qcri.org/crisis_datasets_benchmarks.html)
benchmark — ~142K labelled tweets from 61 real disaster events including the
2015 Nepal earthquake.

The project compares four classical text-classification algorithms (Naive
Bayes, Logistic Regression, Linear SVM, Random Forest) on top of TF-IDF
features and ships a small command-line tool for live predictions.

---

## Setup

```bash
# 1. Create / activate a virtual environment (the repo already has .venv/)
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. (Optional) Pre-download NLTK data — preprocess.py will also do this on demand
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Get the data

The CrisisBench train/dev/test JSON files belong in `data/raw/`. If they are
already present (as in this repo) the loader uses them directly. Otherwise:

```bash
python src/data_loader.py
```

This prepares `data/processed/crisisbench_en.csv` (English tweets only,
preserving the official splits).

## Run the full pipeline

```bash
python run_pipeline.py
```

End-to-end: load → preprocess → TF-IDF → train all 4 models → evaluate →
save the best model to `models/best_model.joblib`. The winner is chosen
by **weighted F1** with a noise-tolerant tiebreaker: when the top
candidates fall within 0.01 weighted F1 of each other (i.e. within seed /
train-test split noise), we cascade to **macro F1** and then training
time. On the current run that resolves to **Linear SVM** — also the
classical baseline used in the published CrisisBench paper.

## Use the CLI

```bash
# Single tweet
python cli.py --tweet "Massive earthquake destroys buildings, many trapped"

# Batch — reads a CSV with a 'text' column, writes predictions to results.csv
python cli.py --file tweets.csv --output results.csv

# Interactive (default)
python cli.py
```

## Notebooks

```bash
jupyter notebook notebooks/
```

- `01_data_loading.ipynb` — raw data exploration
- `02_eda.ipynb` — seven EDA visualisations (class balance, disaster-type
  heatmap, chi-squared discriminative words, category cosine similarity,
  length/vocabulary statistics, Nepal-vs-overall comparison, per-source bias)
- `03_model_comparison.ipynb` — training, evaluation, error analysis,
  per-class F1, model selection rationale

## Project structure

```
disaster-tweet-classifier/
├── CLAUDE.md                    # Build instructions for the AI agent
├── README.md                    # This file
├── requirements.txt
├── data/
│   ├── raw/                     # CrisisBench JSON splits
│   └── processed/               # Cleaned, English-only CSV
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocess.py            # Text cleaning pipeline
│   ├── features.py              # TF-IDF vectorisation
│   ├── train.py                 # All 4 classifiers
│   ├── evaluate.py              # Metrics & plots
│   └── utils.py                 # Shared constants & helpers
├── models/                      # Saved joblib bundles
├── figures/                     # PNG plots from EDA + evaluation
├── cli.py                       # Command-line interface
└── run_pipeline.py              # End-to-end driver
```

## Files at a glance

| File | Purpose |
| --- | --- |
| `src/data_loader.py` | Load CrisisBench from local JSON (HuggingFace fallback), filter English, save processed CSV |
| `src/preprocess.py` | Lowercase / URL & mention removal / hashtag cleaning / stopword removal / lemmatisation |
| `src/features.py` | TF-IDF vectoriser (1-2 grams, sublinear TF, min_df=2) |
| `src/train.py` | Trains MultinomialNB, LogisticRegression, LinearSVC, RandomForest |
| `src/evaluate.py` | Accuracy / precision / recall / F1 / confusion matrices / comparison plots |
| `src/utils.py` | Constants (`RANDOM_STATE`, paths), plot styling, logging helper |
| `cli.py` | Single-tweet, batch, and interactive prediction modes |
| `run_pipeline.py` | One-shot driver that runs every step in order |
