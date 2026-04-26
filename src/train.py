"""
src/train.py
============

Trains and persists the four classifiers compared in this project:

  1. Naive Bayes (MultinomialNB)
  2. Logistic Regression
  3. Linear SVM (LinearSVC)
  4. Random Forest

We define every model in a single dictionary so we can train, evaluate and
save them with a uniform loop instead of writing four nearly-identical code
paths.

The pedagogical "why" behind each algorithm and parameter lives directly
above the model in MODEL_CONFIGS — see the comment blocks there.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# os — file paths for save/load.
import os

# time — measure how long each model takes to train. We report this so we
# can compare not only accuracy but also training cost.
# Example: t0 = time.time(); ...do work...; print(time.time() - t0)
import time

# typing — Dict / Tuple / Any are used in function signatures for clarity.
from typing import Any, Dict, Tuple

# joblib — the de-facto standard for persisting scikit-learn models.
# Like pickle but optimised for large numpy arrays. Files end in `.joblib`.
# Example: joblib.dump(obj, path); obj = joblib.load(path)
import joblib

# scikit-learn classifiers ---------------------------------------------------

# MultinomialNB — Naive Bayes for count/frequency features (perfect for
# TF-IDF). See block comment below for the algorithm description.
from sklearn.naive_bayes import MultinomialNB

# LogisticRegression — despite its name, this is a CLASSIFIER. It learns one
# weight per (feature, class) pair and combines them through a softmax to
# produce probabilities.
from sklearn.linear_model import LogisticRegression

# LinearSVC — a linear support-vector classifier. Same family as a kernelised
# SVM but constrained to a straight-line decision boundary in feature space,
# which makes it dramatically faster on high-dimensional sparse text.
from sklearn.svm import LinearSVC

# RandomForestClassifier — an ensemble of decision trees that vote on the
# answer. Adds non-linearity / interaction modelling to the comparison.
from sklearn.ensemble import RandomForestClassifier

# Project utilities.
from src.utils import RANDOM_STATE, MODELS_DIR, ensure_dir, log_step


# ---------------------------------------------------------------------------
# Model configurations
#
# Each entry: name -> sklearn estimator instance.
# The block comments above each entry explain WHY this algorithm is in the
# comparison and what each parameter means. These comments are the heart of
# the file's pedagogical value — please read them before changing anything.
# ---------------------------------------------------------------------------

# === 1. Naive Bayes (MultinomialNB) ========================================
#
# What it is
# ----------
# Bayes' rule says: P(class | tweet) ∝ P(class) × P(tweet | class).
# To compute P(tweet | class) we'd need the joint distribution of every word
# combination — astronomically expensive. The "naive" assumption is that the
# words in a tweet are conditionally INDEPENDENT given the class, so:
#       P(tweet | class) ≈ ∏ P(word | class)   (product over words in tweet)
# That is obviously false in real language ("storm" and "surge" are
# correlated) but the approximation is "good enough" for many text problems
# and produces a model that is fast to fit and surprisingly competitive.
#
# Why "Multinomial"
# -----------------
# MultinomialNB models word frequencies — counts per document. TF-IDF is not
# strictly a count, but it's a positive real-valued frequency proxy, and
# MultinomialNB handles it gracefully because the implementation only relies
# on (a) feature values being non-negative and (b) per-class sums.
#
# Why we include it
# -----------------
# Baseline floor. NB is fast, simple, and historically the first thing
# anyone tries on text. Any "real" model we choose must beat NB; if it
# doesn't, the added complexity isn't earning its keep.
#
# Parameter: alpha = 1.0 (Laplace / "add-one" smoothing)
# ------------------------------------------------------
# Without smoothing, if a word in a TEST tweet was never seen in TRAINING for
# class C, then P(word | C) = 0, which makes the entire product 0 and that
# class is impossible — even if every other word strongly suggests it.
# alpha=1.0 means we pretend every word was seen one extra time per class,
# so probabilities are tiny but non-zero. Setting alpha=0 would give the
# pathological zero-probability behaviour; setting alpha very high
# over-smooths and washes out genuine signal. 1.0 is the standard default.

# === 2. Logistic Regression ================================================
#
# What it does
# ------------
# Learns one weight per feature per class. To classify a new tweet, it sums
# the weights of every word in the tweet (per class), then runs the result
# through a softmax to get probabilities. Words that strongly co-occur with
# class C in training get large positive weights for C and small or negative
# weights for the others.
#
# Worked example for the tweet
#   "bridge collapsed earthquake people trapped"
# Suppose Logistic Regression has learned (weights for the
# infrastructure_and_utilities_damage class):
#   bridge:    +1.4
#   collapsed: +2.1
#   earthquake:+0.6
#   people:    -0.1
#   trapped:   +0.9
# Sum = 4.9, which softmax turns into a probability close to 1 for that
# class. The same words would yield much smaller sums for, say,
# "donation_and_volunteering", so that class loses.
#
# Where it's used in industry
# ---------------------------
# Spam filtering, click-through-rate prediction, churn modelling, medical
# screening (e.g. "is this lab result anomalous?"). It's the workhorse model
# of applied ML — fast, interpretable, scales to billions of features.
#
# Why we include it
# -----------------
# Top performer on TF-IDF text in most published benchmarks, including
# CrisisBench. Interpretable: per-class weights mean we can directly inspect
# which words drove a prediction.
#
# Parameter: C = 1.0
# ------------------
# C is the INVERSE regularisation strength. Smaller C = stronger
# regularisation = simpler model that may underfit. Larger C = weaker
# regularisation = complex model that may overfit. 1.0 is the sklearn default
# and a reasonable starting point for text.
#
# Parameter: max_iter = 1000
# --------------------------
# The solver runs an iterative optimiser (LBFGS). Default is 100 iterations
# which is too few for a 100k-row, 10k-feature problem and produces a
# ConvergenceWarning. 1000 is comfortable.
#
# Parameter: class_weight = "balanced"
# ------------------------------------
# Our data is heavily skewed: ~37% of tweets are not_humanitarian and only
# ~0.4% are missing_and_found_people. Without intervention the model would
# learn "if in doubt, predict the majority class". `balanced` automatically
# scales each class's loss contribution by the inverse of its frequency, so
# every class effectively contributes equally during training.
#
# Parameter: solver = "lbfgs"
# ---------------------------
# LBFGS is a quasi-Newton optimiser — fast for medium-sized multinomial
# problems and the default for multi-class logistic regression in modern
# sklearn.
#
# Parameter: multi_class = "multinomial"
# --------------------------------------
# Treats all classes simultaneously through a single softmax (vs. fitting
# one-vs-rest binary models). Generally better calibrated probabilities and
# slightly higher accuracy on text.

# === 3. Linear SVM (LinearSVC) =============================================
#
# What an SVM does
# ----------------
# A support-vector machine looks for the decision boundary that maximally
# separates the classes — i.e. the line (or hyperplane in higher dimensions)
# with the largest possible "margin" between the closest points of each
# class. The support vectors are those closest points; everything farther
# away is irrelevant to the boundary.
#
# Why "Linear"
# ------------
# A general SVM can use a non-linear "kernel" to bend the boundary. For
# bag-of-words / TF-IDF features the linear kernel is empirically
# best-in-class — texts already live in a very high-dimensional space, and
# bending that boundary further usually overfits.
#
# How it differs from Logistic Regression
# ---------------------------------------
# LogReg minimises log-loss (probability calibration). LinearSVC minimises
# hinge loss (margin maximisation). Both produce linear decision boundaries,
# but they emphasise different things: LogReg cares about getting
# probabilities right; SVM only cares about getting points on the correct
# side of the margin.
#
# Why we include it
# -----------------
# Replicates the methodology used in the published CrisisBench paper, gives
# a different optimisation philosophy in the comparison, and tends to be
# competitive with LogReg.
#
# Parameter: C = 1.0
# ------------------
# Same role as in LogReg — inverse regularisation strength.
#
# Parameter: class_weight = "balanced"
# ------------------------------------
# Same role as in LogReg — automatically reweight to compensate for class
# imbalance.
#
# Parameter: max_iter = 10000
# ---------------------------
# LinearSVC's default of 1000 frequently fails to converge on text data. We
# bump it to 10000 to silence ConvergenceWarning. (We don't usually hit
# the limit; most fits converge well before.)
#
# NOTE: LinearSVC has no .predict_proba(). At inference time we'll fall
# back to .decision_function() distances if we want pseudo-confidences.

# === 4. Random Forest ======================================================
#
# What a decision tree does
# -------------------------
# A decision tree asks a sequence of yes/no questions about the features.
# E.g. "does the tweet contain 'help'? -> yes -> does it contain 'donate'?
# -> no -> does it contain 'trapped'? -> yes -> predict missing_and_found".
# Each internal node is a feature threshold, each leaf is a class
# prediction. Trees are non-linear and can model interactions ("'help' AND
# 'flood' but NOT 'donate'").
#
# What a random forest does
# -------------------------
# A random forest trains MANY decision trees on bootstrap samples of the
# data and on random subsets of the features, then averages / votes their
# predictions. The randomness across trees decorrelates their errors —
# individual trees can be wildly overfit, but their majority vote is much
# more stable. This is "ensembling".
#
# Why we include it
# -----------------
# It's the non-linear contrast in the comparison. If word interactions
# matter (e.g. "rescue" + "helicopter" together strongly implies one class
# but neither word alone does), the forest can capture that whereas the
# linear models cannot.
#
# Parameter: n_estimators = 200
# -----------------------------
# Number of trees in the forest. More trees = more stable predictions but
# slower training. 200 is a sensible balance.
#
# Parameter: max_depth = None
# ---------------------------
# Let trees grow until every leaf is pure (or hits min_samples_split). Trees
# on TF-IDF features need to be deep to find good splits, and the bagging
# step controls overfitting at the ensemble level even when individual trees
# are huge.
#
# Parameter: class_weight = "balanced"
# ------------------------------------
# Same as the linear models: each tree's split criterion is reweighted to
# compensate for class imbalance.
#
# Parameter: random_state = RANDOM_STATE (42)
# -------------------------------------------
# Forests are intrinsically random (bootstrap samples + feature subsampling).
# Fixing the seed makes the model deterministic — same data and parameters
# always produce the same forest.
#
# Parameter: n_jobs = -1
# ----------------------
# Use every available CPU core to fit trees in parallel. -1 is sklearn's
# convention for "all cores".

MODEL_CONFIGS: Dict[str, Any] = {
    "Naive Bayes": MultinomialNB(alpha=1.0),

    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        # multi_class="multinomial" is the SKLEARN default starting in v1.5+
        # and the only option in 1.7+, so we omit it to silence the
        # deprecation warning. Behaviourally this still uses a single
        # multinomial softmax over all classes — exactly what we want.
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_model(name: str, model: Any, X_train, y_train) -> Tuple[Any, float]:
    """Fit a single classifier on (X_train, y_train) and time how long it took.

    Parameters
    ----------
    name : str
        Display name (used only for log output).
    model : sklearn estimator
        An UNFITTED estimator (e.g. one of MODEL_CONFIGS).
    X_train : array-like or sparse matrix of shape (n_samples, n_features)
        TF-IDF feature matrix for training tweets.
    y_train : array-like of shape (n_samples,)
        Class labels for training tweets.

    Returns
    -------
    (fitted_model, elapsed_seconds) : tuple
        fitted_model — the same `model` object after .fit() (sklearn fits in
                       place and returns self).
        elapsed_seconds — how long the fit took, in seconds.
    """
    log_step(f"  training {name} …")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    log_step(f"    {name} done in {elapsed:.1f}s")
    return model, elapsed


def train_all_models(X_train, y_train) -> Dict[str, Dict[str, Any]]:
    """Train every model in MODEL_CONFIGS.

    Parameters
    ----------
    X_train, y_train : as in train_model().

    Returns
    -------
    dict
        Maps model name to a dict with keys:
            "model"   -> the fitted estimator
            "fit_seconds" -> elapsed training time in seconds
        Example:
            {"Naive Bayes": {"model": MultinomialNB(...), "fit_seconds": 1.4},
             "Logistic Regression": {...}, ...}
    """
    results = {}
    for name, model in MODEL_CONFIGS.items():
        fitted, elapsed = train_model(name, model, X_train, y_train)
        results[name] = {"model": fitted, "fit_seconds": elapsed}
    return results


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """Convert a model display name into a filesystem-safe filename stem.

    Example: "Logistic Regression" -> "logistic_regression"
    """
    return name.lower().replace(" ", "_").replace("/", "_")


def save_model(
    model: Any,
    vectorizer,
    name: str,
    path: str = MODELS_DIR,
    label_classes=None,
) -> str:
    """Persist a model + its vectoriser as a single joblib bundle.

    We save BOTH together because at prediction time you must use the same
    vectoriser the model was trained with — saving them separately invites
    version drift.

    Parameters
    ----------
    model : fitted sklearn estimator
    vectorizer : fitted TfidfVectorizer
    name : str
        Display name; used to build the output filename.
    path : str, default MODELS_DIR
        Output directory.
    label_classes : array-like or None
        The list of class label strings, in the order the model emits them.
        Saved alongside so the CLI / evaluator doesn't have to dig them
        out of the estimator. If None, we read `model.classes_`.

    Returns
    -------
    str — the absolute path the bundle was written to.
    """
    ensure_dir(path)
    filename = f"{_safe_filename(name)}.joblib"
    full_path = os.path.join(path, filename)

    if label_classes is None and hasattr(model, "classes_"):
        label_classes = list(model.classes_)

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "name": name,
        "classes": label_classes,
    }
    joblib.dump(bundle, full_path)
    log_step(f"  saved {name} -> {full_path}")
    return full_path


def load_model(name_or_path: str, path: str = MODELS_DIR) -> Dict[str, Any]:
    """Load a saved model bundle.

    Parameters
    ----------
    name_or_path : str
        Either a display name ("Logistic Regression"), a stem
        ("logistic_regression"), or an absolute / relative path to a
        `.joblib` file. The first two are resolved against `path`.

    Returns
    -------
    dict
        The bundle dict as written by `save_model`. Keys: "model",
        "vectorizer", "name", "classes".
    """
    # Already a path that exists? Use it directly.
    if os.path.isfile(name_or_path):
        full_path = name_or_path
    else:
        stem = _safe_filename(name_or_path)
        # Allow the caller to omit the extension.
        if not stem.endswith(".joblib"):
            stem += ".joblib"
        full_path = os.path.join(path, stem)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No model bundle at {full_path}")

    return joblib.load(full_path)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A tiny end-to-end check: load processed data, sample a few thousand
    # rows, fit the four models, and print accuracies. Useful while
    # iterating on hyperparameters without paying the full pipeline cost.
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from src.utils import PROCESSED_DATA_DIR
    from src.preprocess import preprocess_dataframe
    from src.features import build_tfidf, transform_tfidf

    csv_path = os.path.join(PROCESSED_DATA_DIR, "crisisbench_en.csv")
    df = pd.read_csv(csv_path).sample(15000, random_state=RANDOM_STATE)
    df = preprocess_dataframe(df)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    vec, X_train = build_tfidf(train_df["text_clean"], max_features=5000)
    X_test = transform_tfidf(vec, test_df["text_clean"])

    fitted = train_all_models(X_train, train_df["class_label"])
    for name, info in fitted.items():
        preds = info["model"].predict(X_test)
        acc = accuracy_score(test_df["class_label"], preds)
        print(f"  {name:<22s} acc={acc:.4f}  fit={info['fit_seconds']:.1f}s")
