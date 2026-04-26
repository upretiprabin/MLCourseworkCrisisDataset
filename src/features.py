"""
src/features.py
===============

Turns cleaned tweet text into numerical features that classical ML models
can consume — using TF-IDF.

A 60-second TF-IDF primer
-------------------------
Machine-learning models can't read words; they only understand numbers. So we
need a function that takes a tweet and returns a list of numbers ("a vector")
that summarises what the tweet is about. TF-IDF is one of the simplest and
most effective ways to do that for short documents like tweets.

It produces ONE NUMBER PER WORD per document, computed in two parts:

    TF (Term Frequency): how often the word appears in THIS tweet
        Example: "help help please"
            "help"   appears 2 times out of 3 tokens -> TF = 2/3 ≈ 0.67
            "please" appears 1 time  out of 3 tokens -> TF = 1/3 ≈ 0.33

    IDF (Inverse Document Frequency): how rare the word is across ALL tweets
        Example (corpus of 1,000 tweets):
            "the"        appears in 900 tweets -> common -> low IDF (~0.10)
            "earthquake" appears in  50 tweets -> rare   -> high IDF (~3.00)

    TF-IDF = TF × IDF
        A word gets a HIGH score only if it is frequent in the current tweet
        AND rare in the corpus overall. Common chatter ("the", "is", "and")
        gets crushed, distinctive words ("collapsed", "evacuation",
        "magnitude") rise to the top.

Worked example with three documents
-----------------------------------
    Doc1: "earthquake destroyed building"
    Doc2: "earthquake rescue team"
    Doc3: "weekend football game"

    Vocabulary: ["earthquake", "destroyed", "building", "rescue", "team",
                 "weekend", "football", "game"]

    "earthquake" appears in 2 / 3 docs -> moderate IDF (~0.41 with sublinear)
    "weekend"    appears in 1 / 3 docs -> high IDF
    "destroyed"  appears in 1 / 3 docs -> high IDF

    Doc1 vector (length 8): [0.4, 0.6, 0.6,  0,   0,   0,   0,   0]
    Doc2 vector (length 8): [0.4,  0,   0,  0.6, 0.6,  0,   0,   0]
    Doc3 vector (length 8): [ 0,   0,   0,   0,   0,  0.6, 0.6, 0.6]

    Notice doc1 and doc2 share a non-zero "earthquake" entry — geometrically
    they point in similar directions, so a model can learn that they are
    related. Doc3 sits in a completely different part of the space.

Why a sparse matrix
-------------------
Across 100K+ tweets we may have 50K+ unique words, but the average tweet
contains maybe 15-20 of them. So a (100K × 50K) dense matrix would be 5
billion mostly-zero floats — wasteful. scikit-learn returns a *sparse* matrix
that stores only the non-zero entries. Same numbers, ~100× less memory.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# numpy — the foundational array library. We only need it for `np.asarray`
# / `np.argsort` when ranking top features.
# Example: np.argsort([0.3, 0.7, 0.1]) returns array([2, 0, 1]) (ascending order)
import numpy as np

# scipy.sparse — sparse matrix module. We use `csr_matrix` only as a type hint
# in the docstring; at runtime sklearn returns it for us.
from scipy.sparse import csr_matrix  # noqa: F401 — referenced in docstrings

# sklearn.feature_extraction.text.TfidfVectorizer — the scikit-learn
# implementation of TF-IDF. Two-step usage:
#   vec = TfidfVectorizer(...)
#   X = vec.fit_transform(corpus)   # learns the vocabulary AND transforms
#   X_new = vec.transform(new_text) # uses the already-learned vocabulary
from sklearn.feature_extraction.text import TfidfVectorizer

# Iterable / Sequence type hints for clearer signatures.
from typing import Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tfidf(
    texts: Iterable[str],
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
) -> Tuple[TfidfVectorizer, "csr_matrix"]:
    """Fit a TF-IDF vectoriser and return both the fitted vectoriser and
    the transformed feature matrix.

    Parameters
    ----------
    texts : iterable of str
        Cleaned tweet strings — typically df["text_clean"] from preprocess.py.
        Example: ["help building collapsed nepal", "donate red cross", ...]
    max_features : int, default 10000
        Vocabulary size cap. Of all the unique tokens we see, keep only the
        top `max_features` by document frequency. Why cap?
          - smaller models train faster and use less memory
          - extremely rare words usually only contribute noise / overfitting
        Example trade-off:
          max_features=1000   -> fast, broad strokes, may miss niche words
          max_features=10000  -> balanced, our default
          max_features=50000  -> slow, may overfit
    ngram_range : tuple of (int, int), default (1, 2)
        (min_n, max_n) — which n-gram lengths to include.
          (1, 1) = unigrams only ("collapsed")
          (1, 2) = unigrams AND bigrams ("collapsed", "building collapsed")
          (1, 3) = also trigrams (rarely worth the cost on tweets)
        Bigrams add valuable phrasal context: "building collapsed" is a much
        sharper signal of infrastructure damage than "building" or
        "collapsed" alone.
    sublinear_tf : bool, default True
        Apply log scaling to the term-frequency component:
          tf -> 1 + log(tf)   (only when tf > 0)
        A word appearing 10 times in a tweet is not 10× as informative as
        appearing once — diminishing returns. Sublinear TF flattens that
        curve and reduces sensitivity to repetition (e.g. "help help help").
    min_df : int, default 2
        Ignore tokens that appear in fewer than `min_df` documents. Words
        that show up in just one tweet are usually typos or one-off
        usernames; they can't generalise. min_df=2 throws those out.

    Returns
    -------
    (vectorizer, X) : tuple
        vectorizer : TfidfVectorizer
            The fitted vectoriser. Save this alongside the model — at
            prediction time you MUST pass new tweets through the SAME
            vectoriser (so they live in the same feature space).
        X : scipy.sparse.csr_matrix of shape (n_documents, n_features)
            Sparse TF-IDF matrix. Each row is a tweet, each column a
            vocabulary entry, each cell a TF-IDF score.

    Notes
    -----
    The return shape is (len(texts), <= max_features). It's "<= max_features"
    because the actual vocabulary size is min(max_features, total_unique_tokens).
    """
    # Build the vectoriser. Each parameter explained in the docstring above.
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        # `lowercase=False` because preprocess.py already lowercased
        # everything; doing it again here would just be wasted work.
        lowercase=False,
        # `token_pattern` defines what counts as a token. The default
        # r"(?u)\b\w\w+\b" matches Unicode words of length ≥2. We keep that
        # — it drops single-letter "tokens" that are usually OCR/typo
        # artefacts. r"(?u)" makes \w Unicode-aware.
        token_pattern=r"(?u)\b\w\w+\b",
    )

    # `.fit_transform(corpus)`:
    #   1) Scans the corpus and learns the vocabulary (which words exist,
    #      what their IDF values are).
    #   2) Returns the transformed sparse matrix in one shot.
    # This is faster than calling .fit(corpus) then .transform(corpus)
    # because step 1 already iterated over the data.
    X = vectorizer.fit_transform(texts)

    return vectorizer, X


def transform_tfidf(vectorizer: TfidfVectorizer, texts: Iterable[str]) -> "csr_matrix":
    """Transform new tweets using an already-fitted vectoriser.

    Use this for the test split, the dev split, and any prediction-time
    text. Crucially we do NOT refit — we want new tweets mapped into the
    same feature space the model was trained on.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        A vectoriser previously returned by build_tfidf().
    texts : iterable of str
        Cleaned tweets to transform.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (len(texts), n_features)
    """
    return vectorizer.transform(texts)


def get_top_features(
    vectorizer: TfidfVectorizer,
    X: "csr_matrix",
    n: int = 20,
) -> List[Tuple[str, float]]:
    """Return the n vocabulary terms with the highest mean TF-IDF score
    across the corpus.

    These are the "loudest" tokens overall — words that appear with high
    weight across many tweets. Not the same as the most discriminative tokens
    *between classes* (the EDA notebook uses chi-squared for that), but a
    useful global summary of what the corpus is about.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        The fitted vectoriser.
    X : scipy.sparse.csr_matrix
        The TF-IDF matrix produced by `vectorizer.transform(...)`.
    n : int, default 20
        How many top terms to return.

    Returns
    -------
    list of (term, mean_score) tuples, sorted by mean_score descending.
        Example: [("earthquake", 0.184), ("flood", 0.151), ("help", 0.139), ...]
    """
    # `X.mean(axis=0)` averages each column down all rows -> shape (1, n_feat).
    # On a sparse matrix it returns a numpy *matrix* (note: matrix, not
    # ndarray), so we convert to a flat 1-D array with np.asarray().ravel().
    mean_scores = np.asarray(X.mean(axis=0)).ravel()

    # `argsort()` returns the indices that would sort the array ascending.
    # We want descending, so we negate the array first (this trick avoids
    # sorting twice).
    top_idx = np.argsort(-mean_scores)[:n]

    # `vectorizer.get_feature_names_out()` returns the vocabulary as a list,
    # in the same order the columns of X are in.
    feature_names = vectorizer.get_feature_names_out()

    # Build the (term, score) pairs and return.
    return [(feature_names[i], float(mean_scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Standalone smoke test
#
#     python src/features.py
#
# Loads a small slice of the processed CSV, runs build_tfidf, prints shape
# and the top 15 features. A quick sanity check that the pipeline works
# end-to-end without having to run the full notebooks.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pandas as pd
    from src.utils import PROCESSED_DATA_DIR
    from src.preprocess import preprocess_dataframe

    csv_path = os.path.join(PROCESSED_DATA_DIR, "crisisbench_en.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Processed CSV not found at {csv_path}. "
            f"Run `python src/data_loader.py` first."
        )

    # Load a 5,000-row sample so the smoke test stays fast.
    df = pd.read_csv(csv_path).sample(5000, random_state=42)
    df = preprocess_dataframe(df)

    print(f"Fitting TF-IDF on {len(df):,} cleaned tweets…")
    vec, X = build_tfidf(df["text_clean"])
    print(f"  matrix shape : {X.shape}")
    print(f"  matrix nnz   : {X.nnz:,} non-zero entries")
    # Density = non-zeros / total cells. For a sparse matrix this is tiny.
    density = X.nnz / (X.shape[0] * X.shape[1])
    print(f"  density      : {density:.5%}")
    print("\nTop 15 features by mean TF-IDF:")
    for term, score in get_top_features(vec, X, n=15):
        print(f"  {term:<25s} {score:.4f}")
