"""
cli.py
======

Command-line interface for the disaster-tweet classifier.

Three modes
-----------
1. **Single tweet:**
       python cli.py --tweet "Massive earthquake hits Nepal, buildings down"
2. **Batch:**
       python cli.py --file tweets.csv --output results.csv
   (the input CSV must contain a 'text' column)
3. **Interactive (default):**
       python cli.py
   Type a tweet, get a classification, repeat. Type 'quit' or hit Ctrl-D
   to exit.

Predictions show the top-3 categories with confidence scores plus an ANSI
colour for severity.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# argparse — the standard library's command-line argument parser. You
# declare what flags your script accepts and argparse handles --help
# generation, type conversion, and friendly error messages.
# Example:
#   parser = argparse.ArgumentParser()
#   parser.add_argument("--name", default="world")
#   args = parser.parse_args()
#   print(f"hello, {args.name}")
import argparse

# os, sys — paths and exit codes.
import os
import sys

# typing — for clearer signatures.
from typing import Iterable, List, Optional, Tuple

# numpy — used to softmax the SVM's decision_function() outputs into
# pseudo-probabilities, since LinearSVC has no native predict_proba().
import numpy as np

# pandas — batch mode reads/writes CSVs.
import pandas as pd

# Project modules.
from src.preprocess import clean_text
from src.train import load_model
from src.utils import LABEL_DISPLAY, MODELS_DIR


# ---------------------------------------------------------------------------
# ANSI colour helpers
#
# Terminal text can be coloured by wrapping it in ANSI escape sequences:
#   "\033[31m" turns subsequent output red, "\033[0m" resets it.
# We map our severity tiers (defined in utils.LABEL_DISPLAY) to colour codes.
# ---------------------------------------------------------------------------

ANSI = {
    "critical": "\033[91m",  # bright red
    "high":     "\033[93m",  # bright yellow
    "medium":   "\033[33m",  # dark yellow
    "low":      "\033[92m",  # bright green
    "info":     "\033[37m",  # light grey
    "none":     "\033[90m",  # dark grey
    "reset":    "\033[0m",
    "bold":     "\033[1m",
}


def _colorise(label: str) -> str:
    """Return a colourised, human-friendly version of a class label.

    Parameters
    ----------
    label : str
        Raw class label like "injured_or_dead_people".

    Returns
    -------
    str — terminal-ready string (with ANSI escapes) such as
    "\\033[91m🔴 Injured or Dead People\\033[0m".
    """
    display, severity = LABEL_DISPLAY.get(label, (label, "info"))
    color = ANSI.get(severity, ANSI["info"])
    return f"{color}{display}{ANSI['reset']}"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _scores_for(model, X) -> Tuple[np.ndarray, List[str]]:
    """Return (score_matrix, class_labels) where higher = more confident.

    Tries `predict_proba` first (Naive Bayes, Logistic Regression, Random
    Forest). Falls back to a softmax over `decision_function` for LinearSVC,
    which has no probability output.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : sparse matrix of shape (n_samples, n_features)

    Returns
    -------
    (np.ndarray of shape (n_samples, n_classes), list of class names)
    """
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X), classes

    # decision_function: distance from each class boundary. Convert to
    # pseudo-probabilities with a softmax so the values are comparable.
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # decision_function on binary problems returns a 1-D array; for
        # multiclass it's already 2-D. Reshape for the unlikely binary case
        # so the rest of the code can assume 2-D.
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        # Numerically stable softmax: subtract row max before exp.
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True), classes

    # Last resort: one-hot the prediction.
    preds = model.predict(X)
    out = np.zeros((len(preds), len(classes)))
    cls_index = {c: i for i, c in enumerate(classes)}
    for i, p in enumerate(preds):
        out[i, cls_index[p]] = 1.0
    return out, classes


def _topk(score_row: np.ndarray, classes: List[str], k: int = 3) -> List[Tuple[str, float]]:
    """Return the top-k (class, confidence) pairs from one row of scores."""
    idx = np.argsort(-score_row)[:k]
    return [(classes[i], float(score_row[i])) for i in idx]


def predict_tweets(model, vectorizer, tweets: Iterable[str]):
    """Run full inference on an iterable of raw tweet strings.

    Returns
    -------
    list of dicts — one per input tweet, with keys:
        "raw"       -> original text
        "cleaned"   -> text after preprocess.clean_text()
        "top_label" -> highest-scoring class label
        "topk"      -> list of (label, confidence) pairs (length 3)
    """
    raw = list(tweets)
    cleaned = [clean_text(t) for t in raw]

    # Empty-after-clean tweets: vectoriser still works, but the prediction
    # won't be meaningful. We let it through and rely on the caller to
    # interpret a low-confidence result.
    X = vectorizer.transform(cleaned)
    scores, classes = _scores_for(model, X)

    out = []
    for i, txt in enumerate(raw):
        top = _topk(scores[i], classes, k=3)
        out.append({
            "raw": txt,
            "cleaned": cleaned[i],
            "top_label": top[0][0],
            "topk": top,
        })
    return out


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def _print_prediction(result: dict, show_cleaned: bool = False) -> None:
    """Pretty-print a single prediction to stdout."""
    print(f"\n{ANSI['bold']}Tweet:{ANSI['reset']} {result['raw']}")
    if show_cleaned:
        print(f"  cleaned: {result['cleaned'] or '(empty after cleaning)'}")
    print("  top 3:")
    for label, conf in result["topk"]:
        bar_len = max(1, int(conf * 30))
        bar = "█" * bar_len
        print(f"    {_colorise(label):<55s}  {conf:>6.1%}  {bar}")


def run_single(model, vectorizer, tweet: str) -> int:
    """--tweet mode."""
    res = predict_tweets(model, vectorizer, [tweet])[0]
    _print_prediction(res, show_cleaned=True)
    return 0


def run_batch(model, vectorizer, in_path: str, out_path: str) -> int:
    """--file / --output mode."""
    if not os.path.exists(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(in_path)
    if "text" not in df.columns:
        print("Input CSV must contain a 'text' column.", file=sys.stderr)
        return 1

    print(f"Classifying {len(df):,} tweets from {in_path} …")
    results = predict_tweets(model, vectorizer, df["text"].fillna(""))

    df_out = df.copy()
    df_out["predicted"] = [r["top_label"] for r in results]
    df_out["confidence"] = [r["topk"][0][1] for r in results]
    df_out["top2"] = [r["topk"][1][0] for r in results]
    df_out["top3"] = [r["topk"][2][0] for r in results]

    df_out.to_csv(out_path, index=False)
    print(f"Wrote {len(df_out):,} predictions to {out_path}")
    return 0


def run_interactive(model, vectorizer) -> int:
    """Default REPL mode."""
    print(f"\n{ANSI['bold']}Disaster tweet classifier{ANSI['reset']} — interactive mode")
    print("Type a tweet and hit Enter. Type 'quit' or 'exit' to leave (or Ctrl-D).\n")
    while True:
        try:
            tweet = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not tweet:
            continue
        if tweet.lower() in {"quit", "exit", ":q"}:
            return 0

        res = predict_tweets(model, vectorizer, [tweet])[0]
        _print_prediction(res)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify disaster-related tweets into humanitarian categories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cli.py --tweet 'Bridge collapsed in Nepal, many trapped'\n"
            "  python cli.py --file tweets.csv --output results.csv\n"
            "  python cli.py                   # interactive\n"
        ),
    )
    parser.add_argument(
        "--tweet", type=str,
        help="Classify a single tweet given on the command line.",
    )
    parser.add_argument(
        "--file", type=str,
        help="Path to a CSV containing a 'text' column (one tweet per row).",
    )
    parser.add_argument(
        "--output", type=str, default="predictions.csv",
        help="Where to write batch predictions. Used with --file. Default: predictions.csv",
    )
    parser.add_argument(
        "--model", type=str, default="best_model",
        help="Model bundle to load. Either a name ('Logistic Regression'), a "
             "stem ('logistic_regression'), or a path to a .joblib file. "
             "Default: best_model",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Load the model once. If the requested model isn't present, point the
    # user at run_pipeline.py.
    try:
        bundle = load_model(args.model, path=MODELS_DIR)
    except FileNotFoundError as err:
        print(f"{err}", file=sys.stderr)
        print(
            "Hint: run `python run_pipeline.py` to train and save the models.",
            file=sys.stderr,
        )
        return 1

    model = bundle["model"]
    vectorizer = bundle["vectorizer"]

    if args.tweet:
        return run_single(model, vectorizer, args.tweet)
    if args.file:
        return run_batch(model, vectorizer, args.file, args.output)
    return run_interactive(model, vectorizer)


if __name__ == "__main__":
    sys.exit(main())
