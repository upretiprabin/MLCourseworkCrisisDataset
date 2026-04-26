"""
src/evaluate.py
===============

Compute classification metrics and produce comparison plots.

What's in here
--------------
- evaluate_model            : returns a dict of metrics for one model
- plot_confusion_matrix     : seaborn heatmap of true vs. predicted classes
- plot_classification_report: heatmap version of sklearn's classification_report
- compare_models            : grouped bar chart of all 4 models on key metrics
- plot_per_class_f1         : horizontal bar chart of F1 per class

The metric definitions and trade-offs are explained in detail inside the
notebooks (`notebooks/03_model_comparison.ipynb`). This module's comments
focus on what the *code* does; for the conceptual story see the notebook.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# os — for assembling figure paths.
import os

# typing — clearer signatures.
from typing import Any, Dict, Iterable, List, Optional

# numpy — basic array ops (np.array, np.unique).
import numpy as np

# pandas — used to build small tables for the classification-report heatmap
# and the comparison chart.
import pandas as pd

# matplotlib — plotting backend. We use plt for figure / axes handling.
import matplotlib.pyplot as plt

# seaborn — for prettier heatmaps and bar charts.
import seaborn as sns

# scikit-learn metrics. Imports kept tight: only what we actually call.
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.utils import FIGURES_DIR, ensure_dir, set_plot_style, log_step


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test,
    y_test,
    model_name: str,
) -> Dict[str, Any]:
    """Score a fitted model on the test split and return a metrics dict.

    Parameters
    ----------
    model : fitted sklearn estimator
        Must have `.predict(X)`.
    X_test : array-like or sparse matrix
        Test feature matrix (already transformed by the fitted vectoriser).
    y_test : array-like of shape (n_samples,)
        True labels for the test set.
    model_name : str
        Display name (recorded in the returned dict).

    Returns
    -------
    dict with keys:
        "name"           -> model_name
        "accuracy"       -> overall accuracy in [0, 1]
        "weighted_f1"    -> F1 averaged across classes, weighted by class size
        "macro_f1"       -> F1 averaged across classes, equal weight per class
        "weighted_precision", "weighted_recall"
        "per_class"      -> list of dicts with precision/recall/F1/support
                            for each class
        "y_pred"         -> predicted labels (np.ndarray)
        "report"         -> sklearn classification_report as a string
    """
    log_step(f"  evaluating {model_name} …")
    y_pred = model.predict(X_test)

    # Top-line numbers.
    acc = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # `precision_recall_fscore_support` with average=None gives one number per
    # class — we'll use that to build the "per_class" breakdown.
    classes = sorted(set(y_test) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )

    per_class = [
        {"class": c, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
        for i, c in enumerate(classes)
    ]

    # Aggregate (weighted-by-support) precision and recall for the summary
    # comparison plot.
    wp, wr, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    return {
        "name": model_name,
        "accuracy": float(acc),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "per_class": per_class,
        "y_pred": np.asarray(y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0, digits=3),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """Lower-snake-case + filesystem-safe filename stem."""
    return name.lower().replace(" ", "_").replace("/", "_")


def plot_confusion_matrix(
    y_true: Iterable,
    y_pred: Iterable,
    labels: List[str],
    model_name: str,
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> str:
    """Plot a confusion matrix as a seaborn heatmap.

    A confusion matrix is a square grid:
      rows    = TRUE class
      columns = PREDICTED class
      cell    = count (or fraction) of test tweets that were truly class R
                and predicted as class C
    The diagonal is correct predictions; off-diagonal cells are mistakes.

    Parameters
    ----------
    y_true, y_pred : array-like
        Same length; true and predicted class labels.
    labels : list of str
        Class labels in the order they should appear on the axes.
    model_name : str
        Used for the figure title and default filename.
    save_path : str or None
        Where to write the PNG. If None, defaults to
        figures/confusion_<model_name>.png.
    normalize : bool, default True
        If True, divide each row by its sum (so cells are proportions of the
        true class). Easier to read with imbalanced data.

    Returns
    -------
    str — the path the figure was saved to.
    """
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        # Row-normalise: each row sums to 1. Use float division. We guard
        # against zero-row classes (which can happen if a class has no test
        # samples) by replacing 0 sums with 1 — those rows will just be 0.
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    # Figure size scales with the number of classes so labels stay readable.
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — {model_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    if save_path is None:
        ensure_dir(FIGURES_DIR)
        save_path = os.path.join(FIGURES_DIR, f"confusion_{_safe_filename(model_name)}.png")

    fig.savefig(save_path)
    plt.close(fig)
    log_step(f"    saved {save_path}")
    return save_path


def plot_classification_report(
    y_true: Iterable,
    y_pred: Iterable,
    labels: List[str],
    model_name: str,
    save_path: Optional[str] = None,
) -> str:
    """Plot per-class precision/recall/F1 as a heatmap.

    The standard text output of sklearn's classification_report is hard to
    eyeball — a heatmap makes it instantly obvious which class is the weak
    link.
    """
    set_plot_style()

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    df = pd.DataFrame(
        {"precision": p, "recall": r, "f1-score": f, "support": s},
        index=labels,
    )
    # Drop "support" from the heatmap (different scale: counts not 0-1).
    plot_df = df[["precision", "recall", "f1-score"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.4)))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Score"},
    )
    ax.set_title(f"Per-class metrics — {model_name}")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    if save_path is None:
        ensure_dir(FIGURES_DIR)
        save_path = os.path.join(FIGURES_DIR, f"clsreport_{_safe_filename(model_name)}.png")

    fig.savefig(save_path)
    plt.close(fig)
    log_step(f"    saved {save_path}")
    return save_path


def compare_models(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
) -> str:
    """Grouped bar chart comparing the four models on accuracy / weighted F1
    / macro F1.

    Parameters
    ----------
    results : dict
        Maps model name to the dict returned by evaluate_model().
    save_path : str or None
        Output PNG path. Defaults to figures/model_comparison.png.

    Returns
    -------
    str — the path the figure was saved to.
    """
    set_plot_style()

    rows = []
    for name, info in results.items():
        rows.append({
            "model": name,
            "Accuracy": info["accuracy"],
            "Weighted F1": info["weighted_f1"],
            "Macro F1": info["macro_f1"],
        })
    df = pd.DataFrame(rows).set_index("model")

    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax, edgecolor="black", width=0.75)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    # Annotate each bar with its value.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.tight_layout()

    if save_path is None:
        ensure_dir(FIGURES_DIR)
        save_path = os.path.join(FIGURES_DIR, "model_comparison.png")

    fig.savefig(save_path)
    plt.close(fig)
    log_step(f"    saved {save_path}")
    return save_path


def plot_per_class_f1(
    y_true: Iterable,
    y_pred: Iterable,
    labels: List[str],
    model_name: str,
    save_path: Optional[str] = None,
) -> str:
    """Horizontal bar chart of F1 score per class — sorted ascending so the
    weakest classes are at the top of the chart and visually obvious.
    """
    set_plot_style()

    _, _, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    df = pd.DataFrame({"label": labels, "f1": f, "support": support})
    df = df.sort_values("f1", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.35)))
    bars = ax.barh(df["label"], df["f1"], color=sns.color_palette("viridis", len(df)))
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_title(f"Per-class F1 — {model_name}")
    # Annotate each bar with its F1 and support count.
    for i, bar in enumerate(bars):
        f1_val = df.loc[i, "f1"]
        sup = int(df.loc[i, "support"])
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{f1_val:.2f}  (n={sup})",
            va="center",
            fontsize=8,
        )
    plt.tight_layout()

    if save_path is None:
        ensure_dir(FIGURES_DIR)
        save_path = os.path.join(FIGURES_DIR, f"per_class_f1_{_safe_filename(model_name)}.png")

    fig.savefig(save_path)
    plt.close(fig)
    log_step(f"    saved {save_path}")
    return save_path


def get_misclassified(
    df_test: pd.DataFrame,
    y_pred,
    text_column: str = "text",
    label_column: str = "class_label",
    n: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return n misclassified test rows for error analysis.

    Parameters
    ----------
    df_test : pandas.DataFrame
        The test split (must have `text_column` and `label_column`).
    y_pred : array-like
        Predicted labels, same length and order as df_test.
    n : int, default 10
        How many examples to return.
    random_state : int
        Seed for the random sample.

    Returns
    -------
    pandas.DataFrame with columns: [text_column, label_column, "predicted"].
    """
    df = df_test[[text_column, label_column]].copy().reset_index(drop=True)
    df["predicted"] = list(y_pred)
    wrong = df[df[label_column] != df["predicted"]]
    if len(wrong) == 0:
        return wrong
    sample_n = min(n, len(wrong))
    return wrong.sample(sample_n, random_state=random_state).reset_index(drop=True)
