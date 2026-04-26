"""
run_pipeline.py
===============

Drives the full project end-to-end:

    load → preprocess → TF-IDF → train all 4 models → evaluate → save best

Run from the project root:

    python run_pipeline.py

What "best" means
-----------------
We pick the model with the highest **weighted F1** on the test split.
Weighted F1 is our primary metric because it punishes models that ignore
small classes — see the long discussion in
`notebooks/03_model_comparison.ipynb` for the full reasoning.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# os — file paths.
import os

# sys — to set the script's exit code.
import sys

# time — wall-clock duration of the whole pipeline.
import time

# pandas — already familiar; loads the processed CSV.
import pandas as pd

# Project modules. We import via the `src.` package path so this script can
# be run from the project root with no PYTHONPATH tweaks.
from src.data_loader import load_dataset, filter_english, save_processed, PROCESSED_CSV_PATH
from src.preprocess import preprocess_dataframe
from src.features import build_tfidf, transform_tfidf
from src.train import train_all_models, save_model
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_classification_report,
    plot_per_class_f1,
    compare_models,
)
from src.utils import (
    HUMANITARIAN_LABELS,
    MODELS_DIR,
    ensure_dir,
    log_step,
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    overall_start = time.time()
    ensure_dir(MODELS_DIR)

    # ------------------------------------------------------------------
    # Step 1: data
    #
    # We try to skip the JSON parse if a fresh CSV is already on disk
    # (much faster). Otherwise: load + filter + persist.
    # ------------------------------------------------------------------
    log_step("STEP 1 — load data")
    if os.path.exists(PROCESSED_CSV_PATH):
        log_step(f"  reusing existing processed CSV: {PROCESSED_CSV_PATH}")
        df = pd.read_csv(PROCESSED_CSV_PATH)
    else:
        df_raw = load_dataset(prefer_local=True)
        df = filter_english(df_raw)
        save_processed(df)

    # ------------------------------------------------------------------
    # Step 2: preprocess text
    # ------------------------------------------------------------------
    log_step("STEP 2 — clean text")
    df = preprocess_dataframe(df)
    log_step(f"  {len(df):,} tweets ready after cleaning")

    # ------------------------------------------------------------------
    # Step 3: split using CrisisBench's official train/dev/test
    # ------------------------------------------------------------------
    log_step("STEP 3 — split")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)
    log_step(f"  train={len(train_df):,}  test={len(test_df):,}")

    y_train = train_df["class_label"].values
    y_test = test_df["class_label"].values

    # ------------------------------------------------------------------
    # Step 4: TF-IDF
    # ------------------------------------------------------------------
    log_step("STEP 4 — TF-IDF")
    t0 = time.time()
    vec, X_train = build_tfidf(train_df["text_clean"])
    X_test = transform_tfidf(vec, test_df["text_clean"])
    log_step(f"  vocab={len(vec.get_feature_names_out()):,}  X_train={X_train.shape}  X_test={X_test.shape}  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Step 5: train all four
    # ------------------------------------------------------------------
    log_step("STEP 5 — train")
    fitted = train_all_models(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 6: evaluate + plots
    # ------------------------------------------------------------------
    log_step("STEP 6 — evaluate")
    # Use the actual labels present in the test set (in case a class never
    # appears in the test split, plotting would otherwise create empty rows).
    labels_present = sorted(set(y_train) | set(y_test))

    results = {}
    for name, info in fitted.items():
        eval_out = evaluate_model(info["model"], X_test, y_test, name)
        plot_confusion_matrix(y_test, eval_out["y_pred"], labels_present, name)
        plot_classification_report(y_test, eval_out["y_pred"], labels_present, name)
        results[name] = eval_out

    compare_models(results)

    # ------------------------------------------------------------------
    # Step 7: pick winner by weighted F1, save it as best_model.joblib
    # ------------------------------------------------------------------
    log_step("STEP 7 — choose best model (by weighted F1) and save")
    best_name = max(results, key=lambda n: results[n]["weighted_f1"])
    best_model = fitted[best_name]["model"]
    best_eval = results[best_name]

    # Per-class F1 chart for the winner.
    plot_per_class_f1(y_test, best_eval["y_pred"], labels_present, best_name)

    # Save the winner under both its own filename AND as best_model.joblib so
    # the CLI can rely on a fixed location.
    save_model(best_model, vec, best_name)
    save_model(best_model, vec, "best_model")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Pipeline summary")
    print("=" * 70)
    print(f"{'Model':<22s} {'Acc':>7s} {'WeightedF1':>11s} {'MacroF1':>9s} {'Fit (s)':>9s}")
    print("-" * 70)
    for name, eval_out in results.items():
        marker = " <-- best" if name == best_name else ""
        print(
            f"{name:<22s} "
            f"{eval_out['accuracy']:>7.4f} "
            f"{eval_out['weighted_f1']:>11.4f} "
            f"{eval_out['macro_f1']:>9.4f} "
            f"{fitted[name]['fit_seconds']:>9.1f}"
            f"{marker}"
        )
    print("=" * 70)
    print(f"Total wall time: {time.time() - overall_start:.1f}s")
    print(f"Best model      : {best_name}")
    print(f"Saved to        : {os.path.join(MODELS_DIR, 'best_model.joblib')}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
