"""
src/data_loader.py
==================

Loads the CrisisBench humanitarian dataset and produces a single English-only
pandas DataFrame the rest of the project can consume.

Why this file exists
--------------------
Every ML project needs a *data layer* — a single, well-defined place that knows
how to find the data and what shape it has. Without it, every notebook ends up
re-implementing path logic, schema checks and filtering, and inconsistencies
creep in. This module is that single source of truth.

Loading strategy
----------------
1. **Local JSON first.** CrisisBench publishes train.json / dev.json / test.json
   files. If those exist in `data/raw/`, we load them directly — fast, offline,
   reproducible.
2. **HuggingFace fallback.** If the local files are missing, we try to download
   the dataset from HuggingFace Hub. This is slower and requires a network
   connection but means the project still works on a fresh clone.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# os — operating-system path helpers. We use os.path.exists / os.path.join.
# Example: os.path.exists("foo.txt") returns True if foo.txt is on disk.
import os

# json — parses .json files into Python lists/dicts.
# Example: json.load(open("a.json")) -> [{"id": 1, ...}, {"id": 2, ...}]
import json

# sys — used only to call sys.exit() if the dataset cannot be located.
# Example: sys.exit(1) terminates the script with a non-zero status code.
import sys

# pandas — a library for working with tabular data (like Excel spreadsheets in
# Python). The main object is the DataFrame: a 2D table with named columns.
# Example: pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
import pandas as pd

# Helpers and constants from our own utils module.
# log_step prints a timestamped message; ensure_dir creates folders on demand;
# RAW_DATA_DIR / PROCESSED_DATA_DIR are absolute paths anchored at repo root.
from src.utils import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ensure_dir,
    log_step,
)


# ---------------------------------------------------------------------------
# Constants specific to this module
# ---------------------------------------------------------------------------

# The three split files CrisisBench distributes. We keep their names in one
# place so the loader can iterate over them without hard-coding paths twice.
SPLIT_FILES = {
    "train": "train.json",
    "dev":   "dev.json",
    "test":  "test.json",
}

# The columns we expect every record to have. If any of these are missing
# we'll raise a clear error rather than failing with a confusing KeyError
# deep in pandas.
EXPECTED_COLUMNS = ["id", "event", "source", "text", "lang", "lang_conf", "class_label"]

# Path where the cleaned, English-only DataFrame is persisted for downstream
# notebooks/scripts. Centralising this means there's only one place to change
# if we ever rename or relocate it.
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "crisisbench_en.csv")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_local_dataset(path: str = RAW_DATA_DIR) -> pd.DataFrame:
    """Load CrisisBench from local JSON files.

    Reads train.json / dev.json / test.json from the given directory,
    concatenates them into a single DataFrame, and adds a "split" column so
    we don't lose track of which split each row came from.

    Parameters
    ----------
    path : str
        Directory containing the three JSON files. Defaults to data/raw/.
        Example: "/Users/me/project/data/raw"

    Returns
    -------
    pandas.DataFrame
        A single table with columns:
        ["id", "event", "source", "text", "lang", "lang_conf",
         "class_label", "split"].
        Example row:
            id=295654395306201088, split="dev",
            event="2013_queensland_floods-ontopic", lang="en",
            class_label="not_humanitarian",
            text="Congrats to all my Liverpool supporting fans..."

    Raises
    ------
    FileNotFoundError
        If none of the expected JSON files exist in `path`.
    """
    # `frames` accumulates one DataFrame per split — we'll concatenate at the end.
    # Doing it this way (instead of growing one DataFrame in a loop) is much
    # faster, because pandas concatenation copies the whole frame each time.
    frames = []

    # `.items()` yields (key, value) pairs from a dictionary.
    # split="train", filename="train.json" on the first iteration, etc.
    for split, filename in SPLIT_FILES.items():
        # os.path.join builds OS-correct paths ("/" on macOS/Linux, "\" on Windows).
        full_path = os.path.join(path, filename)

        # Skip missing files politely — we'll raise below if NONE of them existed.
        if not os.path.exists(full_path):
            log_step(f"  [skip] {full_path} not found")
            continue

        log_step(f"  loading {filename}")
        # Open the file in text mode with UTF-8 encoding (tweets contain emoji
        # and non-ASCII characters), then let json parse it into Python objects.
        with open(full_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        # Convert the list-of-dicts into a DataFrame. Each dict becomes a row,
        # the dict keys become column names.
        df_split = pd.DataFrame(records)

        # Add a "split" column tagged with the source. This lets us honour
        # CrisisBench's pre-built splits later (training the model on "train",
        # evaluating on "test", etc.) without having to re-split ourselves.
        df_split["split"] = split

        frames.append(df_split)
        log_step(f"    -> {len(df_split):,} rows")

    if not frames:
        # If frames is empty, no JSON files were found — make the failure mode
        # obvious rather than returning an empty DataFrame.
        raise FileNotFoundError(
            f"No CrisisBench JSON files found in {path}. "
            f"Expected at least one of: {list(SPLIT_FILES.values())}"
        )

    # pd.concat stacks DataFrames vertically. ignore_index=True renumbers the
    # rows 0..N-1 instead of preserving each frame's original index (which
    # would create duplicates after concatenation).
    df = pd.concat(frames, ignore_index=True)
    return df


def download_dataset() -> pd.DataFrame:
    """Download CrisisBench from HuggingFace as a fallback path.

    Uses the `datasets` library (the official HuggingFace dataset client).
    Beginner note: `datasets` provides a unified, cached interface to thousands
    of public ML datasets — you say which one you want, it handles download,
    caching, and gives you back something DataFrame-like.

    Returns
    -------
    pandas.DataFrame
        Same schema as load_local_dataset(), with a "split" column.

    Raises
    ------
    Exception
        Any error from huggingface_hub (network down, dataset renamed, ...).
        We re-raise so the caller sees the real cause.
    """
    # Imported lazily because (a) loading `datasets` is slow and (b) it adds
    # a hard dependency we'd rather only require when we actually need it.
    from datasets import load_dataset  # type: ignore[import]

    log_step("  downloading from HuggingFace: QCRI/CrisisBench-all-lang [humanitarian]")
    # `name="humanitarian"` selects the humanitarian subset (vs. "informativeness").
    ds = load_dataset("QCRI/CrisisBench-all-lang", name="humanitarian")

    # `ds` is a DatasetDict: one entry per split. Convert each to a DataFrame
    # and tag with the split name, the same way load_local_dataset() does.
    frames = []
    for split_name, split_data in ds.items():
        df_split = split_data.to_pandas()
        df_split["split"] = split_name
        frames.append(df_split)

    return pd.concat(frames, ignore_index=True)


def get_dataset_info(df: pd.DataFrame) -> None:
    """Print basic descriptive statistics about a CrisisBench DataFrame.

    Used by the notebooks and the __main__ block to give a quick sanity check
    that the data was loaded correctly. Prints to stdout — does NOT return.

    Parameters
    ----------
    df : pandas.DataFrame
        Any DataFrame, typically the output of load_local_dataset().

    Returns
    -------
    None.
    """
    print("\n" + "=" * 70)
    print("Dataset summary")
    print("=" * 70)
    # f"{x:,}" inserts comma thousands-separators: 99248 -> "99,248"
    print(f"Total rows:    {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns:       {list(df.columns)}")

    # `df.isna()` produces a same-shape DataFrame of True/False (True where
    # the cell is NaN/None). `.sum()` counts True values per column.
    print("\nNull counts per column:")
    print(df.isna().sum())

    # Show how many rows belong to each split (train/dev/test) — useful to
    # confirm CrisisBench's official 99K/14K/28K split sizes.
    if "split" in df.columns:
        print("\nRows per split:")
        print(df["split"].value_counts())

    # value_counts(normalize=True) gives the share (0..1) instead of the raw count.
    # We multiply by 100 and round to get a tidy percentage.
    print("\nClass distribution (top 15):")
    counts = df["class_label"].value_counts()
    pct = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "pct": pct})
    print(summary.head(15))

    # `df["lang"].value_counts()` for a quick language sanity check.
    if "lang" in df.columns:
        print("\nLanguage breakdown (top 10):")
        print(df["lang"].value_counts().head(10))

    print("=" * 70 + "\n")


def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only English tweets.

    CrisisBench is multilingual but our TF-IDF + lemmatiser pipeline is built
    around English-language preprocessing (English stopwords, WordNet
    lemmatiser). Mixing languages would degrade quality, so we filter early.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have a "lang" column.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing only rows where lang == "en". The original
        is left untouched (filtering with a boolean mask creates a copy).
    """
    # Build a boolean Series: True for "en" rows, False otherwise.
    # `df[mask]` then keeps only the rows where mask is True.
    mask = df["lang"] == "en"
    n_kept = int(mask.sum())
    n_dropped = len(df) - n_kept
    log_step(f"  filter_english: kept {n_kept:,} / dropped {n_dropped:,}")
    # `.reset_index(drop=True)` renumbers the rows 0..N-1. drop=True throws
    # away the old index (we don't need it).
    return df[mask].reset_index(drop=True)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a helpful error if any expected column is missing.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    None — but raises ValueError if validation fails.
    """
    # Set difference: which expected columns are NOT present in the frame?
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}. "
            f"Got columns: {list(df.columns)}"
        )


def load_dataset(prefer_local: bool = True) -> pd.DataFrame:
    """Top-level loader that handles fallback logic.

    Tries local JSON first (fast, offline) and falls back to HuggingFace if
    the local files are missing.

    Parameters
    ----------
    prefer_local : bool
        If True (default), attempt to load from data/raw/ before reaching out
        to HuggingFace. Set to False to force the HuggingFace path (useful
        when refreshing the local copy).

    Returns
    -------
    pandas.DataFrame
        Validated CrisisBench DataFrame with a "split" column.
    """
    if prefer_local:
        try:
            log_step("Loading dataset from local JSON files…")
            df = load_local_dataset()
            validate_schema(df)
            return df
        except FileNotFoundError as err:
            # Local files missing — fall through to HuggingFace.
            log_step(f"  local load failed: {err}")
            log_step("  attempting HuggingFace fallback…")

    df = download_dataset()
    validate_schema(df)
    return df


def save_processed(df: pd.DataFrame, path: str = PROCESSED_CSV_PATH) -> None:
    """Persist the cleaned, English-only DataFrame to disk as CSV.

    CSV is the universal interchange format for tabular data — every notebook,
    spreadsheet and downstream tool understands it. We pick CSV over Parquet
    here because it's human-readable: you can `head` it from the terminal to
    quickly verify what was written.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to write.
    path : str
        Destination file path. Defaults to data/processed/crisisbench_en.csv.

    Returns
    -------
    None.
    """
    # ensure_dir creates intermediate folders if they don't exist yet,
    # because `to_csv` doesn't auto-create parent directories.
    ensure_dir(os.path.dirname(path))
    # index=False stops pandas from writing the row-number index as an extra
    # column. We don't need it on disk — it's just a row counter, not data.
    df.to_csv(path, index=False)
    log_step(f"  wrote {len(df):,} rows to {path}")


# ---------------------------------------------------------------------------
# Standalone entry point
#
# Allows running:
#     python src/data_loader.py
# and getting a quick end-to-end load → filter → summarise → save.
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the data loader end-to-end and write the processed CSV.

    Returns
    -------
    int — POSIX exit status. 0 on success, 1 on failure.
    """
    try:
        log_step("Step 1/4: load raw dataset")
        df = load_dataset(prefer_local=True)

        log_step("Step 2/4: dataset summary (raw)")
        get_dataset_info(df)

        log_step("Step 3/4: filter to English-only tweets")
        df_en = filter_english(df)

        log_step("Step 4/4: save processed CSV")
        save_processed(df_en)

        log_step("Done.")
        return 0
    except Exception as exc:  # noqa: BLE001 — top-level safety net
        # `repr(exc)` shows the exception type and message in one line.
        # This is the script's last line of defence — anything caught here
        # is a bug we want surfaced clearly.
        print(f"FATAL: {exc!r}", file=sys.stderr)
        return 1


# The `if __name__ == "__main__":` idiom: run the code below ONLY when the
# file is executed directly (e.g. `python src/data_loader.py`), NOT when it
# is imported by another module. This stops the CSV save from running just
# because some notebook did `from src.data_loader import load_dataset`.
if __name__ == "__main__":
    sys.exit(main())
