"""
src/preprocess.py
=================

Cleans raw tweet text so it can be fed into TF-IDF.

Why preprocessing matters
-------------------------
Raw tweets are noisy: URLs, @mentions, emoji, mixed casing, repeated
whitespace, and "stop words" that appear everywhere ("the", "is", "and"). If
we hand TF-IDF the raw text, the vocabulary explodes and the model wastes
capacity learning that "Help" and "help" mean the same thing. Cleaning is
about *signal density* — removing noise so the remaining tokens carry as much
class-relevant information per byte as possible.

Cleaning pipeline (each step explained inline below):
1. Lowercase
2. Remove URLs
3. Remove @mentions
4. Strip the # symbol from hashtags (keep the word)
5. Remove non-letter characters (numbers, punctuation, emoji)
6. Collapse whitespace
7. Drop English stopwords
8. Lemmatise (reduce words to their base form)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# re — Python's built-in regular expression library.
# A "regex" is a tiny pattern-matching language for text. For example,
# `r"\d+"` matches one or more digits, and re.sub(pattern, repl, text)
# returns `text` with every match replaced by `repl`.
import re

# pandas — used by preprocess_dataframe() to apply clean_text to every row.
# Example: df["text"].apply(clean_text) maps clean_text over a column.
import pandas as pd

# nltk — Natural Language Toolkit. The most popular Python library for
# working with human language. We use it for two things only here:
#   - a curated list of English stopwords (~180 common words to drop)
#   - the WordNet lemmatiser (rewrites words to their dictionary form)
import nltk

# nltk.corpus.stopwords — predefined lists of common, low-information words
# in many languages. We only use the English list. Example contents:
#   {"the", "is", "and", "a", "in", "to", ...}
from nltk.corpus import stopwords

# nltk.stem.WordNetLemmatizer — reduces inflected forms to a base word.
# Example: lemmatize("running")  -> "running"   (default POS=noun)
#          lemmatize("running", pos="v") -> "run"
# We use the default (noun) here, which is the standard simple choice.
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# One-time NLTK setup
#
# NLTK ships *code*, but the language data (stopword lists, WordNet) lives in
# separate downloadable corpora. The first time we use them we need to
# `nltk.download(...)`. We wrap that in a function and call it lazily so the
# module itself can be imported on a brand-new machine without exploding.
# ---------------------------------------------------------------------------

# Resources we need + their lookup paths in the NLTK data registry.
# `find()` raises LookupError if a resource is missing; we catch it and
# download. This is way faster than calling download() unconditionally
# (which checks the network even when the data is already on disk).
_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet":   "corpora/wordnet",
    "omw-1.4":   "corpora/omw-1.4",  # Open Multilingual WordNet — required by
                                     # newer NLTK versions for lemmatiser
}


def _ensure_nltk_data() -> None:
    """Download missing NLTK corpora silently. Idempotent — safe to call on
    every import, but does network work only on first run.
    """
    for name, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            # quiet=True suppresses the verbose progress bar.
            nltk.download(name, quiet=True)


# Run the setup at import time. After this line every public function in the
# module can assume the corpora are available.
_ensure_nltk_data()

# Build the stopword set ONCE (set lookups are O(1); list lookups are O(n)).
# We add a couple of project-specific tokens too: "rt" (Twitter "retweet"
# marker) and "amp" (the `&amp;` HTML entity that survives if we're sloppy).
_STOP_WORDS = set(stopwords.words("english"))
_STOP_WORDS.update({"rt", "amp", "via", "u"})

# A single lemmatiser instance — it's stateless but allocating is cheap; we
# share one to avoid pointless work in tight loops.
_LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
#
# Each call to re.sub() with a *string* pattern recompiles the regex. For text
# we'll be running over 130k+ tweets, so we compile each pattern once at
# import time and reuse the compiled object. This is roughly 5-10× faster.
# ---------------------------------------------------------------------------

# URL pattern: anything starting with http://, https://, or www.
#   r"http\S+"  -> "http" followed by one or more non-whitespace chars
#   r"www\S+"   -> "www"  followed by one or more non-whitespace chars
#   "|" means OR
# Example matches: "https://t.co/abc123", "http://nytimes.com/foo",
#                  "www.redcross.org/donate"
_URL_RE = re.compile(r"http\S+|www\S+", flags=re.IGNORECASE)

# @mention pattern. "@" followed by one or more letters/digits/underscores.
#   \w     = word character ([A-Za-z0-9_])
#   \w+    = one or more word characters
# Example matches: "@FEMA", "@redcross", "@user123"
_MENTION_RE = re.compile(r"@\w+")

# Hashtag pattern. We want to KEEP the word (it's often topical: "earthquake",
# "tsunami") but strip the leading "#". Match a "#" then capture the word.
# In .sub() we substitute the captured group r"\1" — the matched word.
_HASHTAG_RE = re.compile(r"#(\w+)")

# Non-letter characters. After URL/mention/hashtag handling, anything that
# isn't a letter or whitespace is noise: digits, punctuation, emoji,
# symbols. We replace every such character with a single space.
#   [^a-zA-Z\s]  = any character that is NOT a letter and NOT whitespace
_NON_LETTER_RE = re.compile(r"[^a-zA-Z\s]")

# Whitespace collapser: one OR MORE whitespace chars (\s = space, tab,
# newline, …) become a single space. Combined with .strip() at the end
# this guarantees clean single-spaced output.
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Run the full text-cleaning pipeline on one string.

    Steps (each commented in the body):
      1. lowercase
      2. drop URLs
      3. drop @mentions
      4. strip "#" from hashtags but keep the word
      5. drop everything that isn't a letter or whitespace
      6. collapse whitespace + strip leading/trailing space
      7. drop stopwords
      8. lemmatise

    Parameters
    ----------
    text : str
        Raw tweet text. May contain emoji, URLs, mentions, hashtags, etc.
        Example input:
            "RT @FEMA: Help! Buildings collapsed in #Nepal earthquake!!!
             Visit https://t.co/abc for updates."

    Returns
    -------
    str
        Cleaned tweet text. Lowercased, no URLs / mentions / digits / punctuation,
        single-spaced, stopwords removed, words lemmatised.
        Example output:
            "help building collapsed nepal earthquake visit update"
    """
    # Defensive guard: pandas can hand us NaN (a float) where we expected a
    # string. Treat any non-string (including None) as empty.
    if not isinstance(text, str):
        return ""

    # --- Step 1: lowercase ---------------------------------------------------
    # "Help" and "help" carry the same meaning. If we leave casing in place,
    # TF-IDF treats them as separate vocabulary entries, doubling the noise
    # for no benefit. Lowercasing collapses them.
    # Before: "RT @FEMA: Help! Buildings collapsed..."
    # After:  "rt @fema: help! buildings collapsed..."
    text = text.lower()

    # --- Step 2: remove URLs -------------------------------------------------
    # URLs in tweets are usually `t.co` shortlinks — opaque tokens that
    # contribute nothing to category assignment. Drop them entirely.
    # Pattern explained:
    #   http\S+   "http" then one or more non-whitespace chars
    #   www\S+    "www"  then one or more non-whitespace chars
    #   |         OR — match either alternative
    #   re.sub(pat, "", text) replaces every match with the empty string.
    # Before: "...visit https://t.co/abc for updates"
    # After:  "...visit  for updates"
    text = _URL_RE.sub("", text)

    # --- Step 3: remove @mentions --------------------------------------------
    # Usernames are noise: "@FEMA" appearing in a tweet doesn't help us tell
    # whether the tweet is about an injury or a donation. Drop them.
    # Before: "rt @fema: help..."
    # After:  "rt : help..."
    text = _MENTION_RE.sub("", text)

    # --- Step 4: clean hashtags ----------------------------------------------
    # Hashtags are different from mentions: the *word* after the # is often
    # the most informative token in the tweet (#earthquake, #rescue,
    # #nepalquake). We keep the word and only strip the "#".
    # The substitution r"\1" inserts the FIRST capture group from the regex —
    # which we defined as `(\w+)` above, i.e. the hashtag's word.
    # Before: "...help! buildings collapsed #nepal earthquake..."
    # After:  "...help! buildings collapsed nepal earthquake..."
    text = _HASHTAG_RE.sub(r"\1", text)

    # --- Step 5: remove non-letter characters --------------------------------
    # After URL/mention/hashtag handling, we drop everything that isn't a
    # letter or whitespace. That removes:
    #   - numbers ("2015", "5.8") — magnitudes/dates aren't useful per-class
    #   - punctuation (! ? . , : ;)
    #   - emoji and other Unicode symbols
    # We replace with a space (not "") so we don't accidentally glue two
    # words together: "help!buildings" -> "help buildings" (good) rather than
    # "helpbuildings" (bad).
    # Before: "rt : help! buildings collapsed nepal earthquake!!!  for update"
    # After:  "rt   help  buildings collapsed nepal earthquake      for update"
    text = _NON_LETTER_RE.sub(" ", text)

    # --- Step 6: collapse whitespace + strip ---------------------------------
    # Steps 2-5 left holes in the string (multiple spaces). Squash any run
    # of whitespace down to a single space, then strip leading/trailing
    # whitespace with .strip().
    # Before: "rt   help  buildings collapsed nepal earthquake      for update"
    # After:  "rt help buildings collapsed nepal earthquake for update"
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # If the tweet was *only* URLs/mentions/punctuation, we've reduced it to
    # an empty string. Nothing more to do — return early.
    if not text:
        return ""

    # --- Step 7: stopword removal --------------------------------------------
    # "Stopwords" are extremely common English words that appear in nearly
    # every tweet ("the", "is", "and", "a", "in", "to"). Their presence is
    # uninformative — they add columns to TF-IDF without helping the model
    # tell the classes apart, and dilute the IDF weights of more useful
    # words. We drop them.
    # split() with no argument splits on any whitespace, returning a list of
    # tokens. We then keep only tokens that aren't in the stopword set.
    tokens = [t for t in text.split() if t not in _STOP_WORDS]

    # --- Step 8: lemmatisation -----------------------------------------------
    # Lemmatisation reduces a word to its dictionary form ("lemma"):
    #   "buildings" -> "building"
    #   "running"   -> "run"  (when treated as a verb)
    #   "destroyed" -> "destroy"  (verb)
    # This collapses inflected variants of the same concept into a single
    # vocabulary entry, so the model sees "destroy" / "destroyed" / "destroys"
    # as one feature instead of three.
    #
    # We use the default POS (part-of-speech) tag — "n" (noun). The "ideal"
    # approach would tag every word with its POS first, but for short noisy
    # tweets that's diminishing returns: the noun lemmatiser already handles
    # most plurals (the most common inflection in tweets) and the cost is one
    # function call per word.
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    # Reassemble the cleaned tokens with single spaces.
    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    cleaned_column: str = "text_clean",
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Apply clean_text() to every row of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. Must contain `text_column`.
    text_column : str, default "text"
        Name of the column holding raw tweet text.
    cleaned_column : str, default "text_clean"
        Name of the new column to write cleaned text into. The original
        `text_column` is left untouched so we can always re-derive features.
    drop_empty : bool, default True
        Whether to drop rows whose cleaned text is the empty string. Useful
        because tweets that consisted only of URLs/mentions reduce to "" and
        TF-IDF can't do anything with an empty document anyway.

    Returns
    -------
    pandas.DataFrame
        A NEW DataFrame (the original is not mutated) with `cleaned_column`
        added. If drop_empty=True, rows with empty cleaned text are removed
        and the index is reset to 0..N-1.
    """
    # Defensive: copy so callers don't see mutations to their input.
    df = df.copy()

    # `.fillna("")` swaps NaN/None for the empty string before applying.
    # Without this, .apply(clean_text) would still work (clean_text guards
    # against non-strings) but pandas would emit a FutureWarning in some
    # versions. Cleaner to handle it up front.
    df[cleaned_column] = df[text_column].fillna("").apply(clean_text)

    if drop_empty:
        # Boolean mask: True wherever cleaned text is non-empty.
        mask = df[cleaned_column].str.len() > 0
        n_drop = (~mask).sum()
        if n_drop:
            print(f"  preprocess_dataframe: dropping {n_drop:,} rows with empty cleaned text")
        df = df[mask].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Standalone smoke test
#
#     python src/preprocess.py
#
# Loads the processed CSV (if present), cleans the first 5 rows, and prints
# before/after pairs. Useful when iterating on the cleaning rules.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from src.utils import PROCESSED_DATA_DIR

    csv_path = os.path.join(PROCESSED_DATA_DIR, "crisisbench_en.csv")
    if os.path.exists(csv_path):
        df_demo = pd.read_csv(csv_path).head(5)
        df_demo = preprocess_dataframe(df_demo, drop_empty=False)
        for _, row in df_demo.iterrows():
            print("-" * 60)
            print("RAW:    ", row["text"])
            print("CLEAN:  ", row["text_clean"])
    else:
        # No processed CSV yet — show a hand-crafted example so the script
        # still demonstrates something.
        sample = (
            "RT @FEMA: HELP!! Buildings collapsed in #Nepal earthquake. "
            "Visit https://t.co/abc for updates 5.8 magnitude :("
        )
        print("RAW:   ", sample)
        print("CLEAN: ", clean_text(sample))
