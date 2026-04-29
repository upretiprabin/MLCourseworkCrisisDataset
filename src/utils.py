"""
src/utils.py
============

Shared utility functions and constants used by every other module.

Why this file exists
--------------------
When the same constant (e.g. a random seed) is used in five different files,
copy-pasting it everywhere is error-prone — change it in one place and forget
the others, and your "reproducible" experiment is suddenly not reproducible.
So we keep ALL such shared values in this single module and import from here.

Beginner note: a "module" in Python is just a `.py` file. When you write
    from src.utils import RANDOM_STATE
Python loads `src/utils.py`, runs it once (top to bottom), and gives you back
the named object.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# os — the standard library for talking to the operating system.
# Example: os.makedirs("foo/bar") creates a folder hierarchy.
import os

# datetime — for working with dates and times.
# Example: datetime.now().strftime("%H:%M:%S") gives the current wall-clock time.
from datetime import datetime

# matplotlib — the most popular Python plotting library.
# `pyplot` is its main submodule; we always alias it as `plt` by convention.
# Example: plt.plot([1, 2, 3]); plt.show() draws a tiny line chart.
import matplotlib.pyplot as plt

# seaborn — built on top of matplotlib. It produces prettier statistical
# graphics (heatmaps, box plots, etc.) with much shorter code.
# Example: sns.heatmap(my_matrix) draws a colour-coded grid.
import seaborn as sns


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RANDOM_STATE — a fixed seed used by every random operation in the project
# (train/test split, RandomForest tree initialisation, shuffles, etc.).
# Why fix it? Many ML steps use randomness internally. Without a seed, every
# run gives slightly different numbers and you can never tell if a difference
# is "real" or just noise. Fixing the seed turns randomness into a *repeatable*
# pseudo-random sequence: the same code on the same inputs always produces
# the same outputs. 42 is just a popular convention (Hitchhiker's Guide nod).
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Path constants
#
# Every path in the project is computed *relative to the repo root*, never
# assumed to be the user's current working directory. That way `python
# notebooks/02_eda.ipynb` and `python run_pipeline.py` both find the same
# files regardless of where they were launched from.
# ---------------------------------------------------------------------------

# __file__ is the full path to THIS file (src/utils.py).
# os.path.dirname(...) goes up one folder. We do it twice to land at the
# project root (the folder containing, src/, notebooks/, ...).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Convenience paths — every other module imports these instead of hard-coding.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")


# ---------------------------------------------------------------------------
# Class label metadata
#
# These constants describe the 11 humanitarian categories CrisisBench uses.
# Keeping them in one place means the CLI, the notebooks and the evaluation
# code all agree on names and ordering.
# ---------------------------------------------------------------------------

# The 16 humanitarian class labels actually present in the English subset of
# CrisisBench.
#
# Order is roughly "biggest classes first" — handy for plot legends.
HUMANITARIAN_LABELS = [
    "not_humanitarian",
    "other_relevant_information",
    "donation_and_volunteering",
    "requests_or_needs",
    "sympathy_and_support",
    "infrastructure_and_utilities_damage",
    "affected_individual",
    "caution_and_advice",
    "injured_or_dead_people",
    "disease_related",
    "response_efforts",
    "personal_update",
    "missing_and_found_people",
    "displaced_and_evacuations",
    "physical_landslide",
    "terrorism_related",
]

# LABEL_DISPLAY — used by the CLI to colour predictions.
# Each entry is (human_readable_name, severity_level). Severity drives the
# ANSI colour the CLI prints. We separate it out here so the CLI can stay
# focused on argument parsing.
#
# Severity tiers:
#   critical — life-at-risk: dead, missing, casualties
#   high     — urgent: requests, displacement, evacuation, terrorism
#   medium   — significant but not life-threatening: infrastructure, disease,
#              landslide, affected individuals
#   low      — supportive: donations, advice, response coordination
#   info     — relevant but non-actionable: sympathy, general info, updates
#   none     — off-topic / not humanitarian
LABEL_DISPLAY = {
    "injured_or_dead_people":              ("🔴 Injured or Dead People",        "critical"),
    "missing_and_found_people":            ("🔴 Missing/Found People",          "critical"),
    "terrorism_related":                   ("🔴 Terrorism-Related",             "critical"),
    "requests_or_needs":                   ("🟠 Requests or Urgent Needs",      "high"),
    "displaced_and_evacuations":           ("🟠 Displaced & Evacuations",       "high"),
    "affected_individual":                 ("🟡 Affected Individual",           "medium"),
    "infrastructure_and_utilities_damage": ("🟡 Infrastructure Damage",         "medium"),
    "disease_related":                     ("🟡 Disease-Related",               "medium"),
    "physical_landslide":                  ("🟡 Landslide / Physical Damage",   "medium"),
    "donation_and_volunteering":           ("🟢 Donation & Volunteering",       "low"),
    "caution_and_advice":                  ("🟢 Caution & Advice",              "low"),
    "response_efforts":                    ("🟢 Response Efforts",              "low"),
    "sympathy_and_support":                ("⚪ Sympathy & Support",             "info"),
    "other_relevant_information":          ("⚪ Other Relevant Info",            "info"),
    "personal_update":                     ("⚪ Personal Update",                "info"),
    "not_humanitarian":                    ("⚫ Not Humanitarian",               "none"),
}

# DISASTER_TYPE_MAP — used by EDA 2 to bucket events into broader disaster
# types. Note: CrisisBench event names sometimes include a year prefix and
# may end in "-ontopic" / "-offtopic" (e.g. "2013_queensland_floods-ontopic"),
# so callers should match using substring containment, not exact equality.
DISASTER_TYPE_MAP = {
    "earthquake": [
        "nepal_earthquake", "chile_earthquake", "ecuador_earthquake",
        "mexico_earthquake", "iraq_iran_earthquake",
    ],
    "hurricane": [
        "hurricane_harvey", "hurricane_irma", "hurricane_maria",
        "hurricane_florence", "hurricane_dorian", "sandy_hurricane",
        "typhoon_hagupit", "typhoon_yolanda",
    ],
    "flood": [
        "pakistan_floods", "india_floods", "queensland_floods",
        "alberta_floods", "colorado_floods", "maryland_floods",
        "kerala_floods", "midwest_floods", "sri_lanka_floods",
    ],
    "wildfire": ["california_wildfires", "colorado_wildfires"],
    "other": [
        "boston_bombings", "west_texas", "bangladesh_savar",
        "lac_megantic", "glasgow_helicopter",
    ],
}

# A consistent colour palette so every plot in the project looks the same.
# Seaborn understands hex strings, named colours, or its own palette names.
# "deep" is a colour-blind-friendly default with 10 distinct hues.
# 16 distinct colours so every class can have its own. Seaborn's "deep" only
# defines 10, so we use "tab20" — a 20-colour categorical map from matplotlib —
# and slice it down. tab20 is high-contrast and prints reasonably in B&W too.
COLOR_PALETTE = sns.color_palette("tab20", n_colors=len(HUMANITARIAN_LABELS))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create a directory (and any missing parents) if it doesn't exist.

    Why we need this: Python won't auto-create folders when you save a file.
    `df.to_csv("data/processed/foo.csv")` errors out if `data/processed/`
    doesn't exist yet. We call this helper before every save.

    Parameters
    ----------
    path : str
        The directory to create. Example: "/tmp/foo/bar"

    Returns
    -------
    None — the directory exists when this function returns.
    """
    # exist_ok=True means "no error if the directory is already there".
    # makedirs (plural) creates intermediate directories too — so creating
    # "a/b/c/d" works even if only "a/" exists today.
    os.makedirs(path, exist_ok=True)


def log_step(message: str) -> None:
    """Print a timestamped progress message to the console.

    Useful in long-running scripts so the user can see what is happening
    and roughly how long each stage takes (since each line carries the
    wall-clock time it was emitted at).

    Parameters
    ----------
    message : str
        The message to print. Example: "TF-IDF fit complete (12.4s)"

    Returns
    -------
    None.
    """
    # %H:%M:%S — hours:minutes:seconds in 24-hour format.
    timestamp = datetime.now().strftime("%H:%M:%S")
    # f-strings let you embed Python expressions inside string literals using
    # `{...}`. So f"[{timestamp}] {message}" produces e.g. "[14:32:01] hello".
    print(f"[{timestamp}] {message}")


def set_plot_style() -> None:
    """Configure matplotlib + seaborn with the project-wide style.

    Call this once at the top of every notebook or script that produces
    plots. It sets a clean white-grid background, a readable font size and
    our consistent colour palette, so all figures across the project look
    like they belong to the same report.

    Returns
    -------
    None — global matplotlib/seaborn state is mutated.
    """
    # whitegrid: subtle grey gridlines on a white background — easy on the eyes
    # and prints well for the report.
    sns.set_style("whitegrid")
    # Apply our colour palette as the default for every plot.
    sns.set_palette(COLOR_PALETTE)
    # Tweak a few matplotlib runtime config (rc) parameters.
    #
    # The face/edge colour entries below force every figure (both inline in
    # notebooks AND saved to PNG) to render on an OPAQUE WHITE background.
    # Without this, matplotlib's default for inline output is a transparent
    # PNG, which looks fine in a light-themed viewer but renders as
    # "blacked-out" text on dark IDE / notebook themes (PyCharm dark, VS
    # Code dark, JupyterLab dark) — black axis labels and ticks blend
    # straight into the dark background. Forcing white face + opaque save
    # makes the figures legible everywhere regardless of theme.
    plt.rcParams.update({
        "figure.figsize": (10, 6),     # default size in inches (width, height)
        "axes.titlesize": 14,          # plot title font size
        "axes.labelsize": 12,          # x/y axis label font size
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,             # screen resolution (dots per inch)
        "savefig.dpi": 150,            # higher DPI when saving to PNG
        "savefig.bbox": "tight",       # trim whitespace around saved figures
        # Theme-proofing — keep figures readable on dark IDE/notebook themes.
        "figure.facecolor": "white",   # background OUTSIDE the axes
        "axes.facecolor":   "white",   # background INSIDE the axes
        "savefig.facecolor": "white",  # white when written to PNG
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
        # Make text/axis colours explicit too, so they don't drift to a
        # theme-aware grey if matplotlib later honours a dark stylesheet.
        "text.color":         "#222222",
        "axes.labelcolor":    "#222222",
        "axes.edgecolor":     "#444444",
        "xtick.color":        "#222222",
        "ytick.color":        "#222222",
    })


def map_event_to_disaster_type(event: str) -> str:
    """Bucket a CrisisBench event name into one of {earthquake, hurricane,
    flood, wildfire, other, unknown}.

    The dataset uses messy event names like "2013_queensland_floods-ontopic".
    To group those into broader disaster types for EDA 2 we test whether any
    of the keywords from DISASTER_TYPE_MAP appear *anywhere* in the event
    string (substring match, not exact equality).

    Parameters
    ----------
    event : str
        The raw event name. Example: "2015_nepal_earthquake-ontopic"

    Returns
    -------
    str — one of the keys in DISASTER_TYPE_MAP, or "unknown" if no keyword
    matched. Example: map_event_to_disaster_type("...nepal_earthquake...")
    returns "earthquake".
    """
    # Make matching case-insensitive — defensive in case the dataset ever
    # capitalises event names differently across splits.
    event_lower = event.lower()
    # `.items()` yields (key, value) pairs from a dictionary.
    for disaster_type, keywords in DISASTER_TYPE_MAP.items():
        # `any(...)` returns True as soon as ONE element of the iterable is
        # truthy. We stop scanning the keyword list as soon as we find a hit.
        if any(keyword in event_lower for keyword in keywords):
            return disaster_type
    return "unknown"
