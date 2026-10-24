"""Shared paths and constants for the Netflix maturity-classification project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
for p in (DATA, FIG, OUT, MODELS):
    p.mkdir(exist_ok=True)

RAW = DATA / "netflix_titles.csv"
CLEAN = DATA / "netflix_clean.csv"
ARRAYS = OUT / "model_arrays.npz"

SEED = 42
CLASS_NAMES = ["Kids", "Teens", "Adults"]
KIDS = {"TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G"}
TEENS = {"TV-PG", "PG", "TV-14", "PG-13"}
ADULTS = {"TV-MA", "R", "NC-17"}

NETFLIX_RED = "#E50914"
PALETTE = {"Kids": "#F5B700", "Teens": "#2A9D8F", "Adults": "#E50914"}
