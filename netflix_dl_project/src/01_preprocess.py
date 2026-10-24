"""STEP 1-2a: Load the Netflix dataset, run data cleaning and feature engineering.

Outputs: data/netflix_clean.csv, outputs/cleaning_log.json
"""
import json
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import *

raw = pd.read_csv(RAW)
df = raw.copy()
log = {"raw_shape": list(raw.shape), "raw_columns": list(raw.columns),
       "missing_raw": raw.isna().sum().to_dict()}

# --- 1. Repair shifted rows (duration text was typed into the 'rating' column)
shifted = df["rating"].astype(str).str.contains("min", na=False)
log["shifted_rows_repaired"] = int(shifted.sum())
df.loc[shifted, "duration"] = df.loc[shifted, "rating"]
df.loc[shifted, "rating"] = np.nan

# --- 2. Duplicates
log["exact_duplicates"] = int(df.duplicated().sum())
log["key_duplicates"] = int(df.duplicated(["title", "type", "release_year"]).sum())
df = df.drop_duplicates(["title", "type", "release_year"]).reset_index(drop=True)

# --- 3. Missing values
for c in ["director", "cast", "country"]:
    df[c] = df[c].fillna("Unknown").str.strip()
df["date_added_dt"] = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y", errors="coerce")
log["unparsed_dates"] = int(df["date_added_dt"].isna().sum())
df["year_added"] = df["date_added_dt"].dt.year.fillna(df["release_year"]).astype(int)
df["month_added"] = df["date_added_dt"].dt.month
df["month_added"] = df["month_added"].fillna(df["month_added"].mode()[0]).astype(int)

# --- 4. Parse duration into two numeric fields
dur = df["duration"].str.extract(r"(\d+)")[0].astype(float)
df["movie_minutes"] = np.where(df["type"] == "Movie", dur, 0).astype(float)
df["n_seasons"] = np.where(df["type"] == "TV Show", dur, 0).astype(float)
med = df.loc[(df.type == "Movie") & (df.movie_minutes > 0), "movie_minutes"].median()
df.loc[(df["type"] == "Movie") & ((df["movie_minutes"] == 0) | df["movie_minutes"].isna()), "movie_minutes"] = med
df["n_seasons"] = df["n_seasons"].fillna(0)

# --- 5. Target: maturity group derived from rating
def maturity(r):
    if r in KIDS: return "Kids"
    if r in TEENS: return "Teens"
    if r in ADULTS: return "Adults"
    return np.nan
df["maturity"] = df["rating"].map(maturity)
log["rows_before_target_filter"] = int(len(df))
log["rows_dropped_unrated"] = int(df["maturity"].isna().sum())
df = df.dropna(subset=["maturity"]).reset_index(drop=True)

# --- 6. Feature engineering (derived fields)
split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
df["is_movie"] = (df["type"] == "Movie").astype(int)
df["lag_years"] = (df["year_added"] - df["release_year"]).clip(lower=0)
df["n_genres"] = df["listed_in"].map(lambda s: len(split(s)))
df["n_cast"] = df["cast"].map(lambda s: 0 if s == "Unknown" else len(split(s)))
df["n_countries"] = df["country"].map(lambda s: 0 if s == "Unknown" else len(split(s)))
df["n_directors"] = df["director"].map(lambda s: 0 if s == "Unknown" else len(split(s)))
df["has_director"] = (df["n_directors"] > 0).astype(int)
df["has_cast"] = (df["n_cast"] > 0).astype(int)
df["primary_country"] = df["country"].map(lambda s: split(s)[0] if split(s) else "Unknown")
df["desc_words"] = df["description"].str.split().map(len)
df["desc_chars"] = df["description"].str.len()
df["title_len"] = df["title"].str.len()
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["description"].map(lambda t: sia.polarity_scores(t)["compound"])

# --- 7. Outlier report (IQR rule) - treatment (clipping) is done on the train split only
def iqr_count(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
log["outliers_iqr"] = {
    "movie_minutes (movies only)": iqr_count(df.loc[df.is_movie == 1, "movie_minutes"]),
    "n_seasons (TV only)": iqr_count(df.loc[df.is_movie == 0, "n_seasons"]),
    "n_cast": iqr_count(df["n_cast"]), "lag_years": iqr_count(df["lag_years"]),
    "desc_words": iqr_count(df["desc_words"]),
}
log["final_shape"] = list(df.shape)
log["class_counts"] = df["maturity"].value_counts().to_dict()
log["engineered_features"] = ["is_movie", "lag_years", "year_added", "month_added", "movie_minutes",
    "n_seasons", "n_genres", "n_cast", "n_countries", "n_directors", "has_director", "has_cast",
    "primary_country", "desc_words", "desc_chars", "title_len", "sentiment"]

df.to_csv(CLEAN, index=False)
(OUT / "cleaning_log.json").write_text(json.dumps(log, indent=2, default=str))
print(json.dumps(log, indent=2, default=str))
