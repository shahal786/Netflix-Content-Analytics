"""STEP 2b: Encode, scale and split the data (70/15/15, stratified).

All fitted transformers (clipping bounds, scaler, genre vocabulary, TF-IDF, SVD) are fitted on the
TRAIN split only, then applied to validation/test, to avoid data leakage.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import *

df = pd.read_csv(CLEAN)
y = df["maturity"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values

idx = np.arange(len(df))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
print("split sizes", len(tr), len(va), len(te))

NUM = ["release_year", "year_added", "month_added", "lag_years", "movie_minutes", "n_seasons",
       "n_genres", "n_cast", "n_countries", "n_directors", "desc_words", "desc_chars",
       "title_len", "sentiment"]
BIN = ["is_movie", "has_director", "has_cast"]

# numeric: winsorise at train 1st/99th percentile, then standardise
lo, hi = df.loc[tr, NUM].quantile(0.01), df.loc[tr, NUM].quantile(0.99)
numeric = df[NUM].clip(lo, hi, axis=1)
scaler = StandardScaler().fit(numeric.loc[tr])
num_arr = scaler.transform(numeric)
bin_arr = df[BIN].values.astype(float)

# categorical: one-hot of top-12 primary countries (+ Other)
top_c = df.loc[tr, "primary_country"].value_counts().head(12).index.tolist()
ctry = df["primary_country"].where(df["primary_country"].isin(top_c), "Other")
ctry_cols = top_c + ["Other"]
ctry_arr = np.column_stack([(ctry == c).astype(float) for c in ctry_cols])

# multi-label: top-25 genres (listed_in) as multi-hot
genres = df["listed_in"].map(lambda s: [g.strip() for g in s.split(",")])
gcount = pd.Series([g for row in genres.iloc[tr] for g in row]).value_counts()
top_g = gcount.head(25).index.tolist()
genre_arr = np.column_stack([genres.map(lambda r, g=g: float(g in r)) for g in top_g])

# text: TF-IDF (1-2 grams) -> 100-d truncated SVD (latent semantic analysis)
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_features=8000, sublinear_tf=True)
T_tr = tfidf.fit_transform(df.loc[tr, "description"])
svd = TruncatedSVD(n_components=100, random_state=SEED).fit(T_tr)
text_arr = svd.transform(tfidf.transform(df["description"]))
print("SVD explained variance:", round(float(svd.explained_variance_ratio_.sum()), 3))

X = np.hstack([num_arr, bin_arr, ctry_arr, genre_arr, text_arr])
names = NUM + BIN + [f"country={c}" for c in ctry_cols] + [f"genre={g}" for g in top_g] \
        + [f"text_svd_{i}" for i in range(100)]
groups = NUM + BIN + ["primary_country (one-hot)"] * len(ctry_cols) + [f"genre={g}" for g in top_g] \
         + ["description text (TF-IDF+SVD)"] * 100
assert X.shape[1] == len(names) == len(groups)

np.savez_compressed(ARRAYS, X_train=X[tr], X_val=X[va], X_test=X[te], y_train=y[tr], y_val=y[va],
                    y_test=y[te], idx_train=tr, idx_val=va, idx_test=te,
                    names=np.array(names), groups=np.array(groups))
joblib.dump({"scaler": scaler, "tfidf": tfidf, "svd": svd, "clip_lo": lo, "clip_hi": hi,
             "countries": ctry_cols, "genres": top_g}, MODELS / "preprocessors.joblib")
info = {"n_features_total": X.shape[1], "n_numeric": len(NUM), "n_binary": len(BIN),
        "n_country_onehot": len(ctry_cols), "n_genre_multihot": len(top_g), "n_text_svd": 100,
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "class_share_train": np.bincount(y[tr]).tolist(), "top_countries": top_c, "top_genres": top_g}
(OUT / "feature_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps({k: v for k, v in info.items() if k not in ("top_countries", "top_genres")}, indent=2))
