"""STEP 3: Data visualisation covering Ben Fry's 7 stages:
Acquire -> Parse -> Filter -> Mine -> Represent -> Refine -> Interact.

Stage      | Where it appears
-----------|---------------------------------------------------------------------------
Acquire    | raw CSV load + missing-value profile (fig01)
Parse      | date/duration/list-field parsing (01_preprocess.py) -> content timeline (fig02)
Filter     | duplicates, unrated titles, outliers -> outlier boxplots (fig06)
Mine       | statistics & relations: correlation heatmap, pairplot, genre x maturity (fig07-09, 13)
Represent  | the main charts (fig02-fig12)
Refine     | before/after redesign of a default chart (fig14)
Interact   | Plotly HTML dashboards (figures/interactive_*.html)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from config import *

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})
raw = pd.read_csv(RAW)
df = pd.read_csv(CLEAN)
order = CLASS_NAMES
save = lambda name: (plt.savefig(FIG / name, dpi=150), plt.close())

# fig01 ACQUIRE: missing-value profile of the raw file
miss = raw.isna().mean().mul(100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4.2))
colors = [NETFLIX_RED if v > 0 else "#BBBBBB" for v in miss.values]
ax.barh(miss.index[::-1], miss.values[::-1], color=colors[::-1])
for i, v in enumerate(miss.values[::-1]):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("% missing"); ax.set_title("Raw dataset: missing values per column")
save("fig01_missing_values.png")

# fig02 REPRESENT: type split + additions over time
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1.7]})
tc = df["type"].value_counts()
axs[0].pie(tc, labels=tc.index, autopct="%1.1f%%", colors=[NETFLIX_RED, "#221F1F"], startangle=90,
           wedgeprops=dict(width=0.45, edgecolor="white"), textprops={"color": "black"})
axs[0].set_title("Movies vs TV Shows")
yr = df[df.year_added >= 2008].groupby(["year_added", "type"]).size().unstack(fill_value=0)
yr.plot(kind="bar", stacked=True, ax=axs[1], color=[NETFLIX_RED, "#221F1F"], width=0.8)
axs[1].set_title("Titles added to Netflix per year"); axs[1].set_xlabel("Year added"); axs[1].set_ylabel("Titles")
axs[1].legend(title="")
save("fig02_type_and_additions.png")

# fig03: rating + maturity group
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
rc = df["rating"].value_counts()
axs[0].bar(rc.index, rc.values, color=[PALETTE[df.loc[df.rating == r, "maturity"].iloc[0]] for r in rc.index])
axs[0].set_title("Rating distribution (colour = maturity group)"); axs[0].tick_params(axis="x", rotation=45)
mc = df["maturity"].value_counts().reindex(order)
axs[1].bar(mc.index, mc.values, color=[PALETTE[c] for c in mc.index])
for i, v in enumerate(mc.values): axs[1].text(i, v + 40, f"{v}\n({v/len(df):.0%})", ha="center", fontsize=9)
axs[1].set_title("Target classes"); axs[1].set_ylim(0, mc.max() * 1.2)
save("fig03_rating_maturity.png")

# fig04: top countries
top = df[df.primary_country != "Unknown"].groupby(["primary_country", "type"]).size().unstack(fill_value=0)
top = top.loc[top.sum(axis=1).sort_values(ascending=False).head(10).index][::-1]
ax = top.plot(kind="barh", stacked=True, figsize=(8, 4.5), color=[NETFLIX_RED, "#221F1F"], width=0.75)
ax.set_title("Top 10 producing countries (first listed)"); ax.set_xlabel("Titles"); ax.set_ylabel("")
save("fig04_top_countries.png")

# fig05: release year & lag
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["release_year"], bins=40, color=NETFLIX_RED, ax=axs[0]); axs[0].set_title("Release year (strongly left-skewed)")
sns.histplot(df["lag_years"], bins=30, color="#221F1F", ax=axs[1]); axs[1].set_title("Years between release and Netflix addition")
axs[1].set_xlabel("lag_years")
save("fig05_year_distributions.png")

# fig06 FILTER: outliers (box plots)
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
mv = df[df.is_movie == 1]; tv = df[df.is_movie == 0]
sns.boxplot(data=mv, x="maturity", y="movie_minutes", order=order, palette=PALETTE, ax=axs[0]); axs[0].set_title("Movie length (min) by maturity")
sns.boxplot(data=tv, x="maturity", y="n_seasons", order=order, palette=PALETTE, ax=axs[1]); axs[1].set_title("TV seasons by maturity")
sns.boxplot(data=df, x="maturity", y="n_cast", order=order, palette=PALETTE, ax=axs[2]); axs[2].set_title("Cast size by maturity")
save("fig06_outlier_boxplots.png")

# fig07 MINE: correlation heatmap
nums = ["release_year", "year_added", "lag_years", "movie_minutes", "n_seasons", "n_genres", "n_cast",
        "n_countries", "n_directors", "desc_words", "title_len", "sentiment", "is_movie"]
corr = df[nums].corr()
fig, ax = plt.subplots(figsize=(9, 7.2))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), cmap="RdBu_r", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, linewidths=.5, cbar_kws={"shrink": .75}, ax=ax)
ax.set_title("Correlation heatmap of numeric features")
save("fig07_correlation_heatmap.png")

# fig08 MINE: pairplot
sub = df.sample(1500, random_state=SEED)[["release_year", "lag_years", "n_genres", "desc_words", "sentiment", "maturity"]]
g = sns.pairplot(sub, hue="maturity", hue_order=order, palette=PALETTE, corner=True, plot_kws=dict(s=12, alpha=.5), height=1.7)
g.fig.suptitle("Pairplot of key numeric features by maturity", y=1.02, fontweight="bold")
g.savefig(FIG / "fig08_pairplot.png", dpi=130); plt.close("all")

# fig09 MINE: genre x maturity heatmap
gm = df.assign(genre=df.listed_in.str.split(", ")).explode("genre").reset_index(drop=True)
topg = gm.genre.value_counts().head(18).index
ct = pd.crosstab(gm[gm.genre.isin(topg)].genre, gm[gm.genre.isin(topg)].maturity)[order]
ct = ct.div(ct.sum(axis=1), axis=0).sort_values("Adults")
fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(ct, annot=True, fmt=".0%", cmap="YlOrRd", cbar_kws={"label": "share of titles"}, ax=ax)
ax.set_title("Maturity mix inside the 18 biggest genres"); ax.set_ylabel("")
save("fig09_genre_maturity_heatmap.png")

# fig10 REPRESENT (text): word clouds per maturity group
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
extra = {"life", "one", "two", "new", "must", "will", "back", "find", "finds", "get", "gets"}
from wordcloud import STOPWORDS
import random
HUE = {"Kids": 38, "Teens": 170, "Adults": 355}
for ax, c in zip(axs, order):
    txt = " ".join(df.loc[df.maturity == c, "description"])
    cf = lambda *a, h=HUE[c], **k: f"hsl({h}, 75%, {random.randint(22, 42)}%)"
    wc = WordCloud(width=700, height=450, background_color="white", stopwords=STOPWORDS | extra,
                   color_func=cf, max_words=80, random_state=SEED).generate(txt)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(f"{c}")
fig.suptitle("Most frequent words in descriptions", fontweight="bold")
save("fig10_wordclouds.png")

# fig11 REPRESENT (text): top bigrams per maturity
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, c in zip(axs, order):
    cv = CountVectorizer(ngram_range=(2, 2), stop_words="english", min_df=3)
    M = cv.fit_transform(df.loc[df.maturity == c, "description"])
    freq = pd.Series(np.asarray(M.sum(axis=0)).ravel(), index=cv.get_feature_names_out()).nlargest(10)[::-1]
    ax.barh(freq.index, freq.values, color=PALETTE[c]); ax.set_title(f"{c}: top bigrams")
save("fig11_top_bigrams.png")

# fig12: sentiment distribution
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for c in order: sns.kdeplot(df.loc[df.maturity == c, "sentiment"], label=c, color=PALETTE[c], fill=True, alpha=.25, ax=axs[0])
axs[0].set_title("VADER sentiment of descriptions"); axs[0].legend(); axs[0].set_xlabel("compound score")
sns.violinplot(data=df, x="maturity", y="desc_words", order=order, palette=PALETTE, ax=axs[1]); axs[1].set_title("Description length (words)")
save("fig12_sentiment_length.png")

# fig13: type vs maturity share
fig, ax = plt.subplots(figsize=(6.5, 4))
t = pd.crosstab(df.type, df.maturity, normalize="index")[order] * 100
t.plot(kind="bar", stacked=True, color=[PALETTE[c] for c in order], ax=ax, rot=0)
ax.set_ylabel("% of titles"); ax.set_xlabel(""); ax.set_title("Maturity mix: Movies vs TV Shows"); ax.legend(title="", bbox_to_anchor=(1, 1))
save("fig13_type_vs_maturity.png")

# fig14 REFINE: default chart vs refined chart
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
with plt.style.context("default"):
    pass
axs[0].bar(df.rating.value_counts().index, df.rating.value_counts().values)
axs[0].set_title("BEFORE: default styling", fontweight="normal")
rc = df.rating.value_counts()
axs[1].barh(rc.index[::-1], rc.values[::-1], color=[PALETTE[df.loc[df.rating == r, "maturity"].iloc[0]] for r in rc.index[::-1]])
for i, v in enumerate(rc.values[::-1]): axs[1].text(v + 25, i, f"{v}", va="center", fontsize=8)
axs[1].set_title("AFTER: sorted, labelled, colour-coded by maturity group"); axs[1].set_xlabel("Titles")
axs[0].tick_params(axis="x", rotation=60)
save("fig14_refine_before_after.png")

# INTERACT: Plotly dashboards (hover / zoom / drill-down)
sun = df.groupby(["type", "maturity", "rating"]).size().reset_index(name="titles")
px.sunburst(sun, path=["type", "maturity", "rating"], values="titles", color="maturity",
            color_discrete_map={**PALETTE, "(?)": "#cccccc"}, title="Netflix catalogue: type > maturity > rating (click to drill down)"
            ).write_html(FIG / "interactive_sunburst.html", include_plotlyjs="cdn")
tl = df.groupby(["year_added", "maturity"]).size().reset_index(name="titles")
px.line(tl[tl.year_added >= 2012], x="year_added", y="titles", color="maturity", markers=True, color_discrete_map=PALETTE,
        title="Titles added per year by maturity group (hover/zoom)").write_html(FIG / "interactive_timeline.html", include_plotlyjs="cdn")
sc = df.sample(2500, random_state=SEED)
px.scatter(sc, x="release_year", y="sentiment", color="maturity", color_discrete_map=PALETTE, hover_data=["title", "rating", "listed_in"],
           opacity=.6, title="Release year vs description sentiment (hover for title)").write_html(FIG / "interactive_scatter.html", include_plotlyjs="cdn")
print("figures:", sorted(p.name for p in FIG.iterdir()))
