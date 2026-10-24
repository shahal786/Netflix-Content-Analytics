"""Builds the final PDF report (outputs/Netflix_Maturity_DL_Report.pdf) from the generated figures and metrics."""
import json
import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image as RLImage, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from config import *

RED, INK, GREY = colors.HexColor("#E50914"), colors.HexColor("#1A1A1A"), colors.HexColor("#F3F3F3")
M = json.load(open(OUT / "final_metrics.json")); CL = json.load(open(OUT / "cleaning_log.json"))
FI = json.load(open(OUT / "feature_info.json")); H = json.load(open(OUT / "training_history.json"))
TUNE = pd.read_csv(OUT / "hyperparameter_search.csv"); CMP = pd.read_csv(OUT / "model_comparison.csv", index_col=0)
DD = np.load(ARRAYS, allow_pickle=True)
IMP = pd.read_csv(OUT / "feature_importance.csv", index_col=0)

base = ParagraphStyle("b", fontName="Helvetica", fontSize=9.6, leading=13.6, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5)
h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=8, spaceAfter=6, alignment=0, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=RED, spaceBefore=6, spaceAfter=3, alignment=0, keepWithNext=1)
cap = ParagraphStyle("cap", parent=base, fontName="Helvetica-Oblique", fontSize=8.4, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=3)
cell = ParagraphStyle("c", parent=base, fontSize=8.3, leading=10.4, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
bul = ParagraphStyle("bul", parent=base, leftIndent=12, bulletIndent=2, spaceAfter=2)
ref = ParagraphStyle("ref", parent=base, leftIndent=22, firstLineIndent=-22, fontSize=9, leading=12, alignment=0, spaceAfter=3)
P = lambda t, s=base: Paragraph(t, s)
B = lambda t: Paragraph(t, bul, bulletText="\u2022")

def table(rows, widths, head=True):
    t = Table([[Paragraph(str(c), cellb if (head and i == 0) else cell) for c in r] for i, r in enumerate(rows)], colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head: st += [("BACKGROUND", (0, 0), (-1, 0), INK)] + [("BACKGROUND", (0, i), (-1, i), GREY) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st)); return t

def fig(name, caption, maxw=15.5, maxh=8.6):
    w, h = Image.open(FIG / name).size; r = min(maxw / (w / 130 * 2.54), maxh / (h / 130 * 2.54), 1e9)
    sc = min(maxw / (w / h * maxh) if False else 1, 1); width = maxw * cm; height = width * h / w
    if height > maxh * cm: height = maxh * cm; width = height * w / h
    return [RLImage(str(FIG / name), width=width, height=height), P(caption, cap)]

def viz(name, no, title, what, why, ins, maxh=8.0):
    head = KeepTogether([P(f"{no}. {title}", h2)] + fig(name, f"Figure {no}: {title}", maxh=maxh))
    return [head, P(f"<b>What it shows:</b> {what}"), P(f"<b>Why created:</b> {why}"), P(f"<b>Insights:</b> {ins}")]

def page(c, d):
    c.saveState(); c.setStrokeColor(INK); c.setLineWidth(1.1)
    c.rect(1.0 * cm, 1.0 * cm, A4[0] - 2.0 * cm, A4[1] - 2.0 * cm); c.restoreState()

doc = BaseDocTemplate(str(OUT / "Netflix_Maturity_DL_Report.pdf"), pagesize=A4, title="Netflix Content Maturity Classification",
                      author="Sudharshan N", leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(1.9 * cm, 1.8 * cm, A4[0] - 3.8 * cm, A4[1] - 3.6 * cm, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page)])
S = []
tm = M["test_metrics_mlp"]; pc = M["per_class"]; cm_ = M["confusion_matrix"]

# ---------------- cover ----------------
tt = ParagraphStyle("tt", parent=base, fontName="Helvetica-Bold", fontSize=25, leading=31, alignment=TA_CENTER, textColor=INK)
st_ = ParagraphStyle("st", parent=base, fontSize=12, leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
S += [Spacer(1, 4.2 * cm), P("DATA VISUALIZATION AND DEEP LEARNING PROJECT", ParagraphStyle("k", parent=st_, fontName="Helvetica-Bold", textColor=RED, fontSize=10.5)), Spacer(1, 0.5 * cm),
      P("Predicting the Maturity Level of Netflix Titles from Metadata and Plot Descriptions", tt), Spacer(1, 0.6 * cm),
      P("An end-to-end study: EDA, 7-stage visualization and a Multi-Layer Perceptron on the Netflix Movies and TV Shows dataset", st_), Spacer(1, 3.2 * cm),
      table([["Submitted by", "Sudharshan N"], ["Roll number", "24AD119"], ["Programme", "B.Tech Artificial Intelligence and Data Science, III Year"],
             ["Institution", "KPR Institute of Engineering and Technology"], ["Dataset", "netflix_titles.csv (8,807 titles, 12 columns)"], ["Model", "Multi-Layer Perceptron (feed-forward neural network)"]],
            [4.2 * cm, 11.4 * cm], head=False), PageBreak()]

# ---------------- summary & step 1 ----------------
S += [P("Executive Summary", h1),
      P(f"This project studies the public Netflix catalogue (8,807 movies and TV shows) and builds a deep learning model that predicts a title's <b>maturity group</b> (Kids, Teens or Adults, derived from its official rating) using only metadata and the plot description, never the rating itself. After cleaning, 8,717 titles were used, split 70/15/15 into train, validation and test sets. A Multi-Layer Perceptron trained on 155 engineered features reached <b>{tm['accuracy']:.1%} test accuracy, macro-F1 of {tm['macro_f1']:.3f} and macro ROC-AUC of {M['macro_auc']:.3f}</b>, ahead of logistic regression, a random forest and a majority-class baseline. Kids content is identified well (AUC {M['auc_per_class']['Kids']:.3f}); separating Teens from Adults is the hard part, accounting for {(cm_[1][2]+cm_[2][1])/(sum(map(sum,cm_))-sum(cm_[i][i] for i in range(3))):.0%} of all errors."),
      P("1. Step 1: Dataset Selection", h1),
      P("<b>Source:</b> the 'Netflix Movies and TV Shows' dataset, published on Kaggle by S. Bansal [1]. It is a mixed dataset: numerical (release_year, duration), categorical (type, country, rating), multi-label (listed_in genres, cast, director) and free text (description), so it supports both tabular and text analysis."),
      P(f"<b>Feature requirement (15+):</b> the raw file has 12 columns, so to meet the 15-feature requirement it was expanded with {len(CL['engineered_features'])} engineered fields (see below), giving 29 data fields before encoding and <b>{FI['n_features_total']} model input features</b> after encoding. This is stated openly because the raw file alone is below 15 columns."),
      table([["Raw column", "Type", "Use in this project"],
             ["show_id, title", "ID / text", "Identifier; title length engineered"], ["type", "Categorical", "Movie vs TV Show flag"],
             ["director, cast, country", "Text lists", "Counts, presence flags, primary country (one-hot)"], ["date_added", "Date string", "Parsed into year_added, month_added, lag_years"],
             ["release_year", "Numeric", "Used directly"], ["rating", "Categorical", "<b>Target source</b> (grouped into 3 classes); not used as an input"],
             ["duration", "Text ('90 min', '2 Seasons')", "Split into movie_minutes and n_seasons"], ["listed_in", "Multi-label text", "Top-25 genres, multi-hot; n_genres"],
             ["description", "Free text", "Length, VADER sentiment, TF-IDF + SVD (100 dims)"]], [4.0 * cm, 3.6 * cm, 8.0 * cm]),
      Spacer(1, 4),
      P("<b>Engineered fields (17):</b> is_movie, lag_years, year_added, month_added, movie_minutes, n_seasons, n_genres, n_cast, n_countries, n_directors, has_director, has_cast, primary_country, desc_words, desc_chars, title_len, sentiment."),
      P("<b>Problem statement and target.</b> Ratings are grouped as <b>Kids</b> (TV-Y, TV-Y7, TV-Y7-FV, TV-G, G), <b>Teens</b> (TV-PG, PG, TV-14, PG-13) and <b>Adults</b> (TV-MA, R, NC-17). Titles marked NR/UR or with no rating (90 rows) were removed as their maturity is unknown. Genre tags such as 'Kids' TV' are legitimate catalogue metadata and are not derived from the rating, so there is no target leakage. The task is therefore a 3-class classification problem.")]

# ---------------- step 2 ----------------
mr = CL["missing_raw"]
S += [P("2. Step 2: Data Understanding and Preprocessing", h1),
      P("<b>Exploratory data analysis</b> started with the structure of the data (8,807 x 12), data types, missing-value profile and class balance (Figures 1 to 3 below). The cleaning steps and their effect are summarised here."),
      table([["Issue", "Finding", "Treatment"],
             ["Missing values", f"director {mr['director']} ({mr['director']/8807:.1%}), cast {mr['cast']} ({mr['cast']/8807:.1%}), country {mr['country']} ({mr['country']/8807:.1%}), date_added {mr['date_added']}, rating {mr['rating']}, duration {mr['duration']}",
              "Text lists filled with 'Unknown' with has_director / has_cast flags kept; missing date_added falls back to release_year; month to the mode"],
             ["Shifted rows", f"{CL['shifted_rows_repaired']} rows had the duration typed into the rating column", "Moved to duration; their rating set to missing"],
             ["Duplicates", f"{CL['exact_duplicates']} exact, {CL['key_duplicates']} on (title, type, release_year)", "None found; check kept in pipeline"],
             ["Unrated titles", f"{CL['rows_dropped_unrated']} titles (NR, UR or missing)", "Dropped (target unknown); 8,717 rows remain"],
             ["Outliers (IQR rule)", "; ".join(f"{k}: {v}" for k, v in CL["outliers_iqr"].items()), "Winsorised at the 1st/99th percentile, bounds learned on the training set only"],
             ["Transformation", "Dates, durations and list fields stored as text", "Parsed to numbers; sentiment via VADER [6]"]], [2.8 * cm, 7.2 * cm, 5.6 * cm]),
      Spacer(1, 4),
      P(f"<b>Encoding and scaling.</b> The binary flags stay 0/1; the 14 numeric features are standardised (z-score) with a scaler fitted on the training set only; the primary country is one-hot encoded (top 12 plus 'Other' = {FI['n_country_onehot']} columns); genres are multi-hot encoded (top {FI['n_genre_multihot']}); descriptions are converted by TF-IDF (1 to 2-grams, 8,000 terms) [7] and compressed with truncated SVD (latent semantic analysis [11]) into {FI['n_text_svd']} dimensions that retain {FI['svd_explained_variance']:.1%} of the TF-IDF variance. Total: {FI['n_features_total']} features."),
      P(f"<b>Splitting.</b> A stratified 70/15/15 split with a fixed seed keeps the class proportions identical in every subset. All transformers were fitted on the training portion and then applied to validation and test data to prevent leakage."),
      table([["Subset", "Rows", "Kids", "Teens", "Adults"],
             ["Train", FI["split"]["train"], FI["class_share_train"][0], FI["class_share_train"][1], FI["class_share_train"][2]],
             ["Validation", FI["split"]["val"], *np.bincount(DD["y_val"]).tolist()], ["Test", FI["split"]["test"], *np.bincount(DD["y_test"]).tolist()]], [3 * cm, 2.5 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm])]

# ---------------- step 3 ----------------
S += [P("3. Step 3: Data Visualization", h1),
      P("The brief requires all 7 stages of visualization. This project follows Ben Fry's seven-stage process [2]; the table maps each stage to concrete work. Fourteen static figures and three interactive Plotly dashboards were produced (more than the 5 required)."),
      table([["Stage", "What was done", "Where"],
             ["1. Acquire", "Loaded netflix_titles.csv; profiled missing values", "Fig. 1"], ["2. Parse", "Parsed dates, durations and list fields into numeric/categorical columns", "01_preprocess.py, Fig. 2"],
             ["3. Filter", "Removed unrated titles; inspected and treated outliers", "Fig. 6"], ["4. Mine", "Correlations, pairwise relations, genre-by-maturity statistics, text n-grams", "Figs. 7 to 9, 11"],
             ["5. Represent", "Chosen chart type per data type: bars, histograms, boxplots, heatmaps, word clouds", "Figs. 2 to 13"], ["6. Refine", "Redesigned a default chart: sorting, labels, colour coding", "Fig. 14"],
             ["7. Interact", "Hover, zoom and drill-down dashboards in Plotly", "figures/interactive_*.html"]], [2.6 * cm, 9.6 * cm, 3.4 * cm])]
V = [
 ("fig01_missing_values.png", "Missing values per column (Acquire)", "The percentage of empty cells in each raw column.", "To decide what needs imputation before any modelling.", "Director is missing in about 30% of titles and cast and country in about 9.4% each; every other column is almost complete. Because absence itself may carry information (documentaries and stand-up specials often have no cast), presence flags were kept alongside the 'Unknown' fill.", 6.0),
 ("fig02_type_and_additions.png", "Content type and additions per year (Parse, Represent)", "A donut of Movies versus TV Shows and a stacked bar of titles added to Netflix each year.", "To understand the catalogue composition and its growth over time.", "Movies make up 69.4% of the catalogue and TV shows 30.6%. Additions rose steeply from 400 titles in 2016 to a peak of 2,010 in 2019, then fell slightly; the file is a 2021 snapshot, so 2021 is a partial year.", 6.2),
 ("fig03_rating_maturity.png", "Rating distribution and target classes", "Counts of each rating, colour-coded by the maturity group, and the resulting class sizes.", "To define the classification target and check for class imbalance.", "TV-MA (3,207) and TV-14 (2,160) dominate. The target is Adults 46.0%, Teens 43.6%, Kids 10.4%: Kids is a clear minority, so accuracy alone would be misleading and macro-F1 is used as the main metric.", 6.0),
 ("fig04_top_countries.png", "Top 10 producing countries", "Title counts by first-listed country, split by Movie and TV Show.", "To see geographic coverage and decide how many countries to one-hot encode.", "The United States leads with 3,171 titles, followed by India (1,004) and the United Kingdom (618); 830 titles have no country. The distribution has a long tail, so only the top 12 countries plus 'Other' were encoded.", 6.4),
 ("fig05_year_distributions.png", "Release year and release-to-addition lag", "Histograms of release_year and of the years between release and arrival on Netflix.", "To check skewness and the recency of the catalogue.", "The catalogue is recent: median release year 2017 and 70.8% of titles released in 2015 or later. The median lag is one year and 55.2% of titles were added within a year of release, but a long tail of classic titles exists (back to 1925).", 5.4),
 ("fig06_outlier_boxplots.png", "Outlier box plots by maturity group (Filter)", "Movie length, number of seasons and cast size by maturity group, with outliers as dots.", "To find extreme values that could destabilise neural network training.", "Median movie length is 69.5 min for Kids versus 103 (Teens) and 98 (Adults); Kids films are visibly shorter. Many extreme points exist (388 movie lengths, 258 season counts, 122 cast sizes by the IQR rule), which motivated winsorising at the 1st/99th percentile.", 5.4),
 ("fig07_correlation_heatmap.png", "Correlation heatmap (Mine)", "Pairwise Pearson correlations between the numeric features.", "To detect redundant features and linear relationships.", "release_year and lag_years are almost perfectly negatively correlated (-0.98), and is_movie, n_seasons and movie_minutes are structurally linked (-0.68 and -0.61) because each applies to one content type only. Beyond these, correlations are weak, so little linear signal exists in the numeric block alone and a non-linear model is justified.", 8.6),
 ("fig08_pairplot.png", "Pairplot of key numeric features (Mine)", "Scatter and distribution plots for six features, coloured by maturity group.", "To see whether the classes separate in the numeric feature space.", "The three classes overlap heavily in every pair. Only sentiment shows a systematic shift, with Kids descriptions leaning positive. Numeric metadata alone therefore cannot separate the classes; genre and text features are needed.", 9.0),
 ("fig09_genre_maturity_heatmap.png", "Genre by maturity heatmap (Mine)", "For the 18 largest genres, the share of titles falling in each maturity group.", "To test which categorical tag is the strongest signal for the target.", "'Kids' TV' is 91% Kids and 'Children and Family Movies' has no Adult titles at all, while Crime TV Shows (75%), Independent Movies (73%), Horror (73%) and Thrillers (69%) are mostly Adults. Genre is the strongest predictor, which the later feature-importance results confirm.", 8.2),
 ("fig10_wordclouds.png", "Word clouds of descriptions by maturity (Represent, text)", "The most frequent words in the descriptions of each group.", "To compare the vocabulary used by each audience group.", "Kids descriptions stress adventure, friend, save, magic, fun and animal; Adults descriptions contain murder, crime, cop, comedian and drug. Words such as family, love and world are common to all groups, so single words are only a partial signal.", 5.0),
 ("fig11_top_bigrams.png", "Top bigrams by maturity group (Mine, text)", "The ten most frequent two-word phrases per group.", "N-grams capture phrases that word clouds miss.", "Kids: 'save day', 'best friends' and 'power rangers'. 'High school' is frequent in both Teens (82) and Adults (55), a lexical overlap that explains much of the Teens-versus-Adults confusion in the model.", 5.4),
 ("fig12_sentiment_length.png", "Sentiment and description length", "VADER compound sentiment distribution (left) and description length in words (right) per group.", "To test whether tone or verbosity differs by audience.", "Mean sentiment falls steadily with maturity: +0.28 for Kids, +0.05 for Teens and -0.08 for Adults. Description length is almost identical across groups (about 24 words), so it carries no signal.", 5.8),
 ("fig13_type_vs_maturity.png", "Maturity mix of Movies versus TV Shows", "The share of each maturity group inside the two content types.", "To check whether content type is related to the target.", "Kids content is a much larger share of TV shows (17.5%) than of movies (7.3%), while Teens and Adults are roughly balanced in both. Content type is therefore a useful but modest feature.", 5.4),
 ("fig14_refine_before_after.png", "Refine: default versus redesigned chart", "The same rating counts drawn with default styling (left) and after refinement (right).", "To demonstrate the Refine stage: sorting, direct labels and meaningful colour.", "Sorting bars, using horizontal orientation for long labels, annotating counts and colouring by maturity group make the chart readable at a glance without a legend.", 5.4),
]
for i, v in enumerate(V, 1): S.extend(viz(v[0], i, v[1], v[2], v[3], v[4], maxh=v[5]))
S += [P("15. Interactive dashboards (Interact)", h2),
      P("Three Plotly HTML files in the figures folder let the viewer hover, zoom and filter: <b>interactive_sunburst.html</b> (click to drill from type to maturity group to rating), <b>interactive_timeline.html</b> (additions per year by maturity group) and <b>interactive_scatter.html</b> (hover to see the title, rating and genres of each point). They open in any web browser and complete the Interact stage.")]

# ---------------- step 4 ----------------
top5 = TUNE.head(5)
S += [P("4. Step 4: Model Building", h1),
      P(f"<b>Model choice.</b> A Multi-Layer Perceptron (MLP) is a feed-forward deep neural network [3] suited to the mixed tabular and text-embedding input. The final network has an input layer of {FI['n_features_total']} features, hidden layer(s) of size {tuple(H['config']['hidden'])} with ReLU activation, and a 3-unit softmax output; it has {M['n_params']:,} trainable parameters. It is trained with the Adam optimiser [4] and cross-entropy loss, with a mini-batch size of 64. The model was implemented with scikit-learn [5]; the network is trained one epoch at a time so that train and validation loss and accuracy can be recorded after each epoch, and training stops early when validation loss has not improved for 20 epochs, restoring the best weights."),
      P("<b>Feature analysis.</b> Features fall into five groups: 14 numeric, 3 binary flags, 13 country indicators, 25 genre indicators and 100 text components. The EDA showed that numeric metadata is weak (Figures 7 and 8), genre is strong (Figure 9) and text and sentiment add complementary tone (Figures 10 to 12)."),
      P(f"<b>Hyperparameter tuning (basic).</b> A grid search of 24 configurations was run over hidden-layer structure ((32), (64), (64, 32), (128, 64)), L2 penalty alpha (0.0001, 0.01, 0.1) and learning rate (0.0003, 0.001), each trained with early stopping and ranked by <b>validation macro-F1</b>. Validation scores ranged from {TUNE.val_macro_f1.min():.3f} to {TUNE.val_macro_f1.max():.3f}. The widest network (128, 64) reached its best validation loss after only 5 to 15 epochs and scored lower on average (macro-F1 0.708) than the single 64-unit layer (0.718), so a small network with stronger regularisation won."),
      table([["Rank", "Hidden layers", "alpha", "Learning rate", "Best epoch", "Val loss", "Val acc", "Val macro-F1"]] +
            [[i + 1, r.hidden, r.alpha, r.lr, r.best_epoch, f"{r.val_loss:.4f}", f"{r.val_acc:.3f}", f"{r.val_macro_f1:.3f}"] for i, r in top5.iterrows()],
            [1.2 * cm, 2.5 * cm, 1.7 * cm, 2.3 * cm, 2.0 * cm, 1.9 * cm, 1.9 * cm, 2.2 * cm]),
      Spacer(1, 3), *fig("fig21_hyperparameter_search.png", "Figure 16: Hyperparameter search heat map (validation macro-F1)", maxh=5.2),
      P(f"<b>Selected configuration:</b> hidden layers {tuple(H['config']['hidden'])}, alpha = {H['config']['alpha']}, learning rate = {H['config']['lr']}; best epoch {M['best_epoch']} of {M['epochs_run']} run.")]
ti = IMP.head(8)
S += [P("<b>Feature importance.</b> Permutation importance was measured on the validation set: each feature group was shuffled 8 times and the drop in macro-F1 recorded. Text components were permuted together as one group, and country indicators as one group. A Random Forest [10] was fitted for comparison."),
      KeepTogether(fig("fig20_feature_importance.png", "Figure 17: Feature importance, MLP permutation (left) and Random Forest (right)", maxh=6.4)),
      P(f"<b>Reading the chart.</b> The 'Kids' TV' genre flag is by far the most important feature (macro-F1 falls by {IMP.loc[chr(103)+'enre=Kids'+chr(39)+' TV','mean_drop']:.3f} when shuffled), followed by movie length ({IMP.loc['movie_minutes','mean_drop']:.3f}), the 'Children and Family Movies' flag ({IMP.loc['genre=Children & Family Movies','mean_drop']:.3f}), the primary country ({IMP.loc['primary_country (one-hot)','mean_drop']:.3f}) and release year ({IMP.loc['release_year','mean_drop']:.3f}). The Random Forest ranks the 100-dimensional text group first with a summed Gini importance of {IMP.loc['description text (TF-IDF+SVD)','rf_importance']:.2f}; this is largely an artefact, because impurity importance is biased towards continuous features with many candidate split points [9] and the sum is over 100 columns. The permutation view is the more trustworthy one here.")]



# ---------------- step 5 ----------------
S += [P("5. Step 5: Model Evaluation Visualization", h1),
      P(f"All numbers below are on the untouched test set of 1,308 titles, unless stated otherwise. Four evaluation charts were required; six are provided (the error-distribution chart is for regression only, so a confidence distribution is used for error analysis instead).")]
EV = [
 ("fig15_loss_vs_epoch.png", 17, "Loss versus epoch", f"Training loss falls from about 0.92 to {H['history']['train_loss'][-1]:.2f} while validation loss flattens near 0.628 and bottoms at epoch {M['best_epoch']}.", "The widening gap shows the usual overfitting pattern. Early stopping at epoch %d keeps the weights with the lowest validation loss and avoids memorising the training set." % M['best_epoch']),
 ("fig16_accuracy_vs_epoch.png", 18, "Accuracy versus epoch", f"At the selected epoch training accuracy is {M['train_acc_at_best']:.1%} and validation accuracy {M['val_acc_at_best']:.1%}.", "A gap of about 6 points confirms mild overfitting, but validation accuracy is stable rather than collapsing, so the model generalises reasonably well."),
 ("fig17_confusion_matrix.png", 19, "Confusion matrix", f"Counts and per-class recall. Kids recall is {pc['Kids']['recall']:.1%} ({cm_[0][0]} of {sum(cm_[0])}), Teens {pc['Teens']['recall']:.1%}, Adults {pc['Adults']['recall']:.1%}.", f"The model almost never confuses Kids with Adults ({cm_[0][2]} and {cm_[2][0]} cases). Most errors, {cm_[1][2]+cm_[2][1]} of {sum(map(sum,cm_))-sum(cm_[i][i] for i in range(3))}, are between Teens and Adults, which are adjacent maturity levels and share genres (drama, comedy) and vocabulary ('high school')."),
 ("fig18_roc_auc.png", 20, "ROC curves and AUC (one-vs-rest)", f"AUC is {M['auc_per_class']['Kids']:.3f} for Kids, {M['auc_per_class']['Adults']:.3f} for Adults and {M['auc_per_class']['Teens']:.3f} for Teens; macro AUC is {M['macro_auc']:.3f}.", "Kids is separated almost perfectly from the rest. Teens is the hardest class because it sits between the other two, which matches the confusion matrix. All curves lie far above the chance diagonal (AUC 0.5)."),
 ("fig19_model_comparison.png", 21, "Model comparison", f"Accuracy, macro-F1 and macro ROC-AUC for the MLP, logistic regression, random forest and a majority-class baseline.", f"The MLP is best on all three metrics ({CMP.loc['MLP (deep learning)','accuracy']:.3f} / {CMP.loc['MLP (deep learning)','macro_f1']:.3f} / {CMP.loc['MLP (deep learning)','roc_auc_macro_ovr']:.3f}) and far above the majority baseline ({CMP.loc['Majority-class baseline','accuracy']:.3f} accuracy). Its lead over logistic regression is only about 1.4 accuracy points, which is within the sampling noise of a 1,308-row test set (roughly 1.3 points), so it should be read as 'at least as good', not as decisive superiority."),
 ("fig22_confidence_distribution.png", 22, "Error analysis: confidence of correct versus wrong predictions", f"The distribution of the model's top predicted probability for correct and incorrect predictions. Mean confidence is {M['mean_conf_correct']:.2f} for correct and {M['mean_conf_wrong']:.2f} for wrong predictions.", "Wrong predictions are made with lower confidence, so low-confidence predictions could be routed for manual review in a real system. The overlap, however, shows the probabilities are only moderately informative."),
]
for f_, no, t, what, ins in EV:
    S.append(KeepTogether([P(f"{no + 1}. {t}", h2)] + fig(f_, f"Figure {no + 1}: {t}", maxh=5.3 if no in (15, 16) else (6.2 if no == 21 else 6.4))))
    S += [P(f"<b>What it shows:</b> {what}"), P(f"<b>What it conveys:</b> {ins}")]
S += [P("<b>Per-class test metrics</b>", h2),
      table([["Class", "Precision", "Recall", "F1-score", "Support"]] + [[c, f"{pc[c]['precision']:.2f}", f"{pc[c]['recall']:.2f}", f"{pc[c]['f1-score']:.2f}", int(pc[c]['support'])] for c in CLASS_NAMES], [3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]),
      Spacer(1, 4),
      table([["Model", "Accuracy", "Macro-F1", "Weighted-F1", "Log loss", "Macro AUC"]] + [[n, f"{r.accuracy:.4f}", f"{r.macro_f1:.4f}", f"{r.weighted_f1:.4f}", ("-" if pd.isna(r.log_loss) else f"{r.log_loss:.4f}"), f"{r.roc_auc_macro_ovr:.4f}"] for n, r in CMP.iterrows()], [4.4 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm])]

# ---------------- step 6 ----------------
S += [P("6. Step 6: Conclusion and Future Scope", h1),
      P("What was learned from the dataset", h2),
      B("The catalogue is movie-heavy (69.4%), recent (median release 2017) and dominated by US and Indian content; additions peaked in 2019."),
      B("The target is imbalanced (Kids only 10.4%), so macro-F1 and per-class AUC are more honest than accuracy."),
      P("Key patterns and findings", h2),
      B("Genre is the dominant signal: 'Kids' TV' is 91% Kids, whereas Crime, Horror and Independent titles are 73 to 75% Adults."),
      B("Movie length is a secondary signal (Kids median 69.5 minutes versus about 100 for others), and sentiment of the description falls steadily from Kids (+0.28) to Adults (-0.08)."),
      B("Numeric metadata is weakly correlated with maturity and two features are nearly redundant (release_year and lag_years, r = -0.98)."),
      P("How the model performed", h2),
      P(f"The MLP achieved {tm['accuracy']:.1%} accuracy, {tm['macro_f1']:.3f} macro-F1 and {M['macro_auc']:.3f} macro AUC on unseen data, compared with 46.0% for always predicting the majority class. It is slightly ahead of logistic regression and the random forest, but the margin is within test-set noise, which suggests the ceiling is set by the information available in the features rather than by model capacity. Kids detection is strong (F1 {pc['Kids']['f1-score']:.2f}); Teens versus Adults is the weak point (F1 {pc['Teens']['f1-score']:.2f} and {pc['Adults']['f1-score']:.2f})."),
      P("Challenges faced", h2),
      B("Raw column count (12) was below the 15-feature requirement, solved by feature engineering."),
      B("Dirty rows (duration text stored in the rating column), heavy missingness in director and cast, and the many-valued list fields needed custom parsing."),
      B("Fast overfitting: the widest networks peaked after only 5 to 15 epochs, which favoured smaller architectures, L2 regularisation and early stopping."),
      B("The rating boundary between Teens and Adults (TV-14 versus TV-MA, PG-13 versus R) is inherently fuzzy and is partly decided by content (violence, language) that a short description does not reveal."),
      P("Future scope and improvements", h2),
      B("Replace TF-IDF + SVD with pre-trained transformer embeddings (for example BERT or sentence-transformers) or fine-tune a transformer on descriptions to capture context beyond word counts."),
      B("Use PyTorch or TensorFlow to add dropout, batch normalisation and learning-rate schedules, and tune with Bayesian optimisation or k-fold cross-validation for tighter confidence intervals."),
      B("Add richer features: cast and director identity embeddings, IMDb or Rotten Tomatoes scores, runtime of episodes, parental-guide keywords, and trailer or poster images (a multimodal model)."),
      B("Treat the problem as ordinal classification (Kids < Teens < Adults) and use class-weighted or focal loss to improve minority-class recall."),
      B("Explain individual predictions with SHAP or LIME and deploy the model as a small FastAPI service that suggests a maturity group for a new title.")]

# ---------------- step 7 ----------------
S += [P("7. Step 7: Literature Review and References", h1),
      P("<b>Visualization practice.</b> Fry [2] frames visualization as a seven-stage pipeline of acquiring, parsing, filtering, mining, representing, refining and interacting with data, and this project is organised around exactly that sequence. Plots were produced with Matplotlib [8], whose 2D graphics model underlies Seaborn and the word-cloud rendering used here."),
      P("<b>Text representation.</b> Term-weighting schemes such as TF-IDF [7] remain a standard way to turn documents into vectors, and latent semantic analysis [11] shows that a truncated singular-value decomposition of the term-document matrix yields a compact semantic space. Both are combined here to feed descriptions into a neural network. For tone, VADER [6] provides a lexicon- and rule-based sentiment score that needs no training data and works well on short texts such as these one-sentence descriptions."),
      P("<b>Deep learning and evaluation.</b> The theory of feed-forward networks, regularisation and early stopping is covered in the standard textbook by Goodfellow et al. [3]; training uses the Adam optimiser [4] through scikit-learn [5]. Random forests [10] serve as a strong tabular baseline. Permutation importance is used rather than impurity-based importance because the latter is known to favour continuous and high-cardinality features [9]."),
      P("References (IEEE style)", h2)]
R = ["S. Bansal, \"Netflix Movies and TV Shows,\" Kaggle dataset, 2021. [Online]. Available: https://www.kaggle.com/datasets/shivamb/netflix-shows",
     "B. Fry, <i>Visualizing Data: Exploring and Explaining Data with the Processing Environment</i>. Sebastopol, CA, USA: O'Reilly Media, 2007.",
     "I. Goodfellow, Y. Bengio, and A. Courville, <i>Deep Learning</i>. Cambridge, MA, USA: MIT Press, 2016.",
     "D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, San Diego, CA, USA, 2015.",
     "F. Pedregosa <i>et al.</i>, \"Scikit-learn: Machine learning in Python,\" <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825-2830, 2011.",
     "C. J. Hutto and E. Gilbert, \"VADER: A parsimonious rule-based model for sentiment analysis of social media text,\" in <i>Proc. 8th Int. AAAI Conf. Weblogs Social Media (ICWSM)</i>, Ann Arbor, MI, USA, 2014, pp. 216-225.",
     "G. Salton and C. Buckley, \"Term-weighting approaches in automatic text retrieval,\" <i>Inf. Process. Manage.</i>, vol. 24, no. 5, pp. 513-523, 1988.",
     "J. D. Hunter, \"Matplotlib: A 2D graphics environment,\" <i>Comput. Sci. Eng.</i>, vol. 9, no. 3, pp. 90-95, 2007.",
     "C. Strobl, A.-L. Boulesteix, A. Zeileis, and T. Augustin, \"Bias in random forest variable importance measures: Illustrations, sources and a solution,\" <i>BMC Bioinformatics</i>, vol. 8, art. no. 25, 2007.",
     "L. Breiman, \"Random forests,\" <i>Mach. Learn.</i>, vol. 45, no. 1, pp. 5-32, 2001.",
     "S. Deerwester, S. T. Dumais, G. W. Furnas, T. K. Landauer, and R. Harshman, \"Indexing by latent semantic analysis,\" <i>J. Amer. Soc. Inf. Sci.</i>, vol. 41, no. 6, pp. 391-407, 1990."]
S += [Paragraph(f"[{i}] {r}", ref) for i, r in enumerate(R, 1)]
doc.build(S); print("report built")
