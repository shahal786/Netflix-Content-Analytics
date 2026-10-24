# Netflix Content Maturity Classification (Data Visualization + Deep Learning)

Predicts whether a Netflix title is for **Kids / Teens / Adults** (grouped from its rating) using metadata and the
plot description. Model: Multi-Layer Perceptron (feed-forward neural network).

## Run
```bash
pip install -r requirements.txt
python src/run_all.py          # about 1-2 minutes on a laptop CPU
```
Dataset: `data/netflix_titles.csv` (Kaggle "Netflix Movies and TV Shows").

## Pipeline (maps to the assignment steps)
| Script | Step | What it does |
|---|---|---|
| `src/01_preprocess.py` | 1-2 | Load, repair shifted rows, duplicates, missing values, parse dates/durations, engineer 17 features, VADER sentiment |
| `src/02_split_features.py` | 2 | Outlier clipping, scaling, one-hot / multi-hot / TF-IDF+SVD encoding, stratified 70/15/15 split (fit on train only) |
| `src/03_visualize.py` | 3 | 14 static figures + 3 interactive Plotly dashboards covering Fry's 7 stages |
| `src/04_train_mlp.py` | 4 | Grid search (24 configs), epoch-by-epoch MLP training with early stopping |
| `src/05_evaluate.py` | 4-5 | Baselines, permutation feature importance, loss/accuracy curves, confusion matrix, ROC-AUC, model comparison |
| `src/06_build_report.py` | 6-7 | Builds `outputs/Netflix_Maturity_DL_Report.pdf` |

## Folders
`data/` raw + cleaned CSV, `figures/` all charts (+ interactive `*.html`), `outputs/` metrics, logs, report, `models/` trained MLP and fitted preprocessors.

Note: the raw file has 12 columns; 17 engineered fields bring it above the 15-feature requirement (155 model inputs after encoding).
