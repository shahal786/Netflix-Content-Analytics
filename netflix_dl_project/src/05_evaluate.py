"""STEP 4b + 5: Feature importance, baselines and evaluation visualisations on the held-out TEST set."""
import json, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix, f1_score,
                             log_loss, precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from config import *
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})

d = np.load(ARRAYS, allow_pickle=True)
Xtr, Xva, Xte = d["X_train"], d["X_val"], d["X_test"]
ytr, yva, yte = d["y_train"], d["y_val"], d["y_test"]
names, groups = d["names"], d["groups"]
mlp = joblib.load(MODELS / "mlp_best.joblib")
H = json.load(open(OUT / "training_history.json"))
hist = H["history"]; ep = np.arange(1, len(hist["val_loss"]) + 1); be = H["best_epoch"]

# ---------- baselines ----------
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
majority = np.bincount(ytr).argmax()
res = {}
for nm, m in [("MLP (deep learning)", mlp), ("Logistic Regression", lr), ("Random Forest", rf)]:
    p = m.predict_proba(Xte); yp = p.argmax(1)
    res[nm] = dict(accuracy=accuracy_score(yte, yp), macro_f1=f1_score(yte, yp, average="macro"),
                   weighted_f1=f1_score(yte, yp, average="weighted"), log_loss=log_loss(yte, p),
                   roc_auc_macro_ovr=roc_auc_score(yte, p, multi_class="ovr", average="macro"))
res["Majority-class baseline"] = dict(accuracy=float((yte == majority).mean()), macro_f1=f1_score(yte, np.full_like(yte, majority), average="macro"),
                                     weighted_f1=f1_score(yte, np.full_like(yte, majority), average="weighted"), log_loss=np.nan, roc_auc_macro_ovr=0.5)
ptest = mlp.predict_proba(Xte); ypred = ptest.argmax(1)
res_df = pd.DataFrame(res).T.round(4); res_df.to_csv(OUT / "model_comparison.csv"); print(res_df)
rep = classification_report(yte, ypred, target_names=CLASS_NAMES, output_dict=True)
print(classification_report(yte, ypred, target_names=CLASS_NAMES))
cm = confusion_matrix(yte, ypred)

# ---------- fig15 loss vs epoch ----------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(ep, hist["train_loss"], color="#221F1F", lw=2, label="Training loss")
ax.plot(ep, hist["val_loss"], color=NETFLIX_RED, lw=2, label="Validation loss")
ax.axvline(be, ls="--", color="grey"); ax.annotate(f"best epoch = {be}\n(early stopping)", (be, hist["val_loss"][be - 1]),
    xytext=(be + 4, max(hist["val_loss"]) * 0.97), arrowprops=dict(arrowstyle="->", color="grey"), fontsize=9)
ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss vs Epoch"); ax.legend()
plt.savefig(FIG / "fig15_loss_vs_epoch.png", dpi=150); plt.close()

# ---------- fig16 accuracy vs epoch ----------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(ep, hist["train_acc"], color="#221F1F", lw=2, label="Training accuracy")
ax.plot(ep, hist["val_acc"], color=NETFLIX_RED, lw=2, label="Validation accuracy")
ax.axvline(be, ls="--", color="grey"); ax.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy vs Epoch"); ax.legend()
plt.savefig(FIG / "fig16_accuracy_vs_epoch.png", dpi=150); plt.close()

# ---------- fig17 confusion matrices ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axs[0], cbar=False)
axs[0].set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (counts)")
sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".0%", cmap="Reds", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axs[1], cbar=False)
axs[1].set(xlabel="Predicted", ylabel="Actual", title="Row-normalised (recall per class)")
plt.savefig(FIG / "fig17_confusion_matrix.png", dpi=150); plt.close()

# ---------- fig18 ROC / AUC (one-vs-rest) ----------
Y = label_binarize(yte, classes=[0, 1, 2]); fig, ax = plt.subplots(figsize=(6.5, 5.2)); aucs = {}
for i, c in enumerate(CLASS_NAMES):
    f, t, _ = roc_curve(Y[:, i], ptest[:, i]); aucs[c] = auc(f, t)
    ax.plot(f, t, lw=2.2, color=PALETTE[c], label=f"{c} (AUC = {aucs[c]:.3f})")
macro = roc_auc_score(yte, ptest, multi_class="ovr", average="macro")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.500)")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC curves, one-vs-rest (macro AUC = {macro:.3f})"); ax.legend(loc="lower right")
plt.savefig(FIG / "fig18_roc_auc.png", dpi=150); plt.close()

# ---------- fig19 model comparison ----------
cmp = res_df[["accuracy", "macro_f1", "roc_auc_macro_ovr"]]
ax = cmp.plot(kind="bar", figsize=(9, 4.4), color=["#221F1F", NETFLIX_RED, "#F5B700"], width=.75, rot=15)
for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
ax.set_ylim(0, 1.08); ax.set_title("Test-set comparison of models"); ax.legend(["Accuracy", "Macro-F1", "Macro ROC-AUC"], loc="lower right")
plt.savefig(FIG / "fig19_model_comparison.png", dpi=150); plt.close()

# ---------- feature importance: permutation (MLP, grouped) + Random Forest impurity ----------
rng = np.random.default_rng(SEED); base = f1_score(yva, mlp.predict(Xva), average="macro")
ug = list(dict.fromkeys(groups.tolist())); imp = {}
for g in ug:
    cols = np.where(groups == g)[0]; drops = []
    for _ in range(8):
        Xp = Xva.copy(); perm = rng.permutation(len(Xp)); Xp[:, cols] = Xva[perm][:, cols]
        drops.append(base - f1_score(yva, mlp.predict(Xp), average="macro"))
    imp[g] = (np.mean(drops), np.std(drops))
pi = pd.DataFrame(imp, index=["mean_drop", "std"]).T.sort_values("mean_drop", ascending=False)
rfi = pd.Series(rf.feature_importances_, index=groups).groupby(level=0).sum()
pi["rf_importance"] = rfi.reindex(pi.index); pi.to_csv(OUT / "feature_importance.csv")
top = pi.head(15)[::-1]
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
axs[0].barh(top.index, top.mean_drop, xerr=top["std"], color=NETFLIX_RED, ecolor="#555")
axs[0].set(xlabel="Drop in validation macro-F1 when shuffled", title="Permutation importance (MLP, grouped features)")
t2 = pi.sort_values("rf_importance", ascending=False).head(15)[::-1]
axs[1].barh(t2.index, t2.rf_importance, color="#221F1F"); axs[1].set(xlabel="Summed Gini importance", title="Random Forest importance (same groups)")
plt.tight_layout(); plt.savefig(FIG / "fig20_feature_importance.png", dpi=150); plt.close()

# ---------- fig21 hyper-parameter search ----------
tn = pd.read_csv(OUT / "hyperparameter_search.csv")
piv = tn.pivot_table(index="hidden", columns=["alpha", "lr"], values="val_macro_f1")
fig, ax = plt.subplots(figsize=(9, 3.6)); sns.heatmap(piv, annot=True, fmt=".3f", cmap="Reds", ax=ax, cbar_kws={"label": "val macro-F1"})
ax.set_title("Hyper-parameter search (hidden layers x [L2 alpha, learning rate])"); ax.set_xlabel("(alpha, learning rate)")
plt.savefig(FIG / "fig21_hyperparameter_search.png", dpi=150); plt.close()

# ---------- fig22 error analysis: confidence of correct vs wrong predictions ----------
conf = ptest.max(1); ok = ypred == yte
fig, ax = plt.subplots(figsize=(7.5, 4.2))
sns.kdeplot(conf[ok], fill=True, color="#2A9D8F", alpha=.4, label=f"Correct ({ok.sum()})", ax=ax)
sns.kdeplot(conf[~ok], fill=True, color=NETFLIX_RED, alpha=.4, label=f"Wrong ({(~ok).sum()})", ax=ax)
ax.set(xlabel="Model confidence (max softmax probability)", title="Confidence distribution: correct vs wrong"); ax.legend()
plt.savefig(FIG / "fig22_confidence_distribution.png", dpi=150); plt.close()

out = {"test_metrics_mlp": res["MLP (deep learning)"], "per_class": {c: rep[c] for c in CLASS_NAMES}, "confusion_matrix": cm.tolist(),
       "auc_per_class": aucs, "macro_auc": macro, "n_params": H["n_params"], "best_epoch": be, "epochs_run": H["epochs_run"],
       "config": H["config"], "train_acc_at_best": hist["train_acc"][be - 1], "val_acc_at_best": hist["val_acc"][be - 1],
       "mean_conf_correct": float(conf[ok].mean()), "mean_conf_wrong": float(conf[~ok].mean()),
       "top_features": pi.head(10).round(4).to_dict("index")}
json.dump(out, open(OUT / "final_metrics.json", "w"), indent=2)
print(json.dumps({k: out[k] for k in ("auc_per_class", "macro_auc", "n_params", "best_epoch", "mean_conf_correct", "mean_conf_wrong")}, indent=1))
print(pi.head(12).round(4))
