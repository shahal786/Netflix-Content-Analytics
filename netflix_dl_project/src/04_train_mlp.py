"""STEP 4: Deep-learning model - Multi-Layer Perceptron (feed-forward neural network).

* Epoch-by-epoch training with partial_fit so we can log train/validation loss and accuracy
* Basic hyper-parameter tuning (grid over architecture, L2 penalty and learning rate)
* Early stopping on validation loss; best weights are restored
"""
import copy, itertools, json, time, warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from config import *
warnings.filterwarnings("ignore")

d = np.load(ARRAYS, allow_pickle=True)
Xtr, Xva, ytr, yva = d["X_train"], d["X_val"], d["y_train"], d["y_val"]
LABELS = [0, 1, 2]

def train(hidden, alpha, lr, max_epochs, patience, batch=64, seed=SEED, verbose=False):
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=alpha,
                        learning_rate_init=lr, batch_size=batch, random_state=seed)
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "val_f1")}
    best, best_loss, wait, best_ep = None, np.inf, 0, 0
    for ep in range(1, max_epochs + 1):
        clf.partial_fit(Xtr, ytr, classes=LABELS)
        ptr, pva = clf.predict_proba(Xtr), clf.predict_proba(Xva)
        hist["train_loss"].append(log_loss(ytr, ptr, labels=LABELS))
        hist["val_loss"].append(log_loss(yva, pva, labels=LABELS))
        hist["train_acc"].append(accuracy_score(ytr, ptr.argmax(1)))
        hist["val_acc"].append(accuracy_score(yva, pva.argmax(1)))
        hist["val_f1"].append(f1_score(yva, pva.argmax(1), average="macro"))
        if hist["val_loss"][-1] < best_loss - 1e-4:
            best_loss, best, wait, best_ep = hist["val_loss"][-1], copy.deepcopy(clf), 0, ep
        else:
            wait += 1
            if wait >= patience: break
        if verbose and ep % 10 == 0:
            print(f"  epoch {ep:3d} train_loss={hist['train_loss'][-1]:.4f} val_loss={hist['val_loss'][-1]:.4f} val_acc={hist['val_acc'][-1]:.4f}")
    return best, hist, best_ep

# ---------------- hyper-parameter search ----------------
grid = list(itertools.product([(32,), (64,), (64, 32), (128, 64)], [1e-4, 1e-2, 1e-1], [3e-4, 1e-3]))
rows = []
for hidden, alpha, lr in grid:
    t = time.time()
    m, h, be = train(hidden, alpha, lr, max_epochs=60, patience=8)
    pv = m.predict_proba(Xva)
    rows.append(dict(hidden=str(hidden), alpha=alpha, lr=lr, best_epoch=be,
                     val_loss=log_loss(yva, pv, labels=LABELS), val_acc=accuracy_score(yva, pv.argmax(1)),
                     val_macro_f1=f1_score(yva, pv.argmax(1), average="macro"), secs=round(time.time() - t, 1)))
    print(rows[-1], flush=True)
tune = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
tune.to_csv(OUT / "hyperparameter_search.csv", index=False)
b = tune.iloc[0]
best_cfg = dict(hidden=eval(b.hidden), alpha=float(b.alpha), lr=float(b.lr))
print("\nBEST CONFIG:", best_cfg)

# ---------------- final training with longer budget ----------------
print("Final training ...")
model, hist, best_ep = train(best_cfg["hidden"], best_cfg["alpha"], best_cfg["lr"], max_epochs=200, patience=20, verbose=True)
joblib.dump(model, MODELS / "mlp_best.joblib")
json.dump({"config": {**best_cfg, "hidden": list(best_cfg["hidden"])}, "best_epoch": best_ep, "epochs_run": len(hist["val_loss"]),
           "history": hist, "n_params": int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))},
          open(OUT / "training_history.json", "w"), indent=1)
print("best epoch", best_ep, "epochs run", len(hist["val_loss"]))
