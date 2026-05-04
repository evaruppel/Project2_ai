"""
part3_classification.py
────────────────────────
Part 3 — Supervised Machine Learning

Algorithms:
  1. Artificial Neural Network  (MLPClassifier)   — mandatory
  2. Logistic Regression                           — chosen algorithm 1
  3. Random Forest                                 — chosen algorithm 2

Pipeline per algorithm:
  • Stratified train/test split (80% / 20%)
  • 3 hyperparameter experiments on training data (5-fold CV)
  • Best model selected by F1-score (macro)
  • Best model evaluated on held-out test set

Metrics reported:
  Accuracy, Precision, Recall, F1 (macro), ROC-AUC, Confusion Matrix
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve)

from data_loader import load_and_prepare, TARGET_COL

DATA_PATH  = "data/heart_failure_clinical_records.csv"
OUTPUT_DIR = "output_plots/part3"
SEED       = 42
TEST_SIZE  = 0.20
CV_FOLDS   = 5

PALETTE = ["mediumseagreen", "tomato"]


def _save(name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=120, bbox_inches="tight")
    print(f"  [saved] {OUTPUT_DIR}/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def make_split(X: pd.DataFrame, y: np.ndarray):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print("\n" + "="*60)
    print("  TRAIN / TEST SPLIT")
    print("="*60)
    for name, ys in [("Train", y_tr), ("Test", y_te)]:
        n0, n1 = (ys==0).sum(), (ys==1).sum()
        print(f"  {name:<6}: {len(ys):>4} objects  "
              f"({len(ys)/len(y)*100:.1f}%)  |  "
              f"Survived={n0} ({n0/len(ys)*100:.1f}%)  "
              f"Died={n1} ({n1/len(ys)*100:.1f}%)")
    return X_tr, X_te, y_tr, y_te


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _cv_scores(model, X_tr, y_tr) -> dict:
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    res  = cross_validate(
        model, X_tr, y_tr, cv=cv,
        scoring=["accuracy", "f1_macro", "roc_auc"],
        return_train_score=False
    )
    return {
        "Accuracy"  : res["test_accuracy"].mean(),
        "F1 (macro)": res["test_f1_macro"].mean(),
        "ROC-AUC"   : res["test_roc_auc"].mean(),
    }


def run_experiments(name: str, configs: list[dict],
                    X_tr, y_tr) -> tuple[int, list]:
    """
    configs : list of dicts, each with keys:
        "label"  — short description for table
        "model"  — sklearn estimator instance
    Returns index of best experiment (by F1) and list of result dicts.
    """
    print(f"\n{'='*60}")
    print(f"  {name.upper()} — TRAINING EXPERIMENTS")
    print(f"{'='*60}")

    results = []
    for i, cfg in enumerate(configs, 1):
        scores = _cv_scores(cfg["model"], X_tr, y_tr)
        scores["Experiment"] = i
        scores["Config"]     = cfg["label"]
        results.append(scores)

        print(f"\n  Experiment {i} — {cfg['label']}")
        print(f"    Accuracy   : {scores['Accuracy']:.4f}")
        print(f"    F1 (macro) : {scores['F1 (macro)']:.4f}")
        print(f"    ROC-AUC    : {scores['ROC-AUC']:.4f}")

    best_idx = int(np.argmax([r["F1 (macro)"] for r in results]))
    print(f"\n  ► Best experiment: #{best_idx+1} — {configs[best_idx]['label']}")
    return best_idx, results


# ══════════════════════════════════════════════════════════════════════════════
#  TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_test(name: str, model, X_tr, y_tr, X_te, y_te) -> dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = {
        "Algorithm" : name,
        "Accuracy"  : accuracy_score(y_te, y_pred),
        "Precision" : precision_score(y_te, y_pred, average="macro", zero_division=0),
        "Recall"    : recall_score(y_te, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_te, y_pred, average="macro", zero_division=0),
        "ROC-AUC"   : roc_auc_score(y_te, y_prob),
    }

    print(f"\n── Test Results: {name} ─────────────────────────────────────")
    for k, v in metrics.items():
        if k != "Algorithm":
            print(f"   {k:<12}: {v:.4f}")

    cm = confusion_matrix(y_te, y_pred)
    metrics["_cm"]    = cm
    metrics["_prob"]  = y_prob
    metrics["_model"] = model
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_experiment_bars(name: str, results: list, best_idx: int) -> None:
    df = pd.DataFrame(results).set_index("Experiment")
    metrics_cols = ["Accuracy", "F1 (macro)", "ROC-AUC"]

    x    = np.arange(len(results))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#5c8ee0", "#e0a85c", "#7cba6e"]

    for j, (col, color) in enumerate(zip(metrics_cols, colors)):
        bars = ax.bar(x + j*w, df[col].values, w, label=col, color=color, alpha=0.85)
        for bar, val in zip(bars, df[col].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(f"{name} — CV Results per Experiment", fontsize=13)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"Exp {i+1}\n{results[i]['Config']}" for i in range(len(results))],
                       fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.axvspan(best_idx - 0.2, best_idx + 0.8, alpha=0.08,
               color="gold", label="Best")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(f"experiments_{name.replace(' ','_').replace('(','').replace(')','')}") 


def plot_confusion_matrices(test_results: list) -> None:
    n   = len(test_results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, test_results):
        disp = ConfusionMatrixDisplay(res["_cm"],
                                      display_labels=["Survived", "Died"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(res["Algorithm"], fontsize=12)
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14)
    plt.tight_layout()
    _save("confusion_matrices")


def plot_roc_curves(test_results: list, y_te: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#5c8ee0", "#e05c5c", "#7cba6e"]
    for res, color in zip(test_results, colors):
        fpr, tpr, _ = roc_curve(y_te, res["_prob"])
        auc = res["ROC-AUC"]
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{res['Algorithm']} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_title("ROC Curves — Test Set", fontsize=13)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("roc_curves")


def plot_metric_comparison(test_results: list) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1 (macro)", "ROC-AUC"]
    names   = [r["Algorithm"] for r in test_results]
    data    = {m: [r[m] for r in test_results] for m in metrics}

    x    = np.arange(len(metrics))
    w    = 0.22
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#5c8ee0", "#e05c5c", "#7cba6e"]

    for j, (name, color) in enumerate(zip(names, colors)):
        vals = [data[m][j] for m in metrics]
        bars = ax.bar(x + j*w, vals, w, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Algorithm Comparison — Test Set Performance", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_xticks(x + w)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("metric_comparison")


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _ann_configs():
    return [
        {
            "label": "hidden=(64,32), relu, adam, lr=0.001",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32),
                                   activation="relu", solver="adam",
                                   learning_rate_init=0.001, max_iter=500,
                                   random_state=SEED)
        },
        {
            "label": "hidden=(128,64,32), relu, adam, lr=0.001",
            "model": MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                   activation="relu", solver="adam",
                                   learning_rate_init=0.001, max_iter=500,
                                   random_state=SEED)
        },
        {
            "label": "hidden=(64,32), tanh, adam, lr=0.0005",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32),
                                   activation="tanh", solver="adam",
                                   learning_rate_init=0.0005, max_iter=500,
                                   random_state=SEED)
        },
    ]


def _lr_configs():
    return [
        {
            "label": "C=1.0, solver=lbfgs",
            "model": LogisticRegression(C=1.0, solver="lbfgs",
                                        max_iter=1000, random_state=SEED)
        },
        {
            "label": "C=0.1, solver=lbfgs",
            "model": LogisticRegression(C=0.1, solver="lbfgs",
                                        max_iter=1000, random_state=SEED)
        },
        {
            "label": "C=10.0, solver=saga, penalty=l1",
            "model": LogisticRegression(C=10.0, solver="saga", penalty="l1",
                                        max_iter=2000, random_state=SEED)
        },
    ]


def _rf_configs():
    return [
        {
            "label": "n=100, max_depth=None",
            "model": RandomForestClassifier(n_estimators=100, max_depth=None,
                                            random_state=SEED)
        },
        {
            "label": "n=200, max_depth=10, min_samples_split=5",
            "model": RandomForestClassifier(n_estimators=200, max_depth=10,
                                            min_samples_split=5,
                                            random_state=SEED)
        },
        {
            "label": "n=300, max_depth=15, max_features=sqrt",
            "model": RandomForestClassifier(n_estimators=300, max_depth=15,
                                            max_features="sqrt",
                                            random_state=SEED)
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    df_clean, X_raw, X_scaled, y, _ = load_and_prepare(DATA_PATH)
    X_tr, X_te, y_tr, y_te = make_split(X_scaled, y)

    algorithms = [
        ("Neural Network (MLP)",  _ann_configs()),
        ("Logistic Regression",   _lr_configs()),
        ("Random Forest",         _rf_configs()),
    ]

    best_models  = []
    test_results = []

    # ── Training experiments ──────────────────────────────────────────────────
    for alg_name, configs in algorithms:
        best_idx, results = run_experiments(alg_name, configs, X_tr, y_tr)
        plot_experiment_bars(alg_name, results, best_idx)
        best_models.append((alg_name, configs[best_idx]["model"]))

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL TEST SET EVALUATION")
    print("="*60)

    for alg_name, model in best_models:
        res = evaluate_on_test(alg_name, model, X_tr, y_tr, X_te, y_te)
        test_results.append(res)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Test Set — Final Comparison ──────────────────────────────")
    cols = ["Algorithm", "Accuracy", "Precision", "Recall", "F1 (macro)", "ROC-AUC"]
    summary = pd.DataFrame([{k: r[k] for k in cols} for r in test_results])
    summary_print = summary.copy()
    for col in cols[1:]:
        summary_print[col] = summary_print[col].map("{:.4f}".format)
    print(summary_print.to_string(index=False))

    best_alg = summary.loc[summary["F1 (macro)"].idxmax(), "Algorithm"]
    print(f"\n  ► Best overall model: {best_alg}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_confusion_matrices(test_results)
    plot_roc_curves(test_results, y_te)
    plot_metric_comparison(test_results)

    print(f"\n  All Part 3 plots saved to '{OUTPUT_DIR}/'")
    print("  Showing all plots — close the windows to continue...\n")
    plt.show()


if __name__ == "__main__":
    run()
