"""
part2_clustering.py
────────────────────
Part 2 — Unsupervised Machine Learning

Algorithms:
  (A) Hierarchical Clustering — Ward linkage, 3 experiments (cut-off varies)
  (B) K-Means                 — k = 2..6, Silhouette + Elbow analysis

Produces:
  • Dendrogram
  • Hierarchical scatter plots (3 experiments side-by-side)
  • K-Means scatter plots (5 values of k)
  • Silhouette coefficient curve
  • Elbow plot (inertia)
  • True-labels reference scatter
  • Full numeric summary tables
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

from data_loader import load_and_prepare, TARGET_COL

DATA_PATH  = "data/heart_failure_clinical_records.csv"
OUTPUT_DIR = "output_plots/part2"

FEAT_X = "ejection_fraction"
FEAT_Y = "serum_creatinine"

CMAP         = plt.get_cmap("tab10")
PALETTE_TRUE = {0: "mediumseagreen", 1: "tomato"}


def _save(name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=120, bbox_inches="tight")
    print(f"  [saved] {OUTPUT_DIR}/{name}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

LINKAGE_METHOD = "ward"   # FIXED for all 3 experiments


def plot_dendrogram(X: np.ndarray) -> None:
    Z = linkage(X, method=LINKAGE_METHOD, metric="euclidean")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode="lastp", p=30,
               leaf_rotation=90, leaf_font_size=9, ax=ax, color_threshold=0)
    ax.set_title(
        f"Dendrogram — Linkage: {LINKAGE_METHOD.capitalize()}  "
        f"(truncated to last 30 merges)",
        fontsize=13
    )
    ax.set_xlabel("Sample / cluster size")
    ax.set_ylabel("Euclidean distance")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("dendrogram")


def _hierarchical_fit(X: np.ndarray, y_true: np.ndarray,
                      n_clusters: int, exp_num: int) -> np.ndarray:
    model  = AgglomerativeClustering(n_clusters=n_clusters,
                                     linkage=LINKAGE_METHOD,
                                     metric="euclidean")
    labels = model.fit_predict(X)
    sil    = silhouette_score(X, labels)
    ari    = adjusted_rand_score(y_true, labels)

    print(f"\n── Hierarchical Experiment {exp_num} ─────────────────────────────")
    print(f"   Linkage    : {LINKAGE_METHOD}  (fixed)")
    print(f"   n_clusters : {n_clusters}  ← cut-off line position")
    print(f"   Silhouette : {sil:.4f}")
    print(f"   ARI        : {ari:.4f}")
    for c in range(n_clusters):
        m = labels == c
        s0, s1 = (y_true[m] == 0).sum(), (y_true[m] == 1).sum()
        tot = m.sum()
        print(f"   Cluster {c}  : {tot} objects | "
              f"Survived={s0} ({s0/tot*100:.1f}%) | "
              f"Died={s1} ({s1/tot*100:.1f}%)")
    return labels, sil, ari


def plot_hierarchical_results(X_df: pd.DataFrame,
                               labels_list: list,
                               k_list: list) -> None:
    n   = len(labels_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)
    fig.suptitle(
        f"Hierarchical Clustering (Ward) — {FEAT_X} vs {FEAT_Y}",
        fontsize=13, y=1.01
    )
    for ax, labels, k in zip(axes, labels_list, k_list):
        for c in range(k):
            m = labels == c
            ax.scatter(X_df[FEAT_X][m], X_df[FEAT_Y][m],
                       c=[CMAP(c)], label=f"Cluster {c}",
                       alpha=0.6, s=18, edgecolors="none")
        ax.set_title(f"k = {k} clusters", fontsize=12)
        ax.set_xlabel(FEAT_X.replace("_", " ").title())
        ax.set_ylabel(FEAT_Y.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    _save("hierarchical_scatter_experiments")


def run_hierarchical(X_scaled: pd.DataFrame, y_true: np.ndarray) -> None:
    print("\n" + "="*60)
    print("  PART 2A — HIERARCHICAL CLUSTERING")
    print("="*60)

    Xnp = X_scaled.values

    plot_dendrogram(Xnp)

    k_list      = [2, 3, 4]
    labels_list = []
    sil_list    = []
    ari_list    = []

    for i, k in enumerate(k_list, 1):
        lbl, sil, ari = _hierarchical_fit(Xnp, y_true, k, i)
        labels_list.append(lbl)
        sil_list.append(sil)
        ari_list.append(ari)

    print("\n── Hierarchical Summary ─────────────────────────────────────")
    print(f"{'Exp':>4} | {'Linkage':>8} | {'k':>3} | {'Silhouette':>11} | {'ARI':>8}")
    print("─" * 46)
    for i, (k, sil, ari) in enumerate(zip(k_list, sil_list, ari_list), 1):
        print(f"{i:>4} | {LINKAGE_METHOD:>8} | {k:>3} | {sil:>11.4f} | {ari:>8.4f}")

    plot_hierarchical_results(X_scaled, labels_list, k_list)


# ══════════════════════════════════════════════════════════════════════════════
#  K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def _kmeans_fit(X: np.ndarray, y_true: np.ndarray,
                k: int, random_state: int = 42):
    model  = KMeans(n_clusters=k, init="k-means++",
                    n_init=10, max_iter=300, random_state=random_state)
    labels = model.fit_predict(X)
    sil    = silhouette_score(X, labels)
    ari    = adjusted_rand_score(y_true, labels)

    print(f"\n── K-Means k={k} ─────────────────────────────────────────────")
    print(f"   init       : k-means++")
    print(f"   n_init     : 10")
    print(f"   max_iter   : 300")
    print(f"   Silhouette : {sil:.4f}")
    print(f"   Inertia    : {model.inertia_:.2f}")
    print(f"   ARI        : {ari:.4f}")
    for c in range(k):
        m = labels == c
        s0, s1 = (y_true[m] == 0).sum(), (y_true[m] == 1).sum()
        tot = m.sum()
        print(f"   Cluster {c}  : {tot} objects | "
              f"Survived={s0} ({s0/tot*100:.1f}%) | "
              f"Died={s1} ({s1/tot*100:.1f}%)")

    return labels, sil, ari, model.inertia_


def plot_silhouette_curve(k_vals: list, sil_scores: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, sil_scores, marker="o", color="#e05c5c",
            linewidth=2.5, markersize=9)
    for k, s in zip(k_vals, sil_scores):
        ax.annotate(f"{s:.3f}", (k, s),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10)
    ax.axvline(x=2, color="gray", linestyle="--", alpha=0.5,
               label="True number of classes (k=2)")
    ax.set_title("K-Means: Silhouette Coefficient vs k", fontsize=13)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Coefficient")
    ax.set_xticks(k_vals)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("kmeans_silhouette_curve")


def plot_elbow(k_vals: list, inertia: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, inertia, marker="s", color="#5c8ee0",
            linewidth=2.5, markersize=9)
    for k, v in zip(k_vals, inertia):
        ax.annotate(f"{v:,.0f}", (k, v),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    ax.set_title("K-Means: Elbow Plot (Inertia vs k)", fontsize=13)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_xticks(k_vals)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("kmeans_elbow")


def plot_kmeans_results(X_df: pd.DataFrame,
                         labels_list: list, k_vals: list) -> None:
    n   = len(labels_list)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True)
    fig.suptitle(f"K-Means — {FEAT_X} vs {FEAT_Y}", fontsize=13, y=1.01)
    for ax, labels, k in zip(axes, labels_list, k_vals):
        for c in range(k):
            m = labels == c
            ax.scatter(X_df[FEAT_X][m], X_df[FEAT_Y][m],
                       c=[CMAP(c)], label=f"Cluster {c}",
                       alpha=0.6, s=18, edgecolors="none")
        ax.set_title(f"k = {k}", fontsize=12)
        ax.set_xlabel(FEAT_X.replace("_", " ").title())
        ax.set_ylabel(FEAT_Y.replace("_", " ").title())
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    _save("kmeans_scatter_experiments")


def plot_true_labels(X_df: pd.DataFrame, y_true: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in [0, 1]:
        m = y_true == cls
        ax.scatter(X_df[FEAT_X][m], X_df[FEAT_Y][m],
                   c=PALETTE_TRUE[cls], alpha=0.55, s=20, edgecolors="none",
                   label=f"{'Survived' if cls==0 else 'Died'} ({cls})")
    ax.set_title(f"Reference: True Labels — {FEAT_X} vs {FEAT_Y}", fontsize=12)
    ax.set_xlabel(FEAT_X.replace("_", " ").title())
    ax.set_ylabel(FEAT_Y.replace("_", " ").title())
    ax.legend(title="Outcome")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("true_labels_reference")


def run_kmeans(X_scaled: pd.DataFrame, y_true: np.ndarray) -> None:
    print("\n" + "="*60)
    print("  PART 2B — K-MEANS CLUSTERING")
    print("="*60)

    Xnp       = X_scaled.values
    k_vals    = [2, 3, 4, 5, 6]
    sil_list  = []
    inertia   = []
    lbl_list  = []

    for k in k_vals:
        lbl, sil, ari, ine = _kmeans_fit(Xnp, y_true, k)
        sil_list.append(sil)
        inertia.append(ine)
        lbl_list.append(lbl)

    best_idx = int(np.argmax(sil_list))
    print("\n── K-Means Summary ──────────────────────────────────────────")
    print(f"{'k':>4} | {'Silhouette':>11} | {'Inertia':>13} | {'ARI':>8}")
    print("─" * 46)
    for k, sil, ine, lbl in zip(k_vals, sil_list, inertia, lbl_list):
        ari = adjusted_rand_score(y_true, lbl)
        marker = " ◄ best sil" if k == k_vals[best_idx] else ""
        print(f"{k:>4} | {sil:>11.4f} | {ine:>13.2f} | {ari:>8.4f}{marker}")

    plot_silhouette_curve(k_vals, sil_list)
    plot_elbow(k_vals, inertia)
    plot_kmeans_results(X_scaled, lbl_list, k_vals)
    plot_true_labels(X_scaled, y_true)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    df_clean, X_raw, X_scaled, y, _ = load_and_prepare(DATA_PATH)

    run_hierarchical(X_scaled, y)
    run_kmeans(X_scaled, y)

    print(f"\n  All Part 2 plots saved to '{OUTPUT_DIR}/'")
    print("  Showing all plots — close the windows to continue...\n")
    plt.show()


if __name__ == "__main__":
    run()
