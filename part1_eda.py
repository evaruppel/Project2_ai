"""
part1_eda.py
────────────
Part 1 — Data Pre-processing / Exploratory Data Analysis

Produces:
  • Feature roles table
  • Descriptive statistics (central tendency + dispersion)
  • 2D scatter plot  (age vs ejection_fraction)
  • 3D scatter plot  (age vs ejection_fraction vs serum_creatinine)
  • Histogram 1      (ejection_fraction by outcome)
  • Histogram 2      (follow-up time by outcome)
  • Box plots        (ejection_fraction + serum_creatinine)
  • Violin plots     (ejection_fraction + serum_creatinine)
  • Correlation heatmap
  • Data snippet (first 10 rows)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def _save(name: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=120, bbox_inches="tight")
    print(f"  [saved] {OUTPUT_DIR}/{name}.png")

from data_loader import (
    load_and_prepare, print_summary,
    BINARY_COLS, NUMERICAL_COLS, TARGET_COL
)

DATA_PATH  = "data/heart_failure_clinical_records.csv"
OUTPUT_DIR = "output_plots/part1"


# ── 1. Feature roles ───────────────────────────────────────────────────────────
def show_feature_roles(df: pd.DataFrame) -> None:
    print("\n── Feature Roles ────────────────────────────────────────────")
    rows = []
    for col in df.columns:
        role      = "Target" if col == TARGET_COL else "Input"
        ftype     = "Binary"    if col in BINARY_COLS else \
                    "Continuous" if col in NUMERICAL_COLS else "Target"
        val_range = f"{df[col].min()} – {df[col].max()}"
        rows.append({"Feature": col, "Role": role, "Type": ftype, "Range": val_range,
                     "Dtype": str(df[col].dtype)})
    roles_df = pd.DataFrame(rows).sort_values("Role")
    print(roles_df.to_string(index=False))


# ── 2. Descriptive statistics ──────────────────────────────────────────────────
def show_statistics(df: pd.DataFrame) -> None:
    print("\n── Continuous Feature Statistics ────────────────────────────")
    stats = []
    for col in NUMERICAL_COLS:
        d = df[col].dropna()
        stats.append({
            "Feature" : col,
            "Mean"    : d.mean(),
            "Median"  : d.median(),
            "Std Dev" : d.std(),
            "Variance": d.var(),
            "Min"     : d.min(),
            "Max"     : d.max(),
            "Range"   : d.max() - d.min(),
            "IQR"     : d.quantile(0.75) - d.quantile(0.25),
        })
    print(pd.DataFrame(stats).set_index("Feature").round(2).to_string())

    print("\n── Binary / Categorical Feature Counts ──────────────────────")
    for col in BINARY_COLS + [TARGET_COL]:
        vc = df[col].value_counts().sort_index()
        print(f"\n  {col}:")
        for val, cnt in vc.items():
            print(f"    {val}: {cnt}  ({cnt/len(df)*100:.1f}%)")


# ── 3. Data snippet ────────────────────────────────────────────────────────────
def show_snippet(df: pd.DataFrame) -> None:
    print("\n── Data File Structure (first 10 rows) ──────────────────────")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df.head(10).to_string(index=False))


# ── 4. Outlier report ─────────────────────────────────────────────────────────
def show_outliers(df: pd.DataFrame) -> None:
    print("\n── Outlier Report (IQR method, 1.5×IQR fence) ──────────────")
    for col in NUMERICAL_COLS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        pct = n / len(df) * 100
        print(f"  {col:<30}: {n:>4} outliers ({pct:.1f}%)")


# ── 5. Correlation report ─────────────────────────────────────────────────────
def show_correlations(df: pd.DataFrame) -> None:
    print("\n── Pearson Correlation with DEATH_EVENT ─────────────────────")
    corr = df.corr()[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    for feat, val in corr.items():
        bar = "█" * int(abs(val) * 30)
        sign = "+" if val > 0 else "-"
        print(f"  {feat:<30} {sign}{abs(val):.4f}  {bar}")


# ── Plots ──────────────────────────────────────────────────────────────────────
PALETTE   = {0: "mediumseagreen", 1: "tomato"}
LEGEND_H  = [mpatches.Patch(color="mediumseagreen", label="Survived (0)"),
             mpatches.Patch(color="tomato",         label="Died (1)")]


def plot_scatter_2d(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for cls in [0, 1]:
        sub = df[df[TARGET_COL] == cls]
        ax.scatter(sub["age"], sub["ejection_fraction"],
                   c=PALETTE[cls], alpha=0.7, s=55, edgecolors="white",
                   linewidths=0.4, label=f"{'Survived' if cls==0 else 'Died'} ({cls})")
    ax.set_title("Plot 1: Age vs Ejection Fraction", fontsize=14)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Ejection Fraction (%)")
    ax.legend(title="Outcome")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("scatter_2d_age_ejection")


def plot_scatter_2d_time(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for cls in [0, 1]:
        sub = df[df[TARGET_COL] == cls]
        ax.scatter(sub["time"], sub["serum_creatinine"],
                   c=PALETTE[cls], alpha=0.7, s=55, edgecolors="white",
                   linewidths=0.4, label=f"{'Survived' if cls==0 else 'Died'} ({cls})")
    ax.set_title("Plot 2: Follow-up Time vs Serum Creatinine", fontsize=14)
    ax.set_xlabel("Follow-up Time (days)")
    ax.set_ylabel("Serum Creatinine (mg/dL)")
    ax.legend(title="Outcome")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("scatter_2d_time_creatinine")


def plot_scatter_3d(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    colors = df[TARGET_COL].map(PALETTE)
    ax.scatter(df["age"], df["ejection_fraction"], df["serum_creatinine"],
               c=colors, s=40, alpha=0.75, edgecolors="white", linewidths=0.3)
    ax.set_xlabel("Age", fontsize=10)
    ax.set_ylabel("Ejection Fraction", fontsize=10)
    ax.set_zlabel("Serum Creatinine", fontsize=10)
    ax.set_title("3D Scatter: Age × Ejection Fraction × Serum Creatinine", fontsize=12, pad=20)
    legend_elems = [
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="mediumseagreen",
                   markersize=10, label="Survived (0)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="tomato",
                   markersize=10, label="Died (1)"),
    ]
    ax.legend(handles=legend_elems, title="Outcome")
    plt.tight_layout()
    _save("scatter_3d")


def plot_histogram_ejection(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in [0, 1]:
        sub = df[df[TARGET_COL] == cls]["ejection_fraction"]
        ax.hist(sub, bins=20, alpha=0.6, color=PALETTE[cls], edgecolor="white", label=f"{'Survived' if cls==0 else 'Died'} ({cls})")
        sub.plot.kde(ax=ax, color=PALETTE[cls], linewidth=2)
    ax.set_title("Histogram 1: Ejection Fraction Distribution by Outcome", fontsize=13)
    ax.set_xlabel("Ejection Fraction (%)")
    ax.set_ylabel("Count")
    ax.legend(handles=LEGEND_H, title="Outcome")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("histogram_ejection_fraction")


def plot_histogram_time(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in [0, 1]:
        sub = df[df[TARGET_COL] == cls]["time"]
        ax.hist(sub, bins=30, alpha=0.6, color=PALETTE[cls], edgecolor="white", label=f"{'Survived' if cls==0 else 'Died'} ({cls})")
        sub.plot.kde(ax=ax, color=PALETTE[cls], linewidth=2)
    ax.set_title("Histogram 2: Follow-up Time Distribution by Outcome", fontsize=13)
    ax.set_xlabel("Follow-up Time (days)")
    ax.set_ylabel("Count")
    ax.legend(handles=LEGEND_H, title="Outcome")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save("histogram_time")


def plot_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, ylabel in zip(
        axes,
        ["ejection_fraction", "serum_creatinine"],
        ["Ejection Fraction (%)", "Serum Creatinine (mg/dL)"]
    ):
        sns.boxplot(data=df, x=TARGET_COL, y=col, hue=TARGET_COL,
                    palette=["mediumseagreen", "tomato"], legend=False, ax=ax)
        ax.set_title(f"Box Plot: {col.replace('_',' ').title()}")
        ax.set_xlabel("Outcome (0=Survived, 1=Died)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Box Plots — Distribution by Outcome", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save("boxplots")


def plot_violinplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, ylabel in zip(
        axes,
        ["ejection_fraction", "serum_creatinine"],
        ["Ejection Fraction (%)", "Serum Creatinine (mg/dL)"]
    ):
        sns.violinplot(data=df, x=TARGET_COL, y=col, hue=TARGET_COL,
                       palette=["mediumseagreen", "tomato"], legend=False,
                       inner="box", ax=ax)
        ax.set_title(f"Violin Plot: {col.replace('_',' ').title()}")
        ax.set_xlabel("Outcome (0=Survived, 1=Died)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Violin Plots — Distribution by Outcome", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save("violinplots")


def plot_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                mask=mask, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    _save("correlation_heatmap")


# ── Main ───────────────────────────────────────────────────────────────────────
def run():
    df_clean, X_raw, X_scaled, y, scaler = load_and_prepare(DATA_PATH)
    df_raw = __import__("data_loader").load_raw(DATA_PATH)

    print_summary(df_raw, df_clean, y)
    show_snippet(df_clean)
    show_feature_roles(df_clean)
    show_statistics(df_clean)
    show_outliers(df_clean)
    show_correlations(df_clean)

    print("\n  Generating plots...")
    plot_scatter_2d(df_clean)
    plot_scatter_2d_time(df_clean)
    plot_scatter_3d(df_clean)
    plot_histogram_ejection(df_clean)
    plot_histogram_time(df_clean)
    plot_boxplots(df_clean)
    plot_violinplots(df_clean)
    plot_heatmap(df_clean)

    print(f"\n  All Part 1 plots saved to '{OUTPUT_DIR}/'")
    print("  Showing all plots — close the windows to continue...\n")
    plt.show()


if __name__ == "__main__":
    run()
