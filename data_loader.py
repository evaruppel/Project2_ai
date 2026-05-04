"""
data_loader.py
──────────────
Shared data loading, cleaning and preprocessing utilities.
Used by part1_eda.py, part2_clustering.py, part3_classification.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ── Column metadata ────────────────────────────────────────────────────────────
BINARY_COLS   = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
TARGET_COL    = "DEATH_EVENT"
NUMERICAL_COLS = [
    "age", "creatinine_phosphokinase", "ejection_fraction",
    "platelets", "serum_creatinine", "serum_sodium", "time"
]


def load_raw(path: str) -> pd.DataFrame:
    """Load CSV as-is (with duplicates)."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and reset index."""
    return df.drop_duplicates().reset_index(drop=True)


def get_X_y(df: pd.DataFrame):
    """Split into features X and target y."""
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL].values
    return X, y


def scale(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScaler fit+transform. Returns scaled DataFrame + fitted scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), scaler


def load_and_prepare(path: str):
    """
    Full pipeline: load → deduplicate → split → scale.

    Returns
    -------
    df_clean   : cleaned DataFrame (original scale, with DEATH_EVENT)
    X_raw      : features, original scale
    X_scaled   : features, StandardScaler-normalized
    y          : target array (0 / 1)
    scaler     : fitted StandardScaler instance
    """
    df_raw   = load_raw(path)
    df_clean = clean(df_raw)
    X_raw, y = get_X_y(df_clean)
    X_scaled, scaler = scale(X_raw)
    return df_clean, X_raw, X_scaled, y, scaler


def print_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame, y: np.ndarray) -> None:
    print("=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Raw rows        : {len(df_raw)}")
    print(f"  Duplicates      : {df_raw.duplicated().sum()}")
    print(f"  After cleaning  : {len(df_clean)} rows, {df_clean.shape[1]} columns")
    print(f"  Missing values  : {df_clean.isnull().sum().sum()}")
    print(f"  Survived (0)    : {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"  Died     (1)    : {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    print("=" * 60)
