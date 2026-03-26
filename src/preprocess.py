"""
preprocess.py
-------------
Data loading, cleaning, and feature engineering for the LendingClub loan dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# Columns that leak the target (assigned AFTER a loan outcome is known)
LEAKAGE_COLS = [
    "funded_amnt_inv", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "debt_settlement_flag"
]

# High-cardinality or free-text columns to drop
DROP_COLS = [
    "id", "member_id", "url", "desc", "title", "zip_code",
    "earliest_cr_line", "last_credit_pull_d", "issue_d"
]

TARGET_KEEP = ["Fully Paid", "Charged Off"]
TARGET_MAP  = {"Fully Paid": 0, "Charged Off": 1}


def load_data(filepath: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """Load the LendingClub CSV. Optionally sample for faster iteration."""
    print(f"Loading data from {filepath} ...")
    df = pd.read_csv(filepath, low_memory=False)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    print(f"  Shape: {df.shape}")
    return df


def filter_target(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only binary-classifiable loan statuses."""
    df = df[df["loan_status"].isin(TARGET_KEEP)].copy()
    df["default"] = df["loan_status"].map(TARGET_MAP)
    df.drop(columns=["loan_status"], inplace=True)
    print(f"  After target filter: {df.shape}  |  Default rate: {df['default'].mean():.2%}")
    return df


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Remove post-outcome and leakage columns."""
    cols_to_drop = [c for c in LEAKAGE_COLS + DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse percentage strings, extract numeric tenure, and encode categoricals."""

    # int_rate: "13.56%" -> 13.56
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "").astype(float)

    # revol_util: "54.3%" -> 54.3
    if "revol_util" in df.columns:
        df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "")
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

    # emp_length: "10+ years" -> 10, "< 1 year" -> 0
    if "emp_length" in df.columns:
        emp_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8,  "9 years": 9, "10+ years": 10
        }
        df["emp_length"] = df["emp_length"].map(emp_map)

    # term: " 36 months" -> 36
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode low-cardinality string columns."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric NaNs with median; categoricals already encoded as strings."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    return df


def drop_high_null_cols(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with more than `threshold` fraction missing."""
    null_frac = df.isna().mean()
    to_drop = null_frac[null_frac > threshold].index.tolist()
    if to_drop:
        print(f"  Dropping {len(to_drop)} columns with >{threshold:.0%} nulls: {to_drop[:5]} ...")
    df.drop(columns=to_drop, inplace=True)
    return df


def preprocess_pipeline(filepath: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """End-to-end preprocessing. Returns clean DataFrame ready for modeling."""
    df = load_data(filepath, sample_frac)
    df = filter_target(df)
    df = drop_leakage(df)
    df = drop_high_null_cols(df, threshold=0.5)
    df = clean_features(df)
    df = encode_categoricals(df)
    df = impute_missing(df)
    print(f"  Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/loan.csv"
    df = preprocess_pipeline(path, sample_frac=0.1)
    df.to_parquet("data/processed.parquet", index=False)
    print("Saved to data/processed.parquet")
