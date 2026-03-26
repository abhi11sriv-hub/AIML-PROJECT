"""
tests/test_preprocess.py
Unit tests for data preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import (
    filter_target,
    drop_leakage,
    clean_features,
    impute_missing,
    drop_high_null_cols,
    encode_categoricals,
    TARGET_MAP
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_sample():
    """Minimal synthetic dataframe mimicking LendingClub schema."""
    return pd.DataFrame({
        "loan_status":     ["Fully Paid", "Charged Off", "Current", "Fully Paid"],
        "loan_amnt":       [10000, 15000, 8000, 12000],
        "int_rate":        ["13.56%", "22.10%", "8.50%", "11.20%"],
        "revol_util":      ["54.3%", "87.1%", None, "30.2%"],
        "emp_length":      ["3 years", "10+ years", "< 1 year", "5 years"],
        "term":            [" 36 months", " 60 months", " 36 months", " 36 months"],
        "annual_inc":      [60000.0, 35000.0, 55000.0, 80000.0],
        "home_ownership":  ["RENT", "OWN", "MORTGAGE", "RENT"],
        # Leakage columns
        "total_pymnt":     [9800, 3000, None, 12100],
        "recoveries":      [0, 500, 0, 0],
        "last_pymnt_d":    ["Jan-2019", "Mar-2018", None, "Dec-2019"],
    })


# ── filter_target ─────────────────────────────────────────────────────────────

def test_filter_target_keeps_binary(raw_sample):
    df = filter_target(raw_sample)
    assert set(df["default"].unique()).issubset({0, 1})


def test_filter_target_removes_current(raw_sample):
    df = filter_target(raw_sample)
    assert len(df) == 3  # removes "Current"


def test_filter_target_mapping(raw_sample):
    df = filter_target(raw_sample)
    assert df[df["default"] == 1].shape[0] == 1   # one charged off
    assert df[df["default"] == 0].shape[0] == 2   # two fully paid


def test_filter_target_drops_loan_status_col(raw_sample):
    df = filter_target(raw_sample)
    assert "loan_status" not in df.columns


# ── drop_leakage ──────────────────────────────────────────────────────────────

def test_drop_leakage_removes_total_pymnt(raw_sample):
    df = drop_leakage(raw_sample)
    assert "total_pymnt" not in df.columns


def test_drop_leakage_removes_recoveries(raw_sample):
    df = drop_leakage(raw_sample)
    assert "recoveries" not in df.columns


def test_drop_leakage_keeps_loan_amnt(raw_sample):
    df = drop_leakage(raw_sample)
    assert "loan_amnt" in df.columns


# ── clean_features ────────────────────────────────────────────────────────────

def test_int_rate_parsed_as_float(raw_sample):
    df = clean_features(raw_sample.copy())
    assert df["int_rate"].dtype in [np.float64, float]
    assert df["int_rate"].iloc[0] == pytest.approx(13.56)


def test_revol_util_parsed_as_float(raw_sample):
    df = clean_features(raw_sample.copy())
    assert pd.api.types.is_float_dtype(df["revol_util"])


def test_emp_length_mapped_to_int(raw_sample):
    df = clean_features(raw_sample.copy())
    assert df["emp_length"].iloc[0] == 3
    assert df["emp_length"].iloc[1] == 10
    assert df["emp_length"].iloc[2] == 0


def test_term_extracted_as_int(raw_sample):
    df = clean_features(raw_sample.copy())
    assert df["term"].iloc[0] == 36
    assert df["term"].iloc[1] == 60


# ── impute_missing ────────────────────────────────────────────────────────────

def test_impute_fills_numeric_nulls():
    df = pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0],
        "b": [None, 5.0, 6.0, 7.0]
    })
    df_out = impute_missing(df)
    assert df_out.isna().sum().sum() == 0


def test_impute_uses_median():
    df = pd.DataFrame({"x": [1.0, 3.0, None, 5.0]})
    df_out = impute_missing(df)
    assert df_out["x"].iloc[2] == pytest.approx(3.0)


# ── drop_high_null_cols ───────────────────────────────────────────────────────

def test_drop_high_null_cols_removes_sparse():
    df = pd.DataFrame({
        "good": [1, 2, 3, 4],
        "sparse": [None, None, None, 1]  # 75% null
    })
    df_out = drop_high_null_cols(df, threshold=0.5)
    assert "sparse" not in df_out.columns
    assert "good" in df_out.columns


def test_drop_high_null_cols_keeps_dense():
    df = pd.DataFrame({
        "ok": [1, None, 3, 4]  # 25% null — below threshold
    })
    df_out = drop_high_null_cols(df, threshold=0.5)
    assert "ok" in df_out.columns


# ── encode_categoricals ───────────────────────────────────────────────────────

def test_encode_categoricals_no_strings_remain():
    df = pd.DataFrame({
        "cat": ["RENT", "OWN", "MORTGAGE", "RENT"],
        "num": [1.0, 2.0, 3.0, 4.0]
    })
    df_out = encode_categoricals(df)
    assert df_out.select_dtypes(include="object").empty
