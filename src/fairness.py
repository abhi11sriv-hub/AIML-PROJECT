"""
fairness.py
-----------
Audit the trained model for demographic fairness using Fairlearn.
Computes demographic parity, equalized odds, and disparate impact
across sensitive attributes (e.g. home ownership, employment length tier).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate,
    false_negative_rate,
    selection_rate
)
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.postprocessing import ThresholdOptimizer

MODELS_DIR  = "models"
REPORTS_DIR = "reports/figures"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Sensitive feature engineering ────────────────────────────────────────────

def make_sensitive_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable proxy sensitive groups from available features.
    NOTE: Race/gender are not in LendingClub data; we audit on proxies
    that historically correlate with protected attributes.
    """
    sf = pd.DataFrame(index=X.index)

    # Employment length tier (economic stability proxy)
    if "emp_length" in X.columns:
        sf["emp_tier"] = pd.cut(
            X["emp_length"],
            bins=[-1, 1, 5, 10],
            labels=["Short (<2yr)", "Mid (2–5yr)", "Long (6+yr)"]
        ).astype(str)

    # Income tier (economic class proxy)
    if "annual_inc" in X.columns:
        sf["income_tier"] = pd.qcut(
            X["annual_inc"], q=3,
            labels=["Low income", "Mid income", "High income"],
            duplicates="drop"
        ).astype(str)

    # Home ownership group
    if "home_ownership" in X.columns:
        sf["home_ownership"] = X["home_ownership"].astype(str)

    return sf


# ── Metric computation ────────────────────────────────────────────────────────

def audit_sensitive_attr(model, X_test, y_test, sensitive_col, y_prob=None):
    """Compute MetricFrame for a given sensitive attribute."""
    if y_prob is None:
        y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    mf = MetricFrame(
        metrics={
            "accuracy":        accuracy_score,
            "selection_rate":  selection_rate,
            "false_pos_rate":  false_positive_rate,
            "false_neg_rate":  false_negative_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_col
    )

    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_col)
    dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=sensitive_col)
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_col)

    return mf, {"dpd": dpd, "dpr": dpr, "eod": eod}


def plot_metric_by_group(mf, metric_name, group_name, title):
    """Bar chart of a metric broken down by sensitive group."""
    by_group = mf.by_group[metric_name]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E25C5C" if v == by_group.max() else "#4A90D9" for v in by_group]
    ax.bar(by_group.index, by_group.values, color=colors)
    ax.axhline(mf.overall[metric_name], linestyle="--", color="#333", linewidth=1, label="Overall")
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(metric_name)
    ax.set_xlabel(group_name)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = f"{REPORTS_DIR}/fairness_{group_name}_{metric_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Post-processing mitigation ────────────────────────────────────────────────

def mitigate_with_threshold_optimizer(model, X_train, y_train, X_test, y_test, sensitive_train, sensitive_test):
    """
    Apply ThresholdOptimizer to reduce demographic parity difference
    while preserving as much accuracy as possible.
    """
    print("\nApplying ThresholdOptimizer for fairness mitigation ...")
    mitigator = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        objective="balanced_accuracy_score",
        predict_method="predict_proba"
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    y_mitigated = mitigator.predict(X_test, sensitive_features=sensitive_test)

    dpd_before = demographic_parity_difference(
        y_test, (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int),
        sensitive_features=sensitive_test
    )
    dpd_after = demographic_parity_difference(
        y_test, y_mitigated, sensitive_features=sensitive_test
    )
    print(f"  DPD before: {dpd_before:.4f}  →  after: {dpd_after:.4f}")
    return mitigator


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model and test data ...")
    try:
        model = joblib.load(f"{MODELS_DIR}/lightgbm.pkl")
    except FileNotFoundError:
        model = joblib.load(f"{MODELS_DIR}/xgboost.pkl")

    X_test = pd.read_parquet("data/X_test.parquet")
    y_test = pd.read_parquet("data/y_test.parquet")["default"]

    sf = make_sensitive_features(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    all_metrics = {}

    for attr in sf.columns:
        print(f"\nAuditing: {attr}")
        mf, scalar_metrics = audit_sensitive_attr(model, X_test, y_test, sf[attr], y_prob)

        print(f"  Demographic Parity Difference : {scalar_metrics['dpd']:.4f}")
        print(f"  Demographic Parity Ratio      : {scalar_metrics['dpr']:.4f}")
        print(f"  Equalized Odds Difference     : {scalar_metrics['eod']:.4f}")
        print(f"  Metrics by group:\n{mf.by_group.round(4)}")

        all_metrics[attr] = scalar_metrics

        plot_metric_by_group(mf, "selection_rate",  attr, f"Selection Rate by {attr}")
        plot_metric_by_group(mf, "false_pos_rate",  attr, f"False Positive Rate by {attr}")
        plot_metric_by_group(mf, "false_neg_rate",  attr, f"False Negative Rate by {attr}")

    # Save scalar metrics summary
    with open(f"{REPORTS_DIR}/fairness_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved fairness metrics to {REPORTS_DIR}/fairness_metrics.json")

    # Interpretation guide
    print("\n── Interpretation Guide ─────────────────────────────────")
    print("  DPD (Demographic Parity Difference):")
    print("    |DPD| < 0.05  → Low disparity (acceptable)")
    print("    |DPD| 0.05–0.1 → Moderate — investigate")
    print("    |DPD| > 0.1  → High — mitigation recommended")
    print("  DPR (Ratio): Closer to 1.0 = more equal")
    print("  EOD (Equalized Odds Difference): 0 = equal TPR & FPR across groups")


if __name__ == "__main__":
    main()
