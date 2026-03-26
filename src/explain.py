"""
explain.py
----------
Generate SHAP-based explanations for the best trained model.
Produces summary plots, waterfall charts, and per-feature importance CSV.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR  = "models"
REPORTS_DIR = "reports/figures"
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_model_and_data():
    try:
        model = joblib.load(f"{MODELS_DIR}/lightgbm.pkl")
        model_name = "LightGBM"
    except FileNotFoundError:
        model = joblib.load(f"{MODELS_DIR}/xgboost.pkl")
        model_name = "XGBoost"

    X_test = pd.read_parquet("data/X_test.parquet")
    y_test = pd.read_parquet("data/y_test.parquet")["default"]
    return model, model_name, X_test, y_test


def compute_shap_values(model, X_sample):
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values


def plot_summary(shap_values, X_sample, model_name):
    """Global feature importance — beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP Summary — {model_name}", fontsize=14, pad=12)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_waterfall(shap_values, idx, model_name, label):
    """Single-prediction waterfall chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    plt.title(f"SHAP Waterfall — {model_name} | Sample {idx} ({label})", fontsize=13)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/shap_waterfall_sample_{idx}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_mean_importance(shap_values, X_sample):
    """Bar chart of mean absolute SHAP values (global importance)."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(f"{REPORTS_DIR}/feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance_df.head(15)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#4A90D9")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 Features by Mean Absolute SHAP", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/shap_importance_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return importance_df


def plot_dependence(shap_values, X_sample, feature):
    """Dependence plot for a specific feature."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.scatter(shap_values[:, feature], show=False)
    plt.title(f"SHAP Dependence — {feature}", fontsize=13)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/shap_dependence_{feature}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("Loading model and test data ...")
    model, model_name, X_test, y_test = load_model_and_data()

    # Use a representative sample of 2000 rows for speed
    sample_idx = np.random.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

    print(f"Computing SHAP values for {len(X_sample)} samples ...")
    explainer, shap_values = compute_shap_values(model, X_sample)

    print("Generating plots ...")
    plot_summary(shap_values, X_sample, model_name)
    importance_df = plot_mean_importance(shap_values, X_sample)

    # Waterfall for a default and a non-default example
    default_idx    = y_sample[y_sample == 1].index[0]
    non_default_idx = y_sample[y_sample == 0].index[0]
    plot_waterfall(shap_values, default_idx,     model_name, "Charged Off")
    plot_waterfall(shap_values, non_default_idx, model_name, "Fully Paid")

    # Dependence on most important feature
    top_feature = importance_df.iloc[0]["feature"]
    plot_dependence(shap_values, X_sample, top_feature)

    print("\nTop 10 features by SHAP importance:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
