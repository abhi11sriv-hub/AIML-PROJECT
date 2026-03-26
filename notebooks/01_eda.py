# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
# ---

# %% [markdown]
# # Loan Default Risk — Exploratory Data Analysis
#
# **Dataset:** LendingClub Loan Data (2007–2018) from Kaggle  
# **Target:** `loan_status` → `default` (1 = Charged Off, 0 = Fully Paid)
#
# This notebook covers:
# 1. Dataset overview & shape
# 2. Target distribution & class imbalance
# 3. Missing value analysis
# 4. Numerical feature distributions
# 5. Categorical feature breakdown
# 6. Correlation with default
# 7. Key insights for modeling

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import preprocess_pipeline

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
sns.set_theme(style="whitegrid", palette="muted")

# %% [markdown]
# ## 1. Load raw data (10% sample for EDA speed)

# %%
# To use full dataset: sample_frac=1.0
df_raw = pd.read_csv("data/loan.csv", low_memory=False, nrows=100_000)
print(f"Shape: {df_raw.shape}")
df_raw.head(3)

# %% [markdown]
# ## 2. Target distribution

# %%
target_counts = df_raw["loan_status"].value_counts()
print(target_counts)

binary = df_raw[df_raw["loan_status"].isin(["Fully Paid", "Charged Off"])]
rate = (binary["loan_status"] == "Charged Off").mean()
print(f"\nDefault rate (Charged Off): {rate:.2%}")

fig, ax = plt.subplots(figsize=(6, 4))
binary["loan_status"].value_counts().plot.bar(ax=ax, color=["#4A90D9", "#E25C5C"])
ax.set_title("Loan Status Distribution", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/figures/eda_target_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Missing value analysis

# %%
null_pct = df_raw.isna().mean().sort_values(ascending=False)
high_null = null_pct[null_pct > 0.1]

fig, ax = plt.subplots(figsize=(8, max(4, len(high_null) * 0.35)))
high_null.plot.barh(ax=ax, color="#888")
ax.set_xlabel("Fraction missing")
ax.set_title(f"Features with >10% missing values ({len(high_null)} cols)", fontsize=12)
ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="50% threshold")
ax.legend()
plt.tight_layout()
plt.savefig("reports/figures/eda_missing_values.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Numerical feature distributions

# %%
num_features = ["loan_amnt", "int_rate", "annual_inc", "dti",
                "fico_range_low", "revol_util", "installment"]

binary["default"] = (binary["loan_status"] == "Charged Off").astype(int)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feat in enumerate(num_features):
    if feat in binary.columns:
        ax = axes[i]
        binary.groupby("default")[feat].plot.kde(ax=ax, legend=(i == 0))
        ax.set_title(feat, fontsize=11)
        ax.set_ylabel("")
        if i == 0:
            ax.legend(["Fully Paid", "Charged Off"])

for j in range(len(num_features), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Feature distributions: Fully Paid vs Charged Off", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("reports/figures/eda_feature_distributions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Default rate by categorical features

# %%
cat_features = ["grade", "home_ownership", "purpose", "term", "verification_status"]

fig, axes = plt.subplots(1, len(cat_features), figsize=(18, 5))

for ax, col in zip(axes, cat_features):
    if col in binary.columns:
        rate_by_cat = binary.groupby(col)["default"].mean().sort_values(ascending=False)
        rate_by_cat.plot.bar(ax=ax, color="#4A90D9")
        ax.set_title(f"Default rate by {col}", fontsize=11)
        ax.set_ylabel("Default rate")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("reports/figures/eda_categorical_default_rates.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Correlation heatmap

# %%
df_processed = preprocess_pipeline("data/loan.csv", sample_frac=0.05)

num_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
corr = df_processed[num_cols].corr()["default"].drop("default").sort_values()

fig, ax = plt.subplots(figsize=(6, 8))
corr.plot.barh(ax=ax, color=["#E25C5C" if v > 0 else "#4A90D9" for v in corr])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature correlation with default", fontsize=12)
ax.set_xlabel("Pearson correlation")
plt.tight_layout()
plt.savefig("reports/figures/eda_correlation_with_default.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Key Insights
#
# | Insight | Implication |
# |---------|-------------|
# | ~20% default rate — class imbalance | Use SMOTE or class_weight |
# | `int_rate`, `grade` strongly correlated with default | Top predictive features |
# | `annual_inc` has extreme outliers | Log-transform or cap at 99th pct |
# | Several cols >50% null (`mths_since_last_delinq`) | Drop before modeling |
# | `sub_grade` encodes LendingClub's own risk rating | Potential leakage — audit carefully |
# | Default rate rises steeply from Grade A → G | grade is a strong signal |

print("EDA complete. Figures saved to reports/figures/")
