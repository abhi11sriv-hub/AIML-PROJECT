"""
train.py
--------
Train and evaluate Logistic Regression, XGBoost, and LightGBM models
for loan default prediction. Saves the best model to models/.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Feature selection ────────────────────────────────────────────────────────

TOP_FEATURES = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "total_pymnt_inv", "term", "emp_length", "home_ownership",
    "verification_status", "purpose", "grade", "sub_grade",
    "addr_state", "initial_list_status", "application_type"
]


def load_processed(path: str = "data/processed.parquet"):
    df = pd.read_parquet(path)
    feats = [f for f in TOP_FEATURES if f in df.columns]
    X = df[feats]
    y = df["default"]
    print(f"Features: {len(feats)}  |  Samples: {len(y)}  |  Default rate: {y.mean():.2%}")
    return X, y


def split_data(X, y, test_size=0.2, val_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size), stratify=y_train, random_state=42
    )
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train):
    """Oversample minority class to address class imbalance."""
    print("Applying SMOTE ...")
    sm = SMOTE(random_state=42, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {y_res.value_counts().to_dict()}")
    return X_res, y_res


# ── Model definitions ────────────────────────────────────────────────────────

def train_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_xgboost(X_train, y_train, X_val, y_val):
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        early_stopping_rounds=30,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test, scaler=None):
    if scaler is not None:
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test

    y_prob = model.predict_proba(X_input)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    cr   = classification_report(y_test, y_pred, target_names=["Fully Paid", "Charged Off"])

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Classification Report:\n{cr}")

    return {"name": name, "auc": auc, "avg_precision": ap}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    X, y = load_processed()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Apply SMOTE only to training set
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    results = []

    # 1. Logistic Regression (baseline)
    print("\nTraining Logistic Regression ...")
    lr_model, lr_scaler = train_logistic(X_train_res, y_train_res)
    results.append(evaluate("Logistic Regression", lr_model, X_test, y_test, scaler=lr_scaler))
    joblib.dump({"model": lr_model, "scaler": lr_scaler}, f"{MODELS_DIR}/logistic.pkl")

    # 2. XGBoost
    print("\nTraining XGBoost ...")
    xgb_model = train_xgboost(X_train_res, y_train_res, X_val, y_val)
    results.append(evaluate("XGBoost", xgb_model, X_test, y_test))
    joblib.dump(xgb_model, f"{MODELS_DIR}/xgboost.pkl")

    # 3. LightGBM
    print("\nTraining LightGBM ...")
    lgbm_model = train_lightgbm(X_train_res, y_train_res, X_val, y_val)
    results.append(evaluate("LightGBM", lgbm_model, X_test, y_test))
    joblib.dump(lgbm_model, f"{MODELS_DIR}/lightgbm.pkl")

    # Save best model metadata
    best = max(results, key=lambda r: r["auc"])
    print(f"\nBest model: {best['name']}  (AUC={best['auc']:.4f})")
    with open(f"{MODELS_DIR}/best_model_info.json", "w") as f:
        json.dump(best, f, indent=2)

    # Save test split for explainability
    X_test.to_parquet("data/X_test.parquet", index=False)
    y_test.to_frame().to_parquet("data/y_test.parquet", index=False)
    print("Saved test data and model artifacts.")


if __name__ == "__main__":
    main()
