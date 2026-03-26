"""
app.py
------
Streamlit dashboard for loan default risk prediction.
Run with: streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/lightgbm.pkl"), "LightGBM"
    except FileNotFoundError:
        return joblib.load("models/xgboost.pkl"), "XGBoost"

@st.cache_resource
def load_explainer(model):
    return shap.TreeExplainer(model)

model, model_name = load_model()
explainer = load_explainer(model)

# ── Sidebar — Input form ──────────────────────────────────────────────────────
st.sidebar.header("Applicant Information")

loan_amnt      = st.sidebar.number_input("Loan amount ($)", 1000, 40000, 10000, step=500)
int_rate       = st.sidebar.slider("Interest rate (%)", 5.0, 30.0, 13.0, step=0.1)
term           = st.sidebar.selectbox("Term (months)", [36, 60])
annual_inc     = st.sidebar.number_input("Annual income ($)", 10000, 500000, 60000, step=1000)
emp_length     = st.sidebar.slider("Employment length (years)", 0, 10, 3)
dti            = st.sidebar.slider("Debt-to-income ratio", 0.0, 40.0, 15.0, step=0.5)
fico_low       = st.sidebar.slider("FICO score (low)", 580, 850, 690)
inq_last_6mths = st.sidebar.number_input("Credit inquiries (last 6 mo.)", 0, 10, 1)
revol_util     = st.sidebar.slider("Revolving utilization (%)", 0.0, 100.0, 45.0)
delinq_2yrs    = st.sidebar.number_input("Delinquencies (last 2 yr.)", 0, 10, 0)
home_ownership = st.sidebar.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
purpose        = st.sidebar.selectbox("Loan purpose", [
    "debt_consolidation", "credit_card", "home_improvement",
    "other", "major_purchase", "medical", "small_business"
])

# ── Build feature vector ──────────────────────────────────────────────────────
HOME_MAP    = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
PURPOSE_MAP = {
    "debt_consolidation": 0, "credit_card": 1, "home_improvement": 2,
    "other": 3, "major_purchase": 4, "medical": 5, "small_business": 6
}

FEATURE_COLS = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "total_pymnt_inv", "term", "emp_length",
    "home_ownership", "verification_status", "purpose",
    "grade", "sub_grade", "addr_state", "initial_list_status", "application_type"
]

installment = round((loan_amnt * (int_rate / 100 / 12)) /
                    (1 - (1 + int_rate / 100 / 12) ** (-term)), 2)

sample = pd.DataFrame([{
    "loan_amnt": loan_amnt,
    "funded_amnt": loan_amnt,
    "int_rate": int_rate,
    "installment": installment,
    "annual_inc": annual_inc,
    "dti": dti,
    "delinq_2yrs": delinq_2yrs,
    "fico_range_low": fico_low,
    "fico_range_high": fico_low + 4,
    "inq_last_6mths": inq_last_6mths,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": int(annual_inc * revol_util / 100 * 0.05),
    "revol_util": revol_util,
    "total_acc": 20,
    "total_pymnt_inv": 0,
    "term": term,
    "emp_length": emp_length,
    "home_ownership": HOME_MAP.get(home_ownership, 0),
    "verification_status": 1,
    "purpose": PURPOSE_MAP.get(purpose, 3),
    "grade": 3,
    "sub_grade": 10,
    "addr_state": 5,
    "initial_list_status": 0,
    "application_type": 0
}])

# ── Prediction ────────────────────────────────────────────────────────────────
prob = model.predict_proba(sample)[0][1]
risk_pct = round(prob * 100, 1)

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("📊 Loan Default Risk Predictor")
st.caption(f"Model: {model_name} · Adjust inputs in the sidebar")

col1, col2, col3 = st.columns(3)

with col1:
    color = "#E25C5C" if prob >= 0.5 else "#2ecc71"
    st.metric("Default probability", f"{risk_pct}%")
    if prob >= 0.5:
        st.error("⚠ High default risk")
    elif prob >= 0.3:
        st.warning("Moderate risk — review carefully")
    else:
        st.success("Low default risk")

with col2:
    st.metric("Monthly installment", f"${installment:,.2f}")
    st.metric("Annual income", f"${annual_inc:,}")

with col3:
    st.metric("Debt-to-income ratio", f"{dti}%")
    st.metric("FICO score", f"{fico_low}–{fico_low+4}")

st.divider()

# ── SHAP Waterfall ────────────────────────────────────────────────────────────
st.subheader("Why this prediction? (SHAP explanation)")

shap_vals = explainer(sample)

fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.waterfall(shap_vals[0], max_display=12, show=False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.caption(
    "Bars pushing **right** (red) increase default risk. "
    "Bars pushing **left** (blue) reduce it. "
    "The base value is the model's average prediction across training data."
)

st.divider()

# ── Risk factor breakdown table ───────────────────────────────────────────────
st.subheader("Key risk factors")

abs_shap = np.abs(shap_vals.values[0])
feat_names = sample.columns.tolist()
shap_signed = shap_vals.values[0]

factor_df = pd.DataFrame({
    "Feature": feat_names,
    "Your value": sample.values[0],
    "SHAP impact": shap_signed,
    "Direction": ["↑ Risk" if s > 0 else "↓ Risk" for s in shap_signed]
}).sort_values("SHAP impact", key=abs, ascending=False).head(10).reset_index(drop=True)

st.dataframe(factor_df, use_container_width=True)

# ── Fairness notice ───────────────────────────────────────────────────────────
with st.expander("ℹ Fairness & model limitations"):
    st.markdown("""
    **This model is for educational purposes only and must not be used for real lending decisions.**

    - The model was trained on historical LendingClub data, which reflects past societal inequalities.
    - Features like employment length and income can act as proxies for protected attributes.
    - A fairness audit (see `src/fairness.py`) measures demographic parity and equalized odds across income and employment groups.
    - No ML model should be the sole basis for credit decisions. Human review is essential.
    - See the [model card](reports/model_card.md) for full performance and fairness documentation.
    """)
