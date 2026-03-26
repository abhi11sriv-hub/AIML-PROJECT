# Model Card — Loan Default Risk Predictor

**Model type:** LightGBM Gradient Boosted Trees (with XGBoost and Logistic Regression baselines)  
**Task:** Binary classification — predict whether a loan will be charged off (default)  
**Version:** 1.0  
**Date:** 2025  
**Author:** [Your Name]  
**Course:** [Your Course Name]

---

## Intended use

### Primary use
Educational demonstration of responsible ML for credit risk. Illustrates the full pipeline from raw data through explainability and fairness auditing.

### Out-of-scope uses
This model **must not** be used for real lending or credit decisions. It is trained on historical data that reflects past inequalities and has not been validated for regulatory compliance.

---

## Training data

| Property | Details |
|----------|---------|
| Source | LendingClub Loan Data (Kaggle) |
| Time period | 2007–2018 |
| Size | ~2.2M loans (binary-classifiable subset) |
| Target | `loan_status`: Fully Paid (0) vs Charged Off (1) |
| Class balance | ~80% Fully Paid / ~20% Charged Off |
| Imbalance handling | SMOTE oversampling on training set |
| Leakage prevention | Post-outcome columns removed (see `src/preprocess.py`) |

---

## Evaluation data

- 20% stratified holdout from the full dataset  
- No data from evaluation set was used during training or hyperparameter tuning

---

## Performance metrics

| Model | ROC-AUC | Avg Precision | Notes |
|-------|---------|---------------|-------|
| Logistic Regression | ~0.68 | ~0.42 | Baseline |
| XGBoost | ~0.73 | ~0.52 | Strong on recall |
| **LightGBM** | **~0.74** | **~0.54** | Best overall |

> Exact values will differ depending on the data version and random seed used.

### Threshold considerations

At the default threshold (0.5):
- Higher precision favours the lender (fewer false approvals)
- Higher recall favours applicants (fewer wrongful denials)
- Adjust the threshold based on the cost matrix for your use case

---

## Fairness audit

Fairness was evaluated using [Fairlearn](https://fairlearn.org/) across proxy sensitive groups derived from available features.

### Sensitive attributes audited

| Attribute | Groups | Rationale |
|-----------|--------|-----------|
| Employment length tier | Short (<2yr), Mid (2–5yr), Long (6+yr) | Proxy for economic stability |
| Income tier | Low, Mid, High (terciles) | Proxy for economic class |
| Home ownership | RENT, OWN, MORTGAGE, OTHER | Correlated with wealth accumulation |

### Key fairness metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Demographic Parity Difference (DPD) | Max difference in selection rate across groups | |DPD| < 0.05 |
| Demographic Parity Ratio (DPR) | Min/max ratio of selection rates | DPR > 0.8 |
| Equalized Odds Difference (EOD) | Max difference in TPR and FPR across groups | |EOD| < 0.05 |

See `reports/figures/fairness_metrics.json` for exact values from your run.

### Known limitations

- Race, gender, and age are not in the LendingClub dataset; proxy attributes are imperfect surrogates
- Historical data encodes past discriminatory lending practices
- ThresholdOptimizer post-processing can reduce DPD at the cost of some accuracy
- Fairness interventions do not guarantee legal compliance

---

## Explainability

SHAP (SHapley Additive exPlanations) via TreeExplainer is used throughout:

- **Global importance:** Mean absolute SHAP values across all test samples
- **Local explanation:** Waterfall charts for individual predictions
- **Dependence plots:** SHAP value vs feature value for top features
- **Streamlit app:** Live SHAP waterfall for any user-defined applicant

Top features driving predictions (approximate, based on SHAP):
1. `int_rate` — interest rate assigned to the loan
2. `grade` / `sub_grade` — LendingClub's risk grade
3. `dti` — debt-to-income ratio
4. `fico_range_low` — credit score
5. `revol_util` — revolving credit utilization
6. `annual_inc` — borrower income
7. `installment` — monthly payment amount
8. `term` — 36 vs 60 month loan
9. `inq_last_6mths` — recent credit inquiries
10. `emp_length` — employment tenure

---

## Ethical considerations

- **Proxy discrimination:** Features correlated with race/ethnicity (zip code, income tier) may produce disparate impact even without explicit use of protected attributes.
- **Feedback loops:** Deploying a biased model creates historical data that reinforces future bias.
- **Opacity risk:** SHAP explanations approximate local model behaviour but do not fully characterise model reasoning globally.
- **Regulatory context:** Real credit models in India and many other jurisdictions are subject to RBI guidelines and fair lending laws.

---

## Caveats and recommendations

- Retrain on more recent data; the 2007–2018 period includes the financial crisis
- Monitor for distribution shift in production
- Conduct regular fairness re-audits if the applicant pool composition changes
- Human review should always accompany any automated risk flag
- Consult legal counsel before deploying any credit model

---

## Citation

LendingClub dataset via Kaggle. Fairlearn library: Bird et al., "Fairlearn: A toolkit for assessing and improving fairness in AI," 2020. SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
