# Customer Churn Prediction — Technical Report  

*Author: X-Link ML Engineering Team*  
*Date: {{auto-filled at build time}}*  

---

## 1. Executive Overview
The goal of this project is to deliver a production-ready, explainable pipeline that predicts customer churn for X Link Logistics’ connectivity products.  Using 7 148 historical customer records we executed an end-to-end workflow:

1. Automated EDA (`data-exploration/explore_data.py`)  
2. Robust feature cleaning / engineering (`pipeline/data`)  
3. Baseline & advanced modelling (`pipeline/train`)  
4. Hold-out + k-fold evaluation (`evaluate`, `report/modeling/*`)  
5. FastAPI micro-service for real-time inference (`api/*`)  

Key results  
• Hold-out ROC-AUC ≈ 0.85, PR-AUC ≈ 0.66  
• 5-fold ROC-AUC mean = 0.842 ± 0.008 (see report/modeling/kfold\*.json)  
• Top drivers: contract length, tenure, internet type, payment method  

---

## 2. Data Exploration & Quality Findings
Source statistics are persisted in `data-exploration/output/summary_statistics.json`; highlights:

| Aspect | Finding | Impact / Action |
|--------|---------|-----------------|
| Missing | `months_with_provider` (4.9 %), `payment_method` (9.9 %), `lifetime_spend` (12.1 %) | Smart imputer (regression / mode) instead of row-drop |
| Inconsistent categories | 9 typos in `internet_plan`; 12 variants of `payment_method` | Fuzzy-map & lower-case normalisation in `DataCleaner` |
| Class imbalance | 26.6 % churn | Class-weights; threshold tuning |
| Correlations | `months_with_provider` ↔︎ `lifetime_spend` (r = 0.80) | Drop `lifetime_spend` to avoid leakage |

Full tables/plots are in `output/`.

---

## 3. Feature Engineering Decisions

| Feature set | Technique | Rationale |
|-------------|-----------|-----------|
| Binary flags (`gender`, `partner`, …) | Label encode 0/1 | Direct & interpretable |
| Add-ons / lines | One-hot | Low cardinality (≤3) |
| `contract_type` | Ordinal (2 > 1 > 0) | Captures ordering of commitment |
| `payment_method`, `internet_plan` | Target encoding (smoothed) | Captures risk gradient while limiting dummies |
| Derived | Tenure bins, high-risk payment flag, multi-service indicator, high-value customer flag | Encapsulate non-linear patterns seen in EDA |
| Scaling | StandardScaler on numeric | Required for LR; neutral for tree models |

`lifetime_spend`, `account_id`, `customer_hash`, `marketing_opt_in` were **dropped**.

---

## 4. Modelling Approach

1. **Baseline**: Logistic Regression with elasticnet → ROC-AUC ≈ 0.84 (see k-fold JSON).  
2. **Primary**: XGBoost (`n_estimators=100`, `max_depth=4`, `scale_pos_weight=2.76`).  
3. Calibration: Platt-scaling via `CalibratedClassifierCV`.  
4. Threshold optimisation: maximise F1 subject to precision ≥ 0.80.

Hyper-parameter search (grid + early-stopped Bayesian optimisation; scripts in `pipeline/search/`) confirmed diminishing returns; therefore we retained a compact model for latency reasons.

---

## 5. Evaluation Results

### 5.1 Hold-out (20 %)  

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.851** |
| PR-AUC | 0.664 |
| Accuracy | 0.804 |
| Precision (@thr = 0.57) | 0.812 |
| Recall (@thr = 0.57) | 0.534 |
| F1 | 0.643 |
| Brier score | 0.141 |

### 5.2 5-Fold CV (Logistic baseline)
*From* `report/modeling/kfold_kfold_logistic_20250921_193114.json`

| ROC-AUC | PR-AUC |
|---------|--------|
| 0.842 ± 0.008 | 0.647 ± 0.013 |

### 5.3 Segment Analysis (XGBoost hold-out)

| Segment | N | ROC-AUC |
|---------|---|---------|
| Month-to-month | 786 | 0.79 |
| One-year | 306 | 0.88 |
| Two-year | 344 | 0.91 |
| Senior citizens | 231 | 0.83 |
| Tenure 0-12 | 350 | 0.77 |

Interpretation: model struggles on early-tenure customers; future work will explore survival analysis.

---

## 6. Model Explainability
SHAP analysis (notebook in `notebooks/`) shows:

1. Contract length dominates prediction.  
2. Low tenure + high monthly fee increases churn risk.  
3. Electronic check is a red-flag payment method.

Global and local explanations are logged as artefacts in `results/shap/`.

---

## 7. Deployment & Monitoring Plan

| Area | Strategy |
|------|----------|
| Serving mode | Real-time (REST) via FastAPI container (`api/Dockerfile`) |
| Versioning | Semantic model IDs, Git-tagged preprocessing hash |
| Consistency | `pickle` of full `sklearn` pipeline; pyproject.lock ensures lib parity |
| Observability | Prometheus counters: drift (PSI), latency, business KPI (#retained) |
| Retraining | Trigger if ROC-AUC ↓ > 0.05 **or** PSI > 0.2 **or** quarterly |

---

## 8. Risks & Mitigations
1. **Data drift** — new tariff plans → retrain triggers, canary scoring.  
2. **Over-reliance on contract_type** — monitor feature importance; ensemble with simpler model.  
3. **Imputation bias** — live missing-rate alert; consider data collection fix.  
4. **Ethical bias** — audit impact on senior citizens.  

---

## 9. Future Work

1. **Bayesian HPO** — extend `pipeline/search/optuna_search.py` for full parameter sweep; early results show +0.01 ROC-AUC headroom.  
2. **Time-to-event modelling** — use survival forests to predict *when* churn occurs.  
3. **Uplift / causal modelling** — estimate retention-offer effectiveness.  
4. **Feature store** — move cleaning/encoding to company-wide Feast store for multi-team reuse.  

---

## 10. Conclusion
We delivered a statistically robust, explainable, and deployable churn model that meets target performance while keeping operational complexity low.  The project establishes a solid foundation for iterative improvement and reliable business integration.
