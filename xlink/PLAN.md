# Customer Churn Prediction: Statistical ML Approach

## Executive Summary

Based on the comprehensive statistical analysis of 7,148 customer records, we identify key data quality issues and patterns that will inform our ML pipeline design. The dataset exhibits moderate churn rate (26.58%), significant missing values in critical features, and data entry inconsistencies requiring careful preprocessing.

## 1. Data Quality Assessment

### 1.1 Missing Value Analysis
From `summary_statistics.json`, we observe critical missing patterns:

| Feature | Missing Count | Missing % | Impact | Strategy |
|---------|--------------|-----------|---------|----------|
| `lifetime_spend` | 867 | 12.13% | HIGH - Strong predictor | Impute using `monthly_fee × months_with_provider` |
| `payment_method` | 714 | 9.99% | MEDIUM | Mode imputation within contract segments |
| `months_with_provider` | 356 | 4.98% | HIGH - Key tenure metric | Regression-based imputation |

**Rationale**: The strong correlation between `lifetime_spend` and `months_with_provider` (r=0.80) and `monthly_fee` (r=0.55) enables accurate imputation rather than deletion.

### 1.2 Data Quality Issues

#### Categorical Inconsistencies in `internet_plan`:
- Main categories: "Fiber optic" (3,129), "DSL" (2,469), "No" (1,543)
- Data entry errors: "Fiber optiic", "Fiber opttic", "iFber optic", "FFiber optic", "Fiiber optic", "Fibe optic"
- **Action**: String standardization and fuzzy matching correction

#### Payment Method Inconsistencies:
- Duplicate entries with case variations: "Electronic check" vs "electronic check" vs "ELECTRONIC CHECK"
- **Action**: Lowercase standardization and consolidation

### 1.3 Class Imbalance
- Churn rate: 26.58% (1,900 churned / 5,248 retained)
- Imbalance ratio: 1:2.76
- **Strategy**: Moderate imbalance - use class weights and probability calibration

## 2. Feature Engineering Strategy

### 2.1 High-Value Features (Based on Churn Correlation)

#### Strong Predictors:
1. **Contract Type** - Massive churn differential:
   - Month-to-month: 42.86% churn
   - One year: 11.07% churn  
   - Two year: 2.85% churn
   - **Encoding**: Ordinal encoding (0: Two year, 1: One year, 2: Month-to-month)

2. **Tenure (`months_with_provider`)** - Strong negative correlation with churn (r=-0.35)
   - **Transformation**: Create tenure bins: [0-12], [13-24], [25-48], [49+] months

3. **Internet Plan Type** - Fiber optic shows 41.96% churn vs DSL 19.04%
   - **Encoding**: Target encoding with smoothing (high cardinality after cleaning)

4. **Payment Method** - Electronic check: 44.92% churn vs others ~15-19%
   - **Encoding**: Binary flag for high-risk payment methods

### 2.2 Feature Interactions

Based on correlation analysis:
- `lifetime_spend` = f(`monthly_fee`, `months_with_provider`) - Multicollinearity concern
- **Decision**: Drop `lifetime_spend` to avoid leakage and multicollinearity

### 2.3 Encoding Strategy

| Feature Type | Features | Encoding Method | Rationale |
|-------------|----------|-----------------|-----------|
| Binary | `gender`, `partner`, `dependents`, `phone_service`, `paperless_billing` | Label Encoding (0/1) | Natural binary representation |
| Low Cardinality Nominal | `extra_lines`, `addon_*` services | One-Hot Encoding | 3 categories each, interpretable |
| Ordinal | `contract_type` | Ordinal Encoding | Clear ordering by commitment level |
| High Cardinality | `payment_method` (after cleaning) | Target Encoding | 4 main categories with varying churn rates |
| Numerical | `monthly_fee`, `months_with_provider` | StandardScaler | Different scales (fee: 8.74-417.96, tenure: 0-72) |

### 2.4 Feature Selection Rationale

**Drop Features**:
- `account_id`, `customer_hash`: No predictive value (unique identifiers)
- `lifetime_spend`: Redundant with monthly_fee × tenure, potential leakage
- `marketing_opt_in`: Negligible correlation with churn (r=0.0097)

## 3. Modeling Approach

### 3.1 Algorithm Selection

**Primary Model: XGBoost**
- Handles mixed data types naturally
- Built-in missing value handling
- Captures non-linear patterns (e.g., tenure effects)
- Feature importance for interpretability

**Baseline Model: Logistic Regression**
- Simple, interpretable coefficients
- Fast training and inference
- Probability calibration benchmark

### 3.2 Class Imbalance Strategy

Given 26.58% churn rate:
1. **Class Weights**: `scale_pos_weight = 2.76` in XGBoost
2. **Threshold Optimization**: Adjust decision threshold based on business costs
3. **Stratified Splits**: Maintain class distribution in train/val/test

### 3.3 Evaluation Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| ROC-AUC | > 0.85 | Overall discrimination ability |
| PR-AUC | > 0.65 | Performance on minority class |
| Recall@Precision=0.8 | > 0.50 | Business constraint: minimize false positives |
| Calibration (Brier Score) | < 0.15 | Accurate probability estimates for risk scoring |

### 3.4 Segment Analysis Plan

Based on data patterns, analyze performance by:
1. **Contract Type**: Expect model to perform differently across commitment levels
2. **Tenure Bands**: Early churn (0-6 months) vs late churn patterns  
3. **Service Bundle**: Internet-only vs multi-service customers
4. **Senior Citizens**: 41.7% churn rate vs 23.7% for non-seniors

## 4. Production Pipeline Design

### 4.1 Data Pipeline
```python
Pipeline = [
    DataCleaner(),           # Fix categorical inconsistencies
    MissingValueImputer(),   # Smart imputation based on correlations
    FeatureEncoder(),        # Mixed encoding strategies
    FeatureScaler(),         # Numerical standardization
    Model()                  # XGBoost with calibration
]
```

### 4.2 Training/Serving Consistency
- Serialize entire pipeline (not just model)
- Version control preprocessing parameters
- Use same pandas/numpy versions in training and serving

### 4.3 Monitoring Strategy

**Data Drift Monitoring**:
- Track distribution shifts in `contract_type`, `monthly_fee`
- Alert if new categorical values appear (data entry errors)
- Monitor missing value rates

**Performance Monitoring**:
- Weekly churn rate tracking (baseline: 26.58%)
- Monthly recalibration check
- Segment-level performance degradation alerts

### 4.4 Retraining Triggers
1. Churn rate deviation > ±5% from baseline
2. ROC-AUC degradation > 0.05
3. New product/plan introduction
4. Quarterly scheduled retraining

## 5. Risk Assessment

### 5.1 Data Risks
- **Missing Value Imputation**: May introduce bias if missing pattern changes
- **Mitigation**: Monitor imputation statistics in production

### 5.2 Model Risks  
- **Overfitting to Contract Type**: Model may rely too heavily on this feature
- **Mitigation**: Feature importance monitoring, ensemble with simpler models

### 5.3 Business Risks
- **False Positives**: Unnecessary retention offers to stable customers
- **Mitigation**: Probability thresholding, human review for high-value accounts

## 6. Implementation Timeline

1. **Data Cleaning & Feature Engineering** (30 min)
   - Fix categorical inconsistencies
   - Implement imputation strategy
   - Create engineered features

2. **Model Training & Evaluation** (30 min)
   - Train XGBoost and Logistic Regression
   - Optimize thresholds
   - Perform segment analysis

3. **API Development** (20 min)
   - FastAPI service with preprocessing pipeline
   - Input validation
   - Probability and decision endpoints

4. **Documentation & Testing** (10 min)
   - API documentation
   - Example requests
   - Performance benchmarks

## Conclusion

The statistical analysis reveals a dataset with moderate quality issues but strong predictive signals. The proposed pipeline addresses data inconsistencies while preserving the predictive power of key features like contract type and tenure. The XGBoost model with careful preprocessing and calibration should achieve ROC-AUC > 0.85 while maintaining interpretability through feature importance analysis.
