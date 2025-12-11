# Medicare Cost Prediction: Two-Part Model with Machine Learning  
Biostat 625 Final Project · University of Michigan

This project builds a Two-Part modeling framework to predict annual Medicare costs
using the CMS 2008–2010 DE-SynPUF dataset. We compare GLMs with XGBoost-based
models, focusing on predictive accuracy, calibration, and subgroup performance.

- Scope: 116,352 beneficiaries (81,447 train / 34,905 test)
- Task: Predict annual total Medicare payment (`total_payment`)
- Framework: Two-Part model (Frequency × Severity) + GLM / XGBoost
- Baseline: Single-equation GLM (industry-style pricing model)

---

## Highlights

- **Error reduction**:  
  - MAE: **$1,639** vs GLM **$2,013** (**18.6%** improvement)  
  - RMSE: **$5,109** vs GLM **$5,872** (**13.0%** improvement)
- **Stability** (5-fold CV):  
  - MAE CV = **2.7%**, RMSE CV = **5.4%**  
  - RMSLE\_NonZero = **0.645 ± 0.006** (CV < **1%**)
- **Subgroups**:  
  - XGBoost outperforms GLM by **9–13%** (multi-morbidity groups)  
  - **11–21%** gain in hospitalized vs non-hospitalized subgroups  
  - Up to **25.4%** improvement in high-risk, hospitalized patients
- **Calibration**:  
  - Slope ≈ **1.05**, correlation ≈ **0.9998** (near-ideal)

---

## Raw Data Availability

This project uses several CMS DE-SynPUF files (2008–2010), including Beneficiary,
Carrier, Inpatient, and Outpatient claims. Some of these raw files are larger
than 1–2 GB. Because GitHub enforces a 100 MB per-file limit, the original raw
data **cannot be stored in this repo**.

- The `data/` folder is intentionally left empty.
- All preprocessing steps are implemented in `data_prep.R`.
- The full raw files are submitted separately for grading.
- Anyone with access to CMS DE-SynPUF can reproduce the analytic dataset by
  placing the raw files into `data/` and running `data_prep.R`.

This keeps the project reproducible while complying with GitHub storage limits.

---

## Repository Structure

- `run_pipeline.R` – End-to-end pipeline (data prep → modeling → subgroup analysis)
- `data_prep.R` – Preprocessing and feature engineering  
  - Builds an analysis dataset with no data leakage  
  - Features: demographics, chronic conditions, visit counts (no payment fields)
- `modeling.R` – Main Two-Part models  
  - Part 1 (Frequency): Logistic GLM + XGBoost classifier  
    - Features: age, age², sex, race, `chronic_count`  
    - No visit counts (avoid leakage in any-cost prediction)  
  - Part 2 (Severity): Lognormal GLM, Gamma GLM + XGBoost regressor  
    - Features: demographics + visit counts  
  - Outputs: `output/model_performance_v3.csv`, `output/test_predictions_v3.rds`

    - **Note**: The current `modeling.R` may not be compatible with newer versions of XGBoost. A supplementary `modeling.R` file has been provided by team members that is compatible with the latest XGBoost version. Please refer to the supplementary materials if you encounter XGBoost compatibility issues.
    - **注**：当前的 `modeling.R` 文件可能与较新的 XGBoost 版本不兼容。团队成员已提供了一份与之兼容的最新 XGBoost 版本专用的 `modeling.R` 文件。若您遇到 XGBoost 兼容性问题，请查阅相关补充资料。
- `subgroup_analysis.R` – Performance by multi-morbidity and hospitalization status  
  - Outputs: `output/subgroup_analysis_*.csv`
- `eda.R`, `run_eda.R` – Exploratory data analysis helpers
- `output/` – Model results, calibration plots, residual diagnostics

> Note: For newer versions of `xgboost`, a supplementary `modeling.R` is provided
> (same methodology, updated API).

---

## Modeling Overview

We model annual cost as:

> **Cost = P(Y > 0) × E[Y | Y > 0]**

- **Part 1 – Frequency**  
  - Target: `any_cost = I(total_payment > 0)`  
  - Models: Logistic GLM, XGBoost classifier  
  - Features: demographics only (no utilization variables)

- **Part 2 – Severity**  
  - Target: `total_payment` among `total_payment > 0`  
  - Models: Lognormal GLM, Gamma GLM, XGBoost regressor  
  - Features: demographics + visit counts (IP, OP, Carrier)

Key metrics: MAE, RMSE, RMSLE, RMSLE\_NonZero, subgroup deltas vs GLM.

---

## Usage

Install required packages:

```r
install.packages(c(
  "tidyverse",
  "data.table",
  "xgboost",
  "caret",
  "pROC",
  "broom"
))
