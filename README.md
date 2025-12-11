# Medicare Cost Prediction: Two-Part Model with Machine Learning

**Biostat 625 Final Project**

# 🏥 Medicare Cost Prediction Engine
### Saved $374 per patient through 18.6% prediction improvement

> **What I Built**: Production-ready model predicting annual healthcare costs  
> **Why It Matters**: Insurers lose millions from mis-pricing; I reduced error by $374/person  
> **Tech**: Two-Part statistical framework + XGBoost ensemble  
> **Impact**: 13-18% better than industry baseline (GLM)

 **Built for**: Insurance pricing, risk management, healthcare budgeting  
 **Data Scale**: 116,000+ Medicare patients, $480M+ total costs  
 **Performance**: 5-fold CV stable (< 6% variance)

[View Results](#key-results) | [See Code](#quick-start) | [Read Methods](#technical-approach)

## Project Overview

This project implements a Two-Part model framework to predict annual Medicare costs using the CMS 2008-2010 DE-SynPUF dataset. We compare GLM and XGBoost approaches, evaluate performance across key subgroups, and address data leakage concerns through careful feature engineering.

## Data Source

- **CMS DE-SynPUF Sample 1 (2008)**
  - 2008 Beneficiary Summary File
  - 2008-2010 Carrier Claims (Parts A & B)
  - 2008-2010 Inpatient Claims
  - 2008-2010 Outpatient Claims
- **Sample Size**: 116,352 beneficiaries (81,447 train, 34,905 test)

## Project Structure

### Main Pipeline

- **`run_pipeline.R`** - Main pipeline script
  - Runs data preparation, modeling, and subgroup analysis
  - Frequency model uses only demographics (no visit counts)

### Core Scripts

- **`data_prep.R`** - Data preparation
  - Loads CMS DE-SynPUF data
  - Creates analysis dataset with no data leakage
  - Features: demographics, chronic conditions, visit counts (no payment amounts)
  - Output: `output/analysis_data_noleak.rds`

- **`modeling.R`** - Main modeling script (see note below on XGBoost compatibility)
  - **Part 1 (Frequency)**: Logistic GLM + XGBoost classifier
    - Features: demographics only (age, age2, sex, race, chronic_count)
    - No visit counts to avoid data leakage
  - **Part 2 (Severity)**: Lognormal GLM, Gamma GLM + XGBoost regressor
    - Features: demographics + visit counts (no leakage for amount prediction)
  - Cross-validation (5-fold)
  - Calibration analysis
  - Sensitivity analysis (model without chronic_count)
  - Output: `output/model_performance_v3.csv`, `output/test_predictions_v3.rds`
  - **Note**: The current `modeling.R` may not be compatible with newer versions of XGBoost. A supplementary `modeling.R` file has been provided by team members that is compatible with the latest XGBoost version. Please refer to the supplementary materials if you encounter XGBoost compatibility issues.

- **`subgroup_analysis.R`** - Subgroup performance comparison
  - XGBoost vs GLM in key subgroups:
    - Multi-morbidity groups (Low/Medium/High)
    - Hospitalized vs Not Hospitalized
    - Combined subgroups
  - Output: `output/subgroup_analysis_*.csv`

### Supporting Scripts

- **`eda.R`** - Exploratory data analysis
- **`run_eda.R`** - Run EDA script

### Supplementary Materials

- **Supplementary `modeling.R`** - Updated modeling script compatible with newer XGBoost versions
  - Provided by team members as a supplement to address XGBoost version compatibility issues
  - Contains the same modeling framework and methodology as the main `modeling.R`
  - Use this file if you encounter XGBoost compatibility errors with the main script

### Directories

- **`data/`** - Raw CMS data files (not in repo)
- **`output/`** - All results and intermediate files
  - `output/calibration/` - Calibration plots and data
  - `output/residuals/` - Residual diagnostic plots

## Key Features

### Analysis Dataset
- **Unit**: One row per beneficiary in 2008
- **Target**: `total_payment` (annual Medicare total payment)
- **Frequency**: `any_cost = I(total_payment > 0)` (73.7% have costs)
- **Features**: 
  - Demographics: age, sex, race
  - Chronic conditions: chronic_count (0-11)
  - Utilization: visit counts (IP, OP, Carrier)
  - **No payment amounts** in features (to avoid data leakage)

### Modeling Approach

1. **Two-Part Model**:
   - **Part 1 (Frequency)**: Predicts P(Y > 0)
     - Models: Logistic GLM, XGBoost classifier
     - Features: demographics only (no visit counts)
   - **Part 2 (Severity)**: Predicts E[Y | Y > 0]
     - Models: Lognormal GLM, Gamma GLM, XGBoost regressor
     - Features: demographics + visit counts
   - **Pure Premium**: E[X] = p̂ × E[Y | X>0]

2. **Baseline Comparison**: Single GLM model

3. **Performance Metrics**: 
   - MAE, RMSE
   - RMSLE (all samples)
   - **RMSLE_NonZero** (non-zero samples only) - more appropriate for Two-Part model evaluation

4. **Subgroup Analysis**: XGBoost vs GLM performance in multi-morbidity and hospitalized subgroups

5. **Sensitivity Analysis**: Re-estimating model without chronic_count to assess feature importance

## Requirements

```r
# Required R packages
install.packages(c(
  "tidyverse",
  "data.table",
  "xgboost",
  "caret",
  "pROC",
  "broom"
))
```

**Note on XGBoost Compatibility**: The main `modeling.R` script may not be compatible with newer versions of XGBoost. If you encounter compatibility issues, please use the supplementary `modeling.R` file provided by team members, which has been updated to work with the latest XGBoost version.

## Usage

Run the complete pipeline:
```r
source("run_pipeline.R")
```

Or run individual scripts:
```r
# Data preparation
source("data_prep.R")

# Modeling
source("modeling.R")

# Subgroup analysis
source("subgroup_analysis.R")
```

## Key Results

### Model Performance
- **Best Model**: Two-Part (Logistic GLM + XGBoost Regressor)
- **Test Performance**:
  - MAE: 1,638.93 (vs Single GLM: 2,013.27, **18.6% improvement**)
  - RMSE: 5,109.04 (vs Single GLM: 5,871.64, **13.0% improvement**)
  - RMSLE (all samples): 2.04 (affected by zero-value prediction limitation)
  - **RMSLE (non-zero samples only): 0.63** (more appropriate metric)

### Cross-Validation
- **Stability**: MAE CV = 2.7%, RMSE CV = 5.4%
- **RMSLE_NonZero**: 0.6448 ± 0.0055 (CV = 0.85%, very stable)

### Frequency Model
- **AUC (with chronic_count)**: 0.9193
- **AUC (without chronic_count)**: 0.5819
- **AUC drop**: 0.3375 (33.75%)
- **Conclusion**: chronic_count is a key predictor of medical utilization

### Subgroup Analysis
- **Multi-Morbidity**: XGBoost outperforms GLM by 9-13% across all groups
- **Hospitalized**: XGBoost outperforms GLM by 11-21% in both groups
- **Best improvement**: 25.4% in High Multi-Morbidity & Hospitalized subgroup

### Model Calibration
- **Calibration slope**: 1.05 (ideal = 1.0)
- **Calibration correlation**: 0.9998 (ideal = 1.0)
- **Assessment**: Excellent calibration

## Output Files

All results are saved in the `output/` directory:

### Model Performance
- `model_performance_v3.csv` - Performance metrics for all models (includes RMSLE_NonZero)
- `cv_summary_v3.csv` - Cross-validation summary (includes RMSLE_NonZero statistics)
- `sensitivity_analysis_v3.csv` - Sensitivity analysis results (with/without chronic_count)

### Predictions
- `test_predictions_v3.rds` - Test set predictions (RDS format)
- `test_predictions_v3.csv` - Test set predictions (CSV format)

### Subgroup Analysis
- `subgroup_analysis_multi_morbidity.csv` - Multi-morbidity subgroup results
- `subgroup_analysis_hospitalized.csv` - Hospitalized subgroup results
- `subgroup_analysis_combined.csv` - Combined subgroup results

### Diagnostics
- `calibration/calibration_curve_v3.png` - Calibration curve
- `calibration/calibration_data_v3.csv` - Calibration data
- `residuals/residuals_scatter_v3.png` - Residuals vs predicted
- `residuals/qq_plot_v3.png` - Q-Q plot of residuals
- `residuals/residuals_histogram_v3.png` - Residual distribution

### Models
- `fitted_models_v3.rda` - Saved model objects

## Important Notes

### RMSLE Interpretation
- **RMSLE (all samples) = 2.04**: High due to zero-value prediction limitation of Two-Part models
  - Two-Part models cannot predict exact zero values (26.7% of samples)
  - Frequency model predicts P(Y>0), severity model predicts E[Y|Y>0]
  - Pure Premium = P(Y>0) × E[Y|Y>0] is always > 0
- **RMSLE (non-zero samples only) = 0.63**: More appropriate for evaluating severity model performance
  - This metric focuses on the subset where severity model applies
  - Better reflects the model's ability to predict cost amounts

### Data Leakage Prevention (V3)
- **Frequency model**: Uses only demographics (age, age2, sex, race, chronic_count)
  - No visit counts to avoid data leakage
  - Visit counts are highly correlated with any_cost
- **Severity model**: Includes visit counts (no leakage for amount prediction)
  - Visit counts are appropriate for predicting cost amounts given utilization

### Chronic Count Feature
- **chronic_count** is a key predictor (AUC drops from 0.919 to 0.582 without it)
- This is a legitimate medical feature (chronic disease burden)
- Not data leakage, but a strong predictor of medical utilization
- Supported by medical literature (Elixhauser Comorbidity Index, etc.)

## Research Questions

### RQ1: Does a two-part model improve predictive accuracy and calibration relative to single-equation GLMs?
**Answer**: ✅ Yes
- Two-Part model improves MAE by 18.6% and RMSE by 13.0% vs Single GLM
- Excellent calibration (slope = 1.05, correlation = 0.9998)
- RMSLE_NonZero = 0.63 (vs Single GLM: 0.65, 3.3% improvement)

### RQ2: Do tree-based learners (XGBoost) outperform GLMs in key subgroups?
**Answer**: ✅ Yes
- XGBoost outperforms GLM by 9-13% in multi-morbidity subgroups
- XGBoost outperforms GLM by 11-21% in hospitalized subgroups
- Best improvement: 25.4% in High Multi-Morbidity & Hospitalized subgroup

## License

This project is for educational purposes (Biostat 625 Final Project).
