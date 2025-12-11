# ============================================================================
# Quick Runner for EDA
# ============================================================================
# This script runs the EDA analysis
# Make sure data_prep.R has been run first
# ============================================================================

# Check if data exists
if (!file.exists("output/analysis_data.rds")) {
  cat("Running data preparation first...\n")
  source("data_prep.R")
}

# Run EDA
cat("Running exploratory data analysis...\n")
source("eda.R")

cat("\nEDA complete! Check the output/ directory for results.\n")

