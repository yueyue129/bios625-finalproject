# Run V3 pipeline
# Biostat 625

cat("=== Running V3 Pipeline ===\n\n")

# Check if running from correct directory
if (!file.exists("data_prep.R") || !file.exists("modeling.R")) {
  stop("Error: Please run this script from the project root directory.")
}

# Check and install required packages
required_packages <- c(
  "tidyverse", "data.table", "lubridate",
  "xgboost", "caret", "pROC", "broom"
)

cat("Checking required packages...\n")
missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, dependencies = TRUE)
  # Try loading again
  for (pkg in missing_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop("Error: Failed to install package: ", pkg)
    }
  }
}
cat("All required packages are available.\n\n")

# Run pipeline with error handling
tryCatch({
cat("[1/3] Data preparation...\n")
  source("data_prep.R")

cat("\n[2/3] Modeling...\n")
  source("modeling.R")

cat("\n[3/3] Subgroup analysis...\n")
source("subgroup_analysis.R")

cat("\n=== Pipeline Complete ===\n")
}, error = function(e) {
  cat("\n❌ Pipeline failed with error:\n")
  cat(conditionMessage(e), "\n")
  stop("Pipeline execution stopped.")
})

