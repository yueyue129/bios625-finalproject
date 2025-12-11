# Subgroup analysis: XGBoost vs GLM
# Biostat 625

library(tidyverse)

output_dir <- "output/"
set.seed(625)

cat("Subgroup analysis...\n")

if (file.exists(file.path(output_dir, "test_predictions_v3.rds"))) {
  test_pred <- readRDS(file.path(output_dir, "test_predictions_v3.rds"))
} else {
  stop("Please run modeling.R first.")
}

if (file.exists(file.path(output_dir, "analysis_data_noleak.rds"))) {
  analysis_data <- readRDS(file.path(output_dir, "analysis_data_noleak.rds"))
} else {
  stop("Please run data_prep.R first.")
}

# Recreate train-test split (same seed)
library(caret)
set.seed(625)
train_idx <- createDataPartition(analysis_data$any_cost, p = 0.7, list = FALSE)
test_data <- analysis_data[-train_idx, ]

# Helper Functions

compute_metrics <- function(y_true, y_pred) {
  mae <- mean(abs(y_true - y_pred), na.rm = TRUE)
  rmse <- sqrt(mean((y_true - y_pred)^2, na.rm = TRUE))
  rmsle <- sqrt(mean((log1p(y_true) - log1p(y_pred))^2, na.rm = TRUE))
  
  return(data.frame(
    MAE = mae,
    RMSE = rmse,
    RMSLE = rmsle
  ))
}

# V3: frequency uses demographics only, severity uses demographics + visits
build_features_freq <- function(data, for_glm = FALSE) {
  if (for_glm) {
    X <- model.matrix(
      ~ age + age2 + sex + race + chronic_count,
      data = data
    )[, -1]
  } else {
    X <- data %>%
      select(age, age2, chronic_count) %>%
      mutate(
        sex_male = as.numeric(data$sex == "Male"),
        race_black = as.numeric(data$race == "Black"),
        race_other = as.numeric(data$race == "Other"),
        race_asian = as.numeric(data$race == "Asian"),
        race_hispanic = as.numeric(data$race == "Hispanic"),
        race_native = as.numeric(data$race == "Native")
      ) %>%
      as.matrix()
  }
  return(X)
}

build_features_sev <- function(data, for_glm = FALSE) {
  if (for_glm) {
    X <- model.matrix(
      ~ age + age2 + sex + race + chronic_count +
        ip_visits + op_visits + car_visits +
        log_ip_visits + log_op_visits + log_car_visits +
        total_visits + log_total_visits +
        has_ip + has_op + has_car,
      data = data
    )[, -1]
  } else {
    X <- data %>%
      select(age, age2, chronic_count,
             ip_visits, op_visits, car_visits,
             total_visits,
             log_ip_visits, log_op_visits, log_car_visits, log_total_visits,
             has_ip, has_op, has_car) %>%
      mutate(
        sex_male = as.numeric(data$sex == "Male"),
        race_black = as.numeric(data$race == "Black"),
        race_other = as.numeric(data$race == "Other"),
        race_asian = as.numeric(data$race == "Asian"),
        race_hispanic = as.numeric(data$race == "Hispanic"),
        race_native = as.numeric(data$race == "Native")
      ) %>%
      as.matrix()
  }
  return(X)
}

# Recreate Train-Test Split and Predictions


# Recreate the exact same split as modeling_v3.R
set.seed(625)
train_idx <- createDataPartition(analysis_data$any_cost, p = 0.7, list = FALSE)
train_data <- analysis_data[train_idx, ]
test_data <- analysis_data[-train_idx, ]

# Build features (V3: frequency uses demographics only, severity uses demographics + visits)
X_test_freq_ml <- build_features_freq(test_data, for_glm = FALSE)
X_test_sev_ml <- build_features_sev(test_data, for_glm = FALSE)
y_test <- test_data$total_payment

# Load V3 models
load(file.path(output_dir, "fitted_models_v3.rda"))

# Helper function for winsorization
winsorize <- function(x, lower_quantile = 0.01, upper_quantile = 0.99) {
  lower_bound <- quantile(x, lower_quantile, na.rm = TRUE)
  upper_bound <- quantile(x, upper_quantile, na.rm = TRUE)
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

# Helper function to combine two-part predictions
combine_two_part <- function(p_freq, mu_severity_all) {
  return(p_freq * pmax(mu_severity_all, 0.01))
}

# Recreate the exact same predictions as in modeling_v3.R (V3: frequency uses demographics only)
p_freq_glm_test <- predict(freq_glm, newdata = test_data, type = "response")

# Predict severity for all test samples (severity uses demographics + visits)
log_pred_lognorm_test <- predict(severity_lognorm, newdata = test_data)
sigma2 <- summary(severity_lognorm)$dispersion
bias_correction <- exp(sigma2 / 2)
mu_lognorm_test <- pmax(exp(log_pred_lognorm_test) * bias_correction, 0.01)

# Get max_reasonable from training data
pos_idx_train <- which(train_data$any_cost == 1)
y_train_pos <- train_data$total_payment[pos_idx_train]
y_train_pos_trimmed <- winsorize(y_train_pos, 0.01, 0.99)
max_reasonable <- quantile(y_train_pos_trimmed, 0.999, na.rm = TRUE)
mu_lognorm_test <- pmin(mu_lognorm_test, max_reasonable)

# Combine GLM predictions
pred_glm_glm <- combine_two_part(p_freq_glm_test, mu_lognorm_test)

# Get XGBoost predictions (V3: frequency uses demographics only, severity uses demographics + visits)
p_freq_xgb_test <- predict(freq_xgb, X_test_freq_ml)
mu_xgb_test <- pmax(predict(severity_xgb, X_test_sev_ml), 0.01)
mu_xgb_test <- pmin(mu_xgb_test, max_reasonable)

pred_xgb_xgb <- combine_two_part(p_freq_xgb_test, mu_xgb_test)

# Create analysis dataframe
test_analysis <- test_data %>%
  mutate(
    pred_glm_glm = as.numeric(pred_glm_glm),
    pred_xgb_xgb = as.numeric(pred_xgb_xgb),
    actual = y_test
  )


# Subgroup 1: Multi-Morbidity (chronic_count)


# Define multi-morbidity groups based on quantiles
chronic_quantiles <- quantile(test_analysis$chronic_count, probs = c(0, 0.33, 0.67, 1), na.rm = TRUE)
test_analysis <- test_analysis %>%
  mutate(
    multi_morbidity_group = case_when(
      chronic_count <= chronic_quantiles[2] ~ "Low Multi-Morbidity",
      chronic_count <= chronic_quantiles[3] ~ "Medium Multi-Morbidity",
      chronic_count > chronic_quantiles[3] ~ "High Multi-Morbidity"
    )
  )
cat("Chronic count quantiles:", round(chronic_quantiles, 1), "\n")
cat("Multi-morbidity group distribution:\n")
print(table(test_analysis$multi_morbidity_group))
cat("\n")

# Calculate performance by subgroup (optimized: compute all metrics at once)
subgroup_mm <- test_analysis %>%
  group_by(multi_morbidity_group) %>%
  summarise(
    n = n(),
    glm_mae = mean(abs(actual - pred_glm_glm), na.rm = TRUE),
    glm_rmse = sqrt(mean((actual - pred_glm_glm)^2, na.rm = TRUE)),
    glm_rmsle = sqrt(mean((log1p(actual) - log1p(pred_glm_glm))^2, na.rm = TRUE)),
    xgb_mae = mean(abs(actual - pred_xgb_xgb), na.rm = TRUE),
    xgb_rmse = sqrt(mean((actual - pred_xgb_xgb)^2, na.rm = TRUE)),
    xgb_rmsle = sqrt(mean((log1p(actual) - log1p(pred_xgb_xgb))^2, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    rmse_improvement = round((glm_rmse - xgb_rmse) / glm_rmse * 100, 2),
    mae_improvement = round((glm_mae - xgb_mae) / glm_mae * 100, 2),
    rmsle_improvement = round((glm_rmsle - xgb_rmsle) / glm_rmsle * 100, 2),
    glm_mae = round(glm_mae, 2),
    glm_rmse = round(glm_rmse, 2),
    glm_rmsle = round(glm_rmsle, 4),
    xgb_mae = round(xgb_mae, 2),
    xgb_rmse = round(xgb_rmse, 2),
    xgb_rmsle = round(xgb_rmsle, 4)
  )

cat("Multi-Morbidity Subgroup Performance:\n")
print(subgroup_mm)

write.csv(subgroup_mm, 
          file = file.path(output_dir, "subgroup_analysis_multi_morbidity.csv"), 
          row.names = FALSE)

cat("\nMulti-morbidity analysis saved.\n\n")

# Subgroup 2: Hospitalized (has_ip)

cat("=== Subgroup 2: Hospitalized Analysis ===\n\n")

# Define hospitalized groups
test_analysis <- test_analysis %>%
  mutate(
    hospitalized_group = case_when(
      has_ip == 0 ~ "Not Hospitalized",
      has_ip == 1 ~ "Hospitalized"
    )
  )

# Calculate performance by subgroup (optimized)
subgroup_hosp <- test_analysis %>%
  group_by(hospitalized_group) %>%
  summarise(
    n = n(),
    glm_mae = mean(abs(actual - pred_glm_glm), na.rm = TRUE),
    glm_rmse = sqrt(mean((actual - pred_glm_glm)^2, na.rm = TRUE)),
    glm_rmsle = sqrt(mean((log1p(actual) - log1p(pred_glm_glm))^2, na.rm = TRUE)),
    xgb_mae = mean(abs(actual - pred_xgb_xgb), na.rm = TRUE),
    xgb_rmse = sqrt(mean((actual - pred_xgb_xgb)^2, na.rm = TRUE)),
    xgb_rmsle = sqrt(mean((log1p(actual) - log1p(pred_xgb_xgb))^2, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    rmse_improvement = round((glm_rmse - xgb_rmse) / glm_rmse * 100, 2),
    mae_improvement = round((glm_mae - xgb_mae) / glm_mae * 100, 2),
    rmsle_improvement = round((glm_rmsle - xgb_rmsle) / glm_rmsle * 100, 2),
    glm_mae = round(glm_mae, 2),
    glm_rmse = round(glm_rmse, 2),
    glm_rmsle = round(glm_rmsle, 4),
    xgb_mae = round(xgb_mae, 2),
    xgb_rmse = round(xgb_rmse, 2),
    xgb_rmsle = round(xgb_rmsle, 4)
  )

cat("Hospitalized Subgroup Performance:\n")
print(subgroup_hosp)

write.csv(subgroup_hosp, 
          file = file.path(output_dir, "subgroup_analysis_hospitalized.csv"), 
          row.names = FALSE)

cat("\nHospitalized analysis saved.\n\n")

# Combined Subgroup Analysis

cat("=== Combined Subgroup Analysis ===\n\n")

# Combine multi-morbidity and hospitalized
test_analysis <- test_analysis %>%
  mutate(
    combined_group = paste0(
      multi_morbidity_group, " & ",
      ifelse(has_ip == 1, "Hospitalized", "Not Hospitalized")
    )
  )

# Calculate performance for combined subgroups (optimized)
subgroup_combined <- test_analysis %>%
  group_by(combined_group) %>%
  summarise(
    n = n(),
    glm_mae = mean(abs(actual - pred_glm_glm), na.rm = TRUE),
    glm_rmse = sqrt(mean((actual - pred_glm_glm)^2, na.rm = TRUE)),
    xgb_mae = mean(abs(actual - pred_xgb_xgb), na.rm = TRUE),
    xgb_rmse = sqrt(mean((actual - pred_xgb_xgb)^2, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  filter(n >= 100) %>%  # Only include subgroups with at least 100 samples
  mutate(
    rmse_improvement = round((glm_rmse - xgb_rmse) / glm_rmse * 100, 2),
    mae_improvement = round((glm_mae - xgb_mae) / glm_mae * 100, 2),
    glm_mae = round(glm_mae, 2),
    glm_rmse = round(glm_rmse, 2),
    xgb_mae = round(xgb_mae, 2),
    xgb_rmse = round(xgb_rmse, 2)
  ) %>%
  arrange(desc(rmse_improvement))

cat("Combined Subgroup Performance (top 10 by RMSE improvement):\n")
print(head(subgroup_combined, 10))

write.csv(subgroup_combined, 
          file = file.path(output_dir, "subgroup_analysis_combined.csv"), 
          row.names = FALSE)

cat("\nCombined subgroup analysis saved.\n\n")

# Summary

cat("=== Subgroup Analysis Summary ===\n\n")

cat("Multi-Morbidity Subgroups:\n")
cat("  XGBoost outperforms GLM in:\n")
mm_better <- subgroup_mm %>% filter(rmse_improvement > 0)
for (i in 1:nrow(mm_better)) {
  cat("    - ", mm_better$multi_morbidity_group[i], 
      ": RMSE improvement = ", mm_better$rmse_improvement[i], "%\n", sep = "")
}

cat("\nHospitalized Subgroups:\n")
cat("  XGBoost vs GLM:\n")
for (i in 1:nrow(subgroup_hosp)) {
  cat("    - ", subgroup_hosp$hospitalized_group[i], 
      ": RMSE improvement = ", subgroup_hosp$rmse_improvement[i], "%\n", sep = "")
}

cat("\n=== Subgroup Analysis Complete ===\n")
