# Two-part model for Medicare cost prediction
# Biostat 625
# V3: frequency model uses demographics only

library(tidyverse)
library(xgboost)
library(broom)
library(caret)
library(pROC)

output_dir <- "output/"
if (!dir.exists(file.path(output_dir, "residuals"))) dir.create(file.path(output_dir, "residuals"))
if (!dir.exists(file.path(output_dir, "calibration"))) dir.create(file.path(output_dir, "calibration"))

set.seed(625)

# Frequency model: demographics only (no visit counts)
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

# Severity model: includes visit counts
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
      select(
        age, age2, chronic_count,
        ip_visits, op_visits, car_visits,
        total_visits,
        log_ip_visits, log_op_visits, log_car_visits, log_total_visits,
        has_ip, has_op, has_car
      ) %>%
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

compute_metrics <- function(y_true, y_pred, name) {
  mae <- mean(abs(y_true - y_pred))
  rmse <- sqrt(mean((y_true - y_pred)^2))
  rmsle <- sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
  
  # RMSLE for non-zero samples only
  nonzero_idx <- y_true > 0
  rmsle_nonzero <- if (sum(nonzero_idx) > 0) {
    sqrt(mean((log1p(y_true[nonzero_idx]) - log1p(y_pred[nonzero_idx]))^2))
  } else {
    NA
  }
  
  data.frame(
    Model = name,
    MAE = mae,
    RMSE = rmse,
    RMSLE = rmsle,
    RMSLE_NonZero = rmsle_nonzero
  )
}

find_optimal_threshold <- function(y_true, y_pred_proba, method = "youden") {
  roc_obj <- roc(y_true, y_pred_proba, quiet = TRUE)
  
  if (method == "youden") {
    coords_obj <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")
    optimal_threshold <- coords_obj$threshold
  } else if (method == "f1") {
    thresholds <- seq(0.1, 0.9, by = 0.01)
    f1_scores <- sapply(thresholds, function(t) {
      y_pred_binary <- as.numeric(y_pred_proba >= t)
      if (sum(y_pred_binary) == 0 || sum(y_pred_binary) == length(y_pred_binary)) {
        return(0)
      }
      cm <- confusionMatrix(
        factor(y_pred_binary, levels = c(0, 1)),
        factor(y_true, levels = c(0, 1))
      )
      f1 <- cm$byClass["F1"]
      ifelse(is.na(f1), 0, f1)
    })
    optimal_threshold <- thresholds[which.max(f1_scores)]
  } else {
    coords_obj <- coords(roc_obj, "best", ret = "threshold", best.method = "closest.topleft")
    optimal_threshold <- coords_obj$threshold
  }
  
  return(optimal_threshold)
}

winsorize <- function(x, lower_quantile = 0.01, upper_quantile = 0.99) {
  lower_bound <- quantile(x, lower_quantile, na.rm = TRUE)
  upper_bound <- quantile(x, upper_quantile, na.rm = TRUE)
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

combine_two_part <- function(p_freq, mu_severity_all) {
  return(p_freq * pmax(mu_severity_all, 0.01))
}

# Load data
if (file.exists(file.path(output_dir, "analysis_data_noleak.rds"))) {
  analysis_data <- readRDS(file.path(output_dir, "analysis_data_noleak.rds"))
} else {
  stop("Please run data_prep.R first.")
}

cat("Data loaded. N =", nrow(analysis_data), "\n\n")

set.seed(625)
train_idx <- createDataPartition(analysis_data$any_cost, p = 0.7, list = FALSE)
train_data <- analysis_data[train_idx, ]
test_data <- analysis_data[-train_idx, ]

cat("Train:", nrow(train_data), "Test:", nrow(test_data), "\n")

X_train_freq_glm <- build_features_freq(train_data, for_glm = TRUE)
X_test_freq_glm <- build_features_freq(test_data, for_glm = TRUE)
X_train_freq_ml <- build_features_freq(train_data, for_glm = FALSE)
X_test_freq_ml <- build_features_freq(test_data, for_glm = FALSE)
X_train_sev_glm <- build_features_sev(train_data, for_glm = TRUE)
X_test_sev_glm <- build_features_sev(test_data, for_glm = TRUE)
X_train_sev_ml <- build_features_sev(train_data, for_glm = FALSE)
X_test_sev_ml <- build_features_sev(test_data, for_glm = FALSE)

y_train <- train_data$total_payment
y_test <- test_data$total_payment
any_cost_train <- train_data$any_cost
any_cost_test <- test_data$any_cost

pos_idx_train <- which(any_cost_train == 1)
pos_idx_test <- which(any_cost_test == 1)

y_train_pos <- y_train[pos_idx_train]
y_test_pos <- y_test[pos_idx_test]
X_train_sev_glm_pos <- X_train_sev_glm[pos_idx_train, ]
X_test_sev_glm_pos <- X_test_sev_glm[pos_idx_test, ]
X_train_sev_ml_pos <- X_train_sev_ml[pos_idx_train, ]
X_test_sev_ml_pos <- X_test_sev_ml[pos_idx_test, ]

y_train_pos_trimmed <- winsorize(y_train_pos, 0.01, 0.99)
y_test_pos_trimmed <- winsorize(y_test_pos, 0.01, 0.99)

cat("=== Two-Part Model: Part 1 (Frequency) ===\n")
cat("Features: age, age2, sex, race, chronic_count\n\n")

cat("Fitting frequency models...\n")
set.seed(625)
freq_glm <- glm(
  any_cost ~ age + age2 + sex + race + chronic_count,
  data = train_data,
  family = binomial(link = "logit")
)

p_freq_glm_train <- predict(freq_glm, newdata = train_data, type = "response")
p_freq_glm_test <- predict(freq_glm, newdata = test_data, type = "response")

threshold_glm_youden <- find_optimal_threshold(any_cost_train, p_freq_glm_train, "youden")
threshold_glm_f1 <- find_optimal_threshold(any_cost_train, p_freq_glm_train, "f1")

threshold_glm <- threshold_glm_youden
p_freq_glm_train_binary <- as.numeric(p_freq_glm_train >= threshold_glm)
p_freq_glm_test_binary <- as.numeric(p_freq_glm_test >= threshold_glm)

cat("  Logistic GLM complete.\n")
cat("  Training AUC:", round(as.numeric(pROC::auc(any_cost_train, p_freq_glm_train)), 4), "\n")
cat("  Test AUC:", round(as.numeric(pROC::auc(any_cost_test, p_freq_glm_test)), 4), "\n")
cat("  Training Accuracy (optimal threshold):",
    round(mean(p_freq_glm_train_binary == any_cost_train), 4), "\n"
)
cat("  Test Accuracy (optimal threshold):",
    round(mean(p_freq_glm_test_binary == any_cost_test), 4), "\n\n"
)

cat("Fitting XGBoost classifier...\n")

param_grid <- expand.grid(
  max_depth = c(4, 6, 8),
  learning_rate = c(0.05, 0.1),
  nrounds = c(50, 100)
)

best_score <- Inf
best_params <- NULL
best_model <- NULL

for (i in 1:nrow(param_grid)) {
  set.seed(625)
  params <- param_grid[i, ]
  
  dtrain_freq <- xgb.DMatrix(data = X_train_freq_ml, label = any_cost_train)
  
  params_list <- list(
    max_depth = params$max_depth,
    learning_rate = params$learning_rate,
    objective = "binary:logistic",
    eval_metric = "logloss",
    verbosity = 0
  )
  
  model <- xgb.train(
    params = params_list,
    data = dtrain_freq,
    nrounds = params$nrounds
  )
  
  pred <- predict(model, X_train_freq_ml)
  score <- -as.numeric(pROC::auc(any_cost_train, pred))
  
  if (score < best_score) {
    best_score <- score
    best_params <- params
    best_model <- model
  }
}

freq_xgb <- best_model
p_freq_xgb_train <- predict(freq_xgb, X_train_freq_ml)
p_freq_xgb_test <- predict(freq_xgb, X_test_freq_ml)

threshold_xgb <- find_optimal_threshold(any_cost_train, p_freq_xgb_train, "youden")
cat("  XGBoost AUC:", round(as.numeric(pROC::auc(any_cost_test, p_freq_xgb_test)), 3), "\n\n")

cat("\n=== Two-Part Model: Part 2 (Severity) ===\n\n")

cat("[1/3] Fitting Lognormal GLM...\n")
set.seed(625)
severity_lognorm <- glm(
  log(y_train_pos_trimmed) ~ age + age2 + sex + race + chronic_count +
    ip_visits + op_visits + car_visits +
    log_ip_visits + log_op_visits + log_car_visits +
    total_visits + log_total_visits +
    has_ip + has_op + has_car,
  data = train_data %>% filter(any_cost == 1),
  family = gaussian()
)

log_pred_lognorm_train <- predict(
  severity_lognorm,
  newdata = train_data %>% filter(any_cost == 1)
)
log_pred_lognorm_test <- predict(
  severity_lognorm,
  newdata = test_data
)

sigma2 <- summary(severity_lognorm)$dispersion
bias_correction <- exp(sigma2 / 2)

mu_lognorm_train <- pmax(exp(log_pred_lognorm_train) * bias_correction, 0.01)
mu_lognorm_test <- pmax(exp(log_pred_lognorm_test) * bias_correction, 0.01)

max_reasonable <- quantile(y_train_pos_trimmed, 0.999, na.rm = TRUE)
mu_lognorm_train <- pmin(mu_lognorm_train, max_reasonable)
mu_lognorm_test <- pmin(mu_lognorm_test, max_reasonable)

cat("  Lognormal GLM complete.\n")
cat("  Training RMSE:", round(sqrt(mean((y_train_pos - mu_lognorm_train)^2)), 2), "\n")
cat("  Test RMSE (on positive-cost samples):", round(sqrt(mean((y_test_pos - mu_lognorm_test[pos_idx_test])^2)), 2), "\n\n")

set.seed(625)
severity_gamma <- glm(
  y_train_pos_trimmed ~ age + age2 + sex + race + chronic_count +
    ip_visits + op_visits + car_visits +
    log_ip_visits + log_op_visits + log_car_visits +
    total_visits + log_total_visits +
    has_ip + has_op + has_car,
  data = train_data %>% filter(any_cost == 1),
  family = Gamma(link = "log")
)

mu_gamma_train <- predict(
  severity_gamma,
  newdata = train_data %>% filter(any_cost == 1),
  type = "response"
)
mu_gamma_test <- predict(
  severity_gamma,
  newdata = test_data,
  type = "response"
)

mu_gamma_train <- pmax(mu_gamma_train, 0.01)
mu_gamma_test <- pmax(mu_gamma_test, 0.01)
max_reasonable <- quantile(y_train_pos_trimmed, 0.999, na.rm = TRUE)
mu_gamma_train <- pmin(mu_gamma_train, max_reasonable)
mu_gamma_test <- pmin(mu_gamma_test, max_reasonable)

cat("  Gamma GLM complete.\n")
cat("  Training RMSE:", round(sqrt(mean((y_train_pos - mu_gamma_train)^2)), 2), "\n")
cat("  Test RMSE (on positive-cost samples):", round(sqrt(mean((y_test_pos - mu_gamma_test[pos_idx_test])^2)), 2), "\n\n")

param_grid_sev <- expand.grid(
  max_depth = c(4, 6),
  learning_rate = c(0.05, 0.1),
  nrounds = c(50, 100)
)

best_score_sev <- Inf
best_params_sev <- NULL
best_model_sev <- NULL

for (i in 1:nrow(param_grid_sev)) {
  set.seed(625)
  params <- param_grid_sev[i, ]
  
  dtrain_pos <- xgb.DMatrix(data = X_train_sev_ml_pos, label = y_train_pos_trimmed)
  
  params_list <- list(
    max_depth = params$max_depth,
    learning_rate = params$learning_rate,
    objective = "reg:squarederror",
    eval_metric = "rmse",
    verbosity = 0
  )
  
  model <- xgb.train(
    params = params_list,
    data = dtrain_pos,
    nrounds = params$nrounds
  )
  
  pred <- predict(model, X_train_sev_ml_pos)
  score <- sqrt(mean((y_train_pos_trimmed - pred)^2))
  
  if (score < best_score_sev) {
    best_score_sev <- score
    best_params_sev <- params
    best_model_sev <- model
  }
}

severity_xgb <- best_model_sev
mu_xgb_train <- pmax(predict(severity_xgb, X_train_sev_ml_pos), 0.01)
mu_xgb_test <- pmax(predict(severity_xgb, X_test_sev_ml), 0.01)

max_reasonable <- quantile(y_train_pos_trimmed, 0.999, na.rm = TRUE)
mu_xgb_train <- pmin(mu_xgb_train, max_reasonable)
mu_xgb_test <- pmin(mu_xgb_test, max_reasonable)

cat("  XGBoost regressor complete.\n")
cat("  Training RMSE:", round(sqrt(mean((y_train_pos - mu_xgb_train)^2)), 2), "\n")
cat("  Test RMSE (on positive-cost samples):", round(sqrt(mean((y_test_pos - mu_xgb_test[pos_idx_test])^2)), 2), "\n\n")

cat("Combining two-part predictions...\n")

pure_prem_glm_lognorm_test <- combine_two_part(p_freq_glm_test, mu_lognorm_test)
pure_prem_glm_gamma_test <- combine_two_part(p_freq_glm_test, mu_gamma_test)
pure_prem_xgb_xgb_test <- combine_two_part(p_freq_xgb_test, mu_xgb_test)
pure_prem_glm_xgb_test <- combine_two_part(p_freq_glm_test, mu_xgb_test)
pure_prem_xgb_lognorm_test <- combine_two_part(p_freq_xgb_test, mu_lognorm_test)
pure_prem_xgb_gamma_test <- combine_two_part(p_freq_xgb_test, mu_gamma_test)

cat("Two-part predictions combined.\n")

cat("\n=== Single GLM Baseline ===\n")
set.seed(625)
single_glm <- glm(
  log1p(total_payment) ~ age + age2 + sex + race + chronic_count +
    ip_visits + op_visits + car_visits +
    log_ip_visits + log_op_visits + log_car_visits +
    total_visits + log_total_visits +
    has_ip + has_op + has_car,
  data = train_data,
  family = gaussian()
)

log1p_pred_single_test <- predict(single_glm, newdata = test_data)
sigma2_single <- summary(single_glm)$dispersion
mu_single_glm_test <- pmax(exp(log1p_pred_single_test + sigma2_single / 2) - 1, 0)
max_reasonable <- quantile(y_train, 0.999, na.rm = TRUE)
mu_single_glm_test <- pmin(mu_single_glm_test, max_reasonable)

cat("Single GLM complete.\n\n")

cat("\n=== Performance Evaluation ===\n")

predictions <- list(
  "Two-Part (Logistic GLM + Lognormal GLM)" = pure_prem_glm_lognorm_test,
  "Two-Part (Logistic GLM + Gamma GLM)" = pure_prem_glm_gamma_test,
  "Two-Part (XGBoost Classifier + XGBoost Regressor)" = pure_prem_xgb_xgb_test,
  "Two-Part (Logistic GLM + XGBoost Regressor)" = pure_prem_glm_xgb_test,
  "Two-Part (XGBoost Classifier + Lognormal GLM)" = pure_prem_xgb_lognorm_test,
  "Two-Part (XGBoost Classifier + Gamma GLM)" = pure_prem_xgb_gamma_test,
  "Single GLM (Lognormal)" = mu_single_glm_test
)

results <- map_dfr(
  names(predictions),
  ~ compute_metrics(y_test, predictions[[.x]], .x)
)

results_formatted <- results %>%
  mutate(
    MAE = round(MAE, 2),
    RMSE = round(RMSE, 2),
    RMSLE = round(RMSLE, 4),
    RMSLE_NonZero = round(RMSLE_NonZero, 4)
  ) %>%
  arrange(RMSE)

print(results_formatted)
write.csv(results_formatted,
          file = file.path(output_dir, "model_performance_v3.csv"),
          row.names = FALSE
)

cat("\nBest model (by RMSE):", results_formatted$Model[1], "\n")
cat("  MAE:", results_formatted$MAE[1], "\n")
cat("  RMSE:", results_formatted$RMSE[1], "\n")
cat("  RMSLE (all samples):", results_formatted$RMSLE[1], "\n")
cat("  RMSLE (non-zero samples only):", results_formatted$RMSLE_NonZero[1], "\n\n")

cat("\n=== Residual Diagnostics ===\n")

best_pred <- pure_prem_xgb_xgb_test
residuals <- y_test - best_pred

png(file.path(output_dir, "residuals", "residuals_scatter_v3.png"),
    width = 10, height = 6, units = "in", res = 300
)
plot(best_pred, residuals,
     xlab = "Predicted Values", ylab = "Residuals",
     main = "Residuals vs Predicted Values (V3)"
)
abline(h = 0, col = "red", lty = 2)
dev.off()

png(file.path(output_dir, "residuals", "qq_plot_v3.png"),
    width = 10, height = 6, units = "in", res = 300
)
qqnorm(residuals, main = "Q-Q Plot of Residuals (V3)")
qqline(residuals, col = "red")
dev.off()

png(file.path(output_dir, "residuals", "residuals_histogram_v3.png"),
    width = 10, height = 6, units = "in", res = 300
)
hist(residuals,
     breaks = 50,
     xlab = "Residuals", main = "Distribution of Residuals (V3)"
)
dev.off()

cat("Residual plots saved to output/residuals/\n")

cat("\n=== 5-Fold Cross-Validation ===\n")

set.seed(625)
folds <- createFolds(analysis_data$any_cost, k = 5, list = TRUE)

cv_results <- map_dfr(1:5, function(fold) {
  cat("  Fold", fold, "...\n")
  
  test_idx_cv <- folds[[fold]]
  train_idx_cv <- setdiff(1:nrow(analysis_data), test_idx_cv)
  
  train_cv <- analysis_data[train_idx_cv, ]
  test_cv <- analysis_data[test_idx_cv, ]
  
  X_train_cv_freq_ml <- build_features_freq(train_cv, for_glm = FALSE)
  X_test_cv_freq_ml <- build_features_freq(test_cv, for_glm = FALSE)
  X_train_cv_sev_ml <- build_features_sev(train_cv, for_glm = FALSE)
  X_test_cv_sev_ml <- build_features_sev(test_cv, for_glm = FALSE)
  
  any_cost_train_cv <- train_cv$any_cost
  any_cost_test_cv <- test_cv$any_cost
  y_train_cv <- train_cv$total_payment
  y_test_cv <- test_cv$total_payment
  
  pos_idx_train_cv <- which(any_cost_train_cv == 1)
  pos_idx_test_cv <- which(any_cost_test_cv == 1)
  
  y_train_pos_cv <- y_train_cv[pos_idx_train_cv]
  y_test_pos_cv <- y_test_cv[pos_idx_test_cv]
  X_train_cv_sev_ml_pos <- X_train_cv_sev_ml[pos_idx_train_cv, ]
  X_test_cv_sev_ml_pos <- X_test_cv_sev_ml[pos_idx_test_cv, ]
  
  set.seed(625)
  dtrain_freq_cv <- xgb.DMatrix(data = X_train_cv_freq_ml, label = any_cost_train_cv)
  
  params_list_freq_cv <- list(
    max_depth = best_params$max_depth,
    learning_rate = best_params$learning_rate,
    objective = "binary:logistic",
    eval_metric = "logloss",
    verbosity = 0
  )
  
  freq_xgb_cv <- xgb.train(
    params = params_list_freq_cv,
    data = dtrain_freq_cv,
    nrounds = best_params$nrounds
  )
  
  p_freq_cv <- predict(freq_xgb_cv, X_test_cv_freq_ml)
  
  y_train_pos_cv_trimmed <- winsorize(y_train_pos_cv, 0.01, 0.99)
  
  set.seed(625)
  dtrain_pos_cv <- xgb.DMatrix(
    data = X_train_cv_sev_ml_pos,
    label = y_train_pos_cv_trimmed
  )
  
  params_list_sev_cv <- list(
    max_depth = best_params_sev$max_depth,
    learning_rate = best_params_sev$learning_rate,
    objective = "reg:squarederror",
    eval_metric = "rmse",
    verbosity = 0
  )
  
  severity_xgb_cv <- xgb.train(
    params = params_list_sev_cv,
    data = dtrain_pos_cv,
    nrounds = best_params_sev$nrounds
  )
  
  mu_xgb_cv_all <- pmax(predict(severity_xgb_cv, X_test_cv_sev_ml), 0.01)
  max_reasonable_cv <- quantile(y_train_pos_cv_trimmed, 0.999, na.rm = TRUE)
  mu_xgb_cv_all <- pmin(mu_xgb_cv_all, max_reasonable_cv)
  
  pred_cv <- combine_two_part(p_freq_cv, mu_xgb_cv_all)
  
  compute_metrics(y_test_cv, pred_cv, paste0("Fold_", fold))
})

cv_summary <- cv_results %>%
  summarise(
    Model = "Two-Part (XGBoost + XGBoost) - CV (V3)",
    MAE_mean = round(mean(MAE), 2),
    MAE_sd = round(sd(MAE), 2),
    RMSE_mean = round(mean(RMSE), 2),
    RMSE_sd = round(sd(RMSE), 2),
    RMSLE_mean = round(mean(RMSLE), 4),
    RMSLE_sd = round(sd(RMSLE), 4),
    RMSLE_NonZero_mean = round(mean(RMSLE_NonZero, na.rm = TRUE), 4),
    RMSLE_NonZero_sd = round(sd(RMSLE_NonZero, na.rm = TRUE), 4)
  )

cat("\nCross-Validation Results:\n")
print(cv_summary)
write.csv(cv_results,
          file = file.path(output_dir, "cv_results_v3.csv"),
          row.names = FALSE
)
write.csv(cv_summary,
          file = file.path(output_dir, "cv_summary_v3.csv"),
          row.names = FALSE
)

cat("Calibration analysis...\n")

calibration_data <- data.frame(
  actual = y_test,
  predicted = best_pred
) %>%
  mutate(
    pred_decile = ntile(predicted, 10)
  ) %>%
  group_by(pred_decile) %>%
  summarise(
    mean_predicted = mean(predicted),
    mean_actual = mean(actual),
    n = n(),
    .groups = "drop"
  )

png(file.path(output_dir, "calibration", "calibration_curve_v3.png"),
    width = 10, height = 6, units = "in", res = 300
)
plot(calibration_data$mean_predicted, calibration_data$mean_actual,
     xlab = "Mean Predicted", ylab = "Mean Actual",
     main = "Calibration Curve: Predicted vs Actual by Decile (V3)",
     pch = 19, cex = 1.5
)
abline(0, 1, col = "red", lty = 2, lwd = 2)
text(
  calibration_data$mean_predicted, calibration_data$mean_actual,
  labels = calibration_data$pred_decile, pos = 3, cex = 0.8
)
dev.off()

png(file.path(output_dir, "calibration", "qq_calibration_v3.png"),
    width = 10, height = 6, units = "in", res = 300
)
pred_quantiles <- quantile(best_pred, probs = seq(0.1, 0.9, by = 0.1), na.rm = TRUE)
actual_quantiles <- quantile(y_test, probs = seq(0.1, 0.9, by = 0.1), na.rm = TRUE)
plot(pred_quantiles, actual_quantiles,
     xlab = "Predicted Quantiles", ylab = "Actual Quantiles",
     main = "Q-Q Plot: Predicted vs Actual Quantiles (V3)",
     pch = 19
)
abline(0, 1, col = "red", lty = 2, lwd = 2)
dev.off()

write.csv(calibration_data,
          file = file.path(output_dir, "calibration", "calibration_data_v3.csv"),
          row.names = FALSE
)

cat("Calibration plots saved to output/calibration/\n")

cat("\n=== Sensitivity Analysis: Model without chronic_count ===\n\n")

set.seed(625)
freq_glm_no_chronic <- glm(
  any_cost ~ age + age2 + sex + race,
  data = train_data,
  family = binomial(link = "logit")
)

p_freq_glm_no_chronic_test <- predict(
  freq_glm_no_chronic,
  newdata = test_data,
  type = "response"
)
auc_glm_no_chronic <- as.numeric(pROC::auc(any_cost_test, p_freq_glm_no_chronic_test))

cat("Frequency model without chronic_count:\n")
cat("  Test AUC:", round(auc_glm_no_chronic, 4), "\n")
cat("  Main model AUC:", round(as.numeric(pROC::auc(any_cost_test, p_freq_glm_test)), 4), "\n")
cat("  AUC drop:", round(as.numeric(pROC::auc(any_cost_test, p_freq_glm_test)) - auc_glm_no_chronic, 4), "\n\n")

X_train_freq_ml_no_chronic <- train_data %>%
  select(age, age2) %>%
  mutate(
    sex_male = as.numeric(train_data$sex == "Male"),
    race_black = as.numeric(train_data$race == "Black"),
    race_other = as.numeric(train_data$race == "Other"),
    race_asian = as.numeric(train_data$race == "Asian"),
    race_hispanic = as.numeric(train_data$race == "Hispanic"),
    race_native = as.numeric(train_data$race == "Native")
  ) %>%
  as.matrix()

X_test_freq_ml_no_chronic <- test_data %>%
  select(age, age2) %>%
  mutate(
    sex_male = as.numeric(test_data$sex == "Male"),
    race_black = as.numeric(test_data$race == "Black"),
    race_other = as.numeric(test_data$race == "Other"),
    race_asian = as.numeric(test_data$race == "Asian"),
    race_hispanic = as.numeric(test_data$race == "Hispanic"),
    race_native = as.numeric(test_data$race == "Native")
  ) %>%
  as.matrix()

set.seed(625)
dtrain_freq_no_chronic <- xgb.DMatrix(
  data = X_train_freq_ml_no_chronic,
  label = any_cost_train
)

params_no_chronic <- list(
  max_depth = 6,
  learning_rate = 0.1,
  objective = "binary:logistic",
  eval_metric = "logloss",
  verbosity = 0
)

freq_xgb_no_chronic <- xgb.train(
  params = params_no_chronic,
  data = dtrain_freq_no_chronic,
  nrounds = 100
)

p_freq_xgb_no_chronic_test <- predict(freq_xgb_no_chronic, X_test_freq_ml_no_chronic)
auc_xgb_no_chronic <- as.numeric(pROC::auc(any_cost_test, p_freq_xgb_no_chronic_test))

cat("XGBoost frequency model without chronic_count:\n")
cat("  Test AUC:", round(auc_xgb_no_chronic, 4), "\n")
cat("  Main model AUC:", round(as.numeric(pROC::auc(any_cost_test, p_freq_xgb_test)), 4), "\n")
cat("  AUC drop:", round(as.numeric(pROC::auc(any_cost_test, p_freq_xgb_test)) - auc_xgb_no_chronic, 4), "\n\n")

sensitivity_results <- data.frame(
  Model = c(
    "GLM (with chronic_count)", "GLM (without chronic_count)",
    "XGBoost (with chronic_count)", "XGBoost (without chronic_count)"
  ),
  AUC = c(
    round(as.numeric(pROC::auc(any_cost_test, p_freq_glm_test)), 4),
    round(auc_glm_no_chronic, 4),
    round(as.numeric(pROC::auc(any_cost_test, p_freq_xgb_test)), 4),
    round(auc_xgb_no_chronic, 4)
  ),
  AUC_Drop = c(
    0,
    round(as.numeric(pROC::auc(any_cost_test, p_freq_glm_test)) - auc_glm_no_chronic, 4),
    0,
    round(as.numeric(pROC::auc(any_cost_test, p_freq_xgb_test)) - auc_xgb_no_chronic, 4)
  )
)

write.csv(
  sensitivity_results,
  file = file.path(output_dir, "sensitivity_analysis_v3.csv"),
  row.names = FALSE
)

cat("Sensitivity analysis results saved to output/sensitivity_analysis_v3.csv\n\n")

cat("\n=== Saving Models and Predictions ===\n")

save(
  freq_glm, freq_xgb,
  severity_lognorm, severity_gamma, severity_xgb,
  single_glm,
  threshold_glm, threshold_xgb,
  best_params, best_params_sev,
  file = file.path(output_dir, "fitted_models_v3.rda")
)

test_predictions_v3 <- data.frame(
  DESYNPUF_ID = test_data$DESYNPUF_ID,
  total_payment = y_test,
  pred_best_v3 = best_pred,
  any_cost = any_cost_test
)

saveRDS(test_predictions_v3,
        file = file.path(output_dir, "test_predictions_v3.rds")
)
write.csv(
  test_predictions_v3,
  file = file.path(output_dir, "test_predictions_v3.csv"),
  row.names = FALSE
)

cat("All models and predictions saved.\n")
cat("\n=== Modeling Complete ===\n")


