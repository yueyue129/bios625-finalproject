# ============================================================================
# Exploratory Data Analysis (EDA) Script
# Medicare Cost Prediction Project - Biostat 625
# ============================================================================
# This script performs exploratory data analysis:
# 1. Cost distribution histograms (raw and log-transformed)
# 2. Zero-inflation analysis
# 3. Tail risk assessment
# 4. Summary statistics table
# ============================================================================

# Load required libraries
library(tidyverse)
library(ggplot2)
library(knitr)
library(kableExtra)

# Set paths
output_dir <- "output/"
if (!dir.exists(output_dir)) dir.create(output_dir)

# Load processed data
if (!exists("analysis_data")) {
  if (file.exists(paste0(output_dir, "analysis_data.rds"))) {
    analysis_data <- readRDS(paste0(output_dir, "analysis_data.rds"))
    cat("Loaded analysis_data from", paste0(output_dir, "analysis_data.rds"), "\n")
  } else {
    stop("Please run data_prep.R first to create analysis_data.rds")
  }
}

cat("Starting EDA...\n")
cat("Dataset dimensions:", nrow(analysis_data), "x", ncol(analysis_data), "\n\n")

# ============================================================================
# 1. Zero-Inflation Analysis
# ============================================================================

cat("=== Zero-Inflation Analysis ===\n")
n_total <- nrow(analysis_data)
n_zero <- sum(analysis_data$total_payment == 0)
n_positive <- sum(analysis_data$total_payment > 0)
prop_zero <- n_zero / n_total
prop_positive <- n_positive / n_total

cat("Total beneficiaries:", n_total, "\n")
cat("Zero cost:", n_zero, "(", round(prop_zero * 100, 2), "%)\n")
cat("Positive cost:", n_positive, "(", round(prop_positive * 100, 2), "%)\n\n")

# ============================================================================
# 2. Summary Statistics Table
# ============================================================================

cat("=== Summary Statistics ===\n")

# Overall summary
summary_overall <- analysis_data %>%
  summarise(
    Variable = "Total Payment (All)",
    N = n(),
    Mean = mean(total_payment),
    Median = median(total_payment),
    SD = sd(total_payment),
    Min = min(total_payment),
    Q25 = quantile(total_payment, 0.25),
    Q75 = quantile(total_payment, 0.75),
    Q90 = quantile(total_payment, 0.90),
    Q95 = quantile(total_payment, 0.95),
    Q99 = quantile(total_payment, 0.99),
    Max = max(total_payment)
  )

# Summary for positive costs only
summary_positive <- analysis_data %>%
  filter(total_payment > 0) %>%
  summarise(
    Variable = "Total Payment (Positive Only)",
    N = n(),
    Mean = mean(total_payment),
    Median = median(total_payment),
    SD = sd(total_payment),
    Min = min(total_payment),
    Q25 = quantile(total_payment, 0.25),
    Q75 = quantile(total_payment, 0.75),
    Q90 = quantile(total_payment, 0.90),
    Q95 = quantile(total_payment, 0.95),
    Q99 = quantile(total_payment, 0.99),
    Max = max(total_payment)
  )

# Combine summaries
summary_table <- bind_rows(summary_overall, summary_positive) %>%
  mutate(across(Mean:Max, ~ round(.x, 2)))

# Print summary table
print(summary_table)

# Save summary table
write.csv(summary_table, 
          file = paste0(output_dir, "eda_summary_statistics.csv"), 
          row.names = FALSE)

cat("\nSummary statistics saved to:", paste0(output_dir, "eda_summary_statistics.csv"), "\n\n")

# ============================================================================
# 3. Cost Distribution Histograms
# ============================================================================

cat("=== Creating Distribution Plots ===\n")

# 3.1 Raw Cost Histogram (All data)
p1 <- ggplot(analysis_data, aes(x = total_payment)) +
  geom_histogram(bins = 100, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(
    title = "Distribution of Total Medicare Payment (All Beneficiaries)",
    subtitle = paste0("N = ", n_total, " | Zero cost: ", n_zero, " (", round(prop_zero * 100, 1), "%)"),
    x = "Total Payment ($)",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

ggsave(
  filename = paste0(output_dir, "histogram_total_payment.png"),
  plot = p1,
  width = 10,
  height = 6,
  dpi = 300
)
cat("Saved: histogram_total_payment.png\n")

# 3.2 Raw Cost Histogram (Zoomed in, excluding extreme outliers)
# Remove top 1% for better visualization
p2 <- analysis_data %>%
  filter(total_payment <= quantile(total_payment, 0.99, na.rm = TRUE)) %>%
  ggplot(aes(x = total_payment)) +
  geom_histogram(bins = 100, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(
    title = "Distribution of Total Medicare Payment (Excluding Top 1%)",
    subtitle = paste0("Showing payments ≤ $", round(quantile(analysis_data$total_payment, 0.99), 0)),
    x = "Total Payment ($)",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

ggsave(
  filename = paste0(output_dir, "histogram_total_payment_zoomed.png"),
  plot = p2,
  width = 10,
  height = 6,
  dpi = 300
)
cat("Saved: histogram_total_payment_zoomed.png\n")

# 3.3 Log1p Cost Histogram (All data)
p3 <- ggplot(analysis_data, aes(x = log1p_cost)) +
  geom_histogram(bins = 100, fill = "darkgreen", color = "white", alpha = 0.7) +
  labs(
    title = "Distribution of log(1 + Total Payment)",
    subtitle = paste0("N = ", n_total, " | Transformation: log1p(x) = log(1 + x)"),
    x = "log(1 + Total Payment)",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

ggsave(
  filename = paste0(output_dir, "histogram_log1p_cost.png"),
  plot = p3,
  width = 10,
  height = 6,
  dpi = 300
)
cat("Saved: histogram_log1p_cost.png\n")

# 3.4 Log1p Cost Histogram (Positive costs only)
p4 <- analysis_data %>%
  filter(total_payment > 0) %>%
  ggplot(aes(x = log(total_payment))) +
  geom_histogram(bins = 100, fill = "darkgreen", color = "white", alpha = 0.7) +
  labs(
    title = "Distribution of log(Total Payment) - Positive Costs Only",
    subtitle = paste0("N = ", n_positive, " beneficiaries with positive costs"),
    x = "log(Total Payment)",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

ggsave(
  filename = paste0(output_dir, "histogram_log_cost_positive.png"),
  plot = p4,
  width = 10,
  height = 6,
  dpi = 300
)
cat("Saved: histogram_log_cost_positive.png\n")

# ============================================================================
# 4. Tail Risk Analysis
# ============================================================================

cat("\n=== Tail Risk Analysis ===\n")

# Calculate tail statistics
tail_stats <- analysis_data %>%
  filter(total_payment > 0) %>%
  summarise(
    P90 = quantile(total_payment, 0.90),
    P95 = quantile(total_payment, 0.95),
    P99 = quantile(total_payment, 0.99),
    P99.5 = quantile(total_payment, 0.995),
    P99.9 = quantile(total_payment, 0.999),
    Mean_above_P95 = mean(total_payment[total_payment > quantile(total_payment, 0.95)]),
    Mean_above_P99 = mean(total_payment[total_payment > quantile(total_payment, 0.99)])
  ) %>%
  round(2)

print(tail_stats)

# Tail distribution plot
p5 <- analysis_data %>%
  filter(total_payment > 0) %>%
  ggplot(aes(x = total_payment)) +
  geom_histogram(bins = 200, fill = "coral", color = "white", alpha = 0.7) +
  geom_vline(aes(xintercept = quantile(analysis_data$total_payment[analysis_data$total_payment > 0], 0.95)),
             color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = quantile(analysis_data$total_payment[analysis_data$total_payment > 0], 0.99)),
             color = "darkred", linetype = "dashed", size = 1) +
  scale_x_continuous(limits = c(quantile(analysis_data$total_payment[analysis_data$total_payment > 0], 0.90), 
                                max(analysis_data$total_payment))) +
  labs(
    title = "Tail Distribution of Total Payment (Top 10%)",
    subtitle = "Red dashed line: 95th percentile | Dark red: 99th percentile",
    x = "Total Payment ($)",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

ggsave(
  filename = paste0(output_dir, "histogram_tail_distribution.png"),
  plot = p5,
  width = 10,
  height = 6,
  dpi = 300
)
cat("Saved: histogram_tail_distribution.png\n")

# Save tail statistics
write.csv(tail_stats, 
          file = paste0(output_dir, "eda_tail_statistics.csv"), 
          row.names = FALSE)

# ============================================================================
# 5. Additional EDA: Payment by Service Type
# ============================================================================

cat("\n=== Payment by Service Type ===\n")

service_summary <- analysis_data %>%
  summarise(
    Service_Type = c("Inpatient", "Outpatient", "Carrier", "Total"),
    Mean_Payment = c(
      mean(ip_pay),
      mean(op_pay),
      mean(car_pay),
      mean(total_payment)
    ),
    Median_Payment = c(
      median(ip_pay),
      median(op_pay),
      median(car_pay),
      median(total_payment)
    ),
    P95_Payment = c(
      quantile(ip_pay, 0.95),
      quantile(op_pay, 0.95),
      quantile(car_pay, 0.95),
      quantile(total_payment, 0.95)
    ),
    Proportion_NonZero = c(
      mean(ip_pay > 0),
      mean(op_pay > 0),
      mean(car_pay > 0),
      mean(total_payment > 0)
    )
  ) %>%
  mutate(across(Mean_Payment:Proportion_NonZero, ~ round(.x, 2)))

print(service_summary)
write.csv(service_summary, 
          file = paste0(output_dir, "eda_service_type_summary.csv"), 
          row.names = FALSE)

# ============================================================================
# 6. Box Plot: Cost Distribution by Key Groups
# ============================================================================

# Box plot: Cost by sex
p6 <- analysis_data %>%
  filter(total_payment > 0) %>%
  ggplot(aes(x = sex, y = total_payment)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7, outlier.alpha = 0.3) +
  scale_y_log10() +
  labs(
    title = "Total Payment Distribution by Sex (Log Scale)",
    subtitle = "Positive costs only",
    x = "Sex",
    y = "Total Payment ($, log scale)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold")
  )

ggsave(
  filename = paste0(output_dir, "boxplot_cost_by_sex.png"),
  plot = p6,
  width = 8,
  height = 6,
  dpi = 300
)

# Box plot: Cost by chronic condition count
p7 <- analysis_data %>%
  filter(total_payment > 0) %>%
  mutate(chronic_group = cut(chronic_count, 
                             breaks = c(-1, 0, 1, 2, 3, Inf),
                             labels = c("0", "1", "2", "3", "4+"))) %>%
  ggplot(aes(x = chronic_group, y = total_payment)) +
  geom_boxplot(fill = "lightcoral", alpha = 0.7, outlier.alpha = 0.3) +
  scale_y_log10() +
  labs(
    title = "Total Payment Distribution by Chronic Condition Count (Log Scale)",
    subtitle = "Positive costs only",
    x = "Number of Chronic Conditions",
    y = "Total Payment ($, log scale)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold")
  )

ggsave(
  filename = paste0(output_dir, "boxplot_cost_by_chronic.png"),
  plot = p7,
  width = 8,
  height = 6,
  dpi = 300
)

cat("\n=== EDA Complete ===\n")
cat("All plots saved to:", output_dir, "\n")
cat("\nGenerated files:\n")
cat("  - eda_summary_statistics.csv\n")
cat("  - eda_tail_statistics.csv\n")
cat("  - eda_service_type_summary.csv\n")
cat("  - histogram_total_payment.png\n")
cat("  - histogram_total_payment_zoomed.png\n")
cat("  - histogram_log1p_cost.png\n")
cat("  - histogram_log_cost_positive.png\n")
cat("  - histogram_tail_distribution.png\n")
cat("  - boxplot_cost_by_sex.png\n")
cat("  - boxplot_cost_by_chronic.png\n")

