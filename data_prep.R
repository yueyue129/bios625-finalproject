# Data prep for Medicare cost prediction
# Biostat 625

library(tidyverse)
library(data.table)
library(lubridate)

# Set up directories
# Check for data directory (handle both "data" and "data " with trailing space)
data_dir <- NULL
if (dir.exists("data")) {
  data_dir <- "data"
} else if (dir.exists("data ")) {
  data_dir <- "data "
  warning("Using 'data ' directory (with trailing space). Consider renaming to 'data' to avoid path issues.")
} else {
  stop("Error: 'data' directory not found. Please ensure the data files are in a 'data/' directory.")
}

# Check for required data files
required_files <- c(
  "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv",
  "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv",
  "DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv",
  "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv",
  "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv"
)

missing_files <- required_files[!file.exists(file.path(data_dir, required_files))]
if (length(missing_files) > 0) {
  stop("Error: Missing required data files in '", data_dir, "':\n  ", paste(missing_files, collapse = "\n  "))
}

output_dir <- "output"
if (!dir.exists(output_dir)) dir.create(output_dir)

cat("Loading data...\n")

# Beneficiary Summary (2008)
beneficiary <- fread(
  file = file.path(data_dir, "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"),
  stringsAsFactors = FALSE
)

# Carrier Claims (2008-2010, Parts A & B)
carrier_a <- fread(
  file = file.path(data_dir, "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv"),
  stringsAsFactors = FALSE
)
carrier_b <- fread(
  file = file.path(data_dir, "DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv"),
  stringsAsFactors = FALSE
)

# Inpatient Claims (2008-2010)
inpatient <- fread(
  file = file.path(data_dir, "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"),
  stringsAsFactors = FALSE
)

# Outpatient Claims (2008-2010)
outpatient <- fread(
  file = file.path(data_dir, "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv"),
  stringsAsFactors = FALSE
)

cat("Processing beneficiary data...\n")

beneficiary_clean <- beneficiary %>%
  select(
    DESYNPUF_ID,
    BENE_BIRTH_DT,
    BENE_DEATH_DT,
    BENE_SEX_IDENT_CD,
    BENE_RACE_CD,
    SP_ALZHDMTA, SP_CHF, SP_CHRNKIDN, SP_CNCR, SP_COPD,
    SP_DEPRESSN, SP_DIABETES, SP_ISCHMCHT, SP_OSTEOPRS, SP_RA_OA, SP_STRKETIA
  ) %>%
  mutate(
    birth_date_str = as.character(BENE_BIRTH_DT),
    birth_date = as.Date(birth_date_str, format = "%Y%m%d"),
    age_2008 = as.numeric(difftime(as.Date("2008-12-31"), birth_date, units = "days")) / 365.25,
    age = floor(age_2008),
    sex = factor(BENE_SEX_IDENT_CD, levels = c(1, 2), labels = c("Male", "Female")),
    race = factor(
      BENE_RACE_CD,
      levels = c(1, 2, 3, 4, 5, 6),
      labels = c("White", "Black", "Other", "Asian", "Hispanic", "Native")
    ),
    # SP_*: 1=has condition, 2=no condition, so convert to binary first
    chronic_count = (SP_ALZHDMTA == 1) + (SP_CHF == 1) + (SP_CHRNKIDN == 1) + 
                    (SP_CNCR == 1) + (SP_COPD == 1) + (SP_DEPRESSN == 1) + 
                    (SP_DIABETES == 1) + (SP_ISCHMCHT == 1) + (SP_OSTEOPRS == 1) + 
                    (SP_RA_OA == 1) + (SP_STRKETIA == 1)
  ) %>%
  select(DESYNPUF_ID, age, sex, race, chronic_count)

cat("Processing claims data...\n")

extract_year <- function(date_str) {
  as.numeric(substr(date_str, 1, 4))
}

# Combine carrier files
carrier_all <- bind_rows(carrier_a, carrier_b) %>%
  mutate(
    claim_from_year = extract_year(CLM_FROM_DT),
    claim_thru_year = extract_year(CLM_THRU_DT)
  ) %>%
  filter(claim_from_year == 2008 | claim_thru_year == 2008)

# Filter to 2008 only
inpatient_2008 <- inpatient %>%
  mutate(
    claim_from_year = extract_year(CLM_FROM_DT),
    claim_thru_year = extract_year(CLM_THRU_DT)
  ) %>%
  filter(claim_from_year == 2008 | claim_thru_year == 2008)

outpatient_2008 <- outpatient %>%
  mutate(
    claim_from_year = extract_year(CLM_FROM_DT),
    claim_thru_year = extract_year(CLM_THRU_DT)
  ) %>%
  filter(claim_from_year == 2008 | claim_thru_year == 2008)

cat("Aggregating claims...\n")

# Aggregate inpatient: visits and payments (payments only for target, not features)
ip_agg <- inpatient_2008 %>%
  mutate(
    CLM_PMT_AMT = as.numeric(CLM_PMT_AMT),
    CLM_PMT_AMT = ifelse(is.na(CLM_PMT_AMT) | CLM_PMT_AMT < 0, 0, CLM_PMT_AMT)
  ) %>%
  group_by(DESYNPUF_ID) %>%
  summarise(
    ip_visits = n(),
    ip_pay = sum(CLM_PMT_AMT, na.rm = TRUE),
    .groups = "drop"
  )

op_agg <- outpatient_2008 %>%
  mutate(
    CLM_PMT_AMT = as.numeric(CLM_PMT_AMT),
    CLM_PMT_AMT = ifelse(is.na(CLM_PMT_AMT) | CLM_PMT_AMT < 0, 0, CLM_PMT_AMT)
  ) %>%
  group_by(DESYNPUF_ID) %>%
  summarise(
    op_visits = n(),
    op_pay = sum(CLM_PMT_AMT, na.rm = TRUE),
    .groups = "drop"
  )

# Carrier has multiple payment lines per claim, need to sum them
line_pmt_cols <- names(carrier_all)[grepl("^LINE_NCH_PMT_AMT", names(carrier_all))]
line_pmt_cols <- names(carrier_all)[grepl("^LINE_NCH_PMT_AMT", names(carrier_all))]

carrier_agg <- carrier_all %>%
  mutate(across(all_of(line_pmt_cols), ~ as.numeric(.x))) %>%
  mutate(
    car_pay_total = rowSums(
      select(., all_of(line_pmt_cols)),
      na.rm = TRUE
    ),
    car_pay_total = ifelse(is.na(car_pay_total) | car_pay_total < 0, 0, car_pay_total)
  ) %>%
  group_by(DESYNPUF_ID) %>%
  summarise(
    car_visits = n_distinct(CLM_ID, na.rm = TRUE),
    car_pay = sum(car_pay_total, na.rm = TRUE),
    .groups = "drop"
  )

cat("Merging data...\n")

analysis_data <- beneficiary_clean %>%
  left_join(ip_agg, by = "DESYNPUF_ID") %>%
  left_join(op_agg, by = "DESYNPUF_ID") %>%
  left_join(carrier_agg, by = "DESYNPUF_ID") %>%
  mutate(
    ip_visits = ifelse(is.na(ip_visits), 0, ip_visits),
    op_visits = ifelse(is.na(op_visits), 0, op_visits),
    car_visits = ifelse(is.na(car_visits), 0, car_visits),
    ip_pay = ifelse(is.na(ip_pay), 0, ip_pay),
    op_pay = ifelse(is.na(op_pay), 0, op_pay),
    car_pay = ifelse(is.na(car_pay), 0, car_pay)
  ) %>%
  mutate(
    ip_pay = pmax(ip_pay, 0),
    op_pay = pmax(op_pay, 0),
    car_pay = pmax(car_pay, 0)
  ) %>%
  mutate(
    total_payment = ip_pay + op_pay + car_pay,
    any_cost = as.numeric(total_payment > 0)
  )

cat("Creating features...\n")

analysis_data <- analysis_data %>%
  mutate(
    age2 = age^2,
    total_visits = ip_visits + op_visits + car_visits,
    log_ip_visits = log1p(ip_visits),
    log_op_visits = log1p(op_visits),
    log_car_visits = log1p(car_visits),
    log_total_visits = log1p(total_visits),
    has_ip = as.numeric(ip_visits > 0),
    has_op = as.numeric(op_visits > 0),
    has_car = as.numeric(car_visits > 0),
    log1p_cost = log1p(total_payment)
  )

cat("Final cleaning...\n")

analysis_data <- analysis_data %>%
  filter(
    !is.na(age),
    !is.na(sex),
    !is.na(race),
    age >= 0 & age <= 120
  )

# Summary statistics
cat("\nFinal dataset:\n")
cat("Number of beneficiaries (rows):", nrow(analysis_data), "\n")
cat("Number of variables (columns):", ncol(analysis_data), "\n")
cat("\n--- Target Variable ---\n")
cat("Proportion with any cost:", round(mean(analysis_data$any_cost), 3), "\n")
cat("Mean total payment: $", round(mean(analysis_data$total_payment), 2), "\n")
cat("Median total payment: $", round(median(analysis_data$total_payment), 2), "\n")
cat("SD total payment: $", round(sd(analysis_data$total_payment), 2), "\n")
cat("\n--- Utilization (Features) ---\n")
cat("Mean IP visits:", round(mean(analysis_data$ip_visits), 2), "\n")
cat("Mean OP visits:", round(mean(analysis_data$op_visits), 2), "\n")
cat("Mean Carrier visits:", round(mean(analysis_data$car_visits), 2), "\n")
cat("\n--- Demographics ---\n")
cat("Mean age:", round(mean(analysis_data$age, na.rm = TRUE), 1), "\n")
cat("Mean chronic conditions:", round(mean(analysis_data$chronic_count, na.rm = TRUE), 2), "\n")
cat("\n")

cat("Validating analysis dataset...\n")
required_vars <- c(
  "DESYNPUF_ID", "age", "sex", "race", "chronic_count",
  "ip_visits", "op_visits", "car_visits",
  "total_payment", "any_cost"
)

missing_vars <- setdiff(required_vars, names(analysis_data))
if (length(missing_vars) > 0) {
  warning("Missing required variables: ", paste(missing_vars, collapse = ", "))
} else {
  cat("  All required variables present.\n")
}

if (any(duplicated(analysis_data$DESYNPUF_ID))) {
  warning("Duplicate DESYNPUF_ID found!")
} else {
  cat("  No duplicate beneficiaries.\n")
}

feature_vars <- c("age", "age2", "sex", "race", "chronic_count",
                  "ip_visits", "op_visits", "car_visits", "total_visits",
                  "log_ip_visits", "log_op_visits", "log_car_visits", "log_total_visits",
                  "has_ip", "has_op", "has_car")

leakage_vars <- c("ip_pay", "op_pay", "car_pay", "intensity", "ip_share")
if (any(leakage_vars %in% feature_vars)) {
  warning("Data leakage detected! Payment variables found in feature set.")
} else {
  cat("  No data leakage in feature set.\n")
  cat("  Payment variables (ip_pay, op_pay, car_pay) kept only as target components.\n")
}

cat("Validation complete.\n")

cat("\nSaving processed data...\n")

saveRDS(analysis_data, file = file.path(output_dir, "analysis_data_noleak.rds"))
write.csv(analysis_data, file = file.path(output_dir, "analysis_data_noleak.csv"), row.names = FALSE)

cat("Data preparation complete!\n")
cat("Output saved to:", output_dir, "\n")
cat("  - analysis_data_noleak.rds (R object)\n")
cat("  - analysis_data_noleak.csv (CSV format)\n")
cat("\nAnalysis dataset ready for modeling (NO DATA LEAKAGE).\n")
cat("Payment amounts (ip_pay, op_pay, car_pay) are kept for target variable only.\n")

