# ============================================================================
# Multilevel Modeling for Hospital LOS Prediction
# Author: Marzieh Amiri Shahbazi (RIT)
# ============================================================================
#
# This script fits three models to predict hospital length of stay:
#   - Model 1: patient variables only (baseline)
#   - Model 2: patient + hospital features  
#   - Model 3: hierarchical model with hospital random effects
#
# Data: 2019 National Inpatient Sample (NIS)
# ============================================================================


# --- Libraries ---
library(data.table)
library(mgcv)
library(nlme)
library(ggplot2)
library(caret)
library(dplyr)
library(tidyr)
library(purrr)
library(MASS)
library(car) 
library(plotly)
library(stringr)
library(forcats)


# --- Config ---
# Change this to wherever your data lives
DATA_PATH <- "data/NIS_2019_processed.csv"
OUTPUT_DIR <- "results/"

set.seed(42)


# --- Load and prep data ---

my_data <- fread(DATA_PATH)

# Grab a sample if the full dataset is too big
# (NIS has ~7 million records, 1 million is enough for this)
my_data <- my_data[sample(nrow(my_data), 1000000), ]

# Remove outliers — anything above mean + 1.96*SD
# These are usually coding errors or very unusual cases
mean_los <- mean(my_data$LOS, na.rm = TRUE)
sd_los <- sd(my_data$LOS, na.rm = TRUE)
upper_bound <- mean_los + (1.96 * sd_los)
my_data <- my_data[my_data$LOS <= upper_bound, ]

# Convert categorical vars to factors
# (R needs these as factors for the models to work right)
categorical_vars <- c(
  "DIED", "ELECTIVE", "FEMALE", "I10_INJURY", 
  "PAY1", "RACE", "ZIPINC_QRTL",
  "APRDRG_Risk_Mortality", "APRDRG_Severity", "has_comorbidities",
  "HOSP_NIS", "HOSP_BEDSIZE", "HOSP_LOCTEACH", "HOSP_REGION", "H_CONTRL"
)
my_data[, (categorical_vars) := lapply(.SD, as.factor), .SDcols = categorical_vars]

# 80/20 train/test split
train_idx <- sample(1:nrow(my_data), 0.8 * nrow(my_data))
train_data <- my_data[train_idx, ]
test_data <- my_data[-train_idx, ]


# --- Model 1: Patient variables only ---
# This is our baseline — what you'd get if you ignored hospital entirely

fit_model1 <- function(data) {
  gam(
    LOS ~ s(AGE, k = 5) + s(I10_NDX, k = 5) + s(I10_NPR, k = 5) + 
      DIED + ELECTIVE + FEMALE + I10_INJURY + PAY1 + RACE + ZIPINC_QRTL +  
      APRDRG_Risk_Mortality + APRDRG_Severity + has_comorbidities,
    data = data,
    family = Gamma(link = "log"),
    method = "REML"
  )
}

model1 <- fit_model1(train_data)
summary(model1)


# --- Model 2: Add hospital features (but no random effects) ---
# Now we're including hospital characteristics as fixed effects
# This helps, but treats every hospital as independent

fit_model2 <- function(data) {
  gam(
    LOS ~ s(AGE, k = 5) + s(I10_NDX, k = 5) + s(I10_NPR, k = 5) + 
      DIED + ELECTIVE + FEMALE + I10_INJURY + PAY1 + RACE + ZIPINC_QRTL +
      APRDRG_Risk_Mortality + APRDRG_Severity + has_comorbidities +
      HOSP_BEDSIZE + HOSP_LOCTEACH + HOSP_REGION + H_CONTRL,
    data = data,
    family = Gamma(link = "log"),
    method = "REML"
  )
}

model2 <- fit_model2(train_data)
summary(model2)


# --- Model 3: Hierarchical model with hospital random effects ---
# This is the key model — it recognizes that patients within a hospital
# are correlated. Each hospital gets its own intercept.

fit_model3 <- function(data) {
  gamm(
    LOS ~ s(AGE, k = 5) + s(I10_NDX, k = 5) + s(I10_NPR, k = 5) + 
      DIED + ELECTIVE + FEMALE + I10_INJURY + PAY1 + RACE + ZIPINC_QRTL +
      APRDRG_Risk_Mortality + APRDRG_Severity + has_comorbidities +
      HOSP_BEDSIZE + HOSP_LOCTEACH + HOSP_REGION + H_CONTRL,
    random = list(HOSP_NIS = ~1),
    data = data,
    family = Gamma(link = "log"),
    method = "REML"
  )
}

model3 <- fit_model3(train_data)
summary(model3$gam)


# --- Evaluate on test set ---

# Helper function to calculate metrics
calc_metrics <- function(actual, predicted) {
  mse <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual - predicted))
  ss_tot <- sum((actual - mean(actual))^2)
  ss_res <- sum((actual - predicted)^2)
  r2 <- 1 - (ss_res / ss_tot)
  
  return(c(RMSE = rmse, MAE = mae, R2 = r2))
}

# Get predictions
test_data$pred1 <- predict(model1, newdata = test_data, type = "response")
test_data$pred2 <- predict(model2, newdata = test_data, type = "response")
test_data$pred3 <- predict(model3$gam, newdata = test_data, type = "response")

# Compare
cat("\n--- Test Set Results ---\n")
cat("Model 1 (patient only):\n")
print(calc_metrics(test_data$LOS, test_data$pred1))

cat("\nModel 2 (+ hospital features):\n")
print(calc_metrics(test_data$LOS, test_data$pred2))

cat("\nModel 3 (hierarchical):\n")
print(calc_metrics(test_data$LOS, test_data$pred3))


# --- Cross-validation ---
# 5-fold CV to get more robust estimates

run_cv <- function(data, fit_fn, k = 5) {
  folds <- createFolds(data$LOS, k = k, list = TRUE)
  
  results <- lapply(folds, function(test_idx) {
    train_fold <- data[-test_idx, ]
    test_fold <- data[test_idx, ]
    
    mod <- fit_fn(train_fold)
    
    # Handle both gam and gamm objects
    if ("gam" %in% class(mod)) {
      preds <- predict(mod, newdata = test_fold, type = "response")
    } else {
      preds <- predict(mod$gam, newdata = test_fold, type = "response")
    }
    
    return(calc_metrics(test_fold$LOS, preds))
  })
  
  results_df <- do.call(rbind, results)
  return(colMeans(results_df))
}

cat("\n--- Cross-Validation Results ---\n")
cat("Model 1:\n")
print(run_cv(train_data, fit_model1))

cat("\nModel 2:\n")
print(run_cv(train_data, fit_model2))

# Note: Model 3 CV takes a while because GAMM is slow
# Uncomment if you have time:
# cat("\nModel 3:\n")
# print(run_cv(train_data, fit_model3))


# --- Forest plot ---
# This creates a nice visualization of the coefficients

# Rename variables so the plot looks publication-ready
rename_vars <- function(term) {
  case_when(
    str_detect(term, "^APRDRG_Severity") ~ paste0("Severity: ", str_extract(term, "\\d$")),
    str_detect(term, "^APRDRG_Risk_Mortality") ~ paste0("Mortality Risk: ", str_extract(term, "\\d$")),
    term == "DIED1" ~ "Died",
    term == "FEMALE1" ~ "Female",
    term == "has_comorbidities1" ~ "Comorbidities",
    term == "ELECTIVE1" ~ "Elective",
    str_detect(term, "^ZIPINC_QRTL") ~ paste0("Income Q", str_extract(term, "\\d$")),
    str_detect(term, "^PAY1") ~ case_when(
      str_detect(term, "1$") ~ "Medicare",
      str_detect(term, "2$") ~ "Medicaid", 
      str_detect(term, "3$") ~ "Private",
      str_detect(term, "4$") ~ "Self-pay",
      TRUE ~ "Other payer"
    ),
    str_detect(term, "^RACE") ~ paste0("Race: ", str_extract(term, "\\d$")),
    str_detect(term, "^HOSP_REGION") ~ case_when(
      str_detect(term, "1$") ~ "Northeast",
      str_detect(term, "2$") ~ "Midwest",
      str_detect(term, "3$") ~ "South",
      str_detect(term, "4$") ~ "West"
    ),
    str_detect(term, "^HOSP_BEDSIZE") ~ case_when(
      str_detect(term, "1$") ~ "Small hospital",
      str_detect(term, "2$") ~ "Medium hospital",
      str_detect(term, "3$") ~ "Large hospital"
    ),
    str_detect(term, "^HOSP_LOCTEACH") ~ case_when(
      str_detect(term, "1$") ~ "Rural",
      str_detect(term, "2$") ~ "Urban non-teaching",
      str_detect(term, "3$") ~ "Urban teaching"
    ),
    str_detect(term, "^H_CONTRL") ~ case_when(
      str_detect(term, "1$") ~ "Government",
      str_detect(term, "2$") ~ "Private non-profit",
      str_detect(term, "3$") ~ "Private for-profit"
    ),
    term == "s(AGE)" ~ "Age",
    term == "s(I10_NDX)" ~ "# Diagnoses",
    term == "s(I10_NPR)" ~ "# Procedures",
    TRUE ~ term
  )
}

make_forest_plot <- function(model, title, filename) {
  
  # Figure out if it's a gam or gamm
  if ("gam" %in% class(model)) {
    gam_obj <- model
  } else {
    gam_obj <- model$gam
  }
  
  # Get coefficients
  coefs <- as.data.frame(summary(gam_obj)$p.table)
  coefs$term <- rownames(coefs)
  coefs <- coefs %>%
    filter(term != "(Intercept)") %>%
    mutate(
      term = rename_vars(term),
      lower = Estimate - 1.96 * `Std. Error`,
      upper = Estimate + 1.96 * `Std. Error`
    )
  
  # Plot
  p <- ggplot(coefs, aes(x = Estimate, y = reorder(term, Estimate))) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +
    geom_point(size = 2, color = "#2C3E50") +
    geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2, color = "#2C3E50") +
    labs(title = title, x = "Coefficient (log scale)", y = "") +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white"),
      panel.background = element_rect(fill = "white")
    )
  
  ggsave(filename, p, width = 10, height = 8, dpi = 300)
  return(p)
}

# Generate plots
dir.create(OUTPUT_DIR, showWarnings = FALSE)
make_forest_plot(model1, "Model 1: Patient Variables", paste0(OUTPUT_DIR, "forest_model1.png"))
make_forest_plot(model2, "Model 2: Patient + Hospital", paste0(OUTPUT_DIR, "forest_model2.png"))
make_forest_plot(model3, "Model 3: Hierarchical", paste0(OUTPUT_DIR, "forest_model3.png"))


# --- Residual diagnostics ---
# Quick check that the model assumptions aren't totally violated

check_residuals <- function(model, title) {
  if ("gam" %in% class(model)) {
    res <- resid(model)
    fit <- fitted(model)
  } else {
    res <- resid(model$gam)
    fit <- fitted(model$gam)
  }
  
  df <- data.frame(fitted = fit, residuals = res)
  
  p <- ggplot(df, aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.1, size = 0.5) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    geom_smooth(method = "loess", se = FALSE, color = "blue") +
    labs(title = title, x = "Fitted values", y = "Residuals") +
    theme_minimal()
  
  return(p)
}

# Save residual plots
ggsave(paste0(OUTPUT_DIR, "residuals_model1.png"), 
       check_residuals(model1, "Model 1 Residuals"), width = 8, height = 6)
ggsave(paste0(OUTPUT_DIR, "residuals_model2.png"), 
       check_residuals(model2, "Model 2 Residuals"), width = 8, height = 6)
ggsave(paste0(OUTPUT_DIR, "residuals_model3.png"), 
       check_residuals(model3, "Model 3 Residuals"), width = 8, height = 6)


cat("\nDone! Results saved to", OUTPUT_DIR, "\n")
