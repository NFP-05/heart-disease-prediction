#### Load Libraries ####
library(tidyverse)
library(ggplot2)
library(dplyr)

#### Load cleaned dataset from Python preprocessing ####
df_cleaned <- read.csv("Data/heart_cleaned.csv")
print(head(df_cleaned))
cat("\nDataset shape:", nrow(df_cleaned), "rows x", ncol(df_cleaned), "columns\n")

#### Split data by heart disease status ####
no_disease <- df_cleaned %>% filter(HeartDisease == 0)
has_disease <- df_cleaned %>% filter(HeartDisease == 1)

cat("\nNo Heart Disease:", nrow(no_disease), "patients\n")
cat("Has Heart Disease:", nrow(has_disease), "patients\n")

#### Test 1: T-tests for continuous variables ####
cat("\n--- T-tests (Continuous Variables) ---\n")
continuous_vars <- c('Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak')

for (var in continuous_vars) {
  if (var %in% colnames(df_cleaned)) {
    t_test <- t.test(no_disease[[var]], has_disease[[var]])
    p_value <- t_test$p.value
    t_stat <- t_test$statistic
    significance <- ifelse(p_value < 0.05, "SIGNIFICANT", "Not significant")
    cat(sprintf("%-15s | t-stat: %7.3f | p-value: %.4f %s\n", 
                var, t_stat, p_value, significance))
  }
}

#### Test 2: Chi-square tests for categorical variables ####
cat("\n--- Chi-square Tests (Categorical Variables) ---\n")
categorical_vars <- c('Sex', 'ChestPainType', 'FastingBS', 'ExerciseAngina', 'ST_Slope')

for (var in categorical_vars) {
  if (var %in% colnames(df_cleaned)) {
    contingency_table <- table(df_cleaned[[var]], df_cleaned$HeartDisease)
    chi_test <- chisq.test(contingency_table)
    p_value <- chi_test$p.value
    chi2_stat <- chi_test$statistic
    significance <- ifelse(p_value < 0.05, "SIGNIFICANT", "Not significant")
    cat(sprintf("%-15s | chi2: %7.3f | p-value: %.4f %s\n", 
                var, chi2_stat, p_value, significance))
  }
}

#### Test 3: Correlation with target variable ####
cat("\n--- Correlation with Heart Disease ---\n")

# Prepare data for correlation (numeric only)
numeric_with_target <- df_cleaned %>%
  select(all_of(continuous_vars), HeartDisease) %>%
  select(where(is.numeric))

# Calculate correlations
correlations <- cor(numeric_with_target)["HeartDisease", ] %>%
  sort(decreasing = TRUE) %>%
  .[names(.) != "HeartDisease"]

print(correlations)

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
