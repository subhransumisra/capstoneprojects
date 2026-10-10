# Fetal Health Classification Capstone Project

## Project Overview
The objective of this project is to analyze Cardiotocogram (CTG) data to predict fetal health status (Normal, Suspect, or Pathological). Early and accurate detection of abnormal fetal states helps prevent birth complications and supports clinical decision-making.

---

## Dataset Description
- **Source:** Fetal Health Dataset (`fetal_health.csv`) containing 2,126 records of fetal cardiotocographic exams.
- **Target Variable:** `fetal_health` (Categorical: 1 - Normal, 2 - Suspect, 3 - Pathological).
- **Features:** 21 numerical features including baseline fetal heart rate, accelerations, uterine contractions, decelerations, short-term and long-term variability, and various histogram statistics.

---

## Exploratory Data Analysis (EDA) & Data Cleaning
1. **Data Hygiene:** Inspected dataset for missing values and duplicates. Removed 13 duplicate rows, resulting in 2,113 clean records.
2. **Class Distribution:** The target variable exhibits class imbalance, with the vast majority of cases classified as Normal (~78%), followed by Suspect (~14%), and Pathological (~8%).
3. **Outlier & Correlation Analysis:** Boxplots revealed physiological outliers in variables like decelerations and variability. Correlation heatmaps showed strong collinearity among histogram features (e.g., mode, mean, median) and short-term variability metrics.

---

## Feature Engineering
Two new domain-specific features were engineered:
1. `contraction_acceleration_ratio`: Ratio of uterine contractions to accelerations to capture fetal stress response.
2. `variability_index`: Interaction term between abnormal short-term variability and mean short-term variability.

---

## Baseline Machine Learning Model Results
- **Model:** Random Forest Classifier (Baseline)
- **Train/Test Split:** 80% training, 20% testing with stratification across classes.
- **Performance Metrics:**
  - **Overall Accuracy:** ~94%
  - **Macro Average F1-Score:** ~0.91
  - **Key Predictors:** `abnormal_short_term_variability`, `histogram_mean`, `percentage_of_time_with_abnormal_long_term_variability`, and `accelerations` emerged as the most critical features driving classification performance.

---

## Next Steps (Module 24)
- Address class imbalance using SMOTE or class weight adjustments.
- Tune hyperparameters across multiple algorithms (XGBoost, Gradient Boosting, Support Vector Machines).
- Finalize model evaluation using ROC-AUC curves and confusion matrices.
