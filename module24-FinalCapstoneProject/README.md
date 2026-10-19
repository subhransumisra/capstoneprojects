# Capstone Assignment: Comparative Analysis of Classification Models

## 1. Executive Summary & Overview

This project provides a comprehensive comparative evaluation of four fundamental machine learning classification algorithms:
* **Logistic Regression**
* **Decision Trees**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machines (SVM)**

The study evaluates these algorithms across two distinct machine learning tasks—**Customer Churn Prediction** (imbalanced binary classification) and **Handwritten Digit Recognition** (multiclass image classification)—to determine optimal model selection based on performance, speed, interpretability, and class imbalance handling.

---

## 2. Problem Statement & Research Questions

### Research Question
> **How do Logistic Regression, Decision Trees, KNN, and SVM compare in terms of predictive performance, computational efficiency, interpretability, and ability to handle class imbalance across customer churn prediction and handwritten digit recognition tasks?**

### Domain Contexts
* **Use Case 1: Customer Churn Prediction** (`telecom_churn.csv`)  
  Predict whether a telecom customer is likely to leave the service using behavioral and demographic variables.
* **Use Case 2: Handwritten Digit Recognition** (`sklearn.datasets.load_digits`)  
  Classify images of handwritten digits ($0–9$) using $8 \times 8$ pixel intensity values ($64$ feature dimensions).

---

## 3. Business Goals & Objectives

The goal is to establish a clear decision framework for model selection by balancing prediction quality, resource constraints, and explainability.

* **Customer Retention:** Identify at-risk customers early to enable proactive, targeted retention campaigns.
* **Pattern Recognition:** Demonstrate effective image classification across non-linear multi-dimensional vector spaces.
* **Model Trade-off Analysis:** Quantify the operational trade-offs between predictive accuracy, training/inference speed, and model interpretability.

---

## 4. Dataset Overview

| Dataset | Source | Characteristics | Target Variable | Type |
| :--- | :--- | :--- | :--- | :--- |
| **Telecom Customer Churn** | `telecom_churn.csv` | Customer demographics & usage metrics; skewed distribution | `Churn` (True / False) | Imbalanced Binary |
| **Handwritten Digits** | `sklearn.datasets.load_digits` | 1,797 samples, 64 numerical pixel features ($8 \times 8$ matrix) | `Digit` ($0$ through $9$) | Balanced Multiclass |

### Data Pipeline & Preprocessing Workflow
1. **Exploration & Cleaning:** Load datasets, check for missing values, and analyze target class distributions.
2. **Data Partitioning:** Split data into training and test sets.
3. **Feature Scaling:** Apply standard scaling (`StandardScaler`), critical for distance- and margin-based algorithms (KNN, SVM, Logistic Regression).
4. **Evaluation:** Compute appropriate metrics for binary classification (Precision, Recall, F1-Score, ROC-AUC) and multiclass classification (Accuracy, Macro/Weighted F1).

---

## 5. Summary of Recommended Models

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Model Selection Strategy                          │
├──────────────────────────────┬──────────────────────────────────────────┤
│ Customer Churn               │ Handwritten Digit Recognition            │
│ (Imbalanced Binary)          │ (Multiclass Image Data)                  │
├──────────────────────────────┼──────────────────────────────────────────┤
│ 🏆 Logistic Regression /     │ 🏆 Support Vector Machine (SVM, RBF) /   │
│    Decision Trees            │    K-Nearest Neighbors (KNN)             │
│                              │                                          │
│ • High interpretability      │ • High accuracy in non-linear vector     │
│ • Calibrated probabilities   │   spaces                                 │
│ • Supports class weighting   │ • Feature interpretability is secondary  │
└──────────────────────────────┴──────────────────────────────────────────┘
```

---

## 6. Repository Structure

```
.
├── Compare_model.ipynb         # Main exploration notebook analyzing model trade-offs
├── CapstoneEvaluation.ipynb    # Rigorous capstone rubric evaluation document
└── README.md                   # Project overview and documentation
```

---

## 7. Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Launch Jupyter Environment:**
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

3. **Run Analysis:**
   * Open `Compare_model.ipynb` to view feature engineering, model training, and performance metrics.
   * Open `CapstoneEvaluation.ipynb` for the detailed model evaluation against the project rubric.