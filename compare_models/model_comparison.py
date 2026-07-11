
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time

def evaluate_models(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC()
    }

    results = []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
            "Recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
            "F1 Score": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "Training Time (s)": round(train_time, 4)
        })

    results_df = pd.DataFrame(results).sort_values(
        by="F1 Score", ascending=False
    )

    print(f"\\n{'='*80}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    return results_df


# ------------------------------------------------------------------
# TASK 1 - Customer Churn
# ------------------------------------------------------------------
try:
    churn = pd.read_csv("data/telecom_churn.csv")

    target_col = churn.columns[-1]

    X = churn.drop(columns=[target_col])
    y = churn[target_col]

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    evaluate_models(X, y, "Customer Churn")

except Exception as e:
    print("\\nCustomer churn dataset not found or could not be processed.")
    print("Reason:", e)


# ------------------------------------------------------------------
# TASK 2 - Handwritten Digits
# ------------------------------------------------------------------
digits, labels = load_digits(return_X_y=True)

evaluate_models(digits, labels, "Handwritten Digits")
