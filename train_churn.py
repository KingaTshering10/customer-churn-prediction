"""Train and export the notebook's logistic regression baseline."""

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).drop(columns=["customerID"])
    y = df.pop("Churn").map({"No": 0, "Yes": 1})
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    X = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y_test, labels),
        "Precision": precision_score(y_test, labels),
        "Recall": recall_score(y_test, labels),
        "F1-score": f1_score(y_test, labels),
        "ROC AUC": roc_auc_score(y_test, probabilities),
    }
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, labels))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, labels))
    destination = ROOT / "churn_logistic_model.pkl"
    dump({"model": model, "feature_names": X.columns.tolist(),
          "cat_cols": cat_cols}, destination)
    print(f"\nSaved model to: {destination}")


if __name__ == "__main__":
    main()
