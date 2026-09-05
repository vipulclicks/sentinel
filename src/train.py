"""
Trains and compares two fraud classifiers on a held-out test set:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest (non-linear, usually higher recall)

Reports precision, recall, F1, ROC-AUC, confusion matrix, and an estimated
false-positive cost, then saves the better-performing model + a metrics report.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import FEATURE_COLUMNS, CATEGORICAL_COLUMNS

# --- Assumptions, stated explicitly so they're defensible in a demo ---
COST_PER_FALSE_POSITIVE = 50      # INR: cost of a support ticket / customer friction
COST_PER_FALSE_NEGATIVE = 3000    # INR: average unrecovered fraud loss per missed case


def load_data(path="data/transactions_features.csv"):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y = df["isFraud"]
    return X, y


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), FEATURE_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    fp_cost = fp * COST_PER_FALSE_POSITIVE
    fn_cost = fn * COST_PER_FALSE_NEGATIVE
    total_cost = fp_cost + fn_cost

    print(f"\n=== {name} ===")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Estimated cost: FP cost=₹{fp_cost:,} + FN cost=₹{fn_cost:,} = ₹{total_cost:,}")

    return {
        "name": name, "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fp_cost": int(fp_cost), "fn_cost": int(fn_cost), "total_cost": int(total_cost),
    }, y_proba


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} rows ({y_train.mean()*100:.2f}% fraud)")
    print(f"Test:  {len(X_test)} rows ({y_test.mean()*100:.2f}% fraud) — held out, untouched until final eval")

    preprocessor = build_preprocessor()

    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_metrics, lr_proba = evaluate("Logistic Regression (baseline)", lr_pipe, X_test, y_test)

    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_metrics, rf_proba = evaluate("Random Forest", rf_pipe, X_test, y_test)

    # Pick the model with lower total estimated cost (not just highest F1 —
    # this is the "honest metrics including false-positive cost" bar)
    candidates = [(lr_metrics, lr_pipe, lr_proba), (rf_metrics, rf_pipe, rf_proba)]
    best_metrics, best_pipe, best_proba = min(candidates, key=lambda c: c[0]["total_cost"])
    print(f"\n>>> Selected model: {best_metrics['name']} (lowest estimated total cost)")

    joblib.dump(best_pipe, "models/fraud_model.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump({
            "logistic_regression": lr_metrics,
            "random_forest": rf_metrics,
            "selected": best_metrics["name"],
            "assumptions": {
                "cost_per_false_positive_inr": COST_PER_FALSE_POSITIVE,
                "cost_per_false_negative_inr": COST_PER_FALSE_NEGATIVE,
            },
        }, f, indent=2)

    # Precision-recall curve plot for the demo
    plt.figure(figsize=(6, 5))
    for label, proba in [("Logistic Regression", lr_proba), ("Random Forest", rf_proba)]:
        p, r, _ = precision_recall_curve(y_test, proba)
        plt.plot(r, p, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Sentinel")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/pr_curve.png", dpi=120)
    print("\nSaved: models/fraud_model.joblib, models/metrics.json, models/pr_curve.png")


if __name__ == "__main__":
    main()
