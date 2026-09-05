"""
Pulls the actual borderline (0.4-0.8 score) and false-positive transactions
from the held-out test set, using the SAME split as train.py (random_state=42),
so this is real held-out data, not cherry-picked.

Produces:
  data/demo_borderline_cases.csv  -> transactions with score in [0.4, 0.8]
  data/demo_false_positives.csv   -> flagged (score>=0.5) but isFraud==0
  data/demo_showcase.csv          -> combined file, ready to upload in the dashboard
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from features import add_features, FEATURE_COLUMNS, CATEGORICAL_COLUMNS

MODEL_PATH = "models/fraud_model.joblib"
FLAG_THRESHOLD = 0.5


def main():
    df = pd.read_csv("data/transactions_features.csv")
    X = df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    y = df["isFraud"]

    # Reproduce the exact same held-out split used in train.py
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, stratify=y, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    test_df = df.loc[idx_test].copy()
    test_df["fraud_score"] = model.predict_proba(X_test)[:, 1]
    test_df["flagged"] = test_df["fraud_score"] >= FLAG_THRESHOLD

    # Borderline: model genuinely unsure, regardless of ground truth
    borderline = test_df[(test_df["fraud_score"] >= 0.4) & (test_df["fraud_score"] <= 0.8)]
    borderline = borderline.sort_values("fraud_score", ascending=False)

    # Real false positives: flagged but actually legit
    false_positives = test_df[(test_df["flagged"]) & (test_df["isFraud"] == 0)]
    false_positives = false_positives.sort_values("fraud_score", ascending=False)

    # Real false negatives too, since these are the honest "exceptions we couldn't resolve"
    false_negatives = test_df[(~test_df["flagged"]) & (test_df["isFraud"] == 1)]

    print(f"Held-out test set: {len(test_df)} rows")
    print(f"Borderline (0.4-0.8 score): {len(borderline)} rows")
    print(f"False positives (flagged, actually legit): {len(false_positives)} rows")
    print(f"False negatives (missed fraud): {len(false_negatives)} rows")

    borderline.to_csv("data/demo_borderline_cases.csv", index=False)
    false_positives.to_csv("data/demo_false_positives.csv", index=False)
    false_negatives.to_csv("data/demo_false_negatives.csv", index=False)

    # Build a mixed showcase file: some clear catches + all FPs + all FNs + borderline cases,
    # in original schema (so it can be re-uploaded through the dashboard's CSV uploader)
    original_cols = ["step_time", "type", "amount", "nameOrig", "oldbalanceOrg",
                      "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]
    clear_catches = test_df[(test_df["flagged"]) & (test_df["isFraud"] == 1)].sort_values(
        "fraud_score", ascending=False
    ).head(5)

    showcase = pd.concat([
        clear_catches[original_cols],
        false_positives[original_cols],
        false_negatives[original_cols],
        borderline[original_cols],
    ]).drop_duplicates().reset_index(drop=True)
    showcase.to_csv("data/demo_showcase.csv", index=False)
    print(f"\nWrote data/demo_showcase.csv ({len(showcase)} rows) — upload this in the dashboard "
          f"for an honest mixed demo: clear catches + false positives + missed fraud + borderline cases.")


if __name__ == "__main__":
    main()
