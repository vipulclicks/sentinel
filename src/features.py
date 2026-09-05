"""
Feature engineering for Sentinel fraud detector.

Adds:
- errorBalanceOrig / errorBalanceDest: mismatch between expected and actual
  post-transaction balances (a classic PaySim-style tell)
- velocity features: # transactions by this origin account in the last
  5 / 60 minutes (sliding window via groupby + rolling)
- amount_to_balance_ratio: how much of the origin's balance this txn drains
- is_new_dest: whether this is the first time this origin has sent to this dest
- seconds_since_last_txn: gap since this account's previous transaction
"""
import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["step_time"] = pd.to_datetime(df["step_time"])
    df = df.sort_values("step_time").reset_index(drop=True)

    # Balance-consistency error (fraud often breaks the accounting identity)
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # How much of the sender's balance this transaction represents
    df["amount_to_balance_ratio"] = df["amount"] / df["oldbalanceOrg"].replace(0, np.nan)
    df["amount_to_balance_ratio"] = df["amount_to_balance_ratio"].fillna(0).clip(upper=10)

    # Time since this origin account's previous transaction (seconds)
    df["prev_time"] = df.groupby("nameOrig")["step_time"].shift(1)
    df["seconds_since_last_txn"] = (df["step_time"] - df["prev_time"]).dt.total_seconds()
    df["seconds_since_last_txn"] = df["seconds_since_last_txn"].fillna(1e6)  # first txn = "long gap"

    # Velocity: count of this origin's transactions in the preceding 5 / 60 minutes.
    # Use a stable row id for alignment since step_time can have duplicate
    # timestamps across different accounts, which breaks index-based reassignment.
    df["_row_id"] = np.arange(len(df))

    def _rolling_counts(g):
        g = g.set_index("step_time")
        c5 = g["amount"].rolling("5min").count()
        c60 = g["amount"].rolling("60min").count()
        return pd.DataFrame({"_row_id": g["_row_id"].values, "txn_count_5min": c5.values, "txn_count_60min": c60.values})

    rolled = df.groupby("nameOrig", group_keys=False).apply(_rolling_counts, include_groups=False)
    df = df.merge(rolled, on="_row_id", how="left").drop(columns=["_row_id"])

    # Has this origin -> dest pair been seen before?
    seen_pairs = set()
    is_new_dest = []
    for orig, dest in zip(df["nameOrig"], df["nameDest"]):
        pair = (orig, dest)
        is_new_dest.append(pair not in seen_pairs)
        seen_pairs.add(pair)
    df["is_new_dest"] = is_new_dest

    df = df.drop(columns=["prev_time"])
    return df


FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "amount_to_balance_ratio",
    "seconds_since_last_txn",
    "txn_count_5min", "txn_count_60min",
    "is_new_dest",
]
CATEGORICAL_COLUMNS = ["type"]


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv")
    df = add_features(df)
    df.to_csv("data/transactions_features.csv", index=False)
    print(f"Feature-engineered {len(df)} rows -> data/transactions_features.csv")
    print(df[FEATURE_COLUMNS + ["isFraud"]].describe().T[["mean", "std", "min", "max"]])
