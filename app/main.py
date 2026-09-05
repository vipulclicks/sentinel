"""
FastAPI service exposing:
  POST /score       -> score a single transaction, with LLM explanation if flagged
  POST /score_batch  -> score a batch (list) of transactions
  GET  /metrics      -> return the saved model's held-out test metrics

Run: uvicorn app.main:app --reload --port 8000
"""
import os
import sys
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features import add_features, FEATURE_COLUMNS, CATEGORICAL_COLUMNS  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
FLAG_THRESHOLD = 0.5

app = FastAPI(title="Sentinel — Fraud Detection Service (defense-only)")
model = joblib.load(MODEL_PATH)

# --- Defense-only safeguard -------------------------------------------------
# This service must never expose an endpoint that moves money, blocks a
# transaction, freezes an account, or files a chargeback/refund on its own.
# It only scores and explains; a human analyst makes every real decision.
# The check below runs at import time (i.e. at server startup) and refuses
# to boot if a mutating HTTP method is ever added to this app, so this
# constraint can't silently drift as the codebase grows.
ALLOWED_METHODS = {"GET", "HEAD", "POST"}  # HEAD is auto-added by FastAPI for GET routes; POST here is used only for read-style scoring calls
FORBIDDEN_ACTION_KEYWORDS = ["block", "freeze", "refund", "reverse", "chargeback", "transfer", "payout", "cancel"]


def _assert_defense_only():
    for route in app.routes:
        methods = getattr(route, "methods", set()) or set()
        if methods - ALLOWED_METHODS:
            raise RuntimeError(
                f"Defense-only violation: route {route.path} exposes method(s) "
                f"{methods - ALLOWED_METHODS}, which this project must never expose."
            )
        path_lower = getattr(route, "path", "").lower()
        name_lower = (getattr(route, "name", "") or "").lower()
        for kw in FORBIDDEN_ACTION_KEYWORDS:
            if kw in path_lower or kw in name_lower:
                raise RuntimeError(
                    f"Defense-only violation: route '{route.path}' looks like it might "
                    f"perform a money-moving/account action ('{kw}'). This project only "
                    f"scores and explains — it never acts."
                )


DEFENSE_ONLY_POLICY = (
    "This service flags and explains suspicious transactions for human review. "
    "It never blocks, reverses, refunds, or moves money, and never freezes an account."
)
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    step_time: str
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


def explain_flag(row: dict, score: float) -> str:
    """
    Calls Claude API to turn feature values into a human-readable reason.
    Falls back to a templated explanation if no API key is configured, so the
    demo still works offline.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    reasons = []
    if row.get("txn_count_5min", 1) >= 3:
        reasons.append(f"{int(row['txn_count_5min'])} transactions from this account in the last 5 minutes")
    if row.get("amount_to_balance_ratio", 0) >= 0.8:
        reasons.append(f"transaction drains {row['amount_to_balance_ratio']*100:.0f}% of account balance")
    if abs(row.get("errorBalanceOrig", 0)) > 1:
        reasons.append("balance accounting mismatch detected")
    if row.get("is_new_dest", True):
        reasons.append("first-ever transfer to this destination account")
    template_reason = "; ".join(reasons) if reasons else "unusual transaction pattern vs. account history"

    if not api_key:
        return f"[rule-based, no LLM key set] Flagged (score={score:.2f}): {template_reason}."

    try:
        import urllib.request
        payload = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 150,
            "messages": [{
                "role": "user",
                "content": (
                    "You are a fraud-ops assistant. In one short sentence, explain to a "
                    "risk analyst why this transaction was flagged (defense-only, no action "
                    f"taken automatically). Fraud score: {score:.2f}. Signals: {template_reason}. "
                    f"Transaction: {row}"
                ),
            }],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except Exception as e:
        return f"[LLM call failed, fallback] Flagged (score={score:.2f}): {template_reason}. ({e})"


def _score_df(df: pd.DataFrame):
    try:
        feat_df = add_features(df)
    except Exception as e:
        # Malformed input (e.g. an unparseable step_time, or a value that
        # breaks a numeric conversion) should surface as a clean client
        # error, not crash the process with a 500.
        raise HTTPException(
            status_code=400,
            detail=f"Could not process transaction data: {e}",
        )
    X = feat_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    scores = model.predict_proba(X)[:, 1]
    feat_df["fraud_score"] = scores
    feat_df["flagged"] = scores >= FLAG_THRESHOLD
    return feat_df


@app.post("/score")
def score_transaction(txn: Transaction):
    df = pd.DataFrame([txn.dict()])
    scored = _score_df(df)
    row = scored.iloc[0]
    result = {
        "fraud_score": float(row["fraud_score"]),
        "flagged": bool(row["flagged"]),
        "action_taken": "none — flag only, for human review",
        "policy": DEFENSE_ONLY_POLICY,
    }
    if result["flagged"]:
        result["explanation"] = explain_flag(row.to_dict(), result["fraud_score"])
    return result


@app.post("/score_batch")
def score_batch(txns: List[Transaction]):
    if not txns:
        return {
            "total_scored": 0,
            "flagged_count": 0,
            "flagged_rate": 0.0,
            "action_taken": "none — flag only, for human review",
            "policy": DEFENSE_ONLY_POLICY,
            "flagged_transactions": [],
        }
    df = pd.DataFrame([t.dict() for t in txns])
    scored = _score_df(df)
    flagged = scored[scored["flagged"]]
    results = []
    for _, row in flagged.iterrows():
        results.append({
            "nameOrig": row["nameOrig"],
            "nameDest": row["nameDest"],
            "amount": row["amount"],
            "fraud_score": float(row["fraud_score"]),
            "explanation": explain_flag(row.to_dict(), row["fraud_score"]),
        })
    return {
        "total_scored": len(scored),
        "flagged_count": len(flagged),
        "flagged_rate": len(flagged) / max(len(scored), 1),
        "action_taken": "none — flag only, for human review",
        "policy": DEFENSE_ONLY_POLICY,
        "flagged_transactions": results,
    }


@app.get("/metrics")
def get_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.get("/policy")
def get_policy():
    """Explicit, machine-checkable statement of this service's defense-only scope."""
    return {
        "policy": DEFENSE_ONLY_POLICY,
        "routes": [
            {"path": r.path, "methods": sorted(getattr(r, "methods", set()) or set())}
            for r in app.routes if hasattr(r, "path")
        ],
    }


# Run the safeguard check now that all routes above are registered. This
# executes at import time (i.e. when uvicorn loads this module), so the
# server refuses to start at all if the constraint is ever violated.
_assert_defense_only()
