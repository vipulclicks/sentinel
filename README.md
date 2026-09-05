# Sentinel — AI Fraud Detection System (Track 02: AI Risk Manager)

**Sentinel** is a **defense-only** transaction fraud detection system: flags suspicious payments,
scores them, and explains *why* — it never auto-blocks or moves money.

## The bar this project targets
> "Honest metrics including false-positive cost. Strictly defense-only:
> anything offense-capable is disqualified."

- Model selection is made on **estimated total cost** (FP cost + FN cost),
  not just accuracy or F1 — see `models/metrics.json`.
- Every flagged transaction gets a human-readable explanation, so a risk
  analyst (not the model) makes the final call.
- No transaction is ever blocked, reversed, or auto-actioned by this system.

## Architecture
```
data/generate_data.py   -> synthetic PaySim-style dataset with 3 fraud archetypes
src/features.py         -> velocity, balance-error, and ratio feature engineering
src/train.py            -> trains Logistic Regression + Random Forest, picks lower-cost model
app/main.py              -> FastAPI scoring endpoint (+ Claude API explanation layer)
app/dashboard.py          -> Streamlit UI: batch upload, live scoring, metrics
```

## Setup
```bash
pip install -r requirements.txt   # or see individual installs below
python src/generate_data.py --n 50000 --fraud_n 450 --out data/transactions.csv
python src/features.py
cd .. && PYTHONPATH=src python src/train.py
```

## Run the API
```bash
uvicorn app.main:app --reload --port 8000
# POST /score        - single transaction
# POST /score_batch   - list of transactions
# GET  /metrics        - held-out test metrics
```

## Run the dashboard
```bash
streamlit run app/dashboard.py
```

## Enabling the LLM explanation layer
Set `ANTHROPIC_API_KEY` in your environment. Without it, the system falls
back to rule-based explanations (still works, just less natural language).

## Current results (held-out test set, 25% split, stratified)

| Model | Precision | Recall | F1 | ROC-AUC | Est. Cost |
|---|---|---|---|---|---|
| Logistic Regression | 36.9% | 100% | 0.54 | 0.996 | ₹28,850 |
| **Random Forest (selected)** | **93.1%** | **99.1%** | **0.96** | **0.9997** | **₹10,250** |

Cost assumptions (stated explicitly, editable in `src/train.py`):
- False positive cost: ₹50/case (support ticket + friction)
- False negative cost: ₹3,000/case (average unrecovered fraud loss)

## Honest limitations / exceptions
- Trained on synthetic data with 3 injected fraud archetypes (account
  takeover drain, rapid mule fan-out, probe-then-strike). Real fraud
  patterns may differ — this is a proof-of-concept, not production-ready.
- Velocity features depend on having transaction history per account;
  brand-new accounts have a cold-start blind spot.
- Cost assumptions are illustrative placeholders, not sourced from real
  Razorpay loss/support data.
