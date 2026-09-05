"""
Streamlit dashboard for Sentinel (defense-only fraud detector).
Run: streamlit run app/dashboard.py
"""
import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features import add_features, FEATURE_COLUMNS, CATEGORICAL_COLUMNS  # noqa: E402
from main import explain_flag  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
PR_CURVE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pr_curve.png")

st.set_page_config(page_title="Sentinel", layout="wide")
st.title("🛡️ Sentinel")
st.caption("Defense-only: flags and explains suspicious transactions. Never auto-blocks or moves money.")

model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    metrics = json.load(f)

# --- Held-out test metrics ---
st.header("Held-out test set performance")
selected = metrics["selected"]
sel_metrics = metrics["random_forest"] if "Random Forest" in selected else metrics["logistic_regression"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{sel_metrics['precision']*100:.1f}%")
col2.metric("Recall", f"{sel_metrics['recall']*100:.1f}%")
col3.metric("ROC-AUC", f"{sel_metrics['roc_auc']:.3f}")
col4.metric("Est. total cost", f"₹{sel_metrics['total_cost']:,}")

st.write(
    f"Selected model: **{selected}** — chosen for lowest estimated total cost, "
    f"not just highest accuracy (FP cost ₹{metrics['assumptions']['cost_per_false_positive_inr']}/case, "
    f"FN cost ₹{metrics['assumptions']['cost_per_false_negative_inr']}/case)."
)
st.write(
    f"Confusion matrix on held-out test set: TP={sel_metrics['tp']}, FP={sel_metrics['fp']}, "
    f"FN={sel_metrics['fn']}, TN={sel_metrics['tn']}"
)

if os.path.exists(PR_CURVE_PATH):
    st.image(PR_CURVE_PATH, caption="Precision-Recall curve: Logistic Regression vs Random Forest", width=500)

with st.expander("Compare both models"):
    st.json(metrics)

st.divider()

# --- Batch scoring ---
st.header("Score a batch of transactions")
uploaded = st.file_uploader("Upload a CSV (PaySim-style schema)", type="csv")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read this CSV: {e}")
        st.stop()

    if len(df) == 0:
        st.warning("This CSV has no rows — upload a file with at least one transaction.")
        st.stop()

    required_cols = {"step_time", "type", "amount", "nameOrig", "oldbalanceOrg",
                      "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"This CSV is missing required column(s): {sorted(missing)}. "
                  f"Expected schema: {sorted(required_cols)}.")
        st.stop()

    try:
        feat_df = add_features(df)
    except Exception as e:
        st.error(f"Could not process this file — check for malformed dates or values: {e}")
        st.stop()

    X = feat_df[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    scores = model.predict_proba(X)[:, 1]
    feat_df["fraud_score"] = scores
    feat_df["flagged"] = scores >= 0.5

    flagged = feat_df[feat_df["flagged"]].sort_values("fraud_score", ascending=False)
    st.write(f"**{len(flagged)} of {len(feat_df)} transactions flagged** ({len(flagged)/len(feat_df)*100:.2f}%)")

    if "isFraud" in feat_df.columns:
        tp = ((feat_df["flagged"]) & (feat_df["isFraud"] == 1)).sum()
        fp = ((feat_df["flagged"]) & (feat_df["isFraud"] == 0)).sum()
        fn = ((~feat_df["flagged"]) & (feat_df["isFraud"] == 1)).sum()
        st.info(f"Against ground truth in this file: TP={tp}, FP={fp}, FN={fn} (honest exception count, not cherry-picked)")

    st.dataframe(
        flagged[["step_time", "type", "amount", "nameOrig", "nameDest", "fraud_score"]].head(50),
        use_container_width=True,
    )

    if st.button("Generate explanations for top 10 flagged"):
        with st.spinner("Explaining flagged transactions..."):
            for _, row in flagged.head(10).iterrows():
                reason = explain_flag(row.to_dict(), row["fraud_score"])
                st.write(f"**{row['nameOrig']} → {row['nameDest']}** (₹{row['amount']:.2f}, score={row['fraud_score']:.2f}): {reason}")

st.divider()

# --- Single transaction scoring ---
st.header("Score a single transaction (live demo)")
c1, c2, c3 = st.columns(3)
with c1:
    txn_type = st.selectbox("Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
    amount = st.number_input("Amount", value=1000.0)
with c2:
    old_bal_org = st.number_input("Sender old balance", value=5000.0)
    new_bal_org = st.number_input("Sender new balance", value=4000.0)
with c3:
    old_bal_dest = st.number_input("Receiver old balance", value=0.0)
    new_bal_dest = st.number_input("Receiver new balance", value=1000.0)

if st.button("Score this transaction"):
    single = pd.DataFrame([{
        "step_time": "2026-01-01T00:00:00", "type": txn_type, "amount": amount,
        "nameOrig": "acct_demo_orig", "oldbalanceOrg": old_bal_org, "newbalanceOrig": new_bal_org,
        "nameDest": "acct_demo_dest", "oldbalanceDest": old_bal_dest, "newbalanceDest": new_bal_dest,
    }])
    feat = add_features(single)
    X = feat[FEATURE_COLUMNS + CATEGORICAL_COLUMNS]
    score = model.predict_proba(X)[:, 1][0]
    flagged = score >= 0.5
    if flagged:
        st.error(f"🚩 FLAGGED — fraud score {score:.2f}")
        st.write(explain_flag(feat.iloc[0].to_dict(), score))
    else:
        st.success(f"✅ Clean — fraud score {score:.2f}")
