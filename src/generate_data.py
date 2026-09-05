"""
Generates a synthetic mobile-money / payments dataset in the PaySim schema,
with realistic fraud patterns injected (account takeover bursts, cash-out
draining, mule-account fan-out). Designed to mimic Razorpay-style transaction
fields so it's easy to swap in real test-mode data later.

Usage: python src/generate_data.py --n 50000 --out data/transactions.csv
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

TXN_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
TXN_TYPE_WEIGHTS = [0.35, 0.20, 0.25, 0.15, 0.05]


def make_account_pool(n_accounts):
    ids = [f"acct_{i:07d}" for i in range(n_accounts)]
    balances = RNG.gamma(shape=2.0, scale=5000, size=n_accounts)
    return dict(zip(ids, balances))


def gen_legit_transactions(n, accounts):
    acct_ids = list(accounts.keys())
    rows = []
    start = datetime(2026, 1, 1)
    for i in range(n):
        origin = RNG.choice(acct_ids)
        dest = RNG.choice(acct_ids)
        while dest == origin:
            dest = RNG.choice(acct_ids)
        txn_type = RNG.choice(TXN_TYPES, p=TXN_TYPE_WEIGHTS)
        old_bal_org = max(accounts[origin], 0)
        amount = float(np.clip(RNG.gamma(2.0, old_bal_org / 6 + 50), 1, old_bal_org + 500))
        new_bal_org = max(old_bal_org - amount, 0)
        old_bal_dest = accounts.get(dest, 0)
        new_bal_dest = old_bal_dest + amount
        accounts[origin] = new_bal_org
        accounts[dest] = new_bal_dest
        ts = start + timedelta(minutes=int(RNG.integers(0, 60 * 24 * 30)))
        rows.append({
            "step_time": ts,
            "type": txn_type,
            "amount": round(amount, 2),
            "nameOrig": origin,
            "oldbalanceOrg": round(old_bal_org, 2),
            "newbalanceOrig": round(new_bal_org, 2),
            "nameDest": dest,
            "oldbalanceDest": round(old_bal_dest, 2),
            "newbalanceDest": round(new_bal_dest, 2),
            "isFraud": 0,
        })
    return rows


def gen_fraud_transactions(n, accounts):
    """Injects three fraud archetypes: takeover-drain, rapid mule fan-out, tiny-probe-then-big-hit."""
    acct_ids = list(accounts.keys())
    rows = []
    start = datetime(2026, 1, 1)
    n_each = n // 3

    # 1. Account takeover -> full balance drain via CASH_OUT
    for _ in range(n_each):
        victim = RNG.choice(acct_ids)
        mule = f"mule_{RNG.integers(0, 500):04d}"
        old_bal = max(accounts.get(victim, 1000), 100)
        amount = old_bal * RNG.uniform(0.85, 1.0)
        ts = start + timedelta(minutes=int(RNG.integers(0, 60 * 24 * 30)))
        rows.append({
            "step_time": ts, "type": "CASH_OUT", "amount": round(amount, 2),
            "nameOrig": victim, "oldbalanceOrg": round(old_bal, 2),
            "newbalanceOrig": round(old_bal - amount, 2),
            "nameDest": mule, "oldbalanceDest": 0.0,
            "newbalanceDest": round(amount, 2), "isFraud": 1,
        })

    # 2. Rapid mule fan-out: same origin, many small transfers within minutes
    for _ in range(n_each):
        origin = RNG.choice(acct_ids)
        base_ts = start + timedelta(minutes=int(RNG.integers(0, 60 * 24 * 30)))
        burst_size = RNG.integers(4, 9)
        old_bal = max(accounts.get(origin, 2000), 500)
        for k in range(burst_size):
            dest = f"mule_{RNG.integers(0, 500):04d}"
            amount = old_bal / burst_size * RNG.uniform(0.9, 1.1)
            rows.append({
                "step_time": base_ts + timedelta(seconds=int(k * RNG.integers(5, 40))),
                "type": "TRANSFER", "amount": round(amount, 2),
                "nameOrig": origin, "oldbalanceOrg": round(old_bal, 2),
                "newbalanceOrig": round(max(old_bal - amount, 0), 2),
                "nameDest": dest, "oldbalanceDest": 0.0,
                "newbalanceDest": round(amount, 2), "isFraud": 1,
            })
            old_bal = max(old_bal - amount, 0)

    # 3. Probe-then-strike: tiny test transaction, then a large one seconds later
    for _ in range(n_each):
        origin = RNG.choice(acct_ids)
        dest = f"mule_{RNG.integers(0, 500):04d}"
        base_ts = start + timedelta(minutes=int(RNG.integers(0, 60 * 24 * 30)))
        old_bal = max(accounts.get(origin, 2000), 500)
        probe_amt = round(RNG.uniform(1, 5), 2)
        rows.append({
            "step_time": base_ts, "type": "PAYMENT", "amount": probe_amt,
            "nameOrig": origin, "oldbalanceOrg": round(old_bal, 2),
            "newbalanceOrig": round(old_bal - probe_amt, 2),
            "nameDest": dest, "oldbalanceDest": 0.0,
            "newbalanceDest": probe_amt, "isFraud": 1,
        })
        strike_amt = old_bal * RNG.uniform(0.7, 0.95)
        rows.append({
            "step_time": base_ts + timedelta(seconds=int(RNG.integers(20, 90))),
            "type": "CASH_OUT", "amount": round(strike_amt, 2),
            "nameOrig": origin, "oldbalanceOrg": round(old_bal - probe_amt, 2),
            "newbalanceOrig": round(old_bal - probe_amt - strike_amt, 2),
            "nameDest": dest, "oldbalanceDest": probe_amt,
            "newbalanceDest": round(probe_amt + strike_amt, 2), "isFraud": 1,
        })
    return rows


def main(n_legit, n_fraud, out_path):
    accounts = make_account_pool(n_accounts=max(2000, n_legit // 20))
    legit = gen_legit_transactions(n_legit, accounts)
    fraud = gen_fraud_transactions(n_fraud, accounts)
    df = pd.DataFrame(legit + fraud)
    df = df.sort_values("step_time").reset_index(drop=True)
    df["step_time"] = pd.to_datetime(df["step_time"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({df['isFraud'].sum()} fraud, "
          f"{df['isFraud'].mean()*100:.3f}% fraud rate) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000, help="number of legit transactions")
    parser.add_argument("--fraud_n", type=int, default=450, help="number of fraud transactions (approx, rounds to multiple of 3)")
    parser.add_argument("--out", type=str, default="data/transactions.csv")
    args = parser.parse_args()
    main(args.n, args.fraud_n, args.out)
