"""
Fetches real Razorpay TEST-MODE payments via the official API and maps them
into this project's transaction schema, so they can be scored by the fraud
detector alongside (or instead of) synthetic data, to be scored by Sentinel.

Requires environment variables:
    RAZORPAY_KEY_ID
    RAZORPAY_KEY_SECRET
(Generate these in Test Mode from the Razorpay Dashboard: Account & Settings
-> API Keys -> Generate Key. Never commit these to source control.)

Honest limitation, stated up front:
    Razorpay is a payment GATEWAY, not a bank ledger. It has no concept of
    a customer's account balance before/after a payment (oldbalanceOrg,
    newbalanceOrig, etc.) the way our synthetic PaySim-style data does.
    Those fields are therefore NOT real for this data -- we fill them with
    a documented placeholder (see BALANCE_PLACEHOLDER_NOTE below) so the
    existing feature pipeline still runs, but any balance-derived features
    (errorBalanceOrig, amount_to_balance_ratio) are not meaningful on this
    data and should be caveated in a demo, not presented as equivalent to
    the synthetic results.

Usage:
    export RAZORPAY_KEY_ID="rzp_test_xxxxx"
    export RAZORPAY_KEY_SECRET="xxxxx"
    python src/fetch_razorpay_payments.py --count 50 --out data/razorpay_real_transactions.csv
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd

BALANCE_PLACEHOLDER_NOTE = (
    "oldbalance/newbalance fields are NOT real -- Razorpay's Payments API "
    "does not expose customer bank balances. Placeholders assume the "
    "payment amount was the customer's full available balance, purely so "
    "the existing feature pipeline can run. Do not present balance-derived "
    "features (errorBalanceOrig, amount_to_balance_ratio) as meaningful "
    "signal on this data in a demo -- say so explicitly."
)


def get_client():
    try:
        import razorpay
    except ImportError:
        print("Missing dependency. Run: pip install razorpay --break-system-packages")
        sys.exit(1)

    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        print(
            "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set.\n"
            "Generate test-mode keys at https://dashboard.razorpay.com "
            "(Account & Settings -> API Keys -> Generate Key, Test Mode),\n"
            "then run:\n"
            "  export RAZORPAY_KEY_ID=\"rzp_test_xxxxx\"\n"
            "  export RAZORPAY_KEY_SECRET=\"xxxxx\""
        )
        sys.exit(1)
    return razorpay.Client(auth=(key_id, key_secret))


def map_payment_to_schema(payment: dict) -> dict:
    """
    Maps one Razorpay payment object to our PaySim-style schema.
    See module docstring for the balance-field caveat.
    """
    amount_inr = payment["amount"] / 100.0  # Razorpay amounts are in paise
    created_at = datetime.fromtimestamp(payment["created_at"], tz=timezone.utc)

    # Razorpay's `method` (card/upi/netbanking/wallet/emi) doesn't map
    # cleanly to PaySim's PAYMENT/TRANSFER/CASH_OUT/CASH_IN/DEBIT types.
    # We treat every captured payment as a "PAYMENT" (money coming into
    # the merchant from a customer) since that's what a Razorpay payment
    # actually is -- we don't rename it to something PaySim-shaped that
    # would misrepresent what happened.
    txn_type = "PAYMENT"

    origin_id = f"razorpay_customer_{payment.get('contact', payment.get('email', payment['id']))}"
    dest_id = "merchant_account"  # single fixed merchant receiving all test payments

    return {
        "step_time": created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "type": txn_type,
        "amount": amount_inr,
        "nameOrig": origin_id,
        "oldbalanceOrg": amount_inr,   # placeholder -- see BALANCE_PLACEHOLDER_NOTE
        "newbalanceOrig": 0.0,          # placeholder -- see BALANCE_PLACEHOLDER_NOTE
        "nameDest": dest_id,
        "oldbalanceDest": 0.0,          # placeholder -- see BALANCE_PLACEHOLDER_NOTE
        "newbalanceDest": amount_inr,  # placeholder -- see BALANCE_PLACEHOLDER_NOTE
        "razorpay_payment_id": payment["id"],
        "razorpay_method": payment.get("method", "unknown"),
        "razorpay_status": payment.get("status", "unknown"),
    }


def main(count: int, out_path: str):
    client = get_client()
    print(f"Fetching up to {count} test-mode payments from Razorpay...")

    try:
        result = client.payment.all({"count": min(count, 100)})
    except Exception as e:
        print(f"Razorpay API call failed: {e}")
        print("Check that your keys are TEST MODE keys (rzp_test_...) and are still valid.")
        sys.exit(1)

    payments = result.get("items", [])
    if not payments:
        print(
            "\nNo payments found in this test account yet.\n"
            "Razorpay's API can create Orders, but a captured Payment "
            "requires an actual checkout with a test card -- it can't be "
            "created by API call alone.\n\n"
            "To generate some real test payments:\n"
            "  1. Open razorpay_test_checkout.html in a browser (put your "
            "test Key Id in it first).\n"
            "  2. Click Pay, use test card 4111 1111 1111 1111, any future "
            "expiry, any CVV.\n"
            "  3. Repeat 5-10 times with different amounts.\n"
            "  4. Re-run this script.\n"
        )
        sys.exit(0)

    rows = [map_payment_to_schema(p) for p in payments]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} real test-mode payments to {out_path}")
    print(f"\nCAVEAT (include this in any demo using this file):\n  {BALANCE_PLACEHOLDER_NOTE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--out", type=str, default="data/razorpay_real_transactions.csv")
    args = parser.parse_args()
    main(args.count, args.out)
