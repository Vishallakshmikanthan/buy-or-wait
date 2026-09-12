"""Smoke test and real-dataset validation for safe-to-pay and earliest date determination.

Runs across all 250 evaluation requests in dataset/requests.csv, computes metrics,
and performs targeted audits across 8 representative user archetypes.
"""

from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import sys
import time

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.image_resolution import extract_all_images, resolve_ledger_with_images
from code.loaders import load_dataset
from code.reconciliation import reconcile_events
from code.recurrence import detect_all_recurrence, expand_future_events
from code.safe_to_pay import SafeToPayCertificate, evaluate_request_safe_to_pay
from code.simulator import simulate_user


def run_safe_to_pay_smoke_test() -> None:
    dataset_dir = Path(__file__).resolve().parent.parent.parent / "dataset"
    print(f"Loading dataset from: {dataset_dir}")
    t0 = time.perf_counter()
    ds = load_dataset(dataset_dir)
    print(f"Loaded dataset in {time.perf_counter() - t0:.3f}s")

    print("Reconciling canonical ledger and resolving images...")
    t0 = time.perf_counter()
    base_ledger = reconcile_events(ds.events, ds.profiles, ds.exchange_rates, ds.messages, ds.images)
    extractions = extract_all_images(dataset_dir)
    ledger = resolve_ledger_with_images(base_ledger, extractions, ds.exchange_rates)
    print(f"Reconciled ledger in {time.perf_counter() - t0:.3f}s")

    print("Detecting recurrence patterns...")
    t0 = time.perf_counter()
    all_series, _ = detect_all_recurrence(ledger, ds.profiles, ds.messages)
    print(f"Recurrence detection in {time.perf_counter() - t0:.3f}s")

    print("\nEvaluating safe-to-pay across all 250 evaluation requests...")
    t0 = time.perf_counter()

    certificates: Dict[str, SafeToPayCertificate] = {}
    failures = []

    zero_safe_count = 0
    full_safe_count = 0
    partial_safe_count = 0

    total_requested_amount_by_curr = Counter()
    total_safe_amount_by_curr = Counter()

    safe_ratios = []

    safe_today_count = 0
    safe_later_count = 0
    never_safe_count = 0

    earliest_date_offsets = []

    for req in ds.requests:
        uid = req.user_id
        prof = ds.profiles[uid]
        start_d = req.request_date
        end_d = req.request_date + timedelta(days=90)
        future_res = expand_future_events(all_series[uid], start_d, end_d, ledger=ledger)
        user_canonical = ledger.get_events_for_user(uid)

        try:
            baseline = simulate_user(
                user_id=uid,
                simulation_start=start_d,
                simulation_end=end_d,
                canonical_events=user_canonical,
                future_events=future_res.future_events,
                profile=prof,
            )

            cert = evaluate_request_safe_to_pay(
                request=req,
                baseline_simulation=baseline,
                canonical_events=user_canonical,
                future_events=future_res.future_events,
                profile=prof,
            )
            certificates[req.request_id] = cert

            # Tally metrics
            if cert.amount_safe_to_pay == Decimal("0"):
                zero_safe_count += 1
            elif cert.amount_safe_to_pay == cert.requested_amount:
                full_safe_count += 1
            else:
                partial_safe_count += 1

            total_requested_amount_by_curr[cert.currency] += cert.requested_amount
            total_safe_amount_by_curr[cert.currency] += cert.amount_safe_to_pay

            ratio = float(cert.amount_safe_to_pay / cert.requested_amount) if cert.requested_amount > 0 else 0.0
            safe_ratios.append(ratio)

            if cert.is_full_payment_safe_today:
                safe_today_count += 1
                earliest_date_offsets.append(0)
            elif cert.earliest_date_for_full_payment is not None:
                safe_later_count += 1
                offset = (cert.earliest_date_for_full_payment - req.request_date).days
                earliest_date_offsets.append(offset)
            else:
                never_safe_count += 1

        except Exception as e:
            failures.append((req.request_id, uid, str(e)))

    total_eval_time = time.perf_counter() - t0
    print(f"Evaluated all {len(certificates)} requests in {total_eval_time:.3f}s ({total_eval_time/len(ds.requests)*1000:.2f}ms/request)")

    # Print Summary Report
    print("\n==================================================")
    print("SAFE-TO-PAY REAL-DATA VALIDATION REPORT")
    print("==================================================")
    print(f"Requests evaluated:               {len(certificates)} / {len(ds.requests)}")
    print(f"Failures:                         {len(failures)}")
    print(f"Amount safe to pay = 0:           {zero_safe_count} ({zero_safe_count/len(certificates)*100:.1f}%)")
    print(f"Amount safe to pay = requested:   {full_safe_count} ({full_safe_count/len(certificates)*100:.1f}%)")
    print(f"Partial-safe amounts:             {partial_safe_count} ({partial_safe_count/len(certificates)*100:.1f}%)")

    safe_ratios_sorted = sorted(safe_ratios)
    n = len(safe_ratios_sorted)
    print(f"\nSafe / Requested Ratio Summary:")
    print(f"  Min ratio:     {safe_ratios_sorted[0]:.4f}")
    print(f"  P25 ratio:     {safe_ratios_sorted[n//4]:.4f}")
    print(f"  Median ratio:  {safe_ratios_sorted[n//2]:.4f}")
    print(f"  P75 ratio:     {safe_ratios_sorted[3*n//4]:.4f}")
    print(f"  Max ratio:     {safe_ratios_sorted[-1]:.4f}")

    print(f"\nEarliest Full Payment Timing Summary:")
    print(f"  Full payment safe today:        {safe_today_count} ({safe_today_count/len(certificates)*100:.1f}%)")
    print(f"  Full payment safe later:        {safe_later_count} ({safe_later_count/len(certificates)*100:.1f}%)")
    print(f"  Never safe within horizon:      {never_safe_count} ({never_safe_count/len(certificates)*100:.1f}%)")

    later_offsets = [o for o in earliest_date_offsets if o > 0]
    if later_offsets:
        later_sorted = sorted(later_offsets)
        nl = len(later_sorted)
        print(f"  Days to full payment (when later): min={later_sorted[0]}d, median={later_sorted[nl//2]}d, max={later_sorted[-1]}d")

    print("\nTotal Amounts by Currency:")
    for curr in sorted(total_requested_amount_by_curr.keys()):
        req_tot = total_requested_amount_by_curr[curr]
        safe_tot = total_safe_amount_by_curr[curr]
        pct = (safe_tot / req_tot * 100) if req_tot > 0 else Decimal("0")
        print(f"  {curr:4}: Requested = {req_tot:>14.2f} | Safe = {safe_tot:>14.2f} ({pct:>5.1f}%)")

    # 8 Targeted Representative Audits
    archetypes = [
        ("1. Comfortably affordable", "request_26"),
        ("2. Near safety floor", "request_30"),
        ("3. Future salary required", "request_31"),
        ("4. Large purchase relative to cash", "request_36"),
        ("5. Pending debit present", "request_41"),
        ("6. No recurring income", "request_29"),
        ("7. Negative projected baseline", "request_32"),
        ("8. Full payment never safe in horizon", "request_40"),
    ]

    print("\n==================================================")
    print("TARGETED AUDITS (8 REPRESENTATIVE REQUESTS)")
    print("==================================================")
    req_map = {r.request_id: r for r in ds.requests}

    for label, req_id in archetypes:
        cert = certificates[req_id]
        req = req_map[req_id]
        prof = ds.profiles[req.user_id]
        ratio = (cert.amount_safe_to_pay / cert.requested_amount) if cert.requested_amount > 0 else Decimal("0")
        earliest_str = str(cert.earliest_date_for_full_payment) if cert.earliest_date_for_full_payment else "None (Never safe)"

        print(f"\n--- {label} ---")
        print(f"Request:           {cert.request_id} ({cert.user_id}, Date: {cert.request_date}, Currency: {cert.currency})")
        print(f"Requested Amount:  {cert.requested_amount:,.2f}")
        print(f"Available Cash:    {prof.current_available_balance:,.2f} | Safety Floor: {cert.safety_floor:,.2f}")
        print(f"Amount Safe to Pay: {cert.amount_safe_to_pay:,.2f} (Safe Ratio: {ratio*100:.1f}%)")
        print(f"Limiting Date:     {cert.limiting_date} (Min Cash After Purchase: {cert.minimum_available_cash_after_purchase:,.2f})")
        print(f"Safety Margin:     {cert.safety_floor_margin:,.2f}")
        print(f"Earliest Full Date:{earliest_str}")
        print(f"Decision Reason:   {cert.reason_if_unsafe or 'Full requested amount is safely affordable today without breaching the safety floor.'}")


if __name__ == "__main__":
    run_safe_to_pay_smoke_test()
