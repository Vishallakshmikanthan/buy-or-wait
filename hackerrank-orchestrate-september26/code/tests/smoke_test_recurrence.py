"""Real-dataset recurrence detection and future expansion smoke test.

Validates deterministic recurrence detection and future event expansion
across all 275 users and 250 evaluation requests in the dataset.
"""

import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.canonical import Direction
from code.image_resolution import extract_all_images, resolve_ledger_with_images
from code.loaders import load_dataset
from code.reconciliation import reconcile_events
from code.recurrence import detect_all_recurrence, expand_future_events


def run_recurrence_smoke_test():
    dataset_dir = Path(__file__).resolve().parent.parent.parent / "dataset"
    print(f"Loading real dataset from: {dataset_dir}")
    ds = load_dataset(dataset_dir)

    print("Reconciling events into canonical ledger and applying image resolution...")
    base_ledger = reconcile_events(ds.events, ds.profiles, ds.exchange_rates, ds.messages, ds.images)
    extractions = extract_all_images(dataset_dir)
    ledger = resolve_ledger_with_images(base_ledger, extractions, ds.exchange_rates)

    print(f"Total canonical events in ledger: {len(ledger.events)}")

    print("Running recurrence detection across all users...")
    all_series, all_rejected = detect_all_recurrence(
        ledger=ledger,
        profiles=ds.profiles,
        messages=ds.messages,
    )

    total_users = len(all_series)
    total_series = sum(len(s_list) for s_list in all_series.values())
    frequency_counts = Counter(s.frequency.value for s_list in all_series.values() for s in s_list)
    income_series = [s for s_list in all_series.values() for s in s_list if s.direction == Direction.INFLOW]
    expense_series = [s for s_list in all_series.values() for s in s_list if s.direction == Direction.OUTFLOW]
    cancelled_series = [s for s_list in all_series.values() for s in s_list if s.is_cancelled]

    # Calculate one-time vs recurring events in canonical ledger
    recurring_event_ids = set(
        evt_id
        for s_list in all_series.values()
        for s in s_list
        for evt_id in s.historical_event_ids
    )
    one_time_events = [e for e in ledger.events if e.event_id not in recurring_event_ids]

    print("\n=== RECURRENCE DETECTION SUMMARY ===")
    print(f"Users analyzed:                          {total_users}")
    print(f"Total recurrence series detected:        {total_series}")
    print(f"Series by frequency:                     {dict(frequency_counts)}")
    print(f"Recurring income series:                 {len(income_series)}")
    print(f"Recurring expense series:                {len(expense_series)}")
    print(f"Cancelled series (employer terminated):  {len(cancelled_series)}")
    print(f"Historical events in recurring series:   {len(recurring_event_ids)}")
    print(f"One-time / non-recurring events:         {len(one_time_events)}")
    print(f"Rejected candidate groups:               {len(all_rejected)}")

    # Test future expansion over evaluation requests
    print("\nExpanding future events across 250 evaluation requests (90-day horizon)...")
    total_expanded_future_events = 0
    total_suppressed_duplicates = 0
    suppression_reasons = Counter()

    for req in ds.requests:
        user_series = all_series.get(req.user_id, [])
        start_d = req.request_date
        end_d = req.request_date + timedelta(days=90)
        res = expand_future_events(user_series, start_d, end_d, ledger=ledger)
        total_expanded_future_events += len(res.future_events)
        total_suppressed_duplicates += len(res.suppressed_forecasts)
        for sf in res.suppressed_forecasts:
            suppression_reasons[sf.reason] += 1

    print(f"Total expanded future events (all reqs): {total_expanded_future_events}")
    print(f"Total suppressed duplicate forecasts:   {total_suppressed_duplicates}")
    print(f"Suppression breakdown:                  {dict(suppression_reasons)}")

    # Representative sample of detected series
    print("\n=== REPRESENTATIVE DETECTED SERIES SAMPLE ===")
    print(f"{'User':<9} | {'Category':<15} | {'Dir':<7} | {'Freq':<9} | {'Hist':<4} | {'Amount':<12} | {'Anchor Date':<11} | {'Rationale'}")
    print("-" * 115)
    sample_users = ["user_01", "user_02", "user_06", "user_11", "user_12", "user_16", "user_26", "user_73"]
    for uid in sample_users:
        user_s = all_series.get(uid, [])
        for s in user_s[:2]:  # Show top 2 per sample user
            print(
                f"{s.user_id:<9} | {s.category:<15} | {s.direction.value:<7} | {s.frequency.value:<9} | "
                f"{s.historical_count:<4} | {str(s.forecast_amount):<12} | {str(s.anchor_date):<11} | "
                f"{s.evidence_rationale[:40]}..."
            )

    print("\n=== SMOKE TEST COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_recurrence_smoke_test()
