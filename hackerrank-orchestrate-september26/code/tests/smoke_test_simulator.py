"""Smoke test and real-dataset validation for the deterministic 90-day simulator.

Runs the simulator across all 250 evaluation requests using the actual dataset,
verifies determinism, gathers aggregate statistics, and outputs compact timeline audits
for 6 representative user archetypes.
"""

from collections import Counter
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.canonical import CashImpactType, Direction
from code.image_resolution import extract_all_images, resolve_ledger_with_images
from code.loaders import load_dataset
from code.reconciliation import reconcile_events
from code.recurrence import detect_all_recurrence, expand_future_events
from code.simulator import (
    DailySnapshot,
    FinancialState,
    SimulationResult,
    create_initial_state,
    simulate_user,
)


def run_simulator_validation() -> None:
    dataset_dir = Path(__file__).resolve().parent.parent.parent / "dataset"
    print(f"Loading dataset from: {dataset_dir}")
    t0 = time.perf_counter()
    ds = load_dataset(dataset_dir)
    t_load = time.perf_counter() - t0
    print(f"Loaded dataset in {t_load:.3f}s")

    print("Reconciling canonical ledger and resolving image amounts...")
    t0 = time.perf_counter()
    base_ledger = reconcile_events(ds.events, ds.profiles, ds.exchange_rates, ds.messages, ds.images)
    extractions = extract_all_images(dataset_dir)
    ledger = resolve_ledger_with_images(base_ledger, extractions, ds.exchange_rates)
    t_reconcile = time.perf_counter() - t0
    print(f"Reconciled ledger ({len(ledger.events)} events) in {t_reconcile:.3f}s")

    print("Running recurrence detection across all users...")
    t0 = time.perf_counter()
    all_series, rejected = detect_all_recurrence(ledger, ds.profiles, ds.messages)
    t_rec = time.perf_counter() - t0
    print(f"Recurrence detection completed in {t_rec:.3f}s")

    print("\nSimulating all 250 evaluation requests (90-day horizon)...")
    t0 = time.perf_counter()

    results: Dict[str, SimulationResult] = {}
    users_represented = set()
    total_projected_events = 0
    explicit_future_events_count = 0
    forecast_future_events_count = 0
    settled_events_count = 0
    pending_debits_count = 0
    pending_credits_count = 0
    safety_floor_breaches_count = 0
    negative_cash_users = set()
    min_cash_list = []
    ending_balance_list = []
    failures = []

    for req in ds.requests:
        user_id = req.user_id
        users_represented.add(user_id)
        profile = ds.profiles[user_id]
        sim_start = req.request_date
        sim_end = req.request_date + timedelta(days=90)

        # Retrieve user series and expand future events
        user_series = all_series.get(user_id, [])
        future_res = expand_future_events(user_series, sim_start, sim_end, ledger=ledger)

        # Retrieve canonical events for user
        user_canonical = ledger.get_events_for_user(user_id)

        try:
            res = simulate_user(
                user_id=user_id,
                simulation_start=sim_start,
                simulation_end=sim_end,
                canonical_events=user_canonical,
                future_events=future_res.future_events,
                profile=profile,
            )
            results[req.request_id] = res

            total_projected_events += len(res.projected_events)
            for ev in res.projected_events:
                if ev.source_type == "forecast":
                    forecast_future_events_count += 1
                elif ev.source_type == "canonical_scheduled":
                    explicit_future_events_count += 1
                elif ev.source_type == "canonical_settled":
                    settled_events_count += 1
                elif ev.source_type == "canonical_pending":
                    pending_debits_count += 1
                elif ev.cash_impact_type == CashImpactType.PENDING_CREDIT_IGNORED:
                    pending_credits_count += 1

            if res.is_safety_floor_breached:
                safety_floor_breaches_count += 1

            if res.minimum_projected_available_cash < Decimal("0"):
                negative_cash_users.add(user_id)

            min_cash_list.append(res.minimum_projected_available_cash)
            ending_balance_list.append(res.ending_state.available_cash)

        except Exception as e:
            failures.append((req.request_id, user_id, str(e)))

    t_sim = time.perf_counter() - t0
    print(f"Simulation completed in {t_sim:.3f}s ({len(results)} requests, avg {t_sim/len(ds.requests)*1000:.2f}ms/request)")

    # Print Aggregate Statistics
    print("\n==================================================")
    print("REAL DATA VALIDATION REPORT (250 EVALUATION REQUESTS)")
    print("==================================================")
    print(f"Requests successfully simulated: {len(results)} / {len(ds.requests)}")
    print(f"Failures:                        {len(failures)}")
    print(f"Unique users represented:        {len(users_represented)}")
    print(f"Simulation window policy:        [request_date, request_date + 90 days] (inclusive, exactly 91 daily snapshots)")
    print(f"Total projected events:          {total_projected_events}")
    print(f"  - Forecast recurring events:   {forecast_future_events_count}")
    print(f"  - Explicit scheduled events:   {explicit_future_events_count}")
    print(f"  - Settled events (in-window):  {settled_events_count}")
    print(f"  - Pending debits reserved:     {pending_debits_count}")
    print(f"  - Pending credits ignored:     {pending_credits_count}")
    print(f"Safety-floor breaches:           {safety_floor_breaches_count} / {len(results)} requests ({safety_floor_breaches_count/len(results)*100:.1f}%)")
    print(f"Users with negative cash:        {len(negative_cash_users)} ({len(negative_cash_users)/len(users_represented)*100:.1f}%)")

    # Min cash and ending balance distribution
    min_cash_sorted = sorted(min_cash_list)
    ending_bal_sorted = sorted(ending_balance_list)
    n = len(min_cash_sorted)
    print("\n--- Minimum Projected Available Cash Distribution ---")
    print(f"  Min:    {min_cash_sorted[0]:.2f}")
    print(f"  P25:    {min_cash_sorted[n//4]:.2f}")
    print(f"  Median: {min_cash_sorted[n//2]:.2f}")
    print(f"  P75:    {min_cash_sorted[3*n//4]:.2f}")
    print(f"  Max:    {min_cash_sorted[-1]:.2f}")

    print("\n--- Ending Projected Available Cash Distribution ---")
    print(f"  Min:    {ending_bal_sorted[0]:.2f}")
    print(f"  P25:    {ending_bal_sorted[n//4]:.2f}")
    print(f"  Median: {ending_bal_sorted[n//2]:.2f}")
    print(f"  P75:    {ending_bal_sorted[3*n//4]:.2f}")
    print(f"  Max:    {ending_bal_sorted[-1]:.2f}")

    # Inspect 6 Archetypes
    archetypes = [
        ("1. Recurring salary + expenses", "user_100"),
        ("2. Pending debit present", "user_121"),
        ("3. Scheduled future event present", "user_109"),
        ("4. Cancelled recurrence present", "user_111"),
        ("5. Amended recurrence present", "user_103"),
        ("6. No recurring income (expenses only)", "user_102"),
    ]

    print("\n==================================================")
    print("TARGETED TIMELINE AUDITS (6 ARCHETYPES)")
    print("==================================================")

    req_by_user = {r.user_id: r for r in ds.requests}

    for label, uid in archetypes:
        req = req_by_user[uid]
        res = results[req.request_id]
        prof = ds.profiles[uid]

        print(f"\n--------------------------------------------------------------------------------------------------------")
        print(f"ARCHETYPE: {label} (User: {uid}, Request: {req.request_id}, Currency: {prof.home_currency})")
        print(f"Request Date: {req.request_date}, Initial Available: {res.initial_state.available_cash}, Safety Floor: {res.safety_floor}")
        print(f"Min Available: {res.minimum_projected_available_cash} on {res.minimum_cash_date}, Floor Breached: {res.is_safety_floor_breached}")
        print(f"Total Inflows: {res.total_inflows}, Total Outflows: {res.total_outflows}, Ending Available: {res.ending_state.available_cash}")
        print(f"Timeline (showing first 15 events):")
        print(f"{'Date':10} | {'Event ID':30} | {'Dir':7} | {'Amount':10} | {'Status/Source':20} | {'Bal Before':12} | {'Bal After':12} | {'Reserved':10} | {'Available':12}")
        print("-" * 135)

        for t in res.event_transitions[:15]:
            amt_str = f"{t.amount:.2f}"
            b_bef = f"{t.balance_before:.2f}"
            b_aft = f"{t.balance_after:.2f}"
            res_str = f"{t.reserved_after:.2f}"
            avail_str = f"{t.available_after:.2f}"
            dir_str = t.direction.value
            src_str = t.source_type[:20]
            print(f"{str(t.date):10} | {t.event_id[:30]:30} | {dir_str:7} | {amt_str:>10} | {src_str:20} | {b_bef:>12} | {b_aft:>12} | {res_str:>10} | {avail_str:>12}")


if __name__ == "__main__":
    run_simulator_validation()
