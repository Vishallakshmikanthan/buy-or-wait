"""Real-data validation and audit for UserFinancialState across all 250 evaluation requests."""

from collections import Counter
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import sys
import time

# Ensure repo root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.image_resolution import extract_all_images, resolve_ledger_with_images
from code.loaders import load_dataset
from code.reconciliation import reconcile_events
from code.recurrence import detect_all_recurrence, expand_future_events
from code.simulator import simulate_user
from code.user_state import (
    ExpenseStability,
    IncomeStability,
    SpendingTrend,
    UserFinancialState,
    build_user_financial_state,
)


def run_user_state_smoke_test() -> None:
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

    print("\nConstructing UserFinancialState across all 250 evaluation requests...")
    t0 = time.perf_counter()

    states = {}
    failures = []

    spending_trends = Counter()
    income_stabilities = Counter()
    expense_stabilities = Counter()
    currencies = Counter()

    users_covered = set()
    rec_income_count = 0
    rec_expense_count = 0
    baseline_breach_count = 0
    insufficient_income_hist_count = 0
    insufficient_expense_hist_count = 0
    unresolved_events_count = 0

    net_cashflows_30d = []

    for req in ds.requests:
        uid = req.user_id
        users_covered.add(uid)
        prof = ds.profiles[uid]
        start_d = req.request_date
        end_d = req.request_date + timedelta(days=90)

        user_canonical = ledger.get_events_for_user(uid)
        user_series = all_series.get(uid, ())
        future_res = expand_future_events(user_series, start_d, end_d, ledger=ledger)

        try:
            baseline = simulate_user(
                user_id=uid,
                simulation_start=start_d,
                simulation_end=end_d,
                canonical_events=user_canonical,
                future_events=future_res.future_events,
                profile=prof,
            )

            state = build_user_financial_state(
                request=req,
                profile=prof,
                canonical_events=user_canonical,
                future_events=future_res.future_events,
                baseline_simulation=baseline,
                recurring_series=user_series,
            )

            states[req.request_id] = state

            # Track distributions
            spending_trends[state.spending_trend] += 1
            income_stabilities[state.income_stability] += 1
            expense_stabilities[state.expense_stability] += 1
            currencies[state.currency] += 1

            if state.has_validated_recurring_income:
                rec_income_count += 1
            if state.has_validated_recurring_expenses:
                rec_expense_count += 1
            if state.baseline_has_safety_breach:
                baseline_breach_count += 1
            if not state.is_income_history_sufficient:
                insufficient_income_hist_count += 1
            if not state.is_expense_history_sufficient:
                insufficient_expense_hist_count += 1
            if state.has_unresolved_events:
                unresolved_events_count += 1

            # Invariant Assertions across all 250 requests
            expected_cash = state.projected_balance - state.pending_reserved_amount
            assert state.available_cash == expected_cash, (
                f"Invariant violation for {req.request_id}: available_cash ({state.available_cash}) != "
                f"projected_balance ({state.projected_balance}) - pending_reserved ({state.pending_reserved_amount})"
            )
            expected_headroom = max(Decimal("0.00"), state.available_cash - state.safety_floor)
            assert state.current_headroom_above_floor == expected_headroom, (
                f"Headroom violation for {req.request_id}: {state.current_headroom_above_floor} != {expected_headroom}"
            )
            assert state.available_cash == baseline.initial_state.available_cash, (
                f"Safe-to-pay identity mismatch for {req.request_id}: {state.available_cash} != {baseline.initial_state.available_cash}"
            )

            net_cashflows_30d.append(state.recent_30d_net_cashflow)

        except Exception as e:
            failures.append((req.request_id, uid, str(e)))

    eval_dur = time.perf_counter() - t0
    print(f"Constructed all {len(states)} user states in {eval_dur:.3f}s ({eval_dur/len(ds.requests)*1000:.2f}ms/request)")

    # Print Summary Report
    print("\n==================================================")
    print("USER FINANCIAL STATE REAL-DATA AUDIT REPORT")
    print("==================================================")
    print(f"Requests evaluated:               {len(states)} / {len(ds.requests)}")
    print(f"Distinct users covered:           {len(users_covered)}")
    print(f"Failures:                         {len(failures)}")
    print("250/250 satisfy the invariant:    available_cash == projected_balance - pending_reserved_amount")
    print("250/250 satisfy headroom rule:    current_headroom == max(0, available_cash - safety_floor)")
    print("250/250 satisfy identity rule:    available_cash == safe_to_pay baseline starting state")
    print(f"Has Validated Recurring Income:   {rec_income_count} ({rec_income_count/len(states)*100:.1f}%)")
    print(f"Has Validated Recurring Expenses: {rec_expense_count} ({rec_expense_count/len(states)*100:.1f}%)")
    print(f"Baseline Safety Floor Breach:     {baseline_breach_count} ({baseline_breach_count/len(states)*100:.1f}%)")
    print(f"Insufficient Income History:      {insufficient_income_hist_count} ({insufficient_income_hist_count/len(states)*100:.1f}%)")
    print(f"Insufficient Expense History:     {insufficient_expense_hist_count} ({insufficient_expense_hist_count/len(states)*100:.1f}%)")
    print(f"Has Unresolved Historical Events: {unresolved_events_count} ({unresolved_events_count/len(states)*100:.1f}%)")

    print("\nCurrencies Distribution:")
    for curr, count in sorted(currencies.items()):
        print(f"  {curr:4}: {count:3d} requests ({count/len(states)*100:.1f}%)")

    print("\nSpending Trend Distribution:")
    for trend, count in sorted(spending_trends.items(), key=lambda x: x[0].value):
        print(f"  {trend.value:18}: {count:3d} ({count/len(states)*100:.1f}%)")

    print("\nIncome Stability Distribution:")
    for stab, count in sorted(income_stabilities.items(), key=lambda x: x[0].value):
        print(f"  {stab.value:18}: {count:3d} ({count/len(states)*100:.1f}%)")

    print("\nExpense Stability Distribution:")
    for stab, count in sorted(expense_stabilities.items(), key=lambda x: x[0].value):
        print(f"  {stab.value:18}: {count:3d} ({count/len(states)*100:.1f}%)")

    # 8 Representative Archetypes
    archetypes = [
        ("1. Strong Affordable Profile", "request_26"),
        ("2. Low Balance Relative to Request", "request_30"),
        ("3. Recurring Salary Dependent", "request_31"),
        ("4. High Recurring Obligations", "request_36"),
        ("5. Pending Debit Present", "request_41"),
        ("6. No Recurring Income (Windfall Dependent)", "request_29"),
        ("7. Negative Baseline Trajectory (Breach)", "request_32"),
        ("8. Multi-Currency Dataset Case (EUR)", "request_40"),
    ]

    print("\n==================================================")
    print("REPRESENTATIVE USER FINANCIAL STATES (8 ARCHETYPES)")
    print("==================================================")
    for label, req_id in archetypes:
        state = states[req_id]
        print(f"\n--- {label} ---")
        print(f"Request:             {state.request_id} ({state.user_id}, Date: {state.request_date}, Curr: {state.currency})")
        print(f"Requested Amount:    {state.requested_amount:,.2f}")
        print(f"Current Balance:     {state.current_available_balance:,.2f} | Projected Bal: {state.projected_balance:,.2f} | Pending Reserve: {state.pending_reserved_amount:,.2f}")
        print(f"Available Cash:      {state.available_cash:,.2f} | Safety Floor: {state.safety_floor:,.2f}")
        print(f"Current Headroom:    {state.current_headroom_above_floor:,.2f}")
        print(f"Recent 30d In/Out:   +{state.recent_30d_income:,.2f} / -{state.recent_30d_outflow:,.2f} (Net: {state.recent_30d_net_cashflow:,.2f})")
        print(f"Recent 30d Ess/Disc: Ess: {state.recent_30d_essential_outflow:,.2f} | Disc: {state.recent_30d_discretionary_outflow:,.2f}")
        print(f"Monthly Avg (90d):   Income: {state.monthly_average_income_90d:,.2f} | Outflow: {state.monthly_average_outflow_90d:,.2f}")
        print(f"Recurring Streams:   Income: {state.total_monthly_recurring_income:,.2f} | Expenses: {state.total_monthly_recurring_expenses:,.2f} (Net: {state.net_monthly_recurring_cashflow:,.2f})")
        print(f"Upcoming (30d / 90d):30d: {state.upcoming_obligations_30d_total:,.2f} | 90d: {state.upcoming_obligations_90d_total:,.2f} (Count: {state.upcoming_obligations_count}, Suppressed Duplicates: {state.suppressed_forecast_count})")
        print(f"Baseline Projection: Min Cash: {state.baseline_minimum_available_cash:,.2f} on {state.baseline_minimum_cash_date} (Breach: {state.baseline_has_safety_breach})")
        print(f"Signals:             Trend: {state.spending_trend.value} ({state.spending_trend_percentage}%) | Inc Stability: {state.income_stability.value} | Exp Stability: {state.expense_stability.value}")
        print(f"Sufficiency:         History: {state.history_days_available}d ({state.total_historical_events_count} events) | Inc OK: {state.is_income_history_sufficient} | Exp OK: {state.is_expense_history_sufficient}")


if __name__ == "__main__":
    run_user_state_smoke_test()
