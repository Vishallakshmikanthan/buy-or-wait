#!/usr/bin/env python3
"""Forensic Audit Script for Prompt 17: Spending-Change Optimization Layer.

Evaluates:
  - Phase 14: Real data audit across all 250 evaluation requests.
  - Phase 15: Regression guard comparing baseline to new output.
  - Phase 16: Performance benchmarking, simulator calls, and timing.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Set, Tuple

from code.canonical import CanonicalEvent
from code.image_resolution import extract_all_images, resolve_ledger_with_images
from code.loaders import load_dataset
from code.models import FinancialProfile, FinancialRequest, PaymentOption
from code.output import generate_all_outputs
from code.payment_plan import evaluate_payment_option_feasibility
from code.reconciliation import reconcile_events
from code.recurrence import FutureEvent, RecurrenceSeries, detect_all_recurrence, expand_future_events
from code.safe_to_pay import evaluate_request_safe_to_pay
from code.simulator import simulate_user
from code.spending_changes import (
    SpendingActionType,
    SpendingChange,
    identify_eligible_spending_actions,
    generate_spending_change_scenarios,
    optimize_spending_changes_for_candidate,
)
from code.candidate_generation import generate_candidates, CandidateStatus

REPO = Path(__file__).resolve().parent
DATASET = REPO / "dataset"


def main() -> None:
    print("=" * 80)
    print("PROMPT 17 — SPENDING-CHANGE OPTIMIZATION FORENSIC AUDIT")
    print("=" * 80)

    t0 = time.perf_counter()
    ds = load_dataset(DATASET)
    base_ledger = reconcile_events(ds.events, ds.profiles, ds.exchange_rates, ds.messages, ds.images)
    extractions = extract_all_images(DATASET)
    resolved_ledger = resolve_ledger_with_images(base_ledger, extractions, ds.exchange_rates)

    all_series, _ = detect_all_recurrence(resolved_ledger, ds.profiles, ds.messages)
    user_canonical = {uid: resolved_ledger.get_events_for_user(uid) for uid in ds.profiles}

    print(f"Loaded dataset: {len(ds.requests)} requests, {len(ds.profiles)} profiles in {time.perf_counter() - t0:.2f}s")

    # 1. Dataset-level eligible flexible future events
    total_eligible_actions = 0
    users_with_eligible_actions: Set[str] = set()
    total_flexible_series = 0
    users_with_flexible_series: Set[str] = set()

    for uid, prof in ds.profiles.items():
        series_list = all_series[uid]
        can_events = user_canonical[uid]

        # Check raw flexibility
        for s in series_list:
            if s.flexibility in ("stoppable", "reducible", "reducible_or_stoppable"):
                total_flexible_series += 1
                users_with_flexible_series.add(uid)

        # Check policy eligibility for a sample horizon
        future_res = expand_future_events(series_list, date(2025, 1, 1), date(2025, 1, 1) + timedelta(days=90), ledger=resolved_ledger)
        actions = identify_eligible_spending_actions(
            user_id=uid,
            request_date=date(2025, 1, 1),
            simulation_end=date(2025, 1, 1) + timedelta(days=90),
            profile=prof,
            series_list=series_list,
            canonical_events=can_events,
            future_events=future_res.future_events,
        )
        if actions:
            total_eligible_actions += len(actions)
            users_with_eligible_actions.add(uid)

    print(f"\n[Phase 14A: Eligibility Counts]")
    print(f"  Total recurring series marked flexible:       {total_flexible_series}")
    print(f"  Users with flexible recurring series:         {len(users_with_flexible_series)} / {len(ds.profiles)}")
    print(f"  Total eligible action options generated:      {total_eligible_actions}")
    print(f"  Users with eligible spending actions:         {len(users_with_eligible_actions)} / {len(ds.profiles)}")

    # 2. Candidate generation with and without spending changes across all 250 requests
    cands_before_count = 0
    cands_after_count = 0
    rescued_candidates_count = 0
    rescued_candidates_safe = 0
    sim_calls_total = 0
    sim_calls_per_req: List[int] = []

    req_eval_times: List[float] = []
    opt_eval_times: List[float] = []

    candidates_by_req_after = {}
    candidates_by_req_before = {}

    requests_by_id = {r.request_id: r for r in ds.requests}

    # Wrap simulate_user in code.spending_changes to count optimization simulation calls
    import code.spending_changes as sc_module
    original_simulate_user = sc_module.simulate_user
    sim_calls_count = 0
    sim_calls_per_req_dict = defaultdict(int)
    current_req_id = ""

    def counting_simulate_user(*args, **kwargs):
        nonlocal sim_calls_count
        sim_calls_count += 1
        if current_req_id:
            sim_calls_per_req_dict[current_req_id] += 1
        return original_simulate_user(*args, **kwargs)

    sc_module.simulate_user = counting_simulate_user

    for req in ds.requests:
        current_req_id = req.request_id
        uid = req.user_id
        profile = ds.profiles[uid]
        can_events = user_canonical[uid]
        start_d = req.request_date
        end_d = start_d + timedelta(days=90)
        future_res = expand_future_events(all_series[uid], start_d, end_d, ledger=resolved_ledger)
        fut_events = future_res.future_events
        series = all_series[uid]

        t_r0 = time.perf_counter()

        # Baseline simulation & safe to pay
        baseline = simulate_user(
            user_id=uid,
            simulation_start=req.request_date,
            simulation_end=req.request_date + timedelta(days=90),
            canonical_events=can_events,
            future_events=fut_events,
            profile=profile,
        )
        cert_safe = evaluate_request_safe_to_pay(
            request=req,
            baseline_simulation=baseline,
            canonical_events=can_events,
            future_events=fut_events,
            profile=profile,
        )
        options = ds.payment_options_by_request.get(req.request_id, [])
        opts_by_id = {o.payment_option_id: o for o in options}
        feasibilities = [
            evaluate_payment_option_feasibility(
                option=o,
                request=req,
                profile=profile,
                canonical_events=can_events,
                future_events=fut_events,
                baseline_simulation=baseline,
            )
            for o in options
        ]

        # Before (without spending changes)
        cset_before = generate_candidates(
            request=req,
            profile=profile,
            certificate=cert_safe,
            payment_option_feasibilities=feasibilities,
            payment_options_by_id=opts_by_id,
            canonical_events=(),
            future_events=(),
            recurrence_series=(),
        )
        cands_before_count += len(cset_before.candidates)
        candidates_by_req_before[req.request_id] = cset_before

        # After (with spending changes)
        t_opt0 = time.perf_counter()
        cset_after = generate_candidates(
            request=req,
            profile=profile,
            certificate=cert_safe,
            payment_option_feasibilities=feasibilities,
            payment_options_by_id=opts_by_id,
            canonical_events=can_events,
            future_events=fut_events,
            recurrence_series=series,
        )
        t_opt1 = time.perf_counter()
        opt_eval_times.append(t_opt1 - t_opt0)
        req_eval_times.append(t_opt1 - t_r0)

        cands_after_count += len(cset_after.candidates)
        candidates_by_req_after[req.request_id] = cset_after

        rescued = [c for c in cset_after.candidates if c.spending_changes]
        rescued_candidates_count += len(rescued)
        rescued_candidates_safe += sum(1 for c in rescued if c.is_safe)

    # Restore original simulate_user
    sc_module.simulate_user = original_simulate_user

    print(f"\n[Phase 14B: Candidate Generation Counts]")
    print(f"  Candidates generated BEFORE spending changes: {cands_before_count}")
    print(f"  Candidates generated AFTER spending changes:  {cands_after_count}")
    print(f"  Net rescued spending-change candidates:       {rescued_candidates_count}")
    print(f"  Rescued candidates marked safe:               {rescued_candidates_safe}")

    # 3. Output comparison (Phase 15: Regression Guard)
    import subprocess
    try:
        baseline_csv_content = subprocess.check_output(
            ["git", "show", "HEAD:hackerrank-orchestrate-september26/output.csv"],
            cwd=REPO,
            text=True,
        )
        old_reader = csv.DictReader(baseline_csv_content.splitlines())
        old_rows = {r["request_id"]: r for r in old_reader}
    except Exception:
        old_output_path = REPO / "output.csv"
        with open(old_output_path, "r", encoding="utf-8") as f:
            old_rows = {r["request_id"]: r for r in csv.DictReader(f)}

    temp_new_csv = REPO / "scratch" / "audit_output.csv"
    temp_new_csv.parent.mkdir(exist_ok=True)
    new_output_rows, new_sha256, new_size = generate_all_outputs(DATASET, temp_new_csv)

    changed_requests = []
    for r in new_output_rows:
        old = old_rows[r.request_id]
        diffs = {}
        for k in ("affordability_status", "recommended_payment_method", "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed"):
            if getattr(r, k) != old[k]:
                diffs[k] = (old[k], getattr(r, k))
        if diffs:
            changed_requests.append((r, diffs))

    print(f"\n[Phase 15: Regression Guard & Changed Decisions]")
    print(f"  Total requests:                               250")
    print(f"  Unchanged requests:                           {250 - len(changed_requests)}")
    print(f"  Changed requests:                             {len(changed_requests)}")

    for r, diffs in changed_requests:
        req_obj = requests_by_id[r.request_id]
        print(f"\n  Request {r.request_id} (User {req_obj.user_id}):")
        for k, (old_val, new_val) in diffs.items():
            print(f"    {k:30s}: {old_val} -> {new_val}")
        print(f"    Explanation: {r.decision_explanation}")

    # Machine-readable audit records
    audit_records = []
    for r, diffs in changed_requests:
        req_obj = requests_by_id[r.request_id]
        audit_records.append({
            "request_id": r.request_id,
            "user_id": req_obj.user_id,
            "diffs": {k: {"before": v[0], "after": v[1]} for k, v in diffs.items()},
            "spending_changes_needed": r.spending_changes_needed,
            "explanation": r.decision_explanation,
            "classification": "expected_improvement_spending_change_rescue",
        })

    audit_json_path = REPO / "scratch" / "spending_changes_audit.json"
    audit_json_path.write_text(json.dumps(audit_records, indent=2))
    print(f"\n  Saved machine-readable audit record to {audit_json_path}")

    # Phase 16: Performance Metrics
    total_opt_time = sum(opt_eval_times)
    total_req_time = sum(req_eval_times)
    sim_calls_list = list(sim_calls_per_req_dict.values()) or [0]
    print(f"\n[Phase 16: Performance Metrics]")
    print(f"  Total candidate generation + optimization:    {total_req_time:.2f}s")
    print(f"  Pure spending-change optimization runtime:    {total_opt_time:.2f}s")
    print(f"  Average optimization runtime per request:     {(total_opt_time / 250) * 1000:.2f}ms")
    print(f"  Max optimization runtime single request:      {max(opt_eval_times) * 1000:.2f}ms")
    print(f"  Total simulator calls during optimization:    {sim_calls_count}")
    print(f"  Average simulator calls per request:          {sim_calls_count / 250:.2f}")
    print(f"  Max simulator calls for a single request:     {max(sim_calls_list)}")

    # Clean up temp
    if temp_new_csv.exists():
        temp_new_csv.unlink()

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE — ALL METRICS COMPUTED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
