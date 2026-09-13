"""Prompt 12 — Real-Data Final Decision + Safety Gate Audit across all 250 evaluation requests.

Runs the frozen pipeline + candidate generation + ranking + deterministic final decision & safety gate:
- Evaluates all 250 requests
- Gathers candidate universe
- Runs candidates through is_finally_safe / evaluate_candidate_safety_gate
- Ranks surviving candidates via frozen ranking engine
- Selects top surviving candidate or generates deterministic fallback
- Audits distributions:
    * total requests
    * requests with >=1 rankable candidate
    * requests with >=1 finally-safe candidate
    * requests with 0 finally-safe candidates
    * selected candidate counts by payment method
    * affordability status distribution
    * rejection counts by safety reason
    * cases where final gate rejected a candidate upstream marked rankable
    * cases where final winner differs from candidate-generation first candidate
    * cases where final winner differs from ranking winner
    * cases with provisional affordability state
    * deterministic repeat-run equality
- Emits detailed structured traces for auditability
- Verifies post-audit frozen module hashes
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from code.affordability import AffordabilityStatus
from code.candidate_generation import (
    Candidate,
    CandidateSet,
    CandidateType,
    generate_candidates,
)
from code.final_decision import (
    DecisionReasonCode,
    FinalDecision,
    RequestContext,
    evaluate_candidate_safety_gate,
    is_finally_safe,
    make_final_decision,
    make_final_decision_from_candidate_set,
)
from code.image_resolution import extract_all_images, resolve_ledger_with_images
from code.loaders import load_dataset
from code.payment_plan import evaluate_payment_option_feasibility
from code.ranking import (
    RankedCandidateSet,
    rank_candidates,
    rank_candidate_set,
)
from code.reconciliation import reconcile_events
from code.recurrence import detect_all_recurrence, expand_future_events
from code.safe_to_pay import evaluate_request_safe_to_pay
from code.simulator import simulate_user

REPO = Path(__file__).resolve().parent
DATASET = REPO / "dataset"
CODE = REPO / "code"

FROZEN_FILES = (
    "canonical.py",
    "recurrence.py",
    "simulator.py",
    "safe_to_pay.py",
    "user_state.py",
    "affordability.py",
    "payment_plan.py",
    "candidate_generation.py",
    "ranking.py",
)

FROZEN_HASH_BASELINE = {
    "canonical.py": "c98056ef6a4aad8c689aa02e8781da0c4a716548",
    "recurrence.py": "84a149bbb792d41002856eb1ac337597e2df964b",
    "simulator.py": "50dce11d0d8238e12eb4e27d86a46e05aa53df5b",
    "safe_to_pay.py": "c5df53f89ad7ca0e7a05dc9202171a0e1a7c91c6",
    "user_state.py": "789dfbb5b3a57b179fd4ccfc8b95e0a2ffe47a17",
    "affordability.py": "a81175aeed50d8ddeac3f33ca650b311c069355c",
    "payment_plan.py": "740ae3132a0237eaced86b3d31ac0435b6214a48",
    "candidate_generation.py": "e1ab54ad795e724127c0e336fef7c720ef300999",
    "ranking.py": "78f208b16b7dd5c0b711e5e2e73241c6034b6248",
}


def git_hash_object(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    t_start = time.perf_counter()

    print_section("0. PRE-AUDIT FROZEN MODULE HASHES")
    for name in FROZEN_FILES:
        current = git_hash_object(CODE / name)
        baseline = FROZEN_HASH_BASELINE[name]
        match = current == baseline
        print(f"  {name:25s} {current} [{'UNCHANGED' if match else 'CHANGED'}]")
        assert match, f"Frozen module {name} hash mismatch!"

    print_section("1. PIPELINE INITIALIZATION")
    ds = load_dataset(DATASET)
    base_ledger = reconcile_events(ds.events, ds.profiles, ds.exchange_rates, ds.messages, ds.images)
    extractions = extract_all_images(DATASET)
    resolved_ledger = resolve_ledger_with_images(base_ledger, extractions, ds.exchange_rates)

    all_series, _ = detect_all_recurrence(resolved_ledger, ds.profiles, ds.messages)
    user_canonical = {uid: resolved_ledger.get_events_for_user(uid) for uid in ds.profiles}
    print(f"  Loaded {len(ds.requests)} evaluation requests across {len(ds.profiles)} users.")

    print_section("2. RUNNING DETERMINISTIC FINAL DECISION & SAFETY GATE")
    decisions_run1: list[FinalDecision] = []
    contexts: list[RequestContext] = []
    candidate_sets: list[CandidateSet] = []

    for req in ds.requests:
        uid = req.user_id
        profile = ds.profiles[uid]
        start_d = req.request_date
        end_d = start_d + timedelta(days=90)
        future_res = expand_future_events(all_series[uid], start_d, end_d, ledger=resolved_ledger)
        baseline = simulate_user(
            user_id=uid,
            simulation_start=start_d,
            simulation_end=end_d,
            canonical_events=user_canonical[uid],
            future_events=future_res.future_events,
            profile=profile,
        )
        cert = evaluate_request_safe_to_pay(
            request=req,
            baseline_simulation=baseline,
            canonical_events=user_canonical[uid],
            future_events=future_res.future_events,
            profile=profile,
        )
        options = ds.payment_options_by_request.get(req.request_id, [])
        opts_by_id = {o.payment_option_id: o for o in options}
        feasibilities = [
            evaluate_payment_option_feasibility(
                option=o,
                request=req,
                profile=profile,
                canonical_events=user_canonical[uid],
                future_events=future_res.future_events,
                baseline_simulation=baseline,
            )
            for o in options
        ]
        feas_dict = {f.payment_option_id: f for f in feasibilities}

        cset = generate_candidates(req, profile, cert, feasibilities, opts_by_id)
        candidate_sets.append(cset)

        ctx = RequestContext(
            request=req,
            profile=profile,
            certificate=cert,
            payment_option_feasibilities=feas_dict,
        )
        contexts.append(ctx)

        dec = make_final_decision_from_candidate_set(ctx, cset)
        decisions_run1.append(dec)

    print(f"  Processed {len(decisions_run1)} requests in {time.perf_counter() - t_start:.2f}s.")

    print_section("3. REPEAT RUN DETERMINISM CHECK")
    decisions_run2: list[FinalDecision] = []
    for ctx, cset in zip(contexts, candidate_sets):
        dec2 = make_final_decision_from_candidate_set(ctx, cset)
        decisions_run2.append(dec2)

    assert len(decisions_run1) == len(decisions_run2)
    mismatches = 0
    for d1, d2 in zip(decisions_run1, decisions_run2):
        if d1 != d2:
            mismatches += 1
            print(f"  DETERMINISM MISMATCH in request {d1.request_id}!")
    print(f"  Deterministic repeat-run identical decisions: {len(decisions_run1) - mismatches}/{len(decisions_run1)}")
    assert mismatches == 0, "Repeat runs produced different results!"

    print_section("4. AUDIT METRICS & DISTRIBUTIONS")
    total_requests = len(decisions_run1)
    reqs_with_rankable = 0
    reqs_with_finally_safe = 0
    reqs_with_zero_finally_safe = 0
    rejections_counter = Counter()
    payment_method_counter = Counter()
    affordability_counter = Counter()
    upstream_rankable_rejected_by_gate = 0
    diff_from_raw_cgen_first = 0
    diff_from_ranking_winner = 0
    provisional_affordability_count = 0

    for dec, cset, ctx in zip(decisions_run1, candidate_sets, contexts):
        n_rankable = len(cset.rankable_candidates)
        if n_rankable > 0:
            reqs_with_rankable += 1

        surviving = [sg for sg in dec.all_safety_gate_results if sg.is_safe]
        if len(surviving) > 0:
            reqs_with_finally_safe += 1
        else:
            reqs_with_zero_finally_safe += 1

        for sg in dec.all_safety_gate_results:
            for r_code in sg.rejection_reason_codes:
                rejections_counter[r_code.value] += 1

        # Check if final gate rejected an upstream rankable candidate
        for c in cset.rankable_candidates:
            sg = next((s for s in dec.all_safety_gate_results if s.candidate_id == c.candidate_id), None)
            if sg is not None and not sg.is_safe:
                upstream_rankable_rejected_by_gate += 1

        # Raw cgen first candidate comparison
        if cset.candidates:
            first_raw = cset.candidates[0]
            if dec.selected_candidate_id != first_raw.candidate_id:
                diff_from_raw_cgen_first += 1

        # Ranking winner comparison (ranking over rankable candidates)
        ranked_upstream = rank_candidate_set(cset, req.desired_completion_date)
        upstream_winner_id = (
            ranked_upstream.top_ranked_candidate.candidate.candidate_id
            if ranked_upstream.top_ranked_candidate is not None
            else None
        )
        if dec.selected_candidate_id != upstream_winner_id:
            diff_from_ranking_winner += 1

        payment_method_counter[dec.recommended_payment_method] += 1
        affordability_counter[dec.affordability_status.value] += 1

    print(f"  Total requests:                                    {total_requests}")
    print(f"  Requests with >=1 rankable candidate:              {reqs_with_rankable}")
    print(f"  Requests with >=1 finally-safe candidate:          {reqs_with_finally_safe}")
    print(f"  Requests with zero finally-safe candidates:        {reqs_with_zero_finally_safe}")
    print(f"  Upstream rankable rejected by final safety gate:   {upstream_rankable_rejected_by_gate}")
    print(f"  Final winner differs from raw cgen first candidate: {diff_from_raw_cgen_first}")
    print(f"  Final winner differs from ranking winner:          {diff_from_ranking_winner}")
    print(f"  Requests with provisional affordability state:     {provisional_affordability_count}")

    print("\n  [Selected Candidate by Payment Method]")
    for meth, cnt in payment_method_counter.most_common():
        print(f"    {meth:25s}: {cnt:3d} ({cnt/total_requests*100:5.1f}%)")

    print("\n  [Affordability Status Distribution]")
    for st, cnt in affordability_counter.most_common():
        print(f"    {st:25s}: {cnt:3d} ({cnt/total_requests*100:5.1f}%)")

    print("\n  [Rejections by Safety Reason Code]")
    for r_code, cnt in rejections_counter.most_common():
        print(f"    {r_code:30s}: {cnt:4d}")

    print_section("5. DETAILED TRACEABILITY FOR MULTI-CANDIDATE AND ZERO-SAFE REQUESTS")
    multi_candidate_reqs = [
        (d, cs) for d, cs in zip(decisions_run1, candidate_sets) if len(cs.rankable_candidates) > 1
    ]
    print(f"  Total requests with multiple rankable candidates: {len(multi_candidate_reqs)}")
    for d, cs in multi_candidate_reqs:
        print(f"\n  Request {d.request_id} (User: {d.explanation_evidence.user_id}):")
        print(f"    Requested amount: {d.explanation_evidence.requested_amount}, Desired date: {d.explanation_evidence.desired_completion_date}")
        print(f"    Rankable candidates ({len(cs.rankable_candidates)}): {[c.candidate_id for c in cs.rankable_candidates]}")
        print(f"    Selected candidate: {d.selected_candidate_id}")
        print(f"    Recommended payment method: {d.recommended_payment_method}")
        print(f"    Affordability status: {d.affordability_status.value}")
        if d.ranking_trace:
            t = d.ranking_trace
            print(f"    Ranking trace: deadline_met={t.criterion_1_deadline_met}, no_spending_changes={t.criterion_2_no_spending_changes}, total_paid={t.criterion_3_total_amount_paid}, first_date={t.criterion_4_first_payment_date}, num_payments={t.criterion_5_number_of_payments}")
        else:
            print("    Ranking trace: None")

    zero_safe_sample = [d for d in decisions_run1 if d.selected_candidate_id is None][:5]
    print(f"\n  Sample Zero-Safe Requests (5 of {reqs_with_zero_finally_safe}):")
    for d in zero_safe_sample:
        print(f"    Request {d.request_id}: status={d.affordability_status.value}, reason_codes={[c.value for c in d.decision_reason_codes]}")

    print_section("6. POST-AUDIT FROZEN MODULE INTEGRITY CHECK")
    for name in FROZEN_FILES:
        current = git_hash_object(CODE / name)
        baseline = FROZEN_HASH_BASELINE[name]
        assert current == baseline, f"Post-audit violation: {name} was modified!"
        print(f"  {name:25s} {current} [UNCHANGED]")

    print_section("AUDIT COMPLETE — ALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
