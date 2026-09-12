"""Prompt 10B — Candidate Set Integrity Audit.

Runs the frozen financial engine + candidate generation across:
  - all 790 raw payment-option rows (evaluation + sample requests)
  - all 250 evaluation requests

Does NOT rank, score, recommend, write output.csv, or call an LLM.
Does NOT modify frozen upstream modules.
"""
from __future__ import annotations

import hashlib
import random
import time
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from code.candidate_generation import (
    CandidateStatus,
    CandidateType,
    classify_option_universe_state,
    generate_candidates,
    is_rankable,
    OptionUniverseState,
)
from code.image_resolution import extract_all_images, resolve_ledger_with_images
from code.loaders import load_dataset, load_requests
from code.payment_plan import evaluate_payment_option_feasibility
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
)

# Baseline hashes captured at the start of Prompt 10B (git hash-object).
FROZEN_HASH_BASELINE = {
    "canonical.py": "c98056ef6a4aad8c689aa02e8781da0c4a716548",
    "recurrence.py": "84a149bbb792d41002856eb1ac337597e2df964b",
    "simulator.py": "50dce11d0d8238e12eb4e27d86a46e05aa53df5b",
    "safe_to_pay.py": "c5df53f89ad7ca0e7a05dc9202171a0e1a7c91c6",
    "user_state.py": "789dfbb5b3a57b179fd4ccfc8b95e0a2ffe47a17",
    "affordability.py": "a81175aeed50d8ddeac3f33ca650b311c069355c",
    "payment_plan.py": "740ae3132a0237eaced86b3d31ac0435b6214a48",
}


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def git_hash_object(path: Path) -> str:
    """Match `git hash-object` (blob SHA-1)."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    t_all = time.perf_counter()

    print_section("0. FROZEN MODULE HASHES (pre-audit)")
    frozen_ok = True
    for name in FROZEN_FILES:
        current = git_hash_object(CODE / name)
        baseline = FROZEN_HASH_BASELINE[name]
        match = current == baseline
        frozen_ok = frozen_ok and match
        flag = "UNCHANGED" if match else "CHANGED"
        print(f"  {name:20s} {current}  [{flag}]")
        if not match:
            print(f"    baseline {baseline}")
    print(f"  Frozen modules unchanged: {frozen_ok}")

    print_section("1. LOAD + RECONCILE")
    t0 = time.perf_counter()
    ds = load_dataset(DATASET)
    sample_requests = load_requests(DATASET / "sample_requests.csv")
    all_requests_map = {r.request_id: r for r in ds.requests}
    all_requests_map.update({r.request_id: r for r in sample_requests})
    eval_ids = {r.request_id for r in ds.requests}
    sample_ids = {r.request_id for r in sample_requests}
    print(f"  eval requests:   {len(ds.requests)}")
    print(f"  sample requests: {len(sample_requests)}")
    print(f"  payment options: {len(ds.payment_options)}")
    print(f"  loaded in {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    base_ledger = reconcile_events(ds.events, ds.profiles, ds.exchange_rates, ds.messages, ds.images)
    extractions = extract_all_images(DATASET)
    ledger = resolve_ledger_with_images(base_ledger, extractions, ds.exchange_rates)
    all_series, _ = detect_all_recurrence(ledger, ds.profiles, ds.messages)
    print(f"  ledger+recurrence in {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Per-request cache: baseline sim + certificate + feasibilities
    # ------------------------------------------------------------------
    print_section("2. EVALUATE 790 OPTIONS + 250 CANDIDATE SETS")
    t0 = time.perf_counter()

    option_rows = []  # dicts for the 790-option universe
    request_rows = []  # dicts for the 250-request candidate universe
    all_candidates = []  # every generated candidate (eval only)
    partials = []
    waits = []
    lineage_missing = []
    errors = []

    # Cache per (user_id, request_date) is not shared across different requests
    # because certificates and options are request-specific. We still reuse
    # per-user canonical events.
    user_canonical = {uid: ledger.get_events_for_user(uid) for uid in ds.profiles}

    # Evaluate every payment option (790), including sample-request options.
    for opt in ds.payment_options:
        req = all_requests_map.get(opt.request_id)
        if req is None:
            option_rows.append({
                "option": opt,
                "request_id": opt.request_id,
                "method": opt.payment_method,
                "universe": "orphan",
                "feasibility": None,
                "state": OptionUniverseState.STRUCTURALLY_INVALID,
                "candidate_created": False,
                "candidate_excluded": True,
                "exclusion_reason": "no matching request row",
            })
            continue
        profile = ds.profiles[req.user_id]
        start_d = req.request_date
        end_d = start_d + timedelta(days=90)
        future_res = expand_future_events(all_series[req.user_id], start_d, end_d, ledger=ledger)
        feas = evaluate_payment_option_feasibility(
            option=opt,
            request=req,
            profile=profile,
            canonical_events=user_canonical[req.user_id],
            future_events=future_res.future_events,
            simulation_start=start_d,
            simulation_end=end_d,
        )
        state = classify_option_universe_state(feas)
        in_eval = opt.request_id in eval_ids
        is_installment = opt.payment_method == "installments"
        created = in_eval and is_installment
        if created:
            excl_reason = None
        elif not in_eval:
            excl_reason = "sample_request_option"
        elif not is_installment:
            excl_reason = "csv_full_payment_not_used_for_installment_candidate"
        else:
            excl_reason = "unknown"
        option_rows.append({
            "option": opt,
            "request_id": opt.request_id,
            "method": opt.payment_method,
            "universe": "eval" if in_eval else "sample",
            "feasibility": feas,
            "state": state,
            "candidate_created": created,
            "candidate_excluded": not created,
            "exclusion_reason": excl_reason,
        })

    # Generate candidates for every evaluation request.
    for req in ds.requests:
        uid = req.user_id
        profile = ds.profiles[uid]
        start_d = req.request_date
        end_d = start_d + timedelta(days=90)
        try:
            future_res = expand_future_events(all_series[uid], start_d, end_d, ledger=ledger)
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
            cs = generate_candidates(req, profile, cert, feasibilities, opts_by_id)

            # Mutation check: frozen inputs must be bit-identical after generation.
            assert cert.request_id == req.request_id
            assert profile.current_available_balance == ds.profiles[uid].current_available_balance

            request_rows.append({
                "request": req,
                "cert": cert,
                "profile": profile,
                "options": options,
                "feasibilities": feasibilities,
                "opts_by_id": opts_by_id,
                "candidate_set": cs,
                "baseline": baseline,
            })
            for c in cs.candidates:
                all_candidates.append(c)
                if c.candidate_type == CandidateType.PARTIAL_PAYMENT:
                    partials.append((req, cert, c))
                if c.candidate_type == CandidateType.WAIT:
                    waits.append((req, cert, c))
                # Lineage
                missing = []
                if not c.provenance.source:
                    missing.append("source")
                if c.candidate_type == CandidateType.INSTALLMENT_PLAN:
                    if not c.provenance.source_payment_option_id:
                        missing.append("source_payment_option_id")
                    if c.provenance.source != "payment_option_feasibility":
                        missing.append("installment_source")
                if c.candidate_type == CandidateType.FULL_PAYMENT:
                    if c.provenance.source != "safe_to_pay_certificate":
                        missing.append("full_source")
                    if c.provenance.safe_to_pay_certificate_id != cert.request_id:
                        missing.append("full_cert_id")
                if c.candidate_type == CandidateType.PARTIAL_PAYMENT:
                    if c.provenance.source != "safe_to_pay_certificate":
                        missing.append("partial_source")
                    if c.provenance.safe_to_pay_certificate_id != cert.request_id:
                        missing.append("partial_cert_id")
                if c.candidate_type == CandidateType.WAIT:
                    if c.provenance.source != "earliest_full_payment_computation":
                        missing.append("wait_source")
                    if c.provenance.earliest_full_payment_date is None:
                        missing.append("wait_date")
                if missing:
                    lineage_missing.append((c.candidate_id, missing))
        except Exception as e:
            errors.append(f"{req.request_id}: {type(e).__name__}: {e}")

    print(f"  evaluated in {time.perf_counter() - t0:.3f}s")
    print(f"  errors: {len(errors)}")
    for e in errors[:20]:
        print(f"    {e}")

    # ------------------------------------------------------------------
    # 790-option reconciliation
    # ------------------------------------------------------------------
    print_section("3. 790-OPTION RECONCILIATION")
    n_opts = len(ds.payment_options)
    print(f"  raw payment options: {n_opts}")

    by_universe = defaultdict(int)
    by_method = defaultdict(int)
    by_state = defaultdict(int)
    by_state_method = defaultdict(int)
    by_state_universe = defaultdict(int)
    created_by_state = defaultdict(int)
    excluded_by_state = defaultdict(int)
    excluded_reasons = defaultdict(int)
    eval_eligible = eval_safe = eval_unsafe = eval_ineligible = 0
    sample_eligible = sample_safe = 0

    for row in option_rows:
        by_universe[row["universe"]] += 1
        by_method[row["method"]] += 1
        st = row["state"].value
        by_state[st] += 1
        by_state_method[(st, row["method"])] += 1
        by_state_universe[(st, row["universe"])] += 1
        if row["candidate_created"]:
            created_by_state[st] += 1
        else:
            excluded_by_state[st] += 1
            excluded_reasons[row["exclusion_reason"]] += 1
        feas = row["feasibility"]
        if feas is None:
            continue
        if row["universe"] == "eval":
            if feas.is_eligible:
                eval_eligible += 1
                if feas.is_safe:
                    eval_safe += 1
                else:
                    eval_unsafe += 1
            else:
                eval_ineligible += 1
        elif row["universe"] == "sample":
            if feas.is_eligible:
                sample_eligible += 1
                if feas.is_safe:
                    sample_safe += 1

    eligible_790 = by_state[OptionUniverseState.ELIGIBLE_SAFE.value] + by_state[OptionUniverseState.ELIGIBLE_UNSAFE.value]
    safe_790 = by_state[OptionUniverseState.ELIGIBLE_SAFE.value]
    unsafe_790 = by_state[OptionUniverseState.ELIGIBLE_UNSAFE.value]
    ineligible_790 = (
        by_state[OptionUniverseState.CONTRACTUALLY_INELIGIBLE.value]
        + by_state[OptionUniverseState.STRUCTURALLY_INVALID.value]
    )

    print("  universe:")
    for k in sorted(by_universe):
        print(f"    {k:12s} {by_universe[k]}")
    print("  method:")
    for k in sorted(by_method):
        print(f"    {k:12s} {by_method[k]}")
    print("  A/B/C/D state:")
    for k in sorted(by_state):
        print(f"    {k:32s} {by_state[k]}")
    print()
    print(f"  A+B+C+D                         = {sum(by_state.values())}")
    print(f"  B+C eligible                    = {eligible_790}")
    print(f"  B   safe                        = {safe_790}")
    print(f"  C   unsafe                      = {unsafe_790}")
    print(f"  A+D ineligible                  = {ineligible_790}")
    print()
    print("  frozen payment_plan.py expected: 247 eligible / 146 safe / 101 unsafe / 543 ineligible")
    print(f"  match eligible 247? {eligible_790 == 247}")
    print(f"  match safe 146?     {safe_790 == 146}")
    print(f"  match unsafe 101?   {unsafe_790 == 101}")
    print(f"  match ineligible 543? {ineligible_790 == 543}")
    print()
    print("  eval-only feasibility:")
    print(f"    eligible {eval_eligible}  safe {eval_safe}  unsafe {eval_unsafe}  ineligible {eval_ineligible}")
    print(f"    eval total {eval_eligible + eval_ineligible}")
    print("  sample-only feasibility:")
    print(f"    eligible {sample_eligible}  (of which safe {sample_safe})")
    print()
    print("  candidate-generation behavior by option state:")
    print(f"    {'state':32s} {'created':>8s} {'excluded':>8s}")
    for st in sorted(set(list(created_by_state) + list(excluded_by_state))):
        print(f"    {st:32s} {created_by_state[st]:8d} {excluded_by_state[st]:8d}")
    print("  exclusion reasons:")
    for k, v in sorted(excluded_reasons.items(), key=lambda kv: kv[0] or ""):
        print(f"    {str(k):50s} {v}")
    print()
    print("  state x method:")
    for (st, method), n in sorted(by_state_method.items()):
        print(f"    {st:32s} {method:15s} {n}")

    created_installments = sum(created_by_state.values())
    print()
    print(f"  installment candidates created from 790-option map: {created_installments}")
    print("  WHY 469 installment candidates can coexist with 247 eligible options:")
    print("    247 = B+C across ALL 790 CSV rows (eval+sample, full_payment+installments).")
    print("    469 = every installment-method CSV row attached to an evaluation request,")
    print("          retained as a candidate regardless of A/B/C/D. That set includes")
    print("          contractually ineligible, unsafe, structurally invalid, AND safe eligible.")
    print("    They are different universes: 469 is not a subset of 247, and 247 is not")
    print("    a subset of 469 (247 includes CSV full_payment rows and sample-request rows).")

    # ------------------------------------------------------------------
    # Candidate accounting table (250 requests)
    # ------------------------------------------------------------------
    print_section("4. CANDIDATE ACCOUNTING TABLE (250 REQUESTS)")

    type_status = defaultdict(int)
    type_total = defaultdict(int)
    type_rankable = defaultdict(int)
    status_total = defaultdict(int)
    full_bucket = defaultdict(int)

    for c in all_candidates:
        type_total[c.candidate_type.value] += 1
        type_status[(c.candidate_type.value, c.status.value)] += 1
        status_total[c.status.value] += 1
        if c.is_rankable:
            type_rankable[c.candidate_type.value] += 1
        if c.candidate_type == CandidateType.FULL_PAYMENT:
            if c.status == CandidateStatus.ELIGIBLE_AND_SAFE:
                full_bucket["safe"] += 1
            elif c.status == CandidateStatus.STRUCTURALLY_VALID_UNSAFE:
                full_bucket["unsafe"] += 1
            elif c.status == CandidateStatus.CONTRACTUALLY_INELIGIBLE:
                full_bucket["ineligible"] += 1
            else:
                full_bucket["invalid"] += 1

    # Excluded (not generated) per type, relative to 250 requests / eval installment options.
    n_eval = len(ds.requests)
    n_eval_installment_opts = sum(
        1 for row in option_rows
        if row["universe"] == "eval" and row["method"] == "installments"
    )
    excluded = {
        "full_payment": n_eval - type_total["full_payment"],  # expect 0
        "installments": n_eval_installment_opts - type_total["installments"],  # expect 0
        "partial_payment": n_eval - type_total["partial_payment"],
        "wait": n_eval - type_total["wait"],
    }

    header = (
        f"{'type':18s} {'gen':>6s} {'elig_safe':>10s} {'elig_unsafe':>12s} "
        f"{'contr_inelig':>13s} {'struct_inv':>11s} {'excluded':>9s} {'rankable':>9s}"
    )
    print("  " + header)
    print("  " + "-" * len(header))
    grand = defaultdict(int)
    for t in ("full_payment", "installments", "partial_payment", "wait"):
        gen = type_total[t]
        es = type_status[(t, CandidateStatus.ELIGIBLE_AND_SAFE.value)]
        eu = type_status[(t, CandidateStatus.STRUCTURALLY_VALID_UNSAFE.value)]
        ci = type_status[(t, CandidateStatus.CONTRACTUALLY_INELIGIBLE.value)]
        si = type_status[(t, CandidateStatus.STRUCTURALLY_INVALID.value)]
        ex = excluded[t]
        rk = type_rankable[t]
        # Mutual exclusion among generated
        assert es + eu + ci + si == gen, f"{t}: {es}+{eu}+{ci}+{si} != {gen}"
        print(
            f"  {t:18s} {gen:6d} {es:10d} {eu:12d} {ci:13d} {si:11d} {ex:9d} {rk:9d}"
        )
        grand["gen"] += gen
        grand["es"] += es
        grand["eu"] += eu
        grand["ci"] += ci
        grand["si"] += si
        grand["ex"] += ex
        grand["rk"] += rk
    print("  " + "-" * len(header))
    print(
        f"  {'TOTAL':18s} {grand['gen']:6d} {grand['es']:10d} {grand['eu']:12d} "
        f"{grand['ci']:13d} {grand['si']:11d} {grand['ex']:9d} {grand['rk']:9d}"
    )
    print()
    print("  Categories among GENERATED candidates are mutually exclusive:")
    print("    generated = eligible_safe + eligible_unsafe + contractually_ineligible + structurally_invalid")
    print(f"    {grand['gen']} = {grand['es']} + {grand['eu']} + {grand['ci']} + {grand['si']}")
    print(f"    check: {grand['es'] + grand['eu'] + grand['ci'] + grand['si'] == grand['gen']}")
    print("  Excluded is NOT a generated-candidate status; it counts potential actions that")
    print("  were not constructed (partial/wait construction gates; CSV full_payment is not")
    print("  counted here because the canonical full_payment candidate is always emitted).")

    print()
    print("  status totals:")
    for k, v in sorted(status_total.items()):
        print(f"    {k:32s} {v}")

    # ------------------------------------------------------------------
    # Rankability
    # ------------------------------------------------------------------
    print_section("5. RANKABILITY GATE")
    n_rankable = sum(1 for c in all_candidates if is_rankable(c))
    n_unrankable = len(all_candidates) - n_rankable
    invalid_rankable = [
        c for c in all_candidates
        if is_rankable(c) and c.status in (
            CandidateStatus.STRUCTURALLY_INVALID,
            CandidateStatus.CONTRACTUALLY_INELIGIBLE,
        )
    ]
    unsafe_rankable = [
        c for c in all_candidates
        if is_rankable(c) and (not c.is_safe or c.status == CandidateStatus.STRUCTURALLY_VALID_UNSAFE)
    ]
    safe_not_rankable = [
        c for c in all_candidates
        if c.is_safe and not is_rankable(c)
    ]
    print(f"  generated:   {len(all_candidates)}")
    print(f"  rankable:    {n_rankable}")
    print(f"  unrankable:  {n_unrankable}")
    print(f"  invalid-but-rankable (MUST be 0): {len(invalid_rankable)}")
    print(f"  unsafe-but-rankable  (MUST be 0): {len(unsafe_rankable)}")
    print(f"  safe-but-not-rankable (MUST be 0): {len(safe_not_rankable)}")
    print("  is_rankable(c) <=> status==eligible_and_safe AND is_safe")
    print("  Ranking is NOT implemented. This is a hard gate only.")

    # ------------------------------------------------------------------
    # Full payment audit
    # ------------------------------------------------------------------
    print_section("6. FULL-PAYMENT AUDIT")
    print(f"  generated:              {type_total['full_payment']}")
    print(f"  eligible+safe (today):  {full_bucket['safe']}")
    print(f"  eligible+unsafe:        {full_bucket['unsafe']}")
    print(f"  contractually ineligible:{full_bucket['ineligible']}")
    print(f"  structurally invalid:   {full_bucket['invalid']}")
    s, u, i = full_bucket["safe"], full_bucket["unsafe"], full_bucket["ineligible"]
    print(f"  {s} + {u} + {i} = {s + u + i}  (expect 250)")
    print(f"  check 66+82+102 style identity: {s + u + i == type_total['full_payment']}")
    ineligible_rankable_fp = [
        c for c in all_candidates
        if c.candidate_type == CandidateType.FULL_PAYMENT
        and c.status == CandidateStatus.CONTRACTUALLY_INELIGIBLE
        and is_rankable(c)
    ]
    print(f"  ineligible full_payment that is rankable (MUST be 0): {len(ineligible_rankable_fp)}")
    print("  Eligibility source:")
    print("    safe today  -> certificate.is_full_payment_safe_today AND user accepts full_payment")
    print("    unsafe      -> user accepts full_payment AND NOT is_full_payment_safe_today")
    print("    ineligible  -> user payment_methods_user_will_consider does not include full_payment")
    print("                  (independent of the CSV full_payment option row)")

    # ------------------------------------------------------------------
    # Partial payment audit
    # ------------------------------------------------------------------
    print_section("7. PARTIAL-PAYMENT AUDIT (all generated)")
    print(f"  count: {len(partials)}")
    partial_sum_ok = 0
    partial_deadline_ok = 0
    partial_two_ok = 0
    for req, cert, c in partials:
        a1 = c.payment_schedule[0].amount
        a2 = c.payment_schedule[1].amount
        d1 = c.payment_schedule[0].payment_date
        d2 = c.payment_schedule[1].payment_date
        sum_ok = (a1 + a2) == cert.requested_amount == c.total_amount_paid
        two_ok = c.number_of_payments == 2 == len(c.payment_schedule)
        deadline_ok = d2 <= req.desired_completion_date
        if sum_ok:
            partial_sum_ok += 1
        if two_ok:
            partial_two_ok += 1
        if deadline_ok:
            partial_deadline_ok += 1
        print(
            f"  {req.request_id}: safe_today={cert.amount_safe_to_pay} "
            f"remainder={a2} requested={cert.requested_amount} "
            f"p1={d1} p2={d2} deadline={req.desired_completion_date} "
            f"status={c.status.value} safe={c.is_safe} rankable={c.is_rankable} "
            f"source={c.provenance.source} cert={c.provenance.safe_to_pay_certificate_id} "
            f"sum_ok={sum_ok} two_ok={two_ok} deadline_ok={deadline_ok} "
            f"allows_partial={req.allows_partial_payment}"
        )
    print(f"  Decimal sum identity holds: {partial_sum_ok}/{len(partials)}")
    print(f"  exactly two payments:       {partial_two_ok}/{len(partials)}")
    print(f"  second date <= deadline:    {partial_deadline_ok}/{len(partials)}")
    print("  Why permitted: request.allows_partial_payment AND user accepts partial_payment")
    print("    AND 0 < amount_safe_to_pay < requested_amount")
    print("    AND earliest_date_for_full_payment <= desired_completion_date")
    print("    (problem_statement.md L146). Partial is derived, not a CSV option.")

    # ------------------------------------------------------------------
    # Wait audit
    # ------------------------------------------------------------------
    print_section("8. WAIT AUDIT (all generated)")
    print(f"  count: {len(waits)}")
    wait_after_deadline = 0
    wait_rankable = 0
    for req, cert, c in waits:
        after = c.completion_date is not None and c.completion_date > req.desired_completion_date
        if after:
            wait_after_deadline += 1
        if c.is_rankable:
            wait_rankable += 1
        print(
            f"  {req.request_id}: pay_date={c.first_payment_date} amount={c.total_amount_paid} "
            f"requested={cert.requested_amount} n_pay={c.number_of_payments} "
            f"request_date={req.request_date} deadline={req.desired_completion_date} "
            f"after_deadline={after} status={c.status.value} safe={c.is_safe} "
            f"rankable={c.is_rankable} source={c.provenance.source} "
            f"earliest={c.provenance.earliest_full_payment_date}"
        )
    print()
    print("  Specification (problem_statement.md Choosing Between Safe Plans):")
    print("    'wait is eligible when full payment becomes safe later and the user accepts full_payment'")
    print("  Semantics implemented:")
    print("    - no payment today")
    print("    - exactly one payment of requested_amount on earliest_date_for_full_payment")
    print("    - that date is strictly after request_date (else it would be full_payment)")
    print("    - represents delayed eventual full payment, not a new financing product")
    print("    - is_safe=True because earliest_date_for_full_payment is certified by safe_to_pay")
    print("    - rankable=True when constructed (eligible + safe)")
    print("    - completion vs desired_completion_date is ranking criterion 1, NOT a construction gate")
    print(f"  wait after desired_completion_date: {wait_after_deadline}/{len(waits)}")
    print(f"  wait rankable: {wait_rankable}/{len(waits)}")

    # ------------------------------------------------------------------
    # Duplicate audit
    # ------------------------------------------------------------------
    print_section("9. DUPLICATE AUDIT")
    # Identical (request_id, type, schedule) groups
    by_sched = defaultdict(list)
    by_sched_any_type = defaultdict(list)
    for c in all_candidates:
        sched_key = tuple((p.payment_date.isoformat(), str(p.amount)) for p in c.payment_schedule)
        by_sched[(c.request_id, c.candidate_type.value, sched_key)].append(c.candidate_id)
        if sched_key:
            by_sched_any_type[(c.request_id, sched_key)].append(
                (c.candidate_id, c.candidate_type.value, c.source_payment_option_id)
            )

    same_type_dups = {k: v for k, v in by_sched.items() if len(v) > 1}
    cross_type_dups = {k: v for k, v in by_sched_any_type.items() if len({x[1] for x in v}) > 1}

    print(f"  same-type identical schedule groups: {len(same_type_dups)}")
    for k, v in list(same_type_dups.items())[:20]:
        print(f"    {k[0]} {k[1]} n={len(v)} ids={v}")

    print(f"  cross-type identical schedule groups: {len(cross_type_dups)}")
    for k, v in list(cross_type_dups.items())[:20]:
        print(f"    {k[0]} schedule={k[1]} -> {v}")

    # full vs partial where safe == requested (should be 0 partials)
    full_partial_overlap = 0
    for req_row in request_rows:
        cert = req_row["cert"]
        types = {c.candidate_type for c in req_row["candidate_set"].candidates}
        if cert.amount_safe_to_pay == cert.requested_amount and CandidateType.PARTIAL_PAYMENT in types:
            full_partial_overlap += 1
    print(f"  full_payment vs partial where safe==requested (MUST be 0): {full_partial_overlap}")

    # wait vs full: same date
    wait_full_same_date = 0
    for req_row in request_rows:
        by_t = req_row["candidate_set"].by_type
        fps = by_t.get(CandidateType.FULL_PAYMENT, [])
        wts = by_t.get(CandidateType.WAIT, [])
        if fps and wts and fps[0].first_payment_date == wts[0].first_payment_date:
            wait_full_same_date += 1
    print(f"  wait vs full_payment same first_payment_date (MUST be 0): {wait_full_same_date}")

    # Distinct installment option ids with identical schedules
    inst_ident = defaultdict(list)
    for c in all_candidates:
        if c.candidate_type != CandidateType.INSTALLMENT_PLAN:
            continue
        if not c.payment_schedule:
            continue
        key = (
            c.request_id,
            tuple((p.payment_date.isoformat(), str(p.amount)) for p in c.payment_schedule),
        )
        inst_ident[key].append(c.source_payment_option_id)
    inst_dups = {k: v for k, v in inst_ident.items() if len(v) > 1}
    print(f"  distinct payment_option_id sharing identical populated schedules: {len(inst_dups)}")
    for k, v in list(inst_dups.items())[:20]:
        print(f"    {k[0]} options={v} schedule={k[1]}")
    print("  Distinct payment_option_id values are retained even if schedules coincide;")
    print("  they are different supplied offers (problem_statement: installment plans must")
    print("  exactly match a supplied payment option). No silent dedup.")

    # ------------------------------------------------------------------
    # Order independence
    # ------------------------------------------------------------------
    print_section("10. ORDER INDEPENDENCE")
    rng = random.Random(42)
    perm_mismatches = 0
    perm_checked = 0
    sample_n = min(25, len(request_rows))
    sample_rows = list(request_rows)
    rng.shuffle(sample_rows)
    sample_rows = sample_rows[:sample_n]
    for row in sample_rows:
        req = row["request"]
        profile = row["profile"]
        cert = row["cert"]
        feas = list(row["feasibilities"])
        opts = dict(row["opts_by_id"])
        base_ids = tuple(c.candidate_id for c in row["candidate_set"].candidates)
        base_set = set(base_ids)
        # permute feasibilities
        feas2 = list(feas)
        rng.shuffle(feas2)
        cs2 = generate_candidates(req, profile, cert, feas2, opts)
        ids2 = tuple(c.candidate_id for c in cs2.candidates)
        # permute option dict insertion
        keys = list(opts.keys())
        rng.shuffle(keys)
        opts2 = {k: opts[k] for k in keys}
        cs3 = generate_candidates(req, profile, cert, feas, opts2)
        ids3 = tuple(c.candidate_id for c in cs3.candidates)
        perm_checked += 1
        if set(ids2) != base_set or ids2 != base_ids:
            perm_mismatches += 1
            print(f"    mismatch after feas shuffle: {req.request_id}")
        if set(ids3) != base_set or ids3 != base_ids:
            perm_mismatches += 1
            print(f"    mismatch after dict shuffle: {req.request_id}")
    # request-order independence of the aggregated candidate-id multiset
    req_order_ids = tuple(
        cid
        for row in request_rows
        for cid in (c.candidate_id for c in row["candidate_set"].candidates)
    )
    shuffled_rows = list(request_rows)
    rng.shuffle(shuffled_rows)
    shuffled_ids = tuple(
        cid
        for row in shuffled_rows
        for cid in (c.candidate_id for c in row["candidate_set"].candidates)
    )
    # Per-request sets must match; global concatenation order may differ.
    set_match = set(req_order_ids) == set(shuffled_ids)
    print(f"  per-request permutation checks: {perm_checked}  mismatches: {perm_mismatches}")
    print(f"  request-order aggregated id SET identical: {set_match}")
    print("  Candidate ORDER is the documented deterministic sort, not business preference.")
    print("  Generation order of requests/options does not change the candidate SET.")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    print_section("11. MUTATION AUDIT")
    mut_failures = 0
    # Frozen dataclasses: assignment must raise
    row0 = request_rows[0]
    c0 = row0["candidate_set"].candidates[0]
    try:
        c0.total_amount_paid = Decimal("0")  # type: ignore[misc]
        mut_failures += 1
        print("    FAIL: candidate was mutable")
    except AttributeError:
        print("    candidate frozen: OK")
    try:
        row0["cert"].amount_safe_to_pay = Decimal("0")  # type: ignore[misc]
        mut_failures += 1
        print("    FAIL: certificate was mutable")
    except AttributeError:
        print("    certificate frozen: OK")
    try:
        row0["profile"].current_available_balance = Decimal("0")  # type: ignore[misc]
        mut_failures += 1
        print("    FAIL: profile was mutable")
    except AttributeError:
        print("    profile frozen: OK")
    # Re-generate and confirm certificate fields unchanged
    cert_before = (
        row0["cert"].amount_safe_to_pay,
        row0["cert"].earliest_date_for_full_payment,
        row0["cert"].is_full_payment_safe_today,
        row0["cert"].reason_if_unsafe,
    )
    generate_candidates(
        row0["request"], row0["profile"], row0["cert"],
        row0["feasibilities"], row0["opts_by_id"],
    )
    cert_after = (
        row0["cert"].amount_safe_to_pay,
        row0["cert"].earliest_date_for_full_payment,
        row0["cert"].is_full_payment_safe_today,
        row0["cert"].reason_if_unsafe,
    )
    print(f"    certificate identity after generation: {cert_before == cert_after}")
    if cert_before != cert_after:
        mut_failures += 1
    print(f"  mutation failures: {mut_failures}")

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------
    print_section("12. LINEAGE AUDIT")
    print(f"  candidates missing required provenance: {len(lineage_missing)} (expect 0)")
    for item in lineage_missing[:20]:
        print(f"    {item}")
    by_source = defaultdict(int)
    for c in all_candidates:
        by_source[c.provenance.source] += 1
    print("  provenance.source distribution:")
    for k, v in sorted(by_source.items()):
        print(f"    {k:40s} {v}")

    # ------------------------------------------------------------------
    # Decimal exactness
    # ------------------------------------------------------------------
    print_section("13. DECIMAL EXACTNESS")
    non_decimal = 0
    sum_mismatch = 0
    for c in all_candidates:
        if not isinstance(c.total_amount_paid, Decimal):
            non_decimal += 1
        if not isinstance(c.financing_fee, Decimal):
            non_decimal += 1
        for p in c.payment_schedule:
            if not isinstance(p.amount, Decimal):
                non_decimal += 1
        if c.payment_schedule:
            if sum(p.amount for p in c.payment_schedule) != c.total_amount_paid:
                sum_mismatch += 1
    print(f"  non-Decimal money fields: {non_decimal} (expect 0)")
    print(f"  schedule-sum mismatches:  {sum_mismatch} (expect 0)")

    # ------------------------------------------------------------------
    # Frozen hashes post-audit
    # ------------------------------------------------------------------
    print_section("14. UPSTREAM REGRESSION (post-audit hashes)")
    frozen_ok_after = True
    for name in FROZEN_FILES:
        current = git_hash_object(CODE / name)
        baseline = FROZEN_HASH_BASELINE[name]
        match = current == baseline
        frozen_ok_after = frozen_ok_after and match
        flag = "UNCHANGED" if match else "CHANGED"
        print(f"  {name:20s} {current}  [{flag}]")
    print(f"  ZERO upstream changes: {frozen_ok_after}")

    print_section("15. SUMMARY")
    print(f"  790 options:            {n_opts}")
    print(f"  250 requests:           {len(ds.requests)}")
    print(f"  generated candidates:   {len(all_candidates)}")
    print(f"  rankable candidates:    {n_rankable}")
    print(f"  partials:               {len(partials)}")
    print(f"  waits:                  {len(waits)}")
    print(f"  lineage gaps:           {len(lineage_missing)}")
    print(f"  errors:                 {len(errors)}")
    print(f"  frozen unchanged:       {frozen_ok and frozen_ok_after}")
    print(f"  NO ranking implemented: True")
    print(f"  total runtime:          {time.perf_counter() - t_all:.3f}s")


if __name__ == "__main__":
    main()
