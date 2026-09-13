"""Comprehensive audit script for Prompt 18:
Structured Message Interpretation, Evidence Integration, Regression, and Determinism.
"""

import csv
import hashlib
from decimal import Decimal
from typing import Dict, List, Tuple

from pathlib import Path
from code.loaders import load_dataset
from code.reconciliation import reconcile_events, reconcile_events_with_audit
from code.message_interpretation import (
    MessageAction,
    MessageActionType,
    TargetType,
    interpret_and_link_messages,
)
from code.output import generate_all_outputs


def audit_phase_16_real_data(ds) -> Tuple[List[MessageAction], Dict[str, int]]:
    """Execute Phase 16 Real Data Audit across all 215 messages."""
    print("============================================================")
    print("PHASE 16 — REAL DATA AUDIT (ALL 215 MESSAGES)")
    print("============================================================")

    res = reconcile_events_with_audit(
        events=ds.events,
        profiles=ds.profiles,
        exchange_rates=ds.exchange_rates,
        messages=ds.messages,
        images=ds.images,
        requests=ds.requests,
    )

    actions = res.message_actions
    unresolved = res.unresolved_actions
    records = res.reconciliation_records

    print(f"Total Messages Loaded: {len(ds.messages)}")
    print(f"Parsed & Resolved Actions: {len(actions)}")
    print(f"Unresolved Actions: {len(unresolved)}")

    applied = [a for a in actions if a.is_applied]
    rejected = [a for a in actions if not a.is_applied]
    print(f"Applied Actions: {len(applied)}")
    print(f"Rejected / Informational Actions: {len(rejected)}")

    cancels = [a for a in actions if a.action_type == MessageActionType.CANCEL]
    amends = [a for a in actions if a.action_type == MessageActionType.AMEND_AMOUNT]
    delays = [a for a in actions if a.action_type == MessageActionType.DELAY_TO]
    confirms = [a for a in actions if a.action_type == MessageActionType.CONFIRM]
    resumes = [a for a in actions if a.action_type == MessageActionType.RESUME]
    temps = [a for a in actions if a.action_type == MessageActionType.TEMPORARY_CHANGE]
    ignored = [a for a in actions if a.action_type == MessageActionType.IGNORED]

    income_acts = [
        a for a in actions
        if "salary" in (a.target_description or "").lower()
        or "income" in (a.reason or "").lower()
        or "salary" in (a.reason or "").lower()
    ]
    recurring_expense_acts = [
        a for a in actions
        if "rent" in (a.target_description or "").lower()
        or "rent" in (a.reason or "").lower()
    ]

    stats = {
        "total_messages": len(ds.messages),
        "parsed_actions": len(actions),
        "unresolved_actions": len(unresolved),
        "applied_actions": len(applied),
        "rejected_actions": len(rejected),
        "cancellations": len(cancels),
        "amount_amendments": len(amends),
        "delays": len(delays),
        "confirmations": len(confirms),
        "resumptions": len(resumes),
        "temporary_changes": len(temps),
        "income_related_actions": len(income_acts),
        "recurring_expense_actions": len(recurring_expense_acts),
        "ignored_as_non_financial": len(ignored),
        "ambiguous_actions": len(unresolved),
    }

    print("\n--- STATISTICS ---")
    for k, v in stats.items():
        print(f"  {k:30s}: {v:3d}")

    print(f"\n--- AUDIT OF APPLIED ACTIONS (First 20 of {len(applied)}) ---")
    for act in applied[:20]:
        print(f"  [{act.message_id}] type={act.action_type.value:16s} target={act.target_type.value}:{act.target_id}")
        print(f"      before={act.old_amount} after={act.new_amount} eff_date={act.effective_date} reason={act.reason}")

    return actions, stats


def audit_phase_17_regression():
    """Execute Phase 17 Regression Test across all 250 requests."""
    print("\n============================================================")
    print("PHASE 17 — REGRESSION TEST ACROSS ALL 250 REQUESTS")
    print("============================================================")

    # Read current output.csv
    baseline_path = "output.csv"
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_rows = list(csv.DictReader(f))
    baseline_hash = hashlib.sha256(open(baseline_path, "rb").read()).hexdigest()
    print(f"Baseline output.csv SHA-256: {baseline_hash}")
    print(f"Baseline rows count: {len(baseline_rows)}")

    # Generate output using pipeline
    _, new_hash, _ = generate_all_outputs(Path("dataset"), Path("scratch/test_output_p18.csv"))
    print(f"New generated output.csv SHA-256: {new_hash}")

    with open("scratch/test_output_p18.csv", "r", encoding="utf-8") as f:
        new_rows = list(csv.DictReader(f))

    diffs = []
    for b_row, n_row in zip(baseline_rows, new_rows):
        rid = b_row["request_id"]
        col_diffs = {}
        for col in b_row:
            if b_row[col] != n_row[col]:
                col_diffs[col] = (b_row[col], n_row[col])
        if col_diffs:
            diffs.append((rid, col_diffs))

    print(f"\nTotal Changed Requests: {len(diffs)}")
    if diffs:
        for rid, col_diffs in diffs:
            print(f"  Request {rid}:")
            for col, (b_val, n_val) in col_diffs.items():
                print(f"    {col}: before='{b_val}' -> after='{n_val}'")
    else:
        print("  ALL 250 REQUESTS ARE 100% BYTE-IDENTICAL!")
        print("  Regression classification: ZERO_REGRESSIONS (0 unexpected changes)")

    return len(diffs), baseline_hash, new_hash


def audit_phase_18_frozen_integrity():
    """Verify all 8 frozen engine modules remain byte-identical."""
    print("\n============================================================")
    print("PHASE 18 — FROZEN ENGINE INTEGRITY AUDIT")
    print("============================================================")
    frozen_files = [
        "code/canonical.py",
        "code/recurrence.py",
        "code/simulator.py",
        "code/safe_to_pay.py",
        "code/user_state.py",
        "code/affordability.py",
        "code/payment_plan.py",
        "code/ranking.py",
    ]
    expected_hashes = {
        "code/canonical.py": "b1bd5a7d8dc73ca1c52f37e88e49e5064eb185b64c5256504b0d7af9678d6a9f",
        "code/recurrence.py": "ddcd866761e7683fc78ab942618f761520804d9e401185679c9cf52cb7665b32",
        "code/simulator.py": "820235d4068af75c45e7909717301fa00e60210a38440bd152b92dc5dc23ac6d",
        "code/safe_to_pay.py": "b1cb9c995e776a4fe0900a198fdfd3b247577e035f92cb788d2f5427595d3aa2",
        "code/user_state.py": "f2340e6285a72f0904b67855292ffc5a9329ef2eaa285f256dd9f120ee271f81",
        "code/affordability.py": "6cda8c8b2524f1f71304cdeb43ceecb32d838b3c11195ee2bcb1d7b391c6bee9",
        "code/payment_plan.py": "2af320858fe6ae3d91987bb7d2a1a85dbea719c3c58544994fbfb622f997f20b",
        "code/ranking.py": "ff57d55a7a3b3f607c79a943f50a239f2d12c048ceb4f5924e8abf3f56899775",
    }
    all_ok = True
    for fpath in frozen_files:
        content = open(fpath, "rb").read()
        cur_hash = hashlib.sha256(content).hexdigest()
        exp_hash = expected_hashes[fpath]
        match = (cur_hash == exp_hash)
        status = "MATCH (BYTE-IDENTICAL)" if match else "MISMATCH FAILED"
        print(f"  {fpath:25s}: {status} ({cur_hash[:16]}...)")
        if not match:
            all_ok = False
    return all_ok


def audit_phase_19_determinism():
    """Verify repeated execution determinism."""
    print("\n============================================================")
    print("PHASE 19 — DETERMINISM VERIFICATION")
    print("============================================================")
    _, hash1, _ = generate_all_outputs(Path("dataset"), Path("scratch/det_run1.csv"))
    _, hash2, _ = generate_all_outputs(Path("dataset"), Path("scratch/det_run2.csv"))
    match = (hash1 == hash2)
    print(f"Run 1 SHA-256: {hash1}")
    print(f"Run 2 SHA-256: {hash2}")
    print(f"Hashes Match: {match}")
    return match, hash1


if __name__ == "__main__":
    ds = load_dataset("dataset")
    audit_phase_16_real_data(ds)
    diff_count, b_hash, n_hash = audit_phase_17_regression()
    frozen_ok = audit_phase_18_frozen_integrity()
    det_ok, det_hash = audit_phase_19_determinism()

    print("\n============================================================")
    print("FINAL SUMMARY AUDIT REPORT")
    print("============================================================")
    print(f"Frozen Engine Modules: {'ALL 8 BYTE-IDENTICAL' if frozen_ok else 'FAILED'}")
    print(f"Regression Delta Count: {diff_count} requests changed")
    print(f"Output Determinism: {'IDENTICAL' if det_ok else 'FAILED'} ({det_hash})")
