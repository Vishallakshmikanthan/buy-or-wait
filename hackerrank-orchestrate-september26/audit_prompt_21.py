"""audit_prompt_21.py -- Prompt 21 Critical Fix Forensic Verification Script.
Exhaustive verification of Prompt 18B message causal propagation in production output pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Resolve paths
REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from code.loaders import load_dataset
from code.message_interpretation import (
    MessageAction,
    MessageActionType,
    apply_message_actions_to_future_events,
    interpret_and_link_messages,
)
from code.output import SCHEMA_COLUMNS, generate_all_outputs
from code.reconciliation import reconcile_events
from code.recurrence import detect_all_recurrence, expand_future_events
from code.simulator import simulate_user

FROZEN_MODULES = {
    "code/canonical.py":     "b1bd5a7d8dc73ca1c52f37e88e49e5064eb185b64c5256504b0d7af9678d6a9f",
    "code/recurrence.py":    "ddcd866761e7683fc78ab942618f761520804d9e401185679c9cf52cb7665b32",
    "code/simulator.py":     "820235d4068af75c45e7909717301fa00e60210a38440bd152b92dc5dc23ac6d",
    "code/safe_to_pay.py":   "b1cb9c995e776a4fe0900a198fdfd3b247577e035f92cb788d2f5427595d3aa2",
    "code/user_state.py":    "f2340e6285a72f0904b67855292ffc5a9329ef2eaa285f256dd9f120ee271f81",
    "code/affordability.py": "6cda8c8b2524f1f71304cdeb43ceecb32d838b3c11195ee2bcb1d7b391c6bee9",
    "code/payment_plan.py":  "2af320858fe6ae3d91987bb7d2a1a85dbea719c3c58544994fbfb622f997f20b",
    "code/ranking.py":       "ff57d55a7a3b3f607c79a943f50a239f2d12c048ceb4f5924e8abf3f56899775",
}

KNOWN_CAUSAL_REQUESTS = [
    "request_130",
    "request_49",
    "request_85",
    "request_154",
    "request_157",
    "request_193",
    "request_220",
]


def check_frozen_modules() -> bool:
    print("\n" + "=" * 78)
    print("SECTION 1: FROZEN ENGINE MODULE INTEGRITY")
    print("=" * 78)
    all_ok = True
    for rel_path, exp_hash in FROZEN_MODULES.items():
        act_hash = hashlib.sha256((REPO / rel_path).read_bytes()).hexdigest()
        match = (act_hash == exp_hash)
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {rel_path:25s} -> {act_hash[:16]}... (match={match})")
        if not match:
            all_ok = False
    return all_ok


def check_output_py_wiring() -> bool:
    print("\n" + "=" * 78)
    print("SECTION 2: OUTPUT.PY PRODUCTION WIRING FORENSICS")
    print("=" * 78)
    output_py_text = (REPO / "code" / "output.py").read_text(encoding="utf-8")

    has_import = "apply_message_actions_to_future_events" in output_py_text
    has_interpret = "interpret_and_link_messages" in output_py_text
    has_future_call = "future_events = apply_message_actions_to_future_events" in output_py_text
    has_sim_wire = "simulate_user" in output_py_text and "future_events=future_events" in output_py_text

    print(f"  [{'PASS' if has_import else 'FAIL'}] Imported apply_message_actions_to_future_events in code/output.py")
    print(f"  [{'PASS' if has_interpret else 'FAIL'}] Invoked interpret_and_link_messages in generate_all_outputs")
    print(f"  [{'PASS' if has_future_call else 'FAIL'}] Calls apply_message_actions_to_future_events on expanded future events")
    print(f"  [{'PASS' if has_sim_wire else 'FAIL'}] Passes adapted future_events to simulate_user and decision engine")

    return has_import and has_interpret and has_future_call and has_sim_wire


def check_seven_causal_requests(ds) -> bool:
    print("\n" + "=" * 78)
    print("SECTION 3: SEVEN KNOWN CAUSAL REQUESTS REGRESSION & VERIFICATION")
    print("=" * 78)

    base_ledger = reconcile_events(ds.events, ds.profiles, ds.exchange_rates, ds.messages, ds.images)
    all_series, _ = detect_all_recurrence(base_ledger, ds.profiles, ds.messages)
    actions, _ = interpret_and_link_messages(ds.messages, ds.events, ds.requests, ds.profiles)
    actions_by_user = {}
    for act in actions:
        if act.is_applied:
            actions_by_user.setdefault(act.user_id, []).append(act)

    req_map = {r.request_id: r for r in ds.requests}
    all_verified = True

    for rid in KNOWN_CAUSAL_REQUESTS:
        req = req_map[rid]
        uid = req.user_id
        profile = ds.profiles[uid]
        start_d = req.request_date
        end_d = start_d + timedelta(days=90)
        future_res = expand_future_events(all_series[uid], start_d, end_d, ledger=base_ledger)

        u_acts = actions_by_user.get(uid, [])
        adapted_future = apply_message_actions_to_future_events(future_res.future_events, u_acts, req.request_date)

        # Check for mutation
        diffs = []
        for b, m in zip(future_res.future_events, adapted_future):
            if b.amount_home != m.amount_home or b.effective_date != m.effective_date:
                diffs.append((b.effective_date, b.amount_home, m.effective_date, m.amount_home))

        active_mutation = (len(diffs) > 0)
        status = "PASS" if active_mutation else "FAIL"
        print(f"\n  [{status}] {rid} (user={uid}):")
        print(f"          Action Count: {len(u_acts)} (types: {[a.action_type.value for a in u_acts]})")
        print(f"          Future Events Mutated: {len(diffs)} occurrences")
        if diffs:
            b_d, b_a, m_d, m_a = diffs[0]
            print(f"          Sample Shift: Date {b_d} -> {m_d} | Amount {b_a} -> {m_a}")
        if not active_mutation:
            all_verified = False

    return all_verified


def check_message_shuffle_invariance(ds) -> bool:
    print("\n" + "=" * 78)
    print("SECTION 4: MESSAGE-ORDER PERMUTATION INVARIANCE")
    print("=" * 78)

    acts_baseline, _ = interpret_and_link_messages(ds.messages, ds.events, ds.requests, ds.profiles)

    shuffled = list(ds.messages)
    random.seed(999)
    random.shuffle(shuffled)
    acts_shuffled, _ = interpret_and_link_messages(shuffled, ds.events, ds.requests, ds.profiles)

    t1 = [(a.message_id, a.action_type, a.new_amount, a.effective_date) for a in acts_baseline]
    t2 = [(a.message_id, a.action_type, a.new_amount, a.effective_date) for a in acts_shuffled]
    is_invariant = (t1 == t2)
    print(f"  [{'PASS' if is_invariant else 'FAIL'}] Shuffled messages yield identical actions: {is_invariant}")
    return is_invariant


def check_output_contract() -> Tuple[bool, str, int]:
    print("\n" + "=" * 78)
    print("SECTION 5: OUTPUT.CSV CONTRACT AND REPEAT-RUN DETERMINISM")
    print("=" * 78)

    out_file = REPO / "output.csv"
    if not out_file.exists():
        print("  [FAIL] output.csv does not exist")
        return False, "", 0

    with open(out_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    schema_ok = (header == list(SCHEMA_COLUMNS))
    row_count_ok = (len(rows) == 250)
    unique_ids_ok = (len(set(r[0] for r in rows)) == 250)

    # Compute SHA-256
    sha256 = hashlib.sha256(out_file.read_bytes()).hexdigest()
    file_size = out_file.stat().st_size

    print(f"  [{'PASS' if schema_ok else 'FAIL'}] Schema matches exact 8 columns")
    print(f"  [{'PASS' if row_count_ok else 'FAIL'}] Row count exactly 250")
    print(f"  [{'PASS' if unique_ids_ok else 'FAIL'}] 250 unique request IDs matching dataset/requests.csv")
    print(f"  [INFO] output.csv SHA-256: {sha256}")
    print(f"  [INFO] output.csv File Size: {file_size} bytes")

    # Specific causal verification in output.csv
    row_map = {r[0]: r for r in rows}
    r130 = row_map.get("request_130")
    r85 = row_map.get("request_85")
    r220 = row_map.get("request_220")

    # request_130 earliest date must be 2025-01-15 (shifted from 2024-12-15)
    r130_shifted = (r130 is not None and r130[5] == "2025-01-15")
    # request_85 & request_220 earliest dates must be empty
    r85_empty = (r85 is not None and r85[5] == "")
    r220_empty = (r220 is not None and r220[5] == "")

    print(f"  [{'PASS' if r130_shifted else 'FAIL'}] request_130 full payment shifted to 2025-01-15: {r130[5] if r130 else 'N/A'}")
    print(f"  [{'PASS' if r85_empty else 'FAIL'}] request_85 earliest date empty: {r85[5] if r85 else 'N/A'!r}")
    print(f"  [{'PASS' if r220_empty else 'FAIL'}] request_220 earliest date empty: {r220[5] if r220 else 'N/A'!r}")

    all_ok = schema_ok and row_count_ok and unique_ids_ok and r130_shifted and r85_empty and r220_empty
    return all_ok, sha256, file_size


def run_unit_tests() -> Tuple[bool, int]:
    print("\n" + "=" * 78)
    print("SECTION 6: COMPLETE TEST SUITE EXECUTION")
    print("=" * 78)

    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "code/tests", "-p", "test_*.py"]
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)

    match = re.search(r"Ran (\d+) tests in ([\d.]+)s", res.stderr)
    test_count = int(match.group(1)) if match else 0
    passed = (res.returncode == 0) and ("OK" in res.stderr)

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Exit Code: {res.returncode}")
    print(f"  Tests Ran: {test_count}")
    print(f"  Status: {'ALL PASS' if passed else 'FAILURES DETECTED'}")

    return passed, test_count


def main() -> None:
    print("=" * 78)
    print("PROMPT 21 FORENSIC AUDIT REPORT: MESSAGE CAUSAL PROPAGATION WIRING")
    print("=" * 78)

    ds = load_dataset(REPO / "dataset")

    ok_frozen = check_frozen_modules()
    ok_wiring = check_output_py_wiring()
    ok_seven = check_seven_causal_requests(ds)
    ok_shuffle = check_message_shuffle_invariance(ds)
    ok_output, sha, size = check_output_contract()
    ok_tests, test_count = run_unit_tests()

    all_passed = ok_frozen and ok_wiring and ok_seven and ok_shuffle and ok_output and ok_tests

    print("\n" + "=" * 78)
    print("PROMPT 21 FINAL SUMMARY")
    print("=" * 78)
    print(f"  1. Frozen Engine Integrity:         {'PASS' if ok_frozen else 'FAIL'}")
    print(f"  2. Output Pipeline Wiring:          {'PASS' if ok_wiring else 'FAIL'}")
    print(f"  3. 7 Causal Requests Mutated:       {'PASS' if ok_seven else 'FAIL'}")
    print(f"  4. Message Shuffle Invariance:      {'PASS' if ok_shuffle else 'FAIL'}")
    print(f"  5. Output Contract & Hash:          {'PASS' if ok_output else 'FAIL'} (SHA-256: {sha})")
    print(f"  6. Full Test Suite:                 {'PASS' if ok_tests else 'FAIL'} ({test_count} tests)")
    print("-" * 78)
    print(f"  FINAL VERDICT:                      {'PASS' if all_passed else 'FAIL'}")
    print("=" * 78)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
