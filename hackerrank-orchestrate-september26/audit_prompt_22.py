#!/usr/bin/env python3
"""Comprehensive Forensic Audit for Prompt 22 — Final High-Priority Compliance Hardening.

Audits:
1. Authoritative OCR Requirement Analysis (AGENTS.md & problem_statement.md)
2. Image Extraction & Physical File Validation Status
3. requirements.txt Status & AST-based Zero-Dependency Verification
4. log.txt Git Compliance (Index vs Working Tree)
5. Submission Packaging Execution & Determinism
6. ZIP Manifest Verification & Fail-Closed Secret Scan
7. Full Test Suite Execution (python -m unittest discover)
8. Output Schema, Row Count & Bit-for-Bit Determinism
9. 7 Message Causal Regressions (Prompt 18B Pipeline)
10. 8 Frozen Financial Engine SHA-256 Integrity
11. Final Readiness Verdict
"""

from __future__ import annotations

import ast
import csv
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
import zipfile
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent

FROZEN_MODULES: Dict[str, str] = {
    "code/canonical.py": "b1bd5a7d8dc73ca1c52f37e88e49e5064eb185b64c5256504b0d7af9678d6a9f",
    "code/recurrence.py": "ddcd866761e7683fc78ab942618f761520804d9e401185679c9cf52cb7665b32",
    "code/simulator.py": "820235d4068af75c45e7909717301fa00e60210a38440bd152b92dc5dc23ac6d",
    "code/safe_to_pay.py": "b1cb9c995e776a4fe0900a198fdfd3b247577e035f92cb788d2f5427595d3aa2",
    "code/user_state.py": "f2340e6285a72f0904b67855292ffc5a9329ef2eaa285f256dd9f120ee271f81",
    "code/affordability.py": "6cda8c8b2524f1f71304cdeb43ceecb32d838b3c11195ee2bcb1d7b391c6bee9",
    "code/payment_plan.py": "2af320858fe6ae3d91987bb7d2a1a85dbea719c3c58544994fbfb622f997f20b",
    "code/ranking.py": "ff57d55a7a3b3f607c79a943f50a239f2d12c048ceb4f5924e8abf3f56899775",
}

EXPECTED_OUTPUT_SHA256 = "0af51bdca811b558198f81aee78e903d2f659881702f9226a5468f81f008f88d"


def run_cmd(args: List[str], cwd: Path = REPO_ROOT) -> Tuple[int, str, str]:
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def audit_section_1_authoritative_ocr() -> Tuple[bool, str, Dict]:
    agents_path = REPO_ROOT / "AGENTS.md"
    problem_path = REPO_ROOT / "problem_statement.md"

    agents_text = agents_path.read_text(encoding="utf-8")
    problem_text = problem_path.read_text(encoding="utf-8")

    agents_has_ocr = "ocr" in agents_text.lower()
    problem_has_ocr = "ocr" in problem_text.lower()

    # Check AGENTS §6.1 wording
    has_optional_clause = "optional supporting evidence" in agents_text
    has_blank_extract_clause = "When a financial event has a blank amount" in problem_text or "extract the amount from that image" in problem_text

    passed = (not agents_has_ocr) and (not problem_has_ocr) and has_optional_clause and has_blank_extract_clause
    details = {
        "agents_md_mentions_ocr": agents_has_ocr,
        "problem_statement_mentions_ocr": problem_has_ocr,
        "agents_md_optional_supporting_evidence": has_optional_clause,
        "problem_statement_requires_blank_amount_extraction": has_blank_extract_clause,
        "conclusion": "Runtime neural/Tesseract OCR is OPTIONAL / NOT REQUIRED. Image monetary extraction for blank financial events is REQUIRED.",
    }
    msg = "PASS: Specifications designate images as optional supporting evidence and require amount extraction without mandating runtime OCR." if passed else "FAIL: Specifications do not match expected clauses."
    return passed, msg, details


def audit_section_2_image_extraction_and_file_validation() -> Tuple[bool, str, Dict]:
    from code.image_resolution import extract_all_images, IMAGE_DOCUMENT_PROFILES

    # 1. Check all 16 physical images exist and have valid PNG signatures
    images_dir = REPO_ROOT / "dataset" / "media" / "images"
    physical_files_ok = True
    corrupted = []
    for i in range(1, 17):
        p = images_dir / f"image_{i:02d}.png"
        if not p.is_file():
            physical_files_ok = False
            corrupted.append(f"missing:{p.name}")
            continue
        data = p.read_bytes()
        if len(data) < 1000 or data[:8] != b"\x89PNG\r\n\x1a\n":
            physical_files_ok = False
            corrupted.append(f"bad_header:{p.name}")

    # 2. Check extract_all_images validates physical files
    extractions = extract_all_images(REPO_ROOT / "dataset")
    extractions_ok = len(extractions) == 16 and all(r.is_resolved and r.extracted_amount is not None for r in extractions.values())

    # 3. Check matching against dataset/images.csv
    csv_ok = True
    images_csv = REPO_ROOT / "dataset" / "images.csv"
    with open(images_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 16:
        csv_ok = False
    for r in rows:
        img_id = r["image_id"]
        rel_ev = r["related_event_id"]
        if img_id not in extractions or extractions[img_id].related_event_id != rel_ev:
            csv_ok = False

    passed = physical_files_ok and extractions_ok and csv_ok
    details = {
        "physical_images_count": 16,
        "physical_validation_passed": physical_files_ok,
        "extractions_count": len(extractions),
        "all_16_resolved": extractions_ok,
        "images_csv_match": csv_ok,
        "corrupted_or_missing": corrupted,
    }
    msg = "PASS: All 16 physical PNG image files validated on disk, perfectly matching images.csv and extraction profiles." if passed else f"FAIL: Image validation issues: {corrupted}"
    return passed, msg, details


def audit_section_3_requirements_txt() -> Tuple[bool, str, Dict]:
    req_file = REPO_ROOT / "requirements.txt"
    exists = req_file.is_file()
    text = req_file.read_text(encoding="utf-8") if exists else ""

    # AST-based scan of all python files in code/
    stdlib = sys.stdlib_module_names
    external_imports: Set[str] = set()
    for root, _, files in os.walk(REPO_ROOT / "code"):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as fh:
                    try:
                        tree = ast.parse(fh.read(), filename=path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for n in node.names:
                                    top = n.name.split(".")[0]
                                    if top not in stdlib and top != "code":
                                        external_imports.add(top)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    top = node.module.split(".")[0]
                                    if top not in stdlib and top != "code" and node.level == 0:
                                        external_imports.add(top)
                    except Exception as e:
                        external_imports.add(f"PARSE_ERROR:{f}")

    has_zero_ext = len(external_imports) == 0
    passed = exists and has_zero_ext
    details = {
        "requirements_txt_exists": exists,
        "external_imports_in_code": list(external_imports),
        "uses_pure_stdlib_only": has_zero_ext,
    }
    msg = "PASS: requirements.txt created documenting pure Python standard library; zero third-party package dependencies found." if passed else "FAIL: External dependencies found or requirements.txt missing."
    return passed, msg, details


def audit_section_4_log_txt_git_status() -> Tuple[bool, str, Dict]:
    # Check working-tree existence
    log_on_disk = (REPO_ROOT / "log.txt").is_file()

    # Check git ls-files log.txt
    ret, stdout, _ = run_cmd(["git", "ls-files", "log.txt"])
    git_tracked = bool(stdout.strip())

    # Check .gitignore contains log.txt
    gitignore_path = REPO_ROOT / ".gitignore"
    has_log_in_gitignore = "log.txt" in gitignore_path.read_text(encoding="utf-8") if gitignore_path.is_file() else False

    passed = log_on_disk and (not git_tracked) and has_log_in_gitignore
    details = {
        "log_txt_exists_on_disk": log_on_disk,
        "log_txt_tracked_in_git": git_tracked,
        "log_txt_in_gitignore": has_log_in_gitignore,
    }
    msg = "PASS: log.txt preserved on disk, absent from git index, and listed in .gitignore." if passed else "FAIL: log.txt git compliance check failed."
    return passed, msg, details


def audit_section_5_submission_packaging() -> Tuple[bool, str, Dict]:
    from package_submission import build_submission_zip

    zip_path = REPO_ROOT / "code.zip"
    summary1 = build_submission_zip(REPO_ROOT, zip_path)
    hash1 = summary1["zip_sha256"]

    # Re-run packaging to verify deterministic output
    summary2 = build_submission_zip(REPO_ROOT, zip_path)
    hash2 = summary2["zip_sha256"]

    deterministic = (hash1 == hash2)
    passed = deterministic and summary1["total_files"] > 0
    details = {
        "zip_path": str(zip_path),
        "file_count": summary1["total_files"],
        "uncompressed_bytes": summary1["total_uncompressed_bytes"],
        "compressed_bytes": summary1["zip_size_bytes"],
        "zip_sha256": hash1,
        "deterministic": deterministic,
    }
    msg = f"PASS: code.zip successfully generated ({summary1['total_files']} files, {summary1['zip_size_bytes']:,} bytes, deterministic SHA-256: {hash1[:16]}...)." if passed else "FAIL: Packaging is non-deterministic or failed."
    return passed, msg, details


def audit_section_6_zip_manifest_and_secret_scan() -> Tuple[bool, str, Dict]:
    zip_path = REPO_ROOT / "code.zip"
    if not zip_path.is_file():
        return False, "FAIL: code.zip not found", {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()

        # Check required files
        required = [
            "code/main.py",
            "evaluation/usage_report.md",
            "dataset/requests.csv",
            "dataset/images.csv",
            "dataset/media/images/image_01.png",
            "dataset/media/images/image_16.png",
            "README.md",
            "AGENTS.md",
            "problem_statement.md",
            "requirements.txt",
            ".env.example",
            ".gitignore",
        ]
        missing_required = [r for r in required if r not in namelist]

        # Check forbidden files
        forbidden = [
            ".env",
            "log.txt",
            "output.csv",
            "code.zip",
            "package_submission.py",
            "dataset/media/ocr_results.json",
            "code/evaluation/main.py",
            "code/evaluation/usage_report.md",
        ]
        found_forbidden = [f for f in forbidden if f in namelist]
        found_dirs = [n for n in namelist if "__pycache__" in n or n == ".git" or n.startswith(".git/") or "/.git/" in n or n.startswith("scratch/")]

        # Secret scan over all text files in ZIP
        secret_leaks = []
        key_pattern = re.compile(r"nvapi-[A-Za-z0-9_-]{20,}")
        safe_placeholders = {
            "nvapi-your-nvidia-api-key-here",
            "nvapi-mock-valid-key",
            "nvapi-mock-key",
            "nvapi-mock-robust",
            "nvapi-mock-inj",
            "nvapi-SUPER-SECRET-TOKEN-12345",
        }
        for name in namelist:
            if name.endswith((".py", ".md", ".txt", ".csv", ".json", ".example", ".gitignore")):
                content = zf.read(name).decode("utf-8", errors="ignore")
                matches = key_pattern.findall(content)
                real_leaks = [m for m in matches if m not in safe_placeholders and "mock" not in m.lower() and "dummy" not in m.lower()]
                if real_leaks:
                    secret_leaks.append((name, real_leaks))

    passed = (len(missing_required) == 0) and (len(found_forbidden) == 0) and (len(found_dirs) == 0) and (len(secret_leaks) == 0)
    details = {
        "missing_required": missing_required,
        "found_forbidden": found_forbidden,
        "found_forbidden_dirs": found_dirs,
        "secret_leaks": secret_leaks,
        "total_zip_entries": len(namelist),
    }
    msg = "PASS: ZIP manifest strictly verified: all required files present, zero forbidden artifacts, zero secrets detected." if passed else f"FAIL: Manifest/secret scan failed: missing={missing_required}, forbidden={found_forbidden}, leaks={secret_leaks}"
    return passed, msg, details


def audit_section_7_full_tests() -> Tuple[bool, str, Dict]:
    # Run unittest suite directly
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(REPO_ROOT / "code" / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
    result = runner.run(suite)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = (failures == 0 and errors == 0 and total_tests >= 613)

    details = {
        "tests_run": total_tests,
        "failures": failures,
        "errors": errors,
        "all_passed": passed,
    }
    msg = f"PASS: All {total_tests} unit tests passed (0 failures, 0 errors)." if passed else f"FAIL: Unit tests failed: {failures} failures, {errors} errors out of {total_tests} tests."
    return passed, msg, details


def audit_section_8_output_contract_and_determinism() -> Tuple[bool, str, Dict]:
    output_path = REPO_ROOT / "output.csv"
    if not output_path.is_file():
        return False, "FAIL: output.csv not found", {}

    data = output_path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    size = len(data)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    row_count = len(lines) - 1

    expected_cols = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]
    reader = csv.DictReader(lines)
    cols_match = reader.fieldnames == expected_cols

    from code.output import SCHEMA_COLUMNS
    from code.loaders import load_dataset
    from datetime import date
    from decimal import Decimal

    validation_errors = []
    ds = load_dataset(REPO_ROOT / "dataset")
    req_map = {r.request_id: r for r in ds.requests}
    reader = list(csv.DictReader(lines))

    ALLOWED_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
    ALLOWED_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}

    for idx, row in enumerate(reader, start=1):
        req_id = row["request_id"]
        req = req_map.get(req_id)
        if not req:
            validation_errors.append(f"Row {idx}: missing request metadata for {req_id}")
            continue

        # Bounds: 0 <= safe <= requested_amount
        safe_amt = Decimal(row["amount_safe_to_pay"])
        if not (Decimal("0") <= safe_amt <= req.requested_amount):
            validation_errors.append(f"Row {idx} ({req_id}): safe amount {safe_amt} out of bounds [0, {req.requested_amount}]")

        # Allowed values
        if row["affordability_status"] not in ALLOWED_STATUSES:
            validation_errors.append(f"Row {idx} ({req_id}): invalid status {row['affordability_status']}")
        if row["recommended_payment_method"] not in ALLOWED_METHODS:
            validation_errors.append(f"Row {idx} ({req_id}): invalid method {row['recommended_payment_method']}")

        # affordable_now must have earliest == request_date
        if row["affordability_status"] == "affordable_now":
            if row["earliest_date_for_full_payment"] != str(req.request_date):
                validation_errors.append(f"Row {idx} ({req_id}): affordable_now earliest {row['earliest_date_for_full_payment']} != request_date {req.request_date}")

        # partial_payment contract
        if row["recommended_payment_method"] == "partial_payment":
            if row["affordability_status"] != "affordable_with_plan":
                validation_errors.append(f"Row {idx} ({req_id}): partial_payment must be affordable_with_plan")
            plan_parts = row["payment_plan"].split("|")
            if len(plan_parts) != 2:
                validation_errors.append(f"Row {idx} ({req_id}): partial_payment plan must have exactly 2 parts")

        if not row["decision_explanation"].strip():
            validation_errors.append(f"Row {idx} ({req_id}): empty decision_explanation")

    hash_match = (sha256 == EXPECTED_OUTPUT_SHA256)
    passed = (row_count == 250) and cols_match and hash_match and (len(validation_errors) == 0)
    details = {
        "row_count": row_count,
        "file_size": size,
        "sha256": sha256,
        "expected_sha256": EXPECTED_OUTPUT_SHA256,
        "hash_match": hash_match,
        "schema_valid": cols_match,
        "validation_errors": validation_errors,
    }
    msg = f"PASS: output.csv perfectly valid (250 rows, 8 columns, exact SHA-256: {sha256[:16]}...)." if passed else f"FAIL: output.csv contract check failed: errors={validation_errors}"
    return passed, msg, details


def audit_section_9_causal_regressions() -> Tuple[bool, str, Dict]:
    from datetime import timedelta
    from code.loaders import load_dataset
    from code.reconciliation import reconcile_events
    from code.recurrence import detect_all_recurrence, expand_future_events
    from code.message_interpretation import interpret_and_link_messages, apply_message_actions_to_future_events

    output_path = REPO_ROOT / "output.csv"
    with open(output_path, "r", encoding="utf-8") as f:
        rows = {r["request_id"]: r for r in csv.DictReader(f)}

    causal_targets = {
        "request_49": {"status": "affordable_with_plan", "method": "installments", "earliest": "2024-05-15"},
        "request_85": {"status": "not_affordable", "method": "not_recommended", "earliest": ""},
        "request_130": {"status": "affordable_later", "method": "wait", "earliest": "2025-01-15"},
        "request_154": {"status": "affordable_with_plan", "method": "installments", "earliest": "2024-12-04"},
        "request_157": {"status": "affordable_later", "method": "wait", "earliest": "2024-04-15"},
        "request_193": {"status": "affordable_with_plan", "method": "installments", "earliest": "2024-04-15"},
        "request_220": {"status": "not_affordable", "method": "not_recommended", "earliest": ""},
    }

    ds = load_dataset(REPO_ROOT / "dataset")
    base_ledger = reconcile_events(ds.events, ds.profiles, ds.exchange_rates, ds.messages, ds.images)
    all_series, _ = detect_all_recurrence(base_ledger, ds.profiles, ds.messages)
    actions, _ = interpret_and_link_messages(ds.messages, ds.events, ds.requests, ds.profiles)
    actions_by_user = {}
    for act in actions:
        if act.is_applied:
            actions_by_user.setdefault(act.user_id, []).append(act)

    mismatches = []
    mutations_count = 0
    req_dict = {r.request_id: r for r in ds.requests}

    for req_id, expected in causal_targets.items():
        if req_id not in rows:
            mismatches.append(f"{req_id} missing from output.csv")
            continue
        row = rows[req_id]
        if row["earliest_date_for_full_payment"] != expected["earliest"]:
            mismatches.append(f"{req_id}: expected earliest='{expected['earliest']}', got '{row['earliest_date_for_full_payment']}'")
        if row["affordability_status"] != expected["status"]:
            mismatches.append(f"{req_id}: expected status='{expected['status']}', got '{row['affordability_status']}'")
        if row["recommended_payment_method"] != expected["method"]:
            mismatches.append(f"{req_id}: expected method='{expected['method']}', got '{row['recommended_payment_method']}'")

        # Verify active future event mutation
        req = req_dict[req_id]
        uid = req.user_id
        start_d = req.request_date
        end_d = start_d + timedelta(days=90)
        future_res = expand_future_events(all_series[uid], start_d, end_d, ledger=base_ledger)
        u_acts = actions_by_user.get(uid, [])
        adapted_future = apply_message_actions_to_future_events(future_res.future_events, u_acts, req.request_date)
        diffs = [1 for b, m in zip(future_res.future_events, adapted_future) if b.amount_home != m.amount_home or b.effective_date != m.effective_date]
        if diffs:
            mutations_count += 1

    passed = (len(mismatches) == 0) and (mutations_count == len(causal_targets))
    details = {
        "verified_requests": list(causal_targets.keys()),
        "mismatches": mismatches,
        "active_mutations_count": mutations_count,
    }
    msg = "PASS: All 7 message causal mutations active and verified in output.csv." if passed else f"FAIL: Causal regressions detected: {mismatches}"
    return passed, msg, details


def audit_section_10_frozen_engine_hashes() -> Tuple[bool, str, Dict]:
    mismatches = []
    current_hashes = {}
    for mod_path, exp_sha in FROZEN_MODULES.items():
        full_p = REPO_ROOT / mod_path
        if not full_p.is_file():
            mismatches.append(f"{mod_path} is missing")
            continue
        cur_sha = hashlib.sha256(full_p.read_bytes()).hexdigest()
        current_hashes[mod_path] = cur_sha
        if cur_sha != exp_sha:
            mismatches.append(f"{mod_path}: expected {exp_sha}, got {cur_sha}")

    passed = len(mismatches) == 0
    details = {
        "current_hashes": current_hashes,
        "mismatches": mismatches,
    }
    msg = "PASS: All 8 frozen financial engine modules 100% bit-for-bit SHA-256 intact." if passed else f"FAIL: Frozen hash violations: {mismatches}"
    return passed, msg, details


def main():
    print("================================================================================")
    print(" AUDIT PROMPT 22: FINAL HIGH-PRIORITY COMPLIANCE HARDENING FORENSIC AUDIT       ")
    print("================================================================================")

    sections = [
        ("1. Authoritative OCR Requirement", audit_section_1_authoritative_ocr),
        ("2. Image Extraction & File Validation", audit_section_2_image_extraction_and_file_validation),
        ("3. requirements.txt & Zero Dependencies", audit_section_3_requirements_txt),
        ("4. log.txt Git Compliance", audit_section_4_log_txt_git_status),
        ("5. Submission Packaging Determinism", audit_section_5_submission_packaging),
        ("6. ZIP Manifest & Secret Scan", audit_section_6_zip_manifest_and_secret_scan),
        ("7. Full Test Suite Execution", audit_section_7_full_tests),
        ("8. Output Contract & Determinism", audit_section_8_output_contract_and_determinism),
        ("9. Message Causal Regressions (7 Targets)", audit_section_9_causal_regressions),
        ("10. Frozen Engine SHA-256 Hashes", audit_section_10_frozen_engine_hashes),
    ]

    all_passed = True
    results_summary = []

    for name, audit_fn in sections:
        print(f"\n--- Running Section: {name} ---")
        try:
            ok, msg, details = audit_fn()
            print(f"Result: {msg}")
            if details:
                for k, v in details.items():
                    if k != "current_hashes":
                        print(f"  {k}: {v}")
            if not ok:
                all_passed = False
            results_summary.append((name, ok, msg))
        except Exception as e:
            print(f"ERROR executing {name}: {e}")
            all_passed = False
            results_summary.append((name, False, f"EXCEPTION: {e}"))

    print("\n================================================================================")
    print(" FINAL VERDICT SUMMARY                                                          ")
    print("================================================================================")
    for name, ok, msg in results_summary:
        status_str = "[PASS]" if ok else "[FAIL]"
        print(f"{status_str} {name}")

    verdict_str = "PROMPT 22: PASS" if all_passed else "PROMPT 22: FAIL"
    print("================================================================================")
    print(f" OVERALL AUDIT VERDICT: {verdict_str}")
    print("================================================================================")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
