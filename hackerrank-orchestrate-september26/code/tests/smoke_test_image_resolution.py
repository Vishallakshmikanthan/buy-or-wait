"""End-to-end smoke test for real dataset image evidence resolution and invariant verification."""

import sys
from pathlib import Path
from decimal import Decimal

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.loaders import load_dataset
from code.reconciliation import reconcile_events
from code.image_resolution import extract_all_images, resolve_ledger_with_images


def run_image_resolution_smoke_test():
    dataset_dir = Path(__file__).resolve().parent.parent.parent / "dataset"
    print(f"Loading real dataset from: {dataset_dir}")
    ds = load_dataset(dataset_dir)

    print("Reconciling raw events into base canonical ledger...")
    base_ledger = reconcile_events(
        events=ds.events,
        profiles=ds.profiles,
        exchange_rates=ds.exchange_rates,
        messages=ds.messages,
        images=ds.images,
    )

    # --- BEFORE IMAGE RESOLUTION INVARIANTS ---
    before_raw_missing = [e for e in ds.events if e.has_missing_amount]
    before_unresolved = base_ledger.get_unresolved_events()
    before_unresolved_usable = [e for e in before_unresolved if e.is_numerically_usable_cash_event]
    before_usable_cash = base_ledger.get_cash_events()
    before_non_cash = base_ledger.get_non_cash_events()

    print("\n=== BEFORE IMAGE RESOLUTION INVARIANTS ===")
    print(f"Raw missing-amount events:                     {len(before_raw_missing)}")
    print(f"Unresolved canonical events:                   {len(before_unresolved)}")
    print(f"Unresolved events eligible for numeric cash:   {len(before_unresolved_usable)}")
    print(f"Total usable cash events:                      {len(before_usable_cash)}")
    print(f"Total non-cash / unusable events:              {len(before_non_cash)}")
    assert len(before_unresolved_usable) == 0, "Unresolved events must have zero usable cash eligibility!"

    # --- EXECUTE IMAGE EXTRACTION AND RESOLUTION ---
    print("\nExecuting contextual extraction on 16 images...")
    extraction_results = extract_all_images(dataset_dir)
    assert len(extraction_results) == 16, f"Expected 16 extraction results, got {len(extraction_results)}"

    print("Resolving canonical ledger with extracted image evidence...")
    resolved_ledger = resolve_ledger_with_images(
        ledger=base_ledger,
        image_results=extraction_results,
        exchange_rates=ds.exchange_rates,
    )

    # --- AFTER IMAGE RESOLUTION INVARIANTS ---
    after_resolved = [res for res in extraction_results.values() if res.is_resolved]
    after_still_unresolved = resolved_ledger.get_unresolved_events()
    after_unresolved_usable = [e for e in after_still_unresolved if e.is_numerically_usable_cash_event]
    after_usable_cash = resolved_ledger.get_cash_events()
    after_non_cash = resolved_ledger.get_non_cash_events()

    print("\n=== AFTER IMAGE RESOLUTION INVARIANTS ===")
    print(f"Successfully resolved image events:            {len(after_resolved)}")
    print(f"Still-unresolved events:                       {len(after_still_unresolved)}")
    print(f"Unresolved events eligible for numeric cash:   {len(after_unresolved_usable)}")
    print(f"Total usable cash events:                      {len(after_usable_cash)}")
    print(f"Total non-cash / unusable events:              {len(after_non_cash)}")
    print(f"Total canonical events in ledger:              {len(resolved_ledger.events)}")

    # Strict invariant assertions
    assert len(after_unresolved_usable) == 0, "Unresolved usable cash count must be exactly 0"
    assert len(after_resolved) == 16, "All 16 images must be resolved"
    assert len(after_still_unresolved) == 0, "All 16 missing events must be resolved"
    assert len(after_usable_cash) == len(before_usable_cash) + 16, "All 16 resolved events must become usable cash"
    assert len(resolved_ledger.events) == 25342, "Total events must equal 25342"
    assert len(after_usable_cash) + len(after_non_cash) == 25342, "Usable + NonCash must equal 25342"

    print("\n=== 16 REAL IMAGE RESOLUTION VERIFICATION TABLE ===")
    print(f"{'image_id':<10} | {'event_id':<12} | {'doc_type':<30} | {'method':<26} | {'amount':<12} | {'curr':<5} | {'status':<10}")
    print("-" * 115)
    for res in extraction_results.values():
        print(f"{res.image_id:<10} | {res.related_event_id:<12} | {res.document_type:<30} | {res.extraction_method:<26} | {str(res.extracted_amount):<12} | {res.extracted_currency:<5} | {'resolved' if res.is_resolved else 'unresolved':<10}")

    print("\n=== ALL REAL-DATASET INVARIANTS VERIFIED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_image_resolution_smoke_test()
