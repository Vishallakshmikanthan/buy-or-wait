"""Smoke test to ingest real dataset files and verify integrity and counts."""

from pathlib import Path
from code.loaders import load_dataset


def run_smoke_test():
    dataset_dir = Path(__file__).resolve().parent.parent.parent / "dataset"
    print(f"Loading dataset from: {dataset_dir}")

    ds = load_dataset(dataset_dir)

    missing_amount_events = [ev for ev in ds.events if ev.has_missing_amount]

    print("=== SMOKE TEST INGESTION REPORT ===")
    print(f"Requests loaded:          {len(ds.requests)}")
    print(f"Profiles loaded:          {len(ds.profiles)}")
    print(f"Financial events loaded:  {len(ds.events)}")
    print(f"Payment options loaded:   {len(ds.payment_options)}")
    print(f"Exchange rates loaded:    {len(ds.exchange_rates)}")
    print(f"Messages loaded:          {len(ds.messages)}")
    print(f"Image records loaded:     {len(ds.images)}")
    print(f"Events with missing amt:  {len(missing_amount_events)}")

    # Verify that missing amount events match images.csv
    missing_event_ids = {ev.event_id for ev in missing_amount_events}
    image_event_ids = {img.related_event_id for img in ds.images}
    assert missing_event_ids == image_event_ids, (
        f"Mismatch between missing amount events ({missing_event_ids}) "
        f"and images.csv related_event_ids ({image_event_ids})"
    )
    print("Verification passed: Missing amount events match images.csv 1:1.")
    print("=== SMOKE TEST COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_smoke_test()
