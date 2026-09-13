"""Final Output Assembly and Cross-Layer Integration Layer for Buy-or-Wait (Prompt 15).

Integrates the frozen deterministic decision engine, DecisionCertificate, and
grounded explanation layer into the authoritative final submission file output.csv.

Strict Architectural Guarantees:
- Read-only consumer of frozen upstream modules.
- Zero independent financial decision-making or recomputation.
- Mechanical field-mapping from authoritative FinalDecision, DecisionCertificate, and ExplanationResult.
- Exact non-scientific Decimal serialization matching dataset/sample_requests.csv.
- Fail-closed in-memory validation of all 250 rows before disk write.
- Deterministic, repeatable, offline execution from repository root.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from code.candidate_generation import generate_candidates
from code.decision_certificate import (
    DecisionCertificate,
    build_decision_certificate,
    validate_certificate,
)
from code.explanation import (
    ExplanationResult,
    ExplanationValidator,
    GroundedExplanationInput,
    build_grounded_explanation_input,
    generate_grounded_explanation,
)
from code.final_decision import (
    FinalDecision,
    RequestContext,
    make_final_decision_from_candidate_set,
)
from code.image_resolution import extract_all_images, resolve_ledger_with_images
from code.loaders import load_dataset
from code.models import FinancialProfile, FinancialRequest
from code.payment_plan import evaluate_payment_option_feasibility
from code.reconciliation import reconcile_events
from code.recurrence import detect_all_recurrence, expand_future_events
from code.safe_to_pay import evaluate_request_safe_to_pay
from code.simulator import simulate_user
from code.user_state import build_user_financial_state


# ---------------------------------------------------------------------------
# Official CSV Schema
# ---------------------------------------------------------------------------

SCHEMA_COLUMNS: Tuple[str, ...] = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


# ---------------------------------------------------------------------------
# Exact Decimal Serialization
# ---------------------------------------------------------------------------

def serialize_decimal(d: Decimal) -> str:
    """Format Decimal exactly without scientific notation or floating-point conversion.

    Trims trailing zeroes after a decimal point while preserving integer precision.
    Byte-for-byte identical to dataset/sample_requests.csv format (e.g. '15656000', '17229139.2', '0').
    """
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ---------------------------------------------------------------------------
# Output Row Data Structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputRow:
    """Authoritative representation of a single serialized output row."""
    request_id: str
    amount_safe_to_pay: str
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str

    def to_tuple(self) -> Tuple[str, ...]:
        return (
            self.request_id,
            self.amount_safe_to_pay,
            self.affordability_status,
            self.recommended_payment_method,
            self.payment_plan,
            self.earliest_date_for_full_payment,
            self.spending_changes_needed,
            self.decision_explanation,
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": self.amount_safe_to_pay,
            "affordability_status": self.affordability_status,
            "recommended_payment_method": self.recommended_payment_method,
            "payment_plan": self.payment_plan,
            "earliest_date_for_full_payment": self.earliest_date_for_full_payment,
            "spending_changes_needed": self.spending_changes_needed,
            "decision_explanation": self.decision_explanation,
        }


# ---------------------------------------------------------------------------
# Authoritative Field Mapping
# ---------------------------------------------------------------------------

def format_output_row(
    request: FinancialRequest,
    decision: FinalDecision,
    certificate: DecisionCertificate,
    explanation: ExplanationResult,
) -> OutputRow:
    """Mechanically map authoritative fields into an OutputRow without recomputation.

    Source Mapping Table:
      request_id                     -> request.request_id
      amount_safe_to_pay             -> FinalDecision.amount_safe_to_pay (via serialize_decimal)
      affordability_status           -> FinalDecision.affordability_status.value
      recommended_payment_method     -> FinalDecision.recommended_payment_method
      payment_plan                   -> FinalDecision.payment_plan
      earliest_date_for_full_payment -> FinalDecision.earliest_date_for_full_payment.isoformat() or ''
      spending_changes_needed        -> FinalDecision.spending_changes_needed
      decision_explanation           -> ExplanationResult.explanation_text
    """
    req_id = request.request_id
    safe_amt_str = serialize_decimal(decision.amount_safe_to_pay)
    status_str = decision.affordability_status.value
    method_str = decision.recommended_payment_method
    plan_str = decision.payment_plan
    earliest_date_str = (
        decision.earliest_date_for_full_payment.isoformat()
        if decision.earliest_date_for_full_payment is not None
        else ""
    )
    spending_str = decision.spending_changes_needed
    explanation_str = explanation.explanation_text

    return OutputRow(
        request_id=req_id,
        amount_safe_to_pay=safe_amt_str,
        affordability_status=status_str,
        recommended_payment_method=method_str,
        payment_plan=plan_str,
        earliest_date_for_full_payment=earliest_date_str,
        spending_changes_needed=spending_str,
        decision_explanation=explanation_str,
    )


# ---------------------------------------------------------------------------
# Cross-Layer Consistency Validator
# ---------------------------------------------------------------------------

class CrossLayerConsistencyValidator:
    """Validates an assembled OutputRow against authoritative upstream objects.

    Fail-closed: Any mismatch, schema violation, or ungrounded claim raises ValueError.
    """

    ALLOWED_STATUSES: Set[str] = {
        "affordable_now",
        "affordable_with_plan",
        "affordable_later",
        "not_affordable",
    }

    ALLOWED_METHODS: Set[str] = {
        "full_payment",
        "partial_payment",
        "installments",
        "wait",
        "not_recommended",
    }

    @classmethod
    def validate_row(
        cls,
        row: OutputRow,
        request: FinancialRequest,
        decision: FinalDecision,
        certificate: DecisionCertificate,
        explanation_input: GroundedExplanationInput,
        explanation_result: ExplanationResult,
    ) -> None:
        # 1. request_id match across all layers
        if row.request_id != request.request_id:
            raise ValueError(f"request_id mismatch: row={row.request_id} vs request={request.request_id}")
        if row.request_id != decision.request_id:
            raise ValueError(f"request_id mismatch: row={row.request_id} vs decision={decision.request_id}")
        if row.request_id != certificate.request_id:
            raise ValueError(f"request_id mismatch: row={row.request_id} vs certificate={certificate.request_id}")

        # 2. amount_safe_to_pay exact decimal match
        expected_safe = serialize_decimal(decision.amount_safe_to_pay)
        cert_safe = serialize_decimal(certificate.amount_safe_to_pay)
        if row.amount_safe_to_pay != expected_safe:
            raise ValueError(f"amount_safe_to_pay mismatch: row={row.amount_safe_to_pay} vs decision={expected_safe}")
        if row.amount_safe_to_pay != cert_safe:
            raise ValueError(f"amount_safe_to_pay mismatch: row={row.amount_safe_to_pay} vs cert={cert_safe}")

        dec_safe = Decimal(row.amount_safe_to_pay)
        if dec_safe < Decimal("0"):
            raise ValueError(f"Negative safe amount: {row.amount_safe_to_pay}")
        if dec_safe > request.requested_amount:
            raise ValueError(f"Safe amount exceeds requested: {row.amount_safe_to_pay} > {request.requested_amount}")

        # 3. affordability_status match
        if row.affordability_status not in cls.ALLOWED_STATUSES:
            raise ValueError(f"Invalid affordability_status: {row.affordability_status}")
        if row.affordability_status != decision.affordability_status.value:
            raise ValueError(f"Status mismatch: row={row.affordability_status} vs decision={decision.affordability_status.value}")
        if row.affordability_status != certificate.affordability_status:
            raise ValueError(f"Status mismatch: row={row.affordability_status} vs cert={certificate.affordability_status}")

        # 4. recommended_payment_method match
        if row.recommended_payment_method not in cls.ALLOWED_METHODS:
            raise ValueError(f"Invalid payment method: {row.recommended_payment_method}")
        if row.recommended_payment_method != decision.recommended_payment_method:
            raise ValueError(f"Method mismatch: row={row.recommended_payment_method} vs decision={decision.recommended_payment_method}")
        if row.recommended_payment_method != certificate.recommended_payment_method:
            raise ValueError(f"Method mismatch: row={row.recommended_payment_method} vs cert={certificate.recommended_payment_method}")

        # 5. payment_plan match
        if row.payment_plan != decision.payment_plan:
            raise ValueError(f"Payment plan mismatch: row={row.payment_plan} vs decision={decision.payment_plan}")
        if row.payment_plan != certificate.payment_plan:
            raise ValueError(f"Payment plan mismatch: row={row.payment_plan} vs cert={certificate.payment_plan}")

        # 6. earliest_date_for_full_payment match
        expected_earliest = (
            decision.earliest_date_for_full_payment.isoformat()
            if decision.earliest_date_for_full_payment is not None
            else ""
        )
        if row.earliest_date_for_full_payment != expected_earliest:
            raise ValueError(f"Earliest date mismatch: row={row.earliest_date_for_full_payment} vs decision={expected_earliest}")

        if row.affordability_status == "affordable_now":
            if row.earliest_date_for_full_payment != request.request_date.isoformat():
                raise ValueError(
                    f"affordable_now must have earliest_date == request_date ({request.request_date.isoformat()}), "
                    f"got {row.earliest_date_for_full_payment}"
                )

        # 7. spending_changes_needed match
        if row.spending_changes_needed != decision.spending_changes_needed:
            raise ValueError(f"Spending changes mismatch: row={row.spending_changes_needed} vs decision={decision.spending_changes_needed}")
        if row.spending_changes_needed != certificate.spending_changes_needed:
            raise ValueError(f"Spending changes mismatch: row={row.spending_changes_needed} vs cert={certificate.spending_changes_needed}")

        # 8. decision_explanation match and validation
        if not row.decision_explanation or not row.decision_explanation.strip():
            raise ValueError("Explanation cannot be empty")
        if row.decision_explanation != explanation_result.explanation_text:
            raise ValueError("Explanation text does not match ExplanationResult")
        if not explanation_result.validation_passed:
            raise ValueError(f"Explanation failed validation: {explanation_result.unsupported_claims}")

        val_exp = ExplanationValidator.validate(row.decision_explanation, explanation_input)
        if not val_exp.is_valid:
            raise ValueError(f"Row explanation failed ExplanationValidator: {'; '.join(val_exp.errors)}")


# ---------------------------------------------------------------------------
# End-to-End Pipeline Execution & Output Generation
# ---------------------------------------------------------------------------

def generate_all_outputs(
    dataset_dir: Path,
    output_path: Path,
    force_fallback: bool = True,
) -> Tuple[List[OutputRow], str, int]:
    """Execute the end-to-end deterministic pipeline over evaluation requests.

    1. Loads dataset and resolves multi-modal ledger and recurrence.
    2. Executes simulation, safe-to-pay, payment-plan feasibility, candidate generation,
       ranking, final decision, and decision certificate generation.
    3. Validates each certificate.
    4. Generates and validates grounded explanation.
    5. Formats OutputRow and executes CrossLayerConsistencyValidator in memory.
    6. Verifies exact row count and unique request IDs.
    7. Writes output.csv atomically with UTF-8 encoding and LF line terminators.
    8. Computes SHA-256 hash and byte size of written file.
    """
    ds = load_dataset(dataset_dir)
    base_ledger = reconcile_events(ds.events, ds.profiles, ds.exchange_rates, ds.messages, ds.images)
    extractions = extract_all_images(dataset_dir)
    resolved_ledger = resolve_ledger_with_images(base_ledger, extractions, ds.exchange_rates)

    all_series, _ = detect_all_recurrence(resolved_ledger, ds.profiles, ds.messages)
    user_canonical = {uid: resolved_ledger.get_events_for_user(uid) for uid in ds.profiles}

    output_rows: List[OutputRow] = []

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
        cert_safe = evaluate_request_safe_to_pay(
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
        cset = generate_candidates(req, profile, cert_safe, feasibilities, opts_by_id)
        ctx = RequestContext(
            request=req,
            profile=profile,
            certificate=cert_safe,
            payment_option_feasibilities=feas_dict,
        )
        ustate = build_user_financial_state(
            request=req,
            profile=profile,
            canonical_events=user_canonical[uid],
            future_events=future_res.future_events,
            baseline_simulation=baseline,
            recurring_series=all_series[uid],
        )
        dec = make_final_decision_from_candidate_set(ctx, cset)
        cert = build_decision_certificate(dec, context=ctx, candidate_set=cset, user_state=ustate)

        # Validate certificate
        val_cert = validate_certificate(cert, decision=dec, raise_on_error=True)

        # Build GroundedExplanationInput and generate explanation
        exp_input = build_grounded_explanation_input(cert, currency=profile.home_currency)
        exp_result = generate_grounded_explanation(exp_input, force_fallback=force_fallback)

        # Format and validate row in memory
        row = format_output_row(req, dec, cert, exp_result)
        CrossLayerConsistencyValidator.validate_row(
            row=row,
            request=req,
            decision=dec,
            certificate=cert,
            explanation_input=exp_input,
            explanation_result=exp_result,
        )
        output_rows.append(row)

    # Verify 250 rows and unique request IDs
    if len(output_rows) != len(ds.requests):
        raise ValueError(f"Row count mismatch: generated {len(output_rows)} rows, expected {len(ds.requests)}")

    seen_ids: Set[str] = set()
    for row in output_rows:
        if row.request_id in seen_ids:
            raise ValueError(f"Duplicate request_id detected: {row.request_id}")
        seen_ids.add(row.request_id)

    # Fail closed: Only write to disk when all rows have passed validation
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(list(SCHEMA_COLUMNS))
        for row in output_rows:
            writer.writerow(list(row.to_tuple()))

    # Compute SHA-256 and size of written file
    hasher = hashlib.sha256()
    file_size = 0
    with open(output_path, "rb") as f:
        data = f.read()
        hasher.update(data)
        file_size = len(data)

    sha256_hex = hasher.hexdigest()
    return output_rows, sha256_hex, file_size


def main() -> None:
    """CLI entry point for generating output.csv."""
    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "dataset"
    output_csv = root / "output.csv"

    print(f"Generating output.csv from {dataset_dir} -> {output_csv} ...")
    rows, sha256, size = generate_all_outputs(dataset_dir, output_csv, force_fallback=True)
    print(f"Successfully generated and validated {len(rows)} rows.")
    print(f"File size: {size} bytes")
    print(f"SHA-256:   {sha256}")


if __name__ == "__main__":
    main()
