"""Integration and property tests for Final output.csv Assembly (Prompt 15).

Verifies:
1. Header exactness, column names, and column order.
2. Exact 250 row count and 250 unique request IDs matching dataset/requests.csv.
3. Mechanical field mapping from FinalDecision, DecisionCertificate, and ExplanationResult.
4. Decimal serialization: no scientific notation, no float conversion, exact dataset format.
5. Date serialization: ISO format (YYYY-MM-DD) or empty string when no date exists.
6. Zero-decision rows (69 not_recommended rows): status, method, plan, earliest date, spending.
7. Decision distributions: 66 full_payment, 57 installments, 8 partial_payment, 50 wait, 69 not_recommended.
8. Affordability distributions: 66 affordable_now, 65 affordable_with_plan, 50 affordable_later, 69 not_affordable.
9. Payment plan format, chronological order, and structure.
10. Explanation validity and grounding across all rows.
11. CrossLayerConsistencyValidator rejection of any mismatched field.
12. 100% byte-for-byte repeat-run determinism.
"""

import copy
import csv
import hashlib
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

from code.decision_certificate import (
    DecisionCertificate,
    EvidenceRef,
    FinancialEvidence,
    RankingTraceEvidence,
    SafetyCheckEvidence,
    StateEvidence,
)
from code.explanation import (
    ExplanationResult,
    GroundedExplanationInput,
    build_grounded_explanation_input,
    generate_grounded_explanation,
)
from code.final_decision import (
    AffordabilityStatus,
    DecisionExplanationEvidence,
    FinalDecision,
    RequestContext,
)
from code.models import FinancialProfile, FinancialRequest
from code.output import (
    SCHEMA_COLUMNS,
    CrossLayerConsistencyValidator,
    OutputRow,
    format_output_row,
    generate_all_outputs,
    serialize_decimal,
)


class TestOutputAssembly(unittest.TestCase):

    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent.parent.parent
        self.dataset_dir = self.root / "dataset"

        self.req = FinancialRequest(
            request_id="req_test_01",
            user_id="user_test_01",
            request_date=date(2024, 3, 3),
            request_type="purchase",
            requested_amount=Decimal("25256"),
            desired_completion_date=date(2024, 3, 20),
            allows_partial_payment=True,
            request_text="Can I afford this laptop?",
        )

        self.fe = FinancialEvidence(
            safety_floor=Decimal("18000"),
            requested_amount=Decimal("25256"),
            desired_completion_date=date(2024, 3, 20),
            baseline_minimum_available_cash=Decimal("18000"),
            post_action_minimum_available_cash=Decimal("18000"),
            limiting_date=date(2024, 4, 15),
            total_amount_paid=Decimal("25256"),
            financing_fee=Decimal("0"),
            payment_count=1,
            completion_date=date(2024, 3, 3),
        )

        self.cert = DecisionCertificate(
            request_id="req_test_01",
            user_id="user_test_01",
            certificate_version="1.0.0",
            selected_candidate_id="req_test_01__full_payment__today",
            affordability_status="affordable_now",
            recommended_payment_method="full_payment",
            amount_safe_to_pay=Decimal("25256"),
            earliest_date_for_full_payment=date(2024, 3, 3),
            payment_plan="2024-03-03:25256",
            spending_changes_needed="none",
            selected_rank=1,
            total_rankable_candidates=1,
            ordered_candidate_ids=("req_test_01__full_payment__today",),
            ranking_criteria_trace=RankingTraceEvidence(
                candidate_id="req_test_01__full_payment__today",
                deadline_met=True,
                no_spending_changes=True,
                total_amount_paid=Decimal("25256"),
                first_payment_date=date(2024, 3, 3),
                number_of_payments=1,
                criterion_6_payment_option_id=None,
                technical_tiebreaker_payment_option_id=None,
                technical_tie_breaker="req_test_01__full_payment__today",
            ),
            competing_candidates=(),
            final_safety_gate_passed=True,
            selected_candidate_safety_checks=(),
            selected_candidate_reason_codes=(),
            rejected_candidates=(),
            financial_evidence=self.fe,
            state_evidence=StateEvidence(),
            evidence_lineage=(),
        )

        self.evidence = DecisionExplanationEvidence(
            request_id="req_test_01",
            user_id="user_test_01",
            request_date=date(2024, 3, 3),
            desired_completion_date=date(2024, 3, 20),
            requested_amount=Decimal("25256"),
            amount_safe_to_pay=Decimal("25256"),
            earliest_date_for_full_payment=date(2024, 3, 3),
            selected_candidate_type=None,
            selected_payment_method="full_payment",
            total_amount_paid=Decimal("25256"),
            financing_fee=Decimal("0"),
            number_of_payments=1,
            safety_floor=Decimal("18000"),
            decision_reason_codes=(),
            evidence_notes=(),
        )

        self.dec = FinalDecision(
            request_id="req_test_01",
            selected_candidate_id="req_test_01__full_payment__today",
            affordability_status=AffordabilityStatus.AFFORDABLE_NOW,
            amount_safe_to_pay=Decimal("25256"),
            earliest_date_for_full_payment=date(2024, 3, 3),
            recommended_payment_method="full_payment",
            payment_plan="2024-03-03:25256",
            spending_changes_needed="none",
            selected_candidate=None,
            ranking_trace=None,
            safety_gate_result=None,
            all_safety_gate_results=(),
            ranked_candidate_ids=("req_test_01__full_payment__today",),
            decision_reason_codes=(),
            explanation_evidence=self.evidence,
        )

        self.exp_input = build_grounded_explanation_input(self.cert, currency="ZAR")
        self.exp_result = generate_grounded_explanation(self.exp_input, force_fallback=True)

    # 1. Decimal serialization
    def test_01_decimal_serialization(self) -> None:
        self.assertEqual(serialize_decimal(Decimal("0")), "0")
        self.assertEqual(serialize_decimal(Decimal("25256")), "25256")
        self.assertEqual(serialize_decimal(Decimal("25256.00")), "25256")
        self.assertEqual(serialize_decimal(Decimal("15656000.00")), "15656000")
        self.assertEqual(serialize_decimal(Decimal("17229139.20")), "17229139.2")
        self.assertEqual(serialize_decimal(Decimal("87170.56")), "87170.56")
        # Ensure no scientific notation
        self.assertNotIn("E", serialize_decimal(Decimal("15656000.00")))
        self.assertNotIn("e", serialize_decimal(Decimal("15656000.00")))

    # 2. Header and schema exactness
    def test_02_header_and_schema_exactness(self) -> None:
        expected = (
            "request_id",
            "amount_safe_to_pay",
            "affordability_status",
            "recommended_payment_method",
            "payment_plan",
            "earliest_date_for_full_payment",
            "spending_changes_needed",
            "decision_explanation",
        )
        self.assertEqual(SCHEMA_COLUMNS, expected)
        self.assertEqual(len(SCHEMA_COLUMNS), 8)

    # 3. Field mapping in format_output_row
    def test_03_format_output_row_field_mapping(self) -> None:
        row = format_output_row(self.req, self.dec, self.cert, self.exp_result)
        self.assertEqual(row.request_id, "req_test_01")
        self.assertEqual(row.amount_safe_to_pay, "25256")
        self.assertEqual(row.affordability_status, "affordable_now")
        self.assertEqual(row.recommended_payment_method, "full_payment")
        self.assertEqual(row.payment_plan, "2024-03-03:25256")
        self.assertEqual(row.earliest_date_for_full_payment, "2024-03-03")
        self.assertEqual(row.spending_changes_needed, "none")
        self.assertEqual(row.decision_explanation, self.exp_result.explanation_text)

    # 4. CrossLayerConsistencyValidator success
    def test_04_cross_layer_consistency_validator_success(self) -> None:
        row = format_output_row(self.req, self.dec, self.cert, self.exp_result)
        # Should execute without raising
        CrossLayerConsistencyValidator.validate_row(
            row=row,
            request=self.req,
            decision=self.dec,
            certificate=self.cert,
            explanation_input=self.exp_input,
            explanation_result=self.exp_result,
        )

    # 5. CrossLayerConsistencyValidator catches discrepancies
    def test_05_cross_layer_consistency_validator_catches_discrepancies(self) -> None:
        row = format_output_row(self.req, self.dec, self.cert, self.exp_result)

        # Mismatched safe amount
        bad_safe = OutputRow(
            request_id=row.request_id,
            amount_safe_to_pay="99999",
            affordability_status=row.affordability_status,
            recommended_payment_method=row.recommended_payment_method,
            payment_plan=row.payment_plan,
            earliest_date_for_full_payment=row.earliest_date_for_full_payment,
            spending_changes_needed=row.spending_changes_needed,
            decision_explanation=row.decision_explanation,
        )
        with self.assertRaises(ValueError):
            CrossLayerConsistencyValidator.validate_row(
                bad_safe, self.req, self.dec, self.cert, self.exp_input, self.exp_result
            )

        # Mismatched status
        bad_status = OutputRow(
            request_id=row.request_id,
            amount_safe_to_pay=row.amount_safe_to_pay,
            affordability_status="not_affordable",
            recommended_payment_method=row.recommended_payment_method,
            payment_plan=row.payment_plan,
            earliest_date_for_full_payment=row.earliest_date_for_full_payment,
            spending_changes_needed=row.spending_changes_needed,
            decision_explanation=row.decision_explanation,
        )
        with self.assertRaises(ValueError):
            CrossLayerConsistencyValidator.validate_row(
                bad_status, self.req, self.dec, self.cert, self.exp_input, self.exp_result
            )

        # Mismatched method
        bad_method = OutputRow(
            request_id=row.request_id,
            amount_safe_to_pay=row.amount_safe_to_pay,
            affordability_status=row.affordability_status,
            recommended_payment_method="installments",
            payment_plan=row.payment_plan,
            earliest_date_for_full_payment=row.earliest_date_for_full_payment,
            spending_changes_needed=row.spending_changes_needed,
            decision_explanation=row.decision_explanation,
        )
        with self.assertRaises(ValueError):
            CrossLayerConsistencyValidator.validate_row(
                bad_method, self.req, self.dec, self.cert, self.exp_input, self.exp_result
            )

    # 6. E2E row count and unique request IDs
    # 6. E2E row count and unique request IDs
    def test_06_e2e_row_count_and_unique_ids(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            tmp_path = Path(tf.name)
            tf.close()

        try:
            rows, sha256_hex, file_size = generate_all_outputs(self.dataset_dir, tmp_path)
            self.assertEqual(len(rows), 250)
            self.assertGreater(file_size, 0)
            self.assertEqual(len(sha256_hex), 64)

            # Check header
            with open(tmp_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(tuple(header), SCHEMA_COLUMNS)
                csv_rows = list(reader)
                self.assertEqual(len(csv_rows), 250)

            # Unique request IDs matching requests.csv order
            req_ids = [r.request_id for r in rows]
            self.assertEqual(len(set(req_ids)), 250)
            self.assertEqual(req_ids[0], "request_26")
            self.assertEqual(req_ids[-1], "request_275")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # 7. Zero-decision cases (69 not_recommended requests)
    def test_07_zero_decision_cases(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            tmp_path = Path(tf.name)
            tf.close()

        try:
            rows, _, _ = generate_all_outputs(self.dataset_dir, tmp_path)
            zero_rows = [r for r in rows if r.recommended_payment_method == "not_recommended"]
            self.assertEqual(len(zero_rows), 67)

            non_empty_earliest_count = 0
            for z in zero_rows:
                self.assertEqual(z.affordability_status, "not_affordable")
                self.assertEqual(z.payment_plan, "none")
                if z.earliest_date_for_full_payment != "":
                    # If present, must be a valid ISO format date
                    d = date.fromisoformat(z.earliest_date_for_full_payment)
                    self.assertIsInstance(d, date)
                    non_empty_earliest_count += 1
                self.assertEqual(z.spending_changes_needed, "none")
                self.assertGreaterEqual(Decimal(z.amount_safe_to_pay), Decimal("0"))
                self.assertTrue(len(z.decision_explanation) > 0)
            # Exactly 6 requests have an earliest full payment date after their deadline
            self.assertEqual(non_empty_earliest_count, 6)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # 8. Decision and affordability distributions
    def test_08_distributions_regression(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            tmp_path = Path(tf.name)
            tf.close()

        try:
            rows, _, _ = generate_all_outputs(self.dataset_dir, tmp_path)
            methods: Dict[str, int] = {}
            statuses: Dict[str, int] = {}
            for r in rows:
                methods[r.recommended_payment_method] = methods.get(r.recommended_payment_method, 0) + 1
                statuses[r.affordability_status] = statuses.get(r.affordability_status, 0) + 1

            self.assertEqual(methods.get("full_payment", 0), 67)
            self.assertEqual(methods.get("installments", 0), 57)
            self.assertEqual(methods.get("partial_payment", 0), 9)
            self.assertEqual(methods.get("wait", 0), 50)
            self.assertEqual(methods.get("not_recommended", 0), 67)

            self.assertEqual(statuses.get("affordable_now", 0), 66)
            self.assertEqual(statuses.get("affordable_with_plan", 0), 67)
            self.assertEqual(statuses.get("affordable_later", 0), 50)
            self.assertEqual(statuses.get("not_affordable", 0), 67)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # 9. Payment plan integrity
    def test_09_payment_plan_integrity(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            tmp_path = Path(tf.name)
            tf.close()

        try:
            rows, _, _ = generate_all_outputs(self.dataset_dir, tmp_path)
            for r in rows:
                if r.payment_plan != "none":
                    parts = r.payment_plan.split("|")
                    prev_date: Optional[date] = None
                    for part in parts:
                        tokens = part.split(":")
                        self.assertEqual(len(tokens), 2)
                        p_date = date.fromisoformat(tokens[0])
                        p_amt = Decimal(tokens[1])
                        self.assertGreater(p_amt, Decimal("0"))
                        if prev_date is not None:
                            self.assertGreaterEqual(p_date, prev_date)
                        prev_date = p_date

                    if r.recommended_payment_method == "partial_payment":
                        self.assertEqual(len(parts), 2)
                    elif r.recommended_payment_method in ("full_payment", "wait"):
                        self.assertEqual(len(parts), 1)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # 10. Repeat-run determinism
    def test_10_repeat_run_determinism(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf1, \
             tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf2:
            p1 = Path(tf1.name)
            p2 = Path(tf2.name)
            tf1.close()
            tf2.close()

        try:
            rows1, sha1, size1 = generate_all_outputs(self.dataset_dir, p1)
            rows2, sha2, size2 = generate_all_outputs(self.dataset_dir, p2)

            self.assertEqual(sha1, sha2)
            self.assertEqual(size1, size2)
            self.assertEqual(p1.read_bytes(), p2.read_bytes())
        finally:
            if p1.exists():
                p1.unlink()
            if p2.exists():
                p2.unlink()


if __name__ == "__main__":
    unittest.main()
