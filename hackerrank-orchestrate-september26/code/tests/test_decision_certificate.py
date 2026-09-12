"""Comprehensive Unit and Invariant Tests for Decision Certificate Layer (Prompt 13).

Covers all 20 required determinism and safety invariants:
  1. certificate generation is deterministic.
  2. repeated generation produces byte-identical canonical JSON.
  3. SHA-256 fingerprint is stable.
  4. input FinalDecision is not mutated.
  5. certificate validation accepts valid decisions.
  6. certificate validation rejects altered selected candidate.
  7. certificate validation rejects altered amount_safe_to_pay.
  8. certificate validation rejects altered earliest_date_for_full_payment.
  9. certificate validation rejects altered ranking order.
  10. certificate validation rejects altered safety result.
  11. certificate validation rejects invalid affordability status.
  12. certificate validation rejects invalid payment method.
  13. certificate validation rejects missing lineage.
  14. zero-candidate certificate works.
  15. multi-candidate certificate preserves ranking order.
  16. lower-ranked candidates remain traceable.
  17. Decimal serialization is exact.
  18. candidate permutation before ranking does not alter certificate.
  19. same decision generated through repeated process runs has identical hash.
  20. no free-form explanation text is generated.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
from decimal import Decimal
import json
import random
import unittest
from typing import Dict, List, Optional, Sequence, Tuple

from code.affordability import AffordabilityStatus
from code.candidate_generation import (
    Candidate,
    CandidatePayment,
    CandidateProvenance,
    CandidateSet,
    CandidateStatus,
    CandidateType,
)
from code.decision_certificate import (
    CERTIFICATE_VERSION,
    CertificateValidationError,
    CertificateValidationResult,
    CompetingCandidateEvidence,
    DecisionCertificate,
    EvidenceRef,
    FinancialEvidence,
    RankingTraceEvidence,
    RecurringStreamEvidence,
    RejectedCandidateEvidence,
    SafetyCheckEvidence,
    ScheduledObligationEvidence,
    StateEvidence,
    build_decision_certificate,
    certificate_hash,
    validate_certificate,
)
from code.final_decision import (
    DecisionExplanationEvidence,
    DecisionReasonCode,
    FinalDecision,
    RequestContext,
    SafetyGateCheck,
    SafetyGateResult,
    evaluate_candidate_safety_gate,
    make_final_decision,
)
from code.models import FinancialProfile, FinancialRequest
from code.payment_plan import PaymentPlanFeasibility
from code.ranking import RankingCriteriaTrace
from code.safe_to_pay import SafeToPayCertificate


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    user_id: str = "u_01",
    available_balance: Decimal = Decimal("50000"),
    min_balance: Decimal = Decimal("5000"),
    payment_methods: Tuple[str, ...] = ("full_payment", "partial_payment", "installments"),
    max_installment_months: Optional[int] = 12,
) -> FinancialProfile:
    return FinancialProfile(
        user_id=user_id,
        home_currency="INR",
        current_available_balance=available_balance,
        minimum_balance_to_keep=min_balance,
        financial_priorities=("essential_expenses",),
        expense_categories_to_protect=("rent", "utilities"),
        expense_categories_user_is_willing_to_reduce=("dining",),
        expense_categories_user_is_willing_to_stop=("subscriptions",),
        payment_methods_user_will_consider=payment_methods,
        max_installment_months=max_installment_months,
    )


def _make_request(
    request_id: str = "req_01",
    user_id: str = "u_01",
    request_date: date = date(2026, 3, 1),
    requested_amount: Decimal = Decimal("10000"),
    desired_completion_date: date = date(2026, 6, 1),
    allows_partial_payment: bool = True,
) -> FinancialRequest:
    return FinancialRequest(
        request_id=request_id,
        user_id=user_id,
        request_date=request_date,
        request_type="purchase",
        requested_amount=requested_amount,
        desired_completion_date=desired_completion_date,
        allows_partial_payment=allows_partial_payment,
        request_text="Test request",
    )


def _make_certificate(
    request_id: str = "req_01",
    user_id: str = "u_01",
    request_date: date = date(2026, 3, 1),
    requested_amount: Decimal = Decimal("10000"),
    amount_safe_to_pay: Decimal = Decimal("10000"),
    earliest_date_for_full_payment: Optional[date] = date(2026, 3, 1),
    is_full_payment_safe_today: bool = True,
    safety_floor: Decimal = Decimal("5000"),
    safety_floor_margin: Decimal = Decimal("35000"),
) -> SafeToPayCertificate:
    return SafeToPayCertificate(
        request_id=request_id,
        user_id=user_id,
        request_date=request_date,
        simulation_start=request_date,
        simulation_end=request_date + timedelta(days=90),
        requested_amount=requested_amount,
        amount_safe_to_pay=amount_safe_to_pay,
        currency="INR",
        safety_floor=safety_floor,
        baseline_minimum_available_cash=safety_floor + safety_floor_margin,
        baseline_minimum_cash_date=request_date,
        limiting_date=request_date,
        available_cash_after_purchase_today=safety_floor + safety_floor_margin,
        minimum_available_cash_after_purchase=safety_floor + safety_floor_margin,
        safety_floor_margin=safety_floor_margin,
        earliest_date_for_full_payment=earliest_date_for_full_payment,
        is_full_payment_safe_today=is_full_payment_safe_today,
        is_full_payment_safe_later=earliest_date_for_full_payment is not None and earliest_date_for_full_payment > request_date,
        reason_if_unsafe=None if is_full_payment_safe_today else "Unsafe today",
        candidate_search_method="exact_full_amount",
    )


def _make_candidate(
    candidate_id: str = "c_test",
    request_id: str = "req_01",
    candidate_type: CandidateType = CandidateType.FULL_PAYMENT,
    total_amount_paid: Decimal = Decimal("10000"),
    first_payment_date: Optional[date] = date(2026, 3, 1),
    number_of_payments: int = 1,
    completion_date: Optional[date] = date(2026, 3, 1),
    spending_changes: Tuple[str, ...] = (),
    source_payment_option_id: Optional[str] = None,
    is_safe: bool = True,
    status: CandidateStatus = CandidateStatus.ELIGIBLE_AND_SAFE,
    financing_fee: Decimal = Decimal("0"),
    cert_id: Optional[str] = "req_01",
) -> Candidate:
    schedule: List[CandidatePayment] = []
    if status == CandidateStatus.ELIGIBLE_AND_SAFE and is_safe:
        if number_of_payments == 1:
            pay_d = completion_date or first_payment_date or date(2026, 3, 1)
            first_payment_date = pay_d
            completion_date = pay_d
            schedule.append(
                CandidatePayment(
                    payment_date=pay_d,
                    amount=total_amount_paid,
                    sequence_number=1,
                )
            )
        else:
            first_d = first_payment_date or date(2026, 3, 1)
            first_payment_date = first_d
            comp_d = completion_date if (completion_date and completion_date > first_d) else first_d + timedelta(days=30 * (number_of_payments - 1))
            completion_date = comp_d
            part = (total_amount_paid / Decimal(number_of_payments)).quantize(Decimal("0.01"))
            running = Decimal("0")
            for i in range(1, number_of_payments):
                pay_d = first_d + timedelta(days=30 * (i - 1))
                schedule.append(CandidatePayment(payment_date=pay_d, amount=part, sequence_number=i))
                running += part
            schedule.append(
                CandidatePayment(
                    payment_date=comp_d,
                    amount=total_amount_paid - running,
                    sequence_number=number_of_payments,
                )
            )

    return Candidate(
        candidate_id=candidate_id,
        request_id=request_id,
        candidate_type=candidate_type,
        payment_method=candidate_type.value,
        total_amount_paid=total_amount_paid,
        first_payment_date=first_payment_date,
        final_payment_date=completion_date,
        number_of_payments=number_of_payments if schedule else 0,
        payment_schedule=tuple(schedule),
        financing_fee=financing_fee,
        spending_changes=spending_changes,
        source_payment_option_id=source_payment_option_id,
        is_safe=is_safe,
        completion_date=completion_date,
        feasibility_reason=None if is_safe else "Unsafe",
        status=status,
        provenance=CandidateProvenance(
            source=candidate_type.value,
            source_payment_option_id=source_payment_option_id,
            safe_to_pay_certificate_id=cert_id,
            earliest_full_payment_date=first_payment_date if candidate_type == CandidateType.WAIT else None,
        ),
    )


def _make_feasibility(
    payment_option_id: str = "opt_1",
    request_id: str = "req_01",
    is_eligible: bool = True,
    is_safe: bool = True,
) -> PaymentPlanFeasibility:
    return PaymentPlanFeasibility(
        request_id=request_id,
        payment_option_id=payment_option_id,
        is_eligible=is_eligible,
        is_safe=is_safe,
        rejection_reason=None if is_safe and is_eligible else "Ineligible/Unsafe",
        payment_plan=None,
        minimum_available_cash=Decimal("10000"),
        safety_floor=Decimal("5000"),
        limiting_date=date(2026, 3, 1),
        simulator_certificate=None,
    )


def _make_context(
    request: Optional[FinancialRequest] = None,
    profile: Optional[FinancialProfile] = None,
    cert: Optional[SafeToPayCertificate] = None,
    feasibilities: Optional[Dict[str, PaymentPlanFeasibility]] = None,
) -> RequestContext:
    req = request or _make_request()
    prof = profile or _make_profile()
    c = cert or _make_certificate(request_id=req.request_id, user_id=req.user_id)
    return RequestContext(
        request=req,
        profile=prof,
        certificate=c,
        payment_option_feasibilities=feasibilities if feasibilities is not None else {},
    )


class TestDecisionCertificate(unittest.TestCase):
    """Unit tests and invariant assertions for DecisionCertificate."""

    def setUp(self) -> None:
        self.feas_opt1 = _make_feasibility("opt_1")
        self.ctx = _make_context(feasibilities={"opt_1": self.feas_opt1})
        self.c1 = _make_candidate("c1", total_amount_paid=Decimal("10000"))
        self.decision = make_final_decision(self.ctx, [self.c1])

    # 1. Certificate generation is deterministic
    def test_invariant_1_generation_is_deterministic(self) -> None:
        cert1 = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        cert2 = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        self.assertEqual(cert1, cert2)

    # 2. Repeated generation produces byte-identical canonical JSON
    def test_invariant_2_byte_identical_canonical_json(self) -> None:
        cert1 = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        cert2 = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        json1 = cert1.to_canonical_json()
        json2 = cert2.to_canonical_json()
        self.assertEqual(json1, json2)
        self.assertEqual(json1.encode("utf-8"), json2.encode("utf-8"))

    # 3. SHA-256 fingerprint is stable
    def test_invariant_3_sha256_fingerprint_is_stable(self) -> None:
        cert1 = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        cert2 = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        hash1 = certificate_hash(cert1)
        hash2 = certificate_hash(cert2)
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash1, cert1.certificate_hash())
        self.assertEqual(len(hash1), 64)

    # 4. Input FinalDecision is not mutated
    def test_invariant_4_input_decision_not_mutated(self) -> None:
        decision_copy = copy.deepcopy(self.decision)
        _ = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        self.assertEqual(self.decision, decision_copy)

    # 5. Certificate validation accepts valid decisions
    def test_invariant_5_validation_accepts_valid_decisions(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        result = validate_certificate(cert, self.decision)
        self.assertTrue(result.is_valid)
        self.assertTrue(bool(result))
        self.assertEqual(len(result.errors), 0)
        # Strict mode should not raise
        validate_certificate(cert, self.decision, raise_on_error=True)

    # 6. Certificate validation rejects altered selected candidate
    def test_invariant_6_validation_rejects_altered_selected_candidate(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        # Replace selected_candidate_id with a nonexistent one
        bad_dict = cert.to_canonical_dict()
        bad_dict["selected_candidate_id"] = "nonexistent_cand"
        # Manually alter certificate object
        altered_cert = copy.copy(cert)
        object.__setattr__(altered_cert, "selected_candidate_id", "nonexistent_cand")
        result = validate_certificate(altered_cert, self.decision)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("not in ordered_candidate_ids" in e for e in result.errors))
        with self.assertRaises(CertificateValidationError):
            validate_certificate(altered_cert, self.decision, raise_on_error=True)

    # 7. Certificate validation rejects altered amount_safe_to_pay
    def test_invariant_7_validation_rejects_altered_amount_safe_to_pay(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        altered_cert = copy.copy(cert)
        object.__setattr__(altered_cert, "amount_safe_to_pay", Decimal("999999"))
        result = validate_certificate(altered_cert, self.decision)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("amount_safe_to_pay mismatch" in e for e in result.errors))

    # 8. Certificate validation rejects altered earliest_date_for_full_payment
    def test_invariant_8_validation_rejects_altered_earliest_date(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        altered_cert = copy.copy(cert)
        object.__setattr__(altered_cert, "earliest_date_for_full_payment", date(2030, 1, 1))
        result = validate_certificate(altered_cert, self.decision)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("earliest_date_for_full_payment mismatch" in e for e in result.errors))

    # 9. Certificate validation rejects altered ranking order
    def test_invariant_9_validation_rejects_altered_ranking_order(self) -> None:
        c2 = _make_candidate("c2", total_amount_paid=Decimal("12000"))
        dec2 = make_final_decision(self.ctx, [self.c1, c2])
        cert = build_decision_certificate(dec2, context=self.ctx, candidate_set=[self.c1, c2])
        altered_cert = copy.copy(cert)
        # Reverse ordered_candidate_ids
        object.__setattr__(altered_cert, "ordered_candidate_ids", tuple(reversed(cert.ordered_candidate_ids)))
        result = validate_certificate(altered_cert, dec2)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("ordered_candidate_ids does not match" in e for e in result.errors))

    # 10. Certificate validation rejects altered safety result
    def test_invariant_10_validation_rejects_altered_safety_result(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        altered_cert = copy.copy(cert)
        object.__setattr__(altered_cert, "final_safety_gate_passed", False)
        result = validate_certificate(altered_cert, self.decision)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Selected candidate must pass final safety gate" in e for e in result.errors))

    # 11. Certificate validation rejects invalid affordability status
    def test_invariant_11_validation_rejects_invalid_affordability_status(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        altered_cert = copy.copy(cert)
        object.__setattr__(altered_cert, "affordability_status", "somewhat_affordable")
        result = validate_certificate(altered_cert, self.decision)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Invalid affordability_status" in e for e in result.errors))

    # 12. Certificate validation rejects invalid payment method
    def test_invariant_12_validation_rejects_invalid_payment_method(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        altered_cert = copy.copy(cert)
        object.__setattr__(altered_cert, "recommended_payment_method", "barter")
        result = validate_certificate(altered_cert, self.decision)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Invalid recommended_payment_method" in e for e in result.errors))

    # 13. Certificate validation rejects missing lineage
    def test_invariant_13_validation_rejects_missing_lineage(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        altered_cert = copy.copy(cert)
        object.__setattr__(altered_cert, "evidence_lineage", ())
        result = validate_certificate(altered_cert, self.decision)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("evidence_lineage is empty" in e for e in result.errors))

        # Test missing mandatory lineage field
        stripped_lineage = tuple(e for e in cert.evidence_lineage if e.field_name != "amount_safe_to_pay")
        object.__setattr__(altered_cert, "evidence_lineage", stripped_lineage)
        res2 = validate_certificate(altered_cert, self.decision)
        self.assertFalse(res2.is_valid)
        self.assertTrue(any("Missing lineage reference for mandatory field: amount_safe_to_pay" in e for e in res2.errors))

    # 14. Zero-candidate certificate works
    def test_invariant_14_zero_candidate_certificate_works(self) -> None:
        cert_unsafe = _make_certificate(
            is_full_payment_safe_today=False,
            amount_safe_to_pay=Decimal("0"),
            earliest_date_for_full_payment=None,
            safety_floor_margin=Decimal("-1000"),
        )
        ctx_unsafe = _make_context(cert=cert_unsafe)
        c_unsafe = _make_candidate("c_bad", is_safe=False, status=CandidateStatus.STRUCTURALLY_VALID_UNSAFE)
        dec_zero = make_final_decision(ctx_unsafe, [c_unsafe])
        self.assertIsNone(dec_zero.selected_candidate_id)

        cert = build_decision_certificate(dec_zero, context=ctx_unsafe, candidate_set=[c_unsafe])
        self.assertIsNone(cert.selected_candidate_id)
        self.assertIsNone(cert.selected_rank)
        self.assertEqual(cert.recommended_payment_method, "not_recommended")
        self.assertEqual(cert.payment_plan, "none")
        self.assertFalse(cert.final_safety_gate_passed)
        self.assertEqual(cert.affordability_status, AffordabilityStatus.NOT_AFFORDABLE.value)

        # Validate
        res = validate_certificate(cert, dec_zero)
        self.assertTrue(res.is_valid)

        # Canonical JSON
        c_json = cert.to_canonical_json()
        self.assertIn('"selected_candidate_id":null', c_json)
        self.assertIn('"payment_plan":"none"', c_json)

    # 15. Multi-candidate certificate preserves ranking order
    def test_invariant_15_multi_candidate_preserves_ranking_order(self) -> None:
        c_inst = _make_candidate(
            "c_inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("10500"),
            number_of_payments=3,
            source_payment_option_id="opt_1",
        )
        # Note: full_payment (10000) ranks ahead of installments (10500) by total amount paid
        dec_multi = make_final_decision(self.ctx, [self.c1, c_inst])
        self.assertEqual(dec_multi.selected_candidate_id, "c1")
        self.assertEqual(dec_multi.ranked_candidate_ids, ("c1", "c_inst"))

        cert = build_decision_certificate(dec_multi, context=self.ctx, candidate_set=[self.c1, c_inst])
        self.assertEqual(cert.ordered_candidate_ids, ("c1", "c_inst"))
        self.assertEqual(len(cert.competing_candidates), 2)
        self.assertEqual(cert.competing_candidates[0].candidate_id, "c1")
        self.assertEqual(cert.competing_candidates[0].rank, 1)
        self.assertEqual(cert.competing_candidates[1].candidate_id, "c_inst")
        self.assertEqual(cert.competing_candidates[1].rank, 2)

    # 16. Lower-ranked candidates remain traceable
    def test_invariant_16_lower_ranked_candidates_traceable(self) -> None:
        c_inst = _make_candidate(
            "c_inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("10500"),
            number_of_payments=3,
            source_payment_option_id="opt_1",
        )
        dec_multi = make_final_decision(self.ctx, [self.c1, c_inst])
        cert = build_decision_certificate(dec_multi, context=self.ctx, candidate_set=[self.c1, c_inst])
        lower_comp = cert.competing_candidates[1]
        self.assertEqual(lower_comp.candidate_id, "c_inst")
        self.assertEqual(lower_comp.rank, 2)
        self.assertEqual(lower_comp.payment_method, "installments")
        self.assertTrue(lower_comp.deadline_met)
        self.assertFalse(lower_comp.spending_changes_required)
        self.assertEqual(lower_comp.total_amount_paid, Decimal("10500"))
        self.assertEqual(lower_comp.payment_count, 3)
        self.assertEqual(lower_comp.payment_option_id, "opt_1")
        self.assertTrue(lower_comp.is_safe)

    # 17. Decimal serialization is exact
    def test_invariant_17_decimal_serialization_exact(self) -> None:
        precise_amt = Decimal("12345.67")
        c_precise = _make_candidate("c_p", total_amount_paid=precise_amt)
        cert_ctx = _make_certificate(requested_amount=precise_amt, amount_safe_to_pay=precise_amt)
        ctx = _make_context(cert=cert_ctx)
        dec = make_final_decision(ctx, [c_precise])
        cert = build_decision_certificate(dec, context=ctx, candidate_set=[c_precise])

        c_json = cert.to_canonical_json()
        self.assertIn('"amount_safe_to_pay":"12345.67"', c_json)
        self.assertIn('"requested_amount":"12345.67"', c_json)
        self.assertIn('"total_amount_paid":"12345.67"', c_json)
        # Parse back to verify string type
        parsed = json.loads(c_json)
        self.assertIsInstance(parsed["amount_safe_to_pay"], str)
        self.assertEqual(parsed["amount_safe_to_pay"], "12345.67")

    # 18. Candidate permutation before ranking does not alter certificate
    def test_invariant_18_permutation_invariance(self) -> None:
        c_inst = _make_candidate(
            "c_inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("10500"),
            number_of_payments=3,
            source_payment_option_id="opt_1",
        )
        dec_ord1 = make_final_decision(self.ctx, [self.c1, c_inst])
        dec_ord2 = make_final_decision(self.ctx, [c_inst, self.c1])

        cert1 = build_decision_certificate(dec_ord1, context=self.ctx, candidate_set=[self.c1, c_inst])
        cert2 = build_decision_certificate(dec_ord2, context=self.ctx, candidate_set=[c_inst, self.c1])

        self.assertEqual(cert1, cert2)
        self.assertEqual(cert1.to_canonical_json(), cert2.to_canonical_json())
        self.assertEqual(cert1.certificate_hash(), cert2.certificate_hash())

    # 19. Same decision generated through repeated process runs has identical hash
    def test_invariant_19_repeated_process_hash_identity(self) -> None:
        hashes = set()
        for _ in range(5):
            ctx_i = _make_context()
            c_i = _make_candidate("c1", total_amount_paid=Decimal("10000"))
            dec_i = make_final_decision(ctx_i, [c_i])
            cert_i = build_decision_certificate(dec_i, context=ctx_i, candidate_set=[c_i])
            hashes.add(certificate_hash(cert_i))
        self.assertEqual(len(hashes), 1)

    # 20. No free-form explanation text is generated
    def test_invariant_20_no_free_form_explanation_text(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        c_dict = cert.to_canonical_dict()

        # Recursive check that no string value looks like multi-sentence prose or paragraphs
        def check_no_prose(obj: object, path: str = "") -> None:
            if isinstance(obj, str):
                # Ensure no newlines or long paragraph text
                self.assertNotIn("\n", obj, f"Newline found at {path}: {obj}")
                words = obj.split()
                # Labels, reason codes, enum strings, or brief audit notes are all short
                self.assertLess(
                    len(words),
                    30,
                    f"Excessively long prose text found at {path}: {obj}",
                )
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_prose(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    check_no_prose(v, f"{path}[{i}]")

        check_no_prose(c_dict)

    # 21. Certificate generation with UserFinancialState evidence
    def test_certificate_with_user_state(self) -> None:
        rec_in = RecurringStreamEvidence(
            series_id="rec_01",
            category="salary",
            direction="inflow",
            frequency="monthly",
            amount=Decimal("50000"),
            monthly_equivalent=Decimal("50000"),
        )
        rec_out = RecurringStreamEvidence(
            series_id="rec_02",
            category="rent",
            direction="outflow",
            frequency="monthly",
            amount=Decimal("15000"),
            monthly_equivalent=Decimal("15000"),
        )
        sched_ob = ScheduledObligationEvidence(
            event_id="ev_01",
            effective_date=date(2026, 3, 5),
            category="utilities",
            amount=Decimal("2000"),
            source_type="explicit_scheduled",
        )
        st_ev = StateEvidence(
            recurring_obligations=(rec_in, rec_out),
            scheduled_obligations=(sched_ob,),
            spending_trend="stable",
            income_stability="high",
            expense_stability="high",
        )
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        # Manually verify StateEvidence serialization
        d_st = st_ev.to_canonical_dict()
        self.assertEqual(d_st["spending_trend"], "stable")
        self.assertEqual(len(d_st["recurring_obligations"]), 2)
        self.assertEqual(len(d_st["scheduled_obligations"]), 1)

    # 22. Standalone certificate validation (decision=None)
    def test_standalone_certificate_validation(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        res = validate_certificate(cert, decision=None)
        self.assertTrue(res.is_valid)

    # 23. Certificate validation rejects missing or invalid request_id and version
    def test_validation_rejects_missing_request_id_or_version(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        alt1 = copy.copy(cert)
        object.__setattr__(alt1, "request_id", "")
        res1 = validate_certificate(alt1)
        self.assertFalse(res1.is_valid)
        self.assertTrue(any("request_id missing" in e for e in res1.errors))

        alt2 = copy.copy(cert)
        object.__setattr__(alt2, "certificate_version", "")
        res2 = validate_certificate(alt2)
        self.assertFalse(res2.is_valid)
        self.assertTrue(any("certificate_version missing" in e for e in res2.errors))

    # 24. Certificate validation rejects non-Decimal values in financial evidence
    def test_validation_rejects_non_decimal_values(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        alt = copy.copy(cert)
        object.__setattr__(alt, "amount_safe_to_pay", 10000.0)  # float instead of Decimal
        res = validate_certificate(alt)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("amount_safe_to_pay must be Decimal" in e for e in res.errors))

    # 25. Certificate validation rejects ranking_criteria_trace candidate_id mismatch
    def test_validation_rejects_ranking_trace_candidate_mismatch(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        alt = copy.copy(cert)
        bad_trace = RankingTraceEvidence(
            candidate_id="other_cand",
            deadline_met=True,
            no_spending_changes=True,
            total_amount_paid=Decimal("10000"),
            first_payment_date=date(2026, 3, 1),
            number_of_payments=1,
        )
        object.__setattr__(alt, "ranking_criteria_trace", bad_trace)
        res = validate_certificate(alt)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("does not match selected_candidate_id" in e for e in res.errors))

    # 26. Certificate validation rejects rejected_candidate with empty reason codes or bad gate result
    def test_validation_rejects_invalid_rejected_candidate_evidence(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        alt = copy.copy(cert)
        bad_rej = RejectedCandidateEvidence(
            candidate_id="c_rej",
            payment_method="installments",
            rejection_reason_codes=(),  # empty!
            gate_result="PASS",  # should be FAIL
        )
        object.__setattr__(alt, "rejected_candidates", (bad_rej,))
        res = validate_certificate(alt)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("missing rejection reason codes" in e for e in res.errors))
        self.assertTrue(any("gate_result must be 'FAIL'" in e for e in res.errors))

    # 27. Certificate validation rejects mismatching spending_changes_needed or payment_plan
    def test_validation_rejects_decision_plan_or_spending_changes_mismatch(self) -> None:
        cert = build_decision_certificate(self.decision, context=self.ctx, candidate_set=[self.c1])
        alt1 = copy.copy(cert)
        object.__setattr__(alt1, "payment_plan", "different_plan")
        res1 = validate_certificate(alt1, self.decision)
        self.assertFalse(res1.is_valid)
        self.assertTrue(any("payment_plan mismatch" in e for e in res1.errors))

        alt2 = copy.copy(cert)
        object.__setattr__(alt2, "spending_changes_needed", "reduce dining")
        res2 = validate_certificate(alt2, self.decision)
        self.assertFalse(res2.is_valid)
        self.assertTrue(any("spending_changes_needed mismatch" in e for e in res2.errors))

    # 28. Helper dataclasses .to_canonical_dict() verification
    def test_helper_dataclasses_canonical_dict(self) -> None:
        ev_ref = EvidenceRef("test_mod", "TestObj", "id_1", "field_1")
        d_ref = ev_ref.to_canonical_dict()
        self.assertEqual(d_ref, {
            "field_name": "field_1",
            "source_id": "id_1",
            "source_module": "test_mod",
            "source_object": "TestObj",
        })

        chk = SafetyCheckEvidence("sim_safe", True, "SIMULATION_SAFE", "ok")
        d_chk = chk.to_canonical_dict()
        self.assertEqual(d_chk["check_name"], "sim_safe")
        self.assertTrue(d_chk["passed"])

        fin = FinancialEvidence(
            safety_floor=Decimal("5000"),
            requested_amount=Decimal("10000"),
            desired_completion_date=date(2026, 6, 1),
            total_amount_paid=Decimal("10000"),
        )
        d_fin = fin.to_canonical_dict()
        self.assertEqual(d_fin["safety_floor"], "5000")
        self.assertEqual(d_fin["requested_amount"], "10000")
        self.assertEqual(d_fin["desired_completion_date"], "2026-06-01")

    # 29. CertificateValidationResult methods
    def test_validation_result_methods(self) -> None:
        ok_res = CertificateValidationResult(is_valid=True, errors=())
        self.assertTrue(bool(ok_res))
        ok_res.raise_if_invalid()  # does not raise

        fail_res = CertificateValidationResult(is_valid=False, errors=("Error 1", "Error 2"))
        self.assertFalse(bool(fail_res))
        with self.assertRaises(CertificateValidationError):
            fail_res.raise_if_invalid()

    # 30. Canonical JSON preserves all six ranking criteria
    def test_30_canonical_json_preserves_all_six_criteria(self) -> None:
        c_inst = _make_candidate(
            "c_inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("10500"),
            number_of_payments=3,
            source_payment_option_id="opt_1",
        )
        dec = make_final_decision(self.ctx, [c_inst])
        cert = build_decision_certificate(dec, context=self.ctx, candidate_set=[c_inst])

        c_json = cert.to_canonical_json()
        self.assertIn('"criterion_1_deadline_met":true', c_json)
        self.assertIn('"criterion_2_no_spending_changes":true', c_json)
        self.assertIn('"criterion_3_total_amount_paid":"10500"', c_json)
        self.assertIn('"criterion_4_first_payment_date":"2026-03-01"', c_json)
        self.assertIn('"criterion_5_number_of_payments":3', c_json)
        self.assertIn('"criterion_6_payment_option_id":"opt_1"', c_json)
        self.assertIn('"technical_tiebreaker_payment_option_id":"opt_1"', c_json)

    # 31. Changing criterion 6 changes canonical JSON and SHA-256 hash
    def test_31_changing_criterion_6_changes_canonical_json_and_hash(self) -> None:
        c_inst = _make_candidate(
            "c_inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("10500"),
            number_of_payments=3,
            source_payment_option_id="opt_1",
        )
        dec = make_final_decision(self.ctx, [c_inst])
        cert = build_decision_certificate(dec, context=self.ctx, candidate_set=[c_inst])

        orig_json = cert.to_canonical_json()
        orig_hash = certificate_hash(cert)

        # Alter criterion 6
        alt_trace = RankingTraceEvidence(
            candidate_id=cert.ranking_criteria_trace.candidate_id,
            deadline_met=cert.ranking_criteria_trace.deadline_met,
            no_spending_changes=cert.ranking_criteria_trace.no_spending_changes,
            total_amount_paid=cert.ranking_criteria_trace.total_amount_paid,
            first_payment_date=cert.ranking_criteria_trace.first_payment_date,
            number_of_payments=cert.ranking_criteria_trace.number_of_payments,
            criterion_6_payment_option_id="opt_999",
            technical_tiebreaker_payment_option_id="opt_999",
            technical_tie_breaker=cert.ranking_criteria_trace.technical_tie_breaker,
        )
        alt_cert = copy.copy(cert)
        object.__setattr__(alt_cert, "ranking_criteria_trace", alt_trace)

        new_json = alt_cert.to_canonical_json()
        new_hash = certificate_hash(alt_cert)

        self.assertNotEqual(orig_json, new_json)
        self.assertNotEqual(orig_hash, new_hash)

    # 32. Changing criterion 6 cannot silently pass validation against source decision
    def test_32_changing_criterion_6_fails_validation(self) -> None:
        c_inst = _make_candidate(
            "c_inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("10500"),
            number_of_payments=3,
            source_payment_option_id="opt_1",
        )
        dec = make_final_decision(self.ctx, [c_inst])
        cert = build_decision_certificate(dec, context=self.ctx, candidate_set=[c_inst])

        # Alter criterion 6 on certificate
        alt_trace = RankingTraceEvidence(
            candidate_id=cert.ranking_criteria_trace.candidate_id,
            deadline_met=cert.ranking_criteria_trace.deadline_met,
            no_spending_changes=cert.ranking_criteria_trace.no_spending_changes,
            total_amount_paid=cert.ranking_criteria_trace.total_amount_paid,
            first_payment_date=cert.ranking_criteria_trace.first_payment_date,
            number_of_payments=cert.ranking_criteria_trace.number_of_payments,
            criterion_6_payment_option_id="altered_option",
            technical_tiebreaker_payment_option_id="altered_option",
            technical_tie_breaker=cert.ranking_criteria_trace.technical_tie_breaker,
        )
        alt_cert = copy.copy(cert)
        object.__setattr__(alt_cert, "ranking_criteria_trace", alt_trace)

        res = validate_certificate(alt_cert, dec)
        self.assertFalse(res.is_valid)
        self.assertTrue(any("criterion_6_payment_option_id mismatch" in e for e in res.errors))

    # 33. Competing candidates preserve all 6 criteria and decisive criterion
    def test_33_competing_candidates_all_six_criteria_and_decisive(self) -> None:
        c_inst = _make_candidate(
            "c_inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("10500"),
            number_of_payments=3,
            source_payment_option_id="opt_1",
        )
        dec_multi = make_final_decision(self.ctx, [self.c1, c_inst])
        cert = build_decision_certificate(dec_multi, context=self.ctx, candidate_set=[self.c1, c_inst])

        self.assertEqual(len(cert.competing_candidates), 2)
        winner = cert.competing_candidates[0]
        loser = cert.competing_candidates[1]

        self.assertTrue(winner.is_winner)
        self.assertIsNone(winner.decisive_criterion)

        self.assertFalse(loser.is_winner)
        # Lost on total_amount_paid (10000 vs 10500)
        self.assertEqual(loser.decisive_criterion, "CRITERION_3_TOTAL_AMOUNT_PAID")
        self.assertEqual(loser.criterion_6_payment_option_id, "opt_1")


if __name__ == "__main__":
    unittest.main()

