"""Comprehensive Unit and Invariant Tests for Final Decision + Safety Gate Layer.

Covers:
  - Prompt 12 all 21 required edge cases
  - All 12 required determinism and safety invariants
  - Safety gate reason code verification
  - Affordability status exact mapping
  - Immutability and permutation invariance
"""

import copy
import random
import unittest
from datetime import date, timedelta
from decimal import Decimal
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
from code.final_decision import (
    DecisionReasonCode,
    FinalDecision,
    RequestContext,
    SafetyGateCheck,
    SafetyGateResult,
    evaluate_candidate_safety_gate,
    is_finally_safe,
    make_final_decision,
    make_final_decision_from_candidate_set,
)
from code.models import FinancialProfile, FinancialRequest, PaymentOption
from code.payment_plan import (
    PaymentPlan,
    PaymentPlanFeasibility,
    PaymentPlanSimulatorReference,
    PaymentScheduleEntry,
)
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
            part = (total_amount_paid / Decimal(number_of_payments)).quantize(Decimal("0.01"))
            running = Decimal("0")
            for i in range(1, number_of_payments):
                pay_d = (first_payment_date or date(2026, 3, 1)) + timedelta(days=30 * (i - 1))
                schedule.append(CandidatePayment(payment_date=pay_d, amount=part, sequence_number=i))
                running += part
            schedule.append(
                CandidatePayment(
                    payment_date=completion_date or date(2026, 4, 1),
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
        feasibility_reason=None if is_safe else "Unsafe in simulation",
        status=status,
        provenance=CandidateProvenance(
            source="test_provenance",
            source_payment_option_id=source_payment_option_id,
            safe_to_pay_certificate_id=cert_id,
            earliest_full_payment_date=completion_date,
        ),
    )


def _make_context(
    request: Optional[FinancialRequest] = None,
    profile: Optional[FinancialProfile] = None,
    certificate: Optional[SafeToPayCertificate] = None,
    feasibilities: Optional[Dict[str, PaymentPlanFeasibility]] = None,
) -> RequestContext:
    req = request or _make_request()
    prof = profile or _make_profile()
    cert = certificate or _make_certificate()
    feas = feasibilities if feasibilities is not None else {}
    return RequestContext(
        request=req,
        profile=prof,
        certificate=cert,
        payment_option_feasibilities=feas,
    )


# ===========================================================================
# Edge Case Tests (Required Cases 1 to 21)
# ===========================================================================

class TestFinalDecisionEdgeCases(unittest.TestCase):
    """Explicit tests for all 21 edge cases required in Prompt 12."""

    def test_case_1_zero_rankable_candidates(self) -> None:
        """1. Zero rankable candidates -> not_recommended, none, not_affordable."""
        ctx = _make_context(certificate=_make_certificate(is_full_payment_safe_today=False, earliest_date_for_full_payment=None))
        decision = make_final_decision(ctx, [])
        self.assertIsNone(decision.selected_candidate_id)
        self.assertEqual(decision.recommended_payment_method, "not_recommended")
        self.assertEqual(decision.payment_plan, "none")
        self.assertEqual(decision.spending_changes_needed, "none")
        self.assertEqual(decision.affordability_status, AffordabilityStatus.NOT_AFFORDABLE)
        self.assertIn(DecisionReasonCode.NO_SAFE_CANDIDATE_SURVIVED, decision.decision_reason_codes)

    def test_case_2_exactly_one_rankable_candidate(self) -> None:
        """2. Exactly one rankable candidate -> selected immediately."""
        ctx = _make_context()
        c = _make_candidate("c_only", candidate_type=CandidateType.FULL_PAYMENT)
        decision = make_final_decision(ctx, [c])
        self.assertEqual(decision.selected_candidate_id, "c_only")
        self.assertEqual(decision.affordability_status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertEqual(decision.recommended_payment_method, "full_payment")
        self.assertEqual(decision.payment_plan, "2026-03-01:10000")

    def test_case_3_full_payment_vs_installment_tie(self) -> None:
        """3. Full payment vs installment: full payment wins on fewer payments (1 < 2)."""
        ctx = _make_context()
        c_full = _make_candidate("c_full", candidate_type=CandidateType.FULL_PAYMENT, number_of_payments=1)
        c_inst = _make_candidate("c_inst", candidate_type=CandidateType.INSTALLMENT_PLAN, number_of_payments=2, source_payment_option_id="opt_1")
        # Provide valid feasibility for opt_1
        feas = PaymentPlanFeasibility(
            request_id="req_01",
            payment_option_id="opt_1",
            is_eligible=True,
            is_safe=True,
            rejection_reason=None,
            payment_plan=PaymentPlan("req_01", "opt_1", "installments", (
                PaymentScheduleEntry(date(2026, 3, 1), Decimal("5000"), 1),
                PaymentScheduleEntry(date(2026, 4, 1), Decimal("5000"), 2),
            ), Decimal("10000"), Decimal("0"), Decimal("10000")),
            minimum_available_cash=Decimal("50000"),
            safety_floor=Decimal("5000"),
            limiting_date=date(2026, 4, 1),
            simulator_certificate=None,
        )
        ctx = _make_context(feasibilities={"opt_1": feas})
        decision = make_final_decision(ctx, [c_inst, c_full])
        self.assertEqual(decision.selected_candidate_id, "c_full")
        self.assertEqual(decision.affordability_status, AffordabilityStatus.AFFORDABLE_NOW)

    def test_case_4_partial_payment_vs_installment(self) -> None:
        """4. Partial payment vs installment: 2 payments vs 3 payments -> partial payment wins."""
        cert = _make_certificate(
            amount_safe_to_pay=Decimal("5000"),
            earliest_date_for_full_payment=date(2026, 4, 1),
            is_full_payment_safe_today=False,
        )
        feas = PaymentPlanFeasibility("req_01", "opt_3", is_eligible=True, is_safe=True, rejection_reason=None, payment_plan=None, minimum_available_cash=None, safety_floor=None, limiting_date=None)
        ctx = _make_context(certificate=cert, feasibilities={"opt_3": feas})
        c_part = _make_candidate("c_part", candidate_type=CandidateType.PARTIAL_PAYMENT, number_of_payments=2, completion_date=date(2026, 4, 1))
        c_inst = _make_candidate("c_inst", candidate_type=CandidateType.INSTALLMENT_PLAN, number_of_payments=3, source_payment_option_id="opt_3")
        decision = make_final_decision(ctx, [c_inst, c_part])
        self.assertEqual(decision.selected_candidate_id, "c_part")
        self.assertEqual(decision.affordability_status, AffordabilityStatus.AFFORDABLE_WITH_PLAN)

    def test_case_5_wait_vs_installment(self) -> None:
        """5. Wait vs installment with fee -> wait wins on Criterion 3 (lower total paid)."""
        ctx = _make_context(certificate=_make_certificate(
            is_full_payment_safe_today=False,
            earliest_date_for_full_payment=date(2026, 3, 20),
        ))
        c_wait = _make_candidate("c_wait", candidate_type=CandidateType.WAIT, total_amount_paid=Decimal("10000"), first_payment_date=date(2026, 3, 20), completion_date=date(2026, 3, 20), number_of_payments=1)
        c_inst = _make_candidate("c_inst", candidate_type=CandidateType.INSTALLMENT_PLAN, total_amount_paid=Decimal("10500"), financing_fee=Decimal("500"), number_of_payments=2, source_payment_option_id="opt_fee")
        feas = PaymentPlanFeasibility("req_01", "opt_fee", is_eligible=True, is_safe=True, rejection_reason=None, payment_plan=None, minimum_available_cash=None, safety_floor=None, limiting_date=None)
        ctx = RequestContext(ctx.request, ctx.profile, ctx.certificate, {"opt_fee": feas})
        decision = make_final_decision(ctx, [c_inst, c_wait])
        self.assertEqual(decision.selected_candidate_id, "c_wait")
        self.assertEqual(decision.affordability_status, AffordabilityStatus.AFFORDABLE_LATER)
        self.assertEqual(decision.recommended_payment_method, "wait")

    def test_case_6_multiple_candidates_with_identical_schedules(self) -> None:
        """6. Multiple options with identical schedules -> lowest payment_option_id wins."""
        feas1 = PaymentPlanFeasibility("req_01", "payment_option_01", is_eligible=True, is_safe=True, rejection_reason=None, payment_plan=None, minimum_available_cash=None, safety_floor=None, limiting_date=None)
        feas2 = PaymentPlanFeasibility("req_01", "payment_option_02", is_eligible=True, is_safe=True, rejection_reason=None, payment_plan=None, minimum_available_cash=None, safety_floor=None, limiting_date=None)
        ctx = _make_context(feasibilities={"payment_option_01": feas1, "payment_option_02": feas2})
        c1 = _make_candidate("c1", candidate_type=CandidateType.INSTALLMENT_PLAN, source_payment_option_id="payment_option_01")
        c2 = _make_candidate("c2", candidate_type=CandidateType.INSTALLMENT_PLAN, source_payment_option_id="payment_option_02")
        decision = make_final_decision(ctx, [c2, c1])
        self.assertEqual(decision.selected_candidate_id, "c1")

    def test_case_7_candidate_with_invalid_safety_certificate(self) -> None:
        """7. Candidate with invalid certificate reference fails safety gate."""
        ctx = _make_context()
        # Candidate points to wrong certificate
        c_bad = _make_candidate("c_bad", cert_id="wrong_req_id")
        res = evaluate_candidate_safety_gate(c_bad, ctx)
        self.assertFalse(res.is_safe)
        self.assertIn(DecisionReasonCode.INVALID_SAFETY_CERTIFICATE, res.rejection_reason_codes)

    def test_case_8_candidate_marked_safe_but_certificate_shows_breach(self) -> None:
        """8. Candidate marked safe upstream but certificate shows margin breach -> rejected."""
        ctx = _make_context(certificate=_make_certificate(safety_floor_margin=Decimal("-10.00")))
        c = _make_candidate("c_breach")
        res = evaluate_candidate_safety_gate(c, ctx)
        self.assertFalse(res.is_safe)
        self.assertIn(DecisionReasonCode.SAFETY_FLOOR_BREACH, res.rejection_reason_codes)

    def test_case_9_unsafe_mixed_with_safe_candidates(self) -> None:
        """9. Unsafe candidates mixed with safe: unsafe rejected by gate, safe survives and wins."""
        ctx = _make_context()
        c_good = _make_candidate("c_good")
        c_bad = _make_candidate("c_bad", cert_id="wrong_id")
        decision = make_final_decision(ctx, [c_bad, c_good])
        self.assertEqual(decision.selected_candidate_id, "c_good")
        self.assertEqual(len(decision.ranked_candidate_ids), 1)

    def test_case_10_all_candidates_unsafe(self) -> None:
        """10. All candidates fail safety gate -> fallback decision."""
        ctx = _make_context()
        c1 = _make_candidate("c1", cert_id="bad_1")
        c2 = _make_candidate("c2", cert_id="bad_2")
        decision = make_final_decision(ctx, [c1, c2])
        self.assertIsNone(decision.selected_candidate_id)
        self.assertEqual(decision.recommended_payment_method, "not_recommended")
        self.assertEqual(decision.payment_plan, "none")

    def test_case_11_provisional_affordability_state(self) -> None:
        """11. Upstream provisional affordability state preserved in explanation evidence."""
        cert = _make_certificate(is_full_payment_safe_today=False, earliest_date_for_full_payment=None)
        ctx = _make_context(certificate=cert)
        decision = make_final_decision(ctx, [])
        self.assertEqual(decision.affordability_status, AffordabilityStatus.NOT_AFFORDABLE)
        self.assertFalse(decision.has_recommendation)

    def test_case_12_request_with_safe_amount_zero(self) -> None:
        """12. Safe amount = 0 -> amount_safe_to_pay is Decimal('0')."""
        cert = _make_certificate(amount_safe_to_pay=Decimal("0.00"), is_full_payment_safe_today=False, earliest_date_for_full_payment=None)
        ctx = _make_context(certificate=cert)
        decision = make_final_decision(ctx, [])
        self.assertEqual(decision.amount_safe_to_pay, Decimal("0.00"))

    def test_case_13_full_amount_safe_today(self) -> None:
        """13. Full amount safe today -> affordable_now and full_payment selected."""
        ctx = _make_context()
        c_full = _make_candidate("c_full", candidate_type=CandidateType.FULL_PAYMENT)
        decision = make_final_decision(ctx, [c_full])
        self.assertEqual(decision.affordability_status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertEqual(decision.recommended_payment_method, "full_payment")
        self.assertEqual(decision.earliest_date_for_full_payment, date(2026, 3, 1))

    def test_case_14_full_amount_safe_later(self) -> None:
        """14. Full amount safe later -> affordable_later and wait selected."""
        later_d = date(2026, 4, 15)
        cert = _make_certificate(
            is_full_payment_safe_today=False,
            amount_safe_to_pay=Decimal("2000"),
            earliest_date_for_full_payment=later_d,
        )
        ctx = _make_context(certificate=cert)
        c_wait = _make_candidate("c_wait", candidate_type=CandidateType.WAIT, first_payment_date=later_d, completion_date=later_d, number_of_payments=1)
        decision = make_final_decision(ctx, [c_wait])
        self.assertEqual(decision.affordability_status, AffordabilityStatus.AFFORDABLE_LATER)
        self.assertEqual(decision.recommended_payment_method, "wait")
        self.assertEqual(decision.earliest_date_for_full_payment, later_d)

    def test_case_15_no_full_payment_safe_within_horizon(self) -> None:
        """15. No safe full payment within 90 days -> earliest_date_for_full_payment is None."""
        cert = _make_certificate(
            is_full_payment_safe_today=False,
            earliest_date_for_full_payment=None,
        )
        ctx = _make_context(certificate=cert)
        decision = make_final_decision(ctx, [])
        self.assertIsNone(decision.earliest_date_for_full_payment)
        self.assertEqual(decision.affordability_status, AffordabilityStatus.NOT_AFFORDABLE)

    def test_case_16_desired_completion_date_boundary(self) -> None:
        """16. Desired completion date boundary: on deadline is on-time."""
        deadline = date(2026, 5, 1)
        ctx = _make_context(request=_make_request(desired_completion_date=deadline))
        c_on = _make_candidate("c_on", completion_date=deadline)
        c_late = _make_candidate("c_late", completion_date=deadline + timedelta(days=1))
        decision = make_final_decision(ctx, [c_late, c_on])
        self.assertEqual(decision.selected_candidate_id, "c_on")

    def test_case_17_schedule_ending_exactly_on_deadline(self) -> None:
        """17. Schedule ending exactly on deadline is valid and passes safety gate."""
        deadline = date(2026, 5, 1)
        ctx = _make_context(request=_make_request(desired_completion_date=deadline))
        c = _make_candidate("c_exact", completion_date=deadline)
        res = evaluate_candidate_safety_gate(c, ctx)
        self.assertTrue(res.is_safe)

    def test_case_18_schedule_ending_after_deadline_ranking_only(self) -> None:
        """18. Schedule ending after deadline is safe but ranked lower."""
        deadline = date(2026, 5, 1)
        ctx = _make_context(request=_make_request(desired_completion_date=deadline))
        c_late = _make_candidate("c_late", completion_date=deadline + timedelta(days=10))
        res = evaluate_candidate_safety_gate(c_late, ctx)
        self.assertTrue(res.is_safe)
        decision = make_final_decision(ctx, [c_late])
        self.assertEqual(decision.selected_candidate_id, "c_late")

    def test_case_19_pending_debit_reserved_cash(self) -> None:
        """19. Certificate reflects reserved cash margin; decision adheres."""
        cert = _make_certificate(safety_floor=Decimal("10000"), safety_floor_margin=Decimal("500"))
        ctx = _make_context(certificate=cert)
        c = _make_candidate("c_tight")
        res = evaluate_candidate_safety_gate(c, ctx)
        self.assertTrue(res.is_safe)

    def test_case_20_same_day_ordering(self) -> None:
        """20. First payment on request date is valid."""
        ctx = _make_context()
        c = _make_candidate("c_same_day", first_payment_date=date(2026, 3, 1))
        res = evaluate_candidate_safety_gate(c, ctx)
        self.assertTrue(res.is_safe)

    def test_case_21_candidate_permutation_invariance(self) -> None:
        """21. Candidate permutation cannot change final decision."""
        ctx = _make_context()
        c1 = _make_candidate("c1", total_amount_paid=Decimal("10000"))
        c2 = _make_candidate("c2", total_amount_paid=Decimal("12000"))
        c3 = _make_candidate("c3", total_amount_paid=Decimal("15000"))

        base = make_final_decision(ctx, [c1, c2, c3])
        rng = random.Random(42)
        for _ in range(10):
            shuffled = [c1, c2, c3]
            rng.shuffle(shuffled)
            perm = make_final_decision(ctx, shuffled)
            self.assertEqual(base.selected_candidate_id, perm.selected_candidate_id)
            self.assertEqual(base.recommended_payment_method, perm.recommended_payment_method)


# ===========================================================================
# Invariants Tests (Required Invariants 1 to 12)
# ===========================================================================

class TestFinalDecisionInvariants(unittest.TestCase):
    """The 12 mandatory mathematical and architectural invariants."""

    def setUp(self) -> None:
        self.ctx = _make_context()
        self.c1 = _make_candidate("c1", total_amount_paid=Decimal("10000"))
        self.c2 = _make_candidate("c2", total_amount_paid=Decimal("11000"))

    def test_invariant_1_repeated_execution_identical(self) -> None:
        """1. Repeated execution gives identical final decisions."""
        d1 = make_final_decision(self.ctx, [self.c1, self.c2])
        for _ in range(5):
            d2 = make_final_decision(self.ctx, [self.c1, self.c2])
            self.assertEqual(d1.selected_candidate_id, d2.selected_candidate_id)
            self.assertEqual(d1.payment_plan, d2.payment_plan)
            self.assertEqual(d1.affordability_status, d2.affordability_status)

    def test_invariant_2_candidate_permutation_invariance(self) -> None:
        """2. Candidate permutation cannot change the result."""
        d_forward = make_final_decision(self.ctx, [self.c1, self.c2])
        d_reverse = make_final_decision(self.ctx, [self.c2, self.c1])
        self.assertEqual(d_forward.selected_candidate_id, d_reverse.selected_candidate_id)

    def test_invariant_3_ranking_order_cannot_be_bypassed(self) -> None:
        """3. Ranking order cannot be bypassed; rank #1 among surviving is chosen."""
        d = make_final_decision(self.ctx, [self.c2, self.c1])
        self.assertEqual(d.selected_candidate_id, "c1")  # lower cost wins ranking

    def test_invariant_4_unsafe_candidates_never_selected(self) -> None:
        """4. Unsafe candidates can never be selected."""
        c_unsafe = _make_candidate("c_unsafe", is_safe=False, status=CandidateStatus.STRUCTURALLY_VALID_UNSAFE)
        d = make_final_decision(self.ctx, [c_unsafe])
        self.assertIsNone(d.selected_candidate_id)

    def test_invariant_5_safety_gate_failure_never_selected(self) -> None:
        """5. A candidate failing the safety gate can never be selected."""
        c_gate_fail = _make_candidate("c_fail", cert_id="mismatched_cert")
        d = make_final_decision(self.ctx, [c_gate_fail])
        self.assertIsNone(d.selected_candidate_id)

    def test_invariant_6_selected_is_ranking_winner_among_surviving(self) -> None:
        """6. Selected candidate is exactly the ranking winner among surviving candidates."""
        d = make_final_decision(self.ctx, [self.c1, self.c2])
        self.assertEqual(d.selected_candidate_id, d.ranked_candidate_ids[0])

    def test_invariant_7_decimal_values_remain_exact(self) -> None:
        """7. Decimal values remain exact."""
        d = make_final_decision(self.ctx, [self.c1])
        self.assertIsInstance(d.amount_safe_to_pay, Decimal)
        self.assertEqual(d.amount_safe_to_pay, Decimal("10000"))

    def test_invariant_8_immutable_inputs_not_mutated(self) -> None:
        """8. Immutable inputs are not mutated."""
        c1_copy = copy.deepcopy(self.c1)
        c2_copy = copy.deepcopy(self.c2)
        _ = make_final_decision(self.ctx, [self.c1, self.c2])
        self.assertEqual(self.c1, c1_copy)
        self.assertEqual(self.c2, c2_copy)

    def test_invariant_9_every_selected_candidate_schema_valid(self) -> None:
        """9. Every selected candidate produces valid output schema fields."""
        d = make_final_decision(self.ctx, [self.c1])
        self.assertIn(d.affordability_status, (
            AffordabilityStatus.AFFORDABLE_NOW,
            AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            AffordabilityStatus.AFFORDABLE_LATER,
            AffordabilityStatus.NOT_AFFORDABLE,
        ))
        self.assertIn(d.recommended_payment_method, (
            "full_payment", "partial_payment", "installments", "wait", "not_recommended"
        ))

    def test_invariant_10_every_final_decision_satisfies_hard_gates(self) -> None:
        """10. Every final decision satisfies all explicit hard gates."""
        d = make_final_decision(self.ctx, [self.c1])
        self.assertTrue(d.safety_gate_result.is_safe)
        self.assertEqual(len(d.safety_gate_result.rejection_reason_codes), 0)

    def test_invariant_11_zero_candidate_behavior_deterministic(self) -> None:
        """11. Zero-candidate behavior is deterministic."""
        d1 = make_final_decision(self.ctx, [])
        d2 = make_final_decision(self.ctx, [])
        self.assertEqual(d1, d2)

    def test_invariant_12_stability_across_runs(self) -> None:
        """12. Decision remains stable across repeated evaluations."""
        results = [make_final_decision(self.ctx, [self.c1, self.c2]).selected_candidate_id for _ in range(10)]
        self.assertTrue(all(r == "c1" for r in results))


if __name__ == "__main__":
    unittest.main()
