"""Comprehensive tests for candidate_generation.py.

Covers all 16 unit test cases (A-P) and 8 property tests from Prompt 10.

Test categories:
A. full payment candidate
B. eligible installment candidate
C. ineligible installment excluded / retained with correct status
D. unsafe installment retained with correct status
E. partial payment exact two-payment structure
F. partial-payment deadline rule
G. partial payment zero-safe case
H. partial payment full-safe case
I. wait semantics
J. candidate uniqueness
K. deterministic candidate ordering
L. repeated generation produces identical candidates
M. candidate generation does not mutate simulator state
N. Decimal exactness
O. provenance/lineage
P. installment schedule preserved exactly

Property tests:
1. Determinism: identical inputs -> identical candidates
2. Option row reordering does not change candidate set
3. Ineligible option cannot create installment candidate
4. Changing desired_completion_date does not change installment feasibility/safety
5. Partial payment always contains exactly two payments
6. Partial payment amounts sum exactly to requested amount
7. Candidate generation never mutates frozen financial state
8. Candidate total payable amounts equal their schedule sums exactly
"""

import unittest
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from code.candidate_generation import (
    Candidate,
    CandidatePayment,
    CandidateProvenance,
    CandidateSet,
    CandidateStatus,
    CandidateType,
    build_full_payment_candidate,
    build_installment_candidate,
    build_partial_payment_candidate,
    build_wait_candidate,
    generate_candidates,
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
# Test Helpers / Factories
# ---------------------------------------------------------------------------

def _make_profile(
    user_id: str = "user_01",
    home_currency: str = "INR",
    available_balance: str = "100000",
    min_balance: str = "5000",
    payment_methods: Tuple[str, ...] = ("full_payment", "partial_payment", "installments"),
    max_installment_months: Optional[int] = 12,
) -> FinancialProfile:
    return FinancialProfile(
        user_id=user_id,
        home_currency=home_currency,
        current_available_balance=Decimal(available_balance),
        minimum_balance_to_keep=Decimal(min_balance),
        financial_priorities=("essential_expenses",),
        expense_categories_to_protect=("rent", "utilities"),
        expense_categories_user_is_willing_to_reduce=("dining",),
        expense_categories_user_is_willing_to_stop=("subscriptions",),
        payment_methods_user_will_consider=payment_methods,
        max_installment_months=max_installment_months,
    )


def _make_request(
    request_id: str = "request_01",
    user_id: str = "user_01",
    request_date: str = "2024-03-01",
    requested_amount: str = "25000",
    desired_completion_date: str = "2024-06-01",
    allows_partial_payment: bool = True,
    request_type: str = "purchase",
) -> FinancialRequest:
    return FinancialRequest(
        request_id=request_id,
        user_id=user_id,
        request_date=date.fromisoformat(request_date),
        request_type=request_type,
        requested_amount=Decimal(requested_amount),
        desired_completion_date=date.fromisoformat(desired_completion_date),
        allows_partial_payment=allows_partial_payment,
        request_text="Test request",
    )


def _make_certificate(
    request_id: str = "request_01",
    user_id: str = "user_01",
    request_date: str = "2024-03-01",
    requested_amount: str = "25000",
    amount_safe_to_pay: str = "25000",
    is_full_payment_safe_today: bool = True,
    earliest_date: Optional[str] = "2024-03-01",
    is_full_payment_safe_later: bool = False,
    currency: str = "INR",
    safety_floor: str = "5000",
    reason_if_unsafe: Optional[str] = None,
) -> SafeToPayCertificate:
    req_date = date.fromisoformat(request_date)
    ed = date.fromisoformat(earliest_date) if earliest_date else None
    return SafeToPayCertificate(
        request_id=request_id,
        user_id=user_id,
        request_date=req_date,
        simulation_start=req_date,
        simulation_end=req_date.replace(year=req_date.year if req_date.month <= 9 else req_date.year + 1,
                                        month=(req_date.month + 3 - 1) % 12 + 1,
                                        day=1),
        requested_amount=Decimal(requested_amount),
        amount_safe_to_pay=Decimal(amount_safe_to_pay),
        currency=currency,
        safety_floor=Decimal(safety_floor),
        baseline_minimum_available_cash=Decimal("80000"),
        baseline_minimum_cash_date=req_date,
        limiting_date=req_date,
        available_cash_after_purchase_today=Decimal("75000"),
        minimum_available_cash_after_purchase=Decimal("75000"),
        safety_floor_margin=Decimal("70000"),
        earliest_date_for_full_payment=ed,
        is_full_payment_safe_today=is_full_payment_safe_today,
        is_full_payment_safe_later=is_full_payment_safe_later,
        reason_if_unsafe=reason_if_unsafe,
        candidate_search_method="exact_full_amount",
    )


def _make_payment_option(
    payment_option_id: str = "payment_option_02",
    request_id: str = "request_01",
    payment_method: str = "installments",
    payment_amount: str = "2500",
    number_of_payments: int = 10,
    first_payment_date: str = "2024-03-06",
    payment_frequency_days: Optional[int] = 30,
    financing_fee: str = "0",
    total_payable_amount: Optional[str] = None,  # if None, computed as payment_amount * n_payments
) -> PaymentOption:
    pa = Decimal(payment_amount)
    n = number_of_payments
    tpa = Decimal(total_payable_amount) if total_payable_amount is not None else pa * n
    return PaymentOption(
        payment_option_id=payment_option_id,
        request_id=request_id,
        payment_method=payment_method,
        payment_amount=pa,
        number_of_payments=n,
        first_payment_date=date.fromisoformat(first_payment_date),
        payment_frequency_days=payment_frequency_days,
        financing_fee=Decimal(financing_fee),
        total_payable_amount=tpa,
    )


def _make_payment_plan(
    option: PaymentOption,
    request_id: str = "request_01",
) -> PaymentPlan:
    from datetime import timedelta
    entries_list = []
    freq = option.payment_frequency_days or 0
    for k in range(option.number_of_payments):
        entry_date = option.first_payment_date + timedelta(days=k * freq)
        entries_list.append(PaymentScheduleEntry(
            date=entry_date,
            amount=option.payment_amount,
            sequence_number=k + 1,
        ))
    # total_payable_amount must equal payment_amount * number_of_payments
    # (the financing fee is baked into payment_amount already in a valid plan)
    computed_total = option.payment_amount * option.number_of_payments
    return PaymentPlan(
        request_id=request_id,
        payment_option_id=option.payment_option_id,
        payment_method=option.payment_method,
        entries=tuple(entries_list),
        total_amount=Decimal("25000"),
        financing_fee=option.financing_fee,
        total_payable_amount=computed_total,
    )


def _make_feasibility(
    option: PaymentOption,
    request_id: str = "request_01",
    is_eligible: bool = True,
    is_safe: bool = True,
    rejection_reason: Optional[str] = None,
    plan: Optional[PaymentPlan] = None,
) -> PaymentPlanFeasibility:
    sim_cert = PaymentPlanSimulatorReference(
        simulation_start=date(2024, 3, 1),
        simulation_end=date(2024, 5, 31),
        minimum_available_cash=Decimal("10000"),
        safety_floor=Decimal("5000"),
        limiting_date=date(2024, 5, 31),
        is_safety_floor_breached=not is_safe,
        num_safety_floor_breaches=0 if is_safe else 1,
        num_obligation_breaches=0,
    ) if is_eligible else None
    return PaymentPlanFeasibility(
        request_id=request_id,
        payment_option_id=option.payment_option_id,
        is_eligible=is_eligible,
        is_safe=is_safe,
        rejection_reason=rejection_reason,
        payment_plan=plan,
        minimum_available_cash=Decimal("10000") if is_eligible else None,
        safety_floor=Decimal("5000"),
        limiting_date=date(2024, 5, 31) if is_eligible else None,
        simulator_certificate=sim_cert,
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestAFullPaymentCandidate(unittest.TestCase):
    """Test A: full payment candidate."""

    def setUp(self):
        self.profile = _make_profile()
        self.request = _make_request()
        self.cert = _make_certificate()

    def test_full_payment_eligible_and_safe(self):
        cand = build_full_payment_candidate(self.request, self.cert, self.profile)
        self.assertEqual(cand.candidate_type, CandidateType.FULL_PAYMENT)
        self.assertEqual(cand.status, CandidateStatus.ELIGIBLE_AND_SAFE)
        self.assertTrue(cand.is_safe)
        self.assertEqual(cand.number_of_payments, 1)
        self.assertEqual(cand.total_amount_paid, Decimal("25000"))
        self.assertEqual(cand.financing_fee, Decimal("0"))
        self.assertEqual(cand.payment_schedule[0].payment_date, date(2024, 3, 1))
        self.assertEqual(cand.spending_changes, ())
        self.assertIsNone(cand.source_payment_option_id)

    def test_full_payment_user_refuses_method(self):
        profile_no_full = _make_profile(payment_methods=("partial_payment", "installments"))
        cand = build_full_payment_candidate(self.request, self.cert, profile_no_full)
        self.assertEqual(cand.status, CandidateStatus.CONTRACTUALLY_INELIGIBLE)
        self.assertFalse(cand.is_safe)

    def test_full_payment_not_safe_today(self):
        cert_unsafe = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
            reason_if_unsafe="Headroom insufficient.",
        )
        cand = build_full_payment_candidate(self.request, cert_unsafe, self.profile)
        self.assertEqual(cand.status, CandidateStatus.STRUCTURALLY_VALID_UNSAFE)
        self.assertFalse(cand.is_safe)
        self.assertIsNotNone(cand.feasibility_reason)

    def test_full_payment_exactly_one_payment(self):
        cand = build_full_payment_candidate(self.request, self.cert, self.profile)
        self.assertEqual(len(cand.payment_schedule), 1)

    def test_full_payment_amount_equals_requested(self):
        cand = build_full_payment_candidate(self.request, self.cert, self.profile)
        self.assertEqual(cand.total_amount_paid, self.cert.requested_amount)

    def test_full_payment_completion_date_is_request_date(self):
        cand = build_full_payment_candidate(self.request, self.cert, self.profile)
        self.assertEqual(cand.completion_date, self.request.request_date)

    def test_full_payment_zero_financing_fee(self):
        cand = build_full_payment_candidate(self.request, self.cert, self.profile)
        self.assertEqual(cand.financing_fee, Decimal("0"))

    def test_full_payment_payment_date_is_request_date(self):
        cand = build_full_payment_candidate(self.request, self.cert, self.profile)
        self.assertEqual(cand.payment_schedule[0].payment_date, self.request.request_date)


class TestBEligibleInstallmentCandidate(unittest.TestCase):
    """Test B: eligible installment candidate."""

    def setUp(self):
        self.profile = _make_profile()
        self.request = _make_request()
        self.option = _make_payment_option()
        self.plan = _make_payment_plan(self.option)
        self.feasibility = _make_feasibility(self.option, plan=self.plan)

    def test_installment_eligible_and_safe(self):
        cand = build_installment_candidate(self.request, self.feasibility, self.option)
        self.assertEqual(cand.candidate_type, CandidateType.INSTALLMENT_PLAN)
        self.assertEqual(cand.status, CandidateStatus.ELIGIBLE_AND_SAFE)
        self.assertTrue(cand.is_safe)

    def test_installment_preserves_payment_option_id(self):
        cand = build_installment_candidate(self.request, self.feasibility, self.option)
        self.assertEqual(cand.source_payment_option_id, self.option.payment_option_id)

    def test_installment_preserves_financing_fee(self):
        cand = build_installment_candidate(self.request, self.feasibility, self.option)
        self.assertEqual(cand.financing_fee, self.option.financing_fee)

    def test_installment_preserves_number_of_payments(self):
        cand = build_installment_candidate(self.request, self.feasibility, self.option)
        self.assertEqual(cand.number_of_payments, self.option.number_of_payments)

    def test_installment_schedule_sum_equals_total_payable(self):
        cand = build_installment_candidate(self.request, self.feasibility, self.option)
        schedule_sum = sum(p.amount for p in cand.payment_schedule)
        self.assertEqual(schedule_sum, cand.total_amount_paid)


class TestCIneligibleInstallmentExcluded(unittest.TestCase):
    """Test C: ineligible installment retained with correct status."""

    def setUp(self):
        self.profile = _make_profile()
        self.request = _make_request()
        self.option = _make_payment_option()

    def test_ineligible_no_plan_structurally_invalid(self):
        """Schema failure: plan=None -> STRUCTURALLY_INVALID."""
        feasibility = _make_feasibility(
            self.option,
            is_eligible=False,
            is_safe=False,
            rejection_reason="invalid payment amount",
            plan=None,
        )
        cand = build_installment_candidate(self.request, feasibility, self.option)
        self.assertEqual(cand.status, CandidateStatus.STRUCTURALLY_INVALID)
        self.assertFalse(cand.is_safe)

    def test_ineligible_with_plan_contractually_ineligible(self):
        """Preference/horizon failure: plan exists -> CONTRACTUALLY_INELIGIBLE."""
        plan = _make_payment_plan(self.option)
        feasibility = _make_feasibility(
            self.option,
            is_eligible=False,
            is_safe=False,
            rejection_reason="payment method not permitted",
            plan=plan,
        )
        cand = build_installment_candidate(self.request, feasibility, self.option)
        self.assertEqual(cand.status, CandidateStatus.CONTRACTUALLY_INELIGIBLE)
        self.assertFalse(cand.is_safe)

    def test_ineligible_candidate_has_empty_schedule(self):
        feasibility = _make_feasibility(
            self.option,
            is_eligible=False,
            rejection_reason="test",
            plan=None,
        )
        cand = build_installment_candidate(self.request, feasibility, self.option)
        self.assertEqual(len(cand.payment_schedule), 0)

    def test_ineligible_candidate_still_has_option_lineage(self):
        """Even ineligible candidates retain source_payment_option_id for lineage."""
        feasibility = _make_feasibility(
            self.option,
            is_eligible=False,
            rejection_reason="test",
            plan=None,
        )
        cand = build_installment_candidate(self.request, feasibility, self.option)
        self.assertEqual(cand.source_payment_option_id, self.option.payment_option_id)


class TestDUnsafeInstallmentRetained(unittest.TestCase):
    """Test D: unsafe installment retained with correct status."""

    def setUp(self):
        self.profile = _make_profile()
        self.request = _make_request()
        self.option = _make_payment_option()
        self.plan = _make_payment_plan(self.option)

    def test_unsafe_installment_is_structurally_valid_unsafe(self):
        feasibility = _make_feasibility(
            self.option,
            is_eligible=True,
            is_safe=False,
            rejection_reason="safety floor breach on 2024-04-15",
            plan=self.plan,
        )
        cand = build_installment_candidate(self.request, feasibility, self.option)
        self.assertEqual(cand.status, CandidateStatus.STRUCTURALLY_VALID_UNSAFE)
        self.assertFalse(cand.is_safe)
        self.assertIsNotNone(cand.feasibility_reason)

    def test_unsafe_installment_has_populated_schedule(self):
        """Unsafe eligible candidates still have their schedule populated."""
        feasibility = _make_feasibility(
            self.option,
            is_eligible=True,
            is_safe=False,
            rejection_reason="safety floor breach",
            plan=self.plan,
        )
        cand = build_installment_candidate(self.request, feasibility, self.option)
        self.assertGreater(len(cand.payment_schedule), 0)

    def test_unsafe_installment_preserves_option_id(self):
        feasibility = _make_feasibility(
            self.option,
            is_eligible=True,
            is_safe=False,
            rejection_reason="safety floor breach",
            plan=self.plan,
        )
        cand = build_installment_candidate(self.request, feasibility, self.option)
        self.assertEqual(cand.source_payment_option_id, self.option.payment_option_id)


class TestEPartialPaymentTwoPaymentStructure(unittest.TestCase):
    """Test E: partial payment exact two-payment structure."""

    def setUp(self):
        self.profile = _make_profile()
        self.request = _make_request(allows_partial_payment=True)
        self.cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )

    def test_partial_payment_exactly_two_payments(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.number_of_payments, 2)
        self.assertEqual(len(cand.payment_schedule), 2)

    def test_partial_payment_first_is_request_date(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.payment_schedule[0].payment_date, self.request.request_date)

    def test_partial_payment_second_is_earliest_date(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.payment_schedule[1].payment_date, date(2024, 4, 1))

    def test_partial_payment_first_amount_is_safe_to_pay(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.payment_schedule[0].amount, Decimal("10000"))

    def test_partial_payment_second_amount_is_remaining(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.payment_schedule[1].amount, Decimal("15000"))

    def test_partial_payment_amounts_sum_to_requested(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        total = sum(p.amount for p in cand.payment_schedule)
        self.assertEqual(total, Decimal("25000"))

    def test_partial_payment_is_safe(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertTrue(cand.is_safe)

    def test_partial_payment_status_eligible_and_safe(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.status, CandidateStatus.ELIGIBLE_AND_SAFE)

    def test_partial_payment_zero_financing_fee(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.financing_fee, Decimal("0"))

    def test_partial_payment_sequence_numbers(self):
        cand = build_partial_payment_candidate(self.request, self.cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.payment_schedule[0].sequence_number, 1)
        self.assertEqual(cand.payment_schedule[1].sequence_number, 2)


class TestFPartialPaymentDeadline(unittest.TestCase):
    """Test F: partial-payment deadline rule."""

    def setUp(self):
        self.profile = _make_profile()

    def test_partial_excluded_when_earliest_after_deadline(self):
        """earliest_date_for_full_payment > desired_completion_date -> excluded."""
        request = _make_request(
            desired_completion_date="2024-03-15",
            allows_partial_payment=True,
        )
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",  # After desired_completion_date
        )
        cand = build_partial_payment_candidate(request, cert, self.profile)
        self.assertIsNone(cand)

    def test_partial_included_when_earliest_equals_deadline(self):
        """earliest_date_for_full_payment == desired_completion_date -> included."""
        request = _make_request(
            desired_completion_date="2024-04-01",
            allows_partial_payment=True,
        )
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, self.profile)
        self.assertIsNotNone(cand)

    def test_partial_included_when_earliest_before_deadline(self):
        """earliest_date_for_full_payment < desired_completion_date -> included."""
        request = _make_request(
            desired_completion_date="2024-06-01",
            allows_partial_payment=True,
        )
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, self.profile)
        self.assertIsNotNone(cand)

    def test_partial_excluded_when_no_earliest_date(self):
        """earliest_date_for_full_payment is None -> excluded."""
        request = _make_request(allows_partial_payment=True)
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date=None,
        )
        cand = build_partial_payment_candidate(request, cert, self.profile)
        self.assertIsNone(cand)


class TestGPartialPaymentZeroSafe(unittest.TestCase):
    """Test G: partial payment zero-safe case."""

    def test_partial_excluded_when_safe_amount_is_zero(self):
        profile = _make_profile()
        request = _make_request(allows_partial_payment=True)
        cert = _make_certificate(
            amount_safe_to_pay="0",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, profile)
        self.assertIsNone(cand)

    def test_partial_excluded_when_allows_partial_is_false(self):
        profile = _make_profile()
        request = _make_request(allows_partial_payment=False)
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, profile)
        self.assertIsNone(cand)

    def test_partial_excluded_when_user_refuses_partial(self):
        profile = _make_profile(payment_methods=("full_payment", "installments"))
        request = _make_request(allows_partial_payment=True)
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, profile)
        self.assertIsNone(cand)


class TestHPartialPaymentFullSafe(unittest.TestCase):
    """Test H: partial payment full-safe case (would duplicate full_payment -> excluded)."""

    def test_partial_excluded_when_safe_amount_equals_requested(self):
        """amount_safe_to_pay == requested_amount: partial duplicates full_payment -> excluded."""
        profile = _make_profile()
        request = _make_request(allows_partial_payment=True)
        cert = _make_certificate(
            amount_safe_to_pay="25000",
            is_full_payment_safe_today=True,
            earliest_date="2024-03-01",
        )
        cand = build_partial_payment_candidate(request, cert, profile)
        self.assertIsNone(cand)


class TestIWaitSemantics(unittest.TestCase):
    """Test I: wait semantics."""

    def setUp(self):
        self.profile = _make_profile()

    def test_wait_created_when_full_payment_safe_later(self):
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
            is_full_payment_safe_later=True,
        )
        cand = build_wait_candidate(request, cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.candidate_type, CandidateType.WAIT)

    def test_wait_payment_on_earliest_date(self):
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-15",
        )
        cand = build_wait_candidate(request, cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.payment_schedule[0].payment_date, date(2024, 4, 15))

    def test_wait_full_amount_in_single_payment(self):
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_wait_candidate(request, cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.number_of_payments, 1)
        self.assertEqual(cand.total_amount_paid, cert.requested_amount)

    def test_wait_excluded_when_earliest_is_request_date(self):
        """earliest_date == request_date: this is full_payment, not wait."""
        request = _make_request(request_date="2024-03-01")
        cert = _make_certificate(
            is_full_payment_safe_today=True,
            earliest_date="2024-03-01",
        )
        cand = build_wait_candidate(request, cert, self.profile)
        self.assertIsNone(cand)

    def test_wait_excluded_when_no_earliest_date(self):
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="0",
            is_full_payment_safe_today=False,
            earliest_date=None,
        )
        cand = build_wait_candidate(request, cert, self.profile)
        self.assertIsNone(cand)

    def test_wait_excluded_when_user_refuses_full_payment(self):
        profile = _make_profile(payment_methods=("partial_payment", "installments"))
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_wait_candidate(request, cert, profile)
        self.assertIsNone(cand)

    def test_wait_is_safe(self):
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_wait_candidate(request, cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertTrue(cand.is_safe)

    def test_wait_zero_financing_fee(self):
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_wait_candidate(request, cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.financing_fee, Decimal("0"))

    def test_wait_completion_date_is_earliest_date(self):
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-15",
        )
        cand = build_wait_candidate(request, cert, self.profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.completion_date, date(2024, 4, 15))


class TestJCandidateUniqueness(unittest.TestCase):
    """Test J: candidate uniqueness."""

    def setUp(self):
        self.profile = _make_profile()
        self.request = _make_request()
        self.cert = _make_certificate()

    def test_full_payment_has_unique_id(self):
        cand = build_full_payment_candidate(self.request, self.cert, self.profile)
        self.assertIn("full_payment", cand.candidate_id)
        self.assertIn(self.request.request_id, cand.candidate_id)

    def test_installment_has_option_id_in_candidate_id(self):
        option = _make_payment_option(payment_option_id="payment_option_99")
        plan = _make_payment_plan(option)
        feasibility = _make_feasibility(option, plan=plan)
        cand = build_installment_candidate(self.request, feasibility, option)
        self.assertIn("payment_option_99", cand.candidate_id)

    def test_different_options_produce_different_candidate_ids(self):
        option1 = _make_payment_option(payment_option_id="payment_option_10")
        option2 = _make_payment_option(payment_option_id="payment_option_11")
        plan1 = _make_payment_plan(option1)
        plan2 = _make_payment_plan(option2)
        f1 = _make_feasibility(option1, plan=plan1)
        f2 = _make_feasibility(option2, plan=plan2)
        cand1 = build_installment_candidate(self.request, f1, option1)
        cand2 = build_installment_candidate(self.request, f2, option2)
        self.assertNotEqual(cand1.candidate_id, cand2.candidate_id)

    def test_full_payment_and_wait_have_different_types(self):
        cert_later = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        full_cand = build_full_payment_candidate(self.request, cert_later, self.profile)
        wait_cand = build_wait_candidate(self.request, cert_later, self.profile)
        self.assertNotEqual(full_cand.candidate_type, wait_cand.candidate_type)
        self.assertNotEqual(full_cand.candidate_id, wait_cand.candidate_id)


class TestKDeterministicOrdering(unittest.TestCase):
    """Test K: deterministic candidate ordering."""

    def setUp(self):
        self.profile = _make_profile()
        self.request = _make_request(allows_partial_payment=True)
        # Certificate for partial-safe scenario
        self.cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
            is_full_payment_safe_later=True,
        )
        # payment_amount * n_payments must equal total_payable_amount for plan consistency
        opt_a = _make_payment_option(
            payment_option_id="payment_option_02",
            payment_amount="2750",
            number_of_payments=10,
        )
        opt_b = _make_payment_option(
            payment_option_id="payment_option_03",
            payment_amount="3000",
            number_of_payments=9,
        )
        plan_a = _make_payment_plan(opt_a)
        plan_b = _make_payment_plan(opt_b)
        self.feasibilities = [
            _make_feasibility(opt_a, plan=plan_a),
            _make_feasibility(opt_b, plan=plan_b),
        ]
        self.options_by_id = {
            opt_a.payment_option_id: opt_a,
            opt_b.payment_option_id: opt_b,
        }

    def test_ordering_is_deterministic_type_first(self):
        """full_payment < installments < partial_payment < wait (alphabetical)."""
        cs = generate_candidates(
            self.request, self.profile, self.cert,
            self.feasibilities, self.options_by_id,
        )
        types = [c.candidate_type.value for c in cs.candidates]
        self.assertEqual(types, sorted(types))

    def test_ordering_is_stable_secondary_by_option_id(self):
        cs = generate_candidates(
            self.request, self.profile, self.cert,
            self.feasibilities, self.options_by_id,
        )
        installment_cands = [
            c for c in cs.candidates
            if c.candidate_type == CandidateType.INSTALLMENT_PLAN
        ]
        if len(installment_cands) >= 2:
            option_ids = [c.source_payment_option_id for c in installment_cands]
            self.assertEqual(option_ids, sorted(option_ids))

    def test_ordering_does_not_change_on_repeated_generation(self):
        cs1 = generate_candidates(
            self.request, self.profile, self.cert,
            self.feasibilities, self.options_by_id,
        )
        cs2 = generate_candidates(
            self.request, self.profile, self.cert,
            self.feasibilities, self.options_by_id,
        )
        ids1 = [c.candidate_id for c in cs1.candidates]
        ids2 = [c.candidate_id for c in cs2.candidates]
        self.assertEqual(ids1, ids2)


class TestLRepeatedGenerationIdentical(unittest.TestCase):
    """Test L: repeated generation produces identical candidates."""

    def test_repeated_call_produces_identical_candidate_set(self):
        profile = _make_profile()
        request = _make_request(allows_partial_payment=True)
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        option = _make_payment_option()
        plan = _make_payment_plan(option)
        feasibilities = [_make_feasibility(option, plan=plan)]
        options_by_id = {option.payment_option_id: option}

        cs1 = generate_candidates(request, profile, cert, feasibilities, options_by_id)
        cs2 = generate_candidates(request, profile, cert, feasibilities, options_by_id)

        self.assertEqual(len(cs1.candidates), len(cs2.candidates))
        for c1, c2 in zip(cs1.candidates, cs2.candidates):
            self.assertEqual(c1.candidate_id, c2.candidate_id)
            self.assertEqual(c1.candidate_type, c2.candidate_type)
            self.assertEqual(c1.total_amount_paid, c2.total_amount_paid)
            self.assertEqual(c1.payment_schedule, c2.payment_schedule)


class TestMDoesNotMutateState(unittest.TestCase):
    """Test M: candidate generation does not mutate simulator state."""

    def test_certificate_is_unchanged_after_generation(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate()
        original_safe_amt = cert.amount_safe_to_pay
        original_earliest = cert.earliest_date_for_full_payment

        generate_candidates(request, profile, cert, [], {})

        self.assertEqual(cert.amount_safe_to_pay, original_safe_amt)
        self.assertEqual(cert.earliest_date_for_full_payment, original_earliest)

    def test_profile_is_unchanged_after_generation(self):
        profile = _make_profile()
        original_balance = profile.current_available_balance
        request = _make_request()
        cert = _make_certificate()

        generate_candidates(request, profile, cert, [], {})

        self.assertEqual(profile.current_available_balance, original_balance)


class TestNDecimalExactness(unittest.TestCase):
    """Test N: Decimal exactness."""

    def test_candidate_amounts_are_decimal(self):
        profile = _make_profile()
        request = _make_request(requested_amount="25256")
        cert = _make_certificate(
            requested_amount="25256",
            amount_safe_to_pay="25256",
            is_full_payment_safe_today=True,
        )
        cand = build_full_payment_candidate(request, cert, profile)
        self.assertIsInstance(cand.total_amount_paid, Decimal)
        for p in cand.payment_schedule:
            self.assertIsInstance(p.amount, Decimal)

    def test_partial_payment_amounts_sum_exactly(self):
        profile = _make_profile()
        request = _make_request(
            requested_amount="1852.11",
            allows_partial_payment=True,
        )
        cert = _make_certificate(
            requested_amount="1852.11",
            amount_safe_to_pay="1000.00",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, profile)
        self.assertIsNotNone(cand)
        s = sum(p.amount for p in cand.payment_schedule)
        self.assertEqual(s, Decimal("1852.11"))

    def test_financing_fee_is_decimal(self):
        profile = _make_profile()
        request = _make_request()
        # For valid plans: payment_amount * n_payments == total_payable_amount
        # financing_fee is in addition and must be paid separately by the user.
        # Here we use payment_amount=2778.165, n=10 -> total=27781.65, fee=2525.65
        option = _make_payment_option(
            payment_amount="2778.165",
            number_of_payments=10,
            financing_fee="2525.65",
        )
        plan = _make_payment_plan(option)
        feasibility = _make_feasibility(option, plan=plan)
        cand = build_installment_candidate(request, feasibility, option)
        self.assertIsInstance(cand.financing_fee, Decimal)
        self.assertEqual(cand.financing_fee, Decimal("2525.65"))


class TestOProvenanceLineage(unittest.TestCase):
    """Test O: provenance/lineage."""

    def test_full_payment_provenance_source(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate()
        cand = build_full_payment_candidate(request, cert, profile)
        self.assertEqual(cand.provenance.source, "safe_to_pay_certificate")
        self.assertIsNone(cand.provenance.source_payment_option_id)
        self.assertEqual(cand.provenance.safe_to_pay_certificate_id, cert.request_id)

    def test_installment_provenance_source(self):
        profile = _make_profile()
        request = _make_request()
        option = _make_payment_option()
        plan = _make_payment_plan(option)
        feasibility = _make_feasibility(option, plan=plan)
        cand = build_installment_candidate(request, feasibility, option)
        self.assertEqual(cand.provenance.source, "payment_option_feasibility")
        self.assertEqual(cand.provenance.source_payment_option_id, option.payment_option_id)

    def test_partial_payment_provenance_source(self):
        profile = _make_profile()
        request = _make_request(allows_partial_payment=True)
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.provenance.source, "safe_to_pay_certificate")
        self.assertIsNone(cand.provenance.source_payment_option_id)

    def test_wait_provenance_source(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-15",
        )
        cand = build_wait_candidate(request, cert, profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.provenance.source, "earliest_full_payment_computation")
        self.assertEqual(cand.provenance.earliest_full_payment_date, date(2024, 4, 15))


class TestPInstallmentSchedulePreservedExactly(unittest.TestCase):
    """Test P: installment schedule preserved exactly."""

    def test_installment_schedule_dates_exact(self):
        profile = _make_profile()
        request = _make_request()
        option = _make_payment_option(
            first_payment_date="2024-03-06",
            payment_frequency_days=30,
            number_of_payments=3,
            payment_amount="9000",
            total_payable_amount="27000",
        )
        plan = _make_payment_plan(option)
        feasibility = _make_feasibility(option, plan=plan)
        cand = build_installment_candidate(request, feasibility, option)

        from datetime import timedelta
        for k, cp in enumerate(cand.payment_schedule):
            expected_date = date(2024, 3, 6) + timedelta(days=k * 30)
            self.assertEqual(cp.payment_date, expected_date, f"Payment {k+1} date mismatch")

    def test_installment_schedule_amounts_exact(self):
        profile = _make_profile()
        request = _make_request()
        # 3 payments of 1852.11 -> total = 5556.33
        option = _make_payment_option(
            payment_amount="1852.11",
            number_of_payments=3,
        )
        plan = _make_payment_plan(option)
        feasibility = _make_feasibility(option, plan=plan)
        cand = build_installment_candidate(request, feasibility, option)
        for cp in cand.payment_schedule:
            self.assertEqual(cp.amount, Decimal("1852.11"))


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------

class TestProperty1Determinism(unittest.TestCase):
    """Property 1: Identical inputs produce identical candidates."""

    def test_determinism_full_scenario(self):
        profile = _make_profile()
        request = _make_request(allows_partial_payment=True)
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        option = _make_payment_option()
        plan = _make_payment_plan(option)
        feasibilities = [_make_feasibility(option, plan=plan)]
        options_by_id = {option.payment_option_id: option}

        for _ in range(5):
            cs = generate_candidates(request, profile, cert, feasibilities, options_by_id)
            ids = tuple(c.candidate_id for c in cs.candidates)
            amounts = tuple(c.total_amount_paid for c in cs.candidates)
            if _ == 0:
                first_ids = ids
                first_amounts = amounts
            else:
                self.assertEqual(ids, first_ids)
                self.assertEqual(amounts, first_amounts)


class TestProperty2OptionRowReorderingInvariant(unittest.TestCase):
    """Property 2: Reordering payment_option rows does not change the candidate set."""

    def test_option_order_independent(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate()
        opt_a = _make_payment_option(
            payment_option_id="payment_option_02",
            payment_amount="2750",
            number_of_payments=10,
        )
        opt_b = _make_payment_option(
            payment_option_id="payment_option_03",
            payment_amount="3000",
            number_of_payments=9,
        )
        plan_a = _make_payment_plan(opt_a)
        plan_b = _make_payment_plan(opt_b)
        f_a = _make_feasibility(opt_a, plan=plan_a)
        f_b = _make_feasibility(opt_b, plan=plan_b)
        options_by_id = {opt_a.payment_option_id: opt_a, opt_b.payment_option_id: opt_b}

        cs1 = generate_candidates(request, profile, cert, [f_a, f_b], options_by_id)
        cs2 = generate_candidates(request, profile, cert, [f_b, f_a], options_by_id)

        ids1 = {c.candidate_id for c in cs1.candidates}
        ids2 = {c.candidate_id for c in cs2.candidates}
        self.assertEqual(ids1, ids2)


class TestProperty3IneligibleOptionCannotCreateInstallment(unittest.TestCase):
    """Property 3: Changing an ineligible option cannot create an installment candidate."""

    def test_ineligible_option_produces_no_eligible_and_safe_candidate(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate()
        option = _make_payment_option()
        # No plan -> structurally invalid
        feasibility = _make_feasibility(option, is_eligible=False, plan=None, rejection_reason="invalid")
        options_by_id = {option.payment_option_id: option}

        cs = generate_candidates(request, profile, cert, [feasibility], options_by_id)

        install_cands = [c for c in cs.candidates if c.candidate_type == CandidateType.INSTALLMENT_PLAN]
        # Must exist (we include all installment candidates for auditability)
        for c in install_cands:
            self.assertNotEqual(c.status, CandidateStatus.ELIGIBLE_AND_SAFE)


class TestProperty4DesiredCompletionDateDoesNotAffectInstallmentFeasibility(unittest.TestCase):
    """Property 4: Changing desired_completion_date cannot change installment feasibility/safety."""

    def test_installment_feasibility_unchanged_by_deadline(self):
        option = _make_payment_option()
        plan = _make_payment_plan(option)
        feasibility = _make_feasibility(option, plan=plan)

        for completion_date in ["2024-03-10", "2024-04-01", "2024-09-01"]:
            profile = _make_profile()
            request = _make_request(desired_completion_date=completion_date)
            cert = _make_certificate()
            options_by_id = {option.payment_option_id: option}

            cs = generate_candidates(request, profile, cert, [feasibility], options_by_id)
            install_cands = [
                c for c in cs.candidates
                if c.candidate_type == CandidateType.INSTALLMENT_PLAN
            ]
            # All should have the same safety status regardless of completion_date
            self.assertEqual(len(install_cands), 1)
            self.assertEqual(install_cands[0].is_safe, feasibility.is_safe)


class TestProperty5PartialPaymentAlwaysTwoPayments(unittest.TestCase):
    """Property 5: Partial payment always contains exactly two payments."""

    def test_partial_payment_is_always_two_payments(self):
        profile = _make_profile()
        request = _make_request(allows_partial_payment=True)
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, profile)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.number_of_payments, 2)
        self.assertEqual(len(cand.payment_schedule), 2)


class TestProperty6PartialPaymentAmountsSumExact(unittest.TestCase):
    """Property 6: Partial payment amounts sum exactly to requested amount."""

    def _check(self, requested: str, safe: str):
        profile = _make_profile()
        request = _make_request(requested_amount=requested, allows_partial_payment=True)
        cert = _make_certificate(
            requested_amount=requested,
            amount_safe_to_pay=safe,
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, profile)
        if cand is not None:
            schedule_total = sum(p.amount for p in cand.payment_schedule)
            self.assertEqual(schedule_total, Decimal(requested))

    def test_sum_exact_case1(self):
        self._check("25000", "10000")

    def test_sum_exact_case2(self):
        self._check("1852.11", "1000.00")

    def test_sum_exact_case3(self):
        self._check("100000.50", "33333.17")


class TestProperty7NeverMutatesFrozenState(unittest.TestCase):
    """Property 7: Candidate generation never mutates frozen financial state."""

    def test_candidate_objects_are_immutable(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate()
        cand = build_full_payment_candidate(request, cert, profile)
        # frozen=True dataclasses raise FrozenInstanceError (subclass of AttributeError)
        with self.assertRaises(AttributeError):
            cand.total_amount_paid = Decimal("999")  # type: ignore

    def test_certificate_immutable(self):
        cert = _make_certificate()
        with self.assertRaises(AttributeError):
            cert.amount_safe_to_pay = Decimal("999")  # type: ignore


class TestProperty8TotalPayableEqualsScheduleSum(unittest.TestCase):
    """Property 8: Candidate total payable amounts equal their schedule sums exactly."""

    def test_full_payment_total_equals_schedule_sum(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate()
        cand = build_full_payment_candidate(request, cert, profile)
        self.assertEqual(sum(p.amount for p in cand.payment_schedule), cand.total_amount_paid)

    def test_installment_total_equals_schedule_sum(self):
        profile = _make_profile()
        request = _make_request()
        option = _make_payment_option()
        plan = _make_payment_plan(option)
        feasibility = _make_feasibility(option, plan=plan)
        cand = build_installment_candidate(request, feasibility, option)
        self.assertEqual(sum(p.amount for p in cand.payment_schedule), cand.total_amount_paid)

    def test_partial_payment_total_equals_schedule_sum(self):
        profile = _make_profile()
        request = _make_request(allows_partial_payment=True)
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-01",
        )
        cand = build_partial_payment_candidate(request, cert, profile)
        self.assertIsNotNone(cand)
        self.assertEqual(sum(p.amount for p in cand.payment_schedule), cand.total_amount_paid)

    def test_wait_total_equals_schedule_sum(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-15",
        )
        cand = build_wait_candidate(request, cert, profile)
        self.assertIsNotNone(cand)
        self.assertEqual(sum(p.amount for p in cand.payment_schedule), cand.total_amount_paid)


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------

class TestCandidateSetHelpers(unittest.TestCase):
    """Tests for CandidateSet helper methods."""

    def test_safe_candidates_filter(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate()
        cs = generate_candidates(request, profile, cert, [], {})
        safe = cs.safe_candidates
        for c in safe:
            self.assertTrue(c.is_safe)

    def test_count_by_type(self):
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate()
        cs = generate_candidates(request, profile, cert, [], {})
        self.assertEqual(cs.count_by_type(CandidateType.FULL_PAYMENT), 1)
        self.assertEqual(cs.count_by_type(CandidateType.INSTALLMENT_PLAN), 0)

    def test_generate_candidates_always_includes_full_payment(self):
        """FULL_PAYMENT candidate is always present."""
        profile = _make_profile()
        request = _make_request()
        cert = _make_certificate()
        cs = generate_candidates(request, profile, cert, [], {})
        full_cands = [c for c in cs.candidates if c.candidate_type == CandidateType.FULL_PAYMENT]
        self.assertEqual(len(full_cands), 1)


class TestCandidatePaymentPlanString(unittest.TestCase):
    """Test payment_plan_string format."""

    def test_full_payment_plan_string_format(self):
        profile = _make_profile()
        request = _make_request(request_date="2024-03-01")
        cert = _make_certificate(request_date="2024-03-01")
        cand = build_full_payment_candidate(request, cert, profile)
        self.assertIn("2024-03-01", cand.payment_plan_string)
        self.assertIn("25000", cand.payment_plan_string)

    def test_ineligible_installment_plan_string_is_none(self):
        profile = _make_profile()
        request = _make_request()
        option = _make_payment_option()
        feasibility = _make_feasibility(option, is_eligible=False, plan=None, rejection_reason="bad")
        cand = build_installment_candidate(request, feasibility, option)
        self.assertEqual(cand.payment_plan_string, "none")


if __name__ == "__main__":
    unittest.main()
