"""Integrity audit test suite for candidate_generation.py (Prompt 10B §15).

Tests:
A. 790-option -> candidate mapping
B. no invalid candidate is rankable
C. no unsafe candidate is rankable
D. full-payment accounting
E. partial-payment accounting
F. wait semantics
G. duplicate semantics
H. lineage
I. deterministic ordering
J. source-row permutation invariance
K. Decimal exactness
L. candidate immutability
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal
from typing import Optional, Tuple

from code.candidate_generation import (
    Candidate,
    CandidatePayment,
    CandidateProvenance,
    CandidateSet,
    CandidateStatus,
    CandidateType,
    OptionUniverseState,
    build_full_payment_candidate,
    build_installment_candidate,
    build_partial_payment_candidate,
    build_wait_candidate,
    classify_feasibility_status,
    classify_option_universe_state,
    generate_candidates,
    is_rankable,
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
        simulation_end=date(2024, 6, 1),
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
    total_payable_amount: Optional[str] = None,
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
# Test Cases A-L (Prompt 10B §15)
# ---------------------------------------------------------------------------

class TestA_OptionCandidateMapping(unittest.TestCase):
    """A. 790-option -> candidate mapping."""

    def test_feasibility_state_classification(self):
        opt = _make_payment_option()
        plan = _make_payment_plan(opt)

        # B: eligible + safe
        feas_safe = _make_feasibility(opt, is_eligible=True, is_safe=True, plan=plan)
        self.assertEqual(classify_option_universe_state(feas_safe), OptionUniverseState.ELIGIBLE_SAFE)
        self.assertEqual(classify_feasibility_status(feas_safe), CandidateStatus.ELIGIBLE_AND_SAFE)

        # C: eligible + unsafe
        feas_unsafe = _make_feasibility(opt, is_eligible=True, is_safe=False, plan=plan, rejection_reason="Violates safety floor")
        self.assertEqual(classify_option_universe_state(feas_unsafe), OptionUniverseState.ELIGIBLE_UNSAFE)
        self.assertEqual(classify_feasibility_status(feas_unsafe), CandidateStatus.STRUCTURALLY_VALID_UNSAFE)

        # A: contractually ineligible (e.g. user preferences)
        feas_contr = _make_feasibility(opt, is_eligible=False, plan=None, rejection_reason="Option method not permitted by user preferences")
        self.assertEqual(classify_option_universe_state(feas_contr), OptionUniverseState.CONTRACTUALLY_INELIGIBLE)
        self.assertEqual(classify_feasibility_status(feas_contr), CandidateStatus.CONTRACTUALLY_INELIGIBLE)

        # D: structurally invalid (e.g. invalid amount)
        feas_struct = _make_feasibility(opt, is_eligible=False, plan=None, rejection_reason="invalid payment amount")
        self.assertEqual(classify_option_universe_state(feas_struct), OptionUniverseState.STRUCTURALLY_INVALID)
        self.assertEqual(classify_feasibility_status(feas_struct), CandidateStatus.STRUCTURALLY_INVALID)

    def test_csv_full_payment_options_excluded_from_installment_candidates(self):
        """CSV options with method 'full_payment' do not create INSTALLMENT_PLAN candidates."""
        req = _make_request()
        prof = _make_profile()
        cert = _make_certificate()
        opt_full = _make_payment_option(payment_option_id="opt_fp", payment_method="full_payment")
        feas_full = _make_feasibility(opt_full, is_eligible=True, is_safe=True)

        cs = generate_candidates(
            request=req,
            profile=prof,
            certificate=cert,
            payment_option_feasibilities=[feas_full],
            payment_options_by_id={"opt_fp": opt_full},
        )
        installment_cands = [c for c in cs.candidates if c.candidate_type == CandidateType.INSTALLMENT_PLAN]
        self.assertEqual(len(installment_cands), 0)


class TestB_NoInvalidCandidateIsRankable(unittest.TestCase):
    """B. No invalid candidate is rankable."""

    def test_structurally_invalid_candidate_not_rankable(self):
        req = _make_request()
        opt = _make_payment_option()
        feas = _make_feasibility(opt, is_eligible=False, plan=None, rejection_reason="invalid payment amount")
        cand = build_installment_candidate(req, feas, opt)
        self.assertEqual(cand.status, CandidateStatus.STRUCTURALLY_INVALID)
        self.assertFalse(cand.is_safe)
        self.assertFalse(cand.is_rankable)
        self.assertFalse(is_rankable(cand))

    def test_contractually_ineligible_candidate_not_rankable(self):
        req = _make_request()
        opt = _make_payment_option()
        feas = _make_feasibility(opt, is_eligible=False, plan=None, rejection_reason="not permitted by user preferences")
        cand = build_installment_candidate(req, feas, opt)
        self.assertEqual(cand.status, CandidateStatus.CONTRACTUALLY_INELIGIBLE)
        self.assertFalse(cand.is_safe)
        self.assertFalse(cand.is_rankable)
        self.assertFalse(is_rankable(cand))


class TestC_NoUnsafeCandidateIsRankable(unittest.TestCase):
    """C. No unsafe candidate is rankable."""

    def test_structurally_valid_unsafe_candidate_not_rankable(self):
        req = _make_request()
        opt = _make_payment_option()
        plan = _make_payment_plan(opt)
        feas = _make_feasibility(opt, is_eligible=True, is_safe=False, plan=plan, rejection_reason="unsafe")
        cand = build_installment_candidate(req, feas, opt)
        self.assertEqual(cand.status, CandidateStatus.STRUCTURALLY_VALID_UNSAFE)
        self.assertFalse(cand.is_safe)
        self.assertFalse(cand.is_rankable)
        self.assertFalse(is_rankable(cand))

    def test_unsafe_full_payment_not_rankable(self):
        req = _make_request()
        prof = _make_profile()
        cert = _make_certificate(is_full_payment_safe_today=False)
        cand = build_full_payment_candidate(req, cert, prof)
        self.assertEqual(cand.status, CandidateStatus.STRUCTURALLY_VALID_UNSAFE)
        self.assertFalse(cand.is_safe)
        self.assertFalse(cand.is_rankable)
        self.assertFalse(is_rankable(cand))


class TestD_FullPaymentAccounting(unittest.TestCase):
    """D. Full-payment accounting."""

    def test_full_payment_three_states(self):
        req = _make_request()

        # State 1: Eligible and Safe
        prof_ok = _make_profile(payment_methods=("full_payment",))
        cert_safe = _make_certificate(is_full_payment_safe_today=True)
        c1 = build_full_payment_candidate(req, cert_safe, prof_ok)
        self.assertEqual(c1.status, CandidateStatus.ELIGIBLE_AND_SAFE)
        self.assertTrue(c1.is_safe)
        self.assertTrue(c1.is_rankable)

        # State 2: Eligible and Unsafe
        cert_unsafe = _make_certificate(is_full_payment_safe_today=False)
        c2 = build_full_payment_candidate(req, cert_unsafe, prof_ok)
        self.assertEqual(c2.status, CandidateStatus.STRUCTURALLY_VALID_UNSAFE)
        self.assertFalse(c2.is_safe)
        self.assertFalse(c2.is_rankable)

        # State 3: Contractually Ineligible
        prof_no_fp = _make_profile(payment_methods=("installments",))
        c3 = build_full_payment_candidate(req, cert_safe, prof_no_fp)
        self.assertEqual(c3.status, CandidateStatus.CONTRACTUALLY_INELIGIBLE)
        self.assertFalse(c3.is_safe)
        self.assertFalse(c3.is_rankable)


class TestE_PartialPaymentAccounting(unittest.TestCase):
    """E. Partial-payment accounting."""

    def test_partial_payment_exact_sum_and_deadline(self):
        req = _make_request(
            requested_amount="50000",
            desired_completion_date="2024-05-01",
            allows_partial_payment=True,
        )
        prof = _make_profile(payment_methods=("partial_payment",))
        cert = _make_certificate(
            requested_amount="50000",
            amount_safe_to_pay="20000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-15",
        )
        cand = build_partial_payment_candidate(req, cert, prof)
        self.assertIsNotNone(cand)
        self.assertEqual(len(cand.payment_schedule), 2)
        p1, p2 = cand.payment_schedule
        self.assertEqual(p1.amount, Decimal("20000"))
        self.assertEqual(p2.amount, Decimal("30000"))
        self.assertEqual(p1.amount + p2.amount, Decimal("50000"))
        self.assertEqual(p1.payment_date, req.request_date)
        self.assertEqual(p2.payment_date, date(2024, 4, 15))
        self.assertLessEqual(p2.payment_date, req.desired_completion_date)
        self.assertTrue(cand.is_safe)
        self.assertTrue(cand.is_rankable)

    def test_partial_payment_excluded_if_misses_deadline(self):
        req = _make_request(
            desired_completion_date="2024-04-01",
            allows_partial_payment=True,
        )
        prof = _make_profile(payment_methods=("partial_payment",))
        cert = _make_certificate(
            amount_safe_to_pay="10000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-15",  # after desired_completion_date
        )
        cand = build_partial_payment_candidate(req, cert, prof)
        self.assertIsNone(cand)


class TestF_WaitSemantics(unittest.TestCase):
    """F. Wait semantics."""

    def test_wait_candidate_semantics(self):
        req = _make_request(requested_amount="25000", request_date="2024-03-01")
        prof = _make_profile(payment_methods=("full_payment",))
        cert = _make_certificate(
            requested_amount="25000",
            is_full_payment_safe_today=False,
            earliest_date="2024-04-15",
        )
        cand = build_wait_candidate(req, cert, prof)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.candidate_type, CandidateType.WAIT)
        self.assertEqual(cand.total_amount_paid, Decimal("25000"))
        self.assertEqual(cand.number_of_payments, 1)
        self.assertEqual(cand.first_payment_date, date(2024, 4, 15))
        self.assertEqual(cand.completion_date, date(2024, 4, 15))
        self.assertEqual(cand.financing_fee, Decimal("0"))
        self.assertEqual(len(cand.payment_schedule), 1)
        self.assertEqual(cand.payment_schedule[0].amount, Decimal("25000"))
        self.assertEqual(cand.payment_schedule[0].payment_date, date(2024, 4, 15))
        self.assertTrue(cand.is_safe)
        self.assertTrue(cand.is_rankable)

    def test_wait_not_generated_if_safe_today(self):
        req = _make_request(request_date="2024-03-01")
        prof = _make_profile(payment_methods=("full_payment",))
        cert = _make_certificate(
            is_full_payment_safe_today=True,
            earliest_date="2024-03-01",  # on request_date -> full_payment, not wait
        )
        cand = build_wait_candidate(req, cert, prof)
        self.assertIsNone(cand)


class TestG_DuplicateSemantics(unittest.TestCase):
    """G. Duplicate semantics."""

    def test_distinct_payment_options_retained_independently(self):
        """Even if two payment options have identical schedules, both are retained."""
        req = _make_request()
        prof = _make_profile()
        cert = _make_certificate()
        opt1 = _make_payment_option(payment_option_id="opt_1")
        opt2 = _make_payment_option(payment_option_id="opt_2")
        plan1 = _make_payment_plan(opt1)
        plan2 = _make_payment_plan(opt2)
        feas1 = _make_feasibility(opt1, plan=plan1)
        feas2 = _make_feasibility(opt2, plan=plan2)

        cs = generate_candidates(
            request=req,
            profile=prof,
            certificate=cert,
            payment_option_feasibilities=[feas1, feas2],
            payment_options_by_id={"opt_1": opt1, "opt_2": opt2},
        )
        inst_cands = [c for c in cs.candidates if c.candidate_type == CandidateType.INSTALLMENT_PLAN]
        self.assertEqual(len(inst_cands), 2)
        ids = {c.source_payment_option_id for c in inst_cands}
        self.assertEqual(ids, {"opt_1", "opt_2"})

    def test_no_partial_payment_if_safe_amount_equals_requested(self):
        req = _make_request(requested_amount="25000", allows_partial_payment=True)
        prof = _make_profile(payment_methods=("partial_payment",))
        cert = _make_certificate(
            requested_amount="25000",
            amount_safe_to_pay="25000",  # safe_amt == req_amt -> excluded
            is_full_payment_safe_today=True,
        )
        cand = build_partial_payment_candidate(req, cert, prof)
        self.assertIsNone(cand)


class TestH_Lineage(unittest.TestCase):
    """H. Lineage completeness."""

    def test_provenance_populated_for_all_types(self):
        req = _make_request()
        prof = _make_profile()
        cert = _make_certificate(amount_safe_to_pay="10000", is_full_payment_safe_today=False, earliest_date="2024-04-15")
        opt = _make_payment_option()
        plan = _make_payment_plan(opt)
        feas = _make_feasibility(opt, plan=plan)

        cs = generate_candidates(
            request=req,
            profile=prof,
            certificate=cert,
            payment_option_feasibilities=[feas],
            payment_options_by_id={opt.payment_option_id: opt},
        )
        for c in cs.candidates:
            self.assertIsNotNone(c.provenance)
            self.assertTrue(bool(c.provenance.source))
            if c.candidate_type == CandidateType.FULL_PAYMENT:
                self.assertEqual(c.provenance.source, "safe_to_pay_certificate")
                self.assertEqual(c.provenance.safe_to_pay_certificate_id, cert.request_id)
            elif c.candidate_type == CandidateType.INSTALLMENT_PLAN:
                self.assertEqual(c.provenance.source, "payment_option_feasibility")
                self.assertEqual(c.provenance.source_payment_option_id, opt.payment_option_id)
            elif c.candidate_type == CandidateType.PARTIAL_PAYMENT:
                self.assertEqual(c.provenance.source, "safe_to_pay_certificate")
                self.assertEqual(c.provenance.safe_to_pay_certificate_id, cert.request_id)
            elif c.candidate_type == CandidateType.WAIT:
                self.assertEqual(c.provenance.source, "earliest_full_payment_computation")
                self.assertIsNotNone(c.provenance.earliest_full_payment_date)


class TestI_DeterministicOrdering(unittest.TestCase):
    """I. Deterministic ordering."""

    def test_candidates_sorted_by_ordering_key(self):
        req = _make_request()
        prof = _make_profile()
        cert = _make_certificate(amount_safe_to_pay="10000", is_full_payment_safe_today=False, earliest_date="2024-04-15")
        opt = _make_payment_option()
        plan = _make_payment_plan(opt)
        feas = _make_feasibility(opt, plan=plan)

        cs = generate_candidates(
            request=req,
            profile=prof,
            certificate=cert,
            payment_option_feasibilities=[feas],
            payment_options_by_id={opt.payment_option_id: opt},
        )
        keys = [c.ordering_key for c in cs.candidates]
        self.assertEqual(keys, sorted(keys))


class TestJ_SourceRowPermutationInvariance(unittest.TestCase):
    """J. Source-row permutation invariance."""

    def test_permuting_options_does_not_change_candidate_set(self):
        req = _make_request()
        prof = _make_profile()
        cert = _make_certificate()
        opt1 = _make_payment_option(payment_option_id="opt_1", payment_amount="2500")
        opt2 = _make_payment_option(payment_option_id="opt_2", payment_amount="5000", number_of_payments=5)
        plan1 = _make_payment_plan(opt1)
        plan2 = _make_payment_plan(opt2)
        feas1 = _make_feasibility(opt1, plan=plan1)
        feas2 = _make_feasibility(opt2, plan=plan2)
        opts_map = {"opt_1": opt1, "opt_2": opt2}

        cs_order_1 = generate_candidates(req, prof, cert, [feas1, feas2], opts_map)
        cs_order_2 = generate_candidates(req, prof, cert, [feas2, feas1], opts_map)

        self.assertEqual(
            [c.candidate_id for c in cs_order_1.candidates],
            [c.candidate_id for c in cs_order_2.candidates],
        )


class TestK_DecimalExactness(unittest.TestCase):
    """K. Decimal exactness."""

    def test_all_money_fields_decimal(self):
        req = _make_request()
        prof = _make_profile()
        cert = _make_certificate(amount_safe_to_pay="10000", is_full_payment_safe_today=False, earliest_date="2024-04-15")
        opt = _make_payment_option()
        plan = _make_payment_plan(opt)
        feas = _make_feasibility(opt, plan=plan)

        cs = generate_candidates(
            request=req,
            profile=prof,
            certificate=cert,
            payment_option_feasibilities=[feas],
            payment_options_by_id={opt.payment_option_id: opt},
        )
        for c in cs.candidates:
            self.assertIsInstance(c.total_amount_paid, Decimal)
            self.assertIsInstance(c.financing_fee, Decimal)
            for p in c.payment_schedule:
                self.assertIsInstance(p.amount, Decimal)


class TestL_CandidateImmutability(unittest.TestCase):
    """L. Candidate immutability."""

    def test_candidate_is_frozen(self):
        req = _make_request()
        prof = _make_profile()
        cert = _make_certificate()
        cand = build_full_payment_candidate(req, cert, prof)

        with self.assertRaises(FrozenInstanceError):
            cand.total_amount_paid = Decimal("0")  # type: ignore

    def test_payment_and_provenance_frozen(self):
        p = CandidatePayment(payment_date=date(2024, 3, 1), amount=Decimal("100"), sequence_number=1)
        with self.assertRaises(FrozenInstanceError):
            p.amount = Decimal("200")  # type: ignore

        prov = CandidateProvenance(source="test", source_payment_option_id=None, safe_to_pay_certificate_id=None, earliest_full_payment_date=None)
        with self.assertRaises(FrozenInstanceError):
            prov.source = "other"  # type: ignore


if __name__ == "__main__":
    unittest.main()
