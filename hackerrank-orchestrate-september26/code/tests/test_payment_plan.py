"""Unit and metamorphic property test suite for payment_plan.py.

Validates:
- Test Matrix Scenarios A through Y.
- Metamorphic Properties 1 through 8.
- Cross-request option safety and immutability guarantees.
- Full compatibility with frozen simulator.py and canonical models.
"""

from datetime import date, timedelta
from decimal import Decimal
import unittest

from code.canonical import CanonicalEvent, CashImpactType, Direction, RecurrenceClassification
from code.models import FinancialProfile, FinancialRequest, PaymentOption
from code.payment_plan import (
    PaymentPlan,
    PaymentPlanFeasibility,
    PaymentScheduleEntry,
    add_calendar_months,
    construct_payment_schedule,
    evaluate_payment_option_feasibility,
    evaluate_payment_options_for_request,
    validate_option_schema,
)
from code.recurrence import FutureEvent
from code.simulator import simulate_user


def make_profile(
    user_id: str = "user_test",
    home_currency: str = "USD",
    available_balance: Decimal = Decimal("5000.00"),
    minimum_balance: Decimal = Decimal("1000.00"),
    payment_methods: tuple = ("full_payment", "installments", "partial_payment"),
    max_installment_months: int = 6,
) -> FinancialProfile:
    return FinancialProfile(
        user_id=user_id,
        home_currency=home_currency,
        current_available_balance=available_balance,
        minimum_balance_to_keep=minimum_balance,
        financial_priorities=("emergency_savings",),
        expense_categories_to_protect=("rent", "groceries"),
        expense_categories_user_is_willing_to_reduce=("dining",),
        expense_categories_user_is_willing_to_stop=("streaming",),
        payment_methods_user_will_consider=payment_methods,
        max_installment_months=max_installment_months,
    )


def make_request(
    request_id: str = "request_test",
    user_id: str = "user_test",
    request_date: date = date(2026, 1, 1),
    requested_amount: Decimal = Decimal("1200.00"),
    desired_completion_date: date = date(2026, 3, 31),
) -> FinancialRequest:
    return FinancialRequest(
        request_id=request_id,
        user_id=user_id,
        request_date=request_date,
        request_type="purchase",
        requested_amount=requested_amount,
        desired_completion_date=desired_completion_date,
        allows_partial_payment=True,
        request_text="Test purchase",
    )


class TestPaymentPlanMatrix(unittest.TestCase):
    """Test Matrix Scenarios A through Y."""

    def setUp(self) -> None:
        self.profile = make_profile()
        self.request = make_request()

    def test_scenario_a_full_payment_safe_today(self) -> None:
        """A. Full payment safe today."""
        option = PaymentOption(
            payment_option_id="opt_full_safe",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(
            option=option,
            request=self.request,
            profile=self.profile,
            canonical_events=(),
            future_events=(),
        )
        self.assertTrue(res.is_eligible)
        self.assertTrue(res.is_safe)
        self.assertTrue(res.has_valid_plan)
        self.assertEqual(res.plan_type, "full_payment")
        self.assertIsNone(res.rejection_reason)
        self.assertEqual(res.minimum_available_cash, Decimal("3800.00"))

    def test_scenario_b_full_payment_unsafe_today(self) -> None:
        """B. Full payment unsafe today (violates safety floor)."""
        # Balance 5000, safety floor 1000, purchase 4500 -> remaining 500 < 1000
        req = make_request(requested_amount=Decimal("4500.00"))
        option = PaymentOption(
            payment_option_id="opt_full_unsafe",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("4500.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("4500.00"),
        )
        res = evaluate_payment_option_feasibility(
            option=option,
            request=req,
            profile=self.profile,
            canonical_events=(),
            future_events=(),
        )
        self.assertTrue(res.is_eligible)
        self.assertFalse(res.is_safe)
        self.assertFalse(res.has_valid_plan)
        self.assertIn("safety floor breach", res.rejection_reason)

    def test_scenario_c_installment_option_safe(self) -> None:
        """C. Installment option safe (e.g. 3 payments of 400)."""
        option = PaymentOption(
            payment_option_id="opt_inst_safe",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(
            option=option,
            request=self.request,
            profile=self.profile,
            canonical_events=(),
            future_events=(),
        )
        self.assertTrue(res.is_eligible)
        self.assertTrue(res.is_safe)
        self.assertTrue(res.has_valid_plan)
        self.assertEqual(res.plan_type, "installments")
        self.assertEqual(len(res.payment_plan.entries), 3)

    def test_scenario_d_installment_option_unsafe(self) -> None:
        """D. Installment option unsafe due to future obligation collision."""
        # Rent obligation of 3800 on day 30
        canon_rent = CanonicalEvent(
            event_id="rent_01",
            user_id="user_test",
            source_row=1,
            effective_date=date(2026, 1, 31),
            direction=Direction.OUTFLOW,
            direction_original="outflow",
            amount_original=Decimal("3800.00"),
            currency_original="USD",
            amount_home=Decimal("3800.00"),
            home_currency="USD",
            exchange_rate_used=None,
            exchange_rate_date=None,
            status="scheduled",
            is_cash_event=True,
            cash_impact_type=CashImpactType.SCHEDULED_OUTFLOW,
            event_type="rent",
            category="rent",
            description="Monthly Rent",
            flexibility="fixed",
            minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None,
            linked_event_id=None,
            recurrence_type=RecurrenceClassification.ONE_TIME,
            is_unresolved=False,
            unresolved_reason=None,
            evidence_chain=(),
            applied_actions=(),
        )
        option = PaymentOption(
            payment_option_id="opt_inst_unsafe",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        # Starting 5000 - 400 (day 1) - 400 (day 31) - 3800 (day 31 rent) = 400 < safety floor 1000!
        res = evaluate_payment_option_feasibility(
            option=option,
            request=self.request,
            profile=self.profile,
            canonical_events=(canon_rent,),
            future_events=(),
        )
        self.assertTrue(res.is_eligible)
        self.assertFalse(res.is_safe)
        self.assertIn("safety floor breach", res.rejection_reason)

    def test_scenario_e_user_permits_installments(self) -> None:
        """E. User permits installments."""
        prof = make_profile(payment_methods=("installments",))
        opt = PaymentOption(
            payment_option_id="opt_1",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, prof, (), ())
        self.assertTrue(res.is_eligible)

    def test_scenario_f_user_rejects_installments(self) -> None:
        """F. User rejects installments in payment preference."""
        prof = make_profile(payment_methods=("full_payment",))
        opt = PaymentOption(
            payment_option_id="opt_1",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, prof, (), ())
        self.assertFalse(res.is_eligible)
        self.assertFalse(res.is_safe)
        self.assertIn("not permitted by user preferences", res.rejection_reason)

    def test_scenario_g_max_installment_months_exact_boundary(self) -> None:
        """G. max_installment_months exact boundary: duration within limit."""
        # first_payment_date = 2026-01-01, max_months = 2.
        # limit date = 2026-03-01.
        # 3 payments with freq 28: 2026-01-01, 2026-01-29, 2026-02-26.
        # 2026-02-26 <= 2026-03-01 -> ELIGIBLE.
        prof = make_profile(max_installment_months=2)
        opt = PaymentOption(
            payment_option_id="opt_bound",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=28,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, prof, (), ())
        self.assertTrue(res.is_eligible)

    def test_scenario_h_max_installment_months_exceeded(self) -> None:
        """H. max_installment_months exceeded: duration beyond limit."""
        # first_payment_date = 2026-01-01, max_months = 2.
        # limit date = 2026-03-01.
        # 3 payments with freq 31: 2026-01-01, 2026-02-01, 2026-03-04.
        # 2026-03-04 > 2026-03-01 -> INELIGIBLE.
        prof = make_profile(max_installment_months=2)
        opt = PaymentOption(
            payment_option_id="opt_exceeded",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=31,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, prof, (), ())
        self.assertFalse(res.is_eligible)
        self.assertIn("exceeds maximum installment duration", res.rejection_reason)

    def test_scenario_i_request_id_mismatch(self) -> None:
        """I. Request ID mismatch deterministically rejected."""
        opt = PaymentOption(
            payment_option_id="opt_mismatch",
            request_id="request_other",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, self.profile, (), ())
        self.assertFalse(res.is_eligible)
        self.assertIn("option belongs to different request", res.rejection_reason)

    def test_scenario_j_invalid_payment_count(self) -> None:
        """J. Invalid payment count (e.g. 0 payments, or installments with 1 payment)."""
        opt_zero = PaymentOption(
            payment_option_id="opt_zero",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=0,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res_zero = evaluate_payment_option_feasibility(opt_zero, self.request, self.profile, (), ())
        self.assertFalse(res_zero.is_eligible)
        self.assertIn("invalid number of payments", res_zero.rejection_reason)

        opt_inst_one = PaymentOption(
            payment_option_id="opt_inst_one",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res_one = evaluate_payment_option_feasibility(opt_inst_one, self.request, self.profile, (), ())
        self.assertFalse(res_one.is_eligible)
        self.assertIn("installments require > 1 payments", res_one.rejection_reason)

    def test_scenario_k_invalid_frequency(self) -> None:
        """K. Invalid payment frequency for installments."""
        opt_no_freq = PaymentOption(
            payment_option_id="opt_no_freq",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("600.00"),
            number_of_payments=2,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt_no_freq, self.request, self.profile, (), ())
        self.assertFalse(res.is_eligible)
        self.assertIn("installments require positive payment_frequency_days", res.rejection_reason)

    def test_scenario_l_financing_fee_reconciliation(self) -> None:
        """L. Financing fee reconciliation (negative fee rejected)."""
        opt_neg_fee = PaymentOption(
            payment_option_id="opt_neg_fee",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("600.00"),
            number_of_payments=2,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("-10.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt_neg_fee, self.request, self.profile, (), ())
        self.assertFalse(res.is_eligible)
        self.assertIn("invalid financing fee", res.rejection_reason)

    def test_scenario_m_total_payable_reconciliation(self) -> None:
        """M. Total payable reconciliation failure (amount * n != total)."""
        opt_mismatch = PaymentOption(
            payment_option_id="opt_mismatch",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("50.00"),
            total_payable_amount=Decimal("1500.00"),  # 400 * 3 = 1200 != 1500!
        )
        res = evaluate_payment_option_feasibility(opt_mismatch, self.request, self.profile, (), ())
        self.assertFalse(res.is_eligible)
        self.assertIn("amount reconciliation failure", res.rejection_reason)

    def test_scenario_n_first_payment_on_request_date(self) -> None:
        """N. First payment exactly on request date."""
        opt = PaymentOption(
            payment_option_id="opt_on_date",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, self.profile, (), ())
        self.assertTrue(res.is_eligible)
        self.assertEqual(res.payment_plan.first_payment_date, self.request.request_date)

    def test_scenario_o_first_payment_after_request_date(self) -> None:
        """O. First payment after request date."""
        opt = PaymentOption(
            payment_option_id="opt_after_date",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 8),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, self.profile, (), ())
        self.assertTrue(res.is_eligible)
        self.assertEqual(res.payment_plan.first_payment_date, date(2026, 1, 8))

    def test_scenario_p_same_day_salary_and_installment(self) -> None:
        """P. Same-day salary + installment: salary (P0) executes before installment (P3)."""
        # Starting balance 1000, floor 1000 (valid initial state).
        # Salary 2000 on day 1 (P0).
        # Installment 1200 on day 1 (P3).
        # If salary executes first: 1000 + 2000 = 3000 -> 3000 - 1200 = 1800 >= 1000 (SAFE).
        # If installment executed first: 1000 - 1200 = -200 (OVERDRAFT / BREACH).
        prof = make_profile(available_balance=Decimal("1000.00"), minimum_balance=Decimal("1000.00"))
        salary = CanonicalEvent(
            event_id="salary_01",
            user_id="user_test",
            source_row=1,
            effective_date=date(2026, 1, 1),
            direction=Direction.INFLOW,
            direction_original="inflow",
            amount_original=Decimal("2000.00"),
            currency_original="USD",
            amount_home=Decimal("2000.00"),
            home_currency="USD",
            exchange_rate_used=None,
            exchange_rate_date=None,
            status="scheduled",
            is_cash_event=True,
            cash_impact_type=CashImpactType.SCHEDULED_INFLOW,
            event_type="salary",
            category="salary",
            description="Monthly Paycheck",
            flexibility="fixed",
            minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None,
            linked_event_id=None,
            recurrence_type=RecurrenceClassification.EXPLICITLY_RECURRING,
            is_unresolved=False,
            unresolved_reason=None,
            evidence_chain=(),
            applied_actions=(),
        )
        opt = PaymentOption(
            payment_option_id="opt_p",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, prof, (salary,), ())
        self.assertTrue(res.is_eligible)
        self.assertTrue(res.is_safe)
        self.assertEqual(res.minimum_available_cash, Decimal("1000.00"))

    def test_scenario_q_same_day_obligation_and_installment(self) -> None:
        """Q. Same-day obligation + installment: obligation (P2) precedes candidate (P3)."""
        # Starting balance 2500, floor 1000.
        # Obligation 1000 on day 1 (P2).
        # Candidate 1000 on day 1 (P3).
        # Both execute on day 1: 2500 - 1000 - 1000 = 500 < 1000 floor -> UNSAFE.
        ob = CanonicalEvent(
            event_id="bill_01",
            user_id="user_test",
            source_row=1,
            effective_date=date(2026, 1, 1),
            direction=Direction.OUTFLOW,
            direction_original="outflow",
            amount_original=Decimal("1000.00"),
            currency_original="USD",
            amount_home=Decimal("1000.00"),
            home_currency="USD",
            exchange_rate_used=None,
            exchange_rate_date=None,
            status="scheduled",
            is_cash_event=True,
            cash_impact_type=CashImpactType.SCHEDULED_OUTFLOW,
            event_type="utility",
            category="utilities",
            description="Power bill",
            flexibility="fixed",
            minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None,
            linked_event_id=None,
            recurrence_type=RecurrenceClassification.ONE_TIME,
            is_unresolved=False,
            unresolved_reason=None,
            evidence_chain=(),
            applied_actions=(),
        )
        opt = PaymentOption(
            payment_option_id="opt_q",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1000.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1000.00"),
        )
        prof = make_profile(available_balance=Decimal("2500.00"), minimum_balance=Decimal("1000.00"))
        res = evaluate_payment_option_feasibility(opt, self.request, prof, (ob,), ())
        self.assertFalse(res.is_safe)
        self.assertEqual(res.minimum_available_cash, Decimal("500.00"))

    def test_scenario_r_pending_reservation_and_installment(self) -> None:
        """R. Pending reservation + installment: hold reduces available cash."""
        # Available balance already accounts for pending holds in profile.current_available_balance
        prof = make_profile(available_balance=Decimal("2000.00"), minimum_balance=Decimal("1000.00"))
        opt = PaymentOption(
            payment_option_id="opt_r",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, prof, (), ())
        self.assertFalse(res.is_safe)
        self.assertEqual(res.minimum_available_cash, Decimal("800.00"))

    def test_scenario_s_multiple_installments(self) -> None:
        """S. Multiple installments (e.g. 6 payments)."""
        opt = PaymentOption(
            payment_option_id="opt_s",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("200.00"),
            number_of_payments=6,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=15,  # ends on day 75 <= 90
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, self.request, self.profile, (), ())
        self.assertTrue(res.is_eligible)
        self.assertTrue(res.is_safe)
        self.assertEqual(len(res.payment_plan.entries), 6)

    def test_scenario_t_zero_or_negative_amount_rejection(self) -> None:
        """T. Zero or negative amount rejection."""
        opt_neg = PaymentOption(
            payment_option_id="opt_neg",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("-100.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("-100.00"),
        )
        res = evaluate_payment_option_feasibility(opt_neg, self.request, self.profile, (), ())
        self.assertFalse(res.is_eligible)
        self.assertIn("invalid payment amount", res.rejection_reason)

    def test_scenario_u_option_ending_outside_horizon(self) -> None:
        """U. Option ending outside 90-day simulation horizon is rejected."""
        prof = make_profile(max_installment_months=None)
        opt_long = PaymentOption(
            payment_option_id="opt_long",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("100.00"),
            number_of_payments=15,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,  # 14 * 30 = 420 days >> 90 days
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1500.00"),
        )
        res = evaluate_payment_option_feasibility(opt_long, self.request, prof, (), ())
        self.assertFalse(res.is_eligible)
        self.assertIn("payment schedule outside allowed horizon", res.rejection_reason)

    def test_scenario_v_baseline_immutability(self) -> None:
        """V. Baseline immutability: baseline simulation is identical before and after."""
        base_before = simulate_user(
            user_id="user_test",
            simulation_start=date(2026, 1, 1),
            simulation_end=date(2026, 4, 1),
            profile=self.profile,
        )
        opt = PaymentOption(
            payment_option_id="opt_v",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        _ = evaluate_payment_option_feasibility(opt, self.request, self.profile, (), ())

        base_after = simulate_user(
            user_id="user_test",
            simulation_start=date(2026, 1, 1),
            simulation_end=date(2026, 4, 1),
            profile=self.profile,
        )
        self.assertEqual(base_before.minimum_projected_available_cash, base_after.minimum_projected_available_cash)
        self.assertEqual(base_before.ending_state.available_cash, base_after.ending_state.available_cash)
        self.assertEqual(len(base_before.projected_events), len(base_after.projected_events))

    def test_scenario_w_deterministic_repeated_evaluation(self) -> None:
        """W. Deterministic repeated evaluation yields identical results."""
        opt = PaymentOption(
            payment_option_id="opt_w",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res1 = evaluate_payment_option_feasibility(opt, self.request, self.profile, (), ())
        res2 = evaluate_payment_option_feasibility(opt, self.request, self.profile, (), ())
        self.assertEqual(res1, res2)

    def test_scenario_x_arbitrary_option_evaluation_order(self) -> None:
        """X. Arbitrary option evaluation order produces identical individual results."""
        opt1 = PaymentOption(
            payment_option_id="opt_1",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        opt2 = PaymentOption(
            payment_option_id="opt_2",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        forward = evaluate_payment_options_for_request(self.request, (opt1, opt2), self.profile, (), ())
        reverse = evaluate_payment_options_for_request(self.request, (opt2, opt1), self.profile, (), ())

        self.assertEqual(forward[0], reverse[1])
        self.assertEqual(forward[1], reverse[0])

    def test_scenario_y_all_options_evaluated_independently(self) -> None:
        """Y. All options are evaluated independently (no cross-contamination)."""
        opt_bad = PaymentOption(
            payment_option_id="opt_bad",
            request_id="request_other",  # Ineligible
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        opt_good = PaymentOption(
            payment_option_id="opt_good",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        batch = evaluate_payment_options_for_request(self.request, (opt_bad, opt_good), self.profile, (), ())
        self.assertFalse(batch[0].is_eligible)
        self.assertTrue(batch[1].is_eligible)
        self.assertTrue(batch[1].is_safe)


class TestPaymentPlanProperties(unittest.TestCase):
    """Metamorphic Properties 1 through 8."""

    def setUp(self) -> None:
        self.profile = make_profile()
        self.request = make_request()
        self.opt = PaymentOption(
            payment_option_id="opt_prop",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )

    def test_property_1_idempotence(self) -> None:
        """PROPERTY 1: Evaluating an option twice produces identical feasibility and schedule."""
        res1 = evaluate_payment_option_feasibility(self.opt, self.request, self.profile, (), ())
        res2 = evaluate_payment_option_feasibility(self.opt, self.request, self.profile, (), ())
        self.assertEqual(res1, res2)
        self.assertEqual(res1.payment_plan.entries, res2.payment_plan.entries)

    def test_property_2_order_independence(self) -> None:
        """PROPERTY 2: Evaluating options in different input orders produces identical per-option results."""
        opt_full = PaymentOption(
            payment_option_id="opt_full",
            request_id="request_test",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res_a = evaluate_payment_options_for_request(self.request, (self.opt, opt_full), self.profile, (), ())
        res_b = evaluate_payment_options_for_request(self.request, (opt_full, self.opt), self.profile, (), ())
        self.assertEqual(res_a[0], res_b[1])
        self.assertEqual(res_a[1], res_b[0])

    def test_property_3_ineligible_isolation(self) -> None:
        """PROPERTY 3: Adding an ineligible option cannot change any other option's result."""
        opt_ineligible = PaymentOption(
            payment_option_id="opt_ineligible",
            request_id="request_other",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        single = evaluate_payment_option_feasibility(self.opt, self.request, self.profile, (), ())
        batch = evaluate_payment_options_for_request(self.request, (self.opt, opt_ineligible), self.profile, (), ())
        self.assertEqual(single, batch[0])

    def test_property_4_preference_preserves_baseline(self) -> None:
        """PROPERTY 4: Changing only user payment preferences cannot change simulator baseline."""
        prof_full = make_profile(payment_methods=("full_payment",))
        prof_none = make_profile(payment_methods=())

        base1 = simulate_user(user_id="user_test", simulation_start=date(2026, 1, 1), simulation_end=date(2026, 4, 1), profile=prof_full)
        base2 = simulate_user(user_id="user_test", simulation_start=date(2026, 1, 1), simulation_end=date(2026, 4, 1), profile=prof_none)

        self.assertEqual(base1.minimum_projected_available_cash, base2.minimum_projected_available_cash)
        self.assertEqual(base1.ending_state.available_cash, base2.ending_state.available_cash)

    def test_property_5_monotonic_safety_floor(self) -> None:
        """PROPERTY 5: Increasing the safety floor cannot make an option safer."""
        prof_low_floor = make_profile(minimum_balance=Decimal("500.00"))
        prof_high_floor = make_profile(minimum_balance=Decimal("4000.00"))

        res_low = evaluate_payment_option_feasibility(self.opt, self.request, prof_low_floor, (), ())
        res_high = evaluate_payment_option_feasibility(self.opt, self.request, prof_high_floor, (), ())

        # If high floor is safe, low floor MUST be safe
        if res_high.is_safe:
            self.assertTrue(res_low.is_safe)
        # In our case, high floor (4000) causes breach (5000 - 1200 = 3800 < 4000)
        self.assertTrue(res_low.is_safe)
        self.assertFalse(res_high.is_safe)

    def test_property_6_monotonic_future_income(self) -> None:
        """PROPERTY 6: Removing confirmed future income cannot make an option safer."""
        income = CanonicalEvent(
            event_id="bonus_01",
            user_id="user_test",
            source_row=1,
            effective_date=date(2026, 1, 15),
            direction=Direction.INFLOW,
            direction_original="inflow",
            amount_original=Decimal("2000.00"),
            currency_original="USD",
            amount_home=Decimal("2000.00"),
            home_currency="USD",
            exchange_rate_used=None,
            exchange_rate_date=None,
            status="scheduled",
            is_cash_event=True,
            cash_impact_type=CashImpactType.SCHEDULED_INFLOW,
            event_type="bonus",
            category="bonus",
            description="Bonus",
            flexibility="fixed",
            minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None,
            linked_event_id=None,
            recurrence_type=RecurrenceClassification.ONE_TIME,
            is_unresolved=False,
            unresolved_reason=None,
            evidence_chain=(),
            applied_actions=(),
        )
        res_with_income = evaluate_payment_option_feasibility(self.opt, self.request, self.profile, (income,), ())
        res_without_income = evaluate_payment_option_feasibility(self.opt, self.request, self.profile, (), ())

        # Removing income can never make it safer
        self.assertGreaterEqual(res_with_income.minimum_available_cash, res_without_income.minimum_available_cash)

    def test_property_7_monotonic_payment_amount(self) -> None:
        """PROPERTY 7: Increasing a payment amount while holding all else constant cannot make the option safer."""
        opt_cheap = PaymentOption(
            payment_option_id="opt_cheap",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("100.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("300.00"),
        )
        opt_expensive = PaymentOption(
            payment_option_id="opt_expensive",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("1500.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("4500.00"),
        )
        res_cheap = evaluate_payment_option_feasibility(opt_cheap, self.request, self.profile, (), ())
        res_expensive = evaluate_payment_option_feasibility(opt_expensive, self.request, self.profile, (), ())

        self.assertGreaterEqual(res_cheap.minimum_available_cash, res_expensive.minimum_available_cash)
        if res_expensive.is_safe:
            self.assertTrue(res_cheap.is_safe)

    def test_property_8_duration_violation_never_eligible(self) -> None:
        """PROPERTY 8: A plan that violates max_installment_months can never be marked eligible."""
        prof_strict = make_profile(max_installment_months=2)
        # 3 payments of 31 days spans 62 days > 2 calendar months
        opt_violator = PaymentOption(
            payment_option_id="opt_violator",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=31,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt_violator, self.request, prof_strict, (), ())
        self.assertFalse(res.is_eligible)
        self.assertFalse(res.is_safe)
        self.assertFalse(res.has_valid_plan)


class TestPaymentPlanContractualAudit(unittest.TestCase):
    """Specific contractual-semantics audit tests mandated by Prompt 9B."""

    def setUp(self) -> None:
        self.profile = make_profile()
        self.request = make_request()

    def test_money_reconciliation_exact_and_adversarial_tolerances(self) -> None:
        """Section 2: Money reconciliation requires exact equality; reject 0.01, 0.02, 0.04, 0.05, 0.06."""
        # Baseline: 3 payments of 400 = 1200
        base_opt = PaymentOption(
            payment_option_id="opt_recon",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        self.assertIsNone(validate_option_schema(base_opt, self.request))

        # Adversarial discrepancies
        adversarial_diffs = [
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("0.04"),
            Decimal("0.05"),
            Decimal("0.06"),
            Decimal("-0.01"),
            Decimal("-0.05"),
        ]
        for diff in adversarial_diffs:
            bad_opt = PaymentOption(
                payment_option_id=f"opt_bad_{str(diff).replace('.', '_')}",
                request_id="request_test",
                payment_method="installments",
                payment_amount=Decimal("400.00"),
                number_of_payments=3,
                first_payment_date=date(2026, 1, 1),
                payment_frequency_days=30,
                financing_fee=Decimal("0.00"),
                total_payable_amount=Decimal("1200.00") + diff,
            )
            err = validate_option_schema(bad_opt, self.request)
            self.assertIsNotNone(err, f"Expected rejection for reconciliation diff of {diff}")
            self.assertIn("amount reconciliation failure", err)

    def test_month_end_calendar_dates(self) -> None:
        """Section 4: Exact month-end calendar math with leap year and varying month lengths."""
        # Jan 31 + 1 month in non-leap year (2025) -> Feb 28
        self.assertEqual(add_calendar_months(date(2025, 1, 31), 1), date(2025, 2, 28))
        # Jan 31 + 1 month in leap year (2024) -> Feb 29
        self.assertEqual(add_calendar_months(date(2024, 1, 31), 1), date(2024, 2, 29))
        # Jan 31 + 2 months (2025) -> Mar 31
        self.assertEqual(add_calendar_months(date(2025, 1, 31), 2), date(2025, 3, 31))
        # Jan 31 + 3 months (2025) -> Apr 30
        self.assertEqual(add_calendar_months(date(2025, 1, 31), 3), date(2025, 4, 30))
        # Mar 31 + 1 month -> Apr 30
        self.assertEqual(add_calendar_months(date(2025, 3, 31), 1), date(2025, 4, 30))
        # May 31 + 1 month -> Jun 30
        self.assertEqual(add_calendar_months(date(2025, 5, 31), 1), date(2025, 6, 30))
        # Aug 31 + 1 month -> Sep 30
        self.assertEqual(add_calendar_months(date(2025, 8, 31), 1), date(2025, 9, 30))
        # Oct 31 + 1 month -> Nov 30
        self.assertEqual(add_calendar_months(date(2025, 10, 31), 1), date(2025, 11, 30))

    def test_first_payment_date_offset_boundary(self) -> None:
        """Section 4: Compare request_date + 2m vs first_payment_date + 2m."""
        # request_date = 2026-01-01, first_payment_date = 2026-01-15, max_installment_months = 2.
        # limit from first_payment_date = 2026-03-15.
        # 3 payments of freq 28:
        # P1 = 2026-01-15
        # P2 = 2026-02-12
        # P3 = 2026-03-12
        # P3 (2026-03-12) <= limit (2026-03-15) -> ELIGIBLE.
        prof = make_profile(max_installment_months=2)
        req = make_request(request_date=date(2026, 1, 1))
        opt = PaymentOption(
            payment_option_id="opt_offset_bound",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 15),
            payment_frequency_days=28,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt, req, prof, (), ())
        self.assertTrue(res.is_eligible)

    def test_distinct_simulator_events_for_installments(self) -> None:
        """Section 10: Two installments on different dates produce distinct simulator events."""
        plan = construct_payment_schedule(
            PaymentOption(
                payment_option_id="opt_two",
                request_id="request_test",
                payment_method="installments",
                payment_amount=Decimal("500.00"),
                number_of_payments=2,
                first_payment_date=date(2026, 1, 1),
                payment_frequency_days=30,
                financing_fee=Decimal("0.00"),
                total_payable_amount=Decimal("1000.00"),
            ),
            requested_amount=Decimal("1000.00"),
        )
        self.assertEqual(len(plan.entries), 2)
        self.assertNotEqual(plan.entries[0].date, plan.entries[1].date)
        self.assertEqual(plan.entries[0].amount, Decimal("500.00"))
        self.assertEqual(plan.entries[1].amount, Decimal("500.00"))

    def test_prompt_9b_property_1_schedule_entries_exact_count(self) -> None:
        """PROPERTY 1: Payment schedule entries count exactly equals number_of_payments."""
        for n in (2, 3, 4, 6, 15, 24):
            opt = PaymentOption(
                payment_option_id=f"opt_{n}",
                request_id="request_test",
                payment_method="installments",
                payment_amount=Decimal("100.00"),
                number_of_payments=n,
                first_payment_date=date(2026, 1, 1),
                payment_frequency_days=30,
                financing_fee=Decimal("0.00"),
                total_payable_amount=Decimal(str(100 * n)),
            )
            plan = construct_payment_schedule(opt, Decimal(str(100 * n)))
            self.assertEqual(len(plan.entries), n)
            self.assertEqual(plan.number_of_payments, n)

    def test_prompt_9b_property_3_payment_option_id_invariance(self) -> None:
        """PROPERTY 3: Changing payment_option_id only does not change economic feasibility."""
        opt_a = PaymentOption(
            payment_option_id="opt_alpha",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        opt_b = PaymentOption(
            payment_option_id="opt_beta",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res_a = evaluate_payment_option_feasibility(opt_a, self.request, self.profile, (), ())
        res_b = evaluate_payment_option_feasibility(opt_b, self.request, self.profile, (), ())

        self.assertEqual(res_a.is_eligible, res_b.is_eligible)
        self.assertEqual(res_a.is_safe, res_b.is_safe)
        self.assertEqual(res_a.minimum_available_cash, res_b.minimum_available_cash)
        self.assertEqual(res_a.safety_floor, res_b.safety_floor)

    def test_prompt_9b_property_5_financing_fee_monotonicity(self) -> None:
        """PROPERTY 5: Increasing financing fee cannot make option safer if total outflow increases."""
        opt_no_fee = PaymentOption(
            payment_option_id="opt_no_fee",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("400.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        opt_with_fee = PaymentOption(
            payment_option_id="opt_with_fee",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("500.00"),
            number_of_payments=3,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("300.00"),
            total_payable_amount=Decimal("1500.00"),
        )
        res_no_fee = evaluate_payment_option_feasibility(opt_no_fee, self.request, self.profile, (), ())
        res_with_fee = evaluate_payment_option_feasibility(opt_with_fee, self.request, self.profile, (), ())

        self.assertGreaterEqual(res_no_fee.minimum_available_cash, res_with_fee.minimum_available_cash)
        if res_with_fee.is_safe:
            self.assertTrue(res_no_fee.is_safe)

    def test_prompt_9b_property_6_decreasing_max_months_cannot_make_eligible(self) -> None:
        """PROPERTY 6: Decreasing max_installment_months cannot make an already-rejected option eligible."""
        opt_6m = PaymentOption(
            payment_option_id="opt_6m",
            request_id="request_test",
            payment_method="installments",
            payment_amount=Decimal("200.00"),
            number_of_payments=6,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=30,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        # Already rejected at 4 months
        prof_4m = make_profile(max_installment_months=4)
        res_4m = evaluate_payment_option_feasibility(opt_6m, self.request, prof_4m, (), ())
        self.assertFalse(res_4m.is_eligible)

        # Decreasing further to 2 months cannot make it eligible
        prof_2m = make_profile(max_installment_months=2)
        res_2m = evaluate_payment_option_feasibility(opt_6m, self.request, prof_2m, (), ())
        self.assertFalse(res_2m.is_eligible)

    def test_prompt_9b_property_7_cross_request_ineligible(self) -> None:
        """PROPERTY 7: Changing request_id so an option belongs to another request makes it ineligible."""
        opt_other = PaymentOption(
            payment_option_id="opt_cross",
            request_id="request_unrelated_999",
            payment_method="full_payment",
            payment_amount=Decimal("1200.00"),
            number_of_payments=1,
            first_payment_date=date(2026, 1, 1),
            payment_frequency_days=None,
            financing_fee=Decimal("0.00"),
            total_payable_amount=Decimal("1200.00"),
        )
        res = evaluate_payment_option_feasibility(opt_other, self.request, self.profile, (), ())
        self.assertFalse(res.is_eligible)
        self.assertFalse(res.is_safe)
        self.assertIn("option belongs to different request", res.rejection_reason)


if __name__ == "__main__":
    unittest.main()

