"""Comprehensive test suite for deterministic Affordability Classification Layer.

Covers:
- All 15 required scenarios from Section 13 (A through O)
- All 6 metamorphic property tests from Section 15 (Property 1 through 6)
- User payment method policy constraints (e.g. user refusing full payment)
- PaymentPlanFeasibility protocol compliance
- Provisional status vs Plan Evaluation Required handling
"""

from datetime import date, timedelta
from decimal import Decimal
from typing import Optional
import unittest

from code.canonical import CanonicalEvent, CashImpactType, Direction, RecurrenceClassification
from code.models import FinancialProfile, FinancialRequest
from code.recurrence import FutureEvent, RecurrenceFrequency
from code.safe_to_pay import (
    SafeToPayCertificate,
    calculate_amount_safe_to_pay,
    calculate_earliest_full_payment_date,
    evaluate_request_safe_to_pay,
)
from code.simulator import simulate_user
from code.affordability import (
    AffordabilityCertificateReference,
    AffordabilityResult,
    AffordabilityStatus,
    PaymentPlanFeasibility,
    classify_affordability,
    status_rank,
)


class MockPlanFeasibility:
    """Mock implementing PaymentPlanFeasibility protocol."""
    def __init__(self, has_valid: bool, plan_type: Optional[str] = "installments"):
        self._has_valid = has_valid
        self._plan_type = plan_type

    @property
    def has_valid_plan(self) -> bool:
        return self._has_valid

    @property
    def plan_type(self) -> Optional[str]:
        return self._plan_type


class TestAffordabilityClassification(unittest.TestCase):
    """Scenario tests covering Cases A through O and contract requirements."""

    def setUp(self) -> None:
        self.user_id = "test_afford_user"
        self.request_date = date(2025, 1, 1)
        self.sim_start = self.request_date
        self.sim_end = self.request_date + timedelta(days=90)
        self.home_currency = "USD"

        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency=self.home_currency,
            current_available_balance=Decimal("1000.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=("rent", "groceries"),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("streaming",),
            payment_methods_user_will_consider=("full_payment", "installments", "partial_payment"),
            max_installment_months=6,
        )

        self.request = FinancialRequest(
            request_id="req_001",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("300.00"),
            desired_completion_date=self.request_date + timedelta(days=60),
            allows_partial_payment=True,
            request_text="Can I buy this?",
        )

    def _build_cert(
        self,
        requested_amount: Decimal,
        amount_safe_to_pay: Decimal,
        earliest_date: Optional[date],
        safety_floor: Decimal = Decimal("200.00"),
        safety_floor_margin: Decimal = Decimal("500.00"),
    ) -> SafeToPayCertificate:
        """Helper to construct a deterministic SafeToPayCertificate."""
        is_full_today = (amount_safe_to_pay == requested_amount)
        is_full_later = (earliest_date is not None and earliest_date > self.request_date)
        return SafeToPayCertificate(
            request_id=self.request.request_id,
            user_id=self.user_id,
            request_date=self.request_date,
            simulation_start=self.sim_start,
            simulation_end=self.sim_end,
            requested_amount=requested_amount,
            amount_safe_to_pay=amount_safe_to_pay,
            currency=self.home_currency,
            safety_floor=safety_floor,
            baseline_minimum_available_cash=Decimal("1000.00"),
            baseline_minimum_cash_date=self.request_date,
            limiting_date=self.request_date,
            available_cash_after_purchase_today=Decimal("1000.00") - amount_safe_to_pay,
            minimum_available_cash_after_purchase=safety_floor + safety_floor_margin,
            safety_floor_margin=safety_floor_margin,
            earliest_date_for_full_payment=earliest_date,
            is_full_payment_safe_today=is_full_today,
            is_full_payment_safe_later=is_full_later,
            reason_if_unsafe=None if is_full_today else "Not safe today",
            candidate_search_method="exact_full_amount" if is_full_today else "exact_headroom_boundary",
        )

    # -------------------------------------------------------------------------
    # Scenario A: Full amount safe today
    # -------------------------------------------------------------------------
    def test_case_a_full_amount_safe_today(self) -> None:
        cert = self._build_cert(
            requested_amount=Decimal("300.00"),
            amount_safe_to_pay=Decimal("300.00"),
            earliest_date=self.request_date,
        )
        res = classify_affordability(cert, self.profile, self.request)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertEqual(res.amount_safe_to_pay, Decimal("300.00"))
        self.assertEqual(res.earliest_date_for_full_payment, self.request_date)
        self.assertIn("safe to pay on request date", res.reason)
        self.assertFalse(res.is_provisional)
        self.assertFalse(res.requires_payment_plan_evaluation)

    # -------------------------------------------------------------------------
    # Scenario B: Partial safe today + plan feasible
    # -------------------------------------------------------------------------
    def test_case_b_partial_safe_today_plan_feasible(self) -> None:
        cert = self._build_cert(
            requested_amount=Decimal("500.00"),
            amount_safe_to_pay=Decimal("200.00"),
            earliest_date=self.request_date + timedelta(days=30),
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=True)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_WITH_PLAN)
        self.assertEqual(res.amount_safe_to_pay, Decimal("200.00"))
        self.assertEqual(res.earliest_date_for_full_payment, self.request_date + timedelta(days=30))
        self.assertIn("completed safely via valid payment plan", res.reason)
        self.assertFalse(res.is_provisional)
        self.assertFalse(res.requires_payment_plan_evaluation)

    # -------------------------------------------------------------------------
    # Scenario C: Partial safe today + plan infeasible + later full payment
    # -------------------------------------------------------------------------
    def test_case_c_partial_safe_today_plan_infeasible_later_full(self) -> None:
        later_d = self.request_date + timedelta(days=25)
        cert = self._build_cert(
            requested_amount=Decimal("500.00"),
            amount_safe_to_pay=Decimal("200.00"),
            earliest_date=later_d,
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_LATER)
        self.assertEqual(res.amount_safe_to_pay, Decimal("200.00"))
        self.assertEqual(res.earliest_date_for_full_payment, later_d)
        self.assertIn(f"Full amount safe from {later_d.isoformat()}", res.reason)
        self.assertFalse(res.is_provisional)
        self.assertFalse(res.requires_payment_plan_evaluation)

    # -------------------------------------------------------------------------
    # Scenario D: Zero safe today + plan feasible
    # -------------------------------------------------------------------------
    def test_case_d_zero_safe_today_plan_feasible(self) -> None:
        cert = self._build_cert(
            requested_amount=Decimal("500.00"),
            amount_safe_to_pay=Decimal("0.00"),
            earliest_date=self.request_date + timedelta(days=40),
            safety_floor_margin=Decimal("0.00"),
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=True)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_WITH_PLAN)
        self.assertEqual(res.amount_safe_to_pay, Decimal("0.00"))
        self.assertIn("completed safely via valid payment plan", res.reason)
        self.assertFalse(res.is_provisional)

    # -------------------------------------------------------------------------
    # Scenario E: Zero safe today + plan infeasible + later full payment
    # -------------------------------------------------------------------------
    def test_case_e_zero_safe_today_plan_infeasible_later_full(self) -> None:
        later_d = self.request_date + timedelta(days=40)
        cert = self._build_cert(
            requested_amount=Decimal("500.00"),
            amount_safe_to_pay=Decimal("0.00"),
            earliest_date=later_d,
            safety_floor_margin=Decimal("0.00"),
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_LATER)
        self.assertEqual(res.amount_safe_to_pay, Decimal("0.00"))
        self.assertEqual(res.earliest_date_for_full_payment, later_d)
        self.assertIn(f"Full amount safe from {later_d.isoformat()}", res.reason)
        self.assertFalse(res.is_provisional)

    # -------------------------------------------------------------------------
    # Scenario F: Zero safe + no later date (never safe)
    # -------------------------------------------------------------------------
    def test_case_f_zero_safe_no_later_date(self) -> None:
        cert = self._build_cert(
            requested_amount=Decimal("500.00"),
            amount_safe_to_pay=Decimal("0.00"),
            earliest_date=None,
            safety_floor_margin=Decimal("-50.00"),
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.NOT_AFFORDABLE)
        self.assertEqual(res.amount_safe_to_pay, Decimal("0.00"))
        self.assertIsNone(res.earliest_date_for_full_payment)
        self.assertIn("No safe full-payment date", res.reason)
        self.assertFalse(res.is_provisional)

    # -------------------------------------------------------------------------
    # Scenario G: Exact safety-floor equality (margin == 0)
    # -------------------------------------------------------------------------
    def test_case_g_exact_safety_floor_equality(self) -> None:
        # User has exactly 1000 cash, safety floor 200, headroom 800.
        # Requested amount 800. Amount safe to pay 800 leaves available cash at exactly 200.
        cert = self._build_cert(
            requested_amount=Decimal("800.00"),
            amount_safe_to_pay=Decimal("800.00"),
            earliest_date=self.request_date,
            safety_floor_margin=Decimal("0.00"),
        )
        res = classify_affordability(cert, self.profile, self.request)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertEqual(res.amount_safe_to_pay, Decimal("800.00"))
        self.assertEqual(res.certificate_reference.safety_floor_margin, Decimal("0.00"))

    # -------------------------------------------------------------------------
    # Scenario H: Earliest date = request date
    # -------------------------------------------------------------------------
    def test_case_h_earliest_date_equals_request_date(self) -> None:
        cert = self._build_cert(
            requested_amount=Decimal("150.00"),
            amount_safe_to_pay=Decimal("150.00"),
            earliest_date=self.request_date,
        )
        res = classify_affordability(cert, self.profile, self.request)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertEqual(res.earliest_date_for_full_payment, self.request_date)

    # -------------------------------------------------------------------------
    # Scenario I: Earliest date at 90-day horizon boundary
    # -------------------------------------------------------------------------
    def test_case_i_earliest_date_at_horizon_boundary(self) -> None:
        boundary_d = self.sim_end
        cert = self._build_cert(
            requested_amount=Decimal("600.00"),
            amount_safe_to_pay=Decimal("100.00"),
            earliest_date=boundary_d,
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_LATER)
        self.assertEqual(res.earliest_date_for_full_payment, boundary_d)

    # -------------------------------------------------------------------------
    # Scenario J: Earliest date = None
    # -------------------------------------------------------------------------
    def test_case_j_earliest_date_none(self) -> None:
        cert = self._build_cert(
            requested_amount=Decimal("600.00"),
            amount_safe_to_pay=Decimal("100.00"),
            earliest_date=None,
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.NOT_AFFORDABLE)
        self.assertIsNone(res.earliest_date_for_full_payment)
        self.assertIn("No safe full-payment date", res.reason)

    # -------------------------------------------------------------------------
    # Scenario K: Already-breached baseline
    # -------------------------------------------------------------------------
    def test_case_k_already_breached_baseline(self) -> None:
        cert = self._build_cert(
            requested_amount=Decimal("100.00"),
            amount_safe_to_pay=Decimal("0.00"),
            earliest_date=None,
            safety_floor_margin=Decimal("-100.00"),
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.NOT_AFFORDABLE)
        self.assertEqual(res.amount_safe_to_pay, Decimal("0.00"))
        self.assertIsNone(res.earliest_date_for_full_payment)

    # -------------------------------------------------------------------------
    # Scenario L: Requested amount = 0
    # -------------------------------------------------------------------------
    def test_case_l_requested_amount_zero(self) -> None:
        zero_req = FinancialRequest(
            request_id="req_zero",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("0.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="Zero cost item",
        )
        cert = self._build_cert(
            requested_amount=Decimal("0.00"),
            amount_safe_to_pay=Decimal("0.00"),
            earliest_date=self.request_date,
        )
        res = classify_affordability(cert, self.profile, zero_req)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertEqual(res.amount_safe_to_pay, Decimal("0.00"))
        self.assertEqual(res.earliest_date_for_full_payment, self.request_date)
        self.assertIn("Zero requested amount is safe", res.reason)

    # -------------------------------------------------------------------------
    # Scenario M: Pending debit reservation
    # -------------------------------------------------------------------------
    def test_case_m_pending_debit_reservation(self) -> None:
        # Full cash 1000, safety floor 200 -> max 800.
        # But a pending hold of 300 reduces available cash to 700.
        # Safe headroom is 700 - 200 = 500.
        # For requested amount 600, safe amount is 500 (partial).
        cert = self._build_cert(
            requested_amount=Decimal("600.00"),
            amount_safe_to_pay=Decimal("500.00"),
            earliest_date=self.request_date + timedelta(days=15),
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=True)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_WITH_PLAN)
        self.assertEqual(res.amount_safe_to_pay, Decimal("500.00"))

    # -------------------------------------------------------------------------
    # Scenario N: Large future income makes payment affordable later
    # -------------------------------------------------------------------------
    def test_case_n_large_future_income(self) -> None:
        salary_date = self.request_date + timedelta(days=20)
        cert = self._build_cert(
            requested_amount=Decimal("2000.00"),
            amount_safe_to_pay=Decimal("300.00"),
            earliest_date=salary_date,
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_LATER)
        self.assertEqual(res.earliest_date_for_full_payment, salary_date)
        self.assertIn(f"Full amount safe from {salary_date.isoformat()}", res.reason)

    # -------------------------------------------------------------------------
    # Scenario O: No speculative future income
    # -------------------------------------------------------------------------
    def test_case_o_no_speculative_future_income(self) -> None:
        # Without confirmed salary/income, cash remains below target, earliest date is None.
        cert = self._build_cert(
            requested_amount=Decimal("5000.00"),
            amount_safe_to_pay=Decimal("400.00"),
            earliest_date=None,
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.NOT_AFFORDABLE)
        self.assertIsNone(res.earliest_date_for_full_payment)
        self.assertIn("No safe full-payment date", res.reason)

    # -------------------------------------------------------------------------
    # Additional: User preference refusal of full payment
    # -------------------------------------------------------------------------
    def test_user_policy_refusal_of_full_payment(self) -> None:
        # User has sufficient cash today (amount_safe == requested),
        # but user profile explicitly EXCLUDES "full_payment" (e.g. only accepts installments).
        installment_only_profile = FinancialProfile(
            user_id=self.user_id,
            home_currency=self.home_currency,
            current_available_balance=Decimal("1000.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("installments", "partial_payment"),
            max_installment_months=6,
        )
        cert = self._build_cert(
            requested_amount=Decimal("300.00"),
            amount_safe_to_pay=Decimal("300.00"),
            earliest_date=self.request_date,
        )
        # If plan is feasible: affordable_with_plan
        res_plan = classify_affordability(cert, installment_only_profile, self.request, payment_plan_feasible=True)
        self.assertEqual(res_plan.status, AffordabilityStatus.AFFORDABLE_WITH_PLAN)
        self.assertIn("user policy requires an installment or partial payment plan", res_plan.reason)

        # If plan is infeasible: not_affordable
        res_noplan = classify_affordability(cert, installment_only_profile, self.request, payment_plan_feasible=False)
        self.assertEqual(res_noplan.status, AffordabilityStatus.NOT_AFFORDABLE)
        self.assertIn("user will not consider full payment", res_noplan.reason)

    # -------------------------------------------------------------------------
    # Protocol compliance: PaymentPlanFeasibility
    # -------------------------------------------------------------------------
    def test_payment_plan_feasibility_protocol_compliance(self) -> None:
        cert = self._build_cert(
            requested_amount=Decimal("500.00"),
            amount_safe_to_pay=Decimal("100.00"),
            earliest_date=self.request_date + timedelta(days=30),
        )
        valid_plan = MockPlanFeasibility(has_valid=True, plan_type="installments")
        self.assertTrue(isinstance(valid_plan, PaymentPlanFeasibility))
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=valid_plan)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_WITH_PLAN)

        invalid_plan = MockPlanFeasibility(has_valid=False, plan_type=None)
        res_invalid = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=invalid_plan)
        self.assertEqual(res_invalid.status, AffordabilityStatus.AFFORDABLE_LATER)

    # -------------------------------------------------------------------------
    # Provisional state when payment_plan_feasible is None
    # -------------------------------------------------------------------------
    def test_provisional_handling_when_plan_is_none(self) -> None:
        cert = self._build_cert(
            requested_amount=Decimal("500.00"),
            amount_safe_to_pay=Decimal("100.00"),
            earliest_date=self.request_date + timedelta(days=30),
        )
        res = classify_affordability(cert, self.profile, self.request, payment_plan_feasible=None)
        # Status must strictly be one of the four contest statuses:
        self.assertIn(res.status, (
            AffordabilityStatus.AFFORDABLE_NOW,
            AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            AffordabilityStatus.AFFORDABLE_LATER,
            AffordabilityStatus.NOT_AFFORDABLE,
        ))
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_LATER)
        self.assertTrue(res.is_provisional)
        self.assertTrue(res.requires_payment_plan_evaluation)
        self.assertEqual(res.provisional_status, AffordabilityStatus.AFFORDABLE_LATER)

    def test_affordability_status_enum_has_strictly_four_contest_values(self) -> None:
        """Contest rule: AffordabilityStatus must contain strictly four values."""
        self.assertEqual(len(AffordabilityStatus), 4)
        expected_values = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
        actual_values = {s.value for s in AffordabilityStatus}
        self.assertEqual(actual_values, expected_values)



class TestAffordabilityProperties(unittest.TestCase):
    """Metamorphic property tests for Section 15."""

    def setUp(self) -> None:
        self.user_id = "test_prop_user"
        self.request_date = date(2025, 1, 1)
        self.sim_start = self.request_date
        self.sim_end = self.request_date + timedelta(days=90)
        self.home_currency = "USD"

        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency=self.home_currency,
            current_available_balance=Decimal("2000.00"),
            minimum_balance_to_keep=Decimal("500.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=("rent",),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("streaming",),
            payment_methods_user_will_consider=("full_payment", "installments"),
            max_installment_months=6,
        )

    # -------------------------------------------------------------------------
    # PROPERTY 1: Reducing requested amount cannot worsen affordability
    # -------------------------------------------------------------------------
    def test_property_1_reducing_amount_cannot_worsen_affordability(self) -> None:
        canonical_events: Sequence[CanonicalEvent] = []
        future_events: Sequence[FutureEvent] = []

        baseline = simulate_user(
            user_id=self.user_id,
            simulation_start=self.sim_start,
            simulation_end=self.sim_end,
            canonical_events=canonical_events,
            future_events=future_events,
            profile=self.profile,
        )

        amounts = [Decimal("2500.00"), Decimal("1800.00"), Decimal("1500.00"), Decimal("1000.00"), Decimal("500.00"), Decimal("0.00")]
        previous_rank = -1

        for amt in reversed(amounts):  # test monotonically increasing amounts (decreasing hardness)
            req = FinancialRequest(
                request_id="prop_req",
                user_id=self.user_id,
                request_date=self.request_date,
                request_type="purchase",
                requested_amount=amt,
                desired_completion_date=self.request_date + timedelta(days=60),
                allows_partial_payment=True,
                request_text="Testing property 1",
            )
            cert = evaluate_request_safe_to_pay(
                request=req,
                baseline_simulation=baseline,
                canonical_events=canonical_events,
                future_events=future_events,
                profile=self.profile,
            )
            res = classify_affordability(cert, self.profile, req, payment_plan_feasible=False)
            rank = status_rank(res.status)
            if previous_rank != -1:
                self.assertGreaterEqual(
                    previous_rank,
                    rank,
                    f"Affordability worsened when reducing requested amount from larger to {amt}: rank {previous_rank} vs {rank}",
                )
            previous_rank = rank

    # -------------------------------------------------------------------------
    # PROPERTY 2: Increasing safety floor cannot improve affordability
    # -------------------------------------------------------------------------
    def test_property_2_increasing_safety_floor_cannot_improve_affordability(self) -> None:
        req = FinancialRequest(
            request_id="prop_req_2",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("1200.00"),
            desired_completion_date=self.request_date + timedelta(days=60),
            allows_partial_payment=True,
            request_text="Testing property 2",
        )
        canonical_events: Sequence[CanonicalEvent] = []
        future_events: Sequence[FutureEvent] = []

        floors = [Decimal("100.00"), Decimal("300.00"), Decimal("500.00"), Decimal("800.00"), Decimal("1200.00")]
        previous_rank = 999

        for floor in floors:
            prof = FinancialProfile(
                user_id=self.user_id,
                home_currency=self.home_currency,
                current_available_balance=Decimal("2000.00"),
                minimum_balance_to_keep=floor,
                financial_priorities=("savings",),
                expense_categories_to_protect=(),
                expense_categories_user_is_willing_to_reduce=(),
                expense_categories_user_is_willing_to_stop=(),
                payment_methods_user_will_consider=("full_payment",),
                max_installment_months=None,
            )
            baseline = simulate_user(
                user_id=self.user_id,
                simulation_start=self.sim_start,
                simulation_end=self.sim_end,
                canonical_events=canonical_events,
                future_events=future_events,
                profile=prof,
            )
            cert = evaluate_request_safe_to_pay(
                request=req,
                baseline_simulation=baseline,
                canonical_events=canonical_events,
                future_events=future_events,
                profile=prof,
            )
            res = classify_affordability(cert, prof, req, payment_plan_feasible=False)
            rank = status_rank(res.status)
            self.assertLessEqual(
                rank,
                previous_rank,
                f"Affordability improved when safety floor was raised to {floor}: {rank} > {previous_rank}",
            )
            previous_rank = rank

    # -------------------------------------------------------------------------
    def _make_canonical_event(
        self,
        event_id: str,
        effective_date: date,
        direction: Direction,
        amount: Decimal,
        impact_type: CashImpactType,
        category: str = "general",
        status: str = "settled",
    ) -> CanonicalEvent:
        return CanonicalEvent(
            event_id=event_id,
            user_id=self.user_id,
            source_row=1,
            effective_date=effective_date,
            direction=direction,
            direction_original=direction.value,
            amount_original=amount,
            currency_original=self.home_currency,
            amount_home=amount,
            home_currency=self.home_currency,
            exchange_rate_used=Decimal("1"),
            exchange_rate_date=effective_date,
            status=status,
            is_cash_event=True,
            cash_impact_type=impact_type,
            event_type="expense" if direction == Direction.OUTFLOW else "income",
            category=category,
            description=f"Test {event_id}",
            flexibility="fixed",
            minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None,
            linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False,
            unresolved_reason=None,
            evidence_chain=(),
            applied_actions=(),
        )

    # -------------------------------------------------------------------------
    # PROPERTY 3: Removing future confirmed inflow cannot improve affordability
    # -------------------------------------------------------------------------
    def test_property_3_removing_future_inflow_cannot_improve_affordability(self) -> None:
        req = FinancialRequest(
            request_id="prop_req_3",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("1800.00"),
            desired_completion_date=self.request_date + timedelta(days=60),
            allows_partial_payment=True,
            request_text="Testing property 3",
        )
        canonical_events: Sequence[CanonicalEvent] = []
        future_salary = FutureEvent(
            event_id="fut_salary",
            user_id=self.user_id,
            effective_date=self.request_date + timedelta(days=15),
            direction=Direction.INFLOW,
            amount_home=Decimal("1000.00"),
            currency=self.home_currency,
            category="income",
            event_type="salary",
            description="Confirmed monthly salary",
            series_id="series_salary",
            frequency=RecurrenceFrequency.MONTHLY,
        )

        # With future inflow
        base_with_inflow = simulate_user(
            user_id=self.user_id,
            simulation_start=self.sim_start,
            simulation_end=self.sim_end,
            canonical_events=canonical_events,
            future_events=[future_salary],
            profile=self.profile,
        )
        cert_with = evaluate_request_safe_to_pay(
            request=req,
            baseline_simulation=base_with_inflow,
            canonical_events=canonical_events,
            future_events=[future_salary],
            profile=self.profile,
        )
        res_with = classify_affordability(cert_with, self.profile, req, payment_plan_feasible=False)

        # Without future inflow
        base_without_inflow = simulate_user(
            user_id=self.user_id,
            simulation_start=self.sim_start,
            simulation_end=self.sim_end,
            canonical_events=canonical_events,
            future_events=[],
            profile=self.profile,
        )
        cert_without = evaluate_request_safe_to_pay(
            request=req,
            baseline_simulation=base_without_inflow,
            canonical_events=canonical_events,
            future_events=[],
            profile=self.profile,
        )
        res_without = classify_affordability(cert_without, self.profile, req, payment_plan_feasible=False)

        rank_with = status_rank(res_with.status)
        rank_without = status_rank(res_without.status)
        self.assertLessEqual(
            rank_without,
            rank_with,
            f"Removing confirmed future inflow improved status: {rank_without} > {rank_with}",
        )

    # -------------------------------------------------------------------------
    # PROPERTY 4: Increasing pending debit reserve cannot improve affordability
    # -------------------------------------------------------------------------
    def test_property_4_increasing_pending_reserve_cannot_improve_affordability(self) -> None:
        req = FinancialRequest(
            request_id="prop_req_4",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("1200.00"),
            desired_completion_date=self.request_date + timedelta(days=60),
            allows_partial_payment=True,
            request_text="Testing property 4",
        )

        reserves = [Decimal("0.00"), Decimal("200.00"), Decimal("500.00"), Decimal("800.00")]
        previous_rank = 999

        for res_amt in reserves:
            events: List[CanonicalEvent] = []
            if res_amt > Decimal("0"):
                events.append(
                    self._make_canonical_event(
                        event_id=f"pending_hold_{res_amt}",
                        effective_date=self.request_date,
                        direction=Direction.OUTFLOW,
                        amount=res_amt,
                        impact_type=CashImpactType.PENDING_DEBIT_RESERVED,
                        category="shopping",
                        status="pending",
                    )
                )

            baseline = simulate_user(
                user_id=self.user_id,
                simulation_start=self.sim_start,
                simulation_end=self.sim_end,
                canonical_events=events,
                future_events=[],
                profile=self.profile,
            )
            cert = evaluate_request_safe_to_pay(
                request=req,
                baseline_simulation=baseline,
                canonical_events=events,
                future_events=[],
                profile=self.profile,
            )
            res = classify_affordability(cert, self.profile, req, payment_plan_feasible=False)
            rank = status_rank(res.status)
            self.assertLessEqual(
                rank,
                previous_rank,
                f"Affordability improved when pending reserve increased to {res_amt}: {rank} > {previous_rank}",
            )
            previous_rank = rank

    # -------------------------------------------------------------------------
    # PROPERTY 5: Full safe today implies affordable_now regardless of plan feasibility
    # -------------------------------------------------------------------------
    def test_property_5_full_safe_today_implies_affordable_now(self) -> None:
        req = FinancialRequest(
            request_id="prop_req_5",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("500.00"),
            desired_completion_date=self.request_date + timedelta(days=60),
            allows_partial_payment=True,
            request_text="Testing property 5",
        )
        baseline = simulate_user(
            user_id=self.user_id,
            simulation_start=self.sim_start,
            simulation_end=self.sim_end,
            canonical_events=[],
            future_events=[],
            profile=self.profile,
        )
        cert = evaluate_request_safe_to_pay(
            request=req,
            baseline_simulation=baseline,
            canonical_events=[],
            future_events=[],
            profile=self.profile,
        )
        self.assertTrue(cert.is_full_payment_safe_today)

        # Status must be affordable_now whether plan_feasible is None, True, or False:
        for plan_val in [None, True, False]:
            res = classify_affordability(cert, self.profile, req, payment_plan_feasible=plan_val)
            self.assertEqual(
                res.status,
                AffordabilityStatus.AFFORDABLE_NOW,
                f"Full safe today request was not affordable_now when plan_feasible={plan_val}",
            )

    # -------------------------------------------------------------------------
    # PROPERTY 6: classify_affordability is pure and never alters simulator results
    # -------------------------------------------------------------------------
    def test_property_6_affordability_never_changes_simulator_results(self) -> None:
        req = FinancialRequest(
            request_id="prop_req_6",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("800.00"),
            desired_completion_date=self.request_date + timedelta(days=60),
            allows_partial_payment=True,
            request_text="Testing property 6",
        )
        baseline = simulate_user(
            user_id=self.user_id,
            simulation_start=self.sim_start,
            simulation_end=self.sim_end,
            canonical_events=[],
            future_events=[],
            profile=self.profile,
        )
        cert = evaluate_request_safe_to_pay(
            request=req,
            baseline_simulation=baseline,
            canonical_events=[],
            future_events=[],
            profile=self.profile,
        )

        min_cash_before = baseline.minimum_projected_available_cash
        ending_cash_before = baseline.ending_state.available_cash
        safe_amt_before = cert.amount_safe_to_pay
        earliest_before = cert.earliest_date_for_full_payment

        _ = classify_affordability(cert, self.profile, req, payment_plan_feasible=True)
        _ = classify_affordability(cert, self.profile, req, payment_plan_feasible=False)
        _ = classify_affordability(cert, self.profile, req, payment_plan_feasible=None)

        self.assertEqual(baseline.minimum_projected_available_cash, min_cash_before)
        self.assertEqual(baseline.ending_state.available_cash, ending_cash_before)
        self.assertEqual(cert.amount_safe_to_pay, safe_amt_before)
        self.assertEqual(cert.earliest_date_for_full_payment, earliest_before)

    # -------------------------------------------------------------------------
    # PROPERTY 7: Changing only payment preference from accepting full_payment
    # to rejecting full_payment must NEVER change financial capacity.
    # -------------------------------------------------------------------------
    def test_property_7_changing_payment_preference_does_not_change_financial_capacity(self) -> None:
        req = FinancialRequest(
            request_id="prop_req_7",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("500.00"),
            desired_completion_date=self.request_date + timedelta(days=60),
            allows_partial_payment=True,
            request_text="Testing property 7",
        )
        canonical_events: Sequence[CanonicalEvent] = []
        future_events: Sequence[FutureEvent] = []

        prof_accept = FinancialProfile(
            user_id=self.user_id,
            home_currency=self.home_currency,
            current_available_balance=Decimal("2000.00"),
            minimum_balance_to_keep=Decimal("500.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment", "installments"),
            max_installment_months=6,
        )

        prof_reject = FinancialProfile(
            user_id=self.user_id,
            home_currency=self.home_currency,
            current_available_balance=Decimal("2000.00"),
            minimum_balance_to_keep=Decimal("500.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("installments", "partial_payment"),
            max_installment_months=6,
        )

        base_accept = simulate_user(
            user_id=self.user_id,
            simulation_start=self.sim_start,
            simulation_end=self.sim_end,
            canonical_events=canonical_events,
            future_events=future_events,
            profile=prof_accept,
        )
        base_reject = simulate_user(
            user_id=self.user_id,
            simulation_start=self.sim_start,
            simulation_end=self.sim_end,
            canonical_events=canonical_events,
            future_events=future_events,
            profile=prof_reject,
        )

        cert_accept = evaluate_request_safe_to_pay(req, base_accept, canonical_events, future_events, prof_accept)
        cert_reject = evaluate_request_safe_to_pay(req, base_reject, canonical_events, future_events, prof_reject)

        # Simulator outputs must be strictly identical
        self.assertEqual(base_accept.minimum_projected_available_cash, base_reject.minimum_projected_available_cash)
        self.assertEqual(base_accept.ending_state.available_cash, base_reject.ending_state.available_cash)

        # Financial capacity measures must be strictly identical
        self.assertEqual(cert_accept.amount_safe_to_pay, cert_reject.amount_safe_to_pay)
        self.assertEqual(cert_accept.earliest_date_for_full_payment, cert_reject.earliest_date_for_full_payment)
        self.assertEqual(cert_accept.safety_floor, cert_reject.safety_floor)

        # Only affordability classification may change:
        res_accept = classify_affordability(cert_accept, prof_accept, req)
        res_reject = classify_affordability(cert_reject, prof_reject, req)

        self.assertEqual(res_accept.status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertFalse(res_accept.requires_payment_plan_evaluation)

        self.assertNotEqual(res_reject.status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertTrue(res_reject.requires_payment_plan_evaluation)


class TestProvisionalCaseMatrix(unittest.TestCase):
    """Explicit matrix tests for Section 7 (Cases A through J)."""

    def setUp(self) -> None:
        self.user_id = "matrix_user"
        self.request_date = date(2025, 1, 1)
        self.sim_start = self.request_date
        self.sim_end = self.request_date + timedelta(days=90)
        self.currency = "USD"

        self.profile_accept_full = FinancialProfile(
            user_id=self.user_id,
            home_currency=self.currency,
            current_available_balance=Decimal("1000.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment", "installments"),
            max_installment_months=6,
        )

        self.profile_reject_full = FinancialProfile(
            user_id=self.user_id,
            home_currency=self.currency,
            current_available_balance=Decimal("1000.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("installments", "partial_payment"),
            max_installment_months=6,
        )

        self.request = FinancialRequest(
            request_id="req_matrix",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("500.00"),
            desired_completion_date=self.request_date + timedelta(days=60),
            allows_partial_payment=True,
            request_text="Matrix test",
        )

    def _make_cert(
        self,
        requested_amount: Decimal,
        amount_safe_to_pay: Decimal,
        earliest_date: Optional[date],
    ) -> SafeToPayCertificate:
        is_full = (amount_safe_to_pay == requested_amount)
        return SafeToPayCertificate(
            request_id=self.request.request_id,
            user_id=self.user_id,
            request_date=self.request_date,
            simulation_start=self.sim_start,
            simulation_end=self.sim_end,
            requested_amount=requested_amount,
            amount_safe_to_pay=amount_safe_to_pay,
            currency=self.currency,
            safety_floor=Decimal("200.00"),
            baseline_minimum_available_cash=Decimal("1000.00"),
            baseline_minimum_cash_date=self.request_date,
            limiting_date=self.request_date,
            available_cash_after_purchase_today=Decimal("1000.00") - amount_safe_to_pay,
            minimum_available_cash_after_purchase=Decimal("500.00"),
            safety_floor_margin=Decimal("300.00"),
            earliest_date_for_full_payment=earliest_date,
            is_full_payment_safe_today=is_full,
            is_full_payment_safe_later=(earliest_date is not None and earliest_date > self.request_date),
            reason_if_unsafe=None if is_full else "Not fully safe today",
            candidate_search_method="matrix",
        )

    # Matrix A: full safe today + full_payment accepted
    def test_matrix_a_full_safe_accepted(self) -> None:
        cert = self._make_cert(Decimal("500.00"), Decimal("500.00"), self.request_date)
        res = classify_affordability(cert, self.profile_accept_full, self.request, payment_plan_feasible=None)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertFalse(res.requires_payment_plan_evaluation)
        self.assertFalse(res.is_provisional)

    # Matrix B: full safe today + full_payment rejected
    def test_matrix_b_full_safe_rejected(self) -> None:
        cert = self._make_cert(Decimal("500.00"), Decimal("500.00"), self.request_date)
        res = classify_affordability(cert, self.profile_reject_full, self.request, payment_plan_feasible=None)
        self.assertNotEqual(res.status, AffordabilityStatus.AFFORDABLE_NOW)
        self.assertTrue(res.requires_payment_plan_evaluation)
        self.assertTrue(res.is_provisional)
        self.assertEqual(res.status, AffordabilityStatus.NOT_AFFORDABLE)  # conservative fallback

    # Matrix C: partial safe today + plan unknown
    def test_matrix_c_partial_safe_plan_unknown(self) -> None:
        cert = self._make_cert(Decimal("500.00"), Decimal("200.00"), self.request_date + timedelta(days=20))
        res = classify_affordability(cert, self.profile_accept_full, self.request, payment_plan_feasible=None)
        self.assertTrue(res.requires_payment_plan_evaluation)
        self.assertTrue(res.is_provisional)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_LATER)  # conservative fallback

    # Matrix D: zero safe today + plan unknown
    def test_matrix_d_zero_safe_plan_unknown(self) -> None:
        cert = self._make_cert(Decimal("500.00"), Decimal("0.00"), None)
        res = classify_affordability(cert, self.profile_accept_full, self.request, payment_plan_feasible=None)
        self.assertTrue(res.requires_payment_plan_evaluation)
        self.assertTrue(res.is_provisional)
        self.assertEqual(res.status, AffordabilityStatus.NOT_AFFORDABLE)  # conservative fallback

    # Matrix E: partial safe today + plan feasible
    def test_matrix_e_partial_safe_plan_feasible(self) -> None:
        cert = self._make_cert(Decimal("500.00"), Decimal("200.00"), self.request_date + timedelta(days=20))
        res = classify_affordability(cert, self.profile_accept_full, self.request, payment_plan_feasible=True)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_WITH_PLAN)
        self.assertFalse(res.requires_payment_plan_evaluation)
        self.assertFalse(res.is_provisional)

    # Matrix F: partial safe today + plan infeasible + later date
    def test_matrix_f_partial_safe_plan_infeasible_later_date(self) -> None:
        later_d = self.request_date + timedelta(days=20)
        cert = self._make_cert(Decimal("500.00"), Decimal("200.00"), later_d)
        res = classify_affordability(cert, self.profile_accept_full, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_LATER)
        self.assertEqual(res.earliest_date_for_full_payment, later_d)
        self.assertFalse(res.requires_payment_plan_evaluation)
        self.assertFalse(res.is_provisional)

    # Matrix G: partial safe today + plan infeasible + no later date
    def test_matrix_g_partial_safe_plan_infeasible_no_later_date(self) -> None:
        cert = self._make_cert(Decimal("500.00"), Decimal("200.00"), None)
        res = classify_affordability(cert, self.profile_accept_full, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.NOT_AFFORDABLE)
        self.assertIsNone(res.earliest_date_for_full_payment)
        self.assertFalse(res.requires_payment_plan_evaluation)
        self.assertFalse(res.is_provisional)

    # Matrix H: full safe today + full_payment rejected + plan feasible
    def test_matrix_h_full_safe_rejected_plan_feasible(self) -> None:
        cert = self._make_cert(Decimal("500.00"), Decimal("500.00"), self.request_date)
        res = classify_affordability(cert, self.profile_reject_full, self.request, payment_plan_feasible=True)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_WITH_PLAN)
        self.assertFalse(res.requires_payment_plan_evaluation)
        self.assertFalse(res.is_provisional)

    # Matrix I: full safe today + full_payment rejected + plan infeasible
    def test_matrix_i_full_safe_rejected_plan_infeasible(self) -> None:
        cert = self._make_cert(Decimal("500.00"), Decimal("500.00"), self.request_date)
        res = classify_affordability(cert, self.profile_reject_full, self.request, payment_plan_feasible=False)
        self.assertEqual(res.status, AffordabilityStatus.NOT_AFFORDABLE)
        self.assertFalse(res.requires_payment_plan_evaluation)
        self.assertFalse(res.is_provisional)

    # Matrix J: full unsafe today + plan feasible
    def test_matrix_j_full_unsafe_plan_feasible(self) -> None:
        cert = self._make_cert(Decimal("500.00"), Decimal("0.00"), None)
        res = classify_affordability(cert, self.profile_accept_full, self.request, payment_plan_feasible=True)
        self.assertEqual(res.status, AffordabilityStatus.AFFORDABLE_WITH_PLAN)
        self.assertFalse(res.requires_payment_plan_evaluation)
        self.assertFalse(res.is_provisional)


if __name__ == "__main__":
    unittest.main()


