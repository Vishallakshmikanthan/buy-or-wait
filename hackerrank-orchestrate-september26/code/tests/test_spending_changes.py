"""Unit and integration tests for spending change optimization layer.

Covers Phases 13A through 13H:
A. Eligibility
B. Action semantics
C. Safety
D. Determinism
E. Optimization & Disruption Minimization
F. Candidate Integrity
G. Regression & Invariant Preservation
H. Adversarial Cases
"""

from __future__ import annotations

import unittest
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Tuple

from code.canonical import CanonicalEvent
from code.models import FinancialProfile, FinancialRequest, PaymentOption
from code.payment_plan import PaymentPlan, PaymentPlanFeasibility, PaymentScheduleEntry
from code.recurrence import AmountForecastingRule, FutureEvent, RecurrenceFrequency, RecurrenceSeries
from code.safe_to_pay import SafeToPayCertificate
from code.simulator import CashImpactType, Direction, SimulatedEvent, simulate_user
from code.spending_changes import (
    SpendingActionType,
    SpendingChange,
    SpendingChangeScenario,
    apply_spending_changes_overlay,
    format_change_amount,
    generate_spending_change_scenarios,
    identify_eligible_spending_actions,
    optimize_spending_changes_for_candidate,
)
from code.candidate_generation import (
    CandidateStatus,
    CandidateType,
    generate_candidates,
    generate_spending_change_candidates,
)
from code.ranking import rank_candidate_set


class TestSpendingChangesEligibility(unittest.TestCase):
    """Phase 13A: Eligibility Policy Gates."""

    def setUp(self) -> None:
        self.user_id = "user_test"
        self.req_date = date(2025, 1, 1)
        self.sim_end = date(2025, 4, 1)
        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("1000.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=("essential_expenses",),
            expense_categories_to_protect=("rent", "groceries"),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("gym", "streaming"),
            payment_methods_user_will_consider=("full_payment", "installments"),
            max_installment_months=12,
        )

    def _make_series(
        self,
        series_id: str,
        category: str,
        amount: str,
        flexibility: str,
        min_allowed: str = "0.00",
        is_protected: bool = False,
        is_cancelled: bool = False,
        direction: Direction = Direction.OUTFLOW,
    ) -> RecurrenceSeries:
        return RecurrenceSeries(
            series_id=series_id,
            user_id=self.user_id,
            direction=direction,
            category=category,
            event_type="recurring_expense",
            description=category,
            frequency=RecurrenceFrequency.MONTHLY,
            interval_days=30,
            day_of_month=1,
            historical_event_ids=(f"hist_{series_id}",),
            historical_count=3,
            anchor_event_id=f"event_{series_id}",
            anchor_date=date(2024, 12, 1),
            forecast_amount=Decimal(amount),
            currency="USD",
            amount_rule=AmountForecastingRule.EXACT_STABLE,
            flexibility=flexibility,
            minimum_allowed_amount=Decimal(min_allowed) if min_allowed is not None else None,
            is_protected=is_protected,
            is_cancelled=is_cancelled,
        )

    def _make_future_event(self, series_id: str, dt: date, amount: str) -> FutureEvent:
        return FutureEvent(
            event_id=f"fe_{series_id}_{dt.isoformat()}",
            user_id=self.user_id,
            effective_date=dt,
            direction=Direction.OUTFLOW,
            amount_home=Decimal(amount),
            currency="USD",
            category="test",
            event_type="recurring_expense",
            description="test",
            series_id=series_id,
            frequency=RecurrenceFrequency.MONTHLY,
            is_forecast=True,
            anchor_event_id=f"event_{series_id}",
            flexibility="stoppable",
            minimum_allowed_amount=None,
            is_protected=False,
        )

    def test_eligible_stoppable_series(self) -> None:
        """Flexible stoppable expense in willing_to_stop category is eligible."""
        s = self._make_series("gym_1", "gym", "50.00", "stoppable")
        fe = self._make_future_event("gym_1", date(2025, 1, 15), "50.00")
        actions = identify_eligible_spending_actions(
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_end=self.sim_end,
            profile=self.profile,
            series_list=[s],
            canonical_events=[],
            future_events=[fe],
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, SpendingActionType.STOP)
        self.assertEqual(actions[0].event_id, "event_gym_1")
        self.assertEqual(actions[0].modified_amount, Decimal("0"))

    def test_eligible_reducible_series(self) -> None:
        """Flexible reducible expense in willing_to_reduce category is eligible."""
        s = self._make_series("dining_1", "dining", "100.00", "reducible", min_allowed="40.00")
        fe = self._make_future_event("dining_1", date(2025, 1, 10), "100.00")
        actions = identify_eligible_spending_actions(
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_end=self.sim_end,
            profile=self.profile,
            series_list=[s],
            canonical_events=[],
            future_events=[fe],
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, SpendingActionType.REDUCE_TO)
        self.assertEqual(actions[0].event_id, "event_dining_1")
        self.assertEqual(actions[0].modified_amount, Decimal("40.00"))

    def test_fixed_series_rejected(self) -> None:
        """Non-flexible (fixed) expense is rejected."""
        s = self._make_series("gym_fixed", "gym", "50.00", "fixed")
        fe = self._make_future_event("gym_fixed", date(2025, 1, 15), "50.00")
        actions = identify_eligible_spending_actions(
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_end=self.sim_end,
            profile=self.profile,
            series_list=[s],
            canonical_events=[],
            future_events=[fe],
        )
        self.assertEqual(len(actions), 0)

    def test_protected_category_rejected(self) -> None:
        """Expense in expense_categories_to_protect is strictly rejected."""
        s = self._make_series("rent_1", "rent", "800.00", "stoppable")
        fe = self._make_future_event("rent_1", date(2025, 1, 5), "800.00")
        actions = identify_eligible_spending_actions(
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_end=self.sim_end,
            profile=self.profile,
            series_list=[s],
            canonical_events=[],
            future_events=[fe],
        )
        self.assertEqual(len(actions), 0)

    def test_protected_flag_rejected(self) -> None:
        """Expense with is_protected=True is strictly rejected."""
        s = self._make_series("gym_prot", "gym", "50.00", "stoppable", is_protected=True)
        fe = self._make_future_event("gym_prot", date(2025, 1, 15), "50.00")
        actions = identify_eligible_spending_actions(
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_end=self.sim_end,
            profile=self.profile,
            series_list=[s],
            canonical_events=[],
            future_events=[fe],
        )
        self.assertEqual(len(actions), 0)

    def test_cancelled_series_rejected(self) -> None:
        """Expense with is_cancelled=True is rejected."""
        s = self._make_series("gym_canc", "gym", "50.00", "stoppable", is_cancelled=True)
        fe = self._make_future_event("gym_canc", date(2025, 1, 15), "50.00")
        actions = identify_eligible_spending_actions(
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_end=self.sim_end,
            profile=self.profile,
            series_list=[s],
            canonical_events=[],
            future_events=[fe],
        )
        self.assertEqual(len(actions), 0)

    def test_inflow_series_rejected(self) -> None:
        """Income/inflow is never eligible for spending change."""
        s = self._make_series("salary", "salary", "2000.00", "stoppable", direction=Direction.INFLOW)
        fe = self._make_future_event("salary", date(2025, 1, 1), "2000.00")
        actions = identify_eligible_spending_actions(
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_end=self.sim_end,
            profile=self.profile,
            series_list=[s],
            canonical_events=[],
            future_events=[fe],
        )
        self.assertEqual(len(actions), 0)

    def test_unresolved_minimum_amount_rejected(self) -> None:
        """Reducible expense with missing minimum_allowed_amount cannot be reduced."""
        s = self._make_series("dining_no_min", "dining", "100.00", "reducible", min_allowed=None)
        fe = self._make_future_event("dining_no_min", date(2025, 1, 10), "100.00")
        actions = identify_eligible_spending_actions(
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_end=self.sim_end,
            profile=self.profile,
            series_list=[s],
            canonical_events=[],
            future_events=[fe],
        )
        self.assertEqual(len(actions), 0)


class TestSpendingChangesActionSemantics(unittest.TestCase):
    """Phase 13B: Overlay Action Semantics & Immutability."""

    def test_stop_action_removes_future_occurrences(self) -> None:
        fe1 = FutureEvent(
            event_id="fe_1", series_id="s_gym", anchor_event_id="ev_gym", user_id="u1",
            direction=Direction.OUTFLOW, category="gym", effective_date=date(2025, 1, 15),
            amount_home=Decimal("50.00"), currency="USD", event_type="recurring", description="gym",
            frequency=RecurrenceFrequency.MONTHLY, flexibility="stoppable", is_protected=False, is_forecast=True
        )
        fe2 = FutureEvent(
            event_id="fe_2", series_id="s_rent", anchor_event_id="ev_rent", user_id="u1",
            direction=Direction.OUTFLOW, category="rent", effective_date=date(2025, 1, 1),
            amount_home=Decimal("800.00"), currency="USD", event_type="recurring", description="rent",
            frequency=RecurrenceFrequency.MONTHLY, flexibility="fixed", is_protected=True, is_forecast=True
        )
        change = SpendingChange("ev_gym", "s_gym", SpendingActionType.STOP, Decimal("50.00"), Decimal("0"), date(2025, 1, 15), "gym", Decimal("50.00"), "Stop gym")

        orig_fut = (fe1, fe2)
        over_can, over_fut = apply_spending_changes_overlay((), orig_fut, [change])

        self.assertEqual(len(over_fut), 1)
        self.assertEqual(over_fut[0].event_id, "fe_2")
        # Ensure original tuple is untouched
        self.assertEqual(len(orig_fut), 2)

    def test_reduce_to_action_modifies_amount(self) -> None:
        fe1 = FutureEvent(
            event_id="fe_1", series_id="s_din", anchor_event_id="ev_din", user_id="u1",
            direction=Direction.OUTFLOW, category="dining", effective_date=date(2025, 1, 15),
            amount_home=Decimal("100.00"), currency="USD", event_type="recurring", description="dining",
            frequency=RecurrenceFrequency.MONTHLY, flexibility="reducible", is_protected=False, is_forecast=True
        )
        change = SpendingChange("ev_din", "s_din", SpendingActionType.REDUCE_TO, Decimal("100.00"), Decimal("40.00"), date(2025, 1, 15), "dining", Decimal("60.00"), "Reduce dining")

        over_can, over_fut = apply_spending_changes_overlay((), (fe1,), [change])

        self.assertEqual(len(over_fut), 1)
        self.assertEqual(over_fut[0].amount_home, Decimal("40.00"))
        # Original remains 100.00
        self.assertEqual(fe1.amount_home, Decimal("100.00"))


class TestSpendingChangesSafetyAndOptimization(unittest.TestCase):
    """Phases 13C & 13E: Re-Simulation, Safety Gate, and Disruption Minimization."""

    def setUp(self) -> None:
        self.user_id = "user_opt"
        self.start = date(2025, 1, 1)
        self.end = date(2025, 4, 1)
        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("500.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=("essential_expenses",),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("sub1", "sub2"),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=12,
        )

    def test_unsafe_purchase_becomes_safe_after_valid_stop(self) -> None:
        """A purchase that breaches the safety floor is rescued by stopping a flexible subscription."""
        fe_sub = FutureEvent(
            event_id="fe_sub", series_id="s_sub1", anchor_event_id="ev_sub1", user_id=self.user_id,
            direction=Direction.OUTFLOW, category="sub1", effective_date=date(2025, 1, 10),
            amount_home=Decimal("150.00"), currency="USD", event_type="recurring", description="sub1",
            frequency=RecurrenceFrequency.MONTHLY, flexibility="stoppable", is_protected=False, is_forecast=True
        )
        cand_event = SimulatedEvent(
            "p1", self.user_id, date(2025, 1, 5), Direction.OUTFLOW, Decimal("200.00"), "USD",
            "purchase", "action", "Purchase", CashImpactType.SETTLED_OUTFLOW, "fixed", False, "action", False
        )

        # 1. Base simulation breaches safety floor: 500 - 200 - 150 = 150 < 200
        base_res = simulate_user(
            user_id=self.user_id,
            simulation_start=self.start,
            simulation_end=self.end,
            canonical_events=(),
            future_events=(fe_sub,),
            profile=self.profile,
            additional_events=[cand_event],
        )
        self.assertTrue(base_res.is_safety_floor_breached)
        self.assertEqual(base_res.minimum_projected_available_cash, Decimal("150.00"))

        # 2. Optimization finds stop action
        action_stop = SpendingChange("ev_sub1", "s_sub1", SpendingActionType.STOP, Decimal("150.00"), Decimal("0"), date(2025, 1, 10), "sub1", Decimal("150.00"), "Stop sub1")
        scenarios = generate_spending_change_scenarios([action_stop], max_changes=1)
        best = optimize_spending_changes_for_candidate(
            user_id=self.user_id,
            simulation_start=self.start,
            simulation_end=self.end,
            profile=self.profile,
            canonical_events=(),
            future_events=(fe_sub,),
            scenarios=scenarios,
            candidate_events=(cand_event,),
        )
        self.assertIsNotNone(best)
        self.assertEqual(len(best.changes), 1)
        self.assertEqual(best.changes[0].action_type, SpendingActionType.STOP)

    def test_insufficient_reduction_rejected(self) -> None:
        """If a reduction is insufficient to keep cash above the safety floor, it is rejected."""
        # 500 - 300 - 50 = 150 < 200. With reduction to 40, cash is 500 - 300 - 40 = 160 < 200 (still breach).
        fe_din = FutureEvent(
            event_id="fe_din", series_id="s_din", anchor_event_id="ev_din", user_id=self.user_id,
            direction=Direction.OUTFLOW, category="dining", effective_date=date(2025, 1, 10),
            amount_home=Decimal("50.00"), currency="USD", event_type="recurring", description="dining",
            frequency=RecurrenceFrequency.MONTHLY, flexibility="reducible", is_protected=False, is_forecast=True
        )
        cand_event = SimulatedEvent(
            "p1", self.user_id, date(2025, 1, 5), Direction.OUTFLOW, Decimal("300.00"), "USD",
            "purchase", "action", "Purchase", CashImpactType.SETTLED_OUTFLOW, "fixed", False, "action", False
        )
        action_insufficient = SpendingChange("ev_din", "s_din", SpendingActionType.REDUCE_TO, Decimal("50.00"), Decimal("40.00"), date(2025, 1, 10), "dining", Decimal("10.00"), "Reduce dining")
        best = optimize_spending_changes_for_candidate(
            user_id=self.user_id,
            simulation_start=self.start,
            simulation_end=self.end,
            profile=self.profile,
            canonical_events=(),
            future_events=(fe_din,),
            scenarios=generate_spending_change_scenarios([action_insufficient], max_changes=1),
            candidate_events=(cand_event,),
        )
        self.assertIsNone(best)

    def test_minimal_disruption_preference(self) -> None:
        """1 change is preferred over 2 changes; smaller reduction is preferred over larger."""
        act_small = SpendingChange("ev_sub1", "s_sub1", SpendingActionType.STOP, Decimal("30.00"), Decimal("0"), date(2025, 1, 10), "sub1", Decimal("30.00"), "Stop sub1")
        act_big = SpendingChange("ev_sub2", "s_sub2", SpendingActionType.STOP, Decimal("80.00"), Decimal("0"), date(2025, 1, 10), "sub2", Decimal("80.00"), "Stop sub2")

        scenarios = generate_spending_change_scenarios([act_small, act_big], max_changes=2)
        # Verify ordering: single change with smaller reduction comes first
        self.assertEqual(len(scenarios[0].changes), 1)
        self.assertEqual(scenarios[0].changes[0].event_id, "ev_sub1")
        self.assertEqual(len(scenarios[1].changes), 1)
        self.assertEqual(scenarios[1].changes[0].event_id, "ev_sub2")
        self.assertEqual(len(scenarios[2].changes), 2)


class TestSpendingChangesDeterminismAndAdversarial(unittest.TestCase):
    """Phases 13D & 13H: Determinism and Adversarial Cases."""

    def test_permutation_invariance(self) -> None:
        """Permuting input order produces byte-identical serialized scenario ordering."""
        act1 = SpendingChange("ev_1", "s_1", SpendingActionType.STOP, Decimal("10.00"), Decimal("0"), date(2025, 1, 5), "sub", Decimal("10.00"), "Stop 1")
        act2 = SpendingChange("ev_2", "s_2", SpendingActionType.STOP, Decimal("20.00"), Decimal("0"), date(2025, 1, 6), "sub", Decimal("20.00"), "Stop 2")

        scenarios_a = generate_spending_change_scenarios([act1, act2], max_changes=2)
        scenarios_b = generate_spending_change_scenarios([act2, act1], max_changes=2)

        self.assertEqual(
            [s.action_string for s in scenarios_a],
            [s.action_string for s in scenarios_b],
        )

    def test_same_event_mutual_exclusion(self) -> None:
        """Same event cannot have multiple conflicting actions in a single scenario."""
        act_stop = SpendingChange("ev_1", "s_1", SpendingActionType.STOP, Decimal("50.00"), Decimal("0"), date(2025, 1, 5), "dining", Decimal("50.00"), "Stop 1")
        act_red = SpendingChange("ev_1", "s_1", SpendingActionType.REDUCE_TO, Decimal("50.00"), Decimal("20.00"), date(2025, 1, 5), "dining", Decimal("30.00"), "Reduce 1")

        scenarios = generate_spending_change_scenarios([act_stop, act_red], max_changes=2)
        # Every scenario must have at most 1 change targeting ev_1
        for s in scenarios:
            eids = [c.event_id for c in s.changes]
            self.assertEqual(len(eids), len(set(eids)))

    def test_amount_formatting_integer_vs_fractional(self) -> None:
        """Integer amounts (e.g. IDR) format without decimals; fractional (e.g. USD) format with 2 decimals."""
        self.assertEqual(format_change_amount(Decimal("665950")), "665950")
        self.assertEqual(format_change_amount(Decimal("665950.00")), "665950")
        self.assertEqual(format_change_amount(Decimal("23.50")), "23.50")
        self.assertEqual(format_change_amount(Decimal("23.5")), "23.50")


class TestSpendingChangesCandidateIntegration(unittest.TestCase):
    """Phases 13F & 13G: Candidate Integration, Ranking and Invariant Preservation."""

    def setUp(self) -> None:
        self.user_id = "user_integ"
        self.req_date = date(2025, 1, 1)
        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("600.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=("essential_expenses",),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("gym",),
            payment_methods_user_will_consider=("full_payment", "installments"),
            max_installment_months=12,
        )
        self.request = FinancialRequest(
            request_id="req_integ",
            user_id=self.user_id,
            request_date=self.req_date,
            request_type="electronics",
            requested_amount=Decimal("350.00"),
            desired_completion_date=date(2025, 2, 1),
            allows_partial_payment=False,
            request_text="Buy laptop",
        )
        self.certificate_unsafe = SafeToPayCertificate(
            request_id="req_integ",
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_start=self.req_date,
            simulation_end=self.req_date + timedelta(days=90),
            requested_amount=Decimal("350.00"),
            amount_safe_to_pay=Decimal("150.00"),
            currency="USD",
            safety_floor=Decimal("200.00"),
            baseline_minimum_available_cash=Decimal("500.00"),
            baseline_minimum_cash_date=self.req_date,
            limiting_date=self.req_date,
            available_cash_after_purchase_today=Decimal("150.00"),
            minimum_available_cash_after_purchase=Decimal("150.00"),
            safety_floor_margin=Decimal("-50.00"),
            earliest_date_for_full_payment=None,
            is_full_payment_safe_today=False,
            is_full_payment_safe_later=False,
            reason_if_unsafe="Safety floor breached",
            candidate_search_method="standard",
        )
        self.series = RecurrenceSeries(
            series_id="s_gym",
            user_id=self.user_id,
            direction=Direction.OUTFLOW,
            category="gym",
            event_type="recurring_expense",
            description="Gym",
            frequency=RecurrenceFrequency.MONTHLY,
            interval_days=30,
            day_of_month=5,
            historical_event_ids=("h1",),
            historical_count=3,
            anchor_event_id="event_gym",
            anchor_date=date(2024, 12, 5),
            forecast_amount=Decimal("100.00"),
            currency="USD",
            amount_rule=AmountForecastingRule.EXACT_STABLE,
            flexibility="stoppable",
            minimum_allowed_amount=None,
            is_protected=False,
            is_cancelled=False,
        )
        self.fe = FutureEvent(
            event_id="fe_gym",
            series_id="s_gym",
            anchor_event_id="event_gym",
            user_id=self.user_id,
            direction=Direction.OUTFLOW,
            category="gym",
            effective_date=date(2025, 1, 5),
            amount_home=Decimal("100.00"),
            currency="USD",
            event_type="recurring_expense",
            description="Gym",
            frequency=RecurrenceFrequency.MONTHLY,
            flexibility="stoppable",
            minimum_allowed_amount=None,
            is_protected=False,
        )

    def test_candidate_generation_rescues_unsafe_full_payment(self) -> None:
        """When base full payment is unsafe, a rescued candidate with spending changes is generated."""
        cset = generate_candidates(
            request=self.request,
            profile=self.profile,
            certificate=self.certificate_unsafe,
            payment_option_feasibilities=(),
            payment_options_by_id={},
            canonical_events=(),
            future_events=(self.fe,),
            recurrence_series=(self.series,),
        )
        # Base full payment is present (unsafe)
        base_cand = next(c for c in cset.candidates if not c.spending_changes)
        self.assertFalse(base_cand.is_safe)
        self.assertFalse(base_cand.is_rankable)

        # Rescued full payment is present (safe)
        rescued_cand = next(c for c in cset.candidates if c.spending_changes)
        self.assertTrue(rescued_cand.is_safe)
        self.assertTrue(rescued_cand.is_rankable)
        self.assertEqual(rescued_cand.spending_changes, ("stop:event_gym",))
        self.assertEqual(rescued_cand.provenance.source, "spending_change_optimization")

    def test_criterion_2_prefers_no_spending_changes(self) -> None:
        """When an option is safe WITHOUT spending changes, ranking Criterion 2 strictly prefers it."""
        # Create a certificate where full payment is safe today
        cert_safe = SafeToPayCertificate(
            request_id="req_integ",
            user_id=self.user_id,
            request_date=self.req_date,
            simulation_start=self.req_date,
            simulation_end=self.req_date + timedelta(days=90),
            requested_amount=Decimal("350.00"),
            amount_safe_to_pay=Decimal("350.00"),
            currency="USD",
            safety_floor=Decimal("200.00"),
            baseline_minimum_available_cash=Decimal("600.00"),
            baseline_minimum_cash_date=self.req_date,
            limiting_date=self.req_date,
            available_cash_after_purchase_today=Decimal("250.00"),
            minimum_available_cash_after_purchase=Decimal("250.00"),
            safety_floor_margin=Decimal("50.00"),
            earliest_date_for_full_payment=self.req_date,
            is_full_payment_safe_today=True,
            is_full_payment_safe_later=True,
            reason_if_unsafe=None,
            candidate_search_method="standard",
        )
        cset = generate_candidates(
            request=self.request,
            profile=self.profile,
            certificate=cert_safe,
            payment_option_feasibilities=(),
            payment_options_by_id={},
            canonical_events=(),
            future_events=(self.fe,),
            recurrence_series=(self.series,),
        )
        # Because base full payment is safe, no spending changes candidate was generated for full payment
        self.assertTrue(all(len(c.spending_changes) == 0 for c in cset.rankable_candidates))

        ranked = rank_candidate_set(cset, desired_completion_date=self.request.desired_completion_date)
        self.assertIsNotNone(ranked.top_candidate)
        self.assertEqual(len(ranked.top_candidate.spending_changes), 0)


if __name__ == "__main__":
    unittest.main()
