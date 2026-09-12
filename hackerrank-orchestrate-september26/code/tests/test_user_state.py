"""Unit test suite for UserFinancialState layer."""

from datetime import date, timedelta
from decimal import Decimal
import unittest

from code.canonical import CanonicalEvent, CashImpactType, Direction, RecurrenceClassification
from code.models import FinancialProfile, FinancialRequest
from code.recurrence import FutureEvent, RecurrenceFrequency, RecurrenceSeries
from code.simulator import simulate_user
from code.user_state import (
    ExpenseStability,
    IncomeStability,
    SpendingTrend,
    UserFinancialState,
    build_user_financial_state,
    compute_monthly_equivalent,
)


class TestUserFinancialState(unittest.TestCase):
    """Test suite covering deterministic UserFinancialState construction."""

    def setUp(self) -> None:
        self.user_id = "test_user_state"
        self.request_date = date(2025, 4, 1)
        self.home_currency = "USD"

        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency=self.home_currency,
            current_available_balance=Decimal("2000.00"),
            minimum_balance_to_keep=Decimal("500.00"),
            financial_priorities=("savings", "investing"),
            expense_categories_to_protect=("rent", "healthcare"),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("streaming",),
            payment_methods_user_will_consider=("full_payment", "installments"),
            max_installment_months=6,
        )

        self.request = FinancialRequest(
            request_id="req_state_01",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("400.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="Can I afford this laptop?",
        )

    def _make_canonical(
        self,
        event_id: str,
        effective_date: date,
        direction: Direction,
        amount: Decimal,
        category: str = "shopping",
        status: str = "settled",
        impact_type: CashImpactType = CashImpactType.SETTLED_OUTFLOW,
        is_unresolved: bool = False,
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
            is_unresolved=is_unresolved,
            unresolved_reason="test" if is_unresolved else None,
            evidence_chain=(),
            applied_actions=(),
        )

    def _make_future(
        self,
        event_id: str,
        effective_date: date,
        direction: Direction,
        amount: Decimal,
        category: str = "salary",
        series_id: str = "series_1",
    ) -> FutureEvent:
        return FutureEvent(
            event_id=event_id,
            user_id=self.user_id,
            effective_date=effective_date,
            direction=direction,
            amount_home=amount,
            currency=self.home_currency,
            category=category,
            event_type="income" if direction == Direction.INFLOW else "expense",
            description=f"Future {category}",
            series_id=series_id,
            frequency=RecurrenceFrequency.MONTHLY,
            is_forecast=True,
        )

    def test_01_immutability(self) -> None:
        """UserFinancialState must be frozen and immutable."""
        baseline = simulate_user(self.user_id, self.request_date, self.request_date + timedelta(days=90), profile=self.profile)
        state = build_user_financial_state(self.request, self.profile, [], [], baseline, [])
        with self.assertRaises(Exception):
            state.currency = "EUR"  # type: ignore

    def test_02_current_cash_and_pending_reservations(self) -> None:
        """Pending debits reserve cash and reduce available cash without double counting."""
        # Initial balance: 2000. Pending debit of 300 on request date.
        pend = self._make_canonical(
            "p1", self.request_date, Direction.OUTFLOW, Decimal("300.00"),
            status="pending", impact_type=CashImpactType.PENDING_DEBIT_RESERVED,
        )
        baseline = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=[pend], profile=self.profile,
        )
        state = build_user_financial_state(self.request, self.profile, [pend], [], baseline, [])

        self.assertEqual(state.current_available_balance, Decimal("2000.00"))
        self.assertEqual(state.pending_reserved_amount, Decimal("300.00"))
        self.assertEqual(state.available_cash, Decimal("1700.00"))
        self.assertEqual(state.safety_floor, Decimal("500.00"))
        self.assertEqual(state.current_headroom_above_floor, Decimal("1200.00"))

    def test_03_no_lookahead_leakage(self) -> None:
        """Events occurring on or after request_date must NOT enter historical statistics."""
        # Event on request date (April 1)
        same_day = self._make_canonical("e_today", date(2025, 4, 1), Direction.OUTFLOW, Decimal("250.00"))
        # Event after request date (April 5)
        future_ev = self._make_canonical("e_future", date(2025, 4, 5), Direction.OUTFLOW, Decimal("100.00"))
        # Event before request date (March 15)
        past_ev = self._make_canonical("e_past", date(2025, 3, 15), Direction.OUTFLOW, Decimal("50.00"))

        baseline = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=[same_day, future_ev, past_ev], profile=self.profile,
        )
        state = build_user_financial_state(self.request, self.profile, [same_day, future_ev, past_ev], [], baseline, [])

        # Only e_past must appear in historical stats
        self.assertEqual(state.total_historical_events_count, 1)
        self.assertEqual(state.recent_30d_outflow, Decimal("50.00"))
        self.assertEqual(state.recent_90d_outflow, Decimal("50.00"))

    def test_04_historical_windows_partitioning(self) -> None:
        """Verify strict calendar window boundaries: 30d, prev 30d, 90d."""
        # Request date is 2025-04-01.
        # Window 30d: [2025-03-02, 2025-04-01)
        # Prev 30d:   [2025-01-31, 2025-03-02)
        # Window 90d: [2025-01-01, 2025-04-01)
        e_30d = self._make_canonical("e1", date(2025, 3, 15), Direction.OUTFLOW, Decimal("100.00"))
        e_prev = self._make_canonical("e2", date(2025, 2, 15), Direction.OUTFLOW, Decimal("200.00"))
        e_older_90d = self._make_canonical("e3", date(2025, 1, 10), Direction.OUTFLOW, Decimal("300.00"))
        e_outside_90d = self._make_canonical("e4", date(2024, 12, 15), Direction.OUTFLOW, Decimal("400.00"))

        events = [e_30d, e_prev, e_older_90d, e_outside_90d]
        baseline = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=events, profile=self.profile,
        )
        state = build_user_financial_state(self.request, self.profile, events, [], baseline, [])

        self.assertEqual(state.recent_30d_outflow, Decimal("100.00"))
        # 90d outflow includes e1 + e2 + e3 = 100 + 200 + 300 = 600
        self.assertEqual(state.recent_90d_outflow, Decimal("600.00"))
        self.assertEqual(state.monthly_average_outflow_90d, Decimal("200.00"))
        self.assertEqual(state.total_historical_events_count, 4)

    def test_05_spending_trend_calculation(self) -> None:
        """Verify deterministic spending trend detection."""
        # Case A: Outflow increased by > 15% (recent 30d = 300, prev 30d = 200 => +50%)
        e_recent = self._make_canonical("e1", date(2025, 3, 15), Direction.OUTFLOW, Decimal("300.00"))
        e_prev = self._make_canonical("e2", date(2025, 2, 15), Direction.OUTFLOW, Decimal("200.00"))
        e_anchor = self._make_canonical("e3", date(2025, 1, 1), Direction.OUTFLOW, Decimal("50.00"))  # ensure 90d history

        baseline = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=[e_recent, e_prev, e_anchor], profile=self.profile,
        )
        state_inc = build_user_financial_state(self.request, self.profile, [e_recent, e_prev, e_anchor], [], baseline, [])
        self.assertEqual(state_inc.spending_trend, SpendingTrend.INCREASING)
        self.assertEqual(state_inc.spending_trend_percentage, Decimal("50.00"))

        # Case B: Outflow decreased by > 15% (recent 30d = 100, prev 30d = 200 => -50%)
        e_recent_low = self._make_canonical("e1", date(2025, 3, 15), Direction.OUTFLOW, Decimal("100.00"))
        baseline_dec = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=[e_recent_low, e_prev, e_anchor], profile=self.profile,
        )
        state_dec = build_user_financial_state(self.request, self.profile, [e_recent_low, e_prev, e_anchor], [], baseline_dec, [])
        self.assertEqual(state_dec.spending_trend, SpendingTrend.DECREASING)
        self.assertEqual(state_dec.spending_trend_percentage, Decimal("-50.00"))

        # Case C: Stable within 15% (recent 30d = 210, prev 30d = 200 => +5%)
        e_recent_stable = self._make_canonical("e1", date(2025, 3, 15), Direction.OUTFLOW, Decimal("210.00"))
        baseline_stable = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=[e_recent_stable, e_prev, e_anchor], profile=self.profile,
        )
        state_stable = build_user_financial_state(self.request, self.profile, [e_recent_stable, e_prev, e_anchor], [], baseline_stable, [])
        self.assertEqual(state_stable.spending_trend, SpendingTrend.STABLE)
        self.assertEqual(state_stable.spending_trend_percentage, Decimal("5.00"))

    def test_06_income_stability_high(self) -> None:
        """User with validated recurring income with >= 3 events has HIGH income stability."""
        series = RecurrenceSeries(
            series_id="s_inc", user_id=self.user_id, direction=Direction.INFLOW, category="salary",
            event_type="income", description="Monthly Salary", frequency=RecurrenceFrequency.MONTHLY,
            interval_days=30, day_of_month=15, historical_event_ids=("e1", "e2", "e3"), historical_count=3,
            anchor_event_id="e3", anchor_date=date(2025, 3, 15), forecast_amount=Decimal("3000.00"),
            currency="USD", amount_rule="exact_stable", flexibility="fixed", minimum_allowed_amount=None,
        )
        e_anchor = self._make_canonical("e0", date(2025, 1, 1), Direction.OUTFLOW, Decimal("50.00"))
        baseline = simulate_user(self.user_id, self.request_date, self.request_date + timedelta(days=90), canonical_events=[e_anchor], profile=self.profile)
        state = build_user_financial_state(self.request, self.profile, [e_anchor], [], baseline, [series])

        self.assertEqual(state.income_stability, IncomeStability.HIGH)
        self.assertTrue(state.has_validated_recurring_income)
        self.assertEqual(state.total_monthly_recurring_income, Decimal("3000.00"))

    def test_07_expense_breakdown_essential_vs_discretionary(self) -> None:
        """Verify categorization into essential vs discretionary outflows."""
        # Groceries and rent are essential; shopping is discretionary
        e_rent = self._make_canonical("e1", date(2025, 3, 10), Direction.OUTFLOW, Decimal("800.00"), category="rent")
        e_groc = self._make_canonical("e2", date(2025, 3, 20), Direction.OUTFLOW, Decimal("200.00"), category="groceries")
        e_shop = self._make_canonical("e3", date(2025, 3, 25), Direction.OUTFLOW, Decimal("150.00"), category="shopping")

        events = [e_rent, e_groc, e_shop]
        baseline = simulate_user(self.user_id, self.request_date, self.request_date + timedelta(days=90), canonical_events=events, profile=self.profile)
        state = build_user_financial_state(self.request, self.profile, events, [], baseline, [])

        self.assertEqual(state.recent_30d_outflow, Decimal("1150.00"))
        self.assertEqual(state.recent_30d_essential_outflow, Decimal("1000.00"))
        self.assertEqual(state.recent_30d_discretionary_outflow, Decimal("150.00"))

    def test_08_upcoming_obligations_aggregation(self) -> None:
        """Upcoming obligations within 90 days are aggregated without lookahead leakage into history."""
        fut1 = self._make_future("fut1", date(2025, 4, 15), Direction.OUTFLOW, Decimal("500.00"), category="rent")
        fut2 = self._make_future("fut2", date(2025, 5, 15), Direction.OUTFLOW, Decimal("500.00"), category="rent")
        sched_canonical = self._make_canonical(
            "sched1", date(2025, 4, 20), Direction.OUTFLOW, Decimal("150.00"),
            category="bills", status="scheduled", impact_type=CashImpactType.SCHEDULED_OUTFLOW,
        )

        future_events = [fut1, fut2]
        canonical_events = [sched_canonical]

        baseline = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=canonical_events, future_events=future_events, profile=self.profile,
        )
        state = build_user_financial_state(self.request, self.profile, canonical_events, future_events, baseline, [])

        self.assertEqual(state.upcoming_obligations_count, 3)
        # 30d limit is 2025-05-01 => fut1 (April 15) and sched1 (April 20) are in 30d
        self.assertEqual(state.upcoming_obligations_30d_total, Decimal("650.00"))
        # 90d includes all three = 500 + 500 + 150 = 1150
        self.assertEqual(state.upcoming_obligations_90d_total, Decimal("1150.00"))
        # Verify history has 0 events
        self.assertEqual(state.total_historical_events_count, 0)

    def test_09_state_consistency_invariants(self) -> None:
        """State fields must maintain exact mathematical consistency with baseline simulation."""
        sal = self._make_canonical("sal", date(2025, 3, 15), Direction.INFLOW, Decimal("2500.00"), category="salary", impact_type=CashImpactType.SETTLED_INFLOW)
        bill = self._make_canonical("bill", date(2025, 3, 20), Direction.OUTFLOW, Decimal("600.00"), category="bills")

        baseline = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=[sal, bill], profile=self.profile,
        )
        state = build_user_financial_state(self.request, self.profile, [sal, bill], [], baseline, [])

        self.assertEqual(state.currency, self.home_currency)
        self.assertEqual(state.available_cash, state.current_available_balance - state.pending_reserved_amount)
        self.assertEqual(state.baseline_minimum_available_cash, baseline.minimum_projected_available_cash)
        self.assertEqual(state.baseline_has_safety_breach, baseline.is_safety_floor_breached)
        self.assertEqual(state.baseline_safety_margin, state.baseline_minimum_available_cash - state.safety_floor)
        self.assertEqual(state.monthly_average_income_90d, (state.recent_90d_income / Decimal("3")).quantize(Decimal("0.01")))
        self.assertEqual(state.monthly_average_outflow_90d, (state.recent_90d_outflow / Decimal("3")).quantize(Decimal("0.01")))

    def test_10_monthly_equivalent_frequency_conversion(self) -> None:
        """compute_monthly_equivalent converts all standard frequencies accurately."""
        self.assertEqual(compute_monthly_equivalent(Decimal("100.00"), RecurrenceFrequency.MONTHLY), Decimal("100.00"))
        self.assertEqual(compute_monthly_equivalent(Decimal("70.00"), RecurrenceFrequency.WEEKLY), Decimal("303.33"))
        self.assertEqual(compute_monthly_equivalent(Decimal("140.00"), RecurrenceFrequency.BIWEEKLY), Decimal("303.33"))
        self.assertEqual(compute_monthly_equivalent(Decimal("210.00"), RecurrenceFrequency.TRIWEEKLY), Decimal("303.33"))

    def test_11_no_lookahead_contamination_future_income_and_expenses(self) -> None:
        """Future income and expenses in the projection window must not contaminate historical metrics."""
        past_inc = self._make_canonical("inc_past", date(2025, 3, 10), Direction.INFLOW, Decimal("1000.00"), category="salary", impact_type=CashImpactType.SETTLED_INFLOW)
        past_exp = self._make_canonical("exp_past", date(2025, 3, 12), Direction.OUTFLOW, Decimal("200.00"), category="rent")

        # Future events in projection window
        fut_inc = self._make_future("inc_fut", date(2025, 4, 15), Direction.INFLOW, Decimal("5000.00"), category="salary")
        fut_exp = self._make_future("exp_fut", date(2025, 4, 20), Direction.OUTFLOW, Decimal("3000.00"), category="rent")

        baseline = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=[past_inc, past_exp], future_events=[fut_inc, fut_exp], profile=self.profile,
        )
        state = build_user_financial_state(self.request, self.profile, [past_inc, past_exp], [fut_inc, fut_exp], baseline, [])

        # Historical metrics must match ONLY past_inc and past_exp
        self.assertEqual(state.recent_30d_income, Decimal("1000.00"))
        self.assertEqual(state.recent_30d_outflow, Decimal("200.00"))
        self.assertEqual(state.recent_90d_income, Decimal("1000.00"))
        self.assertEqual(state.recent_90d_outflow, Decimal("200.00"))
        self.assertEqual(state.historical_income_count, 1)

        # Projected metrics must capture the future events
        self.assertEqual(state.total_projected_inflows_90d, Decimal("5000.00"))
        self.assertEqual(state.total_projected_outflows_90d, Decimal("3000.00"))

    def test_12_no_double_counting_pending_and_settled(self) -> None:
        """Pending debits reserve cash, while historical settled debits count as historical outflows."""
        # A settled debit from 10 days ago (March 22)
        settled = self._make_canonical("set1", date(2025, 3, 22), Direction.OUTFLOW, Decimal("150.00"), status="settled", impact_type=CashImpactType.SETTLED_OUTFLOW)
        # An active pending debit on request date (April 1)
        pending = self._make_canonical("pend1", date(2025, 4, 1), Direction.OUTFLOW, Decimal("250.00"), status="pending", impact_type=CashImpactType.PENDING_DEBIT_RESERVED)

        baseline = simulate_user(
            self.user_id, self.request_date, self.request_date + timedelta(days=90),
            canonical_events=[settled, pending], profile=self.profile,
        )
        state = build_user_financial_state(self.request, self.profile, [settled, pending], [], baseline, [])

        # Only settled is in historical outflow
        self.assertEqual(state.recent_30d_outflow, Decimal("150.00"))
        # Pending is in pending_reserved_amount
        self.assertEqual(state.pending_reserved_amount, Decimal("250.00"))
        self.assertEqual(state.available_cash, Decimal("1750.00"))

    def test_13_cancelled_and_failed_events_excluded(self) -> None:
        """Cancelled and failed historical transactions must never count as realized spending or income."""
        failed = self._make_canonical("f1", date(2025, 3, 15), Direction.OUTFLOW, Decimal("500.00"), status="failed", impact_type=CashImpactType.FAILED_IGNORED)
        cancelled = self._make_canonical("c1", date(2025, 3, 16), Direction.INFLOW, Decimal("1000.00"), status="cancelled", impact_type=CashImpactType.CANCELLED_IGNORED)

        baseline = simulate_user(self.user_id, self.request_date, self.request_date + timedelta(days=90), canonical_events=[failed, cancelled], profile=self.profile)
        state = build_user_financial_state(self.request, self.profile, [failed, cancelled], [], baseline, [])

        self.assertEqual(state.recent_30d_income, Decimal("0.00"))
        self.assertEqual(state.recent_30d_outflow, Decimal("0.00"))

    def test_14_unresolved_events_telemetry(self) -> None:
        """Unresolved events correctly set telemetry flags without fabricating cash amounts."""
        unres = self._make_canonical("unres1", date(2025, 3, 10), Direction.OUTFLOW, Decimal("0.00"), is_unresolved=True)
        baseline = simulate_user(self.user_id, self.request_date, self.request_date + timedelta(days=90), canonical_events=[unres], profile=self.profile)
        state = build_user_financial_state(self.request, self.profile, [unres], [], baseline, [])

        self.assertTrue(state.has_unresolved_events)
        self.assertEqual(state.unresolved_events_count, 1)

    def test_15_state_determinism_and_reproducibility(self) -> None:
        """Running build_user_financial_state twice on identical inputs must yield identical states."""
        e1 = self._make_canonical("e1", date(2025, 3, 10), Direction.OUTFLOW, Decimal("300.00"))
        baseline = simulate_user(self.user_id, self.request_date, self.request_date + timedelta(days=90), canonical_events=[e1], profile=self.profile)
        s1 = build_user_financial_state(self.request, self.profile, [e1], [], baseline, [])
        s2 = build_user_financial_state(self.request, self.profile, [e1], [], baseline, [])

        self.assertEqual(s1, s2)

    def test_16_expense_stability_classifications(self) -> None:
        """Test HIGH, MODERATE, LOW expense stability thresholds."""
        # Baseline monthly outflow: 1000.00 (90d outflow = 3000.00)
        e_hist = [
            self._make_canonical(f"e_{i}", date(2025, 1, 1) + timedelta(days=i * 10), Direction.OUTFLOW, Decimal("333.33"))
            for i in range(9)
        ]
        baseline = simulate_user(self.user_id, self.request_date, self.request_date + timedelta(days=90), canonical_events=e_hist, profile=self.profile)

        # High stability: recurring expense >= 50% (e.g. 600.00 / 1000.00 = 60%)
        rec_high = RecurrenceSeries(
            series_id="s_rec_high", user_id=self.user_id, direction=Direction.OUTFLOW, category="rent",
            event_type="expense", description="Rent", frequency=RecurrenceFrequency.MONTHLY,
            interval_days=30, day_of_month=1, historical_event_ids=(), historical_count=3,
            anchor_event_id="e1", anchor_date=date(2025, 3, 1), forecast_amount=Decimal("600.00"),
            currency="USD", amount_rule="exact_stable", flexibility="fixed", minimum_allowed_amount=None,
        )
        s_high = build_user_financial_state(self.request, self.profile, e_hist, [], baseline, [rec_high])
        self.assertEqual(s_high.expense_stability, ExpenseStability.HIGH)

        # Moderate stability: recurring expense between 20% and 50% (e.g. 300.00 / 1000.00 = 30%)
        rec_mod = RecurrenceSeries(
            series_id="s_rec_mod", user_id=self.user_id, direction=Direction.OUTFLOW, category="rent",
            event_type="expense", description="Rent", frequency=RecurrenceFrequency.MONTHLY,
            interval_days=30, day_of_month=1, historical_event_ids=(), historical_count=3,
            anchor_event_id="e1", anchor_date=date(2025, 3, 1), forecast_amount=Decimal("300.00"),
            currency="USD", amount_rule="exact_stable", flexibility="fixed", minimum_allowed_amount=None,
        )
        s_mod = build_user_financial_state(self.request, self.profile, e_hist, [], baseline, [rec_mod])
        self.assertEqual(s_mod.expense_stability, ExpenseStability.MODERATE)

        # Low stability: recurring expense < 20% (e.g. 100.00 / 1000.00 = 10%)
        rec_low = RecurrenceSeries(
            series_id="s_rec_low", user_id=self.user_id, direction=Direction.OUTFLOW, category="rent",
            event_type="expense", description="Rent", frequency=RecurrenceFrequency.MONTHLY,
            interval_days=30, day_of_month=1, historical_event_ids=(), historical_count=3,
            anchor_event_id="e1", anchor_date=date(2025, 3, 1), forecast_amount=Decimal("100.00"),
            currency="USD", amount_rule="exact_stable", flexibility="fixed", minimum_allowed_amount=None,
        )
        s_low = build_user_financial_state(self.request, self.profile, e_hist, [], baseline, [rec_low])
        self.assertEqual(s_low.expense_stability, ExpenseStability.LOW)

    def test_17_all_monetary_fields_are_decimal(self) -> None:
        """Verify no float contamination anywhere in UserFinancialState monetary fields."""
        e1 = self._make_canonical("e1", date(2025, 3, 10), Direction.OUTFLOW, Decimal("250.00"))
        baseline = simulate_user(self.user_id, self.request_date, self.request_date + timedelta(days=90), canonical_events=[e1], profile=self.profile)
        state = build_user_financial_state(self.request, self.profile, [e1], [], baseline, [])

        monetary_fields = [
            "requested_amount", "current_available_balance", "safety_floor", "minimum_balance_to_keep",
            "pending_reserved_amount", "available_cash", "current_headroom_above_floor",
            "recent_30d_income", "recent_30d_outflow", "recent_30d_essential_outflow",
            "recent_30d_discretionary_outflow", "recent_30d_net_cashflow",
            "recent_90d_income", "recent_90d_outflow", "recent_90d_essential_outflow",
            "recent_90d_discretionary_outflow", "recent_90d_net_cashflow",
            "monthly_average_income_90d", "monthly_average_outflow_90d",
            "total_monthly_recurring_income", "total_monthly_recurring_expenses",
            "net_monthly_recurring_cashflow", "baseline_minimum_available_cash",
            "baseline_ending_available_cash", "baseline_safety_margin", "baseline_headroom",
            "total_projected_inflows_90d", "total_projected_outflows_90d",
            "upcoming_obligations_30d_total", "upcoming_obligations_90d_total",
        ]
        for field_name in monetary_fields:
            val = getattr(state, field_name)
            self.assertIsInstance(val, Decimal, f"Field {field_name} must be Decimal, got {type(val)}")


if __name__ == "__main__":
    unittest.main()

