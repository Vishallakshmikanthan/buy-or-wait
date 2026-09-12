"""Unit tests for recurrence detection and future-event expansion layer.

Covers all 18 requirements specified in challenge prompt:
1. single event does not create recurrence
2. repeated monthly events create monthly recurrence
3. repeated weekly events create weekly recurrence
4. inconsistent intervals do not create false recurrence
5. stable amounts are forecast correctly
6. varying amounts use documented conservative rule
7. recurring salary requires sufficient evidence
8. pending credit does not create recurring income
9. bonus/commission announcement does not create income
10. recurring expense preserves category/flexibility
11. protected expense is not reclassified as flexible
12. explicit future event prevents duplicate forecast
13. cancellation stops future recurrence
14. amendment changes future recurrence only when explicitly supported
15. generated event IDs are deterministic
16. same input produces identical recurrence output
17. date-range expansion respects start/end boundaries
18. no unresolved event becomes a forecast cash event
"""

import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

from code.canonical import (
    CanonicalEvent,
    CanonicalLedger,
    CashImpactType,
    Direction,
    RecurrenceClassification,
)
from code.loaders import FinancialProfile, Message
from code.recurrence import (
    AmountForecastingRule,
    FutureEvent,
    RecurrenceFrequency,
    RecurrenceSeries,
    detect_recurrence_for_user,
    expand_future_events,
)


def _make_event(
    event_id: str,
    user_id: str = "u_test",
    effective_date: date = date(2025, 1, 15),
    direction: Direction = Direction.OUTFLOW,
    amount_home: Decimal = Decimal("100.00"),
    status: str = "settled",
    is_cash: bool = True,
    cash_impact: CashImpactType = CashImpactType.SETTLED_OUTFLOW,
    category: str = "utilities",
    description: str = "Monthly Water Bill",
    flexibility: str = "fixed",
    minimum_allowed_amount_home: Decimal = None,
    is_unresolved: bool = False,
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        user_id=user_id,
        source_row=1,
        effective_date=effective_date,
        direction=direction,
        direction_original="debit" if direction == Direction.OUTFLOW else "credit",
        amount_original=amount_home,
        currency_original="INR",
        amount_home=amount_home if not is_unresolved else None,
        home_currency="INR",
        exchange_rate_used=Decimal("1"),
        exchange_rate_date=effective_date,
        status=status,
        is_cash_event=is_cash,
        cash_impact_type=cash_impact,
        event_type="expense" if direction == Direction.OUTFLOW else "income",
        category=category,
        description=description,
        flexibility=flexibility,
        minimum_allowed_amount_original=minimum_allowed_amount_home,
        minimum_allowed_amount_home=minimum_allowed_amount_home,
        linked_event_id=None,
        recurrence_type=RecurrenceClassification.UNKNOWN,
        is_unresolved=is_unresolved,
        unresolved_reason="image_required" if is_unresolved else None,
        evidence_chain=(),
        applied_actions=(),
    )


class TestRecurrenceDetection(unittest.TestCase):
    """Tests for recurrence detection and future expansion rules."""

    def test_01_single_event_does_not_create_recurrence(self):
        """A single occurrence must not create a recurring future event."""
        e1 = _make_event("e1", effective_date=date(2025, 1, 15))
        ledger = CanonicalLedger(events=[e1])
        series_list, rejected = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 0)
        self.assertEqual(len(rejected), 1)
        self.assertIn("insufficient_observations", rejected[0]["reason"])

    def test_02_repeated_monthly_events_create_monthly_recurrence(self):
        """Repeated monthly events create monthly recurrence."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10)),
            _make_event("e2", effective_date=date(2025, 2, 10)),
            _make_event("e3", effective_date=date(2025, 3, 10)),
            _make_event("e4", effective_date=date(2025, 4, 10)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 1)
        s = series_list[0]
        self.assertEqual(s.frequency, RecurrenceFrequency.MONTHLY)
        self.assertEqual(s.day_of_month, 10)
        self.assertEqual(s.historical_count, 4)

    def test_03_repeated_weekly_events_create_weekly_recurrence(self):
        """Repeated weekly events create weekly recurrence."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 7)),
            _make_event("e2", effective_date=date(2025, 1, 14)),
            _make_event("e3", effective_date=date(2025, 1, 21)),
            _make_event("e4", effective_date=date(2025, 1, 28)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 1)
        s = series_list[0]
        self.assertEqual(s.frequency, RecurrenceFrequency.WEEKLY)
        self.assertEqual(s.interval_days, 7)

    def test_04_inconsistent_intervals_do_not_create_false_recurrence(self):
        """Inconsistent intervals do not create false recurrence."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 2)),
            _make_event("e2", effective_date=date(2025, 1, 15)),
            _make_event("e3", effective_date=date(2025, 3, 22)),
            _make_event("e4", effective_date=date(2025, 4, 5)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, rejected = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 0)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "irregular_intervals")

    def test_05_stable_amounts_are_forecast_correctly(self):
        """Stable amounts are forecast with exact stable amount."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 15), amount_home=Decimal("250.00")),
            _make_event("e2", effective_date=date(2025, 2, 15), amount_home=Decimal("250.00")),
            _make_event("e3", effective_date=date(2025, 3, 15), amount_home=Decimal("250.00")),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 1)
        self.assertEqual(series_list[0].forecast_amount, Decimal("250.00"))
        self.assertEqual(series_list[0].amount_rule, AmountForecastingRule.EXACT_STABLE.value)

    def test_06_varying_amounts_use_conservative_rule(self):
        """Varying amounts use upper median for expenses, lower median for income."""
        # Expense varying
        exp_events = [
            _make_event("e1", effective_date=date(2025, 1, 15), amount_home=Decimal("100.00")),
            _make_event("e2", effective_date=date(2025, 2, 15), amount_home=Decimal("150.00")),
            _make_event("e3", effective_date=date(2025, 3, 15), amount_home=Decimal("120.00")),
            _make_event("e4", effective_date=date(2025, 4, 15), amount_home=Decimal("180.00")),
        ]
        ledger = CanonicalLedger(events=exp_events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")
        # Sorted amounts: 100, 120, 150, 180 -> upper median (index 2) is 150
        self.assertEqual(series_list[0].forecast_amount, Decimal("150.00"))
        self.assertEqual(series_list[0].amount_rule, AmountForecastingRule.CONSERVATIVE_UPPER_MEDIAN_EXPENSE.value)

        # Income varying
        inc_events = [
            _make_event("i1", direction=Direction.INFLOW, effective_date=date(2025, 1, 15), amount_home=Decimal("1000.00"), category="salary", description="Payroll"),
            _make_event("i2", direction=Direction.INFLOW, effective_date=date(2025, 2, 15), amount_home=Decimal("1500.00"), category="salary", description="Payroll"),
            _make_event("i3", direction=Direction.INFLOW, effective_date=date(2025, 3, 15), amount_home=Decimal("1200.00"), category="salary", description="Payroll"),
            _make_event("i4", direction=Direction.INFLOW, effective_date=date(2025, 4, 15), amount_home=Decimal("1800.00"), category="salary", description="Payroll"),
        ]
        ledger_inc = CanonicalLedger(events=inc_events)
        series_inc, _ = detect_recurrence_for_user(ledger_inc, "u_test")
        # Sorted amounts: 1000, 1200, 1500, 1800 -> lower median (index 1) is 1200
        self.assertEqual(series_inc[0].forecast_amount, Decimal("1200.00"))
        self.assertEqual(series_inc[0].amount_rule, AmountForecastingRule.CONSERVATIVE_LOWER_MEDIAN_INCOME.value)

    def test_07_recurring_salary_requires_sufficient_evidence(self):
        """A recurring salary must have at least 3 historical observations."""
        events = [
            _make_event("s1", direction=Direction.INFLOW, effective_date=date(2025, 1, 15), category="salary", description="Base Salary"),
            _make_event("s2", direction=Direction.INFLOW, effective_date=date(2025, 2, 15), category="salary", description="Base Salary"),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, rejected = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 0)
        self.assertEqual(rejected[0]["reason"], "insufficient_observations (< 3)")

    def test_08_pending_credit_does_not_create_recurring_income(self):
        """Pending credits are non-cash and do not create recurring income."""
        events = [
            _make_event("s1", direction=Direction.INFLOW, effective_date=date(2025, 1, 15), category="salary", description="Base Salary"),
            _make_event("s2", direction=Direction.INFLOW, effective_date=date(2025, 2, 15), category="salary", description="Base Salary"),
            _make_event(
                "s3",
                direction=Direction.INFLOW,
                effective_date=date(2025, 3, 15),
                category="salary",
                description="Base Salary",
                status="pending",
                is_cash=False,
                cash_impact=CashImpactType.PENDING_CREDIT_IGNORED,
            ),
        ]
        ledger = CanonicalLedger(events=events)
        # s3 is non-cash, so only s1 and s2 are usable cash events (< 3)
        series_list, rejected = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 0)

    def test_09_bonus_announcement_does_not_create_income(self):
        """Bonus or unconfirmed messages do not invent recurring income."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10)),
            _make_event("e2", effective_date=date(2025, 2, 10)),
            _make_event("e3", effective_date=date(2025, 3, 10)),
        ]
        msg = Message(
            message_id="m1",
            user_id="u_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your quarterly bonus is under review and pending approval.",
        )
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test", messages=[msg])
        # No bonus income series created
        income_series = [s for s in series_list if s.direction == Direction.INFLOW]
        self.assertEqual(len(income_series), 0)

    def test_10_recurring_expense_preserves_category_and_flexibility(self):
        """Recurring expense retains category, flexibility, and minimum_allowed_amount."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10), category="streaming", flexibility="stoppable", minimum_allowed_amount_home=None),
            _make_event("e2", effective_date=date(2025, 2, 10), category="streaming", flexibility="stoppable", minimum_allowed_amount_home=None),
            _make_event("e3", effective_date=date(2025, 3, 10), category="streaming", flexibility="stoppable", minimum_allowed_amount_home=None),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 1)
        s = series_list[0]
        self.assertEqual(s.category, "streaming")
        self.assertEqual(s.flexibility, "stoppable")
        self.assertIsNone(s.minimum_allowed_amount)

        # Expand future events
        res = expand_future_events(series_list, start_date=date(2025, 4, 1), end_date=date(2025, 6, 30))
        for f_evt in res.future_events:
            self.assertEqual(f_evt.category, "streaming")
            self.assertEqual(f_evt.flexibility, "stoppable")

    def test_11_protected_expense_is_not_reclassified_as_flexible(self):
        """Protected category in profile marks the series as protected."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10), category="utilities", flexibility="fixed"),
            _make_event("e2", effective_date=date(2025, 2, 10), category="utilities", flexibility="fixed"),
            _make_event("e3", effective_date=date(2025, 3, 10), category="utilities", flexibility="fixed"),
        ]
        prof = FinancialProfile(
            user_id="u_test",
            home_currency="INR",
            current_available_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("2000"),
            financial_priorities=("emergency_fund",),
            expense_categories_to_protect=("utilities", "rent"),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=3,
        )
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test", profile=prof)
        self.assertEqual(len(series_list), 1)
        self.assertTrue(series_list[0].is_protected)
        self.assertEqual(series_list[0].flexibility, "fixed")

    def test_12_explicit_future_event_prevents_duplicate_forecast(self):
        """Explicit scheduled canonical event suppresses duplicate forecast."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 15), category="rent", description="Rent"),
            _make_event("e2", effective_date=date(2025, 2, 15), category="rent", description="Rent"),
            _make_event("e3", effective_date=date(2025, 3, 15), category="rent", description="Rent"),
            # Explicit future scheduled rent in April
            _make_event("e_future", effective_date=date(2025, 4, 15), status="scheduled", category="rent", description="Scheduled Rent"),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 1)

        # Expand for April and May
        res = expand_future_events(
            series_list,
            start_date=date(2025, 4, 1),
            end_date=date(2025, 5, 31),
            ledger=ledger,
        )
        # April 15 forecast should be suppressed due to e_future
        self.assertEqual(len(res.suppressed_forecasts), 1)
        self.assertEqual(res.suppressed_forecasts[0].conflicting_event_id, "e_future")
        self.assertEqual(res.suppressed_forecasts[0].forecast_date, date(2025, 4, 15))

        # Only May 15 should be in future_events
        self.assertEqual(len(res.future_events), 1)
        self.assertEqual(res.future_events[0].effective_date, date(2025, 5, 15))

    def test_13_cancellation_stops_future_recurrence(self):
        """Explicit cancellation in message terminates recurrence."""
        events = [
            _make_event("s1", direction=Direction.INFLOW, effective_date=date(2025, 1, 15), category="salary", description="Salary"),
            _make_event("s2", direction=Direction.INFLOW, effective_date=date(2025, 2, 15), category="salary", description="Salary"),
            _make_event("s3", direction=Direction.INFLOW, effective_date=date(2025, 3, 15), category="salary", description="Salary"),
        ]
        msg = Message(
            message_id="m_term",
            user_id="u_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2025, 3, 20, tzinfo=timezone.utc),
            source_type="employer",
            message_text="The current seasonal contract has ended. No off-season income confirmed.",
        )
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test", messages=[msg])
        self.assertEqual(len(series_list), 1)
        self.assertTrue(series_list[0].is_cancelled)
        self.assertIn("contract_ended", series_list[0].cancellation_reason)

        # Expanding events should yield zero future events
        res = expand_future_events(series_list, start_date=date(2025, 4, 1), end_date=date(2025, 6, 30))
        self.assertEqual(len(res.future_events), 0)

    def test_14_amendment_changes_future_recurrence_when_supported(self):
        """Employer message amending salary modifies forecast amount."""
        events = [
            _make_event("s1", direction=Direction.INFLOW, effective_date=date(2025, 1, 15), amount_home=Decimal("1000.00"), category="salary", description="Salary"),
            _make_event("s2", direction=Direction.INFLOW, effective_date=date(2025, 2, 15), amount_home=Decimal("1000.00"), category="salary", description="Salary"),
            _make_event("s3", direction=Direction.INFLOW, effective_date=date(2025, 3, 15), amount_home=Decimal("1000.00"), category="salary", description="Salary"),
        ]
        msg = Message(
            message_id="m_raise",
            user_id="u_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2025, 3, 20, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your monthly salary has increased to INR 1250.00 starting next cycle.",
        )
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test", messages=[msg])
        self.assertEqual(len(series_list), 1)
        self.assertEqual(series_list[0].forecast_amount, Decimal("1250.00"))
        self.assertEqual(series_list[0].amount_rule, AmountForecastingRule.EXPLICIT_AMENDMENT.value)

    def test_15_generated_event_ids_are_deterministic(self):
        """Future event IDs are deterministic based on series and date."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10)),
            _make_event("e2", effective_date=date(2025, 2, 10)),
            _make_event("e3", effective_date=date(2025, 3, 10)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")
        res1 = expand_future_events(series_list, start_date=date(2025, 4, 1), end_date=date(2025, 5, 31))
        res2 = expand_future_events(series_list, start_date=date(2025, 4, 1), end_date=date(2025, 5, 31))

        self.assertEqual([e.event_id for e in res1.future_events], [e.event_id for e in res2.future_events])
        self.assertEqual(res1.future_events[0].event_id, "forecast_rec_u_test_utilities_e3_20250410")

    def test_16_same_input_produces_identical_output(self):
        """Running detection twice produces identical series."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10)),
            _make_event("e2", effective_date=date(2025, 2, 10)),
            _make_event("e3", effective_date=date(2025, 3, 10)),
        ]
        ledger = CanonicalLedger(events=events)
        s1, _ = detect_recurrence_for_user(ledger, "u_test")
        s2, _ = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(s1, s2)

    def test_17_date_range_expansion_respects_boundaries(self):
        """Future expansion generates dates strictly within [start_date, end_date]."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 15)),
            _make_event("e2", effective_date=date(2025, 2, 15)),
            _make_event("e3", effective_date=date(2025, 3, 15)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        # Range only covers May
        res = expand_future_events(series_list, start_date=date(2025, 5, 1), end_date=date(2025, 5, 31))
        self.assertEqual(len(res.future_events), 1)
        self.assertEqual(res.future_events[0].effective_date, date(2025, 5, 15))

    def test_18_no_unresolved_event_becomes_forecast_cash_event(self):
        """Unresolved events cannot form or contribute to recurring cash forecasts."""
        unresolved_events = [
            _make_event("u1", effective_date=date(2025, 1, 10), is_unresolved=True),
            _make_event("u2", effective_date=date(2025, 2, 10), is_unresolved=True),
            _make_event("u3", effective_date=date(2025, 3, 10), is_unresolved=True),
        ]
        ledger = CanonicalLedger(events=unresolved_events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 0)


if __name__ == "__main__":
    unittest.main()
