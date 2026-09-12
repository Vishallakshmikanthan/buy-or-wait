"""Hardened unit test suite for recurrence detection and future-event expansion layer.

Covers all 18 specific requirements from Section 11:
1. valid consecutive monthly recurrence
2. skipped-month same-day false positive rejected
3. month-end calendar recurrence
4. irregular monthly rejected
5. genuine explicit future duplicate suppressed
6. unrelated same-category future event does not suppress
7. pending event does not incorrectly suppress
8. deterministic duplicate matching
9. varying description handling
10. unrelated message cannot cancel recurrence
11. unrelated message cannot amend recurrence
12. cancellation affects future only
13. amendment affects future only
14. no future events outside bounds
15. no duplicate series/date occurrence
16. unresolved event cannot establish recurrence
17. pending credit cannot establish income
18. speculative income cannot establish recurrence
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


class TestHardenedRecurrence(unittest.TestCase):
    """Hardened tests for recurrence detection and future expansion."""

    def test_01_valid_consecutive_monthly_recurrence(self):
        """Jan 15, Feb 15, Mar 15 produces monthly recurrence."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 15)),
            _make_event("e2", effective_date=date(2025, 2, 15)),
            _make_event("e3", effective_date=date(2025, 3, 15)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 1)
        s = series_list[0]
        self.assertEqual(s.frequency, RecurrenceFrequency.MONTHLY)
        self.assertEqual(s.day_of_month, 15)
        self.assertEqual(s.historical_count, 3)

    def test_02_skipped_month_same_day_false_positive_rejected(self):
        """Jan 15, Mar 15, May 15 must NOT become monthly recurrence."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 15)),
            _make_event("e2", effective_date=date(2025, 3, 15)),
            _make_event("e3", effective_date=date(2025, 5, 15)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, rejected = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 0)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "irregular_intervals")

    def test_03_month_end_calendar_recurrence(self):
        """Jan 31, Feb 28, Mar 31 produces valid monthly recurrence."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 31)),
            _make_event("e2", effective_date=date(2025, 2, 28)),
            _make_event("e3", effective_date=date(2025, 3, 31)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 1)
        self.assertEqual(series_list[0].frequency, RecurrenceFrequency.MONTHLY)
        self.assertEqual(series_list[0].day_of_month, 31)

    def test_04_irregular_monthly_rejected(self):
        """Jan 30, Feb 28, Apr 30 (skipped March) must be rejected."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 30)),
            _make_event("e2", effective_date=date(2025, 2, 28)),
            _make_event("e3", effective_date=date(2025, 4, 30)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, rejected = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 0)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "irregular_intervals")

    def test_05_genuine_explicit_future_duplicate_suppressed(self):
        """Genuine explicit scheduled future event suppresses duplicate forecast."""
        events = [
            _make_event("s1", direction=Direction.INFLOW, effective_date=date(2025, 1, 15), amount_home=Decimal("50000"), category="salary", description="Payroll credit"),
            _make_event("s2", direction=Direction.INFLOW, effective_date=date(2025, 2, 15), amount_home=Decimal("50000"), category="salary", description="Payroll credit"),
            _make_event("s3", direction=Direction.INFLOW, effective_date=date(2025, 3, 15), amount_home=Decimal("50000"), category="salary", description="Payroll credit"),
            # Scheduled next confirmed salary in April
            _make_event("s_future", direction=Direction.INFLOW, effective_date=date(2025, 4, 15), amount_home=Decimal("50000"), status="scheduled", category="salary", description="Next confirmed salary"),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 1)

        res = expand_future_events(series_list, start_date=date(2025, 4, 1), end_date=date(2025, 5, 31), ledger=ledger)
        # April 15 forecast suppressed
        self.assertEqual(len(res.suppressed_forecasts), 1)
        self.assertEqual(res.suppressed_forecasts[0].conflicting_event_id, "s_future")
        # May 15 forecast emitted
        self.assertEqual(len(res.future_events), 1)
        self.assertEqual(res.future_events[0].effective_date, date(2025, 5, 15))

    def test_06_unrelated_same_category_future_event_does_not_suppress(self):
        """Unrelated explicit event in same category must NOT suppress recurring forecast."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10), amount_home=Decimal("999"), category="entertainment", description="Netflix subscription"),
            _make_event("e2", effective_date=date(2025, 2, 10), amount_home=Decimal("999"), category="entertainment", description="Netflix subscription"),
            _make_event("e3", effective_date=date(2025, 3, 10), amount_home=Decimal("999"), category="entertainment", description="Netflix subscription"),
            # Unrelated event in same category
            _make_event("e_dinner", effective_date=date(2025, 4, 10), amount_home=Decimal("2500"), status="scheduled", category="entertainment", description="Restaurant dinner"),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 1)

        res = expand_future_events(series_list, start_date=date(2025, 4, 1), end_date=date(2025, 4, 30), ledger=ledger)
        # Netflix forecast must NOT be suppressed by Restaurant dinner
        self.assertEqual(len(res.suppressed_forecasts), 0)
        self.assertEqual(len(res.future_events), 1)
        self.assertEqual(res.future_events[0].effective_date, date(2025, 4, 10))
        self.assertIn("Netflix subscription", res.future_events[0].description)

    def test_07_pending_event_does_not_incorrectly_suppress(self):
        """Generic pending card charge must NOT suppress a recurring forecast."""
        events = [
            _make_event("h1", effective_date=date(2025, 1, 11), amount_home=Decimal("4500"), category="healthcare", description="Therapy appointment"),
            _make_event("h2", effective_date=date(2025, 2, 11), amount_home=Decimal("4500"), category="healthcare", description="Therapy appointment"),
            _make_event("h3", effective_date=date(2025, 3, 11), amount_home=Decimal("4500"), category="healthcare", description="Therapy appointment"),
            # Pending pharmacy card charge on April 11
            _make_event("h_pend", effective_date=date(2025, 4, 11), amount_home=Decimal("2800"), status="pending", is_cash=True, cash_impact=CashImpactType.PENDING_DEBIT_RESERVED, category="healthcare", description="Pending pharmacy card charge"),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 1)

        res = expand_future_events(series_list, start_date=date(2025, 4, 1), end_date=date(2025, 4, 30), ledger=ledger)
        # Must not be suppressed by generic pending debit
        self.assertEqual(len(res.suppressed_forecasts), 0)
        self.assertEqual(len(res.future_events), 1)
        self.assertEqual(res.future_events[0].effective_date, date(2025, 4, 11))

    def test_08_deterministic_duplicate_matching(self):
        """Repeated expansion produces identical suppression results."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 15), category="rent", description="Rent"),
            _make_event("e2", effective_date=date(2025, 2, 15), category="rent", description="Rent"),
            _make_event("e3", effective_date=date(2025, 3, 15), category="rent", description="Rent"),
            _make_event("e_sch", effective_date=date(2025, 4, 15), status="scheduled", category="rent", description="Scheduled rent payment"),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        res1 = expand_future_events(series_list, date(2025, 4, 1), date(2025, 5, 31), ledger=ledger)
        res2 = expand_future_events(series_list, date(2025, 4, 1), date(2025, 5, 31), ledger=ledger)

        self.assertEqual([sf.conflicting_event_id for sf in res1.suppressed_forecasts], [sf.conflicting_event_id for sf in res2.suppressed_forecasts])
        self.assertEqual([f.event_id for f in res1.future_events], [f.event_id for f in res2.future_events])

    def test_09_varying_description_handling(self):
        """Reducible dining with varying descriptions is unified by category/flexibility/min_amount."""
        events = [
            _make_event("d1", effective_date=date(2025, 1, 7), category="dining", description="Coffee shop", flexibility="reducible", minimum_allowed_amount_home=Decimal("500")),
            _make_event("d2", effective_date=date(2025, 1, 28), category="dining", description="Weekend food delivery", flexibility="reducible", minimum_allowed_amount_home=Decimal("500")),
            _make_event("d3", effective_date=date(2025, 2, 18), category="dining", description="Bakery and snacks", flexibility="reducible", minimum_allowed_amount_home=Decimal("500")),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        self.assertEqual(len(series_list), 1)
        self.assertEqual(series_list[0].frequency, RecurrenceFrequency.TRIWEEKLY)
        self.assertEqual(series_list[0].flexibility, "reducible")
        self.assertEqual(series_list[0].minimum_allowed_amount, Decimal("500"))

    def test_10_unrelated_message_cannot_cancel_recurrence(self):
        """Message about salary termination cannot cancel a rent series."""
        events = [
            _make_event("r1", effective_date=date(2025, 1, 5), category="rent", description="Rent"),
            _make_event("r2", effective_date=date(2025, 2, 5), category="rent", description="Rent"),
            _make_event("r3", effective_date=date(2025, 3, 5), category="rent", description="Rent"),
        ]
        msg = Message(
            message_id="m1",
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
        self.assertFalse(series_list[0].is_cancelled)

    def test_11_unrelated_message_cannot_amend_recurrence(self):
        """Message about rent increase cannot amend salary."""
        events = [
            _make_event("s1", direction=Direction.INFLOW, effective_date=date(2025, 1, 15), amount_home=Decimal("5000"), category="salary", description="Salary"),
            _make_event("s2", direction=Direction.INFLOW, effective_date=date(2025, 2, 15), amount_home=Decimal("5000"), category="salary", description="Salary"),
            _make_event("s3", direction=Direction.INFLOW, effective_date=date(2025, 3, 15), amount_home=Decimal("5000"), category="salary", description="Salary"),
        ]
        msg = Message(
            message_id="m_rent",
            user_id="u_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2025, 3, 20, tzinfo=timezone.utc),
            source_type="service_provider",
            message_text="StayLedger notice: The renewed lease increases monthly rent by 12%.",
        )
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test", messages=[msg])

        self.assertEqual(len(series_list), 1)
        self.assertEqual(series_list[0].forecast_amount, Decimal("5000"))
        self.assertIsNone(series_list[0].amendment_reason)

    def test_12_cancellation_affects_future_only(self):
        """Cancellation terminates future expansion without altering past canonical events."""
        past_events = [
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
        ledger = CanonicalLedger(events=past_events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test", messages=[msg])

        self.assertTrue(series_list[0].is_cancelled)
        # Past events remain settled cash in ledger
        self.assertEqual(len(ledger.get_cash_events()), 3)
        self.assertEqual(ledger.events[0].amount_home, Decimal("100.00"))

        # Future expansion yields zero future events
        res = expand_future_events(series_list, date(2025, 4, 1), date(2025, 6, 30))
        self.assertEqual(len(res.future_events), 0)

    def test_13_amendment_affects_future_only(self):
        """Employer raise modifies future forecast amount while past events retain historical amounts."""
        past_events = [
            _make_event("s1", direction=Direction.INFLOW, effective_date=date(2025, 1, 15), amount_home=Decimal("1000.00"), category="salary", description="Salary"),
            _make_event("s2", direction=Direction.INFLOW, effective_date=date(2025, 2, 15), amount_home=Decimal("1000.00"), category="salary", description="Salary"),
            _make_event("s3", direction=Direction.INFLOW, effective_date=date(2025, 3, 15), amount_home=Decimal("1000.00"), category="salary", description="Salary"),
        ]
        msg = Message(
            message_id="m_up",
            user_id="u_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2025, 3, 20, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your monthly salary has increased to INR 1500.00 starting next cycle.",
        )
        ledger = CanonicalLedger(events=past_events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test", messages=[msg])

        self.assertEqual(series_list[0].forecast_amount, Decimal("1500.00"))
        # Historical events in ledger remain 1000.00
        for e in ledger.events:
            self.assertEqual(e.amount_home, Decimal("1000.00"))

        res = expand_future_events(series_list, date(2025, 4, 1), date(2025, 4, 30))
        self.assertEqual(res.future_events[0].amount_home, Decimal("1500.00"))

    def test_14_no_future_events_outside_bounds(self):
        """Future expansion generates occurrences strictly within [start_date, end_date]."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10)),
            _make_event("e2", effective_date=date(2025, 2, 10)),
            _make_event("e3", effective_date=date(2025, 3, 10)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        # Range strictly inside April 1..April 30
        res = expand_future_events(series_list, start_date=date(2025, 4, 1), end_date=date(2025, 4, 30))
        self.assertEqual(len(res.future_events), 1)
        self.assertEqual(res.future_events[0].effective_date, date(2025, 4, 10))

    def test_15_no_duplicate_series_date_occurrence(self):
        """A series generates at most one forecast per scheduled occurrence date."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10)),
            _make_event("e2", effective_date=date(2025, 2, 10)),
            _make_event("e3", effective_date=date(2025, 3, 10)),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")

        res = expand_future_events(series_list, start_date=date(2025, 4, 1), end_date=date(2025, 6, 30))
        dates = [f.effective_date for f in res.future_events]
        self.assertEqual(len(dates), len(set(dates)))

    def test_16_unresolved_event_cannot_establish_recurrence(self):
        """Unresolved events cannot form or contribute to recurring forecasts."""
        unresolved_events = [
            _make_event("u1", effective_date=date(2025, 1, 10), is_unresolved=True),
            _make_event("u2", effective_date=date(2025, 2, 10), is_unresolved=True),
            _make_event("u3", effective_date=date(2025, 3, 10), is_unresolved=True),
        ]
        ledger = CanonicalLedger(events=unresolved_events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 0)

    def test_17_pending_credit_cannot_establish_income(self):
        """Pending credits are non-cash and cannot establish recurring salary."""
        events = [
            _make_event("s1", direction=Direction.INFLOW, effective_date=date(2025, 1, 15), category="salary", description="Salary"),
            _make_event("s2", direction=Direction.INFLOW, effective_date=date(2025, 2, 15), category="salary", description="Salary"),
            _make_event("s3", direction=Direction.INFLOW, effective_date=date(2025, 3, 15), status="pending", is_cash=False, cash_impact=CashImpactType.PENDING_CREDIT_IGNORED, category="salary", description="Salary"),
        ]
        ledger = CanonicalLedger(events=events)
        series_list, rejected = detect_recurrence_for_user(ledger, "u_test")
        self.assertEqual(len(series_list), 0)
        self.assertEqual(rejected[0]["reason"], "insufficient_observations (< 3)")

    def test_18_speculative_income_cannot_establish_recurrence(self):
        """Unconfirmed bonus / prize messages do not invent recurring income."""
        events = [
            _make_event("e1", effective_date=date(2025, 1, 10)),
            _make_event("e2", effective_date=date(2025, 2, 10)),
            _make_event("e3", effective_date=date(2025, 3, 10)),
        ]
        msg = Message(
            message_id="m_prize",
            user_id="u_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
            source_type="financial_service",
            message_text="Your prize claim is verified and pending payment processing. Not credited yet.",
        )
        ledger = CanonicalLedger(events=events)
        series_list, _ = detect_recurrence_for_user(ledger, "u_test", messages=[msg])
        income_series = [s for s in series_list if s.direction == Direction.INFLOW]
        self.assertEqual(len(income_series), 0)


if __name__ == "__main__":
    unittest.main()
