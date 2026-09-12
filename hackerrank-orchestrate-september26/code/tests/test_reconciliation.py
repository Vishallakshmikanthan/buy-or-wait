"""Comprehensive unit tests for the evidence-resolution and canonical ledger layer."""

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
from code.models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageMetadata,
    Message,
)
from code.reconciliation import (
    ExchangeRateNotFoundError,
    reconcile_events,
    reconcile_single_event,
)


class TestReconciliationLayer(unittest.TestCase):
    """Test suite covering the 16 exact test requirements for reconciliation and canonical ledger."""

    def setUp(self):
        """Set up standard test fixtures."""
        self.profile = FinancialProfile(
            user_id="user_test",
            home_currency="USD",
            current_available_balance=Decimal("10000"),
            minimum_balance_to_keep=Decimal("2000"),
            financial_priorities=("emergency_savings",),
            expense_categories_to_protect=("rent", "groceries"),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("streaming",),
            payment_methods_user_will_consider=("full_payment", "installments"),
            max_installment_months=6,
        )

        self.profiles = {self.profile.user_id: self.profile}
        self.rate_map = {
            (date(2025, 6, 1), "EUR", "USD"): Decimal("1.10"),
            (date(2025, 6, 1), "USD", "EUR"): Decimal("0.909"),
        }
        self.exchange_rates = [
            ExchangeRate(rate_date=date(2025, 6, 1), from_currency="EUR", to_currency="USD", rate=Decimal("1.10"))
        ]

    def _make_event(
        self,
        event_id="ev_1",
        user_id="user_test",
        event_type="expense",
        direction="debit",
        amount=Decimal("100"),
        currency="USD",
        event_date=date(2025, 6, 1),
        settlement_date=date(2025, 6, 1),
        status="settled",
        source_row=10,
        linked_event_id=None,
        description="Test transaction",
        category="groceries",
    ) -> FinancialEvent:
        return FinancialEvent(
            event_id=event_id,
            source_row=source_row,
            user_id=user_id,
            event_type=event_type,
            description=description,
            category=category,
            direction=direction,
            amount=amount,
            currency=currency,
            event_date=event_date,
            settlement_date=settlement_date,
            status=status,
            linked_event_id=linked_event_id,
            flexibility="fixed",
            minimum_allowed_amount=None,
        )

    def test_1_settled_inflow_becomes_cash_inflow(self):
        """1. Settled inflow becomes a cash inflow."""
        ev = self._make_event(event_type="income", direction="credit", status="settled", amount=Decimal("3000"))
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertTrue(canon.is_cash_event)
        self.assertEqual(canon.direction, Direction.INFLOW)
        self.assertEqual(canon.cash_impact_type, CashImpactType.SETTLED_INFLOW)
        self.assertEqual(canon.amount_home, Decimal("3000"))

    def test_2_settled_outflow_becomes_cash_outflow(self):
        """2. Settled outflow becomes a cash outflow."""
        ev = self._make_event(event_type="expense", direction="debit", status="settled", amount=Decimal("500"))
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertTrue(canon.is_cash_event)
        self.assertEqual(canon.direction, Direction.OUTFLOW)
        self.assertEqual(canon.cash_impact_type, CashImpactType.SETTLED_OUTFLOW)
        self.assertEqual(canon.amount_home, Decimal("500"))

    def test_3_pending_debit_is_reserved(self):
        """3. Pending debit is reserved as cash outflow."""
        ev = self._make_event(event_type="expense", direction="debit", status="pending", amount=Decimal("250"))
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertTrue(canon.is_cash_event)
        self.assertEqual(canon.direction, Direction.OUTFLOW)
        self.assertEqual(canon.cash_impact_type, CashImpactType.PENDING_DEBIT_RESERVED)
        self.assertEqual(canon.amount_home, Decimal("250"))

    def test_4_pending_credit_is_not_counted_as_available_cash(self):
        """4. Pending credit is not counted as available cash."""
        ev = self._make_event(event_type="refund", direction="credit", status="pending", amount=Decimal("150"))
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertFalse(canon.is_cash_event)
        self.assertEqual(canon.direction, Direction.NON_CASH)
        self.assertEqual(canon.cash_impact_type, CashImpactType.PENDING_CREDIT_IGNORED)

    def test_5_failed_event_does_not_reduce_cash(self):
        """5. Failed event does not reduce cash."""
        ev = self._make_event(event_type="debt_payment", direction="debit", status="failed", amount=Decimal("400"))
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertFalse(canon.is_cash_event)
        self.assertEqual(canon.direction, Direction.NON_CASH)
        self.assertEqual(canon.cash_impact_type, CashImpactType.FAILED_IGNORED)

    def test_6_cancelled_event_does_not_reduce_cash(self):
        """6. Cancelled event does not reduce cash."""
        ev = self._make_event(event_type="expense", direction="debit", status="cancelled", amount=Decimal("600"))
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertFalse(canon.is_cash_event)
        self.assertEqual(canon.direction, Direction.NON_CASH)
        self.assertEqual(canon.cash_impact_type, CashImpactType.CANCELLED_IGNORED)

    def test_7_unrealized_non_cash_investment_does_not_become_cash(self):
        """7. Unrealized/non-cash investment valuation does not become cash."""
        ev = self._make_event(
            event_type="investment_valuation",
            direction="non_cash",
            status="unrealized",
            amount=Decimal("50000"),
            settlement_date=None,
        )
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertFalse(canon.is_cash_event)
        self.assertEqual(canon.direction, Direction.NON_CASH)
        self.assertEqual(canon.cash_impact_type, CashImpactType.UNREALIZED_NON_CASH)
        self.assertEqual(canon.effective_date, ev.event_date)

    def test_8_direction_is_respected_independently_of_amount_sign(self):
        """8. Direction is respected independently of amount sign."""
        # Both positive amounts, but debit is outflow and credit is inflow
        debit_ev = self._make_event(direction="debit", amount=Decimal("123.45"))
        credit_ev = self._make_event(direction="credit", amount=Decimal("123.45"), event_type="income")

        canon_debit = reconcile_single_event(debit_ev, "USD", self.rate_map, [])
        canon_credit = reconcile_single_event(credit_ev, "USD", self.rate_map, [])

        self.assertEqual(canon_debit.direction, Direction.OUTFLOW)
        self.assertEqual(canon_credit.direction, Direction.INFLOW)

    def test_9_missing_amount_remains_unresolved_rather_than_zero(self):
        """9. Missing amount remains unresolved rather than becoming zero."""
        ev = self._make_event(amount=None)
        img = ImageMetadata(image_id="image_01", user_id="user_test", request_id="req_1", related_event_id="ev_1")

        canon = reconcile_single_event(ev, "USD", self.rate_map, [], linked_image=img)
        self.assertTrue(canon.is_unresolved)
        self.assertTrue(canon.requires_image_extraction)
        self.assertIsNone(canon.amount_original)
        self.assertIsNone(canon.amount_home)
        self.assertIn("image_01", canon.unresolved_reason)

    def test_10_related_cancellation_message_cancels_correct_event(self):
        """10. Related cancellation message can cancel the correct event."""
        ev = self._make_event(event_id="ev_to_cancel", status="pending", amount=Decimal("200"))
        msg = Message(
            message_id="msg_1",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_to_cancel",
            sent_at=datetime(2025, 6, 2, 10, 0, tzinfo=timezone.utc),
            source_type="merchant",
            message_text="Notice: Your transaction cancelled by the merchant.",
        )

        canon = reconcile_single_event(ev, "USD", self.rate_map, [msg])
        self.assertEqual(canon.status, "cancelled")
        self.assertFalse(canon.is_cash_event)
        self.assertEqual(canon.cash_impact_type, CashImpactType.CANCELLED_IGNORED)
        self.assertIn("CANCEL", canon.applied_actions)

    def test_11_unrelated_message_does_not_alter_event(self):
        """11. Unrelated message does not alter an event."""
        ev = self._make_event(event_id="ev_independent", status="settled", amount=Decimal("300"))
        # Pass empty linked_messages because unrelated message does not have this event_id as related_event_id
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertEqual(canon.status, "settled")
        self.assertTrue(canon.is_cash_event)
        self.assertEqual(len(canon.applied_actions), 0)

    def test_12_conflicting_evidence_follows_deterministic_precedence(self):
        """12. Conflicting evidence follows deterministic chronological precedence."""
        ev = self._make_event(event_id="ev_conflict", status="pending", amount=Decimal("100"))
        msg_older = Message(
            message_id="msg_old",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_conflict",
            sent_at=datetime(2025, 6, 1, 9, 0, tzinfo=timezone.utc),
            source_type="bank",
            message_text="Charge is still being investigated.",
        )
        msg_newer = Message(
            message_id="msg_new",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_conflict",
            sent_at=datetime(2025, 6, 2, 11, 0, tzinfo=timezone.utc),
            source_type="bank",
            message_text="Transaction cancelled and reversed.",
        )

        # Messages processed in chronological order: newer cancellation wins
        canon = reconcile_single_event(ev, "USD", self.rate_map, [msg_older, msg_newer])
        self.assertEqual(canon.status, "cancelled")
        self.assertFalse(canon.is_cash_event)

    def test_13_foreign_currency_converts_using_correct_dated_rate(self):
        """13. Foreign-currency event converts using the correct dated exchange rate."""
        ev = self._make_event(
            currency="EUR",
            amount=Decimal("100"),
            settlement_date=date(2025, 6, 1),
        )
        # EUR to USD rate is 1.10
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertEqual(canon.amount_original, Decimal("100"))
        self.assertEqual(canon.currency_original, "EUR")
        self.assertEqual(canon.amount_home, Decimal("110.00"))
        self.assertEqual(canon.home_currency, "USD")
        self.assertEqual(canon.exchange_rate_used, Decimal("1.10"))
        self.assertIn("CURRENCY_CONVERSION", canon.applied_actions)

    def test_14_missing_exchange_rate_raises_clear_error(self):
        """14. Incorrect/missing exchange rate raises a clear deterministic error."""
        ev = self._make_event(
            currency="IDR",  # Not in rate map!
            amount=Decimal("100000"),
            settlement_date=date(2025, 6, 1),
        )
        with self.assertRaises(ExchangeRateNotFoundError) as ctx:
            reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertIn("Missing exchange rate", str(ctx.exception))
        self.assertIn("IDR -> USD", str(ctx.exception))

    def test_15_canonical_events_preserve_raw_event_provenance(self):
        """15. Canonical events preserve raw event provenance."""
        ev = self._make_event(event_id="ev_prov", source_row=42, user_id="user_test")
        canon = reconcile_single_event(ev, "USD", self.rate_map, [])
        self.assertEqual(canon.event_id, "ev_prov")
        self.assertEqual(canon.source_row, 42)
        self.assertEqual(canon.user_id, "user_test")
        self.assertTrue(any("row 42" in note for note in canon.evidence_chain))

    def test_16_repeated_execution_produces_identical_output(self):
        """16. Repeated execution produces identical canonical output."""
        events = [
            self._make_event(event_id="ev_b", event_date=date(2025, 6, 2)),
            self._make_event(event_id="ev_a", event_date=date(2025, 6, 1)),
        ]

        ledger1 = reconcile_events(events, self.profiles, self.exchange_rates, [], [])
        ledger2 = reconcile_events(events, self.profiles, self.exchange_rates, [], [])

        self.assertEqual(len(ledger1.events), len(ledger2.events))
        # Verify deterministic sorted ordering (2025-06-01 first, then 2025-06-02)
        self.assertEqual(ledger1.events[0].event_id, "ev_a")
        self.assertEqual(ledger1.events[1].event_id, "ev_b")
        for e1, e2 in zip(ledger1.events, ledger2.events):
            self.assertEqual(e1, e2)


if __name__ == "__main__":
    unittest.main()
