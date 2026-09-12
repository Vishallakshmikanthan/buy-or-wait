"""Comprehensive unit tests for image resolution, contextual extraction, and cash safety."""

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
from code.image_resolution import (
    ImageExtractionResult,
    extract_all_images,
    resolve_ledger_with_images,
)
from code.models import ExchangeRate, FinancialProfile
from code.reconciliation import ExchangeRateNotFoundError


class TestImageResolutionAndCashSafety(unittest.TestCase):
    """Test suite covering Part A (cash safety & conflict resolution) and Part H (image resolution)."""

    def setUp(self):
        """Set up standard test fixtures."""
        self.profile = FinancialProfile(
            user_id="user_test",
            home_currency="INR",
            current_available_balance=Decimal("50000"),
            minimum_balance_to_keep=Decimal("10000"),
            financial_priorities=("emergency_savings",),
            expense_categories_to_protect=("rent", "groceries"),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("streaming",),
            payment_methods_user_will_consider=("full_payment", "installments"),
            max_installment_months=6,
        )

        self.exchange_rates = [
            ExchangeRate(rate_date=date(2025, 10, 1), from_currency="USD", to_currency="INR", rate=Decimal("83.00")),
        ]

    def _make_unresolved_event(
        self,
        event_id="ev_img_1",
        user_id="user_test",
        currency="INR",
        effective_date=date(2025, 10, 1),
        status="settled",
        direction="debit",
        event_type="expense",
        home_currency="INR",
    ) -> CanonicalEvent:
        return CanonicalEvent(
            event_id=event_id,
            user_id=user_id,
            source_row=100,
            effective_date=effective_date,
            direction=Direction.NON_CASH,
            direction_original=direction,
            amount_original=None,
            currency_original=currency,
            amount_home=None,
            home_currency=home_currency,
            exchange_rate_used=None,
            exchange_rate_date=None,
            status=status,
            is_cash_event=False,
            cash_impact_type=CashImpactType.SETTLED_OUTFLOW,
            event_type=event_type,
            category="transport",
            description="Taxi fare",
            flexibility="fixed",
            minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None,
            linked_event_id=None,
            recurrence_type=RecurrenceClassification.ONE_TIME,
            is_unresolved=True,
            unresolved_reason="missing_amount_linked_to_image:image_12",
            evidence_chain=("Requires image extraction from image_12.png",),
            applied_actions=(),
        )

    # -------------------------------------------------------------
    # PART A1: Executable proof that unresolved events contribute zero numeric cash
    # -------------------------------------------------------------

    def test_a1_unresolved_event_zero_numerical_cash_contribution(self):
        """A1. Verify that an unresolved event contributes strictly zero numeric cash to calculations."""
        unresolved_ev = self._make_unresolved_event()
        ledger = CanonicalLedger(events=[unresolved_ev])

        # 1. Inspect property
        self.assertFalse(unresolved_ev.is_numerically_usable_cash_event)
        self.assertEqual(unresolved_ev.get_usable_cash_amount(), Decimal("0"))

        # 2. In calculation / simulation iteration:
        cash_events = ledger.get_cash_events()
        self.assertEqual(len(cash_events), 0, "Unresolved event must not appear in cash events")

        total_cash_flow = sum((e.get_usable_cash_amount() for e in ledger.events), Decimal("0"))
        self.assertEqual(total_cash_flow, Decimal("0"), "Cash flow contribution must be strictly 0")

    # -------------------------------------------------------------
    # PART H: Image resolution test suite (1 to 14)
    # -------------------------------------------------------------

    def test_h1_clear_single_monetary_value(self):
        """H1. Clear single monetary value resolves cleanly."""
        res = ImageExtractionResult(
            image_id="img_single",
            related_event_id="ev_img_1",
            extracted_amount=Decimal("1995.00"),
            extracted_currency="INR",
            extraction_method="ocr_contextual_keyword",
            document_type="Grocery Invoice",
            candidate_amounts=(Decimal("1995.00"),),
            selected_candidate_reason="Single clear invoice total",
            evidence_text="Total: 1995.00",
            unresolved_reason=None,
            is_resolved=True,
        )
        ledger = CanonicalLedger(events=[self._make_unresolved_event()])
        resolved_ledger = resolve_ledger_with_images(ledger, {"img_single": res}, self.exchange_rates)

        ev = resolved_ledger.events[0]
        self.assertFalse(ev.is_unresolved)
        self.assertEqual(ev.amount_original, Decimal("1995.00"))
        self.assertEqual(ev.amount_home, Decimal("1995.00"))

    def test_h2_multiple_monetary_values_with_contextual_selection(self):
        """H2. Multiple monetary values with contextual selection (e.g. Net Pay vs Deductions)."""
        res = ImageExtractionResult(
            image_id="img_multi",
            related_event_id="ev_img_1",
            extracted_amount=Decimal("4365000"),
            extracted_currency="IDR",
            extraction_method="ocr_contextual_keyword",
            document_type="Pay Slip",
            candidate_amounts=(Decimal("4780800"), Decimal("415800"), Decimal("4365000")),
            selected_candidate_reason="Selected Net Pay over gross and deductions",
            evidence_text="Net Pay: 4365000",
            unresolved_reason=None,
            is_resolved=True,
        )
        ledger = CanonicalLedger(events=[self._make_unresolved_event(currency="IDR", home_currency="IDR")])
        resolved_ledger = resolve_ledger_with_images(ledger, {"img_multi": res}, self.exchange_rates)

        ev = resolved_ledger.events[0]
        self.assertEqual(ev.amount_original, Decimal("4365000"))
        self.assertIn("Selected Net Pay", ev.evidence_chain[-1])

    def test_h3_ambiguous_values_remain_unresolved(self):
        """H3. Genuinely ambiguous values remain unresolved."""
        res = ImageExtractionResult(
            image_id="img_ambig",
            related_event_id="ev_img_1",
            extracted_amount=None,
            extracted_currency="INR",
            extraction_method="manual",
            document_type="Torn Receipt",
            candidate_amounts=(Decimal("100"), Decimal("200")),
            selected_candidate_reason="",
            evidence_text="",
            unresolved_reason="conflicting_totals_without_distinguishing_labels",
            is_resolved=False,
        )
        ledger = CanonicalLedger(events=[self._make_unresolved_event()])
        resolved_ledger = resolve_ledger_with_images(ledger, {"img_ambig": res}, self.exchange_rates)

        ev = resolved_ledger.events[0]
        self.assertTrue(ev.is_unresolved)
        self.assertIsNone(ev.amount_original)
        self.assertEqual(ev.unresolved_reason, "conflicting_totals_without_distinguishing_labels")

    def test_h4_ocr_failure_remains_unresolved(self):
        """H4. OCR failure remains unresolved."""
        res = ImageExtractionResult(
            image_id="img_fail",
            related_event_id="ev_img_1",
            extracted_amount=None,
            extracted_currency=None,
            extraction_method="ocr",
            document_type="Corrupt Image",
            candidate_amounts=(),
            selected_candidate_reason="",
            evidence_text="",
            unresolved_reason="unreadable_or_corrupted_image",
            is_resolved=False,
        )
        ledger = CanonicalLedger(events=[self._make_unresolved_event()])
        resolved_ledger = resolve_ledger_with_images(ledger, {"img_fail": res}, self.exchange_rates)

        ev = resolved_ledger.events[0]
        self.assertTrue(ev.is_unresolved)
        self.assertEqual(ev.get_usable_cash_amount(), Decimal("0"))

    def test_h5_invalid_monetary_candidate_remains_unresolved(self):
        """H5. Invalid candidate remains unresolved."""
        res = ImageExtractionResult(
            image_id="img_invalid",
            related_event_id="ev_img_1",
            extracted_amount=None,
            extracted_currency=None,
            extraction_method="ocr",
            document_type="Receipt",
            candidate_amounts=(),
            selected_candidate_reason="",
            evidence_text="",
            unresolved_reason="candidate_failed_arithmetic_crosscheck",
            is_resolved=False,
        )
        ledger = CanonicalLedger(events=[self._make_unresolved_event()])
        resolved_ledger = resolve_ledger_with_images(ledger, {"img_invalid": res}, self.exchange_rates)

        ev = resolved_ledger.events[0]
        self.assertTrue(ev.is_unresolved)

    def test_h6_extracted_amount_is_decimal(self):
        """H6. Extracted amount is strictly Decimal."""
        all_results = extract_all_images()
        for img_id, res in all_results.items():
            if res.is_resolved:
                self.assertIsInstance(res.extracted_amount, Decimal)

    def test_h7_foreign_currency_extracted_amount_converts_using_dated_rate(self):
        """H7. Foreign-currency extracted amount converts using correct dated rate."""
        # image_12 is USD 33.50 for user whose home is INR on 2025-10-01
        res = ImageExtractionResult(
            image_id="image_12",
            related_event_id="ev_img_1",
            extracted_amount=Decimal("33.50"),
            extracted_currency="USD",
            extraction_method="ocr_contextual_keyword",
            document_type="Taxi Receipt",
            candidate_amounts=(Decimal("33.50"),),
            selected_candidate_reason="Taxi total $33.50",
            evidence_text="Total: $33.50",
            unresolved_reason=None,
            is_resolved=True,
        )
        ev = self._make_unresolved_event(currency="USD", home_currency="INR", effective_date=date(2025, 10, 1))
        ledger = CanonicalLedger(events=[ev])
        resolved_ledger = resolve_ledger_with_images(ledger, {"image_12": res}, self.exchange_rates)

        resolved_ev = resolved_ledger.events[0]
        self.assertEqual(resolved_ev.amount_original, Decimal("33.50"))
        self.assertEqual(resolved_ev.currency_original, "USD")
        self.assertEqual(resolved_ev.exchange_rate_used, Decimal("83.00"))
        # 33.50 * 83.00 = 2780.50
        self.assertEqual(resolved_ev.amount_home, Decimal("2780.50"))

    def test_h8_image_cannot_resolve_unrelated_event(self):
        """H8. Image cannot resolve an unrelated event."""
        res = ImageExtractionResult(
            image_id="img_other",
            related_event_id="different_event_id",  # Mismatched ID!
            extracted_amount=Decimal("500"),
            extracted_currency="INR",
            extraction_method="ocr",
            document_type="Receipt",
            candidate_amounts=(Decimal("500"),),
            selected_candidate_reason="Total",
            evidence_text="Total: 500",
            unresolved_reason=None,
            is_resolved=True,
        )
        ledger = CanonicalLedger(events=[self._make_unresolved_event(event_id="ev_img_1")])
        resolved_ledger = resolve_ledger_with_images(ledger, {"img_other": res}, self.exchange_rates)

        ev = resolved_ledger.events[0]
        self.assertTrue(ev.is_unresolved, "Mismatched event ID must not resolve the event")

    def test_h9_image_provenance_is_preserved(self):
        """H9. Image provenance is preserved in evidence_chain and applied_actions."""
        res = ImageExtractionResult(
            image_id="image_05",
            related_event_id="ev_img_1",
            extracted_amount=Decimal("704.05"),
            extracted_currency="INR",
            extraction_method="ocr_contextual_keyword",
            document_type="Telecom Bill",
            candidate_amounts=(Decimal("704.05"),),
            selected_candidate_reason="Amount due",
            evidence_text="Total 704.05",
            unresolved_reason=None,
            is_resolved=True,
        )
        ledger = CanonicalLedger(events=[self._make_unresolved_event()])
        resolved_ledger = resolve_ledger_with_images(ledger, {"image_05": res}, self.exchange_rates)

        ev = resolved_ledger.events[0]
        self.assertTrue(any("Resolved via image_05.png" in note for note in ev.evidence_chain))
        self.assertIn("RESOLVE_IMAGE_AMOUNT", ev.applied_actions)

    def test_h10_resolved_event_becomes_numerically_usable(self):
        """H10. Resolved event becomes numerically usable."""
        res = ImageExtractionResult(
            image_id="img_res",
            related_event_id="ev_img_1",
            extracted_amount=Decimal("1500"),
            extracted_currency="INR",
            extraction_method="ocr",
            document_type="Receipt",
            candidate_amounts=(Decimal("1500"),),
            selected_candidate_reason="Total",
            evidence_text="Total: 1500",
            unresolved_reason=None,
            is_resolved=True,
        )
        ledger = CanonicalLedger(events=[self._make_unresolved_event()])
        resolved_ledger = resolve_ledger_with_images(ledger, {"img_res": res}, self.exchange_rates)

        ev = resolved_ledger.events[0]
        self.assertTrue(ev.is_numerically_usable_cash_event)
        self.assertEqual(len(resolved_ledger.get_cash_events()), 1)

    def test_h11_unresolved_event_remains_numerically_unusable(self):
        """H11. Unresolved event remains numerically unusable."""
        ev = self._make_unresolved_event()
        ledger = CanonicalLedger(events=[ev])
        self.assertFalse(ev.is_numerically_usable_cash_event)
        self.assertEqual(len(ledger.get_cash_events()), 0)

    def test_h12_repeated_extraction_produces_identical_results(self):
        """H12. Repeated extraction produces identical results deterministically."""
        res1 = extract_all_images()
        res2 = extract_all_images()
        self.assertEqual(res1, res2)

    def test_h13_unresolved_events_contribute_zero_numeric_cash_flow(self):
        """H13. Unresolved events contribute exactly zero numeric cash flow."""
        evs = [
            self._make_unresolved_event(event_id="u1"),
            self._make_unresolved_event(event_id="u2"),
        ]
        ledger = CanonicalLedger(events=evs)
        total_cash = sum((e.get_usable_cash_amount() for e in ledger.events), Decimal("0"))
        self.assertEqual(total_cash, Decimal("0"))

    def test_h14_resolved_image_event_contributes_exactly_its_resolved_amount(self):
        """H14. Resolved image event contributes exactly its resolved amount."""
        res = ImageExtractionResult(
            image_id="image_08",
            related_event_id="ev_img_1",
            extracted_amount=Decimal("15339.00"),
            extracted_currency="INR",
            extraction_method="ocr_contextual_keyword",
            document_type="Maintenance Receipt",
            candidate_amounts=(Decimal("15339.00"),),
            selected_candidate_reason="Total Amount Received",
            evidence_text="Total 15339.00",
            unresolved_reason=None,
            is_resolved=True,
        )
        ledger = CanonicalLedger(events=[self._make_unresolved_event()])
        resolved_ledger = resolve_ledger_with_images(ledger, {"image_08": res}, self.exchange_rates)

        ev = resolved_ledger.events[0]
        self.assertEqual(ev.get_usable_cash_amount(), Decimal("15339.00"))


if __name__ == "__main__":
    unittest.main()
