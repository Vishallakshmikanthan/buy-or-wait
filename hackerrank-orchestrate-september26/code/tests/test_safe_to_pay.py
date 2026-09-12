"""Comprehensive unit test suite for deterministic safe-to-pay and earliest full payment date.

Covers:
- All 25 required scenario tests from Section 19
- All 7 metamorphic property tests from Section 20
"""

import csv
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import unittest

from code.canonical import CanonicalEvent, CashImpactType, Direction, RecurrenceClassification
from code.models import FinancialProfile, FinancialRequest
from code.recurrence import FutureEvent, RecurrenceFrequency
from code.safe_to_pay import (
    CURRENCY_QUANTUM,
    SafeToPayCertificate,
    calculate_amount_safe_to_pay,
    calculate_earliest_full_payment_date,
    evaluate_request_safe_to_pay,
    get_currency_quantum,
    is_purchase_safe,
    make_candidate_purchase_event,
)
from code.simulator import (
    EventPriority,
    SimulatedEvent,
    SimulationResult,
    create_initial_state,
    simulate_user,
)


class TestSafeToPayLayer(unittest.TestCase):
    """Test suite covering safe-to-pay calculation and earliest date determination."""

    def setUp(self) -> None:
        self.user_id = "test_user"
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
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )

        self.request = FinancialRequest(
            request_id="req_01",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("500.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="Can I buy this?",
        )

    def _make_canonical(
        self,
        event_id: str,
        effective_date: date,
        direction: Direction,
        amount: Decimal,
        status: str = "settled",
        impact_type: CashImpactType = CashImpactType.SETTLED_OUTFLOW,
        category: str = "shopping",
        is_cash: bool = True,
        is_unresolved: bool = False,
        linked_id: str = None,
    ) -> CanonicalEvent:
        return CanonicalEvent(
            event_id=event_id,
            user_id=self.user_id,
            source_row=1,
            effective_date=effective_date,
            direction=direction,
            direction_original=direction.value,
            amount_original=amount,
            currency_original="USD",
            amount_home=amount,
            home_currency="USD",
            exchange_rate_used=Decimal("1"),
            exchange_rate_date=effective_date,
            status=status,
            is_cash_event=is_cash,
            cash_impact_type=impact_type,
            event_type="expense" if direction == Direction.OUTFLOW else "income",
            category=category,
            description=f"Test {event_id}",
            flexibility="fixed",
            minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None,
            linked_event_id=linked_id,
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
            currency="USD",
            category=category,
            event_type="income" if direction == Direction.INFLOW else "expense",
            description=f"Recurring {category}",
            series_id=series_id,
            frequency=RecurrenceFrequency.MONTHLY,
            is_forecast=True,
        )

    def test_01_zero_safe_amount_when_baseline_breached(self) -> None:
        """1. Zero safe amount when baseline trajectory breaches safety floor."""
        # Initial available: 1000. Floor: 200. Future expense: 900 on Jan 5 => min cash 100 < 200.
        exp = self._make_canonical("e1", date(2025, 1, 5), Direction.OUTFLOW, Decimal("900.00"))
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [exp], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("0.00"))

    def test_02_full_requested_amount_safe(self) -> None:
        """2. Full requested amount safe when headroom exceeds request."""
        # Initial available: 1000. Floor: 200. Headroom: 800. Request: 500 => full 500 safe.
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("500.00"))
        self.assertTrue(cert.is_full_payment_safe_today)
        self.assertEqual(cert.earliest_date_for_full_payment, self.request_date)

    def test_03_partial_safe_amount(self) -> None:
        """3. Partial safe amount when headroom is positive but less than requested."""
        # Initial available: 1000. Floor: 200. Obligation of 500 on Jan 10 => min cash 500.
        # Headroom: 500 - 200 = 300. Request: 500 => amount_safe_to_pay: 300.
        exp = self._make_canonical("e1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("500.00"))
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [exp], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("300.00"))
        self.assertFalse(cert.is_full_payment_safe_today)

    def test_04_maximum_safe_amount_is_exact(self) -> None:
        """4. Maximum safe amount is exact: candidate + 1 cent causes breach."""
        # Headroom is 300.00.
        exp = self._make_canonical("e1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("500.00"))
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=self.profile)
        safe_amt, _, _ = calculate_amount_safe_to_pay(self.request, baseline, [exp], [], self.profile)
        self.assertEqual(safe_amt, Decimal("300.00"))

        # Candidate of 300.00 passes
        cand_ok = make_candidate_purchase_event(self.request, Decimal("300.00"), self.request_date, "USD")
        ok_safe, _ = is_purchase_safe(self.user_id, self.sim_start, self.sim_end, [exp], [], self.profile, cand_ok)
        self.assertTrue(ok_safe)

        # Candidate of 300.01 fails
        cand_fail = make_candidate_purchase_event(self.request, Decimal("300.01"), self.request_date, "USD")
        fail_safe, _ = is_purchase_safe(self.user_id, self.sim_start, self.sim_end, [exp], [], self.profile, cand_fail)
        self.assertFalse(fail_safe)

    def test_05_amount_never_exceeds_request(self) -> None:
        """5. amount_safe_to_pay never exceeds requested_amount."""
        # Headroom: 800. Request: 50.
        req_small = FinancialRequest(
            request_id="req_small",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("50.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="Can I buy this?",
        )
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, profile=self.profile)
        cert = evaluate_request_safe_to_pay(req_small, baseline, [], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("50.00"))

    def test_06_amount_never_negative(self) -> None:
        """6. amount_safe_to_pay is never negative even in massive deficit."""
        exp = self._make_canonical("huge", date(2025, 1, 5), Direction.OUTFLOW, Decimal("5000.00"))
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [exp], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("0.00"))

    def test_07_monotonicity(self) -> None:
        """7. Monotonicity: if X is safe, all Y <= X are safe."""
        # Headroom: 400.
        exp = self._make_canonical("e1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("400.00"))
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=self.profile)
        safe_amt, _, _ = calculate_amount_safe_to_pay(self.request, baseline, [exp], [], self.profile)
        self.assertEqual(safe_amt, Decimal("400.00"))

        # Test monotonicity down to 0
        for test_val in [Decimal("400.00"), Decimal("300.00"), Decimal("150.50"), Decimal("0.01"), Decimal("0.00")]:
            cand = make_candidate_purchase_event(self.request, test_val, self.request_date, "USD")
            s, _ = is_purchase_safe(self.user_id, self.sim_start, self.sim_end, [exp], [], self.profile, cand)
            self.assertTrue(s, f"Value {test_val} should be safe")

    def test_08_safety_floor_equality_boundary(self) -> None:
        """8. Exact equality to safety floor is classified as safe."""
        # Initial available: 1000. Floor: 200. Headroom: 800.
        # Pay 800 => available cash is exactly 200.00 == floor => SAFE.
        cand = make_candidate_purchase_event(self.request, Decimal("800.00"), self.request_date, "USD")
        s, res = is_purchase_safe(self.user_id, self.sim_start, self.sim_end, [], [], self.profile, cand)
        self.assertTrue(s)
        self.assertEqual(res.minimum_projected_available_cash, Decimal("200.00"))
        self.assertFalse(res.is_safety_floor_breached)

    def test_09_one_unit_below_boundary_is_unsafe(self) -> None:
        """9. Available cash one cent below floor is strictly unsafe."""
        # Pay 800.01 => available cash is 199.99 < 200.00 => UNSAFE.
        cand = make_candidate_purchase_event(self.request, Decimal("800.01"), self.request_date, "USD")
        s, res = is_purchase_safe(self.user_id, self.sim_start, self.sim_end, [], [], self.profile, cand)
        self.assertFalse(s)
        self.assertTrue(res.is_safety_floor_breached)

    def test_10_future_obligation_limits_safe_amount(self) -> None:
        """10. Future scheduled obligation restricts today's safe amount."""
        # Rent of 600 due on Jan 20 limits today's headroom to 1000 - 600 - 200 = 200.
        rent = self._make_canonical("rent", date(2025, 1, 20), Direction.OUTFLOW, Decimal("600.00"))
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[rent], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [rent], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("200.00"))

    def test_11_pending_debit_limits_safe_amount(self) -> None:
        """11. Active pending debit reserve reduces safe amount."""
        # Pending debit of 250 on Jan 3 reserves 250: available cash drops from 1000 to 750.
        # Headroom: 750 - 200 = 550. Request is 500 => full 500 safe.
        # If pending debit is 350: available drops to 650. Headroom: 450 => safe amount is 450.
        p_debit = self._make_canonical("p1", date(2025, 1, 3), Direction.OUTFLOW, Decimal("350.00"), status="pending", impact_type=CashImpactType.PENDING_DEBIT_RESERVED)
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[p_debit], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [p_debit], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("450.00"))

    def test_12_future_salary_can_enable_full_payment_later(self) -> None:
        """12. Future confirmed salary enables full payment on salary date."""
        # Initial: 500. Floor: 200. Request: 500. Safe today: 300.
        # Salary of 1500 on Jan 15 makes full payment safe starting Jan 15.
        low_prof = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("500.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=(),
            max_installment_months=None,
        )
        salary = self._make_canonical("sal", date(2025, 1, 15), Direction.INFLOW, Decimal("1500.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW)
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[salary], profile=low_prof)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [salary], [], low_prof)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("300.00"))
        self.assertEqual(cert.earliest_date_for_full_payment, date(2025, 1, 15))

    def test_13_pending_credit_cannot_increase_safe_amount(self) -> None:
        """13. Pending credit contributes zero cash and does not increase safe amount."""
        p_cred = self._make_canonical("pc1", date(2025, 1, 5), Direction.NON_CASH, Decimal("5000.00"), status="pending", impact_type=CashImpactType.PENDING_CREDIT_IGNORED, is_cash=False)
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[p_cred], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [p_cred], [], self.profile)
        # Headroom is still based on 1000 - 200 = 800 (not 5800)
        self.assertEqual(baseline.total_inflows, Decimal("0.00"))
        self.assertEqual(cert.amount_safe_to_pay, Decimal("500.00"))

    def test_14_speculative_bonus_cannot_increase_safe_amount(self) -> None:
        """14. Unconfirmed/unrealized events cannot increase safe amount."""
        unrealized = self._make_canonical("stock", date(2025, 1, 5), Direction.NON_CASH, Decimal("10000.00"), status="unrealized", impact_type=CashImpactType.UNREALIZED_NON_CASH, is_cash=False)
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[unrealized], profile=self.profile)
        self.assertEqual(baseline.total_inflows, Decimal("0.00"))

    def test_15_candidate_does_not_mutate_canonical_ledger(self) -> None:
        """15. Evaluating candidate purchase never modifies source event lists."""
        events = [self._make_canonical("e1", date(2025, 1, 5), Direction.OUTFLOW, Decimal("100.00"))]
        orig_len = len(events)
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=events, profile=self.profile)
        _ = evaluate_request_safe_to_pay(self.request, baseline, events, [], self.profile)
        self.assertEqual(len(events), orig_len)

    def test_16_deterministic_repeated_calculation(self) -> None:
        """16. Running identical calculation twice produces identical byte-for-byte results."""
        exp = self._make_canonical("e1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("350.00"))
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=self.profile)
        c1 = evaluate_request_safe_to_pay(self.request, baseline, [exp], [], self.profile)
        c2 = evaluate_request_safe_to_pay(self.request, baseline, [exp], [], self.profile)
        self.assertEqual(c1, c2)

    def test_17_input_event_order_invariance(self) -> None:
        """17. Reordering input events produces identical safe amount."""
        e1 = self._make_canonical("e1", date(2025, 1, 5), Direction.OUTFLOW, Decimal("100.00"))
        e2 = self._make_canonical("e2", date(2025, 1, 10), Direction.INFLOW, Decimal("500.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW)
        e3 = self._make_canonical("e3", date(2025, 1, 15), Direction.OUTFLOW, Decimal("200.00"))

        base_fwd = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[e1, e2, e3], profile=self.profile)
        cert_fwd = evaluate_request_safe_to_pay(self.request, base_fwd, [e1, e2, e3], [], self.profile)

        base_rev = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[e3, e1, e2], profile=self.profile)
        cert_rev = evaluate_request_safe_to_pay(self.request, base_rev, [e3, e1, e2], [], self.profile)

        self.assertEqual(cert_fwd.amount_safe_to_pay, cert_rev.amount_safe_to_pay)
        self.assertEqual(cert_fwd.earliest_date_for_full_payment, cert_rev.earliest_date_for_full_payment)

    def test_18_decimal_precision(self) -> None:
        """18. Calculations preserve exact currency decimals with no binary floating distortion."""
        # 1000.00 - 749.53 (exp) - 200.00 (floor) = 50.47 headroom.
        exp = self._make_canonical("cents", date(2025, 1, 5), Direction.OUTFLOW, Decimal("749.53"))
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [exp], [], self.profile)
        self.assertIsInstance(cert.amount_safe_to_pay, Decimal)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("50.47"))

    def test_19_earliest_date_request_date_when_safe_today(self) -> None:
        """19. earliest_date_for_full_payment equals request_date when full amount is safe today."""
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, profile=self.profile)
        earliest = calculate_earliest_full_payment_date(self.request, baseline, [], [], self.profile, is_full_safe_today=True)
        self.assertEqual(earliest, self.request_date)

    def test_20_earliest_date_after_future_income(self) -> None:
        """20. earliest_date_for_full_payment identifies first date after salary arrives."""
        # Initial: 300. Floor: 200. Headroom today: 100. Request: 500.
        # Salary on Jan 15: 1000.
        prof = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("300.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=(),
            max_installment_months=None,
        )
        sal = self._make_canonical("sal", date(2025, 1, 15), Direction.INFLOW, Decimal("1000.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW)
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[sal], profile=prof)
        earliest = calculate_earliest_full_payment_date(self.request, baseline, [sal], [], prof, is_full_safe_today=False)
        self.assertEqual(earliest, date(2025, 1, 15))

    def test_21_never_safe_within_horizon(self) -> None:
        """21. earliest_date_for_full_payment returns None when full payment is never safe."""
        # Initial: 300. Floor: 200. Request: 500. No future income.
        prof = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("300.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=(),
            max_installment_months=None,
        )
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, profile=prof)
        earliest = calculate_earliest_full_payment_date(self.request, baseline, [], [], prof, is_full_safe_today=False)
        self.assertIsNone(earliest)

    def test_22_exact_horizon_boundary(self) -> None:
        """22. Salary on the exact end date (Day 90) enables full payment on Day 90."""
        prof = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("300.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=(),
            max_installment_months=None,
        )
        end_d = self.request_date + timedelta(days=90)
        sal = self._make_canonical("sal_end", end_d, Direction.INFLOW, Decimal("1000.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW)
        baseline = simulate_user(self.user_id, self.sim_start, end_d, canonical_events=[sal], profile=prof)
        earliest = calculate_earliest_full_payment_date(self.request, baseline, [sal], [], prof, is_full_safe_today=False)
        self.assertEqual(earliest, end_d)

    def test_23_same_day_event_ordering(self) -> None:
        """23. Same-day salary arriving on request_date clears before purchase, increasing safe amount."""
        # Initial: 300. Floor: 200. Request: 500.
        # Salary of 1000 arrives ON request_date.
        # If salary clears first (Priority 0): available becomes 1300. Headroom: 1100 => full 500 safe!
        prof = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("300.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=(),
            max_installment_months=None,
        )
        sal_today = self._make_canonical("sal_today", self.request_date, Direction.INFLOW, Decimal("1000.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW)
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[sal_today], profile=prof)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [sal_today], [], prof)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("500.00"))

    def test_24_candidate_full_purchase_evaluated_through_same_simulator(self) -> None:
        """24. Candidate purchase verified directly with simulate_user engine."""
        cand = make_candidate_purchase_event(self.request, Decimal("500.00"), self.request_date, "USD")
        safe, res = is_purchase_safe(self.user_id, self.sim_start, self.sim_end, [], [], self.profile, cand)
        self.assertTrue(safe)
        self.assertEqual(res.minimum_projected_available_cash, Decimal("500.00"))

    def test_25_no_spending_optimization_leaks_into_baseline_safe_amount(self) -> None:
        """25. Candidate does not reduce flexible expenses or stop subscriptions in safe amount calculation."""
        # User has reducible dining (100) and stoppable streaming (50).
        # Safe amount MUST NOT assume dining is reduced or streaming is stopped.
        # Available: 1000, Floor: 200. Rent: 600. Dining: 100. Streaming: 50.
        # Baseline min cash: 1000 - 600 - 100 - 50 = 250. Headroom = 250 - 200 = 50.
        # amount_safe_to_pay must be 50, NOT 50 + 150 = 200!
        rent = self._make_canonical("rent", date(2025, 1, 10), Direction.OUTFLOW, Decimal("600.00"))
        dining = self._make_canonical("dining", date(2025, 1, 12), Direction.OUTFLOW, Decimal("100.00"), category="dining")
        streaming = self._make_canonical("stream", date(2025, 1, 14), Direction.OUTFLOW, Decimal("50.00"), category="streaming")

        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[rent, dining, streaming], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [rent, dining, streaming], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("50.00"))


class TestMetamorphicProperties(unittest.TestCase):
    """Metamorphic invariance tests for safe-to-pay calculations."""

    def setUp(self) -> None:
        self.user_id = "test_user"
        self.request_date = date(2025, 1, 1)
        self.sim_start = self.request_date
        self.sim_end = self.request_date + timedelta(days=90)
        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("1000.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=(),
            max_installment_months=None,
        )
        self.request = FinancialRequest(
            request_id="req_meta",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("500.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="meta",
        )

    def _make_canonical(
        self,
        event_id: str,
        effective_date: date,
        direction: Direction,
        amount: Decimal,
        status: str = "settled",
        impact_type: CashImpactType = CashImpactType.SETTLED_OUTFLOW,
        category: str = "shopping",
        is_cash: bool = True,
    ) -> CanonicalEvent:
        return CanonicalEvent(
            event_id=event_id,
            user_id=self.user_id,
            source_row=1,
            effective_date=effective_date,
            direction=direction,
            direction_original=direction.value,
            amount_original=amount,
            currency_original="USD",
            amount_home=amount,
            home_currency="USD",
            exchange_rate_used=Decimal("1"),
            exchange_rate_date=effective_date,
            status=status,
            is_cash_event=is_cash,
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

    def test_property_a_requested_amount_monotonicity(self) -> None:
        """A. If requested amount increases with all else identical, safe amount cannot increase."""
        # Headroom is 300.
        exp = self._make_canonical("e1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("500.00"))
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=self.profile)

        # Request 200 => safe is 200
        req_200 = FinancialRequest("r1", self.user_id, self.request_date, "purchase", Decimal("200.00"), self.request_date, True, "r")
        safe_200, _, _ = calculate_amount_safe_to_pay(req_200, baseline, [exp], [], self.profile)

        # Request 400 => safe is 300
        req_400 = FinancialRequest("r2", self.user_id, self.request_date, "purchase", Decimal("400.00"), self.request_date, True, "r")
        safe_400, _, _ = calculate_amount_safe_to_pay(req_400, baseline, [exp], [], self.profile)

        # Request 600 => safe is 300
        req_600 = FinancialRequest("r3", self.user_id, self.request_date, "purchase", Decimal("600.00"), self.request_date, True, "r")
        safe_600, _, _ = calculate_amount_safe_to_pay(req_600, baseline, [exp], [], self.profile)

        self.assertLessEqual(safe_200, safe_400)
        self.assertEqual(safe_400, safe_600)

    def test_property_b_available_cash_monotonicity(self) -> None:
        """B. If current available cash increases by Δ with all else identical, safe amount should not decrease."""
        exp = self._make_canonical("e1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("500.00"))
        baseline_base = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=self.profile)
        safe_base, _, _ = calculate_amount_safe_to_pay(self.request, baseline_base, [exp], [], self.profile)

        higher_prof = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("1200.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=(),
            max_installment_months=None,
        )
        baseline_higher = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp], profile=higher_prof)
        safe_higher, _, _ = calculate_amount_safe_to_pay(self.request, baseline_higher, [exp], [], higher_prof)

        self.assertGreaterEqual(safe_higher, safe_base)

    def test_property_c_future_mandatory_outflow_monotonicity(self) -> None:
        """C. If a future mandatory outflow increases by Δ, safe amount should not increase."""
        exp1 = self._make_canonical("e1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("300.00"))
        base1 = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp1], profile=self.profile)
        safe1, _, _ = calculate_amount_safe_to_pay(self.request, base1, [exp1], [], self.profile)

        exp2 = self._make_canonical("e2", date(2025, 1, 10), Direction.OUTFLOW, Decimal("400.00"))
        base2 = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[exp2], profile=self.profile)
        safe2, _, _ = calculate_amount_safe_to_pay(self.request, base2, [exp2], [], self.profile)

        self.assertLessEqual(safe2, safe1)

    def test_property_d_confirmed_future_income_monotonicity(self) -> None:
        """D. If confirmed future income increases by Δ, safe amount should not decrease."""
        sal1 = self._make_canonical("sal1", date(2025, 1, 5), Direction.INFLOW, Decimal("200.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW)
        base1 = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[sal1], profile=self.profile)
        safe1, _, _ = calculate_amount_safe_to_pay(self.request, base1, [sal1], [], self.profile)

        sal2 = self._make_canonical("sal2", date(2025, 1, 5), Direction.INFLOW, Decimal("400.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW)
        base2 = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[sal2], profile=self.profile)
        safe2, _, _ = calculate_amount_safe_to_pay(self.request, base2, [sal2], [], self.profile)

        self.assertGreaterEqual(safe2, safe1)

    def test_property_e_unrelated_non_cash_event_invariance(self) -> None:
        """E. Adding an unrelated non-cash event must not change the answer."""
        base_clean = simulate_user(self.user_id, self.sim_start, self.sim_end, profile=self.profile)
        safe_clean, _, _ = calculate_amount_safe_to_pay(self.request, base_clean, [], [], self.profile)

        non_cash = self._make_canonical("nc", date(2025, 1, 5), Direction.NON_CASH, Decimal("1000.00"), status="failed", impact_type=CashImpactType.FAILED_IGNORED, is_cash=False)
        base_nc = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[non_cash], profile=self.profile)
        safe_nc, _, _ = calculate_amount_safe_to_pay(self.request, base_nc, [non_cash], [], self.profile)

        self.assertEqual(safe_clean, safe_nc)

    def test_property_f_reordering_source_events_invariance(self) -> None:
        """F. Reordering source events must not change the safe amount."""
        e1 = self._make_canonical("e1", date(2025, 1, 5), Direction.OUTFLOW, Decimal("100.00"))
        e2 = self._make_canonical("e2", date(2025, 1, 15), Direction.OUTFLOW, Decimal("200.00"))

        base_fwd = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[e1, e2], profile=self.profile)
        safe_fwd, _, _ = calculate_amount_safe_to_pay(self.request, base_fwd, [e1, e2], [], self.profile)

        base_rev = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[e2, e1], profile=self.profile)
        safe_rev, _, _ = calculate_amount_safe_to_pay(self.request, base_rev, [e2, e1], [], self.profile)

        self.assertEqual(safe_fwd, safe_rev)

    def test_property_g_deterministic_output(self) -> None:
        """G. Running the same calculation twice produces identical output."""
        e1 = self._make_canonical("e1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("300.00"))
        base1 = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[e1], profile=self.profile)
        c1 = evaluate_request_safe_to_pay(self.request, base1, [e1], [], self.profile)
        c2 = evaluate_request_safe_to_pay(self.request, base1, [e1], [], self.profile)
        self.assertEqual(c1.amount_safe_to_pay, c2.amount_safe_to_pay)
        self.assertEqual(c1.earliest_date_for_full_payment, c2.earliest_date_for_full_payment)


class TestCurrencyQuantumAudit(unittest.TestCase):
    """Audit and verification of currency quantum across the dataset and contest semantics."""

    def setUp(self) -> None:
        self.dataset_dir = Path(__file__).resolve().parent.parent.parent / "dataset"

    def test_real_requests_quantum_compatibility(self) -> None:
        """Every requested_amount in requests.csv and sample_requests.csv must be a multiple of quantum."""
        profiles = {}
        with open(self.dataset_dir / "financial_profiles.csv", "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                profiles[r["user_id"]] = r["home_currency"]

        for req_file in ["requests.csv", "sample_requests.csv"]:
            file_path = self.dataset_dir / req_file
            if not file_path.exists():
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                for line_no, r in enumerate(csv.DictReader(f), start=2):
                    curr = profiles[r["user_id"]]
                    q = get_currency_quantum(curr)
                    amt = Decimal(r["requested_amount"])
                    remainder = amt % q
                    self.assertEqual(
                        remainder,
                        Decimal("0"),
                        f"Non-quantum amount {amt} for currency {curr} in {req_file}:{line_no}",
                    )

    def test_quantum_map_covers_all_profile_currencies(self) -> None:
        """All distinct currencies present in financial_profiles.csv must be present in CURRENCY_QUANTUM."""
        with open(self.dataset_dir / "financial_profiles.csv", "r", encoding="utf-8") as f:
            currencies = {r["home_currency"] for r in csv.DictReader(f)}

        for curr in currencies:
            self.assertIn(curr, CURRENCY_QUANTUM, f"Currency {curr} not found in CURRENCY_QUANTUM")
            self.assertEqual(
                CURRENCY_QUANTUM[curr],
                Decimal("0.01"),
                f"Currency {curr} quantum must be 0.01 in this dataset",
            )

    def test_dataset_all_monetary_tables_representable_to_two_decimals(self) -> None:
        """Verify no monetary column across profiles, events, or payment options requires finer than 0.01."""
        q = Decimal("0.01")
        # Check profiles
        with open(self.dataset_dir / "financial_profiles.csv", "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.assertEqual(Decimal(r["current_available_balance"]) % q, Decimal("0"))
                self.assertEqual(Decimal(r["minimum_balance_to_keep"]) % q, Decimal("0"))

        # Check events
        with open(self.dataset_dir / "financial_events.csv", "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                amt_str = r.get("amount")
                if amt_str:
                    self.assertEqual(Decimal(amt_str) % q, Decimal("0"))

        # Check payment options
        opt_path = self.dataset_dir / "request_payment_options.csv"
        if opt_path.exists():
            with open(opt_path, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    for col in ["payment_amount", "financing_fee", "total_payable_amount"]:
                        val = r.get(col)
                        if val:
                            self.assertEqual(Decimal(val) % q, Decimal("0"))


class TestHeadroomShortcutTranslationInvariant(unittest.TestCase):
    """Formal proof and verification of the candidate purchase translation property.

    Mathematical Invariant:
    For two candidate amounts X and Y where Y > X, when candidate purchase is injected
    at request_date at EventPriority.OUTFLOW:
        minimum_available_after(Y) = minimum_available_after(X) - (Y - X)
    and for every date t >= request_date:
        available_cash_after(Y, t) = available_cash_after(X, t) - (Y - X)
    """

    def setUp(self) -> None:
        self.user_id = "test_user_invariant"
        self.request_date = date(2025, 1, 1)
        self.sim_start = self.request_date
        self.sim_end = self.request_date + timedelta(days=90)
        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("1500.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=("rent",),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )
        self.request = FinancialRequest(
            request_id="req_inv",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("500.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="Invariant test",
        )

    def _assert_translation_property(
        self,
        canonical: Sequence[CanonicalEvent],
        future: Sequence[FutureEvent],
        amt_x: Decimal,
        amt_y: Decimal,
    ) -> None:
        self.assertGreater(amt_y, amt_x)
        delta = amt_y - amt_x

        ev_x = make_candidate_purchase_event(self.request, amt_x, self.request_date, "USD", event_id="candidate_purchase_inv")
        ev_y = make_candidate_purchase_event(self.request, amt_y, self.request_date, "USD", event_id="candidate_purchase_inv")

        res_x = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=canonical, future_events=future, profile=self.profile, additional_events=[ev_x])
        res_y = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=canonical, future_events=future, profile=self.profile, additional_events=[ev_y])

        # 1. Identify candidate step
        cand_step_x = [t.step for t in res_x.event_transitions if t.event_id == ev_x.event_id][0]
        cand_step_y = [t.step for t in res_y.event_transitions if t.event_id == ev_y.event_id][0]

        post_transitions_x = [t for t in res_x.event_transitions if t.step >= cand_step_x]
        post_transitions_y = [t for t in res_y.event_transitions if t.step >= cand_step_y]

        self.assertEqual(len(post_transitions_x), len(post_transitions_y))

        # Every transition after candidate shifts by exactly delta
        for tx, ty in zip(post_transitions_x, post_transitions_y):
            self.assertEqual(tx.event_id, ty.event_id)
            self.assertEqual(
                ty.available_after,
                tx.available_after - delta,
                f"Transition for {tx.event_id} did not shift by delta {delta}: {ty.available_after} vs {tx.available_after - delta}",
            )

        # 2. Minimum available cash after candidate purchase invariant
        min_after_x = min(t.available_after for t in post_transitions_x)
        min_after_y = min(t.available_after for t in post_transitions_y)
        self.assertEqual(
            min_after_y,
            min_after_x - delta,
            f"Post-candidate minimum available cash did not translate linearly: {min_after_y} vs {min_after_x - delta}",
        )

        # 3. Daily closing available cash invariant for all days on or after request date
        for snap_x, snap_y in zip(res_x.daily_snapshots, res_y.daily_snapshots):
            self.assertEqual(snap_x.date, snap_y.date)
            if snap_x.date >= self.request_date:
                self.assertEqual(
                    snap_y.closing_available_cash,
                    snap_x.closing_available_cash - delta,
                    f"Day {snap_x.date} closing cash did not shift by delta {delta}",
                )

    def test_invariant_no_future_events(self) -> None:
        self._assert_translation_property([], [], Decimal("100.00"), Decimal("250.00"))

    def test_invariant_future_mandatory_outflow(self) -> None:
        exp = CanonicalEvent(
            event_id="exp_inv", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 10),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("400.00"),
            currency_original="USD", amount_home=Decimal("400.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 10), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="rent", description="Rent", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_translation_property([exp], [], Decimal("50.00"), Decimal("200.00"))

    def test_invariant_future_salary(self) -> None:
        sal = FutureEvent(
            event_id="sal_inv", user_id=self.user_id, effective_date=date(2025, 1, 15),
            direction=Direction.INFLOW, amount_home=Decimal("1000.00"), currency="USD",
            category="salary", event_type="income", description="Salary", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        self._assert_translation_property([], [sal], Decimal("100.00"), Decimal("400.00"))

    def test_invariant_pending_debit_reserved(self) -> None:
        pend = CanonicalEvent(
            event_id="pend_inv", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 1),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("200.00"),
            currency_original="USD", amount_home=Decimal("200.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 1), status="pending",
            is_cash_event=True, cash_impact_type=CashImpactType.PENDING_DEBIT_RESERVED, event_type="expense",
            category="groceries", description="Pending grocery hold", flexibility="fixed",
            minimum_allowed_amount_original=None, minimum_allowed_amount_home=None, linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN, is_unresolved=False, unresolved_reason=None,
            evidence_chain=(), applied_actions=(),
        )
        self._assert_translation_property([pend], [], Decimal("100.00"), Decimal("300.00"))

    def test_invariant_same_day_salary(self) -> None:
        sal_today = CanonicalEvent(
            event_id="sal_today", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 1),
            direction=Direction.INFLOW, direction_original="INFLOW", amount_original=Decimal("800.00"),
            currency_original="USD", amount_home=Decimal("800.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 1), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_INFLOW, event_type="income",
            category="salary", description="Salary today", flexibility="fixed",
            minimum_allowed_amount_original=None, minimum_allowed_amount_home=None, linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN, is_unresolved=False, unresolved_reason=None,
            evidence_chain=(), applied_actions=(),
        )
        self._assert_translation_property([sal_today], [], Decimal("150.00"), Decimal("450.00"))

    def test_invariant_same_day_obligation(self) -> None:
        ob_today = CanonicalEvent(
            event_id="ob_today", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 1),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("300.00"),
            currency_original="USD", amount_home=Decimal("300.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 1), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="bills", description="Bill today", flexibility="fixed",
            minimum_allowed_amount_original=None, minimum_allowed_amount_home=None, linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN, is_unresolved=False, unresolved_reason=None,
            evidence_chain=(), applied_actions=(),
        )
        self._assert_translation_property([ob_today], [], Decimal("50.00"), Decimal("200.00"))

    def test_invariant_safety_floor_boundary(self) -> None:
        # Initial cash: 1500, floor: 200. Headroom: 1300.
        # Compare amounts close to the headroom boundary: 1299.00 and 1300.00
        self._assert_translation_property([], [], Decimal("1299.00"), Decimal("1300.00"))

    def test_invariant_baseline_close_to_floor(self) -> None:
        # Outflow leaves only 10.00 headroom above floor
        exp = CanonicalEvent(
            event_id="exp_tight", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 10),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("1290.00"),
            currency_original="USD", amount_home=Decimal("1290.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 10), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="rent", description="Tight rent", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_translation_property([exp], [], Decimal("1.00"), Decimal("5.00"))


class TestExhaustiveAmountCrossCheck(unittest.TestCase):
    """Exhaustive comparison: optimized calculate_amount_safe_to_pay vs brute-force search."""

    def setUp(self) -> None:
        self.user_id = "test_user_exhaustive"
        self.request_date = date(2025, 1, 1)
        self.sim_start = self.request_date
        self.sim_end = self.request_date + timedelta(days=90)
        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("100.00"),
            minimum_balance_to_keep=Decimal("20.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )
        self.request = FinancialRequest(
            request_id="req_ex",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("50.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="Exhaustive test",
        )
        self.q = Decimal("1.00")  # Coarse quantum for fast exhaustive test

    def _brute_force_safe_amount(
        self,
        baseline: SimulationResult,
        canonical: Sequence[CanonicalEvent],
        future: Sequence[FutureEvent],
        q: Decimal,
    ) -> Decimal:
        req_amt = self.request.requested_amount
        max_units = int(req_amt / q)
        best_amt = Decimal("0")
        for u in range(max_units + 1):
            cand_amt = u * q
            cand_ev = make_candidate_purchase_event(self.request, cand_amt, self.request_date, "USD")
            safe, _ = is_purchase_safe(self.user_id, self.sim_start, self.sim_end, canonical, future, self.profile, cand_ev)
            if safe:
                best_amt = cand_amt
        return best_amt

    def _assert_exhaustive_agreement(
        self,
        canonical: Sequence[CanonicalEvent] = (),
        future: Sequence[FutureEvent] = (),
        q: Decimal = Decimal("1.00"),
    ) -> None:
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=canonical, future_events=future, profile=self.profile)
        optimized_amt, _, method = calculate_amount_safe_to_pay(self.request, baseline, canonical, future, self.profile, quantum=q)
        exhaustive_amt = self._brute_force_safe_amount(baseline, canonical, future, q)
        self.assertEqual(
            optimized_amt,
            exhaustive_amt,
            f"Mismatched amount: optimized={optimized_amt} (method={method}), exhaustive={exhaustive_amt}",
        )

    def test_exhaustive_no_future_events(self) -> None:
        self._assert_exhaustive_agreement()

    def test_exhaustive_future_mandatory_outflow(self) -> None:
        exp = CanonicalEvent(
            event_id="e_fut", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 10),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("35.00"),
            currency_original="USD", amount_home=Decimal("35.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 10), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="utilities", description="Electric", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_exhaustive_agreement(canonical=[exp])

    def test_exhaustive_future_salary(self) -> None:
        sal = FutureEvent(
            event_id="sal_fut", user_id=self.user_id, effective_date=date(2025, 1, 15),
            direction=Direction.INFLOW, amount_home=Decimal("100.00"), currency="USD",
            category="salary", event_type="income", description="Salary", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        self._assert_exhaustive_agreement(future=[sal])

    def test_exhaustive_pending_debit(self) -> None:
        pend = CanonicalEvent(
            event_id="p_deb", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 1),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("15.00"),
            currency_original="USD", amount_home=Decimal("15.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 1), status="pending",
            is_cash_event=True, cash_impact_type=CashImpactType.PENDING_DEBIT_RESERVED, event_type="expense",
            category="shopping", description="Hold", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_exhaustive_agreement(canonical=[pend])

    def test_exhaustive_same_day_salary(self) -> None:
        sal = CanonicalEvent(
            event_id="s_today", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 1),
            direction=Direction.INFLOW, direction_original="INFLOW", amount_original=Decimal("50.00"),
            currency_original="USD", amount_home=Decimal("50.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 1), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_INFLOW, event_type="income",
            category="salary", description="Salary today", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_exhaustive_agreement(canonical=[sal])

    def test_exhaustive_same_day_pending_hold(self) -> None:
        pend = CanonicalEvent(
            event_id="p_today", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 1),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("25.00"),
            currency_original="USD", amount_home=Decimal("25.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 1), status="pending",
            is_cash_event=True, cash_impact_type=CashImpactType.PENDING_DEBIT_RESERVED, event_type="expense",
            category="shopping", description="Pending hold", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_exhaustive_agreement(canonical=[pend])

    def test_exhaustive_same_day_obligation(self) -> None:
        ob = CanonicalEvent(
            event_id="o_today", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 1),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("30.00"),
            currency_original="USD", amount_home=Decimal("30.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 1), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="bills", description="Bill today", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_exhaustive_agreement(canonical=[ob])

    def test_exhaustive_safety_floor_boundary(self) -> None:
        # Outflow leaves exactly 0 headroom (100 - 80 = 20 floor)
        exp = CanonicalEvent(
            event_id="e_bound", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 5),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("80.00"),
            currency_original="USD", amount_home=Decimal("80.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 5), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="rent", description="Boundary rent", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_exhaustive_agreement(canonical=[exp])

    def test_exhaustive_baseline_close_to_floor(self) -> None:
        # Outflow leaves 3.00 headroom above floor
        exp = CanonicalEvent(
            event_id="e_close", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 5),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("77.00"),
            currency_original="USD", amount_home=Decimal("77.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 5), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="rent", description="Close rent", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_exhaustive_agreement(canonical=[exp])

    def test_exhaustive_baseline_already_unsafe(self) -> None:
        # Outflow breaches floor (100 - 85 = 15 < 20 floor)
        exp = CanonicalEvent(
            event_id="e_breach", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 5),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("85.00"),
            currency_original="USD", amount_home=Decimal("85.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 5), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="rent", description="Breach rent", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_exhaustive_agreement(canonical=[exp])


class TestStrengthenedMonotonicityPrefix(unittest.TestCase):
    """Formal verification of the strict prefix monotonicity property:

    For any sequence of discrete candidate amounts 0, q, 2q, ..., requested_amount:
    is_purchase_safe(k * q) MUST follow:
        [SAFE, SAFE, ..., SAFE, UNSAFE, UNSAFE, ..., UNSAFE]
    Never:
        [SAFE, UNSAFE, SAFE, ...]
    """

    def setUp(self) -> None:
        self.user_id = "test_user_mono"
        self.request_date = date(2025, 1, 1)
        self.sim_start = self.request_date
        self.sim_end = self.request_date + timedelta(days=90)
        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("100.00"),
            minimum_balance_to_keep=Decimal("20.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )
        self.request = FinancialRequest(
            request_id="req_mono",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("50.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="Monotonicity test",
        )

    def _verify_monotonicity_prefix(
        self,
        canonical: Sequence[CanonicalEvent] = (),
        future: Sequence[FutureEvent] = (),
        q: Decimal = Decimal("1.00"),
    ) -> None:
        req_amt = self.request.requested_amount
        max_units = int(req_amt / q)
        predicates: List[bool] = []

        for u in range(max_units + 1):
            cand_amt = u * q
            cand_ev = make_candidate_purchase_event(self.request, cand_amt, self.request_date, "USD")
            safe, _ = is_purchase_safe(self.user_id, self.sim_start, self.sim_end, canonical, future, self.profile, cand_ev)
            predicates.append(safe)

        # Monotonicity rule: Once a False occurs, all subsequent entries must be False
        seen_false = False
        for idx, p in enumerate(predicates):
            if seen_false and p:
                self.fail(
                    f"Non-monotonic predicate at index {idx} (amount {idx * q}): saw SAFE after UNSAFE! Sequence: {predicates}"
                )
            if not p:
                seen_false = True

    def test_monotonicity_clean_slate(self) -> None:
        self._verify_monotonicity_prefix()

    def test_monotonicity_with_interleaved_events(self) -> None:
        e1 = CanonicalEvent(
            event_id="e1", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 5),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("30.00"),
            currency_original="USD", amount_home=Decimal("30.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 5), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="bills", description="Bill", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        sal = FutureEvent(
            event_id="s1", user_id=self.user_id, effective_date=date(2025, 1, 15),
            direction=Direction.INFLOW, amount_home=Decimal("40.00"), currency="USD",
            category="salary", event_type="income", description="Salary", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        self._verify_monotonicity_prefix(canonical=[e1], future=[sal])


class TestEarliestDateExhaustiveCrossCheck(unittest.TestCase):
    """Exhaustive comparison: calculate_earliest_full_payment_date vs 91-day brute-force evaluation."""

    def setUp(self) -> None:
        self.user_id = "test_user_earliest"
        self.request_date = date(2025, 1, 1)
        self.sim_start = self.request_date
        self.sim_end = self.request_date + timedelta(days=90)
        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("500.00"),
            minimum_balance_to_keep=Decimal("100.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )
        self.request = FinancialRequest(
            request_id="req_early",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("400.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="Earliest test",
        )

    def _brute_force_earliest_date(
        self,
        canonical: Sequence[CanonicalEvent],
        future: Sequence[FutureEvent],
    ) -> Optional[date]:
        for offset in range(91):
            cand_d = self.request_date + timedelta(days=offset)
            cand_ev = make_candidate_purchase_event(self.request, self.request.requested_amount, cand_d, "USD")
            safe, _ = is_purchase_safe(self.user_id, self.sim_start, self.sim_end, canonical, future, self.profile, cand_ev)
            if safe:
                return cand_d
        return None

    def _assert_earliest_date_agreement(
        self,
        canonical: Sequence[CanonicalEvent] = (),
        future: Sequence[FutureEvent] = (),
    ) -> None:
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=canonical, future_events=future, profile=self.profile)
        optimized_date = calculate_earliest_full_payment_date(self.request, baseline, canonical, future, self.profile)
        exhaustive_date = self._brute_force_earliest_date(canonical, future)
        self.assertEqual(
            optimized_date,
            exhaustive_date,
            f"Earliest date mismatch: optimized={optimized_date}, exhaustive={exhaustive_date}",
        )

    def test_earliest_safe_today(self) -> None:
        self._assert_earliest_date_agreement()

    def test_earliest_safe_after_salary(self) -> None:
        # Starting cash: 500, floor: 100. Bill on Jan 5 of 300 => leaves 200 (not enough for 400).
        # Salary on Jan 10 of 1000 => leaves 1200. Full 400 becomes safe on Jan 10.
        exp = CanonicalEvent(
            event_id="b1", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 5),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("300.00"),
            currency_original="USD", amount_home=Decimal("300.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 5), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="bills", description="Bill", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        sal = FutureEvent(
            event_id="sal1", user_id=self.user_id, effective_date=date(2025, 1, 10),
            direction=Direction.INFLOW, amount_home=Decimal("1000.00"), currency="USD",
            category="salary", event_type="income", description="Salary", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        self._assert_earliest_date_agreement(canonical=[exp], future=[sal])

    def test_earliest_safe_after_multiple_salaries(self) -> None:
        # Bill on Jan 5 of 450 => leaves 50.
        # Sal 1 on Jan 10 of 200 => leaves 250 (still not enough for 400 + 100 floor = 500).
        # Sal 2 on Jan 20 of 300 => leaves 550 (now safe!).
        exp = CanonicalEvent(
            event_id="b1", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 5),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("450.00"),
            currency_original="USD", amount_home=Decimal("450.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 5), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="bills", description="Big bill", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        sal1 = FutureEvent(
            event_id="sal1", user_id=self.user_id, effective_date=date(2025, 1, 10),
            direction=Direction.INFLOW, amount_home=Decimal("200.00"), currency="USD",
            category="salary", event_type="income", description="Salary 1", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        sal2 = FutureEvent(
            event_id="sal2", user_id=self.user_id, effective_date=date(2025, 1, 20),
            direction=Direction.INFLOW, amount_home=Decimal("300.00"), currency="USD",
            category="salary", event_type="income", description="Salary 2", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        self._assert_earliest_date_agreement(canonical=[exp], future=[sal1, sal2])

    def test_earliest_safe_after_an_obligation(self) -> None:
        # Jan 2: Obligation of 100. Jan 10: Inflow of 600.
        ob = CanonicalEvent(
            event_id="ob1", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 2),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("100.00"),
            currency_original="USD", amount_home=Decimal("100.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 2), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="rent", description="Rent", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        sal = FutureEvent(
            event_id="sal1", user_id=self.user_id, effective_date=date(2025, 1, 10),
            direction=Direction.INFLOW, amount_home=Decimal("600.00"), currency="USD",
            category="salary", event_type="income", description="Salary", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        self._assert_earliest_date_agreement(canonical=[ob], future=[sal])

    def test_earliest_same_day_salary(self) -> None:
        sal_today = CanonicalEvent(
            event_id="sal_td", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 1),
            direction=Direction.INFLOW, direction_original="INFLOW", amount_original=Decimal("500.00"),
            currency_original="USD", amount_home=Decimal("500.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 1), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_INFLOW, event_type="income",
            category="salary", description="Salary today", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_earliest_date_agreement(canonical=[sal_today])

    def test_earliest_pending_debit(self) -> None:
        pend = CanonicalEvent(
            event_id="pend_d", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 1),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("150.00"),
            currency_original="USD", amount_home=Decimal("150.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 1), status="pending",
            is_cash_event=True, cash_impact_type=CashImpactType.PENDING_DEBIT_RESERVED, event_type="expense",
            category="shopping", description="Pending debit", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        sal = FutureEvent(
            event_id="sal1", user_id=self.user_id, effective_date=date(2025, 1, 10),
            direction=Direction.INFLOW, amount_home=Decimal("500.00"), currency="USD",
            category="salary", event_type="income", description="Salary", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        self._assert_earliest_date_agreement(canonical=[pend], future=[sal])

    def test_earliest_baseline_unsafe(self) -> None:
        # Baseline breaches floor on Jan 5 (500 - 450 = 50 < 100 floor)
        breach = CanonicalEvent(
            event_id="brk", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 5),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("450.00"),
            currency_original="USD", amount_home=Decimal("450.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 5), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="rent", description="Breaching rent", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        self._assert_earliest_date_agreement(canonical=[breach])

    def test_earliest_never_safe(self) -> None:
        # Heavy recurring bills every week, no income
        bills = [
            FutureEvent(
                event_id=f"bill_{i}", user_id=self.user_id, effective_date=date(2025, 1, 1) + timedelta(days=i * 7),
                direction=Direction.OUTFLOW, amount_home=Decimal("100.00"), currency="USD",
                category="bills", event_type="expense", description="Weekly bill", series_id="s_b",
                frequency=RecurrenceFrequency.WEEKLY, is_forecast=True,
            )
            for i in range(1, 12)
        ]
        self._assert_earliest_date_agreement(future=bills)

    def test_earliest_exact_final_horizon_date(self) -> None:
        # Outflow today leaves 200 (not enough for 400).
        # Huge salary arrives on EXACT final day (Day 90 = 2025-04-01).
        final_d = self.request_date + timedelta(days=90)
        exp = CanonicalEvent(
            event_id="b1", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 2),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("250.00"),
            currency_original="USD", amount_home=Decimal("250.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 2), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="bills", description="Bill", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        sal = FutureEvent(
            event_id="sal_final", user_id=self.user_id, effective_date=final_d,
            direction=Direction.INFLOW, amount_home=Decimal("2000.00"), currency="USD",
            category="salary", event_type="income", description="Final day salary", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        self._assert_earliest_date_agreement(canonical=[exp], future=[sal])

    def test_earliest_fluctuating_cash_trajectory(self) -> None:
        # Cash goes up and down over multiple weeks
        events = [
            CanonicalEvent(
                event_id="e_fluct_1", user_id=self.user_id, source_row=1, effective_date=date(2025, 1, 5),
                direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("300.00"),
                currency_original="USD", amount_home=Decimal("300.00"), home_currency="USD",
                exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 5), status="settled",
                is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
                category="rent", description="Rent", flexibility="fixed", minimum_allowed_amount_original=None,
                minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
                is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
            ),
            CanonicalEvent(
                event_id="e_fluct_2", user_id=self.user_id, source_row=2, effective_date=date(2025, 1, 20),
                direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("200.00"),
                currency_original="USD", amount_home=Decimal("200.00"), home_currency="USD",
                exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 20), status="settled",
                is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
                category="bills", description="Bill", flexibility="fixed", minimum_allowed_amount_original=None,
                minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
                is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
            ),
        ]
        sal = FutureEvent(
            event_id="sal_f", user_id=self.user_id, effective_date=date(2025, 1, 15),
            direction=Direction.INFLOW, amount_home=Decimal("400.00"), currency="USD",
            category="salary", event_type="income", description="Salary", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        sal2 = FutureEvent(
            event_id="sal_f2", user_id=self.user_id, effective_date=date(2025, 2, 1),
            direction=Direction.INFLOW, amount_home=Decimal("1000.00"), currency="USD",
            category="salary", event_type="income", description="Salary 2", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )
        self._assert_earliest_date_agreement(canonical=events, future=[sal, sal2])


class TestSameDaySemantics(unittest.TestCase):
    """Explicit verification of same-day event priorities under simulator.py."""

    def setUp(self) -> None:
        self.user_id = "test_user_sameday"
        self.request_date = date(2025, 1, 1)
        self.sim_start = self.request_date
        self.sim_end = self.request_date + timedelta(days=90)
        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("300.00"),
            minimum_balance_to_keep=Decimal("100.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )
        self.request = FinancialRequest(
            request_id="req_sd",
            user_id=self.user_id,
            request_date=self.request_date,
            request_type="purchase",
            requested_amount=Decimal("400.00"),
            desired_completion_date=self.request_date + timedelta(days=30),
            allows_partial_payment=True,
            request_text="Same day test",
        )

    def test_case_a_salary_before_candidate_purchase(self) -> None:
        """Case A: Salary arrives on request date. Candidate purchase executes after salary."""
        # Initial cash 300, floor 100 => Headroom 200 (not enough for 400).
        # But salary of 500 arrives on Jan 1. Inflow is Priority 0 (candidate is Priority 2).
        # Balance becomes 800 before purchase runs, making 400 safe today!
        salary = CanonicalEvent(
            event_id="sal_same_day", user_id=self.user_id, source_row=1, effective_date=self.request_date,
            direction=Direction.INFLOW, direction_original="INFLOW", amount_original=Decimal("500.00"),
            currency_original="USD", amount_home=Decimal("500.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=self.request_date, status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_INFLOW, event_type="income",
            category="salary", description="Salary same day", flexibility="fixed",
            minimum_allowed_amount_original=None, minimum_allowed_amount_home=None, linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN, is_unresolved=False, unresolved_reason=None,
            evidence_chain=(), applied_actions=(),
        )
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[salary], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [salary], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("400.00"))
        self.assertTrue(cert.is_full_payment_safe_today)

    def test_case_b_pending_debit_before_candidate_purchase(self) -> None:
        """Case B: Pending debit occurs on request date. Pending hold (Priority 1) reserves cash before purchase."""
        # Initial cash 500, floor 100 => Headroom 400.
        # But pending hold of 150 occurs on Jan 1. Priority 1 reserves cash to 350 before candidate purchase.
        # Max safe amount today is 350 - 100 = 250.
        prof = FinancialProfile(
            user_id=self.user_id, home_currency="USD", current_available_balance=Decimal("500.00"),
            minimum_balance_to_keep=Decimal("100.00"), financial_priorities=(), expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(), expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",), max_installment_months=None,
        )
        pend = CanonicalEvent(
            event_id="pend_same_day", user_id=self.user_id, source_row=1, effective_date=self.request_date,
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("150.00"),
            currency_original="USD", amount_home=Decimal("150.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=self.request_date, status="pending",
            is_cash_event=True, cash_impact_type=CashImpactType.PENDING_DEBIT_RESERVED, event_type="expense",
            category="shopping", description="Pending same day", flexibility="fixed",
            minimum_allowed_amount_original=None, minimum_allowed_amount_home=None, linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN, is_unresolved=False, unresolved_reason=None,
            evidence_chain=(), applied_actions=(),
        )
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[pend], profile=prof)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [pend], [], prof)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("250.00"))

    def test_case_c_scheduled_obligation_same_day(self) -> None:
        """Case C: Scheduled obligation occurs on request date. Deterministic tie-breaker."""
        prof = FinancialProfile(
            user_id=self.user_id, home_currency="USD", current_available_balance=Decimal("500.00"),
            minimum_balance_to_keep=Decimal("100.00"), financial_priorities=(), expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(), expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",), max_installment_months=None,
        )
        ob = CanonicalEvent(
            event_id="bill_same_day", user_id=self.user_id, source_row=1, effective_date=self.request_date,
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("200.00"),
            currency_original="USD", amount_home=Decimal("200.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=self.request_date, status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="bills", description="Bill same day", flexibility="fixed",
            minimum_allowed_amount_original=None, minimum_allowed_amount_home=None, linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN, is_unresolved=False, unresolved_reason=None,
            evidence_chain=(), applied_actions=(),
        )
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[ob], profile=prof)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [ob], [], prof)
        # 500 - 200 = 300 closing cash. Safety floor 100 => safe amount is 200.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("200.00"))

    def test_case_d_multiple_same_day_events(self) -> None:
        """Case D: Salary + pending hold + obligation + candidate purchase all on request date."""
        sal = CanonicalEvent(
            event_id="sal_d", user_id=self.user_id, source_row=1, effective_date=self.request_date,
            direction=Direction.INFLOW, direction_original="INFLOW", amount_original=Decimal("600.00"),
            currency_original="USD", amount_home=Decimal("600.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=self.request_date, status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_INFLOW, event_type="income",
            category="salary", description="Salary", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        pend = CanonicalEvent(
            event_id="pend_d", user_id=self.user_id, source_row=2, effective_date=self.request_date,
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("100.00"),
            currency_original="USD", amount_home=Decimal("100.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=self.request_date, status="pending",
            is_cash_event=True, cash_impact_type=CashImpactType.PENDING_DEBIT_RESERVED, event_type="expense",
            category="groceries", description="Hold", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        bill = CanonicalEvent(
            event_id="bill_d", user_id=self.user_id, source_row=3, effective_date=self.request_date,
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("200.00"),
            currency_original="USD", amount_home=Decimal("200.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=self.request_date, status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="bills", description="Bill", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )
        # Initial available: 300.
        # + 600 salary (inflow, p0) => 900
        # - 100 pending (hold, p1) => 800
        # - 200 bill (outflow, p2) => 600
        # Floor: 100 => Headroom = 500.
        # Requested: 400 => full 400 is safe!
        baseline = simulate_user(self.user_id, self.sim_start, self.sim_end, canonical_events=[sal, pend, bill], profile=self.profile)
        cert = evaluate_request_safe_to_pay(self.request, baseline, [sal, pend, bill], [], self.profile)
        self.assertEqual(cert.amount_safe_to_pay, Decimal("400.00"))
        self.assertTrue(cert.is_full_payment_safe_today)


class TestCandidateEventTranslationInvariantFormal(unittest.TestCase):
    """Formal test verifying that candidate purchase X vs Y changes ONLY cash states after candidate."""

    def test_formal_event_stream_isolation(self) -> None:
        uid = "u_iso"
        req_d = date(2025, 1, 1)
        prof = FinancialProfile(
            user_id=uid, home_currency="USD", current_available_balance=Decimal("1000.00"),
            minimum_balance_to_keep=Decimal("200.00"), financial_priorities=(), expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(), expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",), max_installment_months=None,
        )
        req = FinancialRequest(
            request_id="req_iso", user_id=uid, request_date=req_d, request_type="purchase",
            requested_amount=Decimal("500.00"), desired_completion_date=req_d + timedelta(days=30),
            allows_partial_payment=True, request_text="Isolation test",
        )

        sal = FutureEvent(
            event_id="s_iso", user_id=uid, effective_date=date(2025, 1, 15),
            direction=Direction.INFLOW, amount_home=Decimal("1500.00"), currency="USD",
            category="salary", event_type="income", description="Salary", series_id="s1",
            frequency=RecurrenceFrequency.MONTHLY, is_forecast=True,
        )

        ev_100 = make_candidate_purchase_event(req, Decimal("100.00"), req_d, "USD", event_id="cand_purchase")
        ev_300 = make_candidate_purchase_event(req, Decimal("300.00"), req_d, "USD", event_id="cand_purchase")

        res_100 = simulate_user(uid, req_d, req_d + timedelta(days=90), future_events=[sal], profile=prof, additional_events=[ev_100])
        res_300 = simulate_user(uid, req_d, req_d + timedelta(days=90), future_events=[sal], profile=prof, additional_events=[ev_300])

        # Verify event stream counts and event_ids are identical
        self.assertEqual(len(res_100.projected_events), len(res_300.projected_events))
        for e1, e2 in zip(res_100.projected_events, res_300.projected_events):
            self.assertEqual(e1.event_id, e2.event_id)
            self.assertEqual(e1.effective_date, e2.effective_date)
            self.assertEqual(e1.direction, e2.direction)
            if e1.event_id != "cand_purchase":
                self.assertEqual(e1.amount, e2.amount)

        # Verify exact cash difference of 200.00 on every post-candidate transition
        delta = Decimal("200.00")
        for t1, t2 in zip(res_100.event_transitions, res_300.event_transitions):
            self.assertEqual(t1.event_id, t2.event_id)
            self.assertEqual(t1.date, t2.date)
            self.assertEqual(t2.available_after, t1.available_after - delta)


class TestCertificateCorrectnessAudit(unittest.TestCase):
    """Audit of SafeToPayCertificate fields and consistency invariants."""

    def test_certificate_fields_consistency(self) -> None:
        uid = "u_cert"
        req_d = date(2025, 1, 1)
        prof = FinancialProfile(
            user_id=uid, home_currency="USD", current_available_balance=Decimal("600.00"),
            minimum_balance_to_keep=Decimal("150.00"), financial_priorities=(), expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(), expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",), max_installment_months=None,
        )
        req = FinancialRequest(
            request_id="req_cert", user_id=uid, request_date=req_d, request_type="purchase",
            requested_amount=Decimal("500.00"), desired_completion_date=req_d + timedelta(days=30),
            allows_partial_payment=True, request_text="Cert test",
        )
        exp = CanonicalEvent(
            event_id="exp_cert", user_id=uid, source_row=1, effective_date=date(2025, 1, 10),
            direction=Direction.OUTFLOW, direction_original="OUTFLOW", amount_original=Decimal("200.00"),
            currency_original="USD", amount_home=Decimal("200.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=date(2025, 1, 10), status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SETTLED_OUTFLOW, event_type="expense",
            category="rent", description="Rent", flexibility="fixed", minimum_allowed_amount_original=None,
            minimum_allowed_amount_home=None, linked_event_id=None, recurrence_type=RecurrenceClassification.UNKNOWN,
            is_unresolved=False, unresolved_reason=None, evidence_chain=(), applied_actions=(),
        )

        baseline = simulate_user(uid, req_d, req_d + timedelta(days=90), canonical_events=[exp], profile=prof)
        cert = evaluate_request_safe_to_pay(req, baseline, [exp], [], prof)

        # Headroom: 400 - 150 = 250. Safe amount: 250.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("250.00"))
        self.assertEqual(cert.safety_floor, Decimal("150.00"))
        self.assertEqual(cert.safety_floor_margin, Decimal("0.00"))
        self.assertEqual(cert.minimum_available_cash_after_purchase, Decimal("150.00"))
        self.assertEqual(cert.limiting_date, date(2025, 1, 10))
        self.assertIsNotNone(cert.reason_if_unsafe)
        self.assertIn("Full payment would breach safety floor", cert.reason_if_unsafe)


class TestSameDayOrderingAndCandidateSemantics(unittest.TestCase):
    """Forensic verification of intra-day event ordering and candidate purchase semantics."""

    def setUp(self) -> None:
        self.uid = "u_forensic_order"
        self.req_d = date(2025, 1, 1)
        self.prof = FinancialProfile(
            user_id=self.uid, home_currency="USD", current_available_balance=Decimal("1000.00"),
            minimum_balance_to_keep=Decimal("200.00"), financial_priorities=(), expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(), expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",), max_installment_months=None,
        )
        self.req = FinancialRequest(
            request_id="req_order", user_id=self.uid, request_date=self.req_d, request_type="purchase",
            requested_amount=Decimal("500.00"), desired_completion_date=self.req_d + timedelta(days=30),
            allows_partial_payment=True, request_text="Order test",
        )

    def _make_ce(self, eid: str, direction: Direction, amt: Decimal, impact: CashImpactType) -> CanonicalEvent:
        return CanonicalEvent(
            event_id=eid, user_id=self.uid, source_row=1, effective_date=self.req_d,
            direction=direction, direction_original=direction.value, amount_original=amt,
            currency_original="USD", amount_home=amt, home_currency="USD", exchange_rate_used=Decimal("1"),
            exchange_rate_date=self.req_d, status="settled" if impact != CashImpactType.PENDING_DEBIT_RESERVED else "pending",
            is_cash_event=True, cash_impact_type=impact, event_type="income" if direction == Direction.INFLOW else "expense",
            category="salary" if direction == Direction.INFLOW else "bills", description=eid, flexibility="fixed",
            minimum_allowed_amount_original=None, minimum_allowed_amount_home=None, linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN, is_unresolved=False, unresolved_reason=None,
            evidence_chain=(), applied_actions=(),
        )

    def test_case_A_salary_plus_candidate(self) -> None:
        """A. salary + candidate: Salary executes first (P0), then candidate purchase (P3)."""
        sal = self._make_ce("sal_1", Direction.INFLOW, Decimal("400.00"), CashImpactType.SCHEDULED_INFLOW)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[sal], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [sal], [], self.prof)

        # Opening 1000 + 400 salary = 1400. Floor 200 -> Headroom 1200. Requested 500 is fully safe.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("500.00"))
        self.assertTrue(cert.is_full_payment_safe_today)

        cand = make_candidate_purchase_event(self.req, Decimal("500.00"), self.req_d, "USD")
        sim = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[sal], profile=self.prof, additional_events=[cand])
        trans = [t for t in sim.event_transitions if t.date == self.req_d]
        self.assertEqual([t.event_id for t in trans], ["sal_1", cand.event_id])
        self.assertEqual(trans[0].available_after, Decimal("1400.00"))
        self.assertEqual(trans[1].available_after, Decimal("900.00"))

    def test_case_B_pending_hold_plus_candidate(self) -> None:
        """B. pending hold + candidate: Hold reserves cash first (P1), then candidate purchase (P3)."""
        hold = self._make_ce("hold_1", Direction.OUTFLOW, Decimal("300.00"), CashImpactType.PENDING_DEBIT_RESERVED)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[hold], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [hold], [], self.prof)

        # Opening 1000 - 300 hold = 700. Floor 200 -> Headroom 500. Requested 500 is exactly safe.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("500.00"))
        self.assertTrue(cert.is_full_payment_safe_today)

        cand = make_candidate_purchase_event(self.req, Decimal("500.00"), self.req_d, "USD")
        sim = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[hold], profile=self.prof, additional_events=[cand])
        trans = [t for t in sim.event_transitions if t.date == self.req_d]
        self.assertEqual([t.event_id for t in trans], ["hold_1", cand.event_id])
        self.assertEqual(trans[0].available_after, Decimal("700.00"))
        self.assertEqual(trans[1].available_after, Decimal("200.00"))

    def test_case_C_scheduled_obligation_plus_candidate(self) -> None:
        """C. scheduled obligation + candidate: Scheduled obligation executes first (P2), then candidate purchase (P3)."""
        ob = self._make_ce("ob_1", Direction.OUTFLOW, Decimal("400.00"), CashImpactType.SCHEDULED_OUTFLOW)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[ob], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [ob], [], self.prof)

        # Opening 1000 - 400 ob = 600. Floor 200 -> Headroom 400. Safe amount = 400.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("400.00"))
        self.assertFalse(cert.is_full_payment_safe_today)

        cand = make_candidate_purchase_event(self.req, Decimal("400.00"), self.req_d, "USD")
        sim = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[ob], profile=self.prof, additional_events=[cand])
        trans = [t for t in sim.event_transitions if t.date == self.req_d]
        self.assertEqual([t.event_id for t in trans], ["ob_1", cand.event_id])
        self.assertEqual(trans[0].available_after, Decimal("600.00"))
        self.assertEqual(trans[1].available_after, Decimal("200.00"))

    def test_case_D_salary_plus_obligation_plus_candidate(self) -> None:
        """D. salary + obligation + candidate: Ordering is strictly Salary (P0) -> Obligation (P2) -> Candidate (P3)."""
        sal = self._make_ce("sal_1", Direction.INFLOW, Decimal("500.00"), CashImpactType.SCHEDULED_INFLOW)
        ob = self._make_ce("ob_1", Direction.OUTFLOW, Decimal("600.00"), CashImpactType.SCHEDULED_OUTFLOW)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[sal, ob], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [sal, ob], [], self.prof)

        # Opening 1000 + 500 - 600 = 900. Floor 200 -> Headroom 700. Requested 500 safe.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("500.00"))

        cand = make_candidate_purchase_event(self.req, Decimal("500.00"), self.req_d, "USD")
        sim = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[sal, ob], profile=self.prof, additional_events=[cand])
        trans = [t for t in sim.event_transitions if t.date == self.req_d]
        self.assertEqual([t.event_id for t in trans], ["sal_1", "ob_1", cand.event_id])
        self.assertEqual(trans[0].available_after, Decimal("1500.00"))
        self.assertEqual(trans[1].available_after, Decimal("900.00"))
        self.assertEqual(trans[2].available_after, Decimal("400.00"))

    def test_case_E_pending_hold_plus_obligation_plus_candidate(self) -> None:
        """E. pending hold + obligation + candidate: Hold (P1) -> Obligation (P2) -> Candidate (P3)."""
        hold = self._make_ce("hold_1", Direction.OUTFLOW, Decimal("200.00"), CashImpactType.PENDING_DEBIT_RESERVED)
        ob = self._make_ce("ob_1", Direction.OUTFLOW, Decimal("300.00"), CashImpactType.SCHEDULED_OUTFLOW)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[hold, ob], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [hold, ob], [], self.prof)

        # Opening 1000 - 200 hold - 300 ob = 500. Floor 200 -> Headroom 300. Safe amount = 300.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("300.00"))

        cand = make_candidate_purchase_event(self.req, Decimal("300.00"), self.req_d, "USD")
        sim = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[hold, ob], profile=self.prof, additional_events=[cand])
        trans = [t for t in sim.event_transitions if t.date == self.req_d]
        self.assertEqual([t.event_id for t in trans], ["hold_1", "ob_1", cand.event_id])
        self.assertEqual(trans[0].available_after, Decimal("800.00"))
        self.assertEqual(trans[1].available_after, Decimal("500.00"))
        self.assertEqual(trans[2].available_after, Decimal("200.00"))

    def test_case_F_multiple_scheduled_obligations_plus_candidate(self) -> None:
        """F. multiple scheduled obligations + candidate: All obligations execute before candidate."""
        ob1 = self._make_ce("ob_1", Direction.OUTFLOW, Decimal("150.00"), CashImpactType.SCHEDULED_OUTFLOW)
        ob2 = self._make_ce("ob_2", Direction.OUTFLOW, Decimal("250.00"), CashImpactType.SCHEDULED_OUTFLOW)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[ob1, ob2], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [ob1, ob2], [], self.prof)

        # 1000 - 400 = 600. Floor 200 -> Headroom 400.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("400.00"))

        cand = make_candidate_purchase_event(self.req, Decimal("400.00"), self.req_d, "USD")
        sim = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[ob1, ob2], profile=self.prof, additional_events=[cand])
        trans = [t for t in sim.event_transitions if t.date == self.req_d]
        self.assertEqual([t.event_id for t in trans], ["ob_1", "ob_2", cand.event_id])

    def test_case_G_multiple_same_day_events_plus_candidate(self) -> None:
        """G. full spectrum: Salary (P0) -> Hold (P1) -> Obligation (P2) -> Candidate (P3)."""
        sal = self._make_ce("sal_1", Direction.INFLOW, Decimal("500.00"), CashImpactType.SCHEDULED_INFLOW)
        hold = self._make_ce("hold_1", Direction.OUTFLOW, Decimal("100.00"), CashImpactType.PENDING_DEBIT_RESERVED)
        ob = self._make_ce("ob_1", Direction.OUTFLOW, Decimal("300.00"), CashImpactType.SCHEDULED_OUTFLOW)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[sal, hold, ob], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [sal, hold, ob], [], self.prof)

        # 1000 + 500 - 100 - 300 = 1100. Floor 200 -> Headroom 900. Requested 500 safe.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("500.00"))

        cand = make_candidate_purchase_event(self.req, Decimal("500.00"), self.req_d, "USD")
        sim = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[sal, hold, ob], profile=self.prof, additional_events=[cand])
        trans = [t for t in sim.event_transitions if t.date == self.req_d]
        self.assertEqual([t.event_id for t in trans], ["sal_1", "hold_1", "ob_1", cand.event_id])

    def test_case_H_tight_safety_floor_boundary(self) -> None:
        """H. tight safety floor boundary: Headroom is very tight (e.g. $10.00)."""
        ob = self._make_ce("ob_1", Direction.OUTFLOW, Decimal("790.00"), CashImpactType.SCHEDULED_OUTFLOW)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[ob], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [ob], [], self.prof)

        # 1000 - 790 = 210. Floor 200 -> Headroom 10.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("10.00"))
        self.assertEqual(cert.safety_floor_margin, Decimal("0.00"))

    def test_case_I_exact_equality_with_safety_floor(self) -> None:
        """I. exact equality: Available cash exactly equals safety floor."""
        ob = self._make_ce("ob_1", Direction.OUTFLOW, Decimal("800.00"), CashImpactType.SCHEDULED_OUTFLOW)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[ob], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [ob], [], self.prof)

        # 1000 - 800 = 200. Floor 200 -> Headroom 0. Safe amount = 0.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("0.00"))
        self.assertFalse(cert.is_full_payment_safe_today)

    def test_case_J_one_cent_above_safety_floor(self) -> None:
        """J. one-cent above: Headroom is exactly $0.01."""
        ob = self._make_ce("ob_1", Direction.OUTFLOW, Decimal("799.99"), CashImpactType.SCHEDULED_OUTFLOW)
        base = simulate_user(self.uid, self.req_d, self.req_d + timedelta(days=90), canonical_events=[ob], profile=self.prof)
        cert = evaluate_request_safe_to_pay(self.req, base, [ob], [], self.prof)

        # 1000 - 799.99 = 200.01. Floor 200 -> Headroom 0.01.
        self.assertEqual(cert.amount_safe_to_pay, Decimal("0.01"))

    def test_case_K_candidate_amount_zero(self) -> None:
        """K. candidate amount = 0: Zero purchase is always safe if baseline is safe."""
        cand_zero = make_candidate_purchase_event(self.req, Decimal("0.00"), self.req_d, "USD")
        safe, res = is_purchase_safe(self.uid, self.req_d, self.req_d + timedelta(days=90), [], [], self.prof, cand_zero)
        self.assertTrue(safe)
        self.assertEqual(res.minimum_projected_available_cash, Decimal("1000.00"))

    def test_case_L_candidate_amount_requested_amount(self) -> None:
        """L. candidate amount = requested amount: Evaluates full amount safety predicate."""
        cand_full = make_candidate_purchase_event(self.req, Decimal("500.00"), self.req_d, "USD")
        safe, res = is_purchase_safe(self.uid, self.req_d, self.req_d + timedelta(days=90), [], [], self.prof, cand_full)
        self.assertTrue(safe)
        self.assertEqual(res.minimum_projected_available_cash, Decimal("500.00"))


class TestCandidateOrderIndependence(unittest.TestCase):
    """Exhaustive verification that candidate purchase event ID never alters safe-to-pay results."""

    def test_event_id_order_independence(self) -> None:
        uid = "u_order_indep"
        req_d = date(2025, 1, 1)
        prof = FinancialProfile(
            user_id=uid, home_currency="USD", current_available_balance=Decimal("1200.00"),
            minimum_balance_to_keep=Decimal("300.00"), financial_priorities=(), expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(), expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",), max_installment_months=None,
        )
        req = FinancialRequest(
            request_id="req_indep", user_id=uid, request_date=req_d, request_type="purchase",
            requested_amount=Decimal("600.00"), desired_completion_date=req_d + timedelta(days=30),
            allows_partial_payment=True, request_text="Indep test",
        )
        ob1 = CanonicalEvent(
            event_id="bill_electricity", user_id=uid, source_row=1, effective_date=req_d,
            direction=Direction.OUTFLOW, direction_original="outflow", amount_original=Decimal("200.00"),
            currency_original="USD", amount_home=Decimal("200.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=req_d, status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SCHEDULED_OUTFLOW, event_type="expense",
            category="utilities", description="Electricity", flexibility="fixed",
            minimum_allowed_amount_original=None, minimum_allowed_amount_home=None, linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN, is_unresolved=False, unresolved_reason=None,
            evidence_chain=(), applied_actions=(),
        )
        ob2 = CanonicalEvent(
            event_id="rent_apartment", user_id=uid, source_row=2, effective_date=req_d,
            direction=Direction.OUTFLOW, direction_original="outflow", amount_original=Decimal("300.00"),
            currency_original="USD", amount_home=Decimal("300.00"), home_currency="USD",
            exchange_rate_used=Decimal("1"), exchange_rate_date=req_d, status="settled",
            is_cash_event=True, cash_impact_type=CashImpactType.SCHEDULED_OUTFLOW, event_type="expense",
            category="rent", description="Rent", flexibility="fixed",
            minimum_allowed_amount_original=None, minimum_allowed_amount_home=None, linked_event_id=None,
            recurrence_type=RecurrenceClassification.UNKNOWN, is_unresolved=False, unresolved_reason=None,
            evidence_chain=(), applied_actions=(),
        )

        base = simulate_user(uid, req_d, req_d + timedelta(days=90), canonical_events=[ob1, ob2], profile=prof)

        # Compare safe-to-pay under different event IDs:
        # 1. default candidate event ID
        # 2. 000_lexically_first
        # 3. zzz_lexically_last
        # 4. arbitrary UUID
        results = []
        for eid in [None, "000_first_cand", "zzz_last_cand", "uuid_98374827_cand"]:
            cand = make_candidate_purchase_event(req, Decimal("400.00"), req_d, "USD", event_id=eid)
            sim = simulate_user(uid, req_d, req_d + timedelta(days=90), canonical_events=[ob1, ob2], profile=prof, additional_events=[cand])
            trans = [t for t in sim.event_transitions if t.date == req_d]
            # Scheduled obligations must always precede candidate purchase
            self.assertEqual(trans[0].event_id, "bill_electricity")
            self.assertEqual(trans[1].event_id, "rent_apartment")
            self.assertEqual(trans[2].event_id, cand.event_id)
            results.append((sim.minimum_projected_available_cash, sim.is_safety_floor_breached, sim.ending_state.available_cash))

        # All runs produce 100% identical financial outcomes
        self.assertEqual(len(set(results)), 1)


if __name__ == "__main__":
    unittest.main()


