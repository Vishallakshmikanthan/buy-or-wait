"""Comprehensive unit test suite for deterministic safe-to-pay and earliest full payment date.

Covers:
- All 25 required scenario tests from Section 19
- All 7 metamorphic property tests from Section 20
"""

from datetime import date, timedelta
from decimal import Decimal
import unittest

from code.canonical import CanonicalEvent, CashImpactType, Direction, RecurrenceClassification
from code.models import FinancialProfile, FinancialRequest
from code.recurrence import FutureEvent, RecurrenceFrequency
from code.safe_to_pay import (
    calculate_amount_safe_to_pay,
    calculate_earliest_full_payment_date,
    evaluate_request_safe_to_pay,
    is_purchase_safe,
    make_candidate_purchase_event,
)
from code.simulator import (
    SimulatedEvent,
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


if __name__ == "__main__":
    unittest.main()
