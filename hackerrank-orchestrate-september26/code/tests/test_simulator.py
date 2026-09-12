"""Comprehensive metamorphic and invariant test suite for the deterministic simulation engine.

Tests all 16 core properties (A through P):
A. Determinism: Same input twice => byte-for-byte/identical result.
B. Zero-event invariance: No events => balance remains unchanged.
C. Conservation: Ending balance equals initial balance + inflows - outflows, adjusted for reservation.
D. Order invariance: Reordering input events does not change result.
E. Boundary: Events outside simulation range do not affect the simulation.
F. Pending debit: Pending debit reserves once and is never double-counted.
G. Pending credit: Pending credit contributes zero cash.
H. Failed/cancelled: No cash impact.
I. Unresolved: No cash impact.
J. Explicit scheduled event: Appears exactly once.
K. Forecast event: Appears exactly once unless explicitly suppressed upstream.
L. Duplicate lineage: Same real-world event represented by multiple statuses cannot double-count.
M. Safety floor: Breach detection is deterministic.
N. Decimal: All money operations remain Decimal.
O. Same-day ordering: Changing input order does not change the same-day result.
P. Arbitrary window: Simulation works for windows other than 90 days.
"""

from datetime import date, timedelta
from decimal import Decimal
import unittest

from code.canonical import CanonicalEvent, CashImpactType, Direction, RecurrenceClassification
from code.models import FinancialProfile
from code.recurrence import FutureEvent, RecurrenceFrequency
from code.simulator import (
    DailySnapshot,
    EventTransition,
    FinancialState,
    SimulatedEvent,
    SimulationResult,
    create_initial_state,
    simulate_user,
)


class TestSimulatorInvariants(unittest.TestCase):
    """Metamorphic and invariant test suite for simulate_user."""

    def setUp(self) -> None:
        """Create standard test fixtures."""
        self.user_id = "test_user"
        self.start_date = date(2025, 1, 1)
        self.end_date = date(2025, 3, 31)  # 90-day horizon

        self.profile = FinancialProfile(
            user_id=self.user_id,
            home_currency="USD",
            current_available_balance=Decimal("1000.00"),
            minimum_balance_to_keep=Decimal("200.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=("rent", "groceries"),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("streaming",),
            payment_methods_user_will_consider=("full_payment", "installments"),
            max_installment_months=3,
        )

        self.initial_state = create_initial_state(
            user_id=self.user_id,
            as_of_date=self.start_date,
            profile=self.profile,
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

    def test_a_determinism(self) -> None:
        """A. Determinism: Same input twice produces identical results."""
        events = [
            self._make_canonical("e1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("100.00")),
            self._make_canonical("e2", date(2025, 1, 15), Direction.INFLOW, Decimal("500.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW),
        ]
        res1 = simulate_user(self.user_id, self.start_date, self.end_date, self.initial_state, canonical_events=events, profile=self.profile)
        res2 = simulate_user(self.user_id, self.start_date, self.end_date, self.initial_state, canonical_events=events, profile=self.profile)

        self.assertEqual(res1.ending_state, res2.ending_state)
        self.assertEqual(res1.minimum_projected_available_cash, res2.minimum_projected_available_cash)
        self.assertEqual(res1.minimum_cash_date, res2.minimum_cash_date)
        self.assertEqual(res1.total_inflows, res2.total_inflows)
        self.assertEqual(res1.total_outflows, res2.total_outflows)
        self.assertEqual(len(res1.event_transitions), len(res2.event_transitions))
        self.assertEqual(len(res1.daily_snapshots), len(res2.daily_snapshots))

    def test_b_zero_event_invariance(self) -> None:
        """B. Zero-event invariance: No events => balance remains unchanged."""
        res = simulate_user(self.user_id, self.start_date, self.end_date, self.initial_state, profile=self.profile)
        self.assertEqual(res.ending_state.available_cash, self.initial_state.available_cash)
        self.assertEqual(res.ending_state.projected_balance, self.initial_state.projected_balance)
        self.assertEqual(res.minimum_projected_available_cash, self.initial_state.available_cash)
        self.assertEqual(res.total_inflows, Decimal("0"))
        self.assertEqual(res.total_outflows, Decimal("0"))
        self.assertEqual(len(res.event_transitions), 0)
        self.assertEqual(len(res.daily_snapshots), 90)

    def test_c_conservation(self) -> None:
        """C. Conservation: Ending balance equals initial balance + inflows - outflows."""
        events = [
            self._make_canonical("e1", date(2025, 1, 5), Direction.OUTFLOW, Decimal("150.00")),
            self._make_canonical("e2", date(2025, 1, 20), Direction.INFLOW, Decimal("600.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW),
            self._make_canonical("e3", date(2025, 2, 1), Direction.OUTFLOW, Decimal("50.00")),
        ]
        res = simulate_user(self.user_id, self.start_date, self.end_date, self.initial_state, canonical_events=events, profile=self.profile)
        expected_balance = self.initial_state.available_cash + Decimal("600.00") - Decimal("200.00")
        self.assertEqual(res.ending_state.available_cash, expected_balance)
        self.assertEqual(res.ending_state.projected_balance, expected_balance)
        self.assertEqual(res.total_inflows, Decimal("600.00"))
        self.assertEqual(res.total_outflows, Decimal("200.00"))

    def test_d_order_invariance(self) -> None:
        """D. Order invariance: Reordering input events does not change the result."""
        e1 = self._make_canonical("e1", date(2025, 1, 5), Direction.OUTFLOW, Decimal("50.00"))
        e2 = self._make_canonical("e2", date(2025, 1, 10), Direction.INFLOW, Decimal("200.00"), status="scheduled", impact_type=CashImpactType.SCHEDULED_INFLOW)
        e3 = self._make_canonical("e3", date(2025, 1, 15), Direction.OUTFLOW, Decimal("100.00"))

        res_forward = simulate_user(self.user_id, self.start_date, self.end_date, self.initial_state, canonical_events=[e1, e2, e3], profile=self.profile)
        res_reverse = simulate_user(self.user_id, self.start_date, self.end_date, self.initial_state, canonical_events=[e3, e1, e2], profile=self.profile)

        self.assertEqual(res_forward.ending_state.available_cash, res_reverse.ending_state.available_cash)
        self.assertEqual(res_forward.minimum_projected_available_cash, res_reverse.minimum_projected_available_cash)
        self.assertEqual(res_forward.minimum_cash_date, res_reverse.minimum_cash_date)

    def test_e_boundary(self) -> None:
        """E. Boundary: Events outside simulation range do not affect the simulation."""
        inside_start = self._make_canonical("e_start", date(2025, 1, 1), Direction.OUTFLOW, Decimal("10.00"))
        inside_end = self._make_canonical("e_end", date(2025, 3, 31), Direction.OUTFLOW, Decimal("20.00"))
        before = self._make_canonical("e_before", date(2024, 12, 31), Direction.OUTFLOW, Decimal("500.00"))
        after = self._make_canonical("e_after", date(2025, 4, 1), Direction.OUTFLOW, Decimal("500.00"))

        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            canonical_events=[before, inside_start, inside_end, after],
            profile=self.profile,
        )

        self.assertEqual(len(res.projected_events), 2)
        self.assertEqual(res.ending_state.available_cash, Decimal("1000.00") - Decimal("30.00"))
        self.assertEqual(res.total_outflows, Decimal("30.00"))

    def test_f_pending_debit_reservation(self) -> None:
        """F. Pending debit reserves once and settles without double counting."""
        # Pending debit arrives on Jan 5, reserving $100
        pending = self._make_canonical(
            "p1",
            date(2025, 1, 5),
            Direction.OUTFLOW,
            Decimal("100.00"),
            status="pending",
            impact_type=CashImpactType.PENDING_DEBIT_RESERVED,
        )
        # Settled debit arrives on Jan 10 for the same $100, linked to p1
        settled = self._make_canonical(
            "s1",
            date(2025, 1, 10),
            Direction.OUTFLOW,
            Decimal("100.00"),
            status="settled",
            impact_type=CashImpactType.SETTLED_OUTFLOW,
            linked_id="p1",
        )

        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            canonical_events=[pending, settled],
            profile=self.profile,
        )

        # On Jan 5: available drops to 900 (reserved=100, ledger=1000)
        # On Jan 10: settled drops ledger to 900, reservation cleared to 0 => available stays 900!
        self.assertEqual(res.ending_state.available_cash, Decimal("900.00"))
        self.assertEqual(res.ending_state.projected_balance, Decimal("900.00"))
        self.assertEqual(res.ending_state.reserved_pending, Decimal("0.00"))
        self.assertEqual(res.minimum_projected_available_cash, Decimal("900.00"))

    def test_g_pending_credit(self) -> None:
        """G. Pending credit contributes zero cash."""
        pending_credit = self._make_canonical(
            "pc1",
            date(2025, 1, 10),
            Direction.NON_CASH,
            Decimal("500.00"),
            status="pending",
            impact_type=CashImpactType.PENDING_CREDIT_IGNORED,
            is_cash=False,
        )
        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            canonical_events=[pending_credit],
            profile=self.profile,
        )
        self.assertEqual(res.ending_state.available_cash, Decimal("1000.00"))
        self.assertEqual(res.total_inflows, Decimal("0"))

    def test_h_failed_cancelled(self) -> None:
        """H. Failed/cancelled events have zero cash impact."""
        failed = self._make_canonical(
            "f1",
            date(2025, 1, 5),
            Direction.NON_CASH,
            Decimal("300.00"),
            status="failed",
            impact_type=CashImpactType.FAILED_IGNORED,
            is_cash=False,
        )
        cancelled = self._make_canonical(
            "c1",
            date(2025, 1, 6),
            Direction.NON_CASH,
            Decimal("200.00"),
            status="cancelled",
            impact_type=CashImpactType.CANCELLED_IGNORED,
            is_cash=False,
        )
        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            canonical_events=[failed, cancelled],
            profile=self.profile,
        )
        self.assertEqual(res.ending_state.available_cash, Decimal("1000.00"))
        self.assertEqual(res.total_inflows, Decimal("0"))
        self.assertEqual(res.total_outflows, Decimal("0"))

    def test_i_unresolved(self) -> None:
        """I. Unresolved events have strictly zero cash impact."""
        unresolved = self._make_canonical(
            "u1",
            date(2025, 1, 10),
            Direction.OUTFLOW,
            Decimal("400.00"),
            status="pending",
            impact_type=CashImpactType.PENDING_DEBIT_RESERVED,
            is_cash=False,
            is_unresolved=True,
        )
        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            canonical_events=[unresolved],
            profile=self.profile,
        )
        self.assertEqual(res.ending_state.available_cash, Decimal("1000.00"))
        self.assertEqual(res.ending_state.reserved_pending, Decimal("0"))

    def test_j_explicit_scheduled_event(self) -> None:
        """J. Explicit scheduled event appears exactly once."""
        sched = self._make_canonical(
            "sched_sal",
            date(2025, 1, 15),
            Direction.INFLOW,
            Decimal("2500.00"),
            status="scheduled",
            impact_type=CashImpactType.SCHEDULED_INFLOW,
            category="salary",
        )
        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            canonical_events=[sched],
            profile=self.profile,
        )
        self.assertEqual(len(res.projected_events), 1)
        self.assertEqual(res.ending_state.available_cash, Decimal("3500.00"))
        self.assertEqual(res.total_inflows, Decimal("2500.00"))

    def test_k_forecast_event(self) -> None:
        """K. Forecast event appears and correctly adjusts balance."""
        forecast = self._make_future("fut_1", date(2025, 1, 20), Direction.OUTFLOW, Decimal("150.00"), category="utilities")
        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            future_events=[forecast],
            profile=self.profile,
        )
        self.assertEqual(len(res.projected_events), 1)
        self.assertEqual(res.ending_state.available_cash, Decimal("850.00"))
        self.assertEqual(res.total_outflows, Decimal("150.00"))

    def test_l_duplicate_lineage(self) -> None:
        """L. Duplicate event IDs cannot double-count."""
        e1 = self._make_canonical("dup_1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("100.00"))
        e2 = self._make_canonical("dup_1", date(2025, 1, 10), Direction.OUTFLOW, Decimal("100.00"))

        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            canonical_events=[e1, e2],
            profile=self.profile,
        )
        self.assertEqual(len(res.projected_events), 1)
        self.assertEqual(res.ending_state.available_cash, Decimal("900.00"))

    def test_m_safety_floor(self) -> None:
        """M. Safety floor breach detection is deterministic."""
        # Floor is 200.00. Balance starts at 1000.00.
        # Expense of 850.00 drops balance to 150.00 (breach of 50.00).
        expense = self._make_canonical("big_exp", date(2025, 1, 10), Direction.OUTFLOW, Decimal("850.00"))
        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            canonical_events=[expense],
            profile=self.profile,
        )
        self.assertTrue(res.is_safety_floor_breached)
        self.assertEqual(len(res.safety_floor_breaches), 1)
        breach = res.safety_floor_breaches[0]
        self.assertEqual(breach.date, date(2025, 1, 10))
        self.assertEqual(breach.available_cash, Decimal("150.00"))
        self.assertEqual(breach.shortfall, Decimal("50.00"))

    def test_n_decimal_precision(self) -> None:
        """N. All operations strictly preserve Decimal precision with no floating-point distortion."""
        cents_1 = self._make_canonical("c1", date(2025, 1, 5), Direction.OUTFLOW, Decimal("0.10"))
        cents_2 = self._make_canonical("c2", date(2025, 1, 6), Direction.OUTFLOW, Decimal("0.20"))
        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            self.initial_state,
            canonical_events=[cents_1, cents_2],
            profile=self.profile,
        )
        self.assertIsInstance(res.ending_state.available_cash, Decimal)
        self.assertEqual(res.ending_state.available_cash, Decimal("999.70"))
        self.assertEqual(res.total_outflows, Decimal("0.30"))

    def test_o_same_day_ordering(self) -> None:
        """O. Inflows clear before outflows on the same day, preventing artificial intra-day breach."""
        # Initial available: $200 (safety floor: $200).
        # On Jan 10:
        # - Salary inflow of $1000 arrives
        # - Rent outflow of $500 is paid
        # If inflow runs first: 200 + 1000 = 1200 -> 1200 - 500 = 700 (never drops below 200).
        # If outflow ran first: 200 - 500 = -300 (artificial breach!).
        state = FinancialState(
            user_id=self.user_id,
            date=self.start_date,
            projected_balance=Decimal("200.00"),
            reserved_pending=Decimal("0.00"),
            available_cash=Decimal("200.00"),
            cumulative_inflows=Decimal("0.00"),
            cumulative_outflows=Decimal("0.00"),
            known_scheduled_obligations=Decimal("0.00"),
            safety_floor=Decimal("200.00"),
            is_safety_floor_breached=False,
        )

        inflow = self._make_canonical(
            "sal",
            date(2025, 1, 10),
            Direction.INFLOW,
            Decimal("1000.00"),
            status="scheduled",
            impact_type=CashImpactType.SCHEDULED_INFLOW,
        )
        outflow = self._make_canonical(
            "rent",
            date(2025, 1, 10),
            Direction.OUTFLOW,
            Decimal("500.00"),
        )

        # Feed outflow first in input list
        res = simulate_user(
            self.user_id,
            self.start_date,
            self.end_date,
            state,
            canonical_events=[outflow, inflow],
            profile=self.profile,
        )

        # Inflow must execute first
        self.assertEqual(res.event_transitions[0].event_id, "sal")
        self.assertEqual(res.event_transitions[1].event_id, "rent")
        self.assertFalse(res.is_safety_floor_breached)
        self.assertEqual(res.minimum_projected_available_cash, Decimal("200.00"))

    def test_p_arbitrary_window(self) -> None:
        """P. Simulation works correctly for arbitrary windows (e.g. 14 days, 180 days)."""
        # 14-day window
        short_start = date(2025, 1, 1)
        short_end = date(2025, 1, 14)
        e_inside = self._make_canonical("e_short", date(2025, 1, 5), Direction.OUTFLOW, Decimal("50.00"))
        e_outside = self._make_canonical("e_long", date(2025, 1, 25), Direction.OUTFLOW, Decimal("50.00"))

        res_short = simulate_user(
            self.user_id,
            short_start,
            short_end,
            self.initial_state,
            canonical_events=[e_inside, e_outside],
            profile=self.profile,
        )
        self.assertEqual(len(res_short.daily_snapshots), 14)
        self.assertEqual(len(res_short.projected_events), 1)
        self.assertEqual(res_short.ending_state.available_cash, Decimal("950.00"))


if __name__ == "__main__":
    unittest.main()
