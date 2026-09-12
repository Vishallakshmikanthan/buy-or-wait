"""Deterministic cash-flow simulation engine for Buy or Wait? financial agent.

Provides a unified, immutable financial state representation and a 90-day (or arbitrary window)
simulation engine that strictly adheres to cash-accounting rules:
- Exclusively uses Decimal for monetary calculations.
- Distinguishes ledger balance, reserved pending debits, and available cash.
- Deterministic event ordering on same-day occurrences (Inflows -> Holds -> Debits -> Non-cash -> event_id).
- Complete double-counting prevention across pending/settled lineages and explicit/forecast duplicates.
- Daily snapshots and per-event transitions with provenance preservation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from code.canonical import CanonicalEvent, CashImpactType, Direction
from code.models import FinancialProfile
from code.recurrence import FutureEvent


class EventPriority(int, Enum):
    """Deterministic priority for same-day event application."""
    INFLOW = 0       # Settled/scheduled/forecast credits arrive first so funds are accessible
    PENDING_HOLD = 1 # Pending debit authorizations reserve cash
    OUTFLOW = 2      # Settled debits, scheduled debits, and forecast expenses execute
    NON_CASH = 3     # Ignored, failed, cancelled, or unrealized records (0 cash impact)


@dataclass(frozen=True)
class SimulatedEvent:
    """Unified internal representation of an event evaluated in the simulation timeline."""
    event_id: str
    user_id: str
    effective_date: date
    direction: Direction
    amount: Decimal
    currency: str
    category: str
    event_type: str
    description: str
    cash_impact_type: CashImpactType
    flexibility: str = "fixed"
    minimum_allowed_amount: Optional[Decimal] = None
    is_protected: bool = False
    source_type: str = "canonical"  # "canonical_settled", "canonical_scheduled", "canonical_pending", "forecast", "action"
    series_id: Optional[str] = None
    linked_event_id: Optional[str] = None
    is_unresolved: bool = False
    is_forecast: bool = False

    @property
    def is_inflow(self) -> bool:
        return self.direction == Direction.INFLOW and self.cash_impact_type in (
            CashImpactType.SETTLED_INFLOW,
            CashImpactType.SCHEDULED_INFLOW,
        )

    @property
    def is_outflow(self) -> bool:
        return self.direction == Direction.OUTFLOW and self.cash_impact_type in (
            CashImpactType.SETTLED_OUTFLOW,
            CashImpactType.SCHEDULED_OUTFLOW,
        )

    @property
    def is_pending_debit(self) -> bool:
        return self.cash_impact_type == CashImpactType.PENDING_DEBIT_RESERVED


def simulated_event_from_canonical(event: CanonicalEvent) -> SimulatedEvent:
    """Convert a CanonicalEvent into a SimulatedEvent."""
    source_type = "canonical_other"
    if event.cash_impact_type == CashImpactType.SETTLED_INFLOW or event.cash_impact_type == CashImpactType.SETTLED_OUTFLOW:
        source_type = "canonical_settled"
    elif event.cash_impact_type == CashImpactType.SCHEDULED_INFLOW or event.cash_impact_type == CashImpactType.SCHEDULED_OUTFLOW:
        source_type = "canonical_scheduled"
    elif event.cash_impact_type == CashImpactType.PENDING_DEBIT_RESERVED:
        source_type = "canonical_pending"

    amount = event.get_usable_cash_amount()

    return SimulatedEvent(
        event_id=event.event_id,
        user_id=event.user_id,
        effective_date=event.effective_date,
        direction=event.direction,
        amount=amount,
        currency=event.home_currency,
        category=event.category,
        event_type=event.event_type,
        description=event.description,
        cash_impact_type=event.cash_impact_type,
        flexibility=event.flexibility,
        minimum_allowed_amount=event.minimum_allowed_amount_home,
        is_protected=False,  # Evaluated against profile
        source_type=source_type,
        series_id=None,
        linked_event_id=event.linked_event_id,
        is_unresolved=event.is_unresolved,
        is_forecast=False,
    )


def simulated_event_from_future(event: FutureEvent) -> SimulatedEvent:
    """Convert a recurring FutureEvent into a SimulatedEvent."""
    impact_type = CashImpactType.SCHEDULED_INFLOW if event.direction == Direction.INFLOW else CashImpactType.SCHEDULED_OUTFLOW
    return SimulatedEvent(
        event_id=event.event_id,
        user_id=event.user_id,
        effective_date=event.effective_date,
        direction=event.direction,
        amount=event.get_usable_cash_amount(),
        currency=event.currency,
        category=event.category,
        event_type=event.event_type,
        description=event.description,
        cash_impact_type=impact_type,
        flexibility=event.flexibility,
        minimum_allowed_amount=event.minimum_allowed_amount,
        is_protected=event.is_protected,
        source_type="forecast",
        series_id=event.series_id,
        linked_event_id=event.anchor_event_id if event.anchor_event_id else None,
        is_unresolved=False,
        is_forecast=True,
    )


@dataclass(frozen=True)
class FinancialState:
    """Immutable snapshot of a user's financial state at a specific point in time."""
    user_id: str
    date: date
    projected_balance: Decimal
    reserved_pending: Decimal
    available_cash: Decimal
    cumulative_inflows: Decimal
    cumulative_outflows: Decimal
    known_scheduled_obligations: Decimal
    safety_floor: Decimal
    is_safety_floor_breached: bool
    active_reservations: Tuple[Tuple[str, Decimal], ...] = ()

    def __post_init__(self) -> None:
        """Enforce strict Decimal types and consistency."""
        if not isinstance(self.projected_balance, Decimal):
            raise TypeError(f"projected_balance must be Decimal, got {type(self.projected_balance)}")
        if not isinstance(self.reserved_pending, Decimal):
            raise TypeError(f"reserved_pending must be Decimal, got {type(self.reserved_pending)}")
        if not isinstance(self.available_cash, Decimal):
            raise TypeError(f"available_cash must be Decimal, got {type(self.available_cash)}")
        if not isinstance(self.safety_floor, Decimal):
            raise TypeError(f"safety_floor must be Decimal, got {type(self.safety_floor)}")


@dataclass(frozen=True)
class EventTransition:
    """Detailed record of a single event application during simulation."""
    step: int
    date: date
    event_id: str
    event_description: str
    category: str
    direction: Direction
    amount: Decimal
    cash_impact_type: CashImpactType
    source_type: str
    balance_before: Decimal
    balance_after: Decimal
    reserved_before: Decimal
    reserved_after: Decimal
    available_before: Decimal
    available_after: Decimal
    safety_floor: Decimal
    is_safety_floor_breached: bool
    is_obligation_breached: bool


@dataclass(frozen=True)
class SafetyFloorBreach:
    """Detailed audit record of a safety floor or cash obligation breach."""
    date: date
    event_id: str
    available_cash: Decimal
    safety_floor: Decimal
    shortfall: Decimal
    breach_type: str  # "safety_floor" or "obligation_overdraft"


@dataclass(frozen=True)
class DailySnapshot:
    """State of the user's finances at the end of a calendar day."""
    date: date
    opening_available_cash: Decimal
    closing_projected_balance: Decimal
    closing_reserved_pending: Decimal
    closing_available_cash: Decimal
    lowest_available_cash: Decimal
    safety_floor: Decimal
    is_safety_floor_breached: bool
    is_obligation_breached: bool
    event_ids: Tuple[str, ...]


@dataclass(frozen=True)
class SimulationResult:
    """Comprehensive, immutable result of a deterministic cash-flow simulation."""
    user_id: str
    start_date: date
    end_date: date
    initial_state: FinancialState
    ending_state: FinancialState
    minimum_projected_available_cash: Decimal
    minimum_cash_date: date
    total_inflows: Decimal
    total_outflows: Decimal
    total_scheduled_obligations: Decimal
    safety_floor: Decimal
    is_safety_floor_breached: bool
    safety_floor_breaches: Tuple[SafetyFloorBreach, ...]
    obligation_breaches: Tuple[SafetyFloorBreach, ...]
    projected_events: Tuple[SimulatedEvent, ...]
    event_transitions: Tuple[EventTransition, ...]
    daily_snapshots: Tuple[DailySnapshot, ...]

    def get_snapshot_for_date(self, target_date: date) -> Optional[DailySnapshot]:
        """Return the daily snapshot for a specific date, if within the simulation window."""
        for snap in self.daily_snapshots:
            if snap.date == target_date:
                return snap
        return None

    def get_available_cash_on(self, target_date: date) -> Decimal:
        """Return the closing available cash on target_date, or initial if before start."""
        if target_date < self.start_date:
            return self.initial_state.available_cash
        if target_date > self.end_date:
            return self.ending_state.available_cash
        snap = self.get_snapshot_for_date(target_date)
        return snap.closing_available_cash if snap else self.ending_state.available_cash

    def is_safe_on(self, target_date: date) -> bool:
        """True if available cash meets or exceeds the safety floor on target_date."""
        return self.get_available_cash_on(target_date) >= self.safety_floor


def create_initial_state(
    user_id: str,
    as_of_date: date,
    profile: FinancialProfile,
    pending_debits: Sequence[CanonicalEvent] = (),
) -> FinancialState:
    """Deterministically construct the initial financial state.

    Anchor Semantics:
    - Authoritative balance: profile.current_available_balance as of as_of_date.
    - Safety floor: profile.minimum_balance_to_keep.
    - Initial reserved pending: If any pending debits exist with effective_date > as_of_date
      and were initiated on or before as_of_date, their reserve is tracked.
      Since profile.current_available_balance is the net *available* balance,
      ledger balance = available + reserved_pending.
    """
    initial_available = profile.current_available_balance
    safety_floor = profile.minimum_balance_to_keep

    active_res: Dict[str, Decimal] = {}
    total_reserved = Decimal("0")

    for pd in pending_debits:
        if pd.user_id == user_id and pd.cash_impact_type == CashImpactType.PENDING_DEBIT_RESERVED:
            amt = pd.get_usable_cash_amount()
            if amt > Decimal("0"):
                active_res[pd.event_id] = amt
                total_reserved += amt

    # Ledger balance is available cash plus any holds already placed
    ledger_balance = initial_available + total_reserved
    is_breached = initial_available < safety_floor

    return FinancialState(
        user_id=user_id,
        date=as_of_date,
        projected_balance=ledger_balance,
        reserved_pending=total_reserved,
        available_cash=initial_available,
        cumulative_inflows=Decimal("0"),
        cumulative_outflows=Decimal("0"),
        known_scheduled_obligations=Decimal("0"),
        safety_floor=safety_floor,
        is_safety_floor_breached=is_breached,
        active_reservations=tuple(sorted(active_res.items())),
    )


def _get_event_priority(event: SimulatedEvent) -> EventPriority:
    """Determine deterministic execution priority for same-day events."""
    if event.is_unresolved or event.cash_impact_type in (
        CashImpactType.PENDING_CREDIT_IGNORED,
        CashImpactType.FAILED_IGNORED,
        CashImpactType.CANCELLED_IGNORED,
        CashImpactType.UNREALIZED_NON_CASH,
    ):
        return EventPriority.NON_CASH
    if event.direction == Direction.INFLOW:
        return EventPriority.INFLOW
    if event.cash_impact_type == CashImpactType.PENDING_DEBIT_RESERVED:
        return EventPriority.PENDING_HOLD
    if event.direction == Direction.OUTFLOW:
        return EventPriority.OUTFLOW
    return EventPriority.NON_CASH


def _event_sort_key(event: SimulatedEvent) -> Tuple[date, int, str]:
    """Sort key guaranteeing 100% deterministic event order across runs.

    Priority:
    1. effective_date ascending
    2. EventPriority (Inflows first -> Holds -> Outflows -> Non-cash)
    3. event_id ascending (tie-breaker)
    """
    priority = _get_event_priority(event).value
    return (event.effective_date, priority, event.event_id)


def simulate_user(
    user_id: str,
    simulation_start: date,
    simulation_end: date,
    initial_state: Optional[FinancialState] = None,
    canonical_events: Sequence[CanonicalEvent] = (),
    future_events: Sequence[FutureEvent] = (),
    profile: Optional[FinancialProfile] = None,
    additional_events: Sequence[SimulatedEvent] = (),
) -> SimulationResult:
    """Execute a single deterministic cash-flow simulation over [simulation_start, simulation_end].

    Args:
        user_id: Target user ID.
        simulation_start: Inclusive start date of the simulation window.
        simulation_end: Inclusive end date of the simulation window.
        initial_state: Optional starting financial state. If omitted, built from profile.
        canonical_events: Canonical ledger events for the user.
        future_events: Recurrence-expanded future events for the user.
        profile: User financial profile (required if initial_state is None).
        additional_events: Optional candidate action events (e.g. proposed purchases or plans).

    Returns:
        SimulationResult containing ending state, daily snapshots, and full audit trace.
    """
    if simulation_end < simulation_start:
        raise ValueError(
            f"simulation_end ({simulation_end}) cannot be before simulation_start ({simulation_start})"
        )

    # 1. Establish initial state
    if initial_state is None:
        if profile is None:
            raise ValueError("Either initial_state or profile must be provided to simulate_user")
        initial_state = create_initial_state(user_id, simulation_start, profile)

    safety_floor = initial_state.safety_floor

    # 2. Ingest and convert events within the simulation window
    sim_events: List[SimulatedEvent] = []
    seen_event_ids: Set[str] = set()

    # Track pre-existing reservations from initial state
    active_reservations: Dict[str, Decimal] = dict(initial_state.active_reservations)

    # Ingest canonical events
    for ce in canonical_events:
        if ce.user_id != user_id:
            continue
        # Window boundary check: [simulation_start, simulation_end] inclusive
        if simulation_start <= ce.effective_date <= simulation_end:
            if ce.event_id in seen_event_ids:
                continue
            seen_event_ids.add(ce.event_id)
            sim_ev = simulated_event_from_canonical(ce)
            # Update protection status from profile
            if profile and ce.category in profile.expense_categories_to_protect:
                sim_ev = SimulatedEvent(
                    event_id=sim_ev.event_id,
                    user_id=sim_ev.user_id,
                    effective_date=sim_ev.effective_date,
                    direction=sim_ev.direction,
                    amount=sim_ev.amount,
                    currency=sim_ev.currency,
                    category=sim_ev.category,
                    event_type=sim_ev.event_type,
                    description=sim_ev.description,
                    cash_impact_type=sim_ev.cash_impact_type,
                    flexibility=sim_ev.flexibility,
                    minimum_allowed_amount=sim_ev.minimum_allowed_amount,
                    is_protected=True,
                    source_type=sim_ev.source_type,
                    series_id=sim_ev.series_id,
                    linked_event_id=sim_ev.linked_event_id,
                    is_unresolved=sim_ev.is_unresolved,
                    is_forecast=False,
                )
            sim_events.append(sim_ev)

    # Ingest future recurrence events
    for fe in future_events:
        if fe.user_id != user_id:
            continue
        if simulation_start <= fe.effective_date <= simulation_end:
            if fe.event_id in seen_event_ids:
                continue
            seen_event_ids.add(fe.event_id)
            sim_ev = simulated_event_from_future(fe)
            if profile and fe.category in profile.expense_categories_to_protect:
                sim_ev = SimulatedEvent(
                    event_id=sim_ev.event_id,
                    user_id=sim_ev.user_id,
                    effective_date=sim_ev.effective_date,
                    direction=sim_ev.direction,
                    amount=sim_ev.amount,
                    currency=sim_ev.currency,
                    category=sim_ev.category,
                    event_type=sim_ev.event_type,
                    description=sim_ev.description,
                    cash_impact_type=sim_ev.cash_impact_type,
                    flexibility=sim_ev.flexibility,
                    minimum_allowed_amount=sim_ev.minimum_allowed_amount,
                    is_protected=True,
                    source_type=sim_ev.source_type,
                    series_id=sim_ev.series_id,
                    linked_event_id=sim_ev.linked_event_id,
                    is_unresolved=False,
                    is_forecast=True,
                )
            sim_events.append(sim_ev)

    # Ingest additional candidate/action events
    for ae in additional_events:
        if ae.user_id != user_id:
            continue
        if simulation_start <= ae.effective_date <= simulation_end:
            if ae.event_id in seen_event_ids:
                continue
            seen_event_ids.add(ae.event_id)
            sim_events.append(ae)

    # 3. Sort deterministically
    sim_events.sort(key=_event_sort_key)

    # Group events by calendar date for daily snapshots
    events_by_date: Dict[date, List[SimulatedEvent]] = {}
    for ev in sim_events:
        events_by_date.setdefault(ev.effective_date, []).append(ev)

    # 4. Simulation tracking variables
    current_balance = initial_state.projected_balance
    current_reserved = initial_state.reserved_pending
    current_available = current_balance - current_reserved

    cumulative_inflows = initial_state.cumulative_inflows
    cumulative_outflows = initial_state.cumulative_outflows
    total_scheduled_obligations = initial_state.known_scheduled_obligations

    min_available_cash = current_available
    min_cash_date = simulation_start

    event_transitions: List[EventTransition] = []
    safety_floor_breaches: List[SafetyFloorBreach] = []
    obligation_breaches: List[SafetyFloorBreach] = []
    daily_snapshots: List[DailySnapshot] = []

    # Check initial state for breaches
    if current_available < safety_floor:
        safety_floor_breaches.append(SafetyFloorBreach(
            date=simulation_start,
            event_id="initial_state",
            available_cash=current_available,
            safety_floor=safety_floor,
            shortfall=safety_floor - current_available,
            breach_type="safety_floor",
        ))
    if current_available < Decimal("0"):
        obligation_breaches.append(SafetyFloorBreach(
            date=simulation_start,
            event_id="initial_state",
            available_cash=current_available,
            safety_floor=safety_floor,
            shortfall=Decimal("0") - current_available,
            breach_type="obligation_overdraft",
        ))

    step_counter = 0

    # 5. Day-by-day simulation loop
    total_days = (simulation_end - simulation_start).days + 1
    for day_offset in range(total_days):
        current_date = simulation_start + timedelta(days=day_offset)
        day_events = events_by_date.get(current_date, [])

        opening_available = current_available
        day_lowest_available = current_available
        day_event_ids: List[str] = []

        for ev in day_events:
            step_counter += 1
            day_event_ids.append(ev.event_id)

            bal_before = current_balance
            res_before = current_reserved
            avail_before = current_available

            # Apply cash semantics
            if ev.is_unresolved:
                # Unresolved events contribute strictly zero cash
                pass

            elif ev.cash_impact_type == CashImpactType.SETTLED_INFLOW:
                current_balance += ev.amount
                cumulative_inflows += ev.amount

            elif ev.cash_impact_type == CashImpactType.SCHEDULED_INFLOW:
                current_balance += ev.amount
                cumulative_inflows += ev.amount

            elif ev.cash_impact_type == CashImpactType.SETTLED_OUTFLOW:
                # Check if this settled event clears an active pending reservation
                # (via linked_event_id or matching event_id)
                reservation_key = None
                if ev.linked_event_id and ev.linked_event_id in active_reservations:
                    reservation_key = ev.linked_event_id
                elif ev.event_id in active_reservations:
                    reservation_key = ev.event_id

                if reservation_key:
                    reserved_amt = active_reservations[reservation_key]
                    release_amt = min(ev.amount, reserved_amt)
                    current_reserved -= release_amt
                    active_reservations[reservation_key] -= release_amt
                    if active_reservations[reservation_key] <= Decimal("0"):
                        del active_reservations[reservation_key]

                current_balance -= ev.amount
                cumulative_outflows += ev.amount

            elif ev.cash_impact_type == CashImpactType.SCHEDULED_OUTFLOW:
                current_balance -= ev.amount
                cumulative_outflows += ev.amount
                total_scheduled_obligations += ev.amount

            elif ev.cash_impact_type == CashImpactType.PENDING_DEBIT_RESERVED:
                # Hold/reserve cash if not already reserved
                if ev.event_id not in active_reservations:
                    active_reservations[ev.event_id] = ev.amount
                    current_reserved += ev.amount
                    total_scheduled_obligations += ev.amount

            elif ev.cash_impact_type in (
                CashImpactType.PENDING_CREDIT_IGNORED,
                CashImpactType.FAILED_IGNORED,
                CashImpactType.CANCELLED_IGNORED,
                CashImpactType.UNREALIZED_NON_CASH,
            ):
                # Strictly zero cash impact
                pass

            # Update available cash
            current_available = current_balance - current_reserved

            # Check intra-day lowest
            if current_available < day_lowest_available:
                day_lowest_available = current_available

            # Check global minimum
            if current_available < min_available_cash:
                min_available_cash = current_available
                min_cash_date = current_date

            is_sf_breached = current_available < safety_floor
            is_ob_breached = current_available < Decimal("0")

            if is_sf_breached and avail_before >= safety_floor:
                safety_floor_breaches.append(SafetyFloorBreach(
                    date=current_date,
                    event_id=ev.event_id,
                    available_cash=current_available,
                    safety_floor=safety_floor,
                    shortfall=safety_floor - current_available,
                    breach_type="safety_floor",
                ))

            if is_ob_breached and avail_before >= Decimal("0"):
                obligation_breaches.append(SafetyFloorBreach(
                    date=current_date,
                    event_id=ev.event_id,
                    available_cash=current_available,
                    safety_floor=safety_floor,
                    shortfall=Decimal("0") - current_available,
                    breach_type="obligation_overdraft",
                ))

            event_transitions.append(EventTransition(
                step=step_counter,
                date=current_date,
                event_id=ev.event_id,
                event_description=ev.description,
                category=ev.category,
                direction=ev.direction,
                amount=ev.amount,
                cash_impact_type=ev.cash_impact_type,
                source_type=ev.source_type,
                balance_before=bal_before,
                balance_after=current_balance,
                reserved_before=res_before,
                reserved_after=current_reserved,
                available_before=avail_before,
                available_after=current_available,
                safety_floor=safety_floor,
                is_safety_floor_breached=is_sf_breached,
                is_obligation_breached=is_ob_breached,
            ))

        # Build DailySnapshot
        daily_snapshots.append(DailySnapshot(
            date=current_date,
            opening_available_cash=opening_available,
            closing_projected_balance=current_balance,
            closing_reserved_pending=current_reserved,
            closing_available_cash=current_available,
            lowest_available_cash=day_lowest_available,
            safety_floor=safety_floor,
            is_safety_floor_breached=day_lowest_available < safety_floor,
            is_obligation_breached=day_lowest_available < Decimal("0"),
            event_ids=tuple(day_event_ids),
        ))

    # 6. Build ending state
    ending_state = FinancialState(
        user_id=user_id,
        date=simulation_end,
        projected_balance=current_balance,
        reserved_pending=current_reserved,
        available_cash=current_available,
        cumulative_inflows=cumulative_inflows,
        cumulative_outflows=cumulative_outflows,
        known_scheduled_obligations=total_scheduled_obligations,
        safety_floor=safety_floor,
        is_safety_floor_breached=current_available < safety_floor,
        active_reservations=tuple(sorted(active_reservations.items())),
    )

    return SimulationResult(
        user_id=user_id,
        start_date=simulation_start,
        end_date=simulation_end,
        initial_state=initial_state,
        ending_state=ending_state,
        minimum_projected_available_cash=min_available_cash,
        minimum_cash_date=min_cash_date,
        total_inflows=cumulative_inflows,
        total_outflows=cumulative_outflows,
        total_scheduled_obligations=total_scheduled_obligations,
        safety_floor=safety_floor,
        is_safety_floor_breached=len(safety_floor_breaches) > 0 or min_available_cash < safety_floor,
        safety_floor_breaches=tuple(safety_floor_breaches),
        obligation_breaches=tuple(obligation_breaches),
        projected_events=tuple(sim_events),
        event_transitions=tuple(event_transitions),
        daily_snapshots=tuple(daily_snapshots),
    )
