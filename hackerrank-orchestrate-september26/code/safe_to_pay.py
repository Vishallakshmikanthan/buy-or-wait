"""Deterministic safe-to-pay and earliest full payment calculation layer.

Computes:
1. amount_safe_to_pay: Maximum monetary amount the user can safely pay today
   without violating their safety floor or causing an overdraft at any point
   across the 90-day simulation horizon.
2. earliest_date_for_full_payment: Earliest conservative calendar date on which
   the full requested amount can be paid safely without violating the safety floor.

Both calculations are backed strictly by the single deterministic simulator engine,
use Decimal monetary precision exclusively, and produce an immutable audit certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Sequence, Tuple

from code.canonical import CanonicalEvent, CashImpactType, Direction
from code.models import FinancialProfile, FinancialRequest
from code.recurrence import FutureEvent
from code.simulator import (
    DailySnapshot,
    SimulatedEvent,
    SimulationResult,
    simulate_user,
)


CURRENCY_QUANTUM: Dict[str, Decimal] = {
    "USD": Decimal("0.01"),
    "EUR": Decimal("0.01"),
    "GBP": Decimal("0.01"),
    "ZAR": Decimal("0.01"),
    "INR": Decimal("0.01"),
    "IDR": Decimal("0.01"),
}


def get_currency_quantum(currency: str) -> Decimal:
    """Return the smallest meaningful currency increment for a given ISO currency code."""
    return CURRENCY_QUANTUM.get(currency.upper().strip(), Decimal("0.01"))


@dataclass(frozen=True)
class SafeToPayCertificate:
    """Deterministic internal audit certificate explaining safe-to-pay results."""
    request_id: str
    user_id: str
    request_date: date
    simulation_start: date
    simulation_end: date
    requested_amount: Decimal
    amount_safe_to_pay: Decimal
    currency: str
    safety_floor: Decimal
    baseline_minimum_available_cash: Decimal
    baseline_minimum_cash_date: date
    limiting_date: date
    available_cash_after_purchase_today: Decimal
    minimum_available_cash_after_purchase: Decimal
    safety_floor_margin: Decimal
    earliest_date_for_full_payment: Optional[date]
    is_full_payment_safe_today: bool
    is_full_payment_safe_later: bool
    reason_if_unsafe: Optional[str]
    candidate_search_method: str


def make_candidate_purchase_event(
    request: FinancialRequest,
    amount: Decimal,
    effective_date: date,
    currency: str,
    event_id: Optional[str] = None,
) -> SimulatedEvent:
    """Construct a deterministic candidate purchase event for evaluation."""
    eid = event_id or f"candidate_purchase_{request.request_id}_{effective_date.strftime('%Y%m%d')}_{str(amount).replace('.', '_')}"
    return SimulatedEvent(
        event_id=eid,
        user_id=request.user_id,
        effective_date=effective_date,
        direction=Direction.OUTFLOW,
        amount=amount,
        currency=currency,
        category=request.request_type,
        event_type="expense",
        description=f"Candidate purchase for request {request.request_id}",
        cash_impact_type=CashImpactType.SETTLED_OUTFLOW,
        flexibility="fixed",
        is_protected=False,
        source_type="action",
        is_forecast=False,
    )


def is_purchase_safe(
    user_id: str,
    simulation_start: date,
    simulation_end: date,
    canonical_events: Sequence[CanonicalEvent],
    future_events: Sequence[FutureEvent],
    profile: FinancialProfile,
    candidate_event: SimulatedEvent,
) -> Tuple[bool, SimulationResult]:
    """Test whether a candidate purchase is safe by executing the deterministic simulator.

    Safety Condition:
    - available_cash >= safety_floor throughout [simulation_start, simulation_end]
    - available_cash >= 0 (no prohibited overdraft condition)
    - all known/forecast obligations remain fundable
    - is_safety_floor_breached is False
    - obligation_breaches count is 0
    """
    res = simulate_user(
        user_id=user_id,
        simulation_start=simulation_start,
        simulation_end=simulation_end,
        canonical_events=canonical_events,
        future_events=future_events,
        profile=profile,
        additional_events=[candidate_event],
    )
    is_safe = (
        not res.is_safety_floor_breached
        and res.minimum_projected_available_cash >= profile.minimum_balance_to_keep
        and len(res.obligation_breaches) == 0
    )
    return is_safe, res


def calculate_amount_safe_to_pay(
    request: FinancialRequest,
    baseline_simulation: SimulationResult,
    canonical_events: Sequence[CanonicalEvent],
    future_events: Sequence[FutureEvent],
    profile: FinancialProfile,
    quantum: Optional[Decimal] = None,
) -> Tuple[Decimal, SimulationResult, str]:
    """Calculate the exact maximum monetary amount the user can safely pay today.

    Formally:
        max X in [0, requested_amount] such that paying X on request_date
        leaves available_cash >= safety_floor and >= 0 across the entire horizon.

    Returns:
        (amount_safe_to_pay, simulation_result_with_safe_amount, search_method)
    """
    safety_floor = profile.minimum_balance_to_keep
    req_amt = request.requested_amount
    q = quantum or get_currency_quantum(profile.home_currency)

    # 1. Check if baseline itself is already in breach
    if baseline_simulation.is_safety_floor_breached or baseline_simulation.minimum_projected_available_cash < safety_floor:
        zero_event = make_candidate_purchase_event(request, Decimal("0"), request.request_date, profile.home_currency)
        _, zero_res = is_purchase_safe(
            user_id=request.user_id,
            simulation_start=baseline_simulation.start_date,
            simulation_end=baseline_simulation.end_date,
            canonical_events=canonical_events,
            future_events=future_events,
            profile=profile,
            candidate_event=zero_event,
        )
        return Decimal("0"), zero_res, "baseline_breached_zero"

    # 2. Check theoretical headroom
    headroom = baseline_simulation.minimum_projected_available_cash - safety_floor
    if headroom <= Decimal("0"):
        zero_event = make_candidate_purchase_event(request, Decimal("0"), request.request_date, profile.home_currency)
        _, zero_res = is_purchase_safe(
            user_id=request.user_id,
            simulation_start=baseline_simulation.start_date,
            simulation_end=baseline_simulation.end_date,
            canonical_events=canonical_events,
            future_events=future_events,
            profile=profile,
            candidate_event=zero_event,
        )
        return Decimal("0"), zero_res, "zero_headroom"

    # Candidate upper bound quantized down to currency quantum
    cand_units = int((min(req_amt, headroom) / q).quantize(Decimal("1"), rounding=ROUND_DOWN))
    candidate_amt = cand_units * q

    # 3. Fast verification with simulator
    cand_event = make_candidate_purchase_event(request, candidate_amt, request.request_date, profile.home_currency)
    safe, cand_res = is_purchase_safe(
        user_id=request.user_id,
        simulation_start=baseline_simulation.start_date,
        simulation_end=baseline_simulation.end_date,
        canonical_events=canonical_events,
        future_events=future_events,
        profile=profile,
        candidate_event=cand_event,
    )

    if safe:
        # If candidate equals requested amount or adding 1 quantum causes a breach,
        # it is provably the exact discrete maximum!
        if candidate_amt == req_amt:
            return candidate_amt, cand_res, "exact_full_amount"

        # Verify upper boundary
        next_cand = candidate_amt + q
        next_event = make_candidate_purchase_event(request, next_cand, request.request_date, profile.home_currency)
        next_safe, _ = is_purchase_safe(
            user_id=request.user_id,
            simulation_start=baseline_simulation.start_date,
            simulation_end=baseline_simulation.end_date,
            canonical_events=canonical_events,
            future_events=future_events,
            profile=profile,
            candidate_event=next_event,
        )
        if not next_safe:
            return candidate_amt, cand_res, "exact_headroom_boundary"

    # 4. Exact discrete binary search fallback over integer units
    low = 0
    high = int((req_amt / q).quantize(Decimal("1"), rounding=ROUND_DOWN))
    best_amt = Decimal("0")
    best_res = baseline_simulation

    while low <= high:
        mid = (low + high) // 2
        test_amt = mid * q
        test_ev = make_candidate_purchase_event(request, test_amt, request.request_date, profile.home_currency)
        test_safe, test_res = is_purchase_safe(
            user_id=request.user_id,
            simulation_start=baseline_simulation.start_date,
            simulation_end=baseline_simulation.end_date,
            canonical_events=canonical_events,
            future_events=future_events,
            profile=profile,
            candidate_event=test_ev,
        )
        if test_safe:
            best_amt = test_amt
            best_res = test_res
            low = mid + 1
        else:
            high = mid - 1

    return best_amt, best_res, "discrete_binary_search"


def calculate_earliest_full_payment_date(
    request: FinancialRequest,
    baseline_simulation: SimulationResult,
    canonical_events: Sequence[CanonicalEvent],
    future_events: Sequence[FutureEvent],
    profile: FinancialProfile,
    is_full_safe_today: bool = False,
) -> Optional[date]:
    """Determine the earliest calendar date on which paying the full requested amount is safe.

    Returns:
        earliest_safe_date within [simulation_start, simulation_end], or None if never safe.
    """
    if is_full_safe_today:
        return request.request_date

    req_amt = request.requested_amount
    safety_floor = profile.minimum_balance_to_keep
    snaps = baseline_simulation.daily_snapshots
    n_days = len(snaps)

    if n_days == 0:
        return None

    # Compute suffix minimums of lowest_available_cash across baseline
    suffix_mins = [Decimal("0")] * n_days
    curr_min = snaps[-1].lowest_available_cash
    for i in range(n_days - 1, -1, -1):
        if snaps[i].lowest_available_cash < curr_min:
            curr_min = snaps[i].lowest_available_cash
        suffix_mins[i] = curr_min

    target_required = safety_floor + req_amt

    # Compute suffix minimums of lowest_available_cash from day i+1 onward
    # suffix_mins[k] = min_{j >= k} snaps[j].lowest_available_cash
    suffix_mins: List[Decimal] = [Decimal("0")] * (n_days + 1)
    suffix_mins[n_days] = Decimal("999999999999999")  # sentinel for after simulation end
    curr_min = snaps[-1].lowest_available_cash
    suffix_mins[n_days - 1] = curr_min
    for j in range(n_days - 2, -1, -1):
        if snaps[j].lowest_available_cash < curr_min:
            curr_min = snaps[j].lowest_available_cash
        suffix_mins[j] = curr_min

    for i in range(n_days):
        cand_date = snaps[i].date

        # If baseline prior to cand_date already breached safety floor,
        # waiting until cand_date cannot fix the historical breach without a plan
        if any(snaps[j].lowest_available_cash < safety_floor for j in range(i)):
            continue

        # Pruning check:
        # 1. On cand_date, closing available cash must be >= target_required
        if snaps[i].closing_available_cash < target_required:
            continue

        # 2. On all subsequent days (j > i), lowest available cash must be >= target_required
        if suffix_mins[i + 1] < target_required:
            continue

        # Candidate passed pruning; verify through the simulator
        cand_ev = make_candidate_purchase_event(
            request=request,
            amount=req_amt,
            effective_date=cand_date,
            currency=profile.home_currency,
        )
        safe, res = is_purchase_safe(
            user_id=request.user_id,
            simulation_start=baseline_simulation.start_date,
            simulation_end=baseline_simulation.end_date,
            canonical_events=canonical_events,
            future_events=future_events,
            profile=profile,
            candidate_event=cand_ev,
        )
        if safe:
            return cand_date

    return None



def evaluate_request_safe_to_pay(
    request: FinancialRequest,
    baseline_simulation: SimulationResult,
    canonical_events: Sequence[CanonicalEvent],
    future_events: Sequence[FutureEvent],
    profile: FinancialProfile,
    quantum: Optional[Decimal] = None,
) -> SafeToPayCertificate:
    """Evaluate both amount_safe_to_pay and earliest_date_for_full_payment for a request."""
    safety_floor = profile.minimum_balance_to_keep
    req_amt = request.requested_amount

    # 1. Compute amount safe to pay today
    safe_amt, cand_res, search_method = calculate_amount_safe_to_pay(
        request=request,
        baseline_simulation=baseline_simulation,
        canonical_events=canonical_events,
        future_events=future_events,
        profile=profile,
        quantum=quantum,
    )

    is_full_safe_today = safe_amt == req_amt

    # 2. Compute earliest date for full payment
    earliest_date = calculate_earliest_full_payment_date(
        request=request,
        baseline_simulation=baseline_simulation,
        canonical_events=canonical_events,
        future_events=future_events,
        profile=profile,
        is_full_safe_today=is_full_safe_today,
    )

    is_full_safe_later = (earliest_date is not None and earliest_date > request.request_date)

    # 3. Audit fields
    today_snap = cand_res.get_snapshot_for_date(request.request_date)
    avail_today = today_snap.closing_available_cash if today_snap else (cand_res.initial_state.available_cash - safe_amt)
    min_avail_after = cand_res.minimum_projected_available_cash
    margin = min_avail_after - safety_floor

    reason_if_unsafe: Optional[str] = None
    if not is_full_safe_today:
        if baseline_simulation.is_safety_floor_breached:
            reason_if_unsafe = (
                f"Baseline trajectory breaches safety floor on {baseline_simulation.minimum_cash_date} "
                f"(projected min cash {baseline_simulation.minimum_projected_available_cash} < floor {safety_floor})."
            )
        elif safe_amt < req_amt:
            limiting_d = baseline_simulation.minimum_cash_date
            headroom = baseline_simulation.minimum_projected_available_cash - safety_floor
            reason_if_unsafe = (
                f"Full payment would breach safety floor on {limiting_d} "
                f"(headroom over 90 days is {headroom}, less than requested {req_amt})."
            )

    return SafeToPayCertificate(
        request_id=request.request_id,
        user_id=request.user_id,
        request_date=request.request_date,
        simulation_start=baseline_simulation.start_date,
        simulation_end=baseline_simulation.end_date,
        requested_amount=req_amt,
        amount_safe_to_pay=safe_amt,
        currency=profile.home_currency,
        safety_floor=safety_floor,
        baseline_minimum_available_cash=baseline_simulation.minimum_projected_available_cash,
        baseline_minimum_cash_date=baseline_simulation.minimum_cash_date,
        limiting_date=baseline_simulation.minimum_cash_date,
        available_cash_after_purchase_today=avail_today,
        minimum_available_cash_after_purchase=min_avail_after,
        safety_floor_margin=margin,
        earliest_date_for_full_payment=earliest_date,
        is_full_payment_safe_today=is_full_safe_today,
        is_full_payment_safe_later=is_full_safe_later,
        reason_if_unsafe=reason_if_unsafe,
        candidate_search_method=search_method,
    )
