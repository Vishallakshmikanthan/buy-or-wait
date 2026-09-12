"""Deterministic Payment Option Feasibility Layer for Buy or Wait?

Evaluates whether specific seller/provider payment options from request_payment_options.csv
are eligible and safe for a given user financial request.

Core Principles:
1. Only explicit options present in request_payment_options.csv are considered.
   Never invent installment counts, amounts, frequencies, financing fees, or dates.
2. Uses the exact same single cash-flow simulator engine (simulator.py).
   Never duplicate balance arithmetic or write ad-hoc loops.
3. Clean separation of Eligibility (preferences, constraints, schema, horizon) from
   Safety (preservation of the safety floor across the 90-day trajectory).
4. Strictly preserves all options evaluated; does NOT perform option ranking,
   scoring, or best-option selection.
5. Uses Decimal exclusively for all monetary values.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from code.canonical import CanonicalEvent, CashImpactType, Direction
from code.models import FinancialProfile, FinancialRequest, PaymentOption
from code.recurrence import FutureEvent
from code.simulator import (
    SimulatedEvent,
    SimulationResult,
    simulate_user,
)


def add_calendar_months(sourcedate: date, months: int) -> date:
    """Add integer calendar months to a date, capping to the last day of the resulting month."""
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class PaymentScheduleEntry:
    """A single scheduled contractual payment outflow."""
    date: date
    amount: Decimal
    sequence_number: int  # 1-indexed (1, 2, ..., N)

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            raise TypeError(f"amount must be Decimal, got {type(self.amount)}")
        if self.amount <= Decimal("0"):
            raise ValueError(f"amount must be positive, got {self.amount}")
        if self.sequence_number < 1:
            raise ValueError(f"sequence_number must be >= 1, got {self.sequence_number}")


@dataclass(frozen=True)
class PaymentPlan:
    """Complete contractual payment plan constructed from a valid PaymentOption."""
    request_id: str
    payment_option_id: str
    payment_method: str
    entries: Tuple[PaymentScheduleEntry, ...]
    total_amount: Decimal          # Base requested purchase amount
    financing_fee: Decimal         # Explicit contractual financing fee
    total_payable_amount: Decimal  # Sum of all contractual scheduled payments

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("PaymentPlan must contain at least one payment entry")
        if not isinstance(self.total_amount, Decimal):
            raise TypeError(f"total_amount must be Decimal, got {type(self.total_amount)}")
        if not isinstance(self.financing_fee, Decimal):
            raise TypeError(f"financing_fee must be Decimal, got {type(self.financing_fee)}")
        if not isinstance(self.total_payable_amount, Decimal):
            raise TypeError(f"total_payable_amount must be Decimal, got {type(self.total_payable_amount)}")

    @property
    def number_of_payments(self) -> int:
        """Total number of contractual payment installments."""
        return len(self.entries)

    @property
    def first_payment_date(self) -> date:
        """Calendar date of the initial payment."""
        return self.entries[0].date

    @property
    def last_payment_date(self) -> date:
        """Calendar date of the final payment completing the plan."""
        return self.entries[-1].date

    @property
    def schedule_string(self) -> str:
        """Contest-compliant serialized schedule format: <YYYY-MM-DD>:<amount>|..."""
        return "|".join(f"{e.date.isoformat()}:{e.amount}" for e in self.entries)


@dataclass(frozen=True)
class PaymentPlanSimulatorReference:
    """Immutable audit trail of simulation evidence for payment plan evaluation."""
    simulation_start: date
    simulation_end: date
    minimum_available_cash: Decimal
    safety_floor: Decimal
    limiting_date: date
    is_safety_floor_breached: bool
    num_safety_floor_breaches: int
    num_obligation_breaches: int


@dataclass(frozen=True)
class PaymentPlanFeasibility:
    """Deterministic result of evaluating a payment option for eligibility and safety.

    Satisfies the PaymentPlanFeasibility protocol expected by downstream/upstream layers.
    """
    request_id: str
    payment_option_id: str
    is_eligible: bool
    is_safe: bool
    rejection_reason: Optional[str]
    payment_plan: Optional[PaymentPlan]
    minimum_available_cash: Optional[Decimal]
    safety_floor: Optional[Decimal]
    limiting_date: Optional[date]
    simulator_certificate: Optional[PaymentPlanSimulatorReference] = None

    @property
    def has_valid_plan(self) -> bool:
        """True if the option is both eligible and safe throughout the simulation."""
        return self.is_eligible and self.is_safe

    @property
    def plan_type(self) -> Optional[str]:
        """Type of payment plan (e.g. 'installments', 'full_payment')."""
        return self.payment_plan.payment_method if self.payment_plan else None


def construct_payment_schedule(
    option: PaymentOption,
    requested_amount: Decimal,
) -> PaymentPlan:
    """Construct the exact contractual payment schedule from a PaymentOption.

    Schedule generation follows:
    - full_payment: 1 entry on first_payment_date with payment_amount.
    - installments: N entries on [first_payment_date + k * frequency_days] for k in 0..N-1.
    """
    if option.payment_method == "full_payment":
        entries = (
            PaymentScheduleEntry(
                date=option.first_payment_date,
                amount=option.payment_amount,
                sequence_number=1,
            ),
        )
    elif option.payment_method == "installments":
        freq = option.payment_frequency_days or 0
        entries_list: List[PaymentScheduleEntry] = []
        for k in range(option.number_of_payments):
            entry_date = option.first_payment_date + timedelta(days=k * freq)
            entries_list.append(
                PaymentScheduleEntry(
                    date=entry_date,
                    amount=option.payment_amount,
                    sequence_number=k + 1,
                )
            )
        entries = tuple(entries_list)
    else:
        raise ValueError(f"Unsupported payment method: {option.payment_method}")

    return PaymentPlan(
        request_id=option.request_id,
        payment_option_id=option.payment_option_id,
        payment_method=option.payment_method,
        entries=entries,
        total_amount=requested_amount,
        financing_fee=option.financing_fee,
        total_payable_amount=option.total_payable_amount,
    )


def validate_option_schema(
    option: PaymentOption,
    request: FinancialRequest,
) -> Optional[str]:
    """Validate payment option data integrity and reconciliation.

    Returns None if valid, or a deterministic rejection reason string if invalid.
    """
    # 1. Request matching
    if option.request_id != request.request_id:
        return f"option belongs to different request (option {option.request_id} != request {request.request_id})"

    # 2. Basic numeric bounds
    if option.payment_amount <= Decimal("0"):
        return f"invalid payment amount ({option.payment_amount} <= 0)"

    if option.number_of_payments < 1:
        return f"invalid number of payments ({option.number_of_payments} < 1)"

    if option.financing_fee < Decimal("0"):
        return f"invalid financing fee ({option.financing_fee} < 0)"

    # 3. Method-specific structural invariants
    if option.payment_method == "full_payment":
        if option.number_of_payments != 1:
            return f"full_payment requires exactly 1 payment, got {option.number_of_payments}"
        if option.financing_fee != Decimal("0"):
            return f"full_payment financing fee must be 0, got {option.financing_fee}"
    elif option.payment_method == "installments":
        if option.number_of_payments <= 1:
            return f"installments require > 1 payments, got {option.number_of_payments}"
        if option.payment_frequency_days is None or option.payment_frequency_days <= 0:
            return f"installments require positive payment_frequency_days, got {option.payment_frequency_days}"
    else:
        return f"unknown payment method: {option.payment_method}"

    # 4. Total payable reconciliation (allow standard penny currency rounding <= 0.05)
    expected_payable = option.payment_amount * option.number_of_payments
    if abs(expected_payable - option.total_payable_amount) > Decimal("0.05"):
        return (
            f"amount reconciliation failure: payment_amount ({option.payment_amount}) * "
            f"payments ({option.number_of_payments}) = {expected_payable} != "
            f"total_payable_amount ({option.total_payable_amount})"
        )

    return None


def check_option_eligibility(
    option: PaymentOption,
    request: FinancialRequest,
    profile: FinancialProfile,
    horizon_days: int = 90,
) -> Tuple[bool, Optional[str], Optional[PaymentPlan]]:
    """Determine whether an option is eligible for the given request and profile.

    Checks in strict deterministic order:
    1. Schema validation & request matching.
    2. User payment method preference.
    3. First payment date bounds (must not precede request date).
    4. Installment schedule construction.
    5. User maximum installment duration (max_installment_months).
    6. 90-day simulation horizon boundary (last payment must not exceed request_date + 90 days).

    Returns:
        (is_eligible, rejection_reason, payment_plan)
    """
    # Step 1: Schema validation
    schema_err = validate_option_schema(option, request)
    if schema_err is not None:
        return False, schema_err, None

    # Step 2: User payment preference
    if option.payment_method not in profile.payment_methods_user_will_consider:
        return False, f"payment method '{option.payment_method}' not permitted by user preferences", None

    # Step 3: First payment date
    if option.first_payment_date < request.request_date:
        return False, f"first payment date ({option.first_payment_date}) precedes request date ({request.request_date})", None

    # Step 4: Construct schedule
    plan = construct_payment_schedule(option, request.requested_amount)

    # Step 5: Installment-month limit check (derived from actual schedule duration)
    if option.payment_method == "installments" and profile.max_installment_months is not None:
        max_months = profile.max_installment_months
        schedule_limit_date = add_calendar_months(option.first_payment_date, max_months)
        if plan.last_payment_date > schedule_limit_date:
            return (
                False,
                f"exceeds maximum installment duration of {max_months} months "
                f"(last payment {plan.last_payment_date} > limit {schedule_limit_date})",
                plan,
            )

    # Step 6: Allowed horizon check (all payments must fall within the 90-day simulation window)
    horizon_end = request.request_date + timedelta(days=horizon_days)
    if plan.last_payment_date > horizon_end:
        return (
            False,
            f"payment schedule outside allowed horizon "
            f"(last payment {plan.last_payment_date} > horizon end {horizon_end})",
            plan,
        )

    return True, None, plan


def evaluate_payment_option_feasibility(
    option: PaymentOption,
    request: FinancialRequest,
    profile: FinancialProfile,
    canonical_events: Sequence[CanonicalEvent],
    future_events: Sequence[FutureEvent],
    baseline_simulation: Optional[SimulationResult] = None,
    simulation_start: Optional[date] = None,
    simulation_end: Optional[date] = None,
) -> PaymentPlanFeasibility:
    """Evaluate a single PaymentOption for eligibility and safety.

    If the option is ineligible, simulation is bypassed and the rejection reason is returned.
    If eligible, candidate payment events are injected into the single deterministic
    simulator (simulator.py). Safety is verified across the complete 90-day trajectory.
    """
    sim_start = simulation_start or request.request_date
    sim_end = simulation_end or (request.request_date + timedelta(days=90))

    is_eligible, rejection_reason, plan = check_option_eligibility(
        option=option,
        request=request,
        profile=profile,
        horizon_days=(sim_end - sim_start).days,
    )

    if not is_eligible or plan is None:
        return PaymentPlanFeasibility(
            request_id=request.request_id,
            payment_option_id=option.payment_option_id,
            is_eligible=False,
            is_safe=False,
            rejection_reason=rejection_reason,
            payment_plan=plan,
            minimum_available_cash=None,
            safety_floor=profile.minimum_balance_to_keep,
            limiting_date=None,
            simulator_certificate=None,
        )

    # Construct candidate action events (Priority 3 in simulator: CANDIDATE_ACTION)
    candidate_events: List[SimulatedEvent] = []
    for entry in plan.entries:
        ce = SimulatedEvent(
            event_id=f"plan_action_{option.payment_option_id}_{entry.sequence_number}_{entry.date.strftime('%Y%m%d')}",
            user_id=request.user_id,
            effective_date=entry.date,
            direction=Direction.OUTFLOW,
            amount=entry.amount,
            currency=profile.home_currency,
            category=request.request_type,
            event_type="expense",
            description=f"Payment {entry.sequence_number}/{plan.number_of_payments} for {option.payment_option_id}",
            cash_impact_type=CashImpactType.SCHEDULED_OUTFLOW,
            flexibility="fixed",
            is_protected=False,
            source_type="action",  # Triggers EventPriority.CANDIDATE_ACTION (Priority 3)
            is_forecast=False,
        )
        candidate_events.append(ce)

    # Run hypothetical simulation using the SAME frozen simulator engine
    sim_res = simulate_user(
        user_id=request.user_id,
        simulation_start=sim_start,
        simulation_end=sim_end,
        canonical_events=canonical_events,
        future_events=future_events,
        profile=profile,
        additional_events=candidate_events,
    )

    is_safe = (
        not sim_res.is_safety_floor_breached
        and sim_res.minimum_projected_available_cash >= profile.minimum_balance_to_keep
        and len(sim_res.obligation_breaches) == 0
    )

    sim_cert = PaymentPlanSimulatorReference(
        simulation_start=sim_start,
        simulation_end=sim_end,
        minimum_available_cash=sim_res.minimum_projected_available_cash,
        safety_floor=sim_res.safety_floor,
        limiting_date=sim_res.minimum_cash_date,
        is_safety_floor_breached=sim_res.is_safety_floor_breached,
        num_safety_floor_breaches=len(sim_res.safety_floor_breaches),
        num_obligation_breaches=len(sim_res.obligation_breaches),
    )

    safety_reason: Optional[str] = None
    if not is_safe:
        if sim_res.obligation_breaches:
            first_ob = sim_res.obligation_breaches[0]
            safety_reason = (
                f"obligation overdraft on {first_ob.date} "
                f"(available {first_ob.available_cash} < 0, shortfall {first_ob.shortfall})"
            )
        elif sim_res.safety_floor_breaches:
            first_br = sim_res.safety_floor_breaches[0]
            safety_reason = (
                f"safety floor breach on {first_br.date} "
                f"(available {first_br.available_cash} < floor {first_br.safety_floor})"
            )
        else:
            safety_reason = (
                f"minimum available cash ({sim_res.minimum_projected_available_cash}) "
                f"violates safety floor ({profile.minimum_balance_to_keep})"
            )

    return PaymentPlanFeasibility(
        request_id=request.request_id,
        payment_option_id=option.payment_option_id,
        is_eligible=True,
        is_safe=is_safe,
        rejection_reason=safety_reason,
        payment_plan=plan,
        minimum_available_cash=sim_res.minimum_projected_available_cash,
        safety_floor=sim_res.safety_floor,
        limiting_date=sim_res.minimum_cash_date,
        simulator_certificate=sim_cert,
    )


def evaluate_payment_options_for_request(
    request: FinancialRequest,
    options: Sequence[PaymentOption],
    profile: FinancialProfile,
    canonical_events: Sequence[CanonicalEvent],
    future_events: Sequence[FutureEvent],
    baseline_simulation: Optional[SimulationResult] = None,
) -> Tuple[PaymentPlanFeasibility, ...]:
    """Evaluate all payment options available for a request.

    Evaluates each option independently in the exact order received.
    Does NOT rank, score, or filter the returned feasibility results.
    """
    results: List[PaymentPlanFeasibility] = []
    for opt in options:
        feasibility = evaluate_payment_option_feasibility(
            option=opt,
            request=request,
            profile=profile,
            canonical_events=canonical_events,
            future_events=future_events,
            baseline_simulation=baseline_simulation,
        )
        results.append(feasibility)
    return tuple(results)
