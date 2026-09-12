"""Deterministic User Financial State Layer.

Constructs an immutable, auditable snapshot of a user's financial standing as of
the specific request date. Synthesizes authoritative evidence from the canonical ledger,
recurrence engine, and baseline cash-flow simulator without duplicating financial logic.

Follows the single-source-of-truth hierarchy:
1. Current cash and limits -> financial_profiles.csv / simulator initial state
2. Pending reservations -> canonical ledger pending-debit semantics
3. Historical statistics -> canonical ledger events strictly before request_date
4. Recurring commitments -> frozen recurrence detection engine
5. Projected trajectory & obligations -> frozen 90-day cash-flow simulator
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from enum import Enum
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from code.canonical import (
    CanonicalEvent,
    CanonicalLedger,
    CashImpactType,
    Direction,
)
from code.models import FinancialProfile, FinancialRequest
from code.recurrence import (
    FutureEvent,
    RecurrenceFrequency,
    RecurrenceSeries,
    expand_future_events,
)
from code.safe_to_pay import get_currency_quantum
from code.simulator import (
    DailySnapshot,
    SimulationResult,
    create_initial_state,
    simulate_user,
)


class SpendingTrend(str, Enum):
    """Deterministic classification of recent 30-day vs previous 30-day spending velocity."""
    INCREASING = "increasing"          # Outflow increased by > 15%
    DECREASING = "decreasing"          # Outflow decreased by > 15%
    STABLE = "stable"                  # Outflow within [-15%, +15%]
    INSUFFICIENT_DATA = "insufficient_data"  # Less than 60 days of history available


class IncomeStability(str, Enum):
    """Deterministic indicator of income reliability for future payment commitments."""
    HIGH = "high"                      # Validated recurring income with high cadence/history
    MODERATE = "moderate"              # Recurring income with shorter history or regular settled inflows
    LOW = "low"                        # Irregular, highly volatile, or sparse income
    INSUFFICIENT_DATA = "insufficient_data"  # Zero income records or insufficient history (<60d)


class ExpenseStability(str, Enum):
    """Deterministic indicator of expense predictability versus discretionary volatility."""
    HIGH = "high"                      # Recurring obligations dominate (>= 50% of budget)
    MODERATE = "moderate"              # Balanced mix of recurring obligations and discretionary spending
    LOW = "low"                        # Discretionary/volatile spending dominates (< 20% recurring)
    INSUFFICIENT_DATA = "insufficient_data"  # Insufficient history (< 30d)


# Standard recognized essential expense categories
ESSENTIAL_EXPENSE_CATEGORIES = frozenset({
    "rent",
    "housing",
    "utilities",
    "groceries",
    "healthcare",
    "insurance",
    "debt_repayment",
    "education",
    "family_support",
    "bills",
})


@dataclass(frozen=True)
class RecurringStreamSummary:
    """Deterministic summary of a verified recurring cashflow stream."""
    series_id: str
    category: str
    direction: Direction
    frequency: RecurrenceFrequency
    amount: Decimal
    currency: str
    is_protected: bool
    is_cancelled: bool
    monthly_equivalent: Decimal


@dataclass(frozen=True)
class UpcomingObligationSummary:
    """Deterministic record of a projected upcoming cash obligation in the 90-day horizon."""
    event_id: str
    effective_date: date
    category: str
    amount: Decimal
    currency: str
    source_type: str  # 'forecast_recurrence' or 'canonical_scheduled'
    series_id: Optional[str]
    is_protected: bool
    flexibility: str


@dataclass(frozen=True)
class UserFinancialState:
    """Immutable snapshot of a user's authoritative financial state at request_date.

    Guarantees:
    - 100% Decimal monetary precision.
    - Zero lookahead contamination: historical stats strictly filter effective_date < request_date.
    - Single source of truth: no independently recalculated simulator or recurrence values.
    - Exact currency alignment with profile.home_currency.
    """
    # 1. Request Context
    request_id: str
    user_id: str
    request_date: date
    currency: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool

    # 2. Current Cash & Policy (As of request_date)
    current_available_balance: Decimal
    safety_floor: Decimal
    minimum_balance_to_keep: Decimal
    pending_reserved_amount: Decimal
    available_cash: Decimal
    current_headroom_above_floor: Decimal

    # 3. Historical Cashflow Statistics (Strictly BEFORE request_date)
    # Recent 30 days: [request_date - 30d, request_date)
    recent_30d_income: Decimal
    recent_30d_outflow: Decimal
    recent_30d_essential_outflow: Decimal
    recent_30d_discretionary_outflow: Decimal
    recent_30d_net_cashflow: Decimal

    # Recent 90 days: [request_date - 90d, request_date)
    recent_90d_income: Decimal
    recent_90d_outflow: Decimal
    recent_90d_essential_outflow: Decimal
    recent_90d_discretionary_outflow: Decimal
    recent_90d_net_cashflow: Decimal
    monthly_average_income_90d: Decimal
    monthly_average_outflow_90d: Decimal

    # Historical income cadence
    historical_income_count: int
    latest_income_date: Optional[date]
    latest_income_amount: Optional[Decimal]

    # 4. Verified Recurring Commitments (From Recurrence Engine)
    recurring_income_streams: Tuple[RecurringStreamSummary, ...]
    recurring_expense_streams: Tuple[RecurringStreamSummary, ...]
    total_monthly_recurring_income: Decimal
    total_monthly_recurring_expenses: Decimal
    net_monthly_recurring_cashflow: Decimal
    has_validated_recurring_income: bool
    has_validated_recurring_expenses: bool

    # 5. Projected Baseline Trajectory (From Single Simulator Engine)
    baseline_minimum_available_cash: Decimal
    baseline_minimum_cash_date: date
    baseline_ending_available_cash: Decimal
    baseline_has_safety_breach: bool
    baseline_safety_margin: Decimal
    baseline_headroom: Decimal
    total_projected_inflows_90d: Decimal
    total_projected_outflows_90d: Decimal

    # 6. Upcoming Obligations (Horizon [request_date, request_date + 90d])
    upcoming_obligations_count: int
    upcoming_obligations_30d_total: Decimal
    upcoming_obligations_90d_total: Decimal
    upcoming_obligations: Tuple[UpcomingObligationSummary, ...]

    # 7. Behavioral & Stability Signals
    spending_trend: SpendingTrend
    spending_trend_percentage: Optional[Decimal]
    income_stability: IncomeStability
    expense_stability: ExpenseStability

    # 8. Data Sufficiency & Evidence Telemetry
    history_days_available: int
    total_historical_events_count: int
    is_income_history_sufficient: bool
    is_expense_history_sufficient: bool
    has_unresolved_events: bool
    unresolved_events_count: int

    # 9. User Preferences & Constraints
    financial_priorities: Tuple[str, ...]
    protected_categories: Tuple[str, ...]
    willing_to_reduce_categories: Tuple[str, ...]
    willing_to_stop_categories: Tuple[str, ...]
    payment_methods_considered: Tuple[str, ...]
    max_installment_months: Optional[int]


def compute_monthly_equivalent(amount: Decimal, freq: RecurrenceFrequency) -> Decimal:
    """Convert an amount of any recurrence frequency into its exact monthly equivalent."""
    if freq == RecurrenceFrequency.WEEKLY:
        # 52 weeks / 12 months
        return (amount * Decimal("52") / Decimal("12")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if freq == RecurrenceFrequency.BIWEEKLY:
        # 26 biweekly periods / 12 months
        return (amount * Decimal("26") / Decimal("12")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if freq == RecurrenceFrequency.TRIWEEKLY:
        # 52 / 3 periods per year / 12 months = 52 / 36
        return (amount * Decimal("52") / Decimal("36")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if freq == RecurrenceFrequency.MONTHLY:
        return amount
    return amount


def build_user_financial_state(
    request: FinancialRequest,
    profile: FinancialProfile,
    canonical_events: Sequence[CanonicalEvent],
    future_events: Sequence[FutureEvent],
    baseline_simulation: SimulationResult,
    recurring_series: Sequence[RecurrenceSeries] = (),
) -> UserFinancialState:
    """Deterministically construct the authoritative UserFinancialState for a request.

    Args:
        request: The evaluation request.
        profile: The user's financial profile.
        canonical_events: Canonical reconciled ledger events for the user.
        future_events: Expanded future forecast events for the user.
        baseline_simulation: Output of simulate_user across the 90-day horizon without candidate purchases.
        recurring_series: Verified RecurrenceSeries certificates from detect_all_recurrence.

    Returns:
        Immutable UserFinancialState snapshot.
    """
    req_d = request.request_date
    curr = profile.home_currency
    q = get_currency_quantum(curr)

    # 1. User preferences & protection sets
    protected_categories = tuple(sorted(profile.expense_categories_to_protect))
    willing_to_reduce = tuple(sorted(profile.expense_categories_user_is_willing_to_reduce))
    willing_to_stop = tuple(sorted(profile.expense_categories_user_is_willing_to_stop))
    payment_methods = tuple(sorted(profile.payment_methods_user_will_consider))

    essential_categories_all = set(ESSENTIAL_EXPENSE_CATEGORIES) | set(protected_categories)

    # 2. Current cash & liquidity reserves
    init_state = baseline_simulation.initial_state
    current_avail_bal = profile.current_available_balance
    safety_floor = profile.minimum_balance_to_keep
    min_bal = profile.minimum_balance_to_keep

    # Pending reserved amount captures holds from initial state or established on request_date
    today_snap = baseline_simulation.get_snapshot_for_date(req_d)
    pending_reserved = init_state.reserved_pending
    if today_snap and today_snap.closing_reserved_pending > pending_reserved:
        pending_reserved = today_snap.closing_reserved_pending

    available_cash = current_avail_bal - pending_reserved
    current_headroom = max(Decimal("0.00"), available_cash - safety_floor)

    # 3. Filter historical events strictly before request_date (NO lookahead leakage)
    historical_events: List[CanonicalEvent] = [
        ce for ce in canonical_events
        if ce.user_id == request.user_id and ce.effective_date < req_d
    ]

    total_hist_count = len(historical_events)
    earliest_date = min((ce.effective_date for ce in historical_events), default=None)
    history_days = (req_d - earliest_date).days if earliest_date else 0

    # Unresolved events check in history
    unresolved_events = [ce for ce in historical_events if ce.is_unresolved]
    unresolved_count = len(unresolved_events)
    has_unresolved = unresolved_count > 0

    # Historical window boundaries (left-inclusive, right-exclusive)
    w30_start = req_d - timedelta(days=30)
    w60_start = req_d - timedelta(days=60)
    w90_start = req_d - timedelta(days=90)

    # Inflow analysis
    # Only settled or scheduled cash inflows with confirmed cash impact are realized
    hist_inflows = [
        ce for ce in historical_events
        if ce.direction == Direction.INFLOW
        and ce.is_cash_event
        and ce.cash_impact_type in (CashImpactType.SETTLED_INFLOW, CashImpactType.SCHEDULED_INFLOW)
        and ce.amount_home is not None
        and ce.amount_home > Decimal("0")
    ]
    hist_income_count = len(hist_inflows)

    # Latest income
    latest_income_date: Optional[date] = None
    latest_income_amount: Optional[Decimal] = None
    if hist_inflows:
        # Sort by effective_date ascending, then source_row ascending
        sorted_inflows = sorted(hist_inflows, key=lambda ce: (ce.effective_date, ce.source_row))
        latest_inflow = sorted_inflows[-1]
        latest_income_date = latest_inflow.effective_date
        latest_income_amount = latest_inflow.amount_home

    # 30-day and 90-day historical income sums
    recent_30d_income = sum(
        (ce.amount_home for ce in hist_inflows if w30_start <= ce.effective_date < req_d),
        Decimal("0.00"),
    )
    recent_90d_income = sum(
        (ce.amount_home for ce in hist_inflows if w90_start <= ce.effective_date < req_d),
        Decimal("0.00"),
    )
    monthly_avg_income_90d = (recent_90d_income / Decimal("3")).quantize(q, rounding=ROUND_HALF_UP)

    # Outflow analysis
    # Only settled cash outflows represent realized historical spending
    hist_outflows = [
        ce for ce in historical_events
        if ce.direction == Direction.OUTFLOW
        and ce.is_cash_event
        and ce.cash_impact_type == CashImpactType.SETTLED_OUTFLOW
        and ce.amount_home is not None
        and ce.amount_home > Decimal("0")
    ]

    recent_30d_outflow = Decimal("0.00")
    recent_30d_essential = Decimal("0.00")
    recent_30d_discretionary = Decimal("0.00")

    prev_30d_outflow = Decimal("0.00")  # [req_d - 60d, req_d - 30d) for spending trend

    recent_90d_outflow = Decimal("0.00")
    recent_90d_essential = Decimal("0.00")
    recent_90d_discretionary = Decimal("0.00")

    for ce in hist_outflows:
        amt = ce.amount_home
        is_essential = ce.category in essential_categories_all

        # 30d window
        if w30_start <= ce.effective_date < req_d:
            recent_30d_outflow += amt
            if is_essential:
                recent_30d_essential += amt
            else:
                recent_30d_discretionary += amt

        # Previous 30d window
        if w60_start <= ce.effective_date < w30_start:
            prev_30d_outflow += amt

        # 90d window
        if w90_start <= ce.effective_date < req_d:
            recent_90d_outflow += amt
            if is_essential:
                recent_90d_essential += amt
            else:
                recent_90d_discretionary += amt

    recent_30d_net = recent_30d_income - recent_30d_outflow
    recent_90d_net = recent_90d_income - recent_90d_outflow
    monthly_avg_outflow_90d = (recent_90d_outflow / Decimal("3")).quantize(q, rounding=ROUND_HALF_UP)

    # 4. Verified Recurring Commitments (From Recurrence Engine)
    recurring_income_streams: List[RecurringStreamSummary] = []
    recurring_expense_streams: List[RecurringStreamSummary] = []
    total_monthly_rec_income = Decimal("0.00")
    total_monthly_rec_expenses = Decimal("0.00")

    for s in recurring_series:
        if s.user_id != request.user_id or s.is_cancelled:
            continue
        m_equiv = compute_monthly_equivalent(s.forecast_amount, s.frequency)
        stream_summary = RecurringStreamSummary(
            series_id=s.series_id,
            category=s.category,
            direction=s.direction,
            frequency=s.frequency,
            amount=s.forecast_amount,
            currency=s.currency,
            is_protected=s.is_protected,
            is_cancelled=s.is_cancelled,
            monthly_equivalent=m_equiv,
        )
        if s.direction == Direction.INFLOW:
            recurring_income_streams.append(stream_summary)
            total_monthly_rec_income += m_equiv
        elif s.direction == Direction.OUTFLOW:
            recurring_expense_streams.append(stream_summary)
            total_monthly_rec_expenses += m_equiv

    net_monthly_rec_cashflow = total_monthly_rec_income - total_monthly_rec_expenses
    has_val_rec_income = len(recurring_income_streams) > 0
    has_val_rec_expenses = len(recurring_expense_streams) > 0

    # 5. Baseline Simulation Summary (Directly from verified simulator result)
    baseline_min_cash = baseline_simulation.minimum_projected_available_cash
    baseline_min_date = baseline_simulation.minimum_cash_date
    baseline_ending_cash = baseline_simulation.ending_state.available_cash
    baseline_sf_breach = baseline_simulation.is_safety_floor_breached
    baseline_margin = baseline_min_cash - safety_floor
    baseline_headroom = max(Decimal("0.00"), baseline_min_cash - safety_floor)

    proj_inflows_90d = baseline_simulation.total_inflows
    proj_outflows_90d = baseline_simulation.total_outflows

    # 6. Upcoming Obligations in the 90-day horizon [request_date, request_date + 90 days]
    # Includes recurrence forecast outflows + canonical scheduled outflows
    upcoming_obligations_list: List[UpcomingObligationSummary] = []
    seen_ob_ids: Set[str] = set()

    sim_end_date = req_d + timedelta(days=90)
    w30_future_limit = req_d + timedelta(days=30)

    # From future recurrence forecast events
    for fe in future_events:
        if fe.user_id != request.user_id:
            continue
        if req_d <= fe.effective_date <= sim_end_date and fe.direction == Direction.OUTFLOW:
            if fe.event_id in seen_ob_ids:
                continue
            seen_ob_ids.add(fe.event_id)
            is_prot = fe.is_protected or (fe.category in protected_categories)
            upcoming_obligations_list.append(UpcomingObligationSummary(
                event_id=fe.event_id,
                effective_date=fe.effective_date,
                category=fe.category,
                amount=fe.amount_home,
                currency=fe.currency,
                source_type="forecast_recurrence",
                series_id=fe.series_id,
                is_protected=is_prot,
                flexibility=fe.flexibility,
            ))

    # From canonical scheduled outflows
    for ce in canonical_events:
        if ce.user_id != request.user_id:
            continue
        if req_d <= ce.effective_date <= sim_end_date and ce.direction == Direction.OUTFLOW:
            if ce.cash_impact_type == CashImpactType.SCHEDULED_OUTFLOW or ce.status == "scheduled":
                if ce.event_id in seen_ob_ids:
                    continue
                seen_ob_ids.add(ce.event_id)
                is_prot = ce.category in protected_categories
                upcoming_obligations_list.append(UpcomingObligationSummary(
                    event_id=ce.event_id,
                    effective_date=ce.effective_date,
                    category=ce.category,
                    amount=ce.amount_home if ce.amount_home is not None else Decimal("0.00"),
                    currency=ce.home_currency,
                    source_type="canonical_scheduled",
                    series_id=None,
                    is_protected=is_prot,
                    flexibility=ce.flexibility,
                ))

    # Sort obligations chronologically, then by event_id
    upcoming_obligations_list.sort(key=lambda ob: (ob.effective_date, ob.event_id))

    upcoming_count = len(upcoming_obligations_list)
    upcoming_30d_tot = sum(
        (ob.amount for ob in upcoming_obligations_list if ob.effective_date < w30_future_limit),
        Decimal("0.00"),
    )
    upcoming_90d_tot = sum(
        (ob.amount for ob in upcoming_obligations_list),
        Decimal("0.00"),
    )

    # 7. Behavioral & Stability Signals
    # A. Spending Trend
    if history_days < 60:
        spending_trend = SpendingTrend.INSUFFICIENT_DATA
        spending_trend_pct = None
    elif prev_30d_outflow == Decimal("0.00"):
        if recent_30d_outflow == Decimal("0.00"):
            spending_trend = SpendingTrend.STABLE
            spending_trend_pct = Decimal("0.00")
        else:
            spending_trend = SpendingTrend.INCREASING
            spending_trend_pct = None
    else:
        diff = recent_30d_outflow - prev_30d_outflow
        ratio = diff / prev_30d_outflow
        spending_trend_pct = (ratio * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if ratio > Decimal("0.15"):
            spending_trend = SpendingTrend.INCREASING
        elif ratio < Decimal("-0.15"):
            spending_trend = SpendingTrend.DECREASING
        else:
            spending_trend = SpendingTrend.STABLE

    # B. Income Stability
    if has_val_rec_income:
        # Check cadence of primary recurring income series
        primary_inc = recurring_income_streams[0]
        rec_matches = [s for s in recurring_series if s.series_id == primary_inc.series_id]
        if rec_matches and rec_matches[0].historical_count >= 3:
            income_stability = IncomeStability.HIGH
        else:
            income_stability = IncomeStability.MODERATE
    elif history_days < 60 or hist_income_count == 0:
        income_stability = IncomeStability.INSUFFICIENT_DATA
    else:
        # Evaluate 90d settled inflows
        inflows_90d = [ce.amount_home for ce in hist_inflows if w90_start <= ce.effective_date < req_d]
        if len(inflows_90d) >= 3:
            # Check coefficient of variation
            mean_inc = sum(inflows_90d) / Decimal(len(inflows_90d))
            if mean_inc > Decimal("0"):
                variance = sum(((amt - mean_inc) ** 2 for amt in inflows_90d)) / Decimal(len(inflows_90d))
                std_dev = Decimal(str(math.sqrt(float(variance))))
                cv = std_dev / mean_inc
                if cv < Decimal("0.25"):
                    income_stability = IncomeStability.MODERATE
                else:
                    income_stability = IncomeStability.LOW
            else:
                income_stability = IncomeStability.LOW
        elif len(inflows_90d) > 0:
            income_stability = IncomeStability.LOW
        else:
            income_stability = IncomeStability.INSUFFICIENT_DATA

    # C. Expense Stability
    if history_days < 30 or len(hist_outflows) == 0:
        expense_stability = ExpenseStability.INSUFFICIENT_DATA
    elif monthly_avg_outflow_90d > Decimal("0.00"):
        rec_share = total_monthly_rec_expenses / monthly_avg_outflow_90d
        if rec_share >= Decimal("0.50"):
            expense_stability = ExpenseStability.HIGH
        elif rec_share >= Decimal("0.20"):
            expense_stability = ExpenseStability.MODERATE
        else:
            expense_stability = ExpenseStability.LOW
    else:
        expense_stability = ExpenseStability.LOW

    # 8. Data Sufficiency Signals
    is_inc_sufficient = history_days >= 60 and hist_income_count >= 1
    is_exp_sufficient = history_days >= 60 and len(hist_outflows) >= 5

    return UserFinancialState(
        request_id=request.request_id,
        user_id=request.user_id,
        request_date=req_d,
        currency=curr,
        requested_amount=request.requested_amount,
        desired_completion_date=request.desired_completion_date,
        allows_partial_payment=request.allows_partial_payment,
        current_available_balance=current_avail_bal,
        safety_floor=safety_floor,
        minimum_balance_to_keep=min_bal,
        pending_reserved_amount=pending_reserved,
        available_cash=available_cash,
        current_headroom_above_floor=current_headroom,
        recent_30d_income=recent_30d_income,
        recent_30d_outflow=recent_30d_outflow,
        recent_30d_essential_outflow=recent_30d_essential,
        recent_30d_discretionary_outflow=recent_30d_discretionary,
        recent_30d_net_cashflow=recent_30d_net,
        recent_90d_income=recent_90d_income,
        recent_90d_outflow=recent_90d_outflow,
        recent_90d_essential_outflow=recent_90d_essential,
        recent_90d_discretionary_outflow=recent_90d_discretionary,
        recent_90d_net_cashflow=recent_90d_net,
        monthly_average_income_90d=monthly_avg_income_90d,
        monthly_average_outflow_90d=monthly_avg_outflow_90d,
        historical_income_count=hist_income_count,
        latest_income_date=latest_income_date,
        latest_income_amount=latest_income_amount,
        recurring_income_streams=tuple(recurring_income_streams),
        recurring_expense_streams=tuple(recurring_expense_streams),
        total_monthly_recurring_income=total_monthly_rec_income,
        total_monthly_recurring_expenses=total_monthly_rec_expenses,
        net_monthly_recurring_cashflow=net_monthly_rec_cashflow,
        has_validated_recurring_income=has_val_rec_income,
        has_validated_recurring_expenses=has_val_rec_expenses,
        baseline_minimum_available_cash=baseline_min_cash,
        baseline_minimum_cash_date=baseline_min_date,
        baseline_ending_available_cash=baseline_ending_cash,
        baseline_has_safety_breach=baseline_sf_breach,
        baseline_safety_margin=baseline_margin,
        baseline_headroom=baseline_headroom,
        total_projected_inflows_90d=proj_inflows_90d,
        total_projected_outflows_90d=proj_outflows_90d,
        upcoming_obligations_count=upcoming_count,
        upcoming_obligations_30d_total=upcoming_30d_tot,
        upcoming_obligations_90d_total=upcoming_90d_tot,
        upcoming_obligations=tuple(upcoming_obligations_list),
        spending_trend=spending_trend,
        spending_trend_percentage=spending_trend_pct,
        income_stability=income_stability,
        expense_stability=expense_stability,
        history_days_available=history_days,
        total_historical_events_count=total_hist_count,
        is_income_history_sufficient=is_inc_sufficient,
        is_expense_history_sufficient=is_exp_sufficient,
        has_unresolved_events=has_unresolved,
        unresolved_events_count=unresolved_count,
        financial_priorities=profile.financial_priorities,
        protected_categories=protected_categories,
        willing_to_reduce_categories=willing_to_reduce,
        willing_to_stop_categories=willing_to_stop,
        payment_methods_considered=payment_methods,
        max_installment_months=profile.max_installment_months,
    )


def evaluate_user_financial_state(
    request: FinancialRequest,
    profile: FinancialProfile,
    ledger: CanonicalLedger,
    all_series: Dict[str, Sequence[RecurrenceSeries]],
) -> UserFinancialState:
    """Convenience pipeline evaluator to construct UserFinancialState from the full ledger.

    Simulates the user baseline across [request_date, request_date + 90 days]
    and extracts the unified immutable financial state.
    """
    uid = request.user_id
    req_d = request.request_date
    sim_end = req_d + timedelta(days=90)

    user_canonical = ledger.get_events_for_user(uid)
    user_series = all_series.get(uid, ())

    future_res = expand_future_events(user_series, req_d, sim_end, ledger=ledger)

    baseline = simulate_user(
        user_id=uid,
        simulation_start=req_d,
        simulation_end=sim_end,
        canonical_events=user_canonical,
        future_events=future_res.future_events,
        profile=profile,
    )

    return build_user_financial_state(
        request=request,
        profile=profile,
        canonical_events=user_canonical,
        future_events=future_res.future_events,
        baseline_simulation=baseline,
        recurring_series=user_series,
    )
