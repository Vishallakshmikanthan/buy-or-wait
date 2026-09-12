"""Deterministic recurrence detection and future-event expansion layer.

Transforms a historical canonical ledger into validated recurring series and expands
future cash-flow occurrences across an arbitrary forecast horizon.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from code.canonical import CanonicalEvent, CanonicalLedger, Direction
from code.loaders import FinancialProfile, Message


class RecurrenceFrequency(str, Enum):
    """Supported recurrence cadences reliably detectable from financial history."""
    WEEKLY = "weekly"          # Interval ~ 7 days (6..8)
    BIWEEKLY = "biweekly"      # Interval ~ 14 days (13..16)
    TRIWEEKLY = "triweekly"    # Interval ~ 21 days (20..22)
    MONTHLY = "monthly"        # Calendar monthly (27..32 days or same day of month)


class AmountForecastingRule(str, Enum):
    """Documented deterministic rules for forecasting future event amounts."""
    EXACT_STABLE = "exact_stable"
    CONSERVATIVE_UPPER_MEDIAN_EXPENSE = "conservative_upper_median_expense"
    CONSERVATIVE_LOWER_MEDIAN_INCOME = "conservative_lower_median_income"
    EXPLICIT_AMENDMENT = "explicit_amendment"


@dataclass(frozen=True)
class RecurrenceSeries:
    """Internal certificate representing a verified historical recurring series."""
    series_id: str
    user_id: str
    direction: Direction
    category: str
    event_type: str
    description: str
    frequency: RecurrenceFrequency
    interval_days: Optional[int]
    day_of_month: Optional[int]
    historical_event_ids: Tuple[str, ...]
    historical_count: int
    anchor_event_id: str
    anchor_date: date
    forecast_amount: Decimal
    currency: str
    amount_rule: str
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]
    is_protected: bool = False
    is_cancelled: bool = False
    cancellation_reason: Optional[str] = None
    amendment_reason: Optional[str] = None
    evidence_rationale: str = ""


@dataclass(frozen=True)
class FutureEvent:
    """Forecast canonical-like financial event generated from a recurring series."""
    event_id: str
    user_id: str
    effective_date: date
    direction: Direction
    amount_home: Decimal
    currency: str
    category: str
    event_type: str
    description: str
    series_id: str
    frequency: RecurrenceFrequency
    is_forecast: bool = True
    anchor_event_id: str = ""
    flexibility: str = "fixed"
    minimum_allowed_amount: Optional[Decimal] = None
    is_protected: bool = False
    source_evidence: Tuple[str, ...] = ()

    @property
    def is_numerically_usable_cash_event(self) -> bool:
        return self.amount_home is not None and self.amount_home > Decimal("0")

    def get_usable_cash_amount(self) -> Decimal:
        return self.amount_home if self.amount_home is not None else Decimal("0")


@dataclass(frozen=True)
class SuppressedForecast:
    """Audit record when a forecast occurrence is suppressed due to an explicit event."""
    series_id: str
    user_id: str
    forecast_date: date
    conflicting_event_id: str
    reason: str


@dataclass
class FutureExpansionResult:
    """Container holding expanded future events, suppressed duplicates, and series metadata."""
    future_events: List[FutureEvent]
    suppressed_forecasts: List[SuppressedForecast]
    detected_series: List[RecurrenceSeries]
    rejected_candidates: List[Dict[str, Any]]


def _calculate_forecast_amount(
    events: List[CanonicalEvent],
    direction: Direction,
) -> Tuple[Decimal, str]:
    """Deterministically determine future amount from historical observations.

    Rules:
    - If all historical amounts are identical: use that exact stable amount.
    - If amounts vary:
        - For OUTFLOW (expense): use upper median (exact median if odd, higher of two if even).
          Conservative for expenses because obligations are not underestimated, using actual observations.
        - For INFLOW (income): use lower median (exact median if odd, lower of two if even).
          Conservative for income because income is not overestimated, using actual observations.
    """
    amounts = sorted([e.get_usable_cash_amount() for e in events])
    if not amounts:
        return Decimal("0"), "zero_fallback"

    if len(set(amounts)) == 1:
        return amounts[0], AmountForecastingRule.EXACT_STABLE.value

    n = len(amounts)
    mid = n // 2
    if n % 2 == 1:
        median_val = amounts[mid]
    else:
        if direction == Direction.OUTFLOW:
            median_val = amounts[mid]  # higher of the two median elements
        else:
            median_val = amounts[mid - 1]  # lower of the two median elements

    rule = (
        AmountForecastingRule.CONSERVATIVE_UPPER_MEDIAN_EXPENSE.value
        if direction == Direction.OUTFLOW
        else AmountForecastingRule.CONSERVATIVE_LOWER_MEDIAN_INCOME.value
    )
    return median_val, rule


def _is_consecutive_monthly(sorted_dates: List[date]) -> Tuple[bool, Optional[int]]:
    """Verify that dates advance month by month in strictly consecutive calendar months.

    Rules:
    - Successive observations must advance by exactly 1 calendar month.
    - Tolerate legitimate month-end clamping (e.g. Jan 31 -> Feb 28/29 -> Mar 31).
    - Tolerate at most +/- 2 days for weekend/banking holiday shifts.
    - Reject skipped months (e.g. Jan 15, Mar 15, May 15).
    """
    if len(sorted_dates) < 3:
        return False, None

    # 1. Strictly consecutive calendar months
    for i in range(len(sorted_dates) - 1):
        d1, d2 = sorted_dates[i], sorted_dates[i + 1]
        m_diff = (d2.year * 12 + d2.month) - (d1.year * 12 + d1.month)
        if m_diff != 1:
            return False, None

    # 2. Day of month consistency and month-end clamping
    days = [d.day for d in sorted_dates]
    base_day = max(days)  # e.g., 31 for month-end clamped to 28/29/30

    for d in sorted_dates:
        dim = calendar.monthrange(d.year, d.month)[1]
        expected_day = min(base_day, dim)
        if abs(d.day - expected_day) > 2:
            return False, None

    # 3. Overall day differences must fall within legitimate month boundaries [27, 33]
    diffs = [(sorted_dates[i + 1] - sorted_dates[i]).days for i in range(len(sorted_dates) - 1)]
    if not all(27 <= diff <= 33 for diff in diffs):
        return False, None

    return True, base_day


def _detect_frequency(sorted_dates: List[date]) -> Tuple[Optional[RecurrenceFrequency], Optional[int], Optional[int]]:
    """Determine recurrence frequency and parameters from sorted event dates.

    Returns:
        (frequency, interval_days, day_of_month)
    """
    if len(sorted_dates) < 3:
        return None, None, None

    diffs = [(sorted_dates[i + 1] - sorted_dates[i]).days for i in range(len(sorted_dates) - 1)]

    # Weekly: all intervals in [6, 8]
    if all(6 <= d <= 8 for d in diffs):
        return RecurrenceFrequency.WEEKLY, 7, None

    # Biweekly: all intervals in [13, 16]
    if all(13 <= d <= 16 for d in diffs):
        return RecurrenceFrequency.BIWEEKLY, 14, None

    # Triweekly: all intervals in [20, 22] (e.g. 21-day reducible dining)
    if all(20 <= d <= 22 for d in diffs):
        return RecurrenceFrequency.TRIWEEKLY, 21, None

    # Monthly: strictly consecutive calendar months with calendar-aware progression
    is_monthly, day_of_month = _is_consecutive_monthly(sorted_dates)
    if is_monthly:
        return RecurrenceFrequency.MONTHLY, None, day_of_month

    return None, None, None


def detect_recurrence_for_user(
    ledger: CanonicalLedger,
    user_id: str,
    profile: Optional[FinancialProfile] = None,
    messages: Optional[List[Message]] = None,
) -> Tuple[List[RecurrenceSeries], List[Dict[str, Any]]]:
    """Detect recurring series for a single user from historical usable cash events.

    Thresholds:
    - Minimum observations: >= 3
    - Usable cash events only (unresolved, non-cash, failed, cancelled events excluded)
    - Rejects irregular or noise candidate series
    - Applies message amendments / cancellations supported by evidence
    """
    # 1. Filter usable cash events for user
    user_events = [
        e for e in ledger.events
        if e.user_id == user_id and e.is_numerically_usable_cash_event
    ]

    # 2. Group candidate series
    # Discretionary flexible dining can have varying descriptions; group by (category, flexibility, minimum_amount)
    # All other events group by (description, direction, category, flexibility)
    groups: Dict[Tuple[str, ...], List[CanonicalEvent]] = {}
    for e in user_events:
        if e.category == "dining" and e.flexibility == "reducible":
            key = (
                "reducible_dining",
                e.direction.value,
                e.category,
                e.flexibility,
                str(e.minimum_allowed_amount_home),
            )
        else:
            key = (
                e.description.strip(),
                e.direction.value,
                e.category,
                e.flexibility,
                str(e.minimum_allowed_amount_home),
            )
        if key not in groups:
            groups[key] = []
        groups[key].append(e)

    detected_series: List[RecurrenceSeries] = []
    rejected_candidates: List[Dict[str, Any]] = []

    # Sort keys deterministically for reproducible iteration
    for key in sorted(groups.keys()):
        events = groups[key]
        desc_label, dir_str, cat, flex, min_amt_str = key
        direction = Direction(dir_str)

        # Observation threshold: at least 3 historical occurrences
        if len(events) < 3:
            rejected_candidates.append({
                "user_id": user_id,
                "key": key,
                "count": len(events),
                "reason": "insufficient_observations (< 3)",
            })
            continue

        sorted_events = sorted(events, key=lambda x: (x.effective_date, x.event_id))
        sorted_dates = [e.effective_date for e in sorted_events]

        freq, interval_days, day_of_month = _detect_frequency(sorted_dates)
        if not freq:
            rejected_candidates.append({
                "user_id": user_id,
                "key": key,
                "count": len(events),
                "reason": "irregular_intervals",
                "diffs": [(sorted_dates[i + 1] - sorted_dates[i]).days for i in range(len(sorted_dates) - 1)],
            })
            continue

        # Valid recurrence found!
        latest_event = sorted_events[-1]
        anchor_event_id = latest_event.event_id
        anchor_date = latest_event.effective_date
        currency = latest_event.home_currency

        # Amount rule
        forecast_amt, amt_rule = _calculate_forecast_amount(sorted_events, direction)

        # Series ID: deterministic slug
        # Format: rec_{user_id}_{cat}_{anchor_event_id}
        series_id = f"rec_{user_id}_{cat}_{anchor_event_id}"

        # Protected status from profile
        is_protected = False
        if profile and profile.expense_categories_to_protect:
            if cat in profile.expense_categories_to_protect:
                is_protected = True

        minimum_allowed = (
            latest_event.minimum_allowed_amount_home
            if latest_event.minimum_allowed_amount_home is not None
            else None
        )

        rationale = (
            f"Detected {freq.value} recurrence across {len(events)} observations "
            f"from {sorted_dates[0]} to {sorted_dates[-1]}. Amount rule: {amt_rule}."
        )

        # Message adjustments (amendments/cancellations)
        is_cancelled = False
        cancellation_reason = None
        amendment_reason = None

        if messages:
            for m in messages:
                if m.user_id == user_id:
                    txt = m.message_text
                    # Check contract ended / employment termination
                    if cat == "salary" and (
                        "seasonal contract has ended" in txt
                        or "Hubungan kerja Anda telah berakhir" in txt
                        or "contract has ended" in txt.lower()
                        or "employment has ended" in txt.lower()
                    ):
                        is_cancelled = True
                        cancellation_reason = f"contract_ended_{m.message_id}"
                        rationale += f" Cancelled by employer update ({m.message_id})."

                    # Check salary increase / decrease amendment
                    elif cat == "salary" and m.source_type == "employer":
                        # Match: "Your monthly salary has increased to EUR 1188"
                        # Match: "Gaji bulanan Anda naik menjadi IDR 42750000"
                        # Match: "Your temporary monthly pay is EUR 1037.52"
                        amt_match = re.search(
                            r"(?:increased to|naik menjadi|monthly pay is|reduced to|salary of|gaji pokok.*adalah)\s+([A-Z]{3})\s+([\d,.]+)",
                            txt,
                            re.IGNORECASE,
                        )
                        if amt_match:
                            try:
                                raw_curr, raw_val = amt_match.group(1), amt_match.group(2).replace(",", "")
                                parsed_amt = Decimal(raw_val)
                                if raw_curr.upper() == currency.upper():
                                    forecast_amt = parsed_amt
                                    amt_rule = AmountForecastingRule.EXPLICIT_AMENDMENT.value
                                    amendment_reason = f"Salary amended by employer in {m.message_id}"
                                    rationale += f" Amended to {parsed_amt} {currency} via {m.message_id}."
                            except Exception:
                                pass

                    # Check rent increase
                    elif cat == "rent" and ("increases monthly rent by" in txt or "kenaikan sewa" in txt):
                        pct_match = re.search(r"(\d+)%", txt)
                        if pct_match:
                            pct = Decimal(pct_match.group(1)) / Decimal("100")
                            forecast_amt = (forecast_amt * (Decimal("1") + pct)).quantize(Decimal("0.01"))
                            amt_rule = AmountForecastingRule.EXPLICIT_AMENDMENT.value
                            amendment_reason = f"Rent increased by {pct_match.group(1)}% in {m.message_id}"
                            rationale += f" Amended with {pct_match.group(1)}% increase via {m.message_id}."

        series = RecurrenceSeries(
            series_id=series_id,
            user_id=user_id,
            direction=direction,
            category=cat,
            event_type=latest_event.event_type,
            description=latest_event.description,
            frequency=freq,
            interval_days=interval_days,
            day_of_month=day_of_month,
            historical_event_ids=tuple(e.event_id for e in sorted_events),
            historical_count=len(events),
            anchor_event_id=anchor_event_id,
            anchor_date=anchor_date,
            forecast_amount=forecast_amt,
            currency=currency,
            amount_rule=amt_rule,
            flexibility=flex,
            minimum_allowed_amount=minimum_allowed,
            is_protected=is_protected,
            is_cancelled=is_cancelled,
            cancellation_reason=cancellation_reason,
            amendment_reason=amendment_reason,
            evidence_rationale=rationale,
        )
        detected_series.append(series)

    return detected_series, rejected_candidates


def detect_all_recurrence(
    ledger: CanonicalLedger,
    profiles: Optional[Dict[str, FinancialProfile]] = None,
    messages: Optional[List[Message]] = None,
) -> Tuple[Dict[str, List[RecurrenceSeries]], List[Dict[str, Any]]]:
    """Detect recurring series across all users in the canonical ledger."""
    all_users = sorted(list(set(e.user_id for e in ledger.events)))
    all_series: Dict[str, List[RecurrenceSeries]] = {}
    all_rejected: List[Dict[str, Any]] = []

    for uid in all_users:
        prof = profiles.get(uid) if profiles else None
        user_msgs = [m for m in messages if m.user_id == uid] if messages else None
        series_list, rejected = detect_recurrence_for_user(ledger, uid, prof, user_msgs)
        all_series[uid] = series_list
        all_rejected.extend(rejected)

    return all_series, all_rejected


def _generate_candidate_dates(
    series: RecurrenceSeries,
    start_date: date,
    end_date: date,
) -> List[date]:
    """Generate candidate recurrence dates between start_date and end_date."""
    dates: List[date] = []
    if series.frequency == RecurrenceFrequency.MONTHLY:
        day = series.day_of_month or series.anchor_date.day
        # Start from the month of start_date or anchor_date
        cur_year = series.anchor_date.year
        cur_month = series.anchor_date.month

        # Step forward month by month
        while True:
            cur_month += 1
            if cur_month > 12:
                cur_month = 1
                cur_year += 1
            days_in_month = calendar.monthrange(cur_year, cur_month)[1]
            actual_day = min(day, days_in_month)
            d = date(cur_year, cur_month, actual_day)
            if d > end_date:
                break
            if d >= start_date:
                dates.append(d)
    else:
        # Fixed-day interval (weekly, biweekly, triweekly)
        step = series.interval_days or 7
        cur_date = series.anchor_date
        while True:
            cur_date += timedelta(days=step)
            if cur_date > end_date:
                break
            if cur_date >= start_date:
                dates.append(cur_date)

    return dates


def _is_genuine_duplicate(
    series: RecurrenceSeries,
    candidate_date: date,
    explicit_event: CanonicalEvent,
) -> bool:
    """Determine whether an explicit canonical event represents the same obligation.

    Safety Rules:
    1. Must match user_id and direction.
    2. Event status MUST be 'scheduled' (not generic pending or past settled).
    3. Date must be close (|explicit.effective_date - candidate_date| <= 3).
    4. Category must match.
    5. Semantic obligation compatibility:
       - Salary: explicit event description must relate to salary/payroll (e.g. 'Next confirmed salary').
       - Expenses: explicit event must be a scheduled obligation (e.g. starts with 'Scheduled', 'bill payment retry')
         or share matching description. Unrelated transactions in the same category must NOT suppress recurring series.
       - Amount compatibility: if amounts exist, must be reasonably compatible (within 35% variance).
    """
    if explicit_event.user_id != series.user_id:
        return False
    if explicit_event.direction != series.direction:
        return False
    if explicit_event.status != "scheduled":
        return False
    if abs((explicit_event.effective_date - candidate_date).days) > 3:
        return False
    if explicit_event.category != series.category:
        return False

    exp_desc = explicit_event.description.lower()
    ser_desc = series.description.lower()

    if series.category == "salary":
        return "salary" in exp_desc or "payroll" in exp_desc

    # For expenses: must denote scheduled obligation or share description keywords
    if "scheduled" in exp_desc or "bill payment" in exp_desc or ser_desc in exp_desc or exp_desc in ser_desc:
        if explicit_event.amount_home and series.forecast_amount > Decimal("0"):
            diff_pct = abs(explicit_event.amount_home - series.forecast_amount) / series.forecast_amount
            return diff_pct <= Decimal("0.35")
        return True

    return False


def expand_future_events(
    series_list: List[RecurrenceSeries],
    start_date: date,
    end_date: date,
    ledger: Optional[CanonicalLedger] = None,
) -> FutureExpansionResult:
    """Expand recurring series into future events over [start_date, end_date].

    Prevents duplication against explicit events in the canonical ledger.
    """
    future_events: List[FutureEvent] = []
    suppressed: List[SuppressedForecast] = []

    # Map existing canonical events by (user_id, direction, category) for deduplication
    explicit_events_by_key: Dict[Tuple[str, str, str], List[CanonicalEvent]] = {}
    if ledger:
        for e in ledger.events:
            # Check explicit events in or around the horizon
            if e.is_numerically_usable_cash_event and e.effective_date >= start_date - timedelta(days=7):
                k = (e.user_id, e.direction.value, e.category)
                if k not in explicit_events_by_key:
                    explicit_events_by_key[k] = []
                explicit_events_by_key[k].append(e)

    for series in series_list:
        # Suppress cancelled series entirely
        if series.is_cancelled:
            continue

        candidate_dates = _generate_candidate_dates(series, start_date, end_date)
        for cand_date in candidate_dates:
            # Check duplicate prevention
            conflicting_event = None
            key = (series.user_id, series.direction.value, series.category)
            if key in explicit_events_by_key:
                for explicit in explicit_events_by_key[key]:
                    if _is_genuine_duplicate(series, cand_date, explicit):
                        conflicting_event = explicit
                        break

            if conflicting_event:
                suppressed.append(SuppressedForecast(
                    series_id=series.series_id,
                    user_id=series.user_id,
                    forecast_date=cand_date,
                    conflicting_event_id=conflicting_event.event_id,
                    reason=f"superseded_by_explicit_{conflicting_event.status}_event",
                ))
                continue

            # Create deterministic future event
            date_str = cand_date.strftime("%Y%m%d")
            event_id = f"forecast_{series.series_id}_{date_str}"

            future_event = FutureEvent(
                event_id=event_id,
                user_id=series.user_id,
                effective_date=cand_date,
                direction=series.direction,
                amount_home=series.forecast_amount,
                currency=series.currency,
                category=series.category,
                event_type=series.event_type,
                description=f"Recurring {series.description} ({series.frequency.value})",
                series_id=series.series_id,
                frequency=series.frequency,
                is_forecast=True,
                anchor_event_id=series.anchor_event_id,
                flexibility=series.flexibility,
                minimum_allowed_amount=series.minimum_allowed_amount,
                is_protected=series.is_protected,
                source_evidence=(
                    f"Generated from {series.series_id} (anchor {series.anchor_event_id}). "
                    f"Rule: {series.amount_rule}."
                ),
            )
            future_events.append(future_event)

    # Sort future events deterministically by effective_date then event_id
    future_events.sort(key=lambda x: (x.effective_date, x.event_id))

    return FutureExpansionResult(
        future_events=future_events,
        suppressed_forecasts=suppressed,
        detected_series=series_list,
        rejected_candidates=[],
    )
