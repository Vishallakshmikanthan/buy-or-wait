"""Deterministic evidence reconciliation and canonical ledger construction."""

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from .canonical import (
    CanonicalEvent,
    CanonicalLedger,
    CashImpactType,
    Direction,
    RecurrenceClassification,
)
from .models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageMetadata,
    Message,
)
from .validators import ValidationError


class ActionType(str, Enum):
    """Reconciliation action applied to an event based on evidence."""
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    AMEND_AMOUNT = "AMEND_AMOUNT"
    DELAY = "DELAY"
    CURRENCY_CONVERSION = "CURRENCY_CONVERSION"


class ExchangeRateNotFoundError(ValidationError):
    """Raised when an exact dated exchange rate is missing for a foreign-currency cash event."""
    pass


def _determine_effective_date(
    event_date: date,
    settlement_date: Optional[date],
    status: str,
) -> date:
    """Determine the deterministic effective cash-flow date for an event.

    Dataset & Problem Statement Semantics:
    1. Settled records: Settle on their `settlement_date` where available; fallback to `event_date`.
    2. Scheduled future records: Execute on their scheduled `settlement_date` where available; fallback to `event_date`.
    3. Pending debits: Reserved conservatively from `settlement_date` if present, else `event_date`.
    4. Non-cash / Unrealized records: Do not impact cash flow; use `event_date` for audit chronology.
    """
    if settlement_date is not None:
        return settlement_date
    return event_date


def _classify_cash_impact(
    status: str,
    direction_original: str,
    event_type: str,
) -> Tuple[bool, CashImpactType, Direction]:
    """Map raw event status and direction to deterministic cash-flow impact.

    Rules:
    - Cancelled / Failed: Never reduce or add cash (non-cash).
    - Unrealized / non_cash: Unrealized valuation is not available cash (non-cash).
    - Pending debits: Must be reserved conservatively (outflow).
    - Pending credits: Must NOT be counted as available future cash (non-cash).
    - Scheduled: Inflow for confirmed income; outflow for scheduled payments.
    - Settled: Inflow for credit; outflow for debit.
    """
    clean_status = status.strip().lower()
    clean_dir = direction_original.strip().lower()

    if clean_status == "cancelled":
        return False, CashImpactType.CANCELLED_IGNORED, Direction.NON_CASH

    if clean_status == "failed":
        return False, CashImpactType.FAILED_IGNORED, Direction.NON_CASH

    if clean_status == "unrealized" or clean_dir == "non_cash":
        return False, CashImpactType.UNREALIZED_NON_CASH, Direction.NON_CASH

    if clean_status == "pending":
        if clean_dir == "debit":
            return True, CashImpactType.PENDING_DEBIT_RESERVED, Direction.OUTFLOW
        else:
            # Pending credit/refund must not be counted as available cash
            return False, CashImpactType.PENDING_CREDIT_IGNORED, Direction.NON_CASH

    if clean_status == "scheduled":
        if clean_dir == "credit":
            return True, CashImpactType.SCHEDULED_INFLOW, Direction.INFLOW
        else:
            return True, CashImpactType.SCHEDULED_OUTFLOW, Direction.OUTFLOW

    if clean_status == "settled":
        if clean_dir == "credit":
            return True, CashImpactType.SETTLED_INFLOW, Direction.INFLOW
        else:
            return True, CashImpactType.SETTLED_OUTFLOW, Direction.OUTFLOW

    # Fallback for unrecognized status: treat conservatively as non-cash
    return False, CashImpactType.UNREALIZED_NON_CASH, Direction.NON_CASH


def _classify_basic_recurrence(
    event_type: str,
    description: str,
    evidence_notes: Sequence[str],
) -> RecurrenceClassification:
    """Basic recurrence classification based on explicit structural indicators."""
    clean_type = event_type.strip().lower()
    clean_desc = description.strip().lower()

    # Subscriptions are explicitly recurring
    if clean_type == "subscription":
        return RecurrenceClassification.EXPLICITLY_RECURRING

    # Single-instance transactional types
    if clean_type in ("refund", "investment_sale", "investment_purchase"):
        return RecurrenceClassification.ONE_TIME

    # Check evidence notes and description for explicit closure/one-time confirmation
    for note in evidence_notes:
        note_lower = note.lower()
        if "claim is now closed" in note_lower or "no further scheduled payments" in note_lower:
            return RecurrenceClassification.ONE_TIME
        if "reimbursement for your earlier work expense" in note_lower:
            return RecurrenceClassification.ONE_TIME

    if "one-time" in clean_desc or "onetime" in clean_desc:
        return RecurrenceClassification.ONE_TIME

    return RecurrenceClassification.UNKNOWN


def reconcile_single_event(
    event: FinancialEvent,
    home_currency: str,
    exchange_rate_map: Dict[Tuple[date, str, str], Decimal],
    linked_messages: Sequence[Message],
    linked_image: Optional[ImageMetadata] = None,
) -> CanonicalEvent:
    """Reconcile one raw FinancialEvent into a CanonicalEvent with evidence chain."""
    evidence: List[str] = [f"Source: financial_events.csv row {event.source_row}"]
    actions: List[str] = []

    # 1. Evaluate linked messages
    status = event.status
    amount_original = event.amount
    settlement_date = event.settlement_date
    event_date = event.event_date

    # Sort messages chronologically by sent_at for deterministic precedence
    sorted_messages = sorted(linked_messages, key=lambda m: m.sent_at)
    for msg in sorted_messages:
        text_lower = msg.message_text.lower()
        evidence.append(f"Message {msg.message_id} ({msg.source_type} at {msg.sent_at.isoformat()}): {msg.message_text}")

        # Check for explicit cancellation
        if any(w in text_lower for w in ["dibatalkan", "cancelled", "transaction cancelled", "payment cancelled"]):
            status = "cancelled"
            actions.append(ActionType.CANCEL.value)
        # Check for explicit confirmation of dispute, failure, or claim closure
        elif any(w in text_lower for w in ["investigated", "failed", "closed", "tidak ada", "selesai", "pending"]):
            actions.append(ActionType.CONFIRM.value)

    # 2. Check for missing image-backed amount
    is_unresolved = False
    unresolved_reason: Optional[str] = None
    if amount_original is None:
        is_unresolved = True
        if linked_image is not None:
            unresolved_reason = f"missing_amount_linked_to_image:{linked_image.image_id}"
            evidence.append(f"Requires image extraction from {linked_image.image_id}.png")
        else:
            unresolved_reason = "missing_amount_no_linked_image"
            evidence.append("Missing monetary amount without linked image record")

    # 3. Determine effective date
    effective_date = _determine_effective_date(event_date, settlement_date, status)

    # 4. Classify cash impact and direction
    is_cash, impact_type, direction = _classify_cash_impact(status, event.direction, event.event_type)

    # 5. Currency normalization
    amount_home: Optional[Decimal] = None
    rate_used: Optional[Decimal] = None
    rate_date: Optional[date] = None
    min_allowed_home: Optional[Decimal] = None

    if event.currency == home_currency:
        amount_home = amount_original
        min_allowed_home = event.minimum_allowed_amount
        rate_used = Decimal("1")
    else:
        # Foreign currency conversion
        rate_key = (effective_date, event.currency, home_currency)
        if rate_key in exchange_rate_map:
            rate_used = exchange_rate_map[rate_key]
            rate_date = effective_date
            actions.append(ActionType.CURRENCY_CONVERSION.value)
            evidence.append(
                f"Exchange rate {rate_used} ({event.currency}->{home_currency}) on {effective_date}"
            )
            if amount_original is not None:
                amount_home = amount_original * rate_used
            if event.minimum_allowed_amount is not None:
                min_allowed_home = event.minimum_allowed_amount * rate_used
        else:
            raise ExchangeRateNotFoundError(
                f"Missing exchange rate for {event.currency} -> {home_currency} on date {effective_date} "
                f"for event '{event.event_id}' (user '{event.user_id}')."
            )

    # 6. Basic recurrence classification
    recurrence = _classify_basic_recurrence(event.event_type, event.description, evidence)

    return CanonicalEvent(
        event_id=event.event_id,
        user_id=event.user_id,
        source_row=event.source_row,
        effective_date=effective_date,
        direction=direction,
        direction_original=event.direction,
        amount_original=amount_original,
        currency_original=event.currency,
        amount_home=amount_home,
        home_currency=home_currency,
        exchange_rate_used=rate_used,
        exchange_rate_date=rate_date,
        status=status,
        is_cash_event=is_cash,
        cash_impact_type=impact_type,
        event_type=event.event_type,
        category=event.category,
        description=event.description,
        flexibility=event.flexibility,
        minimum_allowed_amount_original=event.minimum_allowed_amount,
        minimum_allowed_amount_home=min_allowed_home,
        linked_event_id=event.linked_event_id,
        recurrence_type=recurrence,
        is_unresolved=is_unresolved,
        unresolved_reason=unresolved_reason,
        evidence_chain=tuple(evidence),
        applied_actions=tuple(actions),
    )


def reconcile_events(
    events: Sequence[FinancialEvent],
    profiles: Dict[str, FinancialProfile],
    exchange_rates: Sequence[ExchangeRate],
    messages: Sequence[Message],
    images: Sequence[ImageMetadata],
) -> CanonicalLedger:
    """Reconcile all raw financial events into a deterministic CanonicalLedger.

    Args:
        events: Raw FinancialEvent records from CSV.
        profiles: User financial profiles keyed by user_id.
        exchange_rates: Fixed dated conversion rates.
        messages: Supporting message evidence.
        images: Image metadata links.

    Returns:
        CanonicalLedger containing indexed, verified CanonicalEvents.
    """
    # Build lookup maps
    rate_map: Dict[Tuple[date, str, str], Decimal] = {
        (r.rate_date, r.from_currency, r.to_currency): r.rate for r in exchange_rates
    }

    messages_by_event: Dict[str, List[Message]] = {}
    for msg in messages:
        if msg.related_event_id:
            messages_by_event.setdefault(msg.related_event_id, []).append(msg)

    images_by_event: Dict[str, ImageMetadata] = {
        img.related_event_id: img for img in images
    }

    canonical_events: List[CanonicalEvent] = []
    for event in events:
        profile = profiles.get(event.user_id)
        if profile is None:
            raise ValidationError(
                f"Cannot reconcile event '{event.event_id}': User '{event.user_id}' not found in profiles."
            )

        linked_msgs = messages_by_event.get(event.event_id, [])
        linked_img = images_by_event.get(event.event_id)

        canon = reconcile_single_event(
            event=event,
            home_currency=profile.home_currency,
            exchange_rate_map=rate_map,
            linked_messages=linked_msgs,
            linked_image=linked_img,
        )
        canonical_events.append(canon)

    return CanonicalLedger(events=canonical_events)
