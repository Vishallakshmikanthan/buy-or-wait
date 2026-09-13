"""Deterministic evidence reconciliation and canonical ledger construction."""

from dataclasses import dataclass
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
from .message_interpretation import (
    MessageAction,
    MessageActionType,
    MessageReconciliationRecord,
    TargetType,
    UnresolvedMessageAction,
    interpret_and_link_messages,
    parse_single_message,
    resolve_message_conflicts,
)
from .models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    FinancialRequest,
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


@dataclass(frozen=True)
class ReconciliationResult:
    """Immutable result of reconciling raw events and message actions into a canonical ledger."""
    ledger: CanonicalLedger
    message_actions: Tuple[MessageAction, ...]
    unresolved_actions: Tuple[UnresolvedMessageAction, ...]
    reconciliation_records: Tuple[MessageReconciliationRecord, ...]


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
    linked_actions: Optional[Sequence[MessageAction]] = None,
) -> CanonicalEvent:
    """Reconcile one raw FinancialEvent into a CanonicalEvent with evidence chain.

    Applies structured MessageActions (CANCEL, AMEND_AMOUNT, DELAY_TO, CONFIRM)
    with full provenance retention in evidence_chain and applied_actions.
    """
    evidence: List[str] = [f"Source: financial_events.csv row {event.source_row}"]
    actions: List[str] = []

    # 1. Evaluate linked messages / actions
    status = event.status
    amount_original = event.amount
    settlement_date = event.settlement_date
    event_date = event.event_date

    # Determine actions to apply
    resolved_actions: List[MessageAction] = []
    if linked_actions is not None:
        resolved_actions = list(linked_actions)
    elif linked_messages:
        # Construct and resolve actions from linked messages
        raw_acts: List[MessageAction] = []
        for msg in linked_messages:
            act_type, amt, curr, eff_d, pct, reason = parse_single_message(msg)
            raw_acts.append(
                MessageAction(
                    message_id=msg.message_id,
                    action_type=act_type,
                    request_id=msg.request_id,
                    user_id=msg.user_id,
                    related_event_id=event.event_id,
                    target_type=TargetType.EVENT,
                    target_id=event.event_id,
                    target_description=event.description,
                    effective_date=eff_d or event.event_date,
                    new_amount=amt,
                    old_amount=event.amount,
                    currency=curr or event.currency,
                    confidence=Decimal("1.00"),
                    evidence_text_reference=f"Message {msg.message_id} ({msg.source_type} at {msg.sent_at.isoformat()}): {msg.message_text}",
                    reason=reason,
                    is_applied=True,
                )
            )
        resolved_actions = resolve_message_conflicts(raw_acts)

    for act in resolved_actions:
        evidence.append(
            f"Message {act.message_id} ({act.action_type.value}): {act.evidence_text_reference} - {act.reason}"
        )

        if act.action_type == MessageActionType.CANCEL:
            status = "cancelled"
            if ActionType.CANCEL.value not in actions:
                actions.append(ActionType.CANCEL.value)

        elif act.action_type == MessageActionType.AMEND_AMOUNT:
            if act.new_amount is not None and act.new_amount > Decimal("0"):
                orig_val = str(amount_original)
                amount_original = act.new_amount
                if ActionType.AMEND_AMOUNT.value not in actions:
                    actions.append(ActionType.AMEND_AMOUNT.value)
                evidence.append(f"Amount amended from {orig_val} to {act.new_amount} {act.currency or event.currency}")

        elif act.action_type == MessageActionType.DELAY_TO:
            if act.effective_date is not None:
                orig_d = str(settlement_date or event_date)
                settlement_date = act.effective_date
                if ActionType.DELAY.value not in actions:
                    actions.append(ActionType.DELAY.value)
                evidence.append(f"Effective date delayed from {orig_d} to {act.effective_date}")

        elif act.action_type == MessageActionType.CONFIRM:
            if ActionType.CONFIRM.value not in actions:
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
    if is_unresolved:
        # Unresolved events must never be marked as usable cash events
        is_cash = False

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


def reconcile_events_with_audit(
    events: Sequence[FinancialEvent],
    profiles: Dict[str, FinancialProfile],
    exchange_rates: Sequence[ExchangeRate],
    messages: Sequence[Message],
    images: Sequence[ImageMetadata],
    requests: Optional[Sequence[FinancialRequest]] = None,
) -> ReconciliationResult:
    """Reconcile raw events and messages into a CanonicalLedger with full audit lineage.

    Args:
        events: Raw FinancialEvent records from CSV.
        profiles: User financial profiles keyed by user_id.
        exchange_rates: Fixed dated conversion rates.
        messages: Supporting message evidence.
        images: Image metadata links.
        requests: Optional financial requests for linkage context.

    Returns:
        ReconciliationResult containing CanonicalLedger and complete message audit trail.
    """
    # Build lookup maps
    rate_map: Dict[Tuple[date, str, str], Decimal] = {
        (r.rate_date, r.from_currency, r.to_currency): r.rate for r in exchange_rates
    }

    images_by_event: Dict[str, ImageMetadata] = {
        img.related_event_id: img for img in images
    }

    # 1. Deterministic Message Interpretation & Linkage
    req_list = requests or []
    message_actions, unresolved_actions = interpret_and_link_messages(
        messages=messages,
        events=events,
        requests=req_list,
        profiles=profiles,
    )

    # Index event-targeted actions by target_id
    actions_by_event: Dict[str, List[MessageAction]] = {}
    for act in message_actions:
        if act.target_type == TargetType.EVENT and act.target_id:
            actions_by_event.setdefault(act.target_id, []).append(act)

    # Fallback message mapping by related_event_id for backward compatibility
    messages_by_event: Dict[str, List[Message]] = {}
    for msg in messages:
        if msg.related_event_id:
            messages_by_event.setdefault(msg.related_event_id, []).append(msg)

    # 2. Reconcile events
    canonical_events: List[CanonicalEvent] = []
    audit_records: List[MessageReconciliationRecord] = []

    for event in events:
        profile = profiles.get(event.user_id)
        if profile is None:
            raise ValidationError(
                f"Cannot reconcile event '{event.event_id}': User '{event.user_id}' not found in profiles."
            )

        linked_msgs = messages_by_event.get(event.event_id, [])
        linked_img = images_by_event.get(event.event_id)
        event_acts = actions_by_event.get(event.event_id, [])

        canon = reconcile_single_event(
            event=event,
            home_currency=profile.home_currency,
            exchange_rate_map=rate_map,
            linked_messages=linked_msgs,
            linked_image=linked_img,
            linked_actions=event_acts if event_acts else None,
        )
        canonical_events.append(canon)

        # Record audit entries for applied actions
        for act in event_acts:
            audit_records.append(
                MessageReconciliationRecord(
                    message_id=act.message_id,
                    action_type=act.action_type.value,
                    target_id=event.event_id,
                    original_value=str(act.old_amount) if act.old_amount is not None else None,
                    new_value=str(act.new_amount) if act.new_amount is not None else None,
                    effective_date=act.effective_date,
                    reason=act.reason or "applied_event_action",
                )
            )

    ledger = CanonicalLedger(events=canonical_events)
    return ReconciliationResult(
        ledger=ledger,
        message_actions=tuple(message_actions),
        unresolved_actions=tuple(unresolved_actions),
        reconciliation_records=tuple(audit_records),
    )


def reconcile_events(
    events: Sequence[FinancialEvent],
    profiles: Dict[str, FinancialProfile],
    exchange_rates: Sequence[ExchangeRate],
    messages: Sequence[Message],
    images: Sequence[ImageMetadata],
    requests: Optional[Sequence[FinancialRequest]] = None,
) -> CanonicalLedger:
    """Reconcile all raw financial events into a deterministic CanonicalLedger.

    Backward-compatible entry point returning CanonicalLedger.
    """
    return reconcile_events_with_audit(
        events=events,
        profiles=profiles,
        exchange_rates=exchange_rates,
        messages=messages,
        images=images,
        requests=requests,
    ).ledger
