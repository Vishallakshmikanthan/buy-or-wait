"""Canonical financial ledger and event representation."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class Direction(str, Enum):
    """Normalized cash movement direction."""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    NON_CASH = "non_cash"


class CashImpactType(str, Enum):
    """Categorization of how an event impacts available liquidity."""
    SETTLED_INFLOW = "SETTLED_INFLOW"
    SETTLED_OUTFLOW = "SETTLED_OUTFLOW"
    SCHEDULED_INFLOW = "SCHEDULED_INFLOW"
    SCHEDULED_OUTFLOW = "SCHEDULED_OUTFLOW"
    PENDING_DEBIT_RESERVED = "PENDING_DEBIT_RESERVED"
    PENDING_CREDIT_IGNORED = "PENDING_CREDIT_IGNORED"
    FAILED_IGNORED = "FAILED_IGNORED"
    CANCELLED_IGNORED = "CANCELLED_IGNORED"
    UNREALIZED_NON_CASH = "UNREALIZED_NON_CASH"


class RecurrenceClassification(str, Enum):
    """Basic recurrence classification prior to full historical recurrence modeling."""
    EXPLICITLY_RECURRING = "explicitly_recurring"
    ONE_TIME = "one_time"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CanonicalEvent:
    """Immutable reconciled financial event with full provenance and cash-flow attributes."""
    event_id: str
    user_id: str
    source_row: int
    effective_date: date
    direction: Direction
    direction_original: str
    amount_original: Optional[Decimal]
    currency_original: str
    amount_home: Optional[Decimal]
    home_currency: str
    exchange_rate_used: Optional[Decimal]
    exchange_rate_date: Optional[date]
    status: str
    is_cash_event: bool
    cash_impact_type: CashImpactType
    event_type: str
    category: str
    description: str
    flexibility: str
    minimum_allowed_amount_original: Optional[Decimal]
    minimum_allowed_amount_home: Optional[Decimal]
    linked_event_id: Optional[str]
    recurrence_type: RecurrenceClassification
    is_unresolved: bool
    unresolved_reason: Optional[str]
    evidence_chain: Tuple[str, ...]
    applied_actions: Tuple[str, ...]

    @property
    def requires_image_extraction(self) -> bool:
        """True if the event's amount is missing and needs OCR/vision resolution from media."""
        return self.is_unresolved and self.amount_original is None

    @property
    def is_numerically_usable_cash_event(self) -> bool:
        """True if and only if the event is a cash event, is resolved, and has a non-None amount_home."""
        return self.is_cash_event and (not self.is_unresolved) and (self.amount_home is not None)

    def get_usable_cash_amount(self) -> Decimal:
        """Return the numerical cash amount. Unresolved or non-cash events strictly return Decimal(0)."""
        if not self.is_numerically_usable_cash_event or self.amount_home is None:
            return Decimal("0")
        return self.amount_home


@dataclass
class CanonicalLedger:
    """Unified container for reconciled canonical events with deterministic indexing."""
    events: List[CanonicalEvent]

    # Precomputed index mappings
    events_by_id: Dict[str, CanonicalEvent] = field(init=False)
    events_by_user: Dict[str, List[CanonicalEvent]] = field(init=False)

    def __post_init__(self) -> None:
        """Build indexes for fast user and ID lookups."""
        # Ensure deterministic ordering: effective_date ascending, then event_id ascending
        self.events = sorted(self.events, key=lambda e: (e.effective_date, e.event_id))
        self.events_by_id = {e.event_id: e for e in self.events}

        by_user: Dict[str, List[CanonicalEvent]] = {}
        for e in self.events:
            by_user.setdefault(e.user_id, []).append(e)
        self.events_by_user = by_user

    def get_cash_events(self, user_id: Optional[str] = None) -> List[CanonicalEvent]:
        """Return only events that have a direct, numerically usable cash-flow impact."""
        pool = self.events_by_user.get(user_id, []) if user_id else self.events
        return [e for e in pool if e.is_numerically_usable_cash_event]

    def get_non_cash_events(self, user_id: Optional[str] = None) -> List[CanonicalEvent]:
        """Return non-cash, ignored, failed, cancelled, unrealized, or unresolved events."""
        pool = self.events_by_user.get(user_id, []) if user_id else self.events
        return [e for e in pool if not e.is_numerically_usable_cash_event]

    def get_unresolved_events(self, user_id: Optional[str] = None) -> List[CanonicalEvent]:
        """Return events requiring external/image evidence extraction."""
        pool = self.events_by_user.get(user_id, []) if user_id else self.events
        return [e for e in pool if e.is_unresolved]

    def get_events_for_user(self, user_id: str) -> List[CanonicalEvent]:
        """Return all canonical events for a specific user in chronological order."""
        return self.events_by_user.get(user_id, [])
