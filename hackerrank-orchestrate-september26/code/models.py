"""Data models for Buy or Wait? financial engine."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, Tuple


@dataclass(frozen=True)
class FinancialProfile:
    """User financial profile containing constraints, priorities, and preferences."""
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: Tuple[str, ...]
    expense_categories_to_protect: Tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: Tuple[str, ...]
    expense_categories_user_is_willing_to_stop: Tuple[str, ...]
    payment_methods_user_will_consider: Tuple[str, ...]
    max_installment_months: Optional[int]


@dataclass(frozen=True)
class FinancialEvent:
    """Canonical representation of a financial transaction or valuation event with provenance."""
    event_id: str
    source_row: int
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[Decimal]
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str
    linked_event_id: Optional[str]
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]

    @property
    def has_missing_amount(self) -> bool:
        """True if the monetary amount is missing and needs external/image resolution."""
        return self.amount is None


@dataclass(frozen=True)
class PaymentOption:
    """Seller/provider payment option available for a request."""
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True)
class ExchangeRate:
    """Fixed dated conversion rate between two currencies."""
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


@dataclass(frozen=True)
class Message:
    """Supporting communication/evidence from employers, banks, merchants, or providers."""
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: datetime
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageMetadata:
    """Linkage between an image file and a user, request, and financial event."""
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str


@dataclass(frozen=True)
class FinancialRequest:
    """User request being evaluated by the financial agent."""
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
