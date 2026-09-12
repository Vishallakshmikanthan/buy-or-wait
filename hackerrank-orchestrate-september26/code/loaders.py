"""Deterministic CSV loaders for all dataset entities with strict schema validation."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    FinancialRequest,
    ImageMetadata,
    Message,
    PaymentOption,
)
from .parsers import (
    parse_bool,
    parse_date,
    parse_datetime,
    parse_decimal,
    parse_int,
    parse_pipe_tuple,
)
from .validators import ValidationError, validate_columns

PROFILE_COLUMNS = [
    "user_id",
    "home_currency",
    "current_available_balance",
    "minimum_balance_to_keep",
    "financial_priorities",
    "expense_categories_to_protect",
    "expense_categories_user_is_willing_to_reduce",
    "expense_categories_user_is_willing_to_stop",
    "payment_methods_user_will_consider",
    "max_installment_months",
]

EVENT_COLUMNS = [
    "event_id",
    "user_id",
    "event_type",
    "description",
    "category",
    "direction",
    "amount",
    "currency",
    "event_date",
    "settlement_date",
    "status",
    "linked_event_id",
    "flexibility",
    "minimum_allowed_amount",
]

PAYMENT_OPTION_COLUMNS = [
    "payment_option_id",
    "request_id",
    "payment_method",
    "payment_amount",
    "number_of_payments",
    "first_payment_date",
    "payment_frequency_days",
    "financing_fee",
    "total_payable_amount",
]

EXCHANGE_RATE_COLUMNS = [
    "rate_date",
    "from_currency",
    "to_currency",
    "rate",
]

MESSAGE_COLUMNS = [
    "message_id",
    "user_id",
    "request_id",
    "related_event_id",
    "sent_at",
    "source_type",
    "message_text",
]

IMAGE_COLUMNS = [
    "image_id",
    "user_id",
    "request_id",
    "related_event_id",
]

REQUEST_COLUMNS = [
    "request_id",
    "user_id",
    "request_date",
    "request_type",
    "requested_amount",
    "desired_completion_date",
    "allows_partial_payment",
    "request_text",
]


def _resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a filesystem path to an absolute Path object."""
    return Path(path).resolve()


def load_profiles(filepath: Union[str, Path]) -> Dict[str, FinancialProfile]:
    """Load user financial profiles from CSV into a dictionary keyed by user_id."""
    path = _resolve_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Financial profiles file not found: {path}")

    profiles: Dict[str, FinancialProfile] = {}
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        validate_columns(reader.fieldnames, PROFILE_COLUMNS, path.name)

        for row_idx, row in enumerate(reader, start=2):
            user_id = row["user_id"].strip()
            if not user_id:
                raise ValidationError(f"Missing user_id at row {row_idx} in {path.name}")
            if user_id in profiles:
                raise ValidationError(f"Duplicate user_id '{user_id}' at row {row_idx} in {path.name}")

            profile = FinancialProfile(
                user_id=user_id,
                home_currency=row["home_currency"].strip(),
                current_available_balance=parse_decimal(
                    row["current_available_balance"], "current_available_balance", allow_none=False, row_num=row_idx
                ),
                minimum_balance_to_keep=parse_decimal(
                    row["minimum_balance_to_keep"], "minimum_balance_to_keep", allow_none=False, row_num=row_idx
                ),
                financial_priorities=parse_pipe_tuple(row["financial_priorities"]),
                expense_categories_to_protect=parse_pipe_tuple(row["expense_categories_to_protect"]),
                expense_categories_user_is_willing_to_reduce=parse_pipe_tuple(
                    row["expense_categories_user_is_willing_to_reduce"]
                ),
                expense_categories_user_is_willing_to_stop=parse_pipe_tuple(
                    row["expense_categories_user_is_willing_to_stop"]
                ),
                payment_methods_user_will_consider=parse_pipe_tuple(row["payment_methods_user_will_consider"]),
                max_installment_months=parse_int(
                    row["max_installment_months"], "max_installment_months", allow_none=True, row_num=row_idx
                ),
            )
            profiles[user_id] = profile

    return profiles


def load_events(filepath: Union[str, Path]) -> List[FinancialEvent]:
    """Load financial events preserving exact row provenance and missing amounts."""
    path = _resolve_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Financial events file not found: {path}")

    events: List[FinancialEvent] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        validate_columns(reader.fieldnames, EVENT_COLUMNS, path.name)

        for row_idx, row in enumerate(reader, start=2):
            event_id = row["event_id"].strip()
            user_id = row["user_id"].strip()
            if not event_id:
                raise ValidationError(f"Missing event_id at row {row_idx} in {path.name}")
            if not user_id:
                raise ValidationError(f"Missing user_id at row {row_idx} in {path.name}")

            linked_event = row["linked_event_id"].strip() if row["linked_event_id"] else None
            event = FinancialEvent(
                event_id=event_id,
                source_row=row_idx,
                user_id=user_id,
                event_type=row["event_type"].strip(),
                description=row["description"].strip(),
                category=row["category"].strip(),
                direction=row["direction"].strip(),
                amount=parse_decimal(row["amount"], "amount", allow_none=True, row_num=row_idx),
                currency=row["currency"].strip(),
                event_date=parse_date(row["event_date"], "event_date", allow_none=False, row_num=row_idx),
                settlement_date=parse_date(
                    row["settlement_date"], "settlement_date", allow_none=True, row_num=row_idx
                ),
                status=row["status"].strip(),
                linked_event_id=linked_event if linked_event else None,
                flexibility=row["flexibility"].strip(),
                minimum_allowed_amount=parse_decimal(
                    row["minimum_allowed_amount"], "minimum_allowed_amount", allow_none=True, row_num=row_idx
                ),
            )
            events.append(event)

    return events


def load_payment_options(filepath: Union[str, Path]) -> List[PaymentOption]:
    """Load seller/provider payment options for requests."""
    path = _resolve_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Payment options file not found: {path}")

    options: List[PaymentOption] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        validate_columns(reader.fieldnames, PAYMENT_OPTION_COLUMNS, path.name)

        for row_idx, row in enumerate(reader, start=2):
            option = PaymentOption(
                payment_option_id=row["payment_option_id"].strip(),
                request_id=row["request_id"].strip(),
                payment_method=row["payment_method"].strip(),
                payment_amount=parse_decimal(
                    row["payment_amount"], "payment_amount", allow_none=False, row_num=row_idx
                ),
                number_of_payments=parse_int(
                    row["number_of_payments"], "number_of_payments", allow_none=False, row_num=row_idx
                ),
                first_payment_date=parse_date(
                    row["first_payment_date"], "first_payment_date", allow_none=False, row_num=row_idx
                ),
                payment_frequency_days=parse_int(
                    row["payment_frequency_days"], "payment_frequency_days", allow_none=True, row_num=row_idx
                ),
                financing_fee=parse_decimal(
                    row["financing_fee"], "financing_fee", allow_none=False, row_num=row_idx
                ),
                total_payable_amount=parse_decimal(
                    row["total_payable_amount"], "total_payable_amount", allow_none=False, row_num=row_idx
                ),
            )
            options.append(option)

    return options


def load_exchange_rates(filepath: Union[str, Path]) -> List[ExchangeRate]:
    """Load fixed dated exchange rates with exact Decimal rates."""
    path = _resolve_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Exchange rates file not found: {path}")

    rates: List[ExchangeRate] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        validate_columns(reader.fieldnames, EXCHANGE_RATE_COLUMNS, path.name)

        for row_idx, row in enumerate(reader, start=2):
            rate = ExchangeRate(
                rate_date=parse_date(row["rate_date"], "rate_date", allow_none=False, row_num=row_idx),
                from_currency=row["from_currency"].strip(),
                to_currency=row["to_currency"].strip(),
                rate=parse_decimal(row["rate"], "rate", allow_none=False, row_num=row_idx),
            )
            rates.append(rate)

    return rates


def load_messages(filepath: Union[str, Path]) -> List[Message]:
    """Load supporting messages from employers, banks, merchants, and providers."""
    path = _resolve_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Messages file not found: {path}")

    messages: List[Message] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        validate_columns(reader.fieldnames, MESSAGE_COLUMNS, path.name)

        for row_idx, row in enumerate(reader, start=2):
            req_id = row["request_id"].strip() if row["request_id"] else None
            ev_id = row["related_event_id"].strip() if row["related_event_id"] else None

            msg = Message(
                message_id=row["message_id"].strip(),
                user_id=row["user_id"].strip(),
                request_id=req_id if req_id else None,
                related_event_id=ev_id if ev_id else None,
                sent_at=parse_datetime(row["sent_at"], "sent_at", allow_none=False, row_num=row_idx),
                source_type=row["source_type"].strip(),
                message_text=row["message_text"].strip(),
            )
            messages.append(msg)

    return messages


def load_images(filepath: Union[str, Path]) -> List[ImageMetadata]:
    """Load image metadata links."""
    path = _resolve_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Images metadata file not found: {path}")

    images: List[ImageMetadata] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        validate_columns(reader.fieldnames, IMAGE_COLUMNS, path.name)

        for row in reader:
            img = ImageMetadata(
                image_id=row["image_id"].strip(),
                user_id=row["user_id"].strip(),
                request_id=row["request_id"].strip(),
                related_event_id=row["related_event_id"].strip(),
            )
            images.append(img)

    return images


def load_requests(filepath: Union[str, Path]) -> List[FinancialRequest]:
    """Load evaluation requests."""
    path = _resolve_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Requests file not found: {path}")

    requests: List[FinancialRequest] = []
    with open(path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        validate_columns(reader.fieldnames, REQUEST_COLUMNS, path.name)

        for row_idx, row in enumerate(reader, start=2):
            req = FinancialRequest(
                request_id=row["request_id"].strip(),
                user_id=row["user_id"].strip(),
                request_date=parse_date(row["request_date"], "request_date", allow_none=False, row_num=row_idx),
                request_type=row["request_type"].strip(),
                requested_amount=parse_decimal(
                    row["requested_amount"], "requested_amount", allow_none=False, row_num=row_idx
                ),
                desired_completion_date=parse_date(
                    row["desired_completion_date"], "desired_completion_date", allow_none=False, row_num=row_idx
                ),
                allows_partial_payment=parse_bool(
                    row["allows_partial_payment"], "allows_partial_payment", row_num=row_idx
                ),
                request_text=row["request_text"].strip(),
            )
            requests.append(req)

    return requests


@dataclass
class Dataset:
    """Unified container for all ingested dataset tables with built-in indexing."""
    profiles: Dict[str, FinancialProfile]
    events: List[FinancialEvent]
    payment_options: List[PaymentOption]
    exchange_rates: List[ExchangeRate]
    messages: List[Message]
    images: List[ImageMetadata]
    requests: List[FinancialRequest]

    # Precomputed indices
    events_by_id: Dict[str, FinancialEvent] = field(init=False)
    events_by_user: Dict[str, List[FinancialEvent]] = field(init=False)
    payment_options_by_request: Dict[str, List[PaymentOption]] = field(init=False)
    messages_by_user: Dict[str, List[Message]] = field(init=False)
    messages_by_request: Dict[str, List[Message]] = field(init=False)
    messages_by_event: Dict[str, List[Message]] = field(init=False)
    images_by_event: Dict[str, ImageMetadata] = field(init=False)
    images_by_id: Dict[str, ImageMetadata] = field(init=False)
    requests_by_id: Dict[str, FinancialRequest] = field(init=False)

    def __post_init__(self) -> None:
        """Build indexes for efficient lookup."""
        self.events_by_id = {ev.event_id: ev for ev in self.events}

        ev_by_user: Dict[str, List[FinancialEvent]] = {}
        for ev in self.events:
            ev_by_user.setdefault(ev.user_id, []).append(ev)
        self.events_by_user = ev_by_user

        opts_by_req: Dict[str, List[PaymentOption]] = {}
        for opt in self.payment_options:
            opts_by_req.setdefault(opt.request_id, []).append(opt)
        self.payment_options_by_request = opts_by_req

        msg_by_user: Dict[str, List[Message]] = {}
        msg_by_req: Dict[str, List[Message]] = {}
        msg_by_ev: Dict[str, List[Message]] = {}
        for msg in self.messages:
            msg_by_user.setdefault(msg.user_id, []).append(msg)
            if msg.request_id:
                msg_by_req.setdefault(msg.request_id, []).append(msg)
            if msg.related_event_id:
                msg_by_ev.setdefault(msg.related_event_id, []).append(msg)
        self.messages_by_user = msg_by_user
        self.messages_by_request = msg_by_req
        self.messages_by_event = msg_by_ev

        self.images_by_event = {img.related_event_id: img for img in self.images}
        self.images_by_id = {img.image_id: img for img in self.images}
        self.requests_by_id = {req.request_id: req for req in self.requests}


def get_default_dataset_dir() -> Path:
    """Resolve the default dataset directory relative to this package location."""
    # code/loaders.py -> code/ -> hackerrank-orchestrate-september26/ -> dataset/
    code_dir = Path(__file__).resolve().parent
    return code_dir.parent / "dataset"


def load_dataset(dataset_dir: Optional[Union[str, Path]] = None) -> Dataset:
    """Load all dataset files from directory and return indexed Dataset container."""
    base_dir = Path(dataset_dir).resolve() if dataset_dir is not None else get_default_dataset_dir()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    profiles = load_profiles(base_dir / "financial_profiles.csv")
    events = load_events(base_dir / "financial_events.csv")
    options = load_payment_options(base_dir / "request_payment_options.csv")
    exchange_rates = load_exchange_rates(base_dir / "exchange_rates.csv")
    messages = load_messages(base_dir / "messages.csv")
    images = load_images(base_dir / "images.csv")
    requests = load_requests(base_dir / "requests.csv")

    return Dataset(
        profiles=profiles,
        events=events,
        payment_options=options,
        exchange_rates=exchange_rates,
        messages=messages,
        images=images,
        requests=requests,
    )
