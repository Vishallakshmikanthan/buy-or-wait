"""Robust parsing functions with strict validation and exact Decimal handling."""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Optional, Tuple

from .validators import ValidationError


def parse_decimal(
    value: Optional[str],
    field_name: str,
    allow_none: bool = False,
    row_num: Optional[int] = None,
) -> Optional[Decimal]:
    """Parse a string to an exact Decimal without using binary floating point.

    Args:
        value: Raw string from CSV.
        field_name: Name of the field for error reporting.
        allow_none: If True, empty strings return None instead of raising ValidationError.
        row_num: Optional source row number for diagnostics.

    Returns:
        Exact Decimal value or None if allowed.

    Raises:
        ValidationError: If value is blank and allow_none is False, or if formatting is invalid.
    """
    loc = f" (row {row_num})" if row_num is not None else ""
    if value is None or not value.strip():
        if allow_none:
            return None
        raise ValidationError(f"Missing required monetary/decimal value for '{field_name}'{loc}.")

    clean_str = value.strip()
    try:
        return Decimal(clean_str)
    except InvalidOperation as exc:
        raise ValidationError(
            f"Invalid decimal format '{clean_str}' for field '{field_name}'{loc}."
        ) from exc


def parse_date(
    value: Optional[str],
    field_name: str,
    allow_none: bool = False,
    row_num: Optional[int] = None,
) -> Optional[date]:
    """Parse an ISO format date string (YYYY-MM-DD).

    Args:
        value: Raw string from CSV.
        field_name: Name of the field for error reporting.
        allow_none: If True, empty strings return None instead of raising ValidationError.
        row_num: Optional source row number for diagnostics.

    Returns:
        date object or None if allowed.

    Raises:
        ValidationError: If value is missing/blank when allow_none is False, or if format is invalid.
    """
    loc = f" (row {row_num})" if row_num is not None else ""
    if value is None or not value.strip():
        if allow_none:
            return None
        raise ValidationError(f"Missing required date value for '{field_name}'{loc}.")

    clean_str = value.strip()
    try:
        return date.fromisoformat(clean_str)
    except (ValueError, TypeError) as exc:
        raise ValidationError(
            f"Invalid date format '{clean_str}' for field '{field_name}'{loc}. Expected YYYY-MM-DD."
        ) from exc


def parse_datetime(
    value: Optional[str],
    field_name: str,
    allow_none: bool = False,
    row_num: Optional[int] = None,
) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp string (e.g., 2025-07-29T09:30:00Z).

    Args:
        value: Raw string from CSV.
        field_name: Name of the field for error reporting.
        allow_none: If True, empty strings return None instead of raising ValidationError.
        row_num: Optional source row number for diagnostics.

    Returns:
        datetime object or None if allowed.

    Raises:
        ValidationError: If value is missing/invalid.
    """
    loc = f" (row {row_num})" if row_num is not None else ""
    if value is None or not value.strip():
        if allow_none:
            return None
        raise ValidationError(f"Missing required datetime value for '{field_name}'{loc}.")

    clean_str = value.strip()
    # Normalize ISO 8601 'Z' suffix to '+00:00' for standard library fromisoformat
    normalized = clean_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except (ValueError, TypeError) as exc:
        raise ValidationError(
            f"Invalid datetime format '{clean_str}' for field '{field_name}'{loc}."
        ) from exc


def parse_int(
    value: Optional[str],
    field_name: str,
    allow_none: bool = False,
    row_num: Optional[int] = None,
) -> Optional[int]:
    """Parse an integer value.

    Args:
        value: Raw string from CSV.
        field_name: Name of the field for error reporting.
        allow_none: If True, empty strings return None instead of raising ValidationError.
        row_num: Optional source row number for diagnostics.

    Returns:
        int value or None if allowed.

    Raises:
        ValidationError: If value is missing/invalid.
    """
    loc = f" (row {row_num})" if row_num is not None else ""
    if value is None or not value.strip():
        if allow_none:
            return None
        raise ValidationError(f"Missing required integer value for '{field_name}'{loc}.")

    clean_str = value.strip()
    try:
        return int(clean_str)
    except ValueError as exc:
        raise ValidationError(
            f"Invalid integer format '{clean_str}' for field '{field_name}'{loc}."
        ) from exc


def parse_bool(
    value: Optional[str],
    field_name: str,
    row_num: Optional[int] = None,
) -> bool:
    """Parse a boolean value strictly ('true'/'false', case-insensitive).

    Args:
        value: Raw string from CSV.
        field_name: Name of the field for error reporting.
        row_num: Optional source row number for diagnostics.

    Returns:
        Boolean value.

    Raises:
        ValidationError: If value is missing or unrecognized.
    """
    loc = f" (row {row_num})" if row_num is not None else ""
    if value is None or not value.strip():
        raise ValidationError(f"Missing required boolean value for '{field_name}'{loc}.")

    clean_str = value.strip().lower()
    if clean_str in ("true", "1", "yes"):
        return True
    if clean_str in ("false", "0", "no"):
        return False

    raise ValidationError(
        f"Invalid boolean value '{value}' for field '{field_name}'{loc}. Expected 'true' or 'false'."
    )


def parse_pipe_tuple(value: Optional[str]) -> Tuple[str, ...]:
    """Parse pipe-delimited values into a tuple of clean strings.

    Args:
        value: Raw string e.g. 'education|debt_repayment'.

    Returns:
        Tuple of stripped, non-empty tokens.
    """
    if value is None or not value.strip():
        return ()
    return tuple(item.strip() for item in value.split("|") if item.strip())
