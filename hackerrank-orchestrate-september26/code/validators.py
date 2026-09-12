"""Validation helpers and custom exceptions for data ingestion."""

from typing import Sequence, Optional


class ValidationError(Exception):
    """Raised when data fails schema, type, or constraint validation."""
    pass


def validate_columns(
    actual_columns: Optional[Sequence[str]],
    required_columns: Sequence[str],
    source_name: str,
) -> None:
    """Validate that all required columns are present in the CSV header.

    Args:
        actual_columns: Header column names from the CSV file.
        required_columns: Sequence of required column names.
        source_name: Name or path of the file being validated.

    Raises:
        ValidationError: If actual_columns is None or any required column is missing.
    """
    if actual_columns is None:
        raise ValidationError(f"CSV file '{source_name}' has no header or is empty.")

    actual_set = set(actual_columns)
    missing = [col for col in required_columns if col not in actual_set]
    if missing:
        raise ValidationError(
            f"Missing required column(s) in '{source_name}': {', '.join(missing)}. "
            f"Found columns: {', '.join(actual_columns)}"
        )
