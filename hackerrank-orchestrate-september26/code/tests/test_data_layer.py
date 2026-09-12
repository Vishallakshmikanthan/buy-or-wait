"""Unit tests for the foundational data ingestion and normalization layer."""

import tempfile
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from code.loaders import (
    load_events,
    load_payment_options,
    load_profiles,
    load_requests,
)
from code.models import FinancialEvent, FinancialProfile, PaymentOption
from code.parsers import (
    parse_bool,
    parse_date,
    parse_datetime,
    parse_decimal,
    parse_int,
    parse_pipe_tuple,
)
from code.validators import ValidationError, validate_columns


class TestParsers(unittest.TestCase):
    """Test parser functions for strict typing and exact Decimal representation."""

    def test_decimal_monetary_parsing(self):
        """Test exact Decimal conversion preserving precision without binary floating point errors."""
        val = parse_decimal("15656000.75", "amount")
        self.assertEqual(val, Decimal("15656000.75"))
        self.assertIsInstance(val, Decimal)

        # Test zero
        zero_val = parse_decimal("0", "amount")
        self.assertEqual(zero_val, Decimal("0"))

        # Test small fractional amounts
        small = parse_decimal("0.0001", "fee")
        self.assertEqual(small, Decimal("0.0001"))

        # Test with whitespace
        with_spaces = parse_decimal("  4546.08  ", "payment_amount")
        self.assertEqual(with_spaces, Decimal("4546.08"))

    def test_blank_monetary_fields(self):
        """Test handling of blank monetary fields with and without allow_none."""
        # When allow_none is True, blank string or None returns None
        self.assertIsNone(parse_decimal("", "amount", allow_none=True))
        self.assertIsNone(parse_decimal("   ", "amount", allow_none=True))
        self.assertIsNone(parse_decimal(None, "amount", allow_none=True))

        # When allow_none is False, blank string or None raises ValidationError
        with self.assertRaises(ValidationError) as ctx:
            parse_decimal("", "amount", allow_none=False)
        self.assertIn("Missing required monetary/decimal value", str(ctx.exception))

        with self.assertRaises(ValidationError):
            parse_decimal("not-a-number", "amount", allow_none=False)

    def test_date_parsing(self):
        """Test date parsing for ISO YYYY-MM-DD and error handling."""
        d = parse_date("2026-09-13", "request_date")
        self.assertEqual(d, date(2026, 9, 13))

        # Blank date with allow_none
        self.assertIsNone(parse_date("", "settlement_date", allow_none=True))

        # Blank date when not allowed
        with self.assertRaises(ValidationError) as ctx:
            parse_date("", "event_date", allow_none=False)
        self.assertIn("Missing required date value", str(ctx.exception))

        # Malformed date formats
        for bad_date in ["13-09-2026", "2026/09/13", "invalid", "2026-02-31"]:
            with self.assertRaises(ValidationError):
                parse_date(bad_date, "event_date")

    def test_datetime_parsing(self):
        """Test parsing ISO datetime strings."""
        dt = parse_datetime("2025-07-29T09:30:00Z", "sent_at")
        self.assertEqual(dt.year, 2025)
        self.assertEqual(dt.month, 7)
        self.assertEqual(dt.day, 29)
        self.assertEqual(dt.hour, 9)
        self.assertEqual(dt.minute, 30)

    def test_bool_parsing(self):
        """Test parsing boolean flags."""
        self.assertTrue(parse_bool("true", "allows_partial_payment"))
        self.assertTrue(parse_bool("True", "allows_partial_payment"))
        self.assertTrue(parse_bool("1", "allows_partial_payment"))
        self.assertFalse(parse_bool("false", "allows_partial_payment"))
        self.assertFalse(parse_bool("False", "allows_partial_payment"))
        self.assertFalse(parse_bool("0", "allows_partial_payment"))

        with self.assertRaises(ValidationError):
            parse_bool("maybe", "allows_partial_payment")

    def test_pipe_tuple_parsing(self):
        """Test parsing pipe-delimited strings."""
        result = parse_pipe_tuple("education|debt_repayment|groceries")
        self.assertEqual(result, ("education", "debt_repayment", "groceries"))

        # Empty string should yield empty tuple
        self.assertEqual(parse_pipe_tuple(""), ())
        self.assertEqual(parse_pipe_tuple(None), ())
        self.assertEqual(parse_pipe_tuple("  single_item  "), ("single_item",))


class TestValidators(unittest.TestCase):
    """Test schema validation helpers."""

    def test_required_column_validation_success(self):
        """Test that validation passes when all required columns are present."""
        actual = ["user_id", "home_currency", "current_available_balance"]
        required = ["user_id", "home_currency"]
        # Should not raise
        validate_columns(actual, required, "test.csv")

    def test_required_column_validation_failure(self):
        """Test that validation raises ValidationError when columns are missing."""
        actual = ["user_id", "home_currency"]
        required = ["user_id", "home_currency", "minimum_balance_to_keep"]
        with self.assertRaises(ValidationError) as ctx:
            validate_columns(actual, required, "test.csv")
        self.assertIn("Missing required column(s)", str(ctx.exception))
        self.assertIn("minimum_balance_to_keep", str(ctx.exception))

    def test_required_column_validation_empty_header(self):
        """Test empty/None header."""
        with self.assertRaises(ValidationError) as ctx:
            validate_columns(None, ["user_id"], "test.csv")
        self.assertIn("has no header or is empty", str(ctx.exception))


class TestLoaders(unittest.TestCase):
    """Test CSV file loading and provenance preservation."""

    def test_load_request_and_profile(self):
        """Test loading a mock request and profile file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Write mock profiles CSV
            profiles_csv = tmp / "financial_profiles.csv"
            profiles_csv.write_text(
                "user_id,home_currency,current_available_balance,minimum_balance_to_keep,"
                "financial_priorities,expense_categories_to_protect,expense_categories_user_is_willing_to_reduce,"
                "expense_categories_user_is_willing_to_stop,payment_methods_user_will_consider,max_installment_months\n"
                "user_01,ZAR,58481.10,18000,education|debt,rent|groceries,dining,delivery,full_payment,6\n",
                encoding="utf-8",
            )

            profiles = load_profiles(profiles_csv)
            self.assertIn("user_01", profiles)
            prof = profiles["user_01"]
            self.assertEqual(prof.user_id, "user_01")
            self.assertEqual(prof.home_currency, "ZAR")
            self.assertEqual(prof.current_available_balance, Decimal("58481.10"))
            self.assertEqual(prof.minimum_balance_to_keep, Decimal("18000"))
            self.assertEqual(prof.financial_priorities, ("education", "debt"))
            self.assertEqual(prof.max_installment_months, 6)

            # Write mock requests CSV
            requests_csv = tmp / "requests.csv"
            requests_csv.write_text(
                "request_id,user_id,request_date,request_type,requested_amount,"
                "desired_completion_date,allows_partial_payment,request_text\n"
                "request_01,user_01,2024-03-03,purchase,25256.00,2024-03-20,true,Can I buy laptop?\n",
                encoding="utf-8",
            )

            requests = load_requests(requests_csv)
            self.assertEqual(len(requests), 1)
            req = requests[0]
            self.assertEqual(req.request_id, "request_01")
            self.assertEqual(req.user_id, "user_01")
            self.assertEqual(req.requested_amount, Decimal("25256.00"))
            self.assertTrue(req.allows_partial_payment)
            self.assertEqual(req.request_date, date(2024, 3, 3))

    def test_load_payment_options_for_request(self):
        """Test loading payment options with frequency and fee tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            options_csv = tmp / "request_payment_options.csv"
            options_csv.write_text(
                "payment_option_id,request_id,payment_method,payment_amount,number_of_payments,"
                "first_payment_date,payment_frequency_days,financing_fee,total_payable_amount\n"
                "opt_1,req_1,full_payment,1000.00,1,2025-01-01,,0.00,1000.00\n"
                "opt_2,req_1,installments,260.00,4,2025-01-05,30,40.00,1040.00\n",
                encoding="utf-8",
            )

            options = load_payment_options(options_csv)
            self.assertEqual(len(options), 2)

            opt_full = options[0]
            self.assertEqual(opt_full.payment_method, "full_payment")
            self.assertIsNone(opt_full.payment_frequency_days)
            self.assertEqual(opt_full.financing_fee, Decimal("0.00"))
            self.assertEqual(opt_full.total_payable_amount, Decimal("1000.00"))

            opt_inst = options[1]
            self.assertEqual(opt_inst.payment_method, "installments")
            self.assertEqual(opt_inst.payment_frequency_days, 30)
            self.assertEqual(opt_inst.financing_fee, Decimal("40.00"))
            self.assertEqual(opt_inst.total_payable_amount, Decimal("1040.00"))

    def test_preservation_of_event_provenance_and_missing_amounts(self):
        """Test that event loading preserves source row index and handles missing amounts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            events_csv = tmp / "financial_events.csv"
            events_csv.write_text(
                "event_id,user_id,event_type,description,category,direction,amount,currency,"
                "event_date,settlement_date,status,linked_event_id,flexibility,minimum_allowed_amount\n"
                "ev_1,user_1,expense,Rent,rent,debit,5000.00,ZAR,2025-01-01,2025-01-01,settled,,fixed,\n"
                "ev_2,user_1,income,Salary,salary,credit,,ZAR,2025-01-05,2025-01-05,scheduled,,fixed,\n",
                encoding="utf-8",
            )

            events = load_events(events_csv)
            self.assertEqual(len(events), 2)

            # Event 1: Normal with amount
            ev1 = events[0]
            self.assertEqual(ev1.event_id, "ev_1")
            self.assertEqual(ev1.source_row, 2)  # Row 2 in CSV
            self.assertEqual(ev1.amount, Decimal("5000.00"))
            self.assertFalse(ev1.has_missing_amount)

            # Event 2: Image-backed event with missing amount
            ev2 = events[1]
            self.assertEqual(ev2.event_id, "ev_2")
            self.assertEqual(ev2.source_row, 3)  # Row 3 in CSV
            self.assertIsNone(ev2.amount)
            self.assertTrue(ev2.has_missing_amount)
            # Ensure it is NOT converted to 0
            self.assertNotEqual(ev2.amount, Decimal("0"))


if __name__ == "__main__":
    unittest.main()
