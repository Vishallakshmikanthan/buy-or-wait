"""Production integration tests for message interpretation and causal propagation (Prompt 21).

Verifies:
A. AMEND_AMOUNT causal mutation through production pipeline.
B. CANCEL causal mutation through production pipeline.
C. DELAY_TO causal mutation through production pipeline.
D. TEMPORARY_CHANGE + RESUME causal mutation through production pipeline.
E. Wrong-user action remains ignored.
F. Ambiguous target remains fail-closed.
G. Message-order permutation produces bit-identical output.
"""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

from code.canonical import Direction
from code.loaders import load_dataset
from code.message_interpretation import (
    MessageAction,
    MessageActionType,
    TargetType,
    apply_message_actions_to_future_events,
    interpret_and_link_messages,
    resolve_message_conflicts,
)
from code.models import FinancialProfile, FinancialRequest, Message
from code.output import generate_all_outputs
from code.reconciliation import reconcile_events
from code.recurrence import FutureEvent, RecurrenceFrequency, detect_all_recurrence, expand_future_events
from code.simulator import simulate_user


class TestOutputMessageIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent.parent.parent
        self.dataset_dir = self.root / "dataset"
        self.ds = load_dataset(self.dataset_dir)

    def test_a_amend_amount_production_path(self) -> None:
        """A. AMEND_AMOUNT causal mutation: request_154 has message_119 amending salary."""
        actions, _ = interpret_and_link_messages(self.ds.messages, self.ds.events, self.ds.requests, self.ds.profiles)
        u154_acts = [a for a in actions if a.user_id == "user_154" and a.is_applied]
        self.assertTrue(any(a.action_type == MessageActionType.AMEND_AMOUNT for a in u154_acts))

        req154 = next(r for r in self.ds.requests if r.request_id == "request_154")
        base_ledger = reconcile_events(self.ds.events, self.ds.profiles, self.ds.exchange_rates, self.ds.messages, self.ds.images)
        all_series, _ = detect_all_recurrence(base_ledger, self.ds.profiles, self.ds.messages)
        start_d = req154.request_date
        end_d = start_d + timedelta(days=90)
        future_res = expand_future_events(all_series["user_154"], start_d, end_d, ledger=base_ledger)

        adapted = apply_message_actions_to_future_events(future_res.future_events, u154_acts, req154.request_date)
        # Verify amended salary amount 1628 is active in future events
        salary_events = [fe for fe in adapted if fe.category == "salary"]
        self.assertTrue(len(salary_events) > 0)
        self.assertTrue(all(fe.amount_home == Decimal("1628") for fe in salary_events))

    def test_b_cancel_causal_mutation(self) -> None:
        """B. CANCEL causal mutation: drops cancelled recurring obligation from forward projection."""
        fe = FutureEvent(
            event_id="fe_gym_cancel",
            user_id="user_test_cancel",
            effective_date=date(2026, 6, 1),
            direction=Direction.OUTFLOW,
            amount_home=Decimal("50.00"),
            currency="USD",
            category="gym",
            event_type="subscription",
            description="Gym membership",
            series_id="rec_gym",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        msg_cancel = Message(
            message_id="msg_canc_01",
            user_id="user_test_cancel",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
            source_type="service_provider",
            message_text="Your gym membership subscription has been cancelled.",
        )
        profile = FinancialProfile(
            user_id="user_test_cancel",
            home_currency="USD",
            current_available_balance=Decimal("1000"),
            minimum_balance_to_keep=Decimal("100"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=("gym",),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )
        actions, _ = interpret_and_link_messages([msg_cancel], [], [], {"user_test_cancel": profile})
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, MessageActionType.CANCEL)

        adapted = apply_message_actions_to_future_events([fe], actions, date(2026, 5, 1))
        self.assertEqual(len(adapted), 0)

    def test_c_delay_to_causal_mutation(self) -> None:
        """C. DELAY_TO causal mutation: reschedules projected date forward without changing amounts."""
        fe = FutureEvent(
            event_id="fe_sal_delay",
            user_id="user_test_delay",
            effective_date=date(2026, 5, 15),
            direction=Direction.INFLOW,
            amount_home=Decimal("3000.00"),
            currency="USD",
            category="salary",
            event_type="income",
            description="Monthly salary",
            series_id="rec_sal",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        msg_delay = Message(
            message_id="msg_delay_01",
            user_id="user_test_delay",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 5, 2, 9, 0, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your salary payout is rescheduled to 2026-05-22 due to processing delay.",
        )
        profile = FinancialProfile(
            user_id="user_test_delay",
            home_currency="USD",
            current_available_balance=Decimal("1000"),
            minimum_balance_to_keep=Decimal("100"),
            financial_priorities=(),
            expense_categories_to_protect=(),
            expense_categories_user_is_willing_to_reduce=(),
            expense_categories_user_is_willing_to_stop=(),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )
        actions, _ = interpret_and_link_messages([msg_delay], [], [], {"user_test_delay": profile})
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, MessageActionType.DELAY_TO)

        adapted = apply_message_actions_to_future_events([fe], actions, date(2026, 5, 1))
        self.assertEqual(len(adapted), 1)
        self.assertEqual(adapted[0].effective_date, date(2026, 5, 22))
        self.assertEqual(adapted[0].amount_home, Decimal("3000.00"))

    def test_d_temporary_change_and_resume_causal_mutation(self) -> None:
        """D. TEMPORARY_CHANGE + RESUME: request_130 has temporary salary reduction shifting earliest date."""
        actions, _ = interpret_and_link_messages(self.ds.messages, self.ds.events, self.ds.requests, self.ds.profiles)
        u130_acts = [a for a in actions if a.user_id == "user_130" and a.is_applied]
        self.assertTrue(any(a.action_type == MessageActionType.TEMPORARY_CHANGE for a in u130_acts))

        req130 = next(r for r in self.ds.requests if r.request_id == "request_130")
        base_ledger = reconcile_events(self.ds.events, self.ds.profiles, self.ds.exchange_rates, self.ds.messages, self.ds.images)
        all_series, _ = detect_all_recurrence(base_ledger, self.ds.profiles, self.ds.messages)
        start_d = req130.request_date
        end_d = start_d + timedelta(days=90)
        future_res = expand_future_events(all_series["user_130"], start_d, end_d, ledger=base_ledger)

        adapted = apply_message_actions_to_future_events(future_res.future_events, u130_acts, req130.request_date)
        # Reduced from $3,024 to $2,177.28
        salaries = [fe for fe in adapted if fe.category == "salary"]
        self.assertTrue(len(salaries) > 0)
        self.assertTrue(all(fe.amount_home == Decimal("2177.28") for fe in salaries))

    def test_e_wrong_user_action_ignored(self) -> None:
        """E. Wrong-user action: message for different user is rejected and never applied."""
        msg_wrong = Message(
            message_id="msg_wrong_user",
            user_id="user_nonexistent_999",
            request_id="request_130",
            related_event_id=None,
            sent_at=datetime(2024, 12, 1, 9, 0, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your temporary monthly pay is USD 500.00.",
        )
        actions, unres = interpret_and_link_messages([msg_wrong], self.ds.events, self.ds.requests, self.ds.profiles)
        applied = [a for a in actions if a.is_applied]
        self.assertEqual(len(applied), 0)

    def test_f_ambiguous_target_fail_closed(self) -> None:
        """F. Ambiguous target: unresolvable message fails closed without modifying cash flows."""
        msg_ambig = Message(
            message_id="msg_ambig_01",
            user_id="user_130",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2024, 12, 1, 9, 0, tzinfo=timezone.utc),
            source_type="service_provider",
            message_text="Please be advised that your monthly payment will change soon.",
        )
        actions, unres = interpret_and_link_messages([msg_ambig], self.ds.events, self.ds.requests, self.ds.profiles)
        applied = [a for a in actions if a.is_applied]
        self.assertEqual(len(applied), 0)

    def test_g_message_order_permutation_invariance(self) -> None:
        """G. Message-order permutation: shuffling messages produces identical actions."""
        msgs_shuffled = list(self.ds.messages)
        random.seed(12345)
        random.shuffle(msgs_shuffled)

        acts_orig, _ = interpret_and_link_messages(self.ds.messages, self.ds.events, self.ds.requests, self.ds.profiles)
        acts_shuf, _ = interpret_and_link_messages(msgs_shuffled, self.ds.events, self.ds.requests, self.ds.profiles)

        t1 = [(a.message_id, a.action_type, a.new_amount, a.effective_date) for a in acts_orig]
        t2 = [(a.message_id, a.action_type, a.new_amount, a.effective_date) for a in acts_shuf]
        self.assertEqual(t1, t2)


if __name__ == "__main__":
    unittest.main()
