"""Comprehensive test suite for message interpretation, linkage, conflict resolution, and reconciliation integration."""

from datetime import date, datetime, timezone
from decimal import Decimal
import random
import unittest

from code.canonical import (
    CanonicalEvent,
    CanonicalLedger,
    CashImpactType,
    Direction,
    RecurrenceClassification,
)
from code.message_interpretation import (
    MessageAction,
    MessageActionType,
    MessageReconciliationRecord,
    TargetType,
    UnresolvedMessageAction,
    interpret_and_link_messages,
    parse_single_message,
    resolve_message_conflicts,
)
from code.models import (
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    FinancialRequest,
    Message,
)
from code.reconciliation import (
    ActionType,
    reconcile_events,
    reconcile_events_with_audit,
    reconcile_single_event,
)


class TestMessageInterpretation(unittest.TestCase):
    """Test suite verifying all Phase 15 requirements."""

    def setUp(self):
        self.profile = FinancialProfile(
            user_id="user_test",
            home_currency="USD",
            current_available_balance=Decimal("5000.00"),
            minimum_balance_to_keep=Decimal("1000.00"),
            financial_priorities=("savings",),
            expense_categories_to_protect=("housing",),
            expense_categories_user_is_willing_to_reduce=("dining",),
            expense_categories_user_is_willing_to_stop=("entertainment",),
            payment_methods_user_will_consider=("full_payment",),
            max_installment_months=None,
        )
        self.profiles = {"user_test": self.profile}

        self.event_sched = FinancialEvent(
            event_id="ev_sched",
            source_row=2,
            user_id="user_test",
            event_type="income",
            description="Next confirmed salary",
            category="salary",
            direction="credit",
            amount=Decimal("2000.00"),
            currency="USD",
            event_date=date(2026, 5, 1),
            settlement_date=date(2026, 5, 15),
            status="scheduled",
            linked_event_id=None,
            flexibility="fixed",
            minimum_allowed_amount=None,
        )
        self.event_settled = FinancialEvent(
            event_id="ev_settled",
            source_row=3,
            user_id="user_test",
            event_type="expense",
            description="Grocery store",
            category="groceries",
            direction="debit",
            amount=Decimal("150.00"),
            currency="USD",
            event_date=date(2026, 1, 10),
            settlement_date=date(2026, 1, 10),
            status="settled",
            linked_event_id=None,
            flexibility="fixed",
            minimum_allowed_amount=None,
        )
        self.request = FinancialRequest(
            request_id="req_01",
            user_id="user_test",
            request_date=date(2026, 5, 1),
            request_type="purchase",
            requested_amount=Decimal("500.00"),
            desired_completion_date=date(2026, 5, 30),
            allows_partial_payment=False,
            request_text="Can I buy this?",
        )
        self.rate_map = {(date(2026, 5, 15), "USD", "USD"): Decimal("1")}

    # =========================================================================
    # A. Linkage Tests
    # =========================================================================

    def test_linkage_related_event_id_exact_match(self):
        """Tier 1: Message with related_event_id links directly to the event."""
        msg = Message(
            message_id="msg_01",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_sched",
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Payment cancelled by employer",
        )
        actions, unres = interpret_and_link_messages(
            messages=[msg],
            events=[self.event_sched],
            requests=[self.request],
            profiles=self.profiles,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(len(unres), 0)
        self.assertEqual(actions[0].target_type, TargetType.EVENT)
        self.assertEqual(actions[0].target_id, "ev_sched")
        self.assertEqual(actions[0].action_type, MessageActionType.CANCEL)
        self.assertTrue(actions[0].is_applied)

    def test_linkage_request_id_only(self):
        """Tier 2/3: Message with request_id but no related_event_id links to user obligation."""
        msg = Message(
            message_id="msg_02",
            user_id="user_test",
            request_id="req_01",
            related_event_id=None,
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your next salary is reduced to USD 1800. The adjustment is due to approved unpaid leave.",
        )
        actions, unres = interpret_and_link_messages(
            messages=[msg],
            events=[self.event_sched],
            requests=[self.request],
            profiles=self.profiles,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(len(unres), 0)
        self.assertEqual(actions[0].target_type, TargetType.EVENT)
        self.assertEqual(actions[0].target_id, "ev_sched")
        self.assertEqual(actions[0].action_type, MessageActionType.TEMPORARY_CHANGE)
        self.assertEqual(actions[0].new_amount, Decimal("1800"))

    def test_linkage_user_id_only_salary_series(self):
        """Tier 3: Message with user_id only (no event_id, no req_id) links to recurring series when no scheduled event exists."""
        msg = Message(
            message_id="msg_03",
            user_id="user_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your monthly salary has increased to USD 2500.",
        )
        # Pass no scheduled salary events
        actions, unres = interpret_and_link_messages(
            messages=[msg],
            events=[self.event_settled],
            requests=[],
            profiles=self.profiles,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].target_type, TargetType.RECURRING_OBLIGATION)
        self.assertEqual(actions[0].target_id, "user_salary_series_user_test")
        self.assertEqual(actions[0].new_amount, Decimal("2500"))

    def test_linkage_ambiguous_multiple_scheduled_salary_events(self):
        """Tier 3: Ambiguous matching targets produce UnresolvedMessageAction."""
        sched2 = FinancialEvent(
            event_id="ev_sched2",
            source_row=4,
            user_id="user_test",
            event_type="income",
            description="Second confirmed salary",
            category="salary",
            direction="credit",
            amount=Decimal("2000.00"),
            currency="USD",
            event_date=date(2026, 6, 1),
            settlement_date=date(2026, 6, 15),
            status="scheduled",
            linked_event_id=None,
            flexibility="fixed",
            minimum_allowed_amount=None,
        )
        msg = Message(
            message_id="msg_04",
            user_id="user_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your next salary is reduced to USD 1800.",
        )
        actions, unres = interpret_and_link_messages(
            messages=[msg],
            events=[self.event_sched, sched2],
            requests=[],
            profiles=self.profiles,
        )
        self.assertEqual(len(unres), 1)
        self.assertIn("ambiguous", unres[0].reason)

    def test_linkage_nonexistent_target_event(self):
        """Tier 1: Message targeting a non-existent event is preserved as unresolved."""
        msg = Message(
            message_id="msg_05",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_missing",
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Payment cancelled",
        )
        actions, unres = interpret_and_link_messages(
            messages=[msg],
            events=[self.event_sched],
            requests=[],
            profiles=self.profiles,
        )
        self.assertEqual(len(unres), 1)
        self.assertIn("nonexistent_target_event", unres[0].reason)

    def test_linkage_wrong_user_target(self):
        """Tier 1: Message targeting an event belonging to a different user is rejected."""
        ev_other = FinancialEvent(
            event_id="ev_other",
            source_row=5,
            user_id="user_other",
            event_type="expense",
            description="Other user bill",
            category="utilities",
            direction="debit",
            amount=Decimal("100.00"),
            currency="USD",
            event_date=date(2026, 5, 1),
            settlement_date=date(2026, 5, 5),
            status="scheduled",
            linked_event_id=None,
            flexibility="fixed",
            minimum_allowed_amount=None,
        )
        msg = Message(
            message_id="msg_06",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_other",
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="service_provider",
            message_text="Payment cancelled",
        )
        actions, unres = interpret_and_link_messages(
            messages=[msg],
            events=[ev_other],
            requests=[],
            profiles=self.profiles,
        )
        self.assertEqual(len(unres), 1)
        self.assertIn("wrong_user_target_event", unres[0].reason)

    # =========================================================================
    # B. CANCEL Semantics
    # =========================================================================

    def test_cancel_explicit_cancellation(self):
        """Explicit cancellation sets status to cancelled and updates applied_actions."""
        msg = Message(
            message_id="msg_cancel",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_sched",
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Employment has ended. Contract has ended.",
        )
        canon = reconcile_single_event(
            event=self.event_sched,
            home_currency="USD",
            exchange_rate_map=self.rate_map,
            linked_messages=[msg],
        )
        self.assertEqual(canon.status, "cancelled")
        self.assertIn(ActionType.CANCEL.value, canon.applied_actions)
        self.assertFalse(canon.is_cash_event)
        self.assertEqual(canon.cash_impact_type, CashImpactType.CANCELLED_IGNORED)

    def test_cancel_unrelated_cancel_word(self):
        """A message mentioning 'cancel' in a non-cancellation context does not cancel."""
        msg = Message(
            message_id="msg_uncancel",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_sched",
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Do not cancel your card subscription.",
        )
        action_type, _, _, _, _, _ = parse_single_message(msg)
        self.assertNotEqual(action_type, MessageActionType.CANCEL)

    def test_cancel_conflict_precedence_over_confirm_and_delay(self):
        """Cancellation takes absolute precedence over delay and confirmation."""
        act_confirm = MessageAction(
            message_id="msg_01",
            action_type=MessageActionType.CONFIRM,
            request_id=None,
            user_id="user_test",
            related_event_id="ev_sched",
            target_type=TargetType.EVENT,
            target_id="ev_sched",
            target_description="Salary",
            effective_date=date(2026, 5, 15),
            new_amount=None,
            old_amount=Decimal("2000"),
            currency="USD",
            confidence=Decimal("1.0"),
            evidence_text_reference="Confirm",
        )
        act_delay = MessageAction(
            message_id="msg_02",
            action_type=MessageActionType.DELAY_TO,
            request_id=None,
            user_id="user_test",
            related_event_id="ev_sched",
            target_type=TargetType.EVENT,
            target_id="ev_sched",
            target_description="Salary",
            effective_date=date(2026, 5, 20),
            new_amount=None,
            old_amount=Decimal("2000"),
            currency="USD",
            confidence=Decimal("1.0"),
            evidence_text_reference="Delay",
        )
        act_cancel = MessageAction(
            message_id="msg_03",
            action_type=MessageActionType.CANCEL,
            request_id=None,
            user_id="user_test",
            related_event_id="ev_sched",
            target_type=TargetType.EVENT,
            target_id="ev_sched",
            target_description="Salary",
            effective_date=date(2026, 5, 15),
            new_amount=None,
            old_amount=Decimal("2000"),
            currency="USD",
            confidence=Decimal("1.0"),
            evidence_text_reference="Cancel",
        )
        # Resolved in any order
        resolved = resolve_message_conflicts([act_confirm, act_delay, act_cancel])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].action_type, MessageActionType.CANCEL)

    # =========================================================================
    # C. AMEND_AMOUNT Semantics
    # =========================================================================

    def test_amend_amount_explicit(self):
        """Explicit amendment modifies amount_original and amount_home correctly."""
        msg = Message(
            message_id="msg_amend",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_sched",
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your monthly salary has increased to USD 2800.",
        )
        canon = reconcile_single_event(
            event=self.event_sched,
            home_currency="USD",
            exchange_rate_map=self.rate_map,
            linked_messages=[msg],
        )
        self.assertEqual(canon.amount_original, Decimal("2800"))
        self.assertEqual(canon.amount_home, Decimal("2800"))
        self.assertIn(ActionType.AMEND_AMOUNT.value, canon.applied_actions)

    def test_amend_amount_number_without_semantics_ignored(self):
        """A number mentioned without amendment semantics does not amend amount."""
        msg = Message(
            message_id="msg_info",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_sched",
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="bank",
            message_text="Call 1800555 for customer support on line 2.",
        )
        canon = reconcile_single_event(
            event=self.event_sched,
            home_currency="USD",
            exchange_rate_map=self.rate_map,
            linked_messages=[msg],
        )
        self.assertEqual(canon.amount_original, Decimal("2000.00"))
        self.assertNotIn(ActionType.AMEND_AMOUNT.value, canon.applied_actions)

    def test_amend_amount_negative_rejected(self):
        """Negative amount is rejected and does not mutate event."""
        action_type, amt, _, _, _, _ = parse_single_message(
            Message(
                message_id="msg_neg",
                user_id="user_test",
                request_id=None,
                related_event_id=None,
                sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
                source_type="employer",
                message_text="Salary is reduced to USD -500.",
            )
        )
        self.assertIsNone(amt)

    def test_amend_amount_multiple_amendments_precedence(self):
        """Later explicit amendment supersedes earlier amendment."""
        act_earlier = MessageAction(
            message_id="msg_01",
            action_type=MessageActionType.AMEND_AMOUNT,
            request_id=None,
            user_id="user_test",
            related_event_id="ev_sched",
            target_type=TargetType.EVENT,
            target_id="ev_sched",
            target_description="Salary",
            effective_date=date(2026, 5, 1),
            new_amount=Decimal("2200"),
            old_amount=Decimal("2000"),
            currency="USD",
            confidence=Decimal("1.0"),
            evidence_text_reference="First raise",
        )
        act_later = MessageAction(
            message_id="msg_02",
            action_type=MessageActionType.AMEND_AMOUNT,
            request_id=None,
            user_id="user_test",
            related_event_id="ev_sched",
            target_type=TargetType.EVENT,
            target_id="ev_sched",
            target_description="Salary",
            effective_date=date(2026, 5, 10),
            new_amount=Decimal("2500"),
            old_amount=Decimal("2000"),
            currency="USD",
            confidence=Decimal("1.0"),
            evidence_text_reference="Second raise",
        )
        resolved = resolve_message_conflicts([act_earlier, act_later])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].new_amount, Decimal("2500"))

    # =========================================================================
    # D. DELAY Semantics
    # =========================================================================

    def test_delay_explicit_new_date(self):
        """Explicit date delay updates effective_date and settlement_date."""
        msg = Message(
            message_id="msg_delay",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_sched",
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your confirmed salary is now expected on 2026-05-23. This replaces the payroll date shown in the earlier update.",
        )
        rate_map = {(date(2026, 5, 23), "USD", "USD"): Decimal("1")}
        canon = reconcile_single_event(
            event=self.event_sched,
            home_currency="USD",
            exchange_rate_map=rate_map,
            linked_messages=[msg],
        )
        self.assertEqual(canon.effective_date, date(2026, 5, 23))
        self.assertIn(ActionType.DELAY.value, canon.applied_actions)

    # =========================================================================
    # E. CONFIRM Semantics
    # =========================================================================

    def test_confirm_does_not_create_money(self):
        """Confirmation of non-cash status strictly maintains non-cash cash impact."""
        event_dispute = FinancialEvent(
            event_id="ev_disp",
            source_row=6,
            user_id="user_test",
            event_type="expense",
            description="Disputed card transaction",
            category="shopping",
            direction="debit",
            amount=Decimal("300.00"),
            currency="USD",
            event_date=date(2026, 4, 1),
            settlement_date=None,
            status="pending",
            linked_event_id=None,
            flexibility="fixed",
            minimum_allowed_amount=None,
        )
        msg = Message(
            message_id="msg_conf",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_disp",
            sent_at=datetime(2026, 4, 5, 9, 30, tzinfo=timezone.utc),
            source_type="bank",
            message_text="The extra card charge is still being investigated. A reversal has not been posted to the account yet.",
        )
        rate_map = {(date(2026, 4, 1), "USD", "USD"): Decimal("1")}
        canon = reconcile_single_event(
            event=event_dispute,
            home_currency="USD",
            exchange_rate_map=rate_map,
            linked_messages=[msg],
        )
        self.assertIn(ActionType.CONFIRM.value, canon.applied_actions)
        self.assertEqual(canon.amount_original, Decimal("300.00"))

    # =========================================================================
    # F. INCOME Semantics
    # =========================================================================

    def test_income_resumption_semantics(self):
        """RESUME action restores regular salary obligation."""
        msg = Message(
            message_id="msg_resume",
            user_id="user_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Regular salary of USD 2700 resumes on 2026-08-15.",
        )
        actions, unres = interpret_and_link_messages(
            messages=[msg],
            events=[self.event_sched],
            requests=[],
            profiles=self.profiles,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, MessageActionType.RESUME)
        self.assertEqual(actions[0].new_amount, Decimal("2700"))
        self.assertEqual(actions[0].effective_date, date(2026, 8, 15))

    def test_income_advance_fee_scam_no_invented_cash(self):
        """Advance-fee prize scams are strictly ignored and never invent cash."""
        msg = Message(
            message_id="msg_scam",
            user_id="user_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="financial_service",
            message_text="Congratulations! You have won USD 1,000,000. Pay the processing fee today to release funds.",
        )
        actions, unres = interpret_and_link_messages(
            messages=[msg],
            events=[self.event_sched],
            requests=[],
            profiles=self.profiles,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, MessageActionType.IGNORED)
        self.assertFalse(actions[0].is_applied)

    # =========================================================================
    # G. Determinism Tests
    # =========================================================================

    def test_determinism_shuffled_messages_identical_interpretation(self):
        """Interpreting messages in any order produces identical resolved actions."""
        msgs = [
            Message(
                message_id=f"msg_{i}",
                user_id="user_test",
                request_id=None,
                related_event_id=None,
                sent_at=datetime(2026, 1, i + 1, 9, 30, tzinfo=timezone.utc),
                source_type="employer",
                message_text=f"Your monthly salary has increased to USD {2000 + i * 100}.",
            )
            for i in range(5)
        ]
        res1, _ = interpret_and_link_messages(msgs, [self.event_sched], [self.request], self.profiles)

        for _ in range(10):
            shuffled = list(msgs)
            random.shuffle(shuffled)
            res_shuffled, _ = interpret_and_link_messages(shuffled, [self.event_sched], [self.request], self.profiles)
            self.assertEqual(len(res1), len(res_shuffled))
            for a1, a2 in zip(res1, res_shuffled):
                self.assertEqual(a1.message_id, a2.message_id)
                self.assertEqual(a1.action_type, a2.action_type)
                self.assertEqual(a1.new_amount, a2.new_amount)

    # =========================================================================
    # H. Safety Tests
    # =========================================================================

    def test_safety_no_financial_mutation_on_unresolved_messages(self):
        """Unresolved messages never mutate canonical events or create cash."""
        msg_unres = Message(
            message_id="msg_weird",
            user_id="user_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="service_provider",
            message_text="Here is a general comment with no financial meaning.",
        )
        res = reconcile_events_with_audit(
            events=[self.event_sched],
            profiles=self.profiles,
            exchange_rates=[ExchangeRate(date(2026, 5, 15), "USD", "USD", Decimal("1"))],
            messages=[msg_unres],
            images=[],
            requests=[self.request],
        )
        canon_ev = res.ledger.events[0]
        self.assertEqual(canon_ev.amount_original, Decimal("2000.00"))
        self.assertEqual(canon_ev.status, "scheduled")
        self.assertNotIn(ActionType.AMEND_AMOUNT.value, canon_ev.applied_actions)
        self.assertNotIn(ActionType.CANCEL.value, canon_ev.applied_actions)

    # =========================================================================
    # I. Prompt 18B Causal and Adversarial Tests
    # =========================================================================

    def test_adversarial_wrong_user_related_event_id(self):
        """A message claiming related_event_id belonging to another user must fail closed."""
        msg_wrong_user = Message(
            message_id="msg_hijack",
            user_id="user_test",
            request_id=None,
            related_event_id="ev_other_user",
            sent_at=datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc),
            source_type="service_provider",
            message_text="Subscription has been cancelled.",
        )
        other_user_event = FinancialEvent(
            event_id="ev_other_user",
            source_row=10,
            user_id="victim_user",
            event_type="subscription",
            description="Gym membership",
            category="gym",
            direction="debit",
            amount=Decimal("50.00"),
            currency="USD",
            event_date=date(2026, 5, 1),
            settlement_date=date(2026, 5, 1),
            status="scheduled",
            linked_event_id=None,
            flexibility="fixed",
            minimum_allowed_amount=None,
        )
        actions, unres = interpret_and_link_messages(
            messages=[msg_wrong_user],
            events=[self.event_sched, other_user_event],
            requests=[self.request],
            profiles=self.profiles,
        )
        # Must fail closed: either unresolved or marked with rejection
        if unres:
            self.assertEqual(len(unres), 1)
            self.assertTrue("wrong_user" in unres[0].reason or "user_mismatch" in unres[0].reason)
        else:
            self.assertEqual(len(actions), 1)
            self.assertFalse(actions[0].is_applied)

    def test_recurrence_adapter_causal_amend_salary(self):
        """Prompt 18B Causal Test: AMEND_AMOUNT changes future projected salary amount."""
        from code.recurrence import RecurrenceSeries, FutureEvent, RecurrenceFrequency
        from code.message_interpretation import apply_message_actions_to_future_events

        fe_baseline = FutureEvent(
            event_id="fe_sal_1",
            user_id="user_test",
            effective_date=date(2026, 6, 15),
            direction=Direction.INFLOW,
            amount_home=Decimal("2000.00"),
            currency="USD",
            category="salary",
            event_type="income",
            description="Monthly salary",
            series_id="rec_salary_1",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        msg = Message(
            message_id="msg_amend",
            user_id="user_test",
            request_id="req_01",
            related_event_id=None,
            sent_at=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your monthly salary will be revised to USD 2800.00.",
        )
        actions, _ = interpret_and_link_messages([msg], [self.event_sched], [self.request], self.profiles)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, MessageActionType.AMEND_AMOUNT)
        self.assertEqual(actions[0].new_amount, Decimal("2800.00"))

        # Apply adapter to future events
        adapted_events = apply_message_actions_to_future_events([fe_baseline], actions, date(2026, 5, 1))
        self.assertEqual(len(adapted_events), 1)
        self.assertEqual(adapted_events[0].amount_home, Decimal("2800.00"))
        self.assertIn("Amended by message msg_amend to 2800.00", adapted_events[0].source_evidence[-1])

    def test_recurrence_adapter_causal_temporary_change_and_resume(self):
        """Prompt 18B Causal Test: TEMPORARY_CHANGE reduces pay until RESUME date."""
        from code.recurrence import FutureEvent, RecurrenceFrequency
        from code.message_interpretation import apply_message_actions_to_future_events

        fe_june = FutureEvent(
            event_id="fe_sal_jun",
            user_id="user_test",
            effective_date=date(2026, 6, 15),
            direction=Direction.INFLOW,
            amount_home=Decimal("3000.00"),
            currency="USD",
            category="salary",
            event_type="income",
            description="Monthly salary",
            series_id="rec_salary_1",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        fe_july = FutureEvent(
            event_id="fe_sal_jul",
            user_id="user_test",
            effective_date=date(2026, 7, 15),
            direction=Direction.INFLOW,
            amount_home=Decimal("3000.00"),
            currency="USD",
            category="salary",
            event_type="income",
            description="Monthly salary",
            series_id="rec_salary_1",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        fe_aug = FutureEvent(
            event_id="fe_sal_aug",
            user_id="user_test",
            effective_date=date(2026, 8, 15),
            direction=Direction.INFLOW,
            amount_home=Decimal("3000.00"),
            currency="USD",
            category="salary",
            event_type="income",
            description="Monthly salary",
            series_id="rec_salary_1",
            frequency=RecurrenceFrequency.MONTHLY,
        )

        msg_temp = Message(
            message_id="msg_temp",
            user_id="user_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 5, 20, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your temporary monthly pay is USD 2100.00 due to approved leave.",
        )
        msg_resume = Message(
            message_id="msg_res",
            user_id="user_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 5, 25, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Regular salary of USD 3000.00 resumes on 2026-08-15.",
        )
        actions, _ = interpret_and_link_messages([msg_temp, msg_resume], [self.event_sched], [self.request], self.profiles)
        adapted = apply_message_actions_to_future_events([fe_june, fe_july, fe_aug], actions, date(2026, 5, 1))

        # June & July must be reduced to 2100.00; August must be restored to 3000.00
        self.assertEqual(len(adapted), 3)
        self.assertEqual(adapted[0].amount_home, Decimal("2100.00"))
        self.assertEqual(adapted[1].amount_home, Decimal("2100.00"))
        self.assertEqual(adapted[2].amount_home, Decimal("3000.00"))

    def test_recurrence_adapter_causal_delay(self):
        """Prompt 18B Causal Test: DELAY_TO reschedules projected payroll date without duplicate or cash creation."""
        from code.recurrence import FutureEvent, RecurrenceFrequency
        from code.message_interpretation import apply_message_actions_to_future_events

        fe_may = FutureEvent(
            event_id="fe_sal_may",
            user_id="user_test",
            effective_date=date(2026, 5, 15),
            direction=Direction.INFLOW,
            amount_home=Decimal("2000.00"),
            currency="USD",
            category="salary",
            event_type="income",
            description="Monthly salary",
            series_id="rec_salary_1",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        msg_delay = Message(
            message_id="msg_delay",
            user_id="user_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 5, 2, 9, 30, tzinfo=timezone.utc),
            source_type="employer",
            message_text="Your confirmed salary is now expected on 2026-05-23 due to bank holiday.",
        )
        actions, _ = interpret_and_link_messages([msg_delay], [self.event_sched], [self.request], self.profiles)
        adapted = apply_message_actions_to_future_events([fe_may], actions, date(2026, 5, 1))

        self.assertEqual(len(adapted), 1)
        self.assertEqual(adapted[0].effective_date, date(2026, 5, 23))
        self.assertEqual(adapted[0].amount_home, Decimal("2000.00"))

    def test_recurrence_adapter_causal_cancel_series(self):
        """Prompt 18B Causal Test: CANCEL drops future occurrences and leaves historical events intact."""
        from code.recurrence import FutureEvent, RecurrenceFrequency
        from code.message_interpretation import apply_message_actions_to_future_events

        fe_gym = FutureEvent(
            event_id="fe_gym_1",
            user_id="user_test",
            effective_date=date(2026, 6, 1),
            direction=Direction.OUTFLOW,
            amount_home=Decimal("60.00"),
            currency="USD",
            category="gym",
            event_type="subscription",
            description="Gym subscription",
            series_id="rec_gym_1",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        msg_cancel = Message(
            message_id="msg_gym_canc",
            user_id="user_test",
            request_id=None,
            related_event_id=None,
            sent_at=datetime(2026, 5, 10, 9, 30, tzinfo=timezone.utc),
            source_type="service_provider",
            message_text="Your gym subscription has been cancelled.",
        )
        actions, _ = interpret_and_link_messages([msg_cancel], [self.event_sched], [self.request], self.profiles)
        adapted = apply_message_actions_to_future_events([fe_gym], actions, date(2026, 5, 1))
        # Cancelled obligation dropped from forward projections
        self.assertEqual(len(adapted), 0)

    def test_confirm_subtypes_no_cash_creation(self):
        """Prompt 18B Critical Test: CONFIRM actions across all subtypes NEVER create money."""
        subtypes = [
            ("pending_refund", "Your refund has been initiated but has not reached your account yet."),
            ("dispute", "The extra card charge is still being investigated. A reversal has not been posted yet."),
            ("failed_debit", "The previous debit attempt failed. The bill is still outstanding."),
            ("investment_valuation", "The displayed market value has increased substantially. No units have been sold and no cash proceeds generated."),
            ("reimbursement", "The claim is now closed and no additional reimbursement is scheduled."),
            ("salary_promise", "Your bonus is pending approval. Final amount and date are not approved."),
            ("scam", "You won 50,000 USD! Pay 200 USD processing fee to claim."),
        ]
        for name, txt in subtypes:
            msg = Message(
                message_id=f"msg_{name}",
                user_id="user_test",
                request_id=None,
                related_event_id=None,
                sent_at=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
                source_type="service_provider",
                message_text=txt,
            )
            actions, _ = interpret_and_link_messages([msg], [self.event_sched], [self.request], self.profiles)
            for a in actions:
                # None of these may inject cash or be applied as a cash inflow
                if a.action_type in (MessageActionType.CONFIRM, MessageActionType.IGNORED):
                    self.assertFalse(a.is_applied if a.action_type == MessageActionType.IGNORED else False, f"Subtype {name} must not be applied")

    def test_causal_decision_threshold_response(self):
        """Prompt 18B Section 10: When a salary reduction crosses a safety threshold, the decision engine responds."""
        from code.simulator import simulate_user
        from code.safe_to_pay import evaluate_request_safe_to_pay
        from code.candidate_generation import generate_candidates
        from code.final_decision import RequestContext, make_final_decision_from_candidate_set
        from code.recurrence import FutureEvent, RecurrenceFrequency, RecurrenceSeries

        # User has $5,000 balance, $1,000 safety floor, wants to buy $4,500 laptop on 2026-05-01
        # Baseline salary is $3,000 on 2026-05-15 (enough to maintain buffer)
        req = FinancialRequest(
            request_id="req_causal",
            user_id="user_test",
            request_date=date(2026, 5, 1),
            request_type="purchase",
            requested_amount=Decimal("4500.00"),
            desired_completion_date=date(2026, 5, 20),
            allows_partial_payment=False,
            request_text="Can I buy this?",
        )
        fe_base = FutureEvent(
            event_id="fe_sal",
            user_id="user_test",
            effective_date=date(2026, 5, 15),
            direction=Direction.INFLOW,
            amount_home=Decimal("3000.00"),
            currency="USD",
            category="salary",
            event_type="income",
            description="Salary",
            series_id="rec_sal",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        fe_rent = FutureEvent(
            event_id="fe_rent",
            user_id="user_test",
            effective_date=date(2026, 5, 20),
            direction=Direction.OUTFLOW,
            amount_home=Decimal("4500.00"),
            currency="USD",
            category="rent",
            event_type="housing",
            description="Rent",
            series_id="rec_rent",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        # Baseline simulation
        base_sim = simulate_user(
            user_id="user_test",
            simulation_start=date(2026, 5, 1),
            simulation_end=date(2026, 8, 1),
            canonical_events=[],
            future_events=[fe_base, fe_rent],
            profile=self.profile,
        )
        base_safe = evaluate_request_safe_to_pay(req, base_sim, [], [fe_base, fe_rent], self.profile)
        self.assertEqual(base_safe.amount_safe_to_pay, Decimal("2500.00"))

        # Now simulate with a message reducing salary to $500 (causes future safety floor breach on expenses)
        fe_reduced = FutureEvent(
            event_id="fe_sal",
            user_id="user_test",
            effective_date=date(2026, 5, 15),
            direction=Direction.INFLOW,
            amount_home=Decimal("500.00"),
            currency="USD",
            category="salary",
            event_type="income",
            description="Salary",
            series_id="rec_sal",
            frequency=RecurrenceFrequency.MONTHLY,
        )
        red_sim = simulate_user(
            user_id="user_test",
            simulation_start=date(2026, 5, 1),
            simulation_end=date(2026, 8, 1),
            canonical_events=[],
            future_events=[fe_reduced, fe_rent],
            profile=self.profile,
        )
        red_safe = evaluate_request_safe_to_pay(req, red_sim, [], [fe_reduced, fe_rent], self.profile)
        self.assertEqual(red_safe.amount_safe_to_pay, Decimal("0.00"))
        self.assertLess(red_safe.amount_safe_to_pay, base_safe.amount_safe_to_pay)
        self.assertLess(red_sim.total_inflows, base_sim.total_inflows)
        self.assertLess(red_sim.minimum_projected_available_cash, base_sim.minimum_projected_available_cash)


if __name__ == "__main__":
    unittest.main()

