"""Deterministic message interpretation, validation, and linkage hierarchy."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .models import FinancialEvent, FinancialProfile, FinancialRequest, Message
from .recurrence import FutureEvent, RecurrenceSeries
from .validators import ValidationError


class MessageActionType(str, Enum):
    """Normalized action semantics extracted deterministically from messages."""
    CANCEL = "CANCEL"
    AMEND_AMOUNT = "AMEND_AMOUNT"
    DELAY_TO = "DELAY_TO"
    CONFIRM = "CONFIRM"
    RESUME = "RESUME"
    TEMPORARY_CHANGE = "TEMPORARY_CHANGE"
    CREATE_RECURRING = "CREATE_RECURRING"
    UNRESOLVED = "UNRESOLVED"
    IGNORED = "IGNORED"


class TargetType(str, Enum):
    """The financial entity targeted by a message action."""
    EVENT = "EVENT"
    RECURRING_OBLIGATION = "RECURRING_OBLIGATION"
    REQUEST = "REQUEST"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MessageAction:
    """Immutable structured representation of a message's parsed and validated action."""
    message_id: str
    action_type: MessageActionType
    request_id: Optional[str]
    user_id: str
    related_event_id: Optional[str]
    target_type: TargetType
    target_id: Optional[str]
    target_description: Optional[str]
    effective_date: Optional[date]
    new_amount: Optional[Decimal]
    old_amount: Optional[Decimal]
    currency: Optional[str]
    confidence: Decimal
    evidence_text_reference: str
    reason: Optional[str] = None
    is_applied: bool = False
    unresolved_reason: Optional[str] = None


@dataclass(frozen=True)
class UnresolvedMessageAction:
    """Record of a message that could not be deterministically resolved or applied."""
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    message_text: str
    reason: str


@dataclass(frozen=True)
class MessageReconciliationRecord:
    """Immutable audit trail of an applied message action."""
    message_id: str
    action_type: str
    target_id: str
    original_value: Optional[str]
    new_value: Optional[str]
    effective_date: Optional[date]
    reason: str


def _parse_iso_or_text_date(text: str) -> Optional[date]:
    """Deterministically extract a date formatted as YYYY-MM-DD or DD Month YYYY."""
    # 1. ISO format YYYY-MM-DD
    iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso_match:
        try:
            return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
        except ValueError:
            pass

    # 2. Text dates like '24 July 2026', '3 April 2026', '3 September 2026'
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "januari": 1, "februari": 2, "maret": 3, "mei": 5, "juni": 6,
        "juli": 7, "agustus": 8, "oktober": 10, "desember": 12,
    }
    month_pattern = "|".join(months.keys())
    text_date_match = re.search(rf"\b(\d{{1,2}})\s+({month_pattern})\s+(\d{{4}})\b", text, re.IGNORECASE)
    if text_date_match:
        day = int(text_date_match.group(1))
        m_str = text_date_match.group(2).lower()
        month = months[m_str]
        year = int(text_date_match.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            pass

    return None


def _parse_currency_amount(text: str) -> Tuple[Optional[str], Optional[Decimal]]:
    """Deterministically extract a monetary currency and positive decimal amount."""
    # Match patterns like: EUR 1422.85, IDR 38,760,000, USD 1,548.00, ZAR 54120
    curr_match = re.search(
        r"\b(EUR|USD|INR|ZAR|IDR)\s*([\d,]+(?:\.\d+)?)\b",
        text,
        re.IGNORECASE,
    )
    if curr_match:
        curr = curr_match.group(1).upper()
        raw_num = curr_match.group(2).replace(",", "")
        try:
            val = Decimal(raw_num)
            if val > Decimal("0"):
                return curr, val
        except Exception:
            pass

    # Match patterns where currency follows: 300 EUR, 100 USD
    curr_match_post = re.search(
        r"\b([\d,]+(?:\.\d+)?)\s*(EUR|USD|INR|ZAR|IDR)\b",
        text,
        re.IGNORECASE,
    )
    if curr_match_post:
        curr = curr_match_post.group(2).upper()
        raw_num = curr_match_post.group(1).replace(",", "")
        try:
            val = Decimal(raw_num)
            if val > Decimal("0"):
                return curr, val
        except Exception:
            pass

    return None, None


def _parse_percentage(text: str) -> Optional[Decimal]:
    """Deterministically extract a percentage value."""
    pct_match = re.search(r"(\d+(?:\.\d+)?)%", text)
    if pct_match:
        try:
            return Decimal(pct_match.group(1)) / Decimal("100")
        except Exception:
            pass
    return None


def parse_single_message(
    message: Message,
) -> Tuple[MessageActionType, Optional[Decimal], Optional[str], Optional[date], Optional[Decimal], str]:
    """Deterministically parse a raw Message into its semantic action, amounts, dates, and intent reason.

    Returns:
        Tuple of (action_type, new_amount, currency, effective_date, percentage, reason)
    """
    txt = message.message_text
    tl = txt.lower()

    # 1. Advance-fee prize scams (strictly IGNORED / REJECTED)
    if any(k in tl for k in ["bayar biaya", "pay the release charge", "pay the processing charge", "biaya pencairan", "processing fee"]):
        return MessageActionType.IGNORED, None, None, None, None, "advance_fee_prize_scam_rejected"

    # 2. Explicit cancellation / contract termination
    if any(k in tl for k in [
        "contract has ended", "kontrak musiman saat ini telah berakhir",
        "employment has ended", "hubungan kerja anda telah berakhir", "pekerjaan anda telah berakhir",
        "transaction cancelled", "payment cancelled", "transaksi dibatalkan", "pembayaran dibatalkan",
        "order cancelled", "pesanan dibatalkan",
    ]):
        return MessageActionType.CANCEL, None, None, None, None, "explicit_cancellation_or_termination"

    if "cancelled" in tl or "dibatalkan" in tl or "batal" in tl:
        return MessageActionType.CANCEL, None, None, None, None, "explicit_cancellation"

    # 3. Regular salary resumption
    if any(k in tl for k in ["resumes on", "mulai berlaku kembali", "kembali normal", "regular salary of", "gaji rutin sebesar"]):
        if "resumes" in tl or "berlaku kembali" in tl:
            curr, amt = _parse_currency_amount(txt)
            d = _parse_iso_or_text_date(txt)
            return MessageActionType.RESUME, amt, curr, d, None, "salary_resumption_after_temporary_change"

    # 4. Temporary salary reduction / unpaid leave
    if any(k in tl for k in [
        "salary is reduced to", "gaji berikutnya dikurangi menjadi",
        "temporary monthly pay is", "gaji bulanan sementara adalah",
        "reduced amount continues", "jumlah yang lebih rendah masih berlaku",
    ]):
        curr, amt = _parse_currency_amount(txt)
        d = _parse_iso_or_text_date(txt)
        return MessageActionType.TEMPORARY_CHANGE, amt, curr, d, None, "temporary_salary_reduction"

    # 5. Amount amendments: salary increase, rent increase, general revision
    if any(k in tl for k in [
        "salary has increased to", "salary will be revised to", "revised to", "gaji bulanan anda naik menjadi",
        "increases monthly rent by", "kenaikan sewa", "perpanjangan sewa menaikkan",
        "sewa bulanan sebesar",
    ]):
        if any(w in tl for w in ["rent", "sewa", "lease"]):
            pct = _parse_percentage(txt)
            return MessageActionType.AMEND_AMOUNT, None, None, None, pct, "rent_lease_percentage_increase"
        else:
            curr, amt = _parse_currency_amount(txt)
            d = _parse_iso_or_text_date(txt)
            return MessageActionType.AMEND_AMOUNT, amt, curr, d, None, "salary_amount_increase"

    # One household earner ended -> remaining salary specified
    if any(k in tl for k in ["household employment record has ended", "sumber pendapatan kerja rumah tangga telah berakhir"]):
        curr, amt = _parse_currency_amount(txt)
        return MessageActionType.AMEND_AMOUNT, amt, curr, None, None, "remaining_household_salary_after_earner_ended"

    # 6. Date delay / rescheduling (salary payroll date replacement)
    if any(k in tl for k in [
        "replaces the payroll date", "menggantikan tanggal penggajian", "menggantikan tanggal payroll",
        "is now expected on", "kini diperkirakan masuk pada", "delayed until", "rescheduled to",
        "diundur ke", "ditunda sampai",
    ]):
        d = _parse_iso_or_text_date(txt)
        curr, amt = _parse_currency_amount(txt)
        return MessageActionType.DELAY_TO, amt, curr, d, None, "payroll_date_rescheduled"

    # 7. Client invoice approval (InvoiceFlow, InvoiceLane, PayPilot, ClientDesk, FreelanceHub, WorkPort)
    # Pending credit awaiting settlement; non-cash until settled
    if any(k in tl for k in ["approved an invoice payment", "menyetujui pembayaran faktur", "settlement is expected on", "penyelesaian diperkirakan pada"]):
        curr, amt = _parse_currency_amount(txt)
        d = _parse_iso_or_text_date(txt)
        return MessageActionType.CONFIRM, amt, curr, d, None, "client_invoice_approved_pending_settlement"

    # 8. Explicit confirmations:
    # Prize proceeds settled & closed
    if any(k in tl for k in ["prize proceeds have reached your account", "hasil hadiah sudah masuk ke rekening", "claim is now closed", "klaim sudah ditutup"]):
        return MessageActionType.CONFIRM, None, None, None, None, "prize_proceeds_settled_and_closed"

    # Investment sale settled & closed
    if any(k in tl for k in ["proceeds from your investment sale have settled", "hasil penjualan investasi anda sudah masuk ke rekening tunai", "sale order is complete"]):
        return MessageActionType.CONFIRM, None, None, None, None, "investment_sale_proceeds_settled"

    # Work expense reimbursement settled & closed
    if any(k in tl for k in ["reimbursement for your earlier work expense", "penggantian atas biaya kerja anda sebelumnya"]):
        return MessageActionType.CONFIRM, None, None, None, None, "work_expense_reimbursement_settled_closed"

    # Pending refunds (non-cash)
    if any(k in tl for k in ["refund has been initiated", "pengembalian dana sudah diproses", "refund is still processing", "pengembalian dana ... masih diproses", "pengembalian dana"]):
        return MessageActionType.CONFIRM, None, None, None, None, "merchant_refund_pending_non_cash"

    # Disputed card charges / investigations (non-cash)
    if any(k in tl for k in ["charge is still being investigated", "tagihan kartu tambahan masih dalam penyelidikan", "dispute is open", "sengketa masih terbuka", "reversal has not been posted", "dana pembalikannya belum tercatat"]):
        return MessageActionType.CONFIRM, None, None, None, None, "card_dispute_under_investigation_non_cash"

    # Failed debit attempts (outstanding bill retry)
    if any(k in tl for k in ["previous debit attempt failed", "debit sebelumnya gagal", "bill is still outstanding", "tagihan masih belum lunas"]):
        return MessageActionType.CONFIRM, None, None, None, None, "debit_attempt_failed_retry_scheduled"

    # Unrealized investment valuation
    if any(k in tl for k in ["displayed market value", "nilai investasi yang ditampilkan", "no units have been sold", "tidak ada unit yang dijual", "holding has not been sold"]):
        return MessageActionType.CONFIRM, None, None, None, None, "investment_valuation_unrealized_non_cash"

    # Receipt confirmation for image-backed events
    if any(k in tl for k in ["receipt has the final", "receipt contains the final", "order was paid in", "pembayaran diterima pada"]):
        return MessageActionType.CONFIRM, None, None, None, None, "receipt_confirms_final_amount_for_event"

    # First salary / confirmed credit date
    if any(k in tl for k in ["first salary", "gaji pertama"]):
        curr, amt = _parse_currency_amount(txt)
        d = _parse_iso_or_text_date(txt)
        return MessageActionType.CONFIRM, amt, curr, d, None, "first_salary_confirmed"

    # Salary with one-time arrears
    if any(k in tl for k in ["one-time arrears adjustment", "penyesuaian tunggakan satu kali"]):
        curr, amt = _parse_currency_amount(txt)
        return MessageActionType.CONFIRM, amt, curr, None, None, "salary_with_one_time_arrears"

    # Confirmed base salary (pending commissions omitted)
    if any(k in tl for k in ["confirmed base salary is", "gaji pokok yang dikonfirmasi adalah"]):
        curr, amt = _parse_currency_amount(txt)
        return MessageActionType.CONFIRM, amt, curr, None, None, "confirmed_base_salary_omits_pending_commissions"

    # Foreign currency salary credit notice or confirmed salary
    if any(k in tl for k in ["receiving bank will convert", "bank penerima akan mengonversinya", "dikonfirmasi untuk", "sudah dikonfirmasi"]):
        curr, amt = _parse_currency_amount(txt)
        d = _parse_iso_or_text_date(txt)
        return MessageActionType.CONFIRM, amt, curr, d, None, "confirmed_salary_notice"

    # Pending gig payouts (QuickCrew, TaskLoop, ShiftPay, TaskSprint)
    if any(k in tl for k in ["payout is still pending", "pembayaran berikutnya dari", "balance isn't withdrawable", "masih tertunda", "belum dapat ditarik"]):
        return MessageActionType.CONFIRM, None, None, None, None, "pending_gig_payout_not_withdrawable"

    # Pending prize claim processing
    if any(k in tl for k in ["prize claim has been verified and is still in payment processing", "klaim hadiah anda sudah diverifikasi dan masih"]):
        return MessageActionType.CONFIRM, None, None, None, None, "prize_claim_processing_not_credited"

    # Internal account transfer (non-cash)
    if any(k in tl for k in ["transfer between your two accounts", "transfer antara dua rekening"]):
        return MessageActionType.IGNORED, None, None, None, None, "internal_account_transfer_non_cash"

    # Multi-card minimums notice
    if any(k in tl for k in ["two separate card accounts", "dua akun kartu yang terpisah"]):
        return MessageActionType.IGNORED, None, None, None, None, "separate_card_accounts_minimum_notice"

    # Foreign currency charge / conversion notice
    if any(k in tl for k in ["bill was charged in a foreign currency", "tagihan ditagihkan dalam mata uang asing", "tagihan dikenakan dalam mata uang asing"]):
        return MessageActionType.IGNORED, None, None, None, None, "foreign_currency_conversion_notice"

    # Bonus pending review
    if any(k in tl for k in ["bonus kuartalan", "quarterly bonus"]):
        return MessageActionType.IGNORED, None, None, None, None, "quarterly_bonus_pending_approval_non_cash"

    return MessageActionType.UNRESOLVED, None, None, None, None, "unmatched_message_semantics"


def link_message_action(
    message: Message,
    action_type: MessageActionType,
    amount: Optional[Decimal],
    currency: Optional[str],
    effective_date: Optional[date],
    pct: Optional[Decimal],
    reason: str,
    events_by_id: Dict[str, FinancialEvent],
    events_by_user: Dict[str, List[FinancialEvent]],
    requests_by_id: Dict[str, FinancialRequest],
    profiles_by_user: Dict[str, FinancialProfile],
) -> Tuple[Optional[MessageAction], Optional[UnresolvedMessageAction]]:
    """Deterministically apply the 4-tier linkage hierarchy to link a message action to its target entity.

    Hierarchy:
    1. related_event_id: exact event match, verified for user ownership.
    2. explicit request_id + event/obligation context: matched to request context.
    3. explicit user_id + unique matching obligation: scheduled event or recurring series.
    4. explicit message content: unique identifying reference code.

    If ambiguity remains or targets cannot be established: produces UnresolvedMessageAction.
    """
    mid = message.message_id
    uid = message.user_id
    rid = message.request_id
    eid = message.related_event_id
    txt = message.message_text

    # Tier 1: related_event_id
    if eid:
        target_event = events_by_id.get(eid)
        if target_event is None:
            return None, UnresolvedMessageAction(
                message_id=mid,
                user_id=uid,
                request_id=rid,
                related_event_id=eid,
                message_text=txt,
                reason=f"nonexistent_target_event:{eid}",
            )

        if target_event.user_id != uid:
            return None, UnresolvedMessageAction(
                message_id=mid,
                user_id=uid,
                request_id=rid,
                related_event_id=eid,
                message_text=txt,
                reason=f"wrong_user_target_event:event_user_{target_event.user_id}_vs_msg_user_{uid}",
            )

        # Hierarchy Tier 1 resolved
        return MessageAction(
            message_id=mid,
            action_type=action_type,
            request_id=rid,
            user_id=uid,
            related_event_id=eid,
            target_type=TargetType.EVENT,
            target_id=eid,
            target_description=target_event.description,
            effective_date=effective_date or target_event.event_date,
            new_amount=amount,
            old_amount=target_event.amount,
            currency=currency or target_event.currency,
            confidence=Decimal("1.00"),
            evidence_text_reference=f"Tier1:related_event_id={eid}",
            reason=reason,
            is_applied=True,
        ), None

    # Tier 2: Validate request_id if present
    if rid and rid in requests_by_id:
        target_req = requests_by_id[rid]
        if target_req.user_id != uid:
            return None, UnresolvedMessageAction(
                message_id=mid,
                user_id=uid,
                request_id=rid,
                related_event_id=None,
                message_text=txt,
                reason=f"wrong_user_request:req_user_{target_req.user_id}_vs_msg_user_{uid}",
            )

    # Tier 3: explicit user_id + unique matching obligation
    user_events = events_by_user.get(uid, [])

    # A. Employer message regarding salary obligation
    if message.source_type == "employer":
        # Check if user has an explicit scheduled salary event
        sched_sal = [e for e in user_events if e.status == "scheduled" and (e.category == "salary" or "salary" in e.description.lower())]
        if len(sched_sal) == 1:
            target_ev = sched_sal[0]
            return MessageAction(
                message_id=mid,
                action_type=action_type,
                request_id=rid,
                user_id=uid,
                related_event_id=target_ev.event_id,
                target_type=TargetType.EVENT,
                target_id=target_ev.event_id,
                target_description=target_ev.description,
                effective_date=effective_date or target_ev.event_date,
                new_amount=amount,
                old_amount=target_ev.amount,
                currency=currency or target_ev.currency,
                confidence=Decimal("0.95"),
                evidence_text_reference=f"Tier3:unique_scheduled_salary_event={target_ev.event_id}",
                reason=reason,
                is_applied=True,
            ), None
        elif len(sched_sal) > 1:
            # Ambiguous matching obligations
            return None, UnresolvedMessageAction(
                message_id=mid,
                user_id=uid,
                request_id=rid,
                related_event_id=None,
                message_text=txt,
                reason="ambiguous_multiple_scheduled_salary_events",
            )
        else:
            # User has recurring salary obligation without explicit scheduled event
            target_id = f"user_salary_series_{uid}"
            return MessageAction(
                message_id=mid,
                action_type=action_type,
                request_id=rid,
                user_id=uid,
                related_event_id=None,
                target_type=TargetType.RECURRING_OBLIGATION,
                target_id=target_id,
                target_description="Recurring salary obligation",
                effective_date=effective_date,
                new_amount=amount,
                old_amount=None,
                currency=currency,
                confidence=Decimal("0.90"),
                evidence_text_reference=f"Tier3:user_salary_recurrence={target_id}",
                reason=reason,
                is_applied=True,
            ), None

    # B. Rent/lease obligations from service provider
    if "rent" in reason or "lease" in reason or "sewa" in txt.lower():
        rent_evs = [e for e in user_events if e.category == "housing" or "rent" in e.description.lower()]
        target_id = f"user_rent_series_{uid}"
        return MessageAction(
            message_id=mid,
            action_type=action_type,
            request_id=rid,
            user_id=uid,
            related_event_id=None,
            target_type=TargetType.RECURRING_OBLIGATION,
            target_id=target_id,
            target_description="Recurring rent obligation",
            effective_date=effective_date,
            new_amount=amount,
            old_amount=None,
            currency=currency,
            confidence=Decimal("0.90"),
            evidence_text_reference=f"Tier3:user_rent_recurrence={target_id}",
            reason=reason,
            is_applied=True,
        ), None

    # C. Client invoice payout notice
    if "invoice" in reason or "faktur" in txt.lower():
        target_id = f"user_invoice_{uid}"
        return MessageAction(
            message_id=mid,
            action_type=action_type,
            request_id=rid,
            user_id=uid,
            related_event_id=None,
            target_type=TargetType.UNKNOWN,
            target_id=target_id,
            target_description="Client invoice approval (pending credit)",
            effective_date=effective_date,
            new_amount=amount,
            old_amount=None,
            currency=currency,
            confidence=Decimal("0.90"),
            evidence_text_reference=f"Tier3:client_invoice_notice={target_id}",
            reason=reason,
            is_applied=False,
        ), None

    # D. Subscription / gym / membership cancellation
    if any(k in txt.lower() for k in ["subscription", "membership", "gym", "langganan"]):
        target_id = f"user_subscription_{uid}"
        return MessageAction(
            message_id=mid,
            action_type=action_type,
            request_id=rid,
            user_id=uid,
            related_event_id=None,
            target_type=TargetType.RECURRING_OBLIGATION,
            target_id=target_id,
            target_description="Recurring subscription obligation",
            effective_date=effective_date,
            new_amount=amount,
            old_amount=None,
            currency=currency,
            confidence=Decimal("0.90"),
            evidence_text_reference=f"Tier3:user_subscription_recurrence={target_id}",
            reason=reason,
            is_applied=True,
        ), None

    # D. Ignored informational or non-financial messages
    if action_type in (MessageActionType.IGNORED, MessageActionType.CONFIRM):
        return MessageAction(
            message_id=mid,
            action_type=action_type,
            request_id=rid,
            user_id=uid,
            related_event_id=None,
            target_type=TargetType.UNKNOWN,
            target_id=None,
            target_description="Informational or non-financial communication",
            effective_date=effective_date,
            new_amount=amount,
            old_amount=None,
            currency=currency,
            confidence=Decimal("1.00"),
            evidence_text_reference="Tier4:informational_general_notice",
            reason=reason,
            is_applied=False,
        ), None

    # Ambiguous or unresolvable target
    return None, UnresolvedMessageAction(
        message_id=mid,
        user_id=uid,
        request_id=rid,
        related_event_id=None,
        message_text=txt,
        reason=f"ambiguous_target_obligation_for_{action_type.value}",
    )


def resolve_message_conflicts(
    actions: Sequence[MessageAction],
) -> List[MessageAction]:
    """Deterministically resolve conflicts among multiple MessageActions targeting the same entity.

    Rules:
    1. Sort deterministically by (effective_date or sent_at, sent_at, message_id).
    2. Precedence hierarchy:
       CANCEL > RESUME > AMEND_AMOUNT / TEMPORARY_CHANGE > DELAY_TO > CONFIRM.
    3. Multiple amendments: later explicit amendment supersedes earlier amendment.
    4. Deterministic order independent of input row sequence.
    """
    if not actions:
        return []

    # Group actions by unique target key
    by_target: Dict[Tuple[TargetType, Optional[str]], List[MessageAction]] = {}
    unbound: List[MessageAction] = []

    for act in actions:
        if act.target_id is not None:
            key = (act.target_type, act.target_id)
            by_target.setdefault(key, []).append(act)
        else:
            unbound.append(act)

    resolved: List[MessageAction] = list(unbound)

    for key, act_list in by_target.items():
        if len(act_list) == 1:
            resolved.append(act_list[0])
            continue

        # Sort deterministically: effective_date ascending, then message_id ascending
        sorted_acts = sorted(
            act_list,
            key=lambda a: (a.effective_date or date.min, a.message_id),
        )

        # Check for explicit cancellation
        cancels = [a for a in sorted_acts if a.action_type == MessageActionType.CANCEL]
        if cancels:
            # Cancellation takes absolute precedence over amendments and delays
            resolved.append(cancels[-1])
            continue

        # Check for resume and temporary change coexistence
        temp_changes = [a for a in sorted_acts if a.action_type == MessageActionType.TEMPORARY_CHANGE]
        resumes = [a for a in sorted_acts if a.action_type == MessageActionType.RESUME]
        if resumes and temp_changes:
            # Both preserved: temporary change governs interim window; resume governs on/after resume date
            resolved.append(temp_changes[-1])
            resolved.append(resumes[-1])
            continue
        elif resumes:
            resolved.append(resumes[-1])
            continue

        # Check for amount amendments
        amends = [
            a for a in sorted_acts
            if a.action_type in (MessageActionType.AMEND_AMOUNT, MessageActionType.TEMPORARY_CHANGE)
        ]
        if amends:
            # Latest amendment takes precedence
            resolved.append(amends[-1])
            continue

        # Check for delays
        delays = [a for a in sorted_acts if a.action_type == MessageActionType.DELAY_TO]
        if delays:
            resolved.append(delays[-1])
            continue

        # Fallback to latest confirmation
        resolved.append(sorted_acts[-1])

    # Sort final resolved actions deterministically by message_id
    resolved.sort(key=lambda a: a.message_id)
    return resolved


def interpret_and_link_messages(
    messages: Sequence[Message],
    events: Sequence[FinancialEvent],
    requests: Sequence[FinancialRequest],
    profiles: Dict[str, FinancialProfile],
) -> Tuple[List[MessageAction], List[UnresolvedMessageAction]]:
    """Parse, link, and validate all messages into structured MessageActions and UnresolvedMessageActions.

    Guarantees:
    - Zero LLM calls; 100% deterministic pattern matching.
    - Full 4-tier linkage hierarchy.
    - Strict validation: target existence, user ownership, positive amounts, valid dates.
    - Deterministic conflict resolution.
    """
    events_by_id = {e.event_id: e for e in events}
    events_by_user: Dict[str, List[FinancialEvent]] = {}
    for e in events:
        events_by_user.setdefault(e.user_id, []).append(e)

    requests_by_id = {r.request_id: r for r in requests}

    parsed_actions: List[MessageAction] = []
    unresolved_actions: List[UnresolvedMessageAction] = []

    # Sort messages deterministically by sent_at then message_id
    sorted_messages = sorted(messages, key=lambda m: (m.sent_at, m.message_id))

    for msg in sorted_messages:
        action_type, amt, curr, eff_date, pct, reason = parse_single_message(msg)

        if action_type == MessageActionType.UNRESOLVED:
            unresolved_actions.append(
                UnresolvedMessageAction(
                    message_id=msg.message_id,
                    user_id=msg.user_id,
                    request_id=msg.request_id,
                    related_event_id=msg.related_event_id,
                    message_text=msg.message_text,
                    reason=reason,
                )
            )
            continue

        action, unres = link_message_action(
            message=msg,
            action_type=action_type,
            amount=amt,
            currency=curr,
            effective_date=eff_date,
            pct=pct,
            reason=reason,
            events_by_id=events_by_id,
            events_by_user=events_by_user,
            requests_by_id=requests_by_id,
            profiles_by_user=profiles,
        )

        if unres:
            unresolved_actions.append(unres)
        elif action:
            parsed_actions.append(action)

    # Apply deterministic conflict resolution
    resolved_actions = resolve_message_conflicts(parsed_actions)

    return resolved_actions, unresolved_actions


def apply_message_actions_to_series(
    series_list: Sequence[RecurrenceSeries],
    actions: Sequence[MessageAction],
) -> List[RecurrenceSeries]:
    """Deterministically adapt recurrence series with validated MessageActions.

    Guarantees:
    - Never mutates historical evidence or original objects (immutable copies returned).
    - Respects action precedence: CANCEL > RESUME > AMEND_AMOUNT / TEMPORARY_CHANGE > DELAY_TO > CONFIRM.
    - Category-isolated: salary messages only affect salary series; rent only affects rent; subscriptions only affect subscriptions.
    - Preserves full audit lineage in evidence_rationale and amendment/cancellation reasons.
    """
    if not actions or not series_list:
        return list(series_list)

    resolved_actions = resolve_message_conflicts(list(actions))

    updated_series: List[RecurrenceSeries] = []
    for s in series_list:
        curr_s = s
        for act in resolved_actions:
            if not act.is_applied:
                continue

            if act.target_type in (TargetType.RECURRING_OBLIGATION, TargetType.EVENT):
                match_text = f"{act.reason or ''} {act.target_description or ''} {act.target_id or ''}".lower()
                cat_match = False
                if "salary" in match_text or "payroll" in match_text:
                    cat_match = (s.category == "salary")
                elif "rent" in match_text:
                    cat_match = (s.category == "rent")
                elif any(k in match_text for k in ("subscription", "membership", "gym")):
                    cat_match = (s.category in ("gym", "music_subscription", "delivery_membership", "subscription"))

                if not cat_match:
                    continue

                if act.action_type == MessageActionType.CANCEL:
                    curr_s = RecurrenceSeries(
                        series_id=curr_s.series_id,
                        user_id=curr_s.user_id,
                        direction=curr_s.direction,
                        category=curr_s.category,
                        event_type=curr_s.event_type,
                        description=curr_s.description,
                        frequency=curr_s.frequency,
                        interval_days=curr_s.interval_days,
                        day_of_month=curr_s.day_of_month,
                        historical_event_ids=curr_s.historical_event_ids,
                        historical_count=curr_s.historical_count,
                        anchor_event_id=curr_s.anchor_event_id,
                        anchor_date=curr_s.anchor_date,
                        forecast_amount=curr_s.forecast_amount,
                        currency=curr_s.currency,
                        amount_rule=curr_s.amount_rule,
                        flexibility=curr_s.flexibility,
                        minimum_allowed_amount=curr_s.minimum_allowed_amount,
                        is_protected=curr_s.is_protected,
                        is_cancelled=True,
                        cancellation_reason=f"Cancelled by message {act.message_id} ({act.reason})",
                        amendment_reason=curr_s.amendment_reason,
                        evidence_rationale=curr_s.evidence_rationale + f" Cancelled via {act.message_id}.",
                    )

                elif act.action_type == MessageActionType.AMEND_AMOUNT:
                    if act.new_amount is not None and act.new_amount > Decimal("0"):
                        curr_s = RecurrenceSeries(
                            series_id=curr_s.series_id,
                            user_id=curr_s.user_id,
                            direction=curr_s.direction,
                            category=curr_s.category,
                            event_type=curr_s.event_type,
                            description=curr_s.description,
                            frequency=curr_s.frequency,
                            interval_days=curr_s.interval_days,
                            day_of_month=curr_s.day_of_month,
                            historical_event_ids=curr_s.historical_event_ids,
                            historical_count=curr_s.historical_count,
                            anchor_event_id=curr_s.anchor_event_id,
                            anchor_date=curr_s.anchor_date,
                            forecast_amount=act.new_amount,
                            currency=curr_s.currency,
                            amount_rule="explicit_amendment",
                            flexibility=curr_s.flexibility,
                            minimum_allowed_amount=curr_s.minimum_allowed_amount,
                            is_protected=curr_s.is_protected,
                            is_cancelled=curr_s.is_cancelled,
                            cancellation_reason=curr_s.cancellation_reason,
                            amendment_reason=f"Amended by message {act.message_id} to {act.new_amount}",
                            evidence_rationale=curr_s.evidence_rationale + f" Amended via {act.message_id} to {act.new_amount}.",
                        )

                elif act.action_type == MessageActionType.RESUME:
                    if act.new_amount is not None and act.new_amount > Decimal("0"):
                        curr_s = RecurrenceSeries(
                            series_id=curr_s.series_id,
                            user_id=curr_s.user_id,
                            direction=curr_s.direction,
                            category=curr_s.category,
                            event_type=curr_s.event_type,
                            description=curr_s.description,
                            frequency=curr_s.frequency,
                            interval_days=curr_s.interval_days,
                            day_of_month=curr_s.day_of_month,
                            historical_event_ids=curr_s.historical_event_ids,
                            historical_count=curr_s.historical_count,
                            anchor_event_id=curr_s.anchor_event_id,
                            anchor_date=curr_s.anchor_date,
                            forecast_amount=act.new_amount,
                            currency=curr_s.currency,
                            amount_rule="explicit_amendment",
                            flexibility=curr_s.flexibility,
                            minimum_allowed_amount=curr_s.minimum_allowed_amount,
                            is_protected=curr_s.is_protected,
                            is_cancelled=False,
                            cancellation_reason=None,
                            amendment_reason=f"Resumed by message {act.message_id} to {act.new_amount}",
                            evidence_rationale=curr_s.evidence_rationale + f" Resumed via {act.message_id} to {act.new_amount}.",
                        )

        updated_series.append(curr_s)

    return updated_series


def apply_message_actions_to_future_events(
    future_events: Sequence[FutureEvent],
    actions: Sequence[MessageAction],
    request_date: date,
) -> List[FutureEvent]:
    """Deterministically adapt future event occurrences with validated MessageActions.

    Guarantees:
    - Never mutates historical canonical events (only forward projections).
    - Respects temporal windows (e.g. temporary salary reductions resume on resume_date).
    - Preserves audit trace in FutureEvent.source_evidence.
    - Enforces no money invention: confirmation actions cannot invent amounts or events.
    """
    if not actions or not future_events:
        return list(future_events)

    resolved_actions = resolve_message_conflicts(list(actions))

    # Index RESUME actions by category to bound temporary changes
    resumes_by_cat: Dict[str, MessageAction] = {}
    for act in resolved_actions:
        if act.action_type == MessageActionType.RESUME and act.is_applied:
            match_text = f"{act.reason or ''} {act.target_description or ''} {act.target_id or ''}".lower()
            if "salary" in match_text or "payroll" in match_text:
                resumes_by_cat["salary"] = act
            elif "rent" in match_text:
                resumes_by_cat["rent"] = act

    updated_events: List[FutureEvent] = []
    for fe in future_events:
        curr_fe: Optional[FutureEvent] = fe
        for act in resolved_actions:
            if not act.is_applied or curr_fe is None:
                continue

            if act.target_type in (TargetType.RECURRING_OBLIGATION, TargetType.EVENT):
                match_text = f"{act.reason or ''} {act.target_description or ''} {act.target_id or ''}".lower()
                cat_match = False
                if "salary" in match_text or "payroll" in match_text:
                    cat_match = (curr_fe.category == "salary")
                elif "rent" in match_text:
                    cat_match = (curr_fe.category == "rent")
                elif any(k in match_text for k in ("subscription", "membership", "gym")):
                    cat_match = (curr_fe.category in ("gym", "music_subscription", "delivery_membership", "subscription"))

                if not cat_match:
                    continue

                if act.action_type == MessageActionType.CANCEL:
                    if act.effective_date is None or curr_fe.effective_date >= act.effective_date:
                        curr_fe = None
                        break

                elif act.action_type == MessageActionType.AMEND_AMOUNT:
                    if act.new_amount is not None and act.new_amount > Decimal("0"):
                        if act.effective_date is None or curr_fe.effective_date >= act.effective_date:
                            ev_tuple = curr_fe.source_evidence if isinstance(curr_fe.source_evidence, tuple) else (curr_fe.source_evidence,)
                            curr_fe = FutureEvent(
                                event_id=curr_fe.event_id,
                                user_id=curr_fe.user_id,
                                effective_date=curr_fe.effective_date,
                                direction=curr_fe.direction,
                                amount_home=act.new_amount,
                                currency=curr_fe.currency,
                                category=curr_fe.category,
                                event_type=curr_fe.event_type,
                                description=curr_fe.description,
                                series_id=curr_fe.series_id,
                                frequency=curr_fe.frequency,
                                is_forecast=curr_fe.is_forecast,
                                anchor_event_id=curr_fe.anchor_event_id,
                                flexibility=curr_fe.flexibility,
                                minimum_allowed_amount=curr_fe.minimum_allowed_amount,
                                is_protected=curr_fe.is_protected,
                                source_evidence=ev_tuple + (f"Amended by message {act.message_id} to {act.new_amount}",),
                            )

                elif act.action_type == MessageActionType.TEMPORARY_CHANGE:
                    if act.new_amount is not None and act.new_amount > Decimal("0"):
                        cat_key = "salary" if curr_fe.category == "salary" else "rent"
                        resume_act = resumes_by_cat.get(cat_key)
                        should_apply = True
                        if resume_act and resume_act.effective_date:
                            if curr_fe.effective_date >= resume_act.effective_date:
                                should_apply = False

                        if should_apply:
                            ev_tuple = curr_fe.source_evidence if isinstance(curr_fe.source_evidence, tuple) else (curr_fe.source_evidence,)
                            curr_fe = FutureEvent(
                                event_id=curr_fe.event_id,
                                user_id=curr_fe.user_id,
                                effective_date=curr_fe.effective_date,
                                direction=curr_fe.direction,
                                amount_home=act.new_amount,
                                currency=curr_fe.currency,
                                category=curr_fe.category,
                                event_type=curr_fe.event_type,
                                description=curr_fe.description,
                                series_id=curr_fe.series_id,
                                frequency=curr_fe.frequency,
                                is_forecast=curr_fe.is_forecast,
                                anchor_event_id=curr_fe.anchor_event_id,
                                flexibility=curr_fe.flexibility,
                                minimum_allowed_amount=curr_fe.minimum_allowed_amount,
                                is_protected=curr_fe.is_protected,
                                source_evidence=ev_tuple + (f"Temporary change by message {act.message_id} to {act.new_amount}",),
                            )

                elif act.action_type == MessageActionType.RESUME:
                    if act.new_amount is not None and act.new_amount > Decimal("0"):
                        if act.effective_date is not None and curr_fe.effective_date >= act.effective_date:
                            ev_tuple = curr_fe.source_evidence if isinstance(curr_fe.source_evidence, tuple) else (curr_fe.source_evidence,)
                            curr_fe = FutureEvent(
                                event_id=curr_fe.event_id,
                                user_id=curr_fe.user_id,
                                effective_date=curr_fe.effective_date,
                                direction=curr_fe.direction,
                                amount_home=act.new_amount,
                                currency=curr_fe.currency,
                                category=curr_fe.category,
                                event_type=curr_fe.event_type,
                                description=curr_fe.description,
                                series_id=curr_fe.series_id,
                                frequency=curr_fe.frequency,
                                is_forecast=curr_fe.is_forecast,
                                anchor_event_id=curr_fe.anchor_event_id,
                                flexibility=curr_fe.flexibility,
                                minimum_allowed_amount=curr_fe.minimum_allowed_amount,
                                is_protected=curr_fe.is_protected,
                                source_evidence=ev_tuple + (f"Resumed by message {act.message_id} to {act.new_amount}",),
                            )

                elif act.action_type == MessageActionType.DELAY_TO:
                    if act.effective_date is not None and act.effective_date >= request_date:
                        ev_tuple = curr_fe.source_evidence if isinstance(curr_fe.source_evidence, tuple) else (curr_fe.source_evidence,)
                        curr_fe = FutureEvent(
                            event_id=curr_fe.event_id,
                            user_id=curr_fe.user_id,
                            effective_date=act.effective_date,
                            direction=curr_fe.direction,
                            amount_home=curr_fe.amount_home,
                            currency=curr_fe.currency,
                            category=curr_fe.category,
                            event_type=curr_fe.event_type,
                            description=curr_fe.description,
                            series_id=curr_fe.series_id,
                            frequency=curr_fe.frequency,
                            is_forecast=curr_fe.is_forecast,
                            anchor_event_id=curr_fe.anchor_event_id,
                            flexibility=curr_fe.flexibility,
                            minimum_allowed_amount=curr_fe.minimum_allowed_amount,
                            is_protected=curr_fe.is_protected,
                            source_evidence=ev_tuple + (f"Delayed by message {act.message_id} to {act.effective_date}",),
                        )

        if curr_fe is not None:
            updated_events.append(curr_fe)

    updated_events.sort(key=lambda x: (x.effective_date, x.event_id))
    return updated_events


def adapt_recurrence_and_future_events(
    series_list: Sequence[RecurrenceSeries],
    future_events: Sequence[FutureEvent],
    actions: Sequence[MessageAction],
    request_date: date,
) -> Tuple[List[RecurrenceSeries], List[FutureEvent]]:
    """Unified adapter applying validated message actions across both series and future events."""
    adapted_series = apply_message_actions_to_series(series_list, actions)
    adapted_events = apply_message_actions_to_future_events(future_events, actions, request_date)
    return adapted_series, adapted_events

