"""Deterministic Affordability Classification Layer for Buy or Wait?

Consumes authoritative outputs from:
1. safe_to_pay.py (amount_safe_to_pay, earliest_date_for_full_payment, SafeToPayCertificate)
2. Downstream payment_plan layer contract (PaymentPlanFeasibility protocol)

Implements a pure deterministic classification function that categorizes financial requests
into the official contest statuses:
- affordable_now
- affordable_with_plan
- affordable_later
- not_affordable
- plan_evaluation_required (provisional state when downstream plan has not been evaluated)

Does not modify upstream simulator or safe_to_pay state.
Does not call any LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional, Protocol, Union, runtime_checkable

from code.models import FinancialProfile, FinancialRequest
from code.safe_to_pay import SafeToPayCertificate


class AffordabilityStatus(str, Enum):
    """Allowed affordability status values under contest rules, plus provisional state."""
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"
    PLAN_EVALUATION_REQUIRED = "plan_evaluation_required"


def status_rank(status: Union[AffordabilityStatus, str]) -> int:
    """Return ordinal rank for monotonic comparison in property tests.

    Higher rank corresponds to strictly more favorable / immediate affordability.
    AFFORDABLE_NOW: 3
    AFFORDABLE_WITH_PLAN: 2
    AFFORDABLE_LATER: 1
    NOT_AFFORDABLE: 0
    PLAN_EVALUATION_REQUIRED: -1
    """
    val = status.value if isinstance(status, AffordabilityStatus) else str(status)
    if val == AffordabilityStatus.AFFORDABLE_NOW.value:
        return 3
    if val == AffordabilityStatus.AFFORDABLE_WITH_PLAN.value:
        return 2
    if val == AffordabilityStatus.AFFORDABLE_LATER.value:
        return 1
    if val == AffordabilityStatus.NOT_AFFORDABLE.value:
        return 0
    if val == AffordabilityStatus.PLAN_EVALUATION_REQUIRED.value:
        return -1
    raise ValueError(f"Unknown status value: {val}")


@runtime_checkable
class PaymentPlanFeasibility(Protocol):
    """Contract for downstream payment plan evaluation layer."""
    @property
    def has_valid_plan(self) -> bool:
        """True if a valid deterministic payment plan exists satisfying all safety constraints."""
        ...

    @property
    def plan_type(self) -> Optional[str]:
        """Type of payment plan (e.g. 'installments', 'partial_payment', 'spending_changes')."""
        ...


@dataclass(frozen=True)
class AffordabilityCertificateReference:
    """Read-only reference to upstream SafeToPayCertificate evidence."""
    request_id: str
    safety_floor: Decimal
    baseline_minimum_available_cash: Decimal
    limiting_date: date
    safety_floor_margin: Decimal
    candidate_search_method: str


@dataclass(frozen=True)
class AffordabilityResult:
    """Immutable deterministic result of affordability classification."""
    request_id: str
    user_id: str
    status: AffordabilityStatus
    requested_amount: Decimal
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: Optional[date]
    reason: str
    is_provisional: bool
    requires_payment_plan_evaluation: bool
    provisional_status: Optional[AffordabilityStatus]
    certificate_reference: AffordabilityCertificateReference
    certificate: Optional[SafeToPayCertificate] = None

    @property
    def is_affordable_now(self) -> bool:
        return self.status == AffordabilityStatus.AFFORDABLE_NOW

    @property
    def is_affordable_with_plan(self) -> bool:
        return self.status == AffordabilityStatus.AFFORDABLE_WITH_PLAN

    @property
    def is_affordable_later(self) -> bool:
        return self.status == AffordabilityStatus.AFFORDABLE_LATER

    @property
    def is_not_affordable(self) -> bool:
        return self.status == AffordabilityStatus.NOT_AFFORDABLE

    @property
    def status_value(self) -> str:
        return self.status.value


def classify_affordability(
    certificate: SafeToPayCertificate,
    profile: FinancialProfile,
    request: FinancialRequest,
    payment_plan_feasible: Optional[Union[bool, PaymentPlanFeasibility]] = None,
    allow_provisional: bool = False,
    respect_deadline: bool = False,
    require_full_payment_preference_for_later: bool = False,
) -> AffordabilityResult:
    """Deterministic affordability classification function.

    Precedence:
    1. affordable_now:
       - Full requested amount is safe to pay on request_date AND user accepts full_payment.
    2. affordable_with_plan:
       - Downstream payment plan is feasible (valid installments, partial payment, or spending changes)
         when full payment is not safe today OR user refuses full payment.
    3. affordable_later:
       - Full payment is safe on an earliest date later in the 90-day simulation horizon,
         and no valid payment plan exists.
    4. not_affordable:
       - No safe full payment within horizon and no valid payment plan exists.
    5. plan_evaluation_required (provisional):
       - Full payment not safe today (or refused by user policy), and payment_plan_feasible is None.

    Args:
        certificate: Authoritative audit certificate from safe_to_pay.py.
        profile: User financial profile containing payment preferences and constraints.
        request: Financial request under evaluation.
        payment_plan_feasible: Optional bool or PaymentPlanFeasibility protocol object.
            None = payment-plan layer not evaluated yet.
            True = a valid deterministic plan exists.
            False = no valid plan exists.
        allow_provisional: If True and payment_plan_feasible is None, resolves status to the
            provisional fallback (affordable_later or not_affordable) with is_provisional=True.
            If False (default), preserves plan_evaluation_required.
        respect_deadline: If True, requires earliest_date <= desired_completion_date for affordable_later.
            Defaults to False (pure 90-day horizon financial capacity).
        require_full_payment_preference_for_later: If True, requires user to accept full_payment for
            affordable_later. Defaults to False (measures capacity independently of payment method preference).

    Returns:
        Immutable AffordabilityResult.
    """
    req_amt = certificate.requested_amount
    safe_amt = certificate.amount_safe_to_pay
    earliest_date = certificate.earliest_date_for_full_payment
    user_accepts_full_payment = "full_payment" in profile.payment_methods_user_will_consider

    cert_ref = AffordabilityCertificateReference(
        request_id=certificate.request_id,
        safety_floor=certificate.safety_floor,
        baseline_minimum_available_cash=certificate.baseline_minimum_available_cash,
        limiting_date=certificate.limiting_date,
        safety_floor_margin=certificate.safety_floor_margin,
        candidate_search_method=certificate.candidate_search_method,
    )

    # 1. Unpack payment plan feasibility contract
    plan_valid: Optional[bool] = None
    if payment_plan_feasible is None:
        plan_valid = None
    elif isinstance(payment_plan_feasible, bool):
        plan_valid = payment_plan_feasible
    elif hasattr(payment_plan_feasible, "has_valid_plan"):
        plan_valid = bool(payment_plan_feasible.has_valid_plan)
    else:
        raise TypeError(f"Invalid payment_plan_feasible type: {type(payment_plan_feasible)}")

    # =========================================================================
    # PRECEDENCE LEVEL 1: affordable_now
    # =========================================================================
    # Contest Rule: Full amount is safe to pay on request_date and user accepts full_payment.
    if certificate.is_full_payment_safe_today and user_accepts_full_payment:
        if req_amt == Decimal("0"):
            reason = "Zero requested amount is safe on request date."
        else:
            reason = f"Full requested amount {req_amt} is safe to pay on request date {request.request_date.isoformat()}."
        return AffordabilityResult(
            request_id=request.request_id,
            user_id=request.user_id,
            status=AffordabilityStatus.AFFORDABLE_NOW,
            requested_amount=req_amt,
            amount_safe_to_pay=safe_amt,
            earliest_date_for_full_payment=request.request_date,
            reason=reason,
            is_provisional=False,
            requires_payment_plan_evaluation=False,
            provisional_status=None,
            certificate_reference=cert_ref,
            certificate=certificate,
        )

    # =========================================================================
    # PRECEDENCE LEVEL 2: affordable_with_plan
    # =========================================================================
    # If a payment plan is verified feasible by the downstream plan layer:
    if plan_valid is True:
        if certificate.is_full_payment_safe_today and not user_accepts_full_payment:
            reason = (
                f"Full amount {req_amt} is financially safe today, but user policy requires an "
                f"installment or partial payment plan."
            )
        else:
            reason = (
                f"Full amount not safe today (safe: {safe_amt} of {req_amt}); "
                f"completed safely via valid payment plan."
            )
        return AffordabilityResult(
            request_id=request.request_id,
            user_id=request.user_id,
            status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            requested_amount=req_amt,
            amount_safe_to_pay=safe_amt,
            earliest_date_for_full_payment=earliest_date,
            reason=reason,
            is_provisional=False,
            requires_payment_plan_evaluation=False,
            provisional_status=None,
            certificate_reference=cert_ref,
            certificate=certificate,
        )

    # Helper evaluating whether later full payment is acceptable
    deadline_ok = (not respect_deadline) or (earliest_date is not None and earliest_date <= request.desired_completion_date)
    is_later_date = (earliest_date is not None and earliest_date > request.request_date)
    method_ok = (not require_full_payment_preference_for_later) or user_accepts_full_payment
    is_affordable_later_eligible = is_later_date and deadline_ok and method_ok

    # =========================================================================
    # PRECEDENCE LEVEL 3 & 4: affordable_later vs not_affordable (Plan Infeasible)
    # =========================================================================
    if plan_valid is False:
        if is_affordable_later_eligible and earliest_date is not None:
            reason = f"Full amount safe from {earliest_date.isoformat()}."
            return AffordabilityResult(
                request_id=request.request_id,
                user_id=request.user_id,
                status=AffordabilityStatus.AFFORDABLE_LATER,
                requested_amount=req_amt,
                amount_safe_to_pay=safe_amt,
                earliest_date_for_full_payment=earliest_date,
                reason=reason,
                is_provisional=False,
                requires_payment_plan_evaluation=False,
                provisional_status=None,
                certificate_reference=cert_ref,
                certificate=certificate,
            )

        # Otherwise not_affordable
        if earliest_date is None:
            reason = "No safe full-payment date within 90-day simulation horizon and no valid payment plan exists."
        elif not user_accepts_full_payment and certificate.is_full_payment_safe_today:
            reason = "Full amount is safe today, but user will not consider full payment and no valid payment plan exists."
        elif not user_accepts_full_payment and require_full_payment_preference_for_later:
            reason = "Full payment safe later, but user will not consider full payment and no valid payment plan exists."
        elif not deadline_ok:
            reason = (
                f"Full payment safe on {earliest_date.isoformat()}, but exceeds desired completion date "
                f"{request.desired_completion_date.isoformat()} and no valid payment plan exists."
            )
        else:
            reason = "Request cannot be completed safely within forecast period."

        return AffordabilityResult(
            request_id=request.request_id,
            user_id=request.user_id,
            status=AffordabilityStatus.NOT_AFFORDABLE,
            requested_amount=req_amt,
            amount_safe_to_pay=safe_amt,
            earliest_date_for_full_payment=earliest_date,
            reason=reason,
            is_provisional=False,
            requires_payment_plan_evaluation=False,
            provisional_status=None,
            certificate_reference=cert_ref,
            certificate=certificate,
        )

    # =========================================================================
    # PRECEDENCE LEVEL 5: Plan Feasibility Unknown (plan_valid is None)
    # =========================================================================
    # Downstream payment-plan layer has not evaluated this request yet.
    # Determine what the provisional fallback would be if no plan exists:
    if is_affordable_later_eligible and earliest_date is not None:
        fallback_status = AffordabilityStatus.AFFORDABLE_LATER
        fallback_reason = f"Full amount safe from {earliest_date.isoformat()} (provisional; pending payment plan evaluation)."
    else:
        fallback_status = AffordabilityStatus.NOT_AFFORDABLE
        if earliest_date is None:
            fallback_reason = "No safe full payment within horizon (provisional; pending payment plan evaluation)."
        elif not user_accepts_full_payment and certificate.is_full_payment_safe_today:
            fallback_reason = "Full amount safe today but user policy refuses full payment (provisional; pending payment plan evaluation)."
        elif not user_accepts_full_payment and require_full_payment_preference_for_later:
            fallback_reason = "User will not consider full payment (provisional; pending payment plan evaluation)."
        else:
            fallback_reason = "Earliest full payment exceeds deadline (provisional; pending payment plan evaluation)."

    if allow_provisional:
        status = fallback_status
        reason = fallback_reason
    else:
        status = AffordabilityStatus.PLAN_EVALUATION_REQUIRED
        reason = f"Full amount not safe today; requires payment-plan evaluation (provisional fallback: {fallback_status.value})."

    return AffordabilityResult(
        request_id=request.request_id,
        user_id=request.user_id,
        status=status,
        requested_amount=req_amt,
        amount_safe_to_pay=safe_amt,
        earliest_date_for_full_payment=earliest_date,
        reason=reason,
        is_provisional=True,
        requires_payment_plan_evaluation=True,
        provisional_status=fallback_status,
        certificate_reference=cert_ref,
        certificate=certificate,
    )

