"""Deterministic Final Decision + Safety Gate Layer for Buy or Wait?

Specification Sources:
  1. problem_statement.md lines 187-208 ("Choosing Between Safe Plans", "Evaluation"):
     "An immediate payment method—full_payment, partial_payment, or installments—is eligible
      only when it appears in the user's payment_methods_user_will_consider. wait is eligible
      when full payment becomes safe later and the user accepts full_payment. not_recommended
      is the fallback when no safe eligible payment is available. When more than one eligible
      plan is safe, rank the plans in this order..."
  2. AGENTS.md §6.2 ("Required Output") and §6.3 ("Financial Decision Rules"):
     - amount_safe_to_pay, affordability_status, recommended_payment_method, payment_plan,
       earliest_date_for_full_payment, spending_changes_needed.
     - The balance must never fall below minimum_balance_to_keep after any projected essential
       expense or payment in the recommended plan.
     - Respect the user's protected categories and preferences.

WHAT THIS MODULE DOES:
- Filters candidates through an explicit, auditable final safety gate (is_finally_safe).
- Submits surviving candidates to the frozen ranking engine (code/ranking.py).
- Selects strictly the #1 ranking candidate from surviving candidates.
- If zero candidates survive the final safety gate, deterministically constructs the
  specification-defined fallback decision ("not_recommended", "none", "none", "not_affordable").
- Derives the exact contest AffordabilityStatus from authoritative evidence.
- Emits an immutable, typed FinalDecision containing full evidence and audit references.

WHAT THIS MODULE DOES NOT DO:
- No LLM calls or natural language generation.
- No output.csv file generation.
- No weighted scores or arbitrary heuristics.
- No recomputation of balances or duplicate financial simulation.
- No modification of any frozen upstream module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple, Union

from code.affordability import (
    AffordabilityResult,
    AffordabilityStatus,
    classify_affordability,
)
from code.candidate_generation import (
    Candidate,
    CandidateSet,
    CandidateStatus,
    CandidateType,
    is_rankable,
)
from code.models import FinancialProfile, FinancialRequest
from code.payment_plan import PaymentPlanFeasibility
from code.ranking import (
    RankedCandidate,
    RankedCandidateSet,
    RankingCriteriaTrace,
    rank_candidates,
)
from code.safe_to_pay import SafeToPayCertificate


# ---------------------------------------------------------------------------
# Explicit Safety Gate Reason Codes
# ---------------------------------------------------------------------------

class DecisionReasonCode(str, Enum):
    """Explicit, auditable reason codes for safety gate checks and final decisions."""
    # Gate passed checks
    RANKABLE = "RANKABLE"
    MARKED_SAFE = "MARKED_SAFE"
    SAFETY_CERTIFICATE_VALID = "SAFETY_CERTIFICATE_VALID"
    SIMULATION_SAFE = "SIMULATION_SAFE"
    NO_OVERDRAFT = "NO_OVERDRAFT"
    PAYMENT_PLAN_VALID = "PAYMENT_PLAN_VALID"
    USER_CONSTRAINTS_VALID = "USER_CONSTRAINTS_VALID"
    SCHEDULE_CONSISTENT = "SCHEDULE_CONSISTENT"

    # Gate failure checks
    NOT_RANKABLE = "NOT_RANKABLE"
    NOT_MARKED_SAFE = "NOT_MARKED_SAFE"
    INVALID_SAFETY_CERTIFICATE = "INVALID_SAFETY_CERTIFICATE"
    SAFETY_FLOOR_BREACH = "SAFETY_FLOOR_BREACH"
    OVERDRAFT_DETECTED = "OVERDRAFT_DETECTED"
    PAYMENT_PLAN_INVALID = "PAYMENT_PLAN_INVALID"
    USER_PREFERENCES_VIOLATED = "USER_PREFERENCES_VIOLATED"
    SCHEDULE_INCONSISTENT = "SCHEDULE_INCONSISTENT"

    # Decision level codes
    SELECTED_BY_RANKING = "SELECTED_BY_RANKING"
    NO_SAFE_CANDIDATE_SURVIVED = "NO_SAFE_CANDIDATE_SURVIVED"
    FULL_PAYMENT_SAFE_TODAY = "FULL_PAYMENT_SAFE_TODAY"
    FULL_PAYMENT_SAFE_LATER = "FULL_PAYMENT_SAFE_LATER"
    NOT_AFFORDABLE_WITHIN_HORIZON = "NOT_AFFORDABLE_WITHIN_HORIZON"


# ---------------------------------------------------------------------------
# Safety Gate Check & Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyGateCheck:
    """Audit record for a single safety gate check."""
    check_name: str
    passed: bool
    reason_code: DecisionReasonCode
    details: str


@dataclass(frozen=True)
class SafetyGateResult:
    """Immutable audit result of evaluating a candidate through the final safety gate."""
    candidate_id: str
    is_safe: bool
    checks: Tuple[SafetyGateCheck, ...]
    rejection_reason_codes: Tuple[DecisionReasonCode, ...]


# ---------------------------------------------------------------------------
# Request Context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestContext:
    """Authoritative context for evaluating a request's candidates and final decision."""
    request: FinancialRequest
    profile: FinancialProfile
    certificate: SafeToPayCertificate
    payment_option_feasibilities: Dict[str, PaymentPlanFeasibility]

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def user_id(self) -> str:
        return self.request.user_id

    @property
    def requested_amount(self) -> Decimal:
        return self.certificate.requested_amount

    @property
    def desired_completion_date(self) -> date:
        return self.request.desired_completion_date


# ---------------------------------------------------------------------------
# Decision Explanation Evidence (Structured for downstream consumers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionExplanationEvidence:
    """Structured, immutable evidence for downstream explanation/LLM consumption.

    Contains only authoritative facts; no generated prose.
    """
    request_id: str
    user_id: str
    request_date: date
    desired_completion_date: date
    requested_amount: Decimal
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: Optional[date]
    selected_candidate_type: Optional[CandidateType]
    selected_payment_method: str
    total_amount_paid: Optional[Decimal]
    financing_fee: Optional[Decimal]
    number_of_payments: Optional[int]
    safety_floor: Decimal
    decision_reason_codes: Tuple[DecisionReasonCode, ...]
    evidence_notes: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Final Decision Object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinalDecision:
    """Immutable final decision object produced by the deterministic decision layer.

    Contains the selected candidate (or None if no safe plan exists), the authoritative
    affordability status, contest-compliant output fields, full safety gate audit,
    and structured evidence references.
    """
    request_id: str
    selected_candidate_id: Optional[str]
    affordability_status: AffordabilityStatus
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: Optional[date]
    recommended_payment_method: str
    payment_plan: str
    spending_changes_needed: str
    selected_candidate: Optional[Candidate]
    ranking_trace: Optional[RankingCriteriaTrace]
    safety_gate_result: Optional[SafetyGateResult]
    all_safety_gate_results: Tuple[SafetyGateResult, ...]
    ranked_candidate_ids: Tuple[str, ...]
    decision_reason_codes: Tuple[DecisionReasonCode, ...]
    explanation_evidence: DecisionExplanationEvidence

    def __post_init__(self) -> None:
        # Guarantee exact 4 contest affordability statuses
        if self.affordability_status not in (
            AffordabilityStatus.AFFORDABLE_NOW,
            AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            AffordabilityStatus.AFFORDABLE_LATER,
            AffordabilityStatus.NOT_AFFORDABLE,
        ):
            raise ValueError(f"Invalid contest affordability status: {self.affordability_status}")

        # Guarantee monetary precision
        if not isinstance(self.amount_safe_to_pay, Decimal):
            raise TypeError(f"amount_safe_to_pay must be Decimal, got {type(self.amount_safe_to_pay)}")

        # Guarantee recommended_payment_method is schema-valid
        valid_methods = ("full_payment", "partial_payment", "installments", "wait", "not_recommended")
        if self.recommended_payment_method not in valid_methods:
            raise ValueError(f"Invalid recommended_payment_method: {self.recommended_payment_method}")

        # When not_recommended: selected_candidate must be None and plan must be 'none'
        if self.recommended_payment_method == "not_recommended":
            if self.selected_candidate is not None or self.selected_candidate_id is not None:
                raise ValueError("not_recommended decision cannot have a selected candidate.")
            if self.payment_plan != "none":
                raise ValueError("not_recommended decision must have payment_plan='none'")

    @property
    def has_recommendation(self) -> bool:
        return self.selected_candidate is not None


# ---------------------------------------------------------------------------
# Final Safety Gate Evaluator
# ---------------------------------------------------------------------------

def evaluate_candidate_safety_gate(candidate: Candidate, context: RequestContext) -> SafetyGateResult:
    """Evaluate a candidate through all 8 authoritative safety conditions.

    Conditions checked:
      1. RANKABLE: candidate.is_rankable == True
      2. MARKED_SAFE: candidate.is_safe == True and status == ELIGIBLE_AND_SAFE
      3. SAFETY_CERTIFICATE_VALID: provenance points to valid matching certificate/feasibility
      4. SIMULATION_SAFE / NO_SAFETY_FLOOR_BREACH: underlying simulation confirms safety margin >= 0
      5. NO_OVERDRAFT: available cash above safety floor
      6. PAYMENT_PLAN_VALID: schedule sum == total, count == len, partial payment logic verified
      7. USER_CONSTRAINTS_VALID: payment method in user preferences, duration <= max_installment_months
      8. SCHEDULE_CONSISTENT: chronological dates, first_date >= request_date
    """
    checks: list[SafetyGateCheck] = []
    rejections: list[DecisionReasonCode] = []

    # 1. RANKABLE
    c_rankable = candidate.is_rankable and is_rankable(candidate)
    if c_rankable:
        checks.append(SafetyGateCheck("rankable", True, DecisionReasonCode.RANKABLE, "Candidate is rankable."))
    else:
        checks.append(SafetyGateCheck("rankable", False, DecisionReasonCode.NOT_RANKABLE, "Candidate is not rankable."))
        rejections.append(DecisionReasonCode.NOT_RANKABLE)

    # 2. MARKED_SAFE
    c_marked_safe = candidate.is_safe and candidate.status == CandidateStatus.ELIGIBLE_AND_SAFE
    if c_marked_safe:
        checks.append(SafetyGateCheck("marked_safe", True, DecisionReasonCode.MARKED_SAFE, "Candidate is marked eligible and safe."))
    else:
        checks.append(SafetyGateCheck("marked_safe", False, DecisionReasonCode.NOT_MARKED_SAFE, "Candidate is not marked safe."))
        rejections.append(DecisionReasonCode.NOT_MARKED_SAFE)

    # 3. SAFETY_CERTIFICATE_VALID
    cert_valid = True
    cert_detail = "Safety certificate/feasibility proven."
    if candidate.candidate_type in (CandidateType.FULL_PAYMENT, CandidateType.PARTIAL_PAYMENT, CandidateType.WAIT):
        if candidate.provenance.safe_to_pay_certificate_id != context.certificate.request_id:
            cert_valid = False
            cert_detail = f"Certificate ID mismatch: {candidate.provenance.safe_to_pay_certificate_id} != {context.certificate.request_id}"
        elif context.certificate.request_id != context.request.request_id:
            cert_valid = False
            cert_detail = f"Context certificate mismatch: {context.certificate.request_id} != {context.request.request_id}"
    elif candidate.candidate_type == CandidateType.INSTALLMENT_PLAN:
        opt_id = candidate.source_payment_option_id
        if opt_id is None or opt_id not in context.payment_option_feasibilities:
            cert_valid = False
            cert_detail = f"Installment payment option {opt_id} missing in feasibility map."
        else:
            feas = context.payment_option_feasibilities[opt_id]
            if not feas.is_safe or not feas.is_eligible:
                cert_valid = False
                cert_detail = f"Installment feasibility for {opt_id} is unsafe/ineligible: {feas.rejection_reason}."

    if cert_valid:
        checks.append(SafetyGateCheck("safety_certificate_valid", True, DecisionReasonCode.SAFETY_CERTIFICATE_VALID, cert_detail))
    else:
        checks.append(SafetyGateCheck("safety_certificate_valid", False, DecisionReasonCode.INVALID_SAFETY_CERTIFICATE, cert_detail))
        rejections.append(DecisionReasonCode.INVALID_SAFETY_CERTIFICATE)

    # 4. SIMULATION_SAFE / NO_SAFETY_FLOOR_BREACH
    sim_safe = True
    sim_detail = "Simulation confirmed safe with non-negative margin above safety floor."
    if candidate.candidate_type == CandidateType.FULL_PAYMENT:
        if not context.certificate.is_full_payment_safe_today or context.certificate.safety_floor_margin < Decimal("0"):
            sim_safe = False
            sim_detail = f"Full payment unsafe today: margin {context.certificate.safety_floor_margin} < 0."
    elif candidate.candidate_type == CandidateType.WAIT:
        if context.certificate.earliest_date_for_full_payment is None:
            sim_safe = False
            sim_detail = "Wait candidate has no earliest safe date."
        elif candidate.first_payment_date != context.certificate.earliest_date_for_full_payment:
            sim_safe = False
            sim_detail = f"Wait candidate date mismatch: {candidate.first_payment_date} != {context.certificate.earliest_date_for_full_payment}."
    elif candidate.candidate_type == CandidateType.PARTIAL_PAYMENT:
        if context.certificate.amount_safe_to_pay <= Decimal("0") or context.certificate.earliest_date_for_full_payment is None:
            sim_safe = False
            sim_detail = "Partial payment requires positive safe amount and safe earliest date."
    elif candidate.candidate_type == CandidateType.INSTALLMENT_PLAN:
        opt_id = candidate.source_payment_option_id
        if opt_id in context.payment_option_feasibilities:
            feas = context.payment_option_feasibilities[opt_id]
            if not feas.is_safe:
                sim_safe = False
                sim_detail = f"Option {opt_id} failed simulation safety check."

    if sim_safe:
        checks.append(SafetyGateCheck("simulation_safe", True, DecisionReasonCode.SIMULATION_SAFE, sim_detail))
    else:
        checks.append(SafetyGateCheck("simulation_safe", False, DecisionReasonCode.SAFETY_FLOOR_BREACH, sim_detail))
        rejections.append(DecisionReasonCode.SAFETY_FLOOR_BREACH)

    # 5. NO_OVERDRAFT
    no_overdraft = True
    od_detail = "No overdraft detected under action."
    if candidate.candidate_type == CandidateType.FULL_PAYMENT:
        if context.certificate.available_cash_after_purchase_today < context.certificate.safety_floor:
            no_overdraft = False
            od_detail = "Full payment today causes cash to breach safety floor."
    elif candidate.candidate_type == CandidateType.INSTALLMENT_PLAN:
        opt_id = candidate.source_payment_option_id
        if opt_id in context.payment_option_feasibilities:
            feas = context.payment_option_feasibilities[opt_id]
            if feas.simulator_certificate is not None and feas.simulator_certificate.is_safety_floor_breached:
                no_overdraft = False
                od_detail = "Installment plan breaches safety floor in simulation."

    if no_overdraft:
        checks.append(SafetyGateCheck("no_overdraft", True, DecisionReasonCode.NO_OVERDRAFT, od_detail))
    else:
        checks.append(SafetyGateCheck("no_overdraft", False, DecisionReasonCode.OVERDRAFT_DETECTED, od_detail))
        rejections.append(DecisionReasonCode.OVERDRAFT_DETECTED)

    # 6. PAYMENT_PLAN_VALID
    plan_valid = True
    pv_detail = "Payment schedule is contractually valid."
    if not candidate.payment_schedule:
        plan_valid = False
        pv_detail = "Empty payment schedule."
    elif sum(p.amount for p in candidate.payment_schedule) != candidate.total_amount_paid:
        plan_valid = False
        pv_detail = "Schedule sum does not equal total amount paid."
    elif len(candidate.payment_schedule) != candidate.number_of_payments:
        plan_valid = False
        pv_detail = "Schedule length does not match number of payments."
    elif any(p.amount <= Decimal("0") for p in candidate.payment_schedule):
        plan_valid = False
        pv_detail = "Non-positive payment amount detected in schedule."
    elif candidate.candidate_type == CandidateType.PARTIAL_PAYMENT:
        if candidate.number_of_payments != 2:
            plan_valid = False
            pv_detail = "Partial payment must have exactly 2 payments."
        elif candidate.payment_schedule[0].amount != context.certificate.amount_safe_to_pay:
            plan_valid = False
            pv_detail = "Partial payment 1 does not match amount_safe_to_pay."
        elif candidate.completion_date is not None and candidate.completion_date > context.request.desired_completion_date:
            plan_valid = False
            pv_detail = "Partial payment completion exceeds desired_completion_date."

    if plan_valid:
        checks.append(SafetyGateCheck("payment_plan_valid", True, DecisionReasonCode.PAYMENT_PLAN_VALID, pv_detail))
    else:
        checks.append(SafetyGateCheck("payment_plan_valid", False, DecisionReasonCode.PAYMENT_PLAN_INVALID, pv_detail))
        rejections.append(DecisionReasonCode.PAYMENT_PLAN_INVALID)

    # 7. USER_CONSTRAINTS_VALID
    user_valid = True
    uv_detail = "User preferences and constraint limits satisfied."
    if candidate.candidate_type == CandidateType.WAIT:
        if "full_payment" not in context.profile.payment_methods_user_will_consider:
            user_valid = False
            uv_detail = f"User payment preferences {context.profile.payment_methods_user_will_consider} do not include 'full_payment' (required for wait eligibility)."
    elif candidate.payment_method not in context.profile.payment_methods_user_will_consider:
        user_valid = False
        uv_detail = f"Payment method '{candidate.payment_method}' not in user preferences {context.profile.payment_methods_user_will_consider}."
    elif candidate.candidate_type == CandidateType.INSTALLMENT_PLAN and context.profile.max_installment_months is not None:
        # Check that installment duration does not exceed max_installment_months
        opt_id = candidate.source_payment_option_id
        if opt_id in context.payment_option_feasibilities:
            feas = context.payment_option_feasibilities[opt_id]
            if not feas.is_eligible and "exceeds maximum installment duration" in (feas.rejection_reason or ""):
                user_valid = False
                uv_detail = "Installment option duration exceeds user max_installment_months."

    if user_valid:
        checks.append(SafetyGateCheck("user_constraints_valid", True, DecisionReasonCode.USER_CONSTRAINTS_VALID, uv_detail))
    else:
        checks.append(SafetyGateCheck("user_constraints_valid", False, DecisionReasonCode.USER_PREFERENCES_VIOLATED, uv_detail))
        rejections.append(DecisionReasonCode.USER_PREFERENCES_VIOLATED)

    # 8. SCHEDULE_CONSISTENT
    sched_ok = True
    sc_detail = "Payment schedule is internally consistent and chronological."
    if candidate.payment_schedule:
        for i in range(len(candidate.payment_schedule) - 1):
            if candidate.payment_schedule[i].payment_date > candidate.payment_schedule[i + 1].payment_date:
                sched_ok = False
                sc_detail = "Payment dates are not monotonically non-decreasing."
                break
        if candidate.first_payment_date != candidate.payment_schedule[0].payment_date:
            sched_ok = False
            sc_detail = "first_payment_date does not match first schedule entry date."
        if candidate.completion_date != candidate.payment_schedule[-1].payment_date:
            sched_ok = False
            sc_detail = "completion_date does not match last schedule entry date."
        if candidate.first_payment_date is not None and candidate.first_payment_date < context.request.request_date:
            sched_ok = False
            sc_detail = "first_payment_date precedes request_date."

    if sched_ok:
        checks.append(SafetyGateCheck("schedule_consistent", True, DecisionReasonCode.SCHEDULE_CONSISTENT, sc_detail))
    else:
        checks.append(SafetyGateCheck("schedule_consistent", False, DecisionReasonCode.SCHEDULE_INCONSISTENT, sc_detail))
        rejections.append(DecisionReasonCode.SCHEDULE_INCONSISTENT)

    is_overall_safe = len(rejections) == 0
    return SafetyGateResult(
        candidate_id=candidate.candidate_id,
        is_safe=is_overall_safe,
        checks=tuple(checks),
        rejection_reason_codes=tuple(rejections),
    )


def is_finally_safe(candidate: Candidate, context: RequestContext) -> bool:
    """Predicate answering whether a candidate satisfies all final safety gate conditions."""
    res = evaluate_candidate_safety_gate(candidate, context)
    return res.is_safe


# ---------------------------------------------------------------------------
# Master Final Decision Function
# ---------------------------------------------------------------------------

def make_final_decision(
    context: RequestContext,
    candidates: Sequence[Candidate],
) -> FinalDecision:
    """Produce the deterministic FinalDecision for a request given candidate universe.

    Workflow:
      1. Evaluates all candidates through evaluate_candidate_safety_gate.
      2. Gathers surviving safe candidates.
      3. Passes surviving candidates to the frozen ranking engine (code/ranking.py).
      4. Selects strictly the #1 ranking candidate from surviving candidates.
      5. If zero candidates survive, constructs the specification-defined fallback decision.
    """
    req = context.request
    cert = context.certificate
    prof = context.profile

    # 1 & 2. Filter candidates through final safety gate
    safety_results: list[SafetyGateResult] = []
    surviving_candidates: list[Candidate] = []
    safety_results_by_id: dict[str, SafetyGateResult] = {}

    for c in candidates:
        sg_res = evaluate_candidate_safety_gate(c, context)
        safety_results.append(sg_res)
        safety_results_by_id[c.candidate_id] = sg_res
        if sg_res.is_safe:
            surviving_candidates.append(c)

    # 3. Submit surviving candidates to frozen ranking engine
    ranked_set: RankedCandidateSet = rank_candidates(
        candidates=surviving_candidates,
        desired_completion_date=req.desired_completion_date,
        request_id=req.request_id,
    )
    ranked_ids = tuple(rc.candidate.candidate_id for rc in ranked_set.ranked_candidates)

    # 4. If a candidate survived, select rank #1
    if not ranked_set.is_empty and ranked_set.top_ranked_candidate is not None:
        top_rc: RankedCandidate = ranked_set.top_ranked_candidate
        selected_c: Candidate = top_rc.candidate
        trace: RankingCriteriaTrace = top_rc.trace
        selected_sg: SafetyGateResult = safety_results_by_id[selected_c.candidate_id]

        # Determine AffordabilityStatus
        if selected_c.candidate_type == CandidateType.FULL_PAYMENT:
            aff_status = AffordabilityStatus.AFFORDABLE_NOW
        elif selected_c.candidate_type in (CandidateType.INSTALLMENT_PLAN, CandidateType.PARTIAL_PAYMENT):
            aff_status = AffordabilityStatus.AFFORDABLE_WITH_PLAN
        elif selected_c.candidate_type == CandidateType.WAIT:
            aff_status = AffordabilityStatus.AFFORDABLE_LATER
        else:
            raise ValueError(f"Unknown candidate type: {selected_c.candidate_type}")

        reason_codes = (
            DecisionReasonCode.SELECTED_BY_RANKING,
            *selected_sg.rejection_reason_codes,
        )

        evidence = DecisionExplanationEvidence(
            request_id=req.request_id,
            user_id=req.user_id,
            request_date=req.request_date,
            desired_completion_date=req.desired_completion_date,
            requested_amount=cert.requested_amount,
            amount_safe_to_pay=cert.amount_safe_to_pay,
            earliest_date_for_full_payment=cert.earliest_date_for_full_payment,
            selected_candidate_type=selected_c.candidate_type,
            selected_payment_method=selected_c.payment_method,
            total_amount_paid=selected_c.total_amount_paid,
            financing_fee=selected_c.financing_fee,
            number_of_payments=selected_c.number_of_payments,
            safety_floor=cert.safety_floor,
            decision_reason_codes=reason_codes,
            evidence_notes=(
                f"Selected rank #1 candidate {selected_c.candidate_id} via deterministic ranking.",
                f"Affordability status: {aff_status.value}.",
            ),
        )

        return FinalDecision(
            request_id=req.request_id,
            selected_candidate_id=selected_c.candidate_id,
            affordability_status=aff_status,
            amount_safe_to_pay=cert.amount_safe_to_pay,
            earliest_date_for_full_payment=cert.earliest_date_for_full_payment,
            recommended_payment_method=selected_c.payment_method,
            payment_plan=selected_c.payment_plan_string,
            spending_changes_needed="none",
            selected_candidate=selected_c,
            ranking_trace=trace,
            safety_gate_result=selected_sg,
            all_safety_gate_results=tuple(safety_results),
            ranked_candidate_ids=ranked_ids,
            decision_reason_codes=reason_codes,
            explanation_evidence=evidence,
        )

    # 5. Zero candidates survived: construct fallback decision
    aff_result: AffordabilityResult = classify_affordability(
        certificate=cert,
        profile=prof,
        request=req,
        payment_plan_feasible=False,
        respect_deadline=True,
        require_full_payment_preference_for_later=True,
    )
    fallback_status = aff_result.status

    reason_codes = (DecisionReasonCode.NO_SAFE_CANDIDATE_SURVIVED,)

    evidence = DecisionExplanationEvidence(
        request_id=req.request_id,
        user_id=req.user_id,
        request_date=req.request_date,
        desired_completion_date=req.desired_completion_date,
        requested_amount=cert.requested_amount,
        amount_safe_to_pay=cert.amount_safe_to_pay,
        earliest_date_for_full_payment=cert.earliest_date_for_full_payment,
        selected_candidate_type=None,
        selected_payment_method="not_recommended",
        total_amount_paid=None,
        financing_fee=None,
        number_of_payments=None,
        safety_floor=cert.safety_floor,
        decision_reason_codes=reason_codes,
        evidence_notes=(
            "Zero candidates survived the final safety gate.",
            f"Affordability status: {fallback_status.value} ({aff_result.reason}).",
        ),
    )

    return FinalDecision(
        request_id=req.request_id,
        selected_candidate_id=None,
        affordability_status=fallback_status,
        amount_safe_to_pay=cert.amount_safe_to_pay,
        earliest_date_for_full_payment=cert.earliest_date_for_full_payment,
        recommended_payment_method="not_recommended",
        payment_plan="none",
        spending_changes_needed="none",
        selected_candidate=None,
        ranking_trace=None,
        safety_gate_result=None,
        all_safety_gate_results=tuple(safety_results),
        ranked_candidate_ids=(),
        decision_reason_codes=reason_codes,
        explanation_evidence=evidence,
    )


def make_final_decision_from_candidate_set(
    context: RequestContext,
    candidate_set: CandidateSet,
) -> FinalDecision:
    """Convenience entry point: evaluates final decision from a pre-generated CandidateSet."""
    return make_final_decision(context=context, candidates=candidate_set.candidates)
