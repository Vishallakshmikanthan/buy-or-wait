"""Deterministic Decision Certificate / Evidence Package for Buy or Wait?

Prompt 13:
This module transforms an already-computed FinalDecision into a compact,
immutable, deterministic evidence package for downstream auditability and
explanation models.

Core Principles:
    DECISION ENGINE DECIDES.
    CERTIFICATE PROVES.
    LLM LATER EXPLAINS.

WHAT THIS MODULE DOES:
- Collects, normalizes, validates, and exposes authoritative evidence already
  produced by upstream modules.
- Tracks granular provenance / lineage (EvidenceRef) for every material fact.
- Supports both selected-candidate certificates and zero-candidate certificates.
- Exposes structured competing candidate evidence explaining why lower-ranked
  candidates lost.
- Serializes deterministically to canonical dict / JSON (SHA-256 integrity fingerprint).
- Validates certificate structural integrity and consistency with FinalDecision.

WHAT THIS MODULE DOES NOT DO:
- Does NOT make a new financial decision.
- Does NOT recompute affordability.
- Does NOT recompute balances.
- Does NOT re-run candidate ranking to choose a different candidate.
- Does NOT generate free-form prose, paragraphs, or LLM text.
- Does NOT produce output.csv or submission packaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from code.affordability import AffordabilityStatus
from code.candidate_generation import (
    Candidate,
    CandidatePayment,
    CandidateSet,
    CandidateStatus,
    CandidateType,
)
from code.final_decision import (
    DecisionReasonCode,
    FinalDecision,
    RequestContext,
    SafetyGateCheck,
    SafetyGateResult,
)
from code.payment_plan import PaymentPlanFeasibility
from code.ranking import RankingCriteriaTrace
from code.safe_to_pay import SafeToPayCertificate
from code.user_state import (
    ExpenseStability,
    IncomeStability,
    RecurringStreamSummary,
    SpendingTrend,
    UpcomingObligationSummary,
    UserFinancialState,
)

CERTIFICATE_VERSION: str = "1.0.0"

OFFICIAL_AFFORDABILITY_STATUSES: frozenset[str] = frozenset({
    AffordabilityStatus.AFFORDABLE_NOW.value,
    AffordabilityStatus.AFFORDABLE_WITH_PLAN.value,
    AffordabilityStatus.AFFORDABLE_LATER.value,
    AffordabilityStatus.NOT_AFFORDABLE.value,
})

VALID_PAYMENT_METHODS: frozenset[str] = frozenset({
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
})


# ---------------------------------------------------------------------------
# Evidence Lineage Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceRef:
    """Structured lineage reference for a material fact in the certificate.

    Attributes:
        source_module: Upstream module name (e.g. 'final_decision', 'safe_to_pay').
        source_object: Class/type name (e.g. 'FinalDecision', 'SafeToPayCertificate').
        source_id: Authoritative identifier (e.g. request_id, user_id, candidate_id).
        field_name: Name of the fact or attribute referenced.
    """
    source_module: str
    source_object: str
    source_id: str
    field_name: str

    def to_canonical_dict(self) -> Dict[str, str]:
        return {
            "field_name": self.field_name,
            "source_id": self.source_id,
            "source_module": self.source_module,
            "source_object": self.source_object,
        }


# ---------------------------------------------------------------------------
# Structured Ranking & Competing Candidate Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankingTraceEvidence:
    """Structured ranking trace for the top-ranked candidate across all 6 criteria."""
    candidate_id: str
    deadline_met: bool
    no_spending_changes: bool
    total_amount_paid: Decimal
    first_payment_date: Optional[date]
    number_of_payments: int
    criterion_6_payment_option_id: Optional[str] = None
    technical_tiebreaker_payment_option_id: Optional[str] = None
    technical_tie_breaker: Optional[str] = None

    @property
    def criterion_1_deadline_met(self) -> bool:
        return self.deadline_met

    @property
    def criterion_2_no_spending_changes(self) -> bool:
        return self.no_spending_changes

    @property
    def criterion_3_total_amount_paid(self) -> Decimal:
        return self.total_amount_paid

    @property
    def criterion_4_first_payment_date(self) -> Optional[date]:
        return self.first_payment_date

    @property
    def criterion_5_number_of_payments(self) -> int:
        return self.number_of_payments

    def to_canonical_dict(self) -> Dict[str, Any]:
        opt_id = self.criterion_6_payment_option_id or self.technical_tiebreaker_payment_option_id
        return {
            "candidate_id": self.candidate_id,
            "criterion_1_deadline_met": self.deadline_met,
            "criterion_2_no_spending_changes": self.no_spending_changes,
            "criterion_3_total_amount_paid": str(self.total_amount_paid),
            "criterion_4_first_payment_date": self.first_payment_date.isoformat() if self.first_payment_date else None,
            "criterion_5_number_of_payments": self.number_of_payments,
            "criterion_6_payment_option_id": opt_id,
            "deadline_met": self.deadline_met,
            "first_payment_date": self.first_payment_date.isoformat() if self.first_payment_date else None,
            "no_spending_changes": self.no_spending_changes,
            "number_of_payments": self.number_of_payments,
            "payment_option_id": opt_id,
            "technical_tie_breaker": self.technical_tie_breaker,
            "technical_tiebreaker_payment_option_id": opt_id,
            "total_amount_paid": str(self.total_amount_paid),
        }


@dataclass(frozen=True)
class CompetingCandidateEvidence:
    """Structured evidence for competing candidates in ranking order across all 6 criteria."""
    candidate_id: str
    rank: int
    payment_method: str
    deadline_met: bool
    spending_changes_required: bool
    total_amount_paid: Decimal
    start_date: Optional[date]
    payment_count: int
    payment_option_id: Optional[str]
    is_safe: bool
    criterion_1_deadline_met: bool = True
    criterion_2_no_spending_changes: bool = True
    criterion_3_total_amount_paid: Optional[Decimal] = None
    criterion_4_first_payment_date: Optional[date] = None
    criterion_5_number_of_payments: Optional[int] = None
    criterion_6_payment_option_id: Optional[str] = None
    decisive_criterion: Optional[str] = None
    is_winner: bool = False

    def to_canonical_dict(self) -> Dict[str, Any]:
        opt_id = self.criterion_6_payment_option_id if self.criterion_6_payment_option_id is not None else self.payment_option_id
        t_amt = self.criterion_3_total_amount_paid if self.criterion_3_total_amount_paid is not None else self.total_amount_paid
        f_date = self.criterion_4_first_payment_date if self.criterion_4_first_payment_date is not None else self.start_date
        n_pay = self.criterion_5_number_of_payments if self.criterion_5_number_of_payments is not None else self.payment_count
        return {
            "candidate_id": self.candidate_id,
            "criterion_1_deadline_met": self.criterion_1_deadline_met,
            "criterion_2_no_spending_changes": self.criterion_2_no_spending_changes,
            "criterion_3_total_amount_paid": str(t_amt),
            "criterion_4_first_payment_date": f_date.isoformat() if f_date else None,
            "criterion_5_number_of_payments": n_pay,
            "criterion_6_payment_option_id": opt_id,
            "deadline_met": self.deadline_met,
            "decisive_criterion": self.decisive_criterion,
            "is_safe": self.is_safe,
            "is_winner": self.is_winner,
            "payment_count": self.payment_count,
            "payment_method": self.payment_method,
            "payment_option_id": self.payment_option_id,
            "rank": self.rank,
            "spending_changes_required": self.spending_changes_required,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "technical_tiebreaker_payment_option_id": opt_id,
            "total_amount_paid": str(self.total_amount_paid),
        }


# ---------------------------------------------------------------------------
# Structured Safety Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyCheckEvidence:
    """Evidence for a single safety gate condition check."""
    check_name: str
    passed: bool
    reason_code: str
    details: str

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "details": self.details,
            "passed": self.passed,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class RejectedCandidateEvidence:
    """Evidence record for candidates rejected by safety or eligibility gates."""
    candidate_id: str
    payment_method: str
    rejection_reason_codes: Tuple[str, ...]
    gate_result: str = "FAIL"

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "gate_result": self.gate_result,
            "payment_method": self.payment_method,
            "rejection_reason_codes": list(self.rejection_reason_codes),
        }


# ---------------------------------------------------------------------------
# Structured Financial Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinancialEvidence:
    """Authoritative already-computed financial evidence metrics."""
    safety_floor: Decimal
    requested_amount: Decimal
    desired_completion_date: date
    baseline_minimum_available_cash: Optional[Decimal] = None
    post_action_minimum_available_cash: Optional[Decimal] = None
    limiting_date: Optional[date] = None
    financing_fee: Optional[Decimal] = None
    total_amount_paid: Optional[Decimal] = None
    completion_date: Optional[date] = None
    payment_count: Optional[int] = None

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "baseline_minimum_available_cash": (
                str(self.baseline_minimum_available_cash)
                if self.baseline_minimum_available_cash is not None else None
            ),
            "completion_date": self.completion_date.isoformat() if self.completion_date else None,
            "desired_completion_date": self.desired_completion_date.isoformat(),
            "financing_fee": str(self.financing_fee) if self.financing_fee is not None else None,
            "limiting_date": self.limiting_date.isoformat() if self.limiting_date else None,
            "payment_count": self.payment_count,
            "post_action_minimum_available_cash": (
                str(self.post_action_minimum_available_cash)
                if self.post_action_minimum_available_cash is not None else None
            ),
            "requested_amount": str(self.requested_amount),
            "safety_floor": str(self.safety_floor),
            "total_amount_paid": str(self.total_amount_paid) if self.total_amount_paid is not None else None,
        }


# ---------------------------------------------------------------------------
# Structured State & Obligation Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecurringStreamEvidence:
    """Authoritative summary of a verified recurring cashflow stream."""
    series_id: str
    category: str
    direction: str
    frequency: str
    amount: Decimal
    monthly_equivalent: Decimal

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "amount": str(self.amount),
            "category": self.category,
            "direction": self.direction,
            "frequency": self.frequency,
            "monthly_equivalent": str(self.monthly_equivalent),
            "series_id": self.series_id,
        }


@dataclass(frozen=True)
class ScheduledObligationEvidence:
    """Authoritative summary of an upcoming obligation relevant to request horizon."""
    event_id: str
    effective_date: date
    category: str
    amount: Decimal
    source_type: str

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "amount": str(self.amount),
            "category": self.category,
            "effective_date": self.effective_date.isoformat(),
            "event_id": self.event_id,
            "source_type": self.source_type,
        }


@dataclass(frozen=True)
class StateEvidence:
    """Authoritative user state and obligations evidence."""
    recurring_obligations: Tuple[RecurringStreamEvidence, ...] = ()
    scheduled_obligations: Tuple[ScheduledObligationEvidence, ...] = ()
    spending_trend: Optional[str] = None
    income_stability: Optional[str] = None
    expense_stability: Optional[str] = None

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "expense_stability": self.expense_stability,
            "income_stability": self.income_stability,
            "recurring_obligations": [r.to_canonical_dict() for r in self.recurring_obligations],
            "scheduled_obligations": [s.to_canonical_dict() for s in self.scheduled_obligations],
            "spending_trend": self.spending_trend,
        }


# ---------------------------------------------------------------------------
# Decision Certificate (Master Immutable Evidence Package)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionCertificate:
    """Compact, immutable, deterministic evidence package for a FinalDecision.

    Contains only authoritative facts already computed and validated upstream.
    No free-form prose, paragraphs, or natural language explanations.
    """
    # IDENTITY
    request_id: str
    user_id: str
    certificate_version: str

    # DECISION
    selected_candidate_id: Optional[str]
    affordability_status: str
    recommended_payment_method: str
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: Optional[date]
    payment_plan: str
    spending_changes_needed: str

    # RANKING
    selected_rank: Optional[int]
    total_rankable_candidates: int
    ordered_candidate_ids: Tuple[str, ...]
    ranking_criteria_trace: Optional[RankingTraceEvidence]
    competing_candidates: Tuple[CompetingCandidateEvidence, ...]

    # SAFETY
    final_safety_gate_passed: bool
    selected_candidate_safety_checks: Tuple[SafetyCheckEvidence, ...]
    selected_candidate_reason_codes: Tuple[str, ...]

    # REJECTED CANDIDATES
    rejected_candidates: Tuple[RejectedCandidateEvidence, ...]

    # FINANCIAL EVIDENCE
    financial_evidence: FinancialEvidence

    # UPCOMING / STATE EVIDENCE
    state_evidence: StateEvidence

    # LINEAGE
    evidence_lineage: Tuple[EvidenceRef, ...]

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Convert certificate to a deterministic dictionary with canonical primitives."""
        return {
            "affordability_status": self.affordability_status,
            "amount_safe_to_pay": str(self.amount_safe_to_pay),
            "certificate_version": self.certificate_version,
            "competing_candidates": [c.to_canonical_dict() for c in self.competing_candidates],
            "earliest_date_for_full_payment": (
                self.earliest_date_for_full_payment.isoformat()
                if self.earliest_date_for_full_payment else None
            ),
            "evidence_lineage": [e.to_canonical_dict() for e in self.evidence_lineage],
            "final_safety_gate_passed": self.final_safety_gate_passed,
            "financial_evidence": self.financial_evidence.to_canonical_dict(),
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "payment_plan": self.payment_plan,
            "ranking_criteria_trace": (
                self.ranking_criteria_trace.to_canonical_dict()
                if self.ranking_criteria_trace else None
            ),
            "recommended_payment_method": self.recommended_payment_method,
            "rejected_candidates": [r.to_canonical_dict() for r in self.rejected_candidates],
            "request_id": self.request_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_reason_codes": list(self.selected_candidate_reason_codes),
            "selected_candidate_safety_checks": [
                s.to_canonical_dict() for s in self.selected_candidate_safety_checks
            ],
            "selected_rank": self.selected_rank,
            "spending_changes_needed": self.spending_changes_needed,
            "state_evidence": self.state_evidence.to_canonical_dict(),
            "total_rankable_candidates": self.total_rankable_candidates,
            "user_id": self.user_id,
        }

    def to_canonical_json(self) -> str:
        """Serialize certificate to canonical byte-identical JSON string."""
        d = self.to_canonical_dict()
        return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def certificate_hash(self) -> str:
        """Compute deterministic SHA-256 integrity fingerprint over canonical JSON bytes."""
        return certificate_hash(self)


def certificate_hash(certificate: DecisionCertificate) -> str:
    """Compute deterministic SHA-256 integrity fingerprint over canonical JSON bytes.

    This is an integrity fingerprint only, not a cryptographic security signature.
    """
    canonical_bytes = certificate.to_canonical_json().encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Certificate Builder
# ---------------------------------------------------------------------------

def build_decision_certificate(
    decision: FinalDecision,
    context: Optional[RequestContext] = None,
    candidate_set: Optional[Union[CandidateSet, Sequence[Candidate]]] = None,
    user_state: Optional[UserFinancialState] = None,
) -> DecisionCertificate:
    """Transform an already-computed FinalDecision into a DecisionCertificate.

    Extracts all facts from authoritative upstream sources without recomputing.
    """
    req_id = decision.request_id
    user_id = decision.explanation_evidence.user_id
    selected_c = decision.selected_candidate
    selected_c_id = decision.selected_candidate_id

    # 1. Ranking Trace
    ranking_trace_evidence: Optional[RankingTraceEvidence] = None
    if decision.ranking_trace is not None and selected_c_id is not None:
        t = decision.ranking_trace
        ranking_trace_evidence = RankingTraceEvidence(
            candidate_id=selected_c_id,
            deadline_met=t.criterion_1_deadline_met,
            no_spending_changes=t.criterion_2_no_spending_changes,
            total_amount_paid=t.criterion_3_total_amount_paid,
            first_payment_date=t.criterion_4_first_payment_date,
            number_of_payments=t.criterion_5_number_of_payments,
            criterion_6_payment_option_id=t.criterion_6_payment_option_id,
            technical_tiebreaker_payment_option_id=t.criterion_6_payment_option_id,
            technical_tie_breaker=t.technical_tie_breaker,
        )

    # 2. Competing Candidates Evidence
    # Gather candidate pool
    candidates_pool: List[Candidate] = []
    if candidate_set is not None:
        if isinstance(candidate_set, CandidateSet):
            candidates_pool = list(candidate_set.candidates)
        else:
            candidates_pool = list(candidate_set)
    elif selected_c is not None:
        candidates_pool = [selected_c]

    c_by_id: Dict[str, Candidate] = {c.candidate_id: c for c in candidates_pool}
    top_cand: Optional[Candidate] = None
    if decision.ranked_candidate_ids:
        top_cand = c_by_id.get(decision.ranked_candidate_ids[0]) or selected_c
    elif selected_c is not None:
        top_cand = selected_c

    desired_d = decision.explanation_evidence.desired_completion_date

    competing_candidates: List[CompetingCandidateEvidence] = []
    for rank_idx, c_id in enumerate(decision.ranked_candidate_ids, start=1):
        c = c_by_id.get(c_id) or (selected_c if c_id == selected_c_id else None)
        if c is not None:
            d_met = (
                (c.completion_date <= desired_d)
                if c.completion_date else False
            )
            no_sc = (len(c.spending_changes) == 0)
            is_winner = (rank_idx == 1)
            decisive_crit: Optional[str] = None
            if not is_winner and top_cand is not None:
                top_d_met = (
                    (top_cand.completion_date <= desired_d)
                    if top_cand.completion_date else False
                )
                top_no_sc = (len(top_cand.spending_changes) == 0)
                if top_d_met != d_met:
                    decisive_crit = "CRITERION_1_DEADLINE_MET"
                elif top_no_sc != no_sc:
                    decisive_crit = "CRITERION_2_NO_SPENDING_CHANGES"
                elif top_cand.total_amount_paid != c.total_amount_paid:
                    decisive_crit = "CRITERION_3_TOTAL_AMOUNT_PAID"
                elif top_cand.first_payment_date != c.first_payment_date:
                    decisive_crit = "CRITERION_4_FIRST_PAYMENT_DATE"
                elif top_cand.number_of_payments != c.number_of_payments:
                    decisive_crit = "CRITERION_5_NUMBER_OF_PAYMENTS"
                elif (
                    top_cand.source_payment_option_id is not None
                    and c.source_payment_option_id is not None
                    and top_cand.source_payment_option_id != c.source_payment_option_id
                ):
                    decisive_crit = "CRITERION_6_PAYMENT_OPTION_ID"
                elif top_cand.candidate_id != c.candidate_id:
                    decisive_crit = "TECHNICAL_TIE_BREAKER_CANDIDATE_ID"

            competing_candidates.append(CompetingCandidateEvidence(
                candidate_id=c.candidate_id,
                rank=rank_idx,
                payment_method=c.payment_method,
                deadline_met=d_met,
                spending_changes_required=not no_sc,
                total_amount_paid=c.total_amount_paid,
                start_date=c.first_payment_date,
                payment_count=c.number_of_payments,
                payment_option_id=c.source_payment_option_id,
                is_safe=c.is_safe,
                criterion_1_deadline_met=d_met,
                criterion_2_no_spending_changes=no_sc,
                criterion_3_total_amount_paid=c.total_amount_paid,
                criterion_4_first_payment_date=c.first_payment_date,
                criterion_5_number_of_payments=c.number_of_payments,
                criterion_6_payment_option_id=c.source_payment_option_id,
                decisive_criterion=decisive_crit,
                is_winner=is_winner,
            ))

    # 3. Safety Checks & Reason Codes
    selected_safety_checks: List[SafetyCheckEvidence] = []
    selected_reason_codes: List[str] = []
    final_safety_gate_passed = False

    if decision.safety_gate_result is not None:
        final_safety_gate_passed = decision.safety_gate_result.is_safe
        for chk in decision.safety_gate_result.checks:
            selected_safety_checks.append(SafetyCheckEvidence(
                check_name=chk.check_name,
                passed=chk.passed,
                reason_code=chk.reason_code.value,
                details=chk.details,
            ))
        selected_reason_codes = sorted(r.value for r in decision.safety_gate_result.rejection_reason_codes)

    # 4. Rejected Candidates
    rejected_candidates: List[RejectedCandidateEvidence] = []
    for sg in decision.all_safety_gate_results:
        if not sg.is_safe:
            c_cand = c_by_id.get(sg.candidate_id)
            if c_cand:
                method = c_cand.payment_method
            elif "__" in sg.candidate_id:
                parts = sg.candidate_id.split("__")
                method = parts[1] if len(parts) > 1 else "unknown"
            else:
                method = "unknown"

            rejected_candidates.append(RejectedCandidateEvidence(
                candidate_id=sg.candidate_id,
                payment_method=method,
                rejection_reason_codes=tuple(sorted(r.value for r in sg.rejection_reason_codes)),
                gate_result="FAIL",
            ))

    rejected_candidates.sort(key=lambda r: r.candidate_id)

    # 5. Financial Evidence
    exp_ev = decision.explanation_evidence
    base_min_cash: Optional[Decimal] = None
    post_action_min_cash: Optional[Decimal] = None
    limiting_d: Optional[date] = None

    if context is not None:
        base_min_cash = context.certificate.baseline_minimum_available_cash
        limiting_d = context.certificate.limiting_date

        if selected_c is not None:
            if selected_c.candidate_type == CandidateType.FULL_PAYMENT:
                post_action_min_cash = context.certificate.minimum_available_cash_after_purchase
            elif selected_c.candidate_type == CandidateType.INSTALLMENT_PLAN:
                opt_id = selected_c.source_payment_option_id
                if opt_id and opt_id in context.payment_option_feasibilities:
                    feas = context.payment_option_feasibilities[opt_id]
                    if feas.simulator_certificate is not None:
                        post_action_min_cash = feas.simulator_certificate.minimum_available_cash
                    elif feas.minimum_available_cash is not None:
                        post_action_min_cash = feas.minimum_available_cash
            elif selected_c.candidate_type == CandidateType.WAIT:
                post_action_min_cash = context.certificate.baseline_minimum_available_cash

    fin_evidence = FinancialEvidence(
        safety_floor=exp_ev.safety_floor,
        requested_amount=exp_ev.requested_amount,
        desired_completion_date=exp_ev.desired_completion_date,
        baseline_minimum_available_cash=base_min_cash,
        post_action_minimum_available_cash=post_action_min_cash,
        limiting_date=limiting_d,
        financing_fee=exp_ev.financing_fee,
        total_amount_paid=exp_ev.total_amount_paid,
        completion_date=selected_c.completion_date if selected_c else None,
        payment_count=exp_ev.number_of_payments,
    )

    # 6. State & Upcoming Obligations Evidence
    rec_streams_ev: List[RecurringStreamEvidence] = []
    sched_obs_ev: List[ScheduledObligationEvidence] = []
    spending_trend_val: Optional[str] = None
    income_stability_val: Optional[str] = None
    expense_stability_val: Optional[str] = None

    if user_state is not None:
        spending_trend_val = user_state.spending_trend.value
        income_stability_val = user_state.income_stability.value
        expense_stability_val = user_state.expense_stability.value

        # Summarize verified recurring streams
        for rs in user_state.recurring_income_streams:
            rec_streams_ev.append(RecurringStreamEvidence(
                series_id=rs.series_id,
                category=rs.category,
                direction=rs.direction.value,
                frequency=rs.frequency.value,
                amount=rs.amount,
                monthly_equivalent=rs.monthly_equivalent,
            ))
        for rs in user_state.recurring_expense_streams:
            rec_streams_ev.append(RecurringStreamEvidence(
                series_id=rs.series_id,
                category=rs.category,
                direction=rs.direction.value,
                frequency=rs.frequency.value,
                amount=rs.amount,
                monthly_equivalent=rs.monthly_equivalent,
            ))
        rec_streams_ev.sort(key=lambda r: r.series_id)

        # Scheduled obligations relevant to selected candidate horizon
        if selected_c is not None:
            h_start = selected_c.first_payment_date or exp_ev.request_date
            h_end = selected_c.completion_date or (h_start + timedelta(days=30))
        else:
            h_start = exp_ev.request_date
            h_end = h_start + timedelta(days=30)

        rel_obs = [
            ob for ob in user_state.upcoming_obligations
            if h_start <= ob.effective_date <= h_end
        ][:10]

        for ob in rel_obs:
            sched_obs_ev.append(ScheduledObligationEvidence(
                event_id=ob.event_id,
                effective_date=ob.effective_date,
                category=ob.category,
                amount=ob.amount,
                source_type=ob.source_type,
            ))
        sched_obs_ev.sort(key=lambda s: (s.effective_date, s.event_id))

    st_evidence = StateEvidence(
        recurring_obligations=tuple(rec_streams_ev),
        scheduled_obligations=tuple(sched_obs_ev),
        spending_trend=spending_trend_val,
        income_stability=income_stability_val,
        expense_stability=expense_stability_val,
    )

    # 7. Lineage Records (EvidenceRef)
    lineage_records: List[EvidenceRef] = [
        EvidenceRef("final_decision", "FinalDecision", req_id, "request_id"),
        EvidenceRef("final_decision", "FinalDecision", req_id, "user_id"),
        EvidenceRef("final_decision", "FinalDecision", req_id, "selected_candidate_id"),
        EvidenceRef("final_decision", "FinalDecision", req_id, "affordability_status"),
        EvidenceRef("final_decision", "FinalDecision", req_id, "recommended_payment_method"),
        EvidenceRef("safe_to_pay", "SafeToPayCertificate", req_id, "amount_safe_to_pay"),
        EvidenceRef("safe_to_pay", "SafeToPayCertificate", req_id, "earliest_date_for_full_payment"),
        EvidenceRef("final_decision", "FinalDecision", req_id, "payment_plan"),
        EvidenceRef("final_decision", "FinalDecision", req_id, "spending_changes_needed"),
        EvidenceRef("safe_to_pay", "SafeToPayCertificate", req_id, "safety_floor"),
        EvidenceRef("safe_to_pay", "SafeToPayCertificate", req_id, "requested_amount"),
        EvidenceRef("final_decision", "FinalDecision", req_id, "desired_completion_date"),
    ]

    if selected_c is not None:
        c_src_obj = "Candidate"
        lineage_records.extend([
            EvidenceRef("candidate_generation", c_src_obj, selected_c.candidate_id, "total_amount_paid"),
            EvidenceRef("candidate_generation", c_src_obj, selected_c.candidate_id, "financing_fee"),
            EvidenceRef("candidate_generation", c_src_obj, selected_c.candidate_id, "number_of_payments"),
            EvidenceRef("candidate_generation", c_src_obj, selected_c.candidate_id, "completion_date"),
        ])

    if decision.ranking_trace is not None:
        lineage_records.extend([
            EvidenceRef("ranking", "RankingCriteriaTrace", req_id, "ranking_criteria_trace"),
            EvidenceRef("ranking", "RankingCriteriaTrace", req_id, "criterion_1_deadline_met"),
            EvidenceRef("ranking", "RankingCriteriaTrace", req_id, "criterion_2_no_spending_changes"),
            EvidenceRef("ranking", "RankingCriteriaTrace", req_id, "criterion_3_total_amount_paid"),
            EvidenceRef("ranking", "RankingCriteriaTrace", req_id, "criterion_4_first_payment_date"),
            EvidenceRef("ranking", "RankingCriteriaTrace", req_id, "criterion_5_number_of_payments"),
            EvidenceRef("ranking", "RankingCriteriaTrace", req_id, "criterion_6_payment_option_id"),
            EvidenceRef("ranking", "RankingCriteriaTrace", req_id, "technical_tie_breaker"),
        ])

    if base_min_cash is not None:
        lineage_records.append(
            EvidenceRef("safe_to_pay", "SafeToPayCertificate", req_id, "baseline_minimum_available_cash")
        )
    if limiting_d is not None:
        lineage_records.append(
            EvidenceRef("safe_to_pay", "SafeToPayCertificate", req_id, "limiting_date")
        )
    if post_action_min_cash is not None:
        if selected_c and selected_c.candidate_type == CandidateType.INSTALLMENT_PLAN and selected_c.source_payment_option_id:
            lineage_records.append(
                EvidenceRef("payment_plan", "PaymentPlanFeasibility", selected_c.source_payment_option_id, "minimum_available_cash")
            )
        else:
            lineage_records.append(
                EvidenceRef("safe_to_pay", "SafeToPayCertificate", req_id, "minimum_available_cash_after_purchase")
            )

    if user_state is not None:
        lineage_records.extend([
            EvidenceRef("user_state", "UserFinancialState", user_id, "spending_trend"),
            EvidenceRef("user_state", "UserFinancialState", user_id, "income_stability"),
            EvidenceRef("user_state", "UserFinancialState", user_id, "expense_stability"),
        ])

    lineage_records.sort(key=lambda e: (e.field_name, e.source_module, e.source_object, e.source_id))

    # Selected rank
    selected_rank: Optional[int] = None
    if selected_c_id is not None:
        if selected_c_id in decision.ranked_candidate_ids:
            selected_rank = decision.ranked_candidate_ids.index(selected_c_id) + 1
        else:
            selected_rank = 1

    return DecisionCertificate(
        request_id=req_id,
        user_id=user_id,
        certificate_version=CERTIFICATE_VERSION,
        selected_candidate_id=selected_c_id,
        affordability_status=decision.affordability_status.value,
        recommended_payment_method=decision.recommended_payment_method,
        amount_safe_to_pay=decision.amount_safe_to_pay,
        earliest_date_for_full_payment=decision.earliest_date_for_full_payment,
        payment_plan=decision.payment_plan,
        spending_changes_needed=decision.spending_changes_needed,
        selected_rank=selected_rank,
        total_rankable_candidates=len(decision.ranked_candidate_ids),
        ordered_candidate_ids=decision.ranked_candidate_ids,
        ranking_criteria_trace=ranking_trace_evidence,
        competing_candidates=tuple(competing_candidates),
        final_safety_gate_passed=final_safety_gate_passed,
        selected_candidate_safety_checks=tuple(selected_safety_checks),
        selected_candidate_reason_codes=tuple(selected_reason_codes),
        rejected_candidates=tuple(rejected_candidates),
        financial_evidence=fin_evidence,
        state_evidence=st_evidence,
        evidence_lineage=tuple(lineage_records),
    )


# ---------------------------------------------------------------------------
# Certificate Validation
# ---------------------------------------------------------------------------

class CertificateValidationError(ValueError):
    """Raised when certificate validation fails in strict mode."""
    pass


@dataclass(frozen=True)
class CertificateValidationResult:
    """Immutable audit result of validating a DecisionCertificate."""
    is_valid: bool
    errors: Tuple[str, ...]

    def __bool__(self) -> bool:
        return self.is_valid

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise CertificateValidationError(
                f"Certificate validation failed ({len(self.errors)} errors): {'; '.join(self.errors)}"
            )


def validate_certificate(
    certificate: DecisionCertificate,
    decision: Optional[FinalDecision] = None,
    raise_on_error: bool = False,
) -> CertificateValidationResult:
    """Deterministically validate structural and semantic integrity of a DecisionCertificate.

    Validation Rules:
      1. request_id exists and is non-empty.
      2. certificate_version exists and is non-empty.
      3. affordability_status is one of the 4 official statuses.
      4. recommended_payment_method is schema-valid.
      5. Decimal values are valid Decimals.
      6. selected candidate exists if selected_candidate_id is not None.
      7. selected candidate appears in ordered_candidate_ids.
      8. selected rank matches its position in ordered_candidate_ids.
      9. selected candidate safety gate is PASS.
      10. every rejected candidate has explicit rejection reason codes.
      11. ranking trace corresponds to selected candidate.
      12. amount_safe_to_pay matches authoritative upstream evidence.
      13. earliest_date_for_full_payment matches authoritative upstream evidence.
      14. payment_plan matches selected candidate where applicable.
      15. no certificate field contradicts FinalDecision (when provided).
      16. evidence_lineage is populated with valid EvidenceRef objects.
      17. zero-candidate invariants hold if selected_candidate_id is None.
    """
    errors: List[str] = []

    # 1. request_id exists
    if not certificate.request_id or not isinstance(certificate.request_id, str):
        errors.append("request_id missing or invalid")

    # 2. certificate_version exists
    if not certificate.certificate_version or not isinstance(certificate.certificate_version, str):
        errors.append("certificate_version missing or invalid")

    # 3. affordability_status official status
    if certificate.affordability_status not in OFFICIAL_AFFORDABILITY_STATUSES:
        errors.append(f"Invalid affordability_status: {certificate.affordability_status}")

    # 4. recommended_payment_method schema-valid
    if certificate.recommended_payment_method not in VALID_PAYMENT_METHODS:
        errors.append(f"Invalid recommended_payment_method: {certificate.recommended_payment_method}")

    # 5. Decimal values are valid
    if not isinstance(certificate.amount_safe_to_pay, Decimal):
        errors.append(f"amount_safe_to_pay must be Decimal, got {type(certificate.amount_safe_to_pay)}")
    if not isinstance(certificate.financial_evidence.safety_floor, Decimal):
        errors.append("financial_evidence.safety_floor must be Decimal")
    if not isinstance(certificate.financial_evidence.requested_amount, Decimal):
        errors.append("financial_evidence.requested_amount must be Decimal")

    # 6. selected candidate exists if selected_candidate_id is not None
    if certificate.selected_candidate_id is not None:
        if certificate.selected_rank is None or certificate.selected_rank < 1:
            errors.append("selected_rank must be >= 1 when selected_candidate_id is present")
        if certificate.recommended_payment_method == "not_recommended":
            errors.append("recommended_payment_method cannot be 'not_recommended' when candidate is selected")

        # 7. selected candidate appears in ordered_candidate_ids
        if certificate.selected_candidate_id not in certificate.ordered_candidate_ids:
            errors.append(
                f"selected_candidate_id {certificate.selected_candidate_id} not in ordered_candidate_ids"
            )
        else:
            # 8. selected rank matches its position
            expected_rank = certificate.ordered_candidate_ids.index(certificate.selected_candidate_id) + 1
            if certificate.selected_rank != expected_rank:
                errors.append(
                    f"selected_rank {certificate.selected_rank} does not match position {expected_rank}"
                )

        # 9. selected candidate safety gate is PASS
        if not certificate.final_safety_gate_passed:
            errors.append("Selected candidate must pass final safety gate (final_safety_gate_passed=True)")

    else:
        # Zero-candidate invariants
        if certificate.selected_rank is not None:
            errors.append("selected_rank must be None when selected_candidate_id is None")
        if certificate.recommended_payment_method != "not_recommended":
            errors.append("recommended_payment_method must be 'not_recommended' when selected_candidate_id is None")
        if certificate.payment_plan != "none":
            errors.append("payment_plan must be 'none' when selected_candidate_id is None")
        if certificate.final_safety_gate_passed:
            errors.append("final_safety_gate_passed must be False when selected_candidate_id is None")

    # 10. every rejected candidate has explicit rejection reason codes
    for r in certificate.rejected_candidates:
        if not r.rejection_reason_codes:
            errors.append(f"Rejected candidate {r.candidate_id} missing rejection reason codes")
        if r.gate_result != "FAIL":
            errors.append(f"Rejected candidate {r.candidate_id} gate_result must be 'FAIL'")

    # 11. ranking trace corresponds to the selected candidate
    if certificate.ranking_criteria_trace is not None:
        if certificate.selected_candidate_id is None:
            errors.append("ranking_criteria_trace present but selected_candidate_id is None")
        elif certificate.ranking_criteria_trace.candidate_id != certificate.selected_candidate_id:
            errors.append(
                f"ranking_criteria_trace candidate_id {certificate.ranking_criteria_trace.candidate_id} "
                f"does not match selected_candidate_id {certificate.selected_candidate_id}"
            )

    # 16. evidence_lineage validation
    if not certificate.evidence_lineage:
        errors.append("evidence_lineage is empty")
    else:
        lineage_fields = set()
        for ref in certificate.evidence_lineage:
            if not ref.source_module or not ref.source_object or not ref.source_id or not ref.field_name:
                errors.append(f"Invalid EvidenceRef: {ref}")
            lineage_fields.add(ref.field_name)

        # Mandatory facts that must have lineage
        for mandatory_field in (
            "request_id",
            "user_id",
            "amount_safe_to_pay",
            "affordability_status",
            "recommended_payment_method",
            "payment_plan",
        ):
            if mandatory_field not in lineage_fields:
                errors.append(f"Missing lineage reference for mandatory field: {mandatory_field}")

    # 12, 13, 14, 15: Cross-check against FinalDecision when provided
    if decision is not None:
        if certificate.request_id != decision.request_id:
            errors.append(f"request_id mismatch: {certificate.request_id} != {decision.request_id}")
        if certificate.selected_candidate_id != decision.selected_candidate_id:
            errors.append(
                f"selected_candidate_id mismatch: {certificate.selected_candidate_id} != {decision.selected_candidate_id}"
            )
        if certificate.affordability_status != decision.affordability_status.value:
            errors.append(
                f"affordability_status mismatch: {certificate.affordability_status} != {decision.affordability_status.value}"
            )
        if certificate.recommended_payment_method != decision.recommended_payment_method:
            errors.append(
                f"recommended_payment_method mismatch: {certificate.recommended_payment_method} != {decision.recommended_payment_method}"
            )
        # 12. amount_safe_to_pay matches authoritative upstream evidence
        if certificate.amount_safe_to_pay != decision.amount_safe_to_pay:
            errors.append(
                f"amount_safe_to_pay mismatch: {certificate.amount_safe_to_pay} != {decision.amount_safe_to_pay}"
            )
        # 13. earliest_date_for_full_payment matches authoritative upstream evidence
        if certificate.earliest_date_for_full_payment != decision.earliest_date_for_full_payment:
            errors.append(
                f"earliest_date_for_full_payment mismatch: {certificate.earliest_date_for_full_payment} != {decision.earliest_date_for_full_payment}"
            )
        # 14. payment_plan matches selected candidate where applicable
        if certificate.payment_plan != decision.payment_plan:
            errors.append(
                f"payment_plan mismatch: {certificate.payment_plan} != {decision.payment_plan}"
            )
        if certificate.spending_changes_needed != decision.spending_changes_needed:
            errors.append(
                f"spending_changes_needed mismatch: {certificate.spending_changes_needed} != {decision.spending_changes_needed}"
            )
        if certificate.ordered_candidate_ids != decision.ranked_candidate_ids:
            errors.append("ordered_candidate_ids does not match decision.ranked_candidate_ids")

        # 11b. Cross-check all 6 ranking criteria when ranking trace is present
        if decision.ranking_trace is not None:
            if certificate.ranking_criteria_trace is None:
                errors.append("ranking_criteria_trace missing in certificate when present in decision")
            else:
                rt = certificate.ranking_criteria_trace
                d_rt = decision.ranking_trace
                if rt.criterion_1_deadline_met != d_rt.criterion_1_deadline_met:
                    errors.append("criterion_1_deadline_met mismatch")
                if rt.criterion_2_no_spending_changes != d_rt.criterion_2_no_spending_changes:
                    errors.append("criterion_2_no_spending_changes mismatch")
                if rt.criterion_3_total_amount_paid != d_rt.criterion_3_total_amount_paid:
                    errors.append("criterion_3_total_amount_paid mismatch")
                if rt.criterion_4_first_payment_date != d_rt.criterion_4_first_payment_date:
                    errors.append("criterion_4_first_payment_date mismatch")
                if rt.criterion_5_number_of_payments != d_rt.criterion_5_number_of_payments:
                    errors.append("criterion_5_number_of_payments mismatch")
                if rt.criterion_6_payment_option_id != d_rt.criterion_6_payment_option_id:
                    errors.append("criterion_6_payment_option_id mismatch")

    res = CertificateValidationResult(
        is_valid=(len(errors) == 0),
        errors=tuple(errors),
    )
    if raise_on_error:
        res.raise_if_invalid()
    return res
