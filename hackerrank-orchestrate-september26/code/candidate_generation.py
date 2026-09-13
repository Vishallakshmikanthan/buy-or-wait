"""Deterministic Candidate Action Generation Layer for Buy or Wait?

Constructs the complete, deterministic set of legitimate candidate actions for each
financial request. Placed ABOVE the frozen financial engine and BELOW the ranking layer.

WHAT THIS MODULE DOES:
- Constructs FULL_PAYMENT, INSTALLMENT_PLAN, PARTIAL_PAYMENT, and WAIT candidates.
- Preserves frozen upstream feasibility/safety results verbatim.
- Returns an immutable, deterministically ordered collection of candidates.
- Records provenance/lineage for every candidate.

WHAT THIS MODULE DOES NOT DO:
- No ranking or scoring.
- No best-candidate selection.
- No recommendation.
- No output.csv generation.
- No LLM calls.
- No modification of any frozen upstream module.

CANDIDATE TYPE SOURCES:
  FULL_PAYMENT      problem_statement.md L127, AGENTS.md s6.2
  INSTALLMENT_PLAN  problem_statement.md L129 / L144 must exactly match a supplied payment option
  PARTIAL_PAYMENT   problem_statement.md L146 exactly two payments; strict eligibility conditions
  WAIT              problem_statement.md L129; AGENTS.md s6.3: eligible when full payment becomes
                    safe later and user accepts full_payment

ORDERING (s13): Deterministic, NOT business-preference ranked.
  Primary: candidate_type.value (alphabetical)
  Secondary: source_payment_option_id (lexicographic; None -> empty string)
  Tertiary: first_payment_date
  Quaternary: candidate_id

All monetary values use Decimal. No float. No randomness. No wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from code.canonical import CanonicalEvent
from code.models import FinancialProfile, FinancialRequest, PaymentOption
from code.payment_plan import (
    PaymentPlan,
    PaymentPlanFeasibility,
)
from code.recurrence import FutureEvent, RecurrenceSeries
from code.safe_to_pay import SafeToPayCertificate
from code.simulator import CashImpactType, Direction, SimulatedEvent
from code.spending_changes import (
    SpendingActionType,
    SpendingChange,
    generate_spending_change_scenarios,
    identify_eligible_spending_actions,
    optimize_spending_changes_for_candidate,
)


# ---------------------------------------------------------------------------
# Candidate Types
# ---------------------------------------------------------------------------

class CandidateType(str, Enum):
    """All specification-permitted candidate action types.

    Source: problem_statement.md Allowed values / recommended_payment_method
    """
    FULL_PAYMENT = "full_payment"
    INSTALLMENT_PLAN = "installments"
    PARTIAL_PAYMENT = "partial_payment"
    WAIT = "wait"


class CandidateStatus(str, Enum):
    """Structural/contractual/safety classification (s10 - keep concepts separate).

    STRUCTURALLY_INVALID:      missing/bad data; cannot form schedule.
    CONTRACTUALLY_INELIGIBLE:  fails user prefs, date bounds, or horizon.
    STRUCTURALLY_VALID_UNSAFE: valid schedule but safety simulation fails.
    ELIGIBLE_AND_SAFE:         passed all eligibility and safety checks.

    Rankability: ONLY ELIGIBLE_AND_SAFE is rankable. See is_rankable().
    """
    STRUCTURALLY_INVALID = "structurally_invalid"
    CONTRACTUALLY_INELIGIBLE = "contractually_ineligible"
    STRUCTURALLY_VALID_UNSAFE = "structurally_valid_unsafe"
    ELIGIBLE_AND_SAFE = "eligible_and_safe"


class OptionUniverseState(str, Enum):
    """Exactly-one state for a raw CSV payment option (Prompt 10B §4).

    A. contractually ineligible
    B. eligible + safe
    C. eligible + unsafe
    D. structurally invalid
    """
    CONTRACTUALLY_INELIGIBLE = "A_contractually_ineligible"
    ELIGIBLE_SAFE = "B_eligible_safe"
    ELIGIBLE_UNSAFE = "C_eligible_unsafe"
    STRUCTURALLY_INVALID = "D_structurally_invalid"


# Markers copied from frozen payment_plan.validate_option_schema rejection strings.
# Preference / first-date / max-months / horizon failures are CONTRACTUAL, not structural,
# even when check_option_eligibility returns payment_plan=None (it constructs the plan
# only after preference and first-date checks pass).
_STRUCTURAL_REJECTION_MARKERS: Tuple[str, ...] = (
    "option belongs to different request",
    "invalid payment amount",
    "invalid number of payments",
    "invalid financing fee",
    "full_payment requires exactly",
    "full_payment financing fee must be 0",
    "installments require > 1 payments",
    "installments require positive payment_frequency_days",
    "unknown payment method",
    "amount reconciliation failure",
)

_CONTRACTUAL_REJECTION_MARKERS: Tuple[str, ...] = (
    "not permitted by user preferences",
    "precedes request date",
    "exceeds maximum installment duration",
    "outside allowed horizon",
)


def classify_feasibility_status(feasibility: PaymentPlanFeasibility) -> CandidateStatus:
    """Map a frozen PaymentPlanFeasibility onto exactly one CandidateStatus.

    Eligible + safe   -> ELIGIBLE_AND_SAFE
    Eligible + unsafe -> STRUCTURALLY_VALID_UNSAFE
    Ineligible:
      schema/reconciliation failure -> STRUCTURALLY_INVALID
      prefs / first-date / max-months / horizon -> CONTRACTUALLY_INELIGIBLE

    Does NOT rank. Does NOT mutate feasibility.
    """
    if feasibility.is_eligible:
        if feasibility.is_safe:
            return CandidateStatus.ELIGIBLE_AND_SAFE
        return CandidateStatus.STRUCTURALLY_VALID_UNSAFE

    reason = feasibility.rejection_reason or ""
    for marker in _STRUCTURAL_REJECTION_MARKERS:
        if marker in reason:
            return CandidateStatus.STRUCTURALLY_INVALID
    for marker in _CONTRACTUAL_REJECTION_MARKERS:
        if marker in reason:
            return CandidateStatus.CONTRACTUALLY_INELIGIBLE
    # Unknown ineligible reason: plan present implies contractual; else structural.
    if feasibility.payment_plan is not None:
        return CandidateStatus.CONTRACTUALLY_INELIGIBLE
    return CandidateStatus.STRUCTURALLY_INVALID


def classify_option_universe_state(feasibility: PaymentPlanFeasibility) -> OptionUniverseState:
    """Map a frozen feasibility result onto exactly one of A/B/C/D."""
    status = classify_feasibility_status(feasibility)
    if status == CandidateStatus.ELIGIBLE_AND_SAFE:
        return OptionUniverseState.ELIGIBLE_SAFE
    if status == CandidateStatus.STRUCTURALLY_VALID_UNSAFE:
        return OptionUniverseState.ELIGIBLE_UNSAFE
    if status == CandidateStatus.CONTRACTUALLY_INELIGIBLE:
        return OptionUniverseState.CONTRACTUALLY_INELIGIBLE
    return OptionUniverseState.STRUCTURALLY_INVALID


# ---------------------------------------------------------------------------
# Payment entry (candidate-level)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidatePayment:
    """A single payment entry in a candidate schedule. All amounts Decimal."""
    payment_date: date
    amount: Decimal
    sequence_number: int  # 1-indexed

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            raise TypeError(f"amount must be Decimal, got {type(self.amount)}")
        if self.amount < Decimal("0"):
            raise ValueError(f"amount must be non-negative, got {self.amount}")
        if self.sequence_number < 1:
            raise ValueError(f"sequence_number must be >= 1, got {self.sequence_number}")

    @property
    def schedule_entry(self) -> str:
        """Contest-compliant YYYY-MM-DD:amount format."""
        return f"{self.payment_date.isoformat()}:{self.amount}"


# ---------------------------------------------------------------------------
# Candidate Provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateProvenance:
    """Immutable lineage record for every candidate (s14).

    FULL_PAYMENT:     source = safe_to_pay_certificate
    INSTALLMENT_PLAN: source = payment_option_feasibility; source_payment_option_id set
    PARTIAL_PAYMENT:  source = safe_to_pay_certificate
    WAIT:             source = earliest_full_payment_computation
    """
    source: str
    source_payment_option_id: Optional[str]
    safe_to_pay_certificate_id: Optional[str]
    earliest_full_payment_date: Optional[date]


# ---------------------------------------------------------------------------
# Candidate (Immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """Immutable deterministic candidate action.

    All money fields are Decimal. No float. No randomness. No wall-clock time.

    Ordering key (s13 - reproducibility only, NOT business ranking):
    (candidate_type.value, source_payment_option_id or "", first_payment_date, candidate_id)
    """
    candidate_id: str
    request_id: str
    candidate_type: CandidateType
    payment_method: str
    total_amount_paid: Decimal
    first_payment_date: Optional[date]
    final_payment_date: Optional[date]
    number_of_payments: int
    payment_schedule: Tuple[CandidatePayment, ...]
    financing_fee: Decimal
    spending_changes: Tuple[str, ...]
    source_payment_option_id: Optional[str]
    is_safe: bool
    completion_date: Optional[date]
    feasibility_reason: Optional[str]
    status: CandidateStatus
    provenance: CandidateProvenance

    def __post_init__(self) -> None:
        # For ineligible/invalid candidates the schedule is intentionally empty;
        # we only enforce the sum invariant for candidates with a populated schedule.
        if self.payment_schedule:
            schedule_sum = sum(p.amount for p in self.payment_schedule)
            if schedule_sum != self.total_amount_paid:
                raise ValueError(
                    f"Candidate {self.candidate_id}: total_amount_paid ({self.total_amount_paid}) "
                    f"!= sum of schedule ({schedule_sum})"
                )
        if self.payment_schedule and self.number_of_payments != len(self.payment_schedule):
            raise ValueError(
                f"Candidate {self.candidate_id}: number_of_payments ({self.number_of_payments}) "
                f"!= len(payment_schedule) ({len(self.payment_schedule)})"
            )
        # Rankability lockstep: ELIGIBLE_AND_SAFE <=> is_safe. Prevents an unsafe
        # or ineligible candidate from ever looking rankable.
        if self.status == CandidateStatus.ELIGIBLE_AND_SAFE:
            if not self.is_safe:
                raise ValueError(
                    f"Candidate {self.candidate_id}: ELIGIBLE_AND_SAFE requires is_safe=True"
                )
            if not self.payment_schedule:
                raise ValueError(
                    f"Candidate {self.candidate_id}: ELIGIBLE_AND_SAFE requires a populated schedule"
                )
        if self.is_safe and self.status != CandidateStatus.ELIGIBLE_AND_SAFE:
            raise ValueError(
                f"Candidate {self.candidate_id}: is_safe=True requires ELIGIBLE_AND_SAFE, "
                f"got {self.status.value}"
            )

    @property
    def is_rankable(self) -> bool:
        """Hard gate before ranking. TRUE only for eligible + safe candidates.

        See module-level is_rankable() for the full constraint list.
        This is NOT a ranking function.
        """
        return self.status == CandidateStatus.ELIGIBLE_AND_SAFE and self.is_safe

    @property
    def payment_plan_string(self) -> str:
        """Contest-compliant payment plan string."""
        if not self.payment_schedule:
            return "none"
        return "|".join(p.schedule_entry for p in self.payment_schedule)

    @property
    def ordering_key(self) -> Tuple:
        """Deterministic sort key (s13). NOT a business-preference ranking."""
        return (
            self.candidate_type.value,
            self.source_payment_option_id or "",
            self.first_payment_date or date.min,
            self.candidate_id,
        )


# ---------------------------------------------------------------------------
# Candidate Collection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateSet:
    """Immutable deterministic set of candidates for a single request.

    candidates is sorted by ordering_key (s13: reproducibility only, NOT ranking).
    """
    request_id: str
    candidates: Tuple[Candidate, ...]

    @property
    def safe_candidates(self) -> Tuple[Candidate, ...]:
        return tuple(c for c in self.candidates if c.is_safe)

    @property
    def rankable_candidates(self) -> Tuple[Candidate, ...]:
        """Candidates that may enter the later ranking layer. NOT a ranking."""
        return tuple(c for c in self.candidates if c.is_rankable)

    @property
    def by_type(self) -> Dict[CandidateType, List[Candidate]]:
        result: Dict[CandidateType, List[Candidate]] = {}
        for c in self.candidates:
            result.setdefault(c.candidate_type, []).append(c)
        return result

    def count_by_type(self, t: CandidateType) -> int:
        return sum(1 for c in self.candidates if c.candidate_type == t)

    def count_safe_by_type(self, t: CandidateType) -> int:
        return sum(1 for c in self.candidates if c.candidate_type == t and c.is_safe)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_candidate_id(request_id: str, candidate_type: CandidateType, suffix: str) -> str:
    """Construct a deterministic, collision-resistant candidate_id."""
    return f"{request_id}__{candidate_type.value}__{suffix}"


def _candidates_sort_key(c: Candidate) -> Tuple:
    return c.ordering_key


def is_rankable(candidate: Candidate) -> bool:
    """Deterministic hard gate that decides whether a candidate may enter ranking.

    THIS IS NOT A RANKING FUNCTION. It does not score, order, or select a winner.

    TRUE only when every hard constraint required BEFORE ranking is satisfied:

      1. Structural validity
         The candidate has a constructible payment schedule (not schema-broken).

      2. Contract eligibility
         - FULL_PAYMENT / WAIT: user lists full_payment in payment_methods_user_will_consider
         - INSTALLMENT: the supplied option passed check_option_eligibility
           (method permitted, first_payment_date >= request_date, max_installment_months,
           90-day horizon, schema)
         - PARTIAL_PAYMENT: request.allows_partial_payment AND user lists partial_payment

      3. Safety
         90-day forecast never falls below minimum_balance_to_keep
         (frozen simulator / SafeToPayCertificate / PaymentPlanFeasibility).

      4. Completion / construction requirements
         - FULL_PAYMENT: exactly one payment of requested_amount on request_date
         - INSTALLMENT: schedule copied verbatim from a supplied payment option
         - PARTIAL_PAYMENT (hard, problem_statement.md L146):
             0 < amount_safe_to_pay < requested_amount
             earliest_date_for_full_payment exists
             earliest_date_for_full_payment <= desired_completion_date
             exactly two payments summing to requested_amount
         - WAIT: earliest_date_for_full_payment exists AND > request_date
             (one payment of requested_amount on that date)

    NOT a hard gate (ranking criterion 1, not eligibility):
      installment / wait / full_payment completion versus desired_completion_date.
      Exception: PARTIAL_PAYMENT, where the deadline IS a construction condition,
      so a partial candidate that misses the deadline is never generated.

    Implementation: status == ELIGIBLE_AND_SAFE AND is_safe, which Candidate.__post_init__
    keeps in lockstep. Structurally invalid, contractually ineligible, and unsafe
    candidates may exist in the universe for auditability; they are never rankable.
    """
    return candidate.status == CandidateStatus.ELIGIBLE_AND_SAFE and candidate.is_safe


# ---------------------------------------------------------------------------
# FULL_PAYMENT Candidate Construction (s4)
# ---------------------------------------------------------------------------

def build_full_payment_candidate(
    request: FinancialRequest,
    certificate: SafeToPayCertificate,
    profile: FinancialProfile,
) -> Candidate:
    """Construct the canonical FULL_PAYMENT candidate (s4).

    Specification:
    - payment_method = full_payment
    - exactly one payment on request_date for requested_amount
    - financing_fee = 0
    - no spending changes
    - safety from frozen safe_to_pay certificate (is_full_payment_safe_today)
    - user must accept full_payment in payment_methods_user_will_consider
    """
    user_accepts_full = "full_payment" in profile.payment_methods_user_will_consider
    is_safe_today = certificate.is_full_payment_safe_today

    if not user_accepts_full:
        status = CandidateStatus.CONTRACTUALLY_INELIGIBLE
        is_safe = False
        feasibility_reason = "User payment preferences do not include full_payment."
    elif not is_safe_today:
        status = CandidateStatus.STRUCTURALLY_VALID_UNSAFE
        is_safe = False
        feasibility_reason = certificate.reason_if_unsafe or (
            f"Full payment of {certificate.requested_amount} is not safe on {request.request_date}."
        )
    else:
        status = CandidateStatus.ELIGIBLE_AND_SAFE
        is_safe = True
        feasibility_reason = None

    payment = CandidatePayment(
        payment_date=request.request_date,
        amount=certificate.requested_amount,
        sequence_number=1,
    )
    provenance = CandidateProvenance(
        source="safe_to_pay_certificate",
        source_payment_option_id=None,
        safe_to_pay_certificate_id=certificate.request_id,
        earliest_full_payment_date=certificate.earliest_date_for_full_payment,
    )

    return Candidate(
        candidate_id=_make_candidate_id(request.request_id, CandidateType.FULL_PAYMENT, "today"),
        request_id=request.request_id,
        candidate_type=CandidateType.FULL_PAYMENT,
        payment_method=CandidateType.FULL_PAYMENT.value,
        total_amount_paid=certificate.requested_amount,
        first_payment_date=request.request_date,
        final_payment_date=request.request_date,
        number_of_payments=1,
        payment_schedule=(payment,),
        financing_fee=Decimal("0"),
        spending_changes=(),
        source_payment_option_id=None,
        is_safe=is_safe,
        completion_date=request.request_date,
        feasibility_reason=feasibility_reason,
        status=status,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# INSTALLMENT_PLAN Candidates Construction (s5)
# ---------------------------------------------------------------------------

def build_installment_candidate(
    request: FinancialRequest,
    feasibility: PaymentPlanFeasibility,
    option: PaymentOption,
) -> Candidate:
    """Construct one INSTALLMENT_PLAN candidate from a payment option feasibility result (s5).

    Preserves exactly: payment amounts, number of payments, first date,
    frequency (embedded in schedule), financing fee, total payable amount, payment_option_id.
    Safety comes from the frozen simulator result inside feasibility.
    """
    plan: Optional[PaymentPlan] = feasibility.payment_plan
    status = classify_feasibility_status(feasibility)

    if not feasibility.is_eligible:
        # Retain the option as a non-rankable candidate for auditability.
        # Status is CONTRACTUALLY_INELIGIBLE or STRUCTURALLY_INVALID; never rankable.
        return Candidate(
            candidate_id=_make_candidate_id(
                request.request_id, CandidateType.INSTALLMENT_PLAN, option.payment_option_id
            ),
            request_id=request.request_id,
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            payment_method=CandidateType.INSTALLMENT_PLAN.value,
            total_amount_paid=option.total_payable_amount,
            first_payment_date=option.first_payment_date,
            final_payment_date=None,
            number_of_payments=option.number_of_payments,
            payment_schedule=(),
            financing_fee=option.financing_fee,
            spending_changes=(),
            source_payment_option_id=option.payment_option_id,
            is_safe=False,
            completion_date=None,
            feasibility_reason=feasibility.rejection_reason,
            status=status,
            provenance=CandidateProvenance(
                source="payment_option_feasibility",
                source_payment_option_id=option.payment_option_id,
                safe_to_pay_certificate_id=None,
                earliest_full_payment_date=None,
            ),
        )

    assert plan is not None
    schedule: List[CandidatePayment] = []
    for entry in plan.entries:
        schedule.append(CandidatePayment(
            payment_date=entry.date,
            amount=entry.amount,
            sequence_number=entry.sequence_number,
        ))

    is_safe = feasibility.is_safe
    status = classify_feasibility_status(feasibility)
    feasibility_reason = None if is_safe else feasibility.rejection_reason
    completion_date = plan.last_payment_date if plan.entries else None

    return Candidate(
        candidate_id=_make_candidate_id(
            request.request_id, CandidateType.INSTALLMENT_PLAN, option.payment_option_id
        ),
        request_id=request.request_id,
        candidate_type=CandidateType.INSTALLMENT_PLAN,
        payment_method=CandidateType.INSTALLMENT_PLAN.value,
        total_amount_paid=plan.total_payable_amount,
        first_payment_date=plan.first_payment_date,
        final_payment_date=plan.last_payment_date,
        number_of_payments=plan.number_of_payments,
        payment_schedule=tuple(schedule),
        financing_fee=plan.financing_fee,
        spending_changes=(),
        source_payment_option_id=option.payment_option_id,
        is_safe=is_safe,
        completion_date=completion_date,
        feasibility_reason=feasibility_reason,
        status=status,
        provenance=CandidateProvenance(
            source="payment_option_feasibility",
            source_payment_option_id=option.payment_option_id,
            safe_to_pay_certificate_id=None,
            earliest_full_payment_date=plan.last_payment_date,
        ),
    )


# ---------------------------------------------------------------------------
# PARTIAL_PAYMENT Candidate Construction (s6)
# ---------------------------------------------------------------------------

def build_partial_payment_candidate(
    request: FinancialRequest,
    certificate: SafeToPayCertificate,
    profile: FinancialProfile,
) -> Optional[Candidate]:
    """Construct a PARTIAL_PAYMENT candidate when all specification conditions are met (s6).

    Specification (problem_statement.md L146):
    - request.allows_partial_payment must be True
    - user accepts partial_payment in payment_methods_user_will_consider
    - 0 < amount_safe_to_pay < requested_amount (strictly between; not zero, not full amount)
    - earliest_date_for_full_payment must exist and be <= desired_completion_date (HARD DEADLINE)

    If amount_safe_to_pay == 0:            excluded (zero-safe case, s6)
    If amount_safe_to_pay == requested:    excluded (would duplicate full_payment, s6)

    Exactly two payments:
      Payment 1: request_date, amount_safe_to_pay
      Payment 2: earliest_date_for_full_payment, requested_amount - amount_safe_to_pay

    Returns None if any condition is not met.
    """
    if not request.allows_partial_payment:
        return None
    if "partial_payment" not in profile.payment_methods_user_will_consider:
        return None

    safe_amt = certificate.amount_safe_to_pay
    req_amt = certificate.requested_amount
    earliest_date = certificate.earliest_date_for_full_payment

    # Strictly 0 < amount_safe_to_pay < requested_amount
    if safe_amt <= Decimal("0"):
        return None
    if safe_amt >= req_amt:
        return None

    # Hard deadline: earliest_date must exist and be on or before desired_completion_date
    if earliest_date is None:
        return None
    if earliest_date > request.desired_completion_date:
        return None

    remaining = req_amt - safe_amt
    payment1 = CandidatePayment(
        payment_date=request.request_date,
        amount=safe_amt,
        sequence_number=1,
    )
    payment2 = CandidatePayment(
        payment_date=earliest_date,
        amount=remaining,
        sequence_number=2,
    )
    total = safe_amt + remaining
    assert total == req_amt, f"Partial payment amounts {safe_amt} + {remaining} != {req_amt}"

    return Candidate(
        candidate_id=_make_candidate_id(
            request.request_id, CandidateType.PARTIAL_PAYMENT, "two_payment"
        ),
        request_id=request.request_id,
        candidate_type=CandidateType.PARTIAL_PAYMENT,
        payment_method=CandidateType.PARTIAL_PAYMENT.value,
        total_amount_paid=total,
        first_payment_date=request.request_date,
        final_payment_date=earliest_date,
        number_of_payments=2,
        payment_schedule=(payment1, payment2),
        financing_fee=Decimal("0"),
        spending_changes=(),
        source_payment_option_id=None,
        is_safe=True,  # Both payments individually validated by safe_to_pay machinery
        completion_date=earliest_date,
        feasibility_reason=None,
        status=CandidateStatus.ELIGIBLE_AND_SAFE,
        provenance=CandidateProvenance(
            source="safe_to_pay_certificate",
            source_payment_option_id=None,
            safe_to_pay_certificate_id=certificate.request_id,
            earliest_full_payment_date=earliest_date,
        ),
    )


# ---------------------------------------------------------------------------
# WAIT Candidate Construction (s7)
# ---------------------------------------------------------------------------

def build_wait_candidate(
    request: FinancialRequest,
    certificate: SafeToPayCertificate,
    profile: FinancialProfile,
) -> Optional[Candidate]:
    """Construct a WAIT candidate when specification conditions are met (s7).

    Specification (AGENTS.md s6.3 / problem_statement.md Choosing Between Safe Plans):
    "wait is eligible when full payment becomes safe later and the user accepts full_payment"

    WAIT semantics:
    - No payment today.
    - Exactly one payment on earliest_date_for_full_payment for the full requested amount.
    - completion_date = earliest_date_for_full_payment.
    - No financing fee, no spending changes.

    Conditions:
    - user accepts full_payment in payment_methods_user_will_consider
    - earliest_date_for_full_payment exists and > request_date (strictly later than today)

    Returns None if conditions not met.
    """
    if "full_payment" not in profile.payment_methods_user_will_consider:
        return None

    earliest_date = certificate.earliest_date_for_full_payment
    if earliest_date is None:
        return None
    if earliest_date <= request.request_date:
        # On request_date itself = FULL_PAYMENT, not WAIT
        return None

    payment = CandidatePayment(
        payment_date=earliest_date,
        amount=certificate.requested_amount,
        sequence_number=1,
    )

    return Candidate(
        candidate_id=_make_candidate_id(
            request.request_id, CandidateType.WAIT, earliest_date.isoformat()
        ),
        request_id=request.request_id,
        candidate_type=CandidateType.WAIT,
        payment_method=CandidateType.WAIT.value,
        total_amount_paid=certificate.requested_amount,
        first_payment_date=earliest_date,
        final_payment_date=earliest_date,
        number_of_payments=1,
        payment_schedule=(payment,),
        financing_fee=Decimal("0"),
        spending_changes=(),
        source_payment_option_id=None,
        is_safe=True,  # earliest_date_for_full_payment is validated safe by the frozen engine
        completion_date=earliest_date,
        feasibility_reason=None,
        status=CandidateStatus.ELIGIBLE_AND_SAFE,
        provenance=CandidateProvenance(
            source="earliest_full_payment_computation",
            source_payment_option_id=None,
            safe_to_pay_certificate_id=certificate.request_id,
            earliest_full_payment_date=earliest_date,
        ),
    )


# ---------------------------------------------------------------------------
# SPENDING_CHANGE Rescue Candidates Construction (Phases 9 & 10)
# ---------------------------------------------------------------------------

def generate_spending_change_candidates(
    request: FinancialRequest,
    profile: FinancialProfile,
    certificate: SafeToPayCertificate,
    payment_option_feasibilities: Sequence[PaymentPlanFeasibility],
    payment_options_by_id: Dict[str, PaymentOption],
    canonical_events: Sequence[CanonicalEvent] = (),
    future_events: Sequence[FutureEvent] = (),
    recurrence_series: Sequence[RecurrenceSeries] = (),
) -> List[Candidate]:
    """Generate minimal-disruption spending-change rescue candidates.

    Evaluates whether spending changes can rescue:
      1. FULL_PAYMENT on request_date (if user accepts full_payment and base is unsafe).
      2. INSTALLMENT_PLAN options (for eligible but unsafe installment options).
      3. PARTIAL_PAYMENT (if allows_partial_payment and base partial is unsafe/missing).
      4. WAIT on desired_completion_date (if user accepts full_payment and base wait is unsafe/missing).

    Guarantees:
      - Does NOT generate candidates if base candidate is already safe.
      - Never uses float.
      - Completely deterministic.
      - Resimulates every candidate through simulate_user via optimize_spending_changes_for_candidate.
    """
    if not recurrence_series and not canonical_events:
        return []

    eligible_actions = identify_eligible_spending_actions(
        user_id=request.user_id,
        request_date=request.request_date,
        simulation_end=request.request_date + timedelta(days=90),
        profile=profile,
        series_list=recurrence_series,
        canonical_events=canonical_events,
        future_events=future_events,
    )
    if not eligible_actions:
        return []

    scenarios = generate_spending_change_scenarios(eligible_actions, max_changes=3)
    if not scenarios:
        return []

    rescued: List[Candidate] = []
    sim_start = request.request_date
    sim_end = request.request_date + timedelta(days=90)

    # 1. FULL_PAYMENT rescue
    if (
        "full_payment" in profile.payment_methods_user_will_consider
        and not certificate.is_full_payment_safe_today
    ):
        full_event = SimulatedEvent(
            event_id=f"candidate_purchase_{request.request_id}_{request.request_date.strftime('%Y%m%d')}_{str(certificate.requested_amount).replace('.', '_')}",
            user_id=request.user_id,
            effective_date=request.request_date,
            direction=Direction.OUTFLOW,
            amount=certificate.requested_amount,
            currency=profile.home_currency,
            category=request.request_type,
            event_type="expense",
            description=f"Candidate purchase for request {request.request_id}",
            cash_impact_type=CashImpactType.SETTLED_OUTFLOW,
            flexibility="fixed",
            is_protected=False,
            source_type="action",
            is_forecast=False,
        )
        best_scenario = optimize_spending_changes_for_candidate(
            user_id=request.user_id,
            simulation_start=sim_start,
            simulation_end=sim_end,
            profile=profile,
            canonical_events=canonical_events,
            future_events=future_events,
            scenarios=scenarios,
            candidate_events=(full_event,),
        )
        if best_scenario is not None:
            payment = CandidatePayment(
                payment_date=request.request_date,
                amount=certificate.requested_amount,
                sequence_number=1,
            )
            rescued.append(
                Candidate(
                    candidate_id=_make_candidate_id(request.request_id, CandidateType.FULL_PAYMENT, "today_with_spending_changes"),
                    request_id=request.request_id,
                    candidate_type=CandidateType.FULL_PAYMENT,
                    payment_method=CandidateType.FULL_PAYMENT.value,
                    total_amount_paid=certificate.requested_amount,
                    first_payment_date=request.request_date,
                    final_payment_date=request.request_date,
                    number_of_payments=1,
                    payment_schedule=(payment,),
                    financing_fee=Decimal("0"),
                    spending_changes=tuple(c.to_action_string() for c in best_scenario.changes),
                    source_payment_option_id=None,
                    is_safe=True,
                    completion_date=request.request_date,
                    feasibility_reason=None,
                    status=CandidateStatus.ELIGIBLE_AND_SAFE,
                    provenance=CandidateProvenance(
                        source="spending_change_optimization",
                        source_payment_option_id=None,
                        safe_to_pay_certificate_id=certificate.request_id,
                        earliest_full_payment_date=certificate.earliest_date_for_full_payment,
                    ),
                )
            )

    # 2. INSTALLMENT_PLAN rescue
    for feasibility in payment_option_feasibilities:
        option = payment_options_by_id.get(feasibility.payment_option_id)
        if option is None or option.payment_method != "installments":
            continue
        if option.payment_method not in profile.payment_methods_user_will_consider:
            continue
        if not feasibility.is_eligible or feasibility.payment_plan is None:
            continue
        if feasibility.is_safe:
            continue  # Already safe without spending changes

        plan = feasibility.payment_plan
        plan_events: List[SimulatedEvent] = []
        schedule: List[CandidatePayment] = []
        for entry in plan.entries:
            plan_events.append(
                SimulatedEvent(
                    event_id=f"plan_action_{option.payment_option_id}_{entry.sequence_number}_{entry.date.strftime('%Y%m%d')}",
                    user_id=request.user_id,
                    effective_date=entry.date,
                    direction=Direction.OUTFLOW,
                    amount=entry.amount,
                    currency=profile.home_currency,
                    category="candidate_payment",
                    event_type="action",
                    description=f"Payment {entry.sequence_number} for option {option.payment_option_id}",
                    cash_impact_type=CashImpactType.SETTLED_OUTFLOW,
                    flexibility="fixed",
                    is_protected=False,
                    source_type="action",
                    is_forecast=False,
                )
            )
            schedule.append(
                CandidatePayment(
                    payment_date=entry.date,
                    amount=entry.amount,
                    sequence_number=entry.sequence_number,
                )
            )

        best_scenario = optimize_spending_changes_for_candidate(
            user_id=request.user_id,
            simulation_start=sim_start,
            simulation_end=sim_end,
            profile=profile,
            canonical_events=canonical_events,
            future_events=future_events,
            scenarios=scenarios,
            candidate_events=tuple(plan_events),
        )
        if best_scenario is not None:
            rescued.append(
                Candidate(
                    candidate_id=_make_candidate_id(
                        request.request_id, CandidateType.INSTALLMENT_PLAN, f"{option.payment_option_id}_with_spending_changes"
                    ),
                    request_id=request.request_id,
                    candidate_type=CandidateType.INSTALLMENT_PLAN,
                    payment_method=CandidateType.INSTALLMENT_PLAN.value,
                    total_amount_paid=plan.total_payable_amount,
                    first_payment_date=plan.first_payment_date,
                    final_payment_date=plan.last_payment_date,
                    number_of_payments=plan.number_of_payments,
                    payment_schedule=tuple(schedule),
                    financing_fee=plan.financing_fee,
                    spending_changes=tuple(c.to_action_string() for c in best_scenario.changes),
                    source_payment_option_id=option.payment_option_id,
                    is_safe=True,
                    completion_date=plan.last_payment_date,
                    feasibility_reason=None,
                    status=CandidateStatus.ELIGIBLE_AND_SAFE,
                    provenance=CandidateProvenance(
                        source="spending_change_optimization",
                        source_payment_option_id=option.payment_option_id,
                        safe_to_pay_certificate_id=None,
                        earliest_full_payment_date=plan.last_payment_date,
                    ),
                )
            )

    # 3. PARTIAL_PAYMENT rescue
    if (
        request.allows_partial_payment
        and "partial_payment" in profile.payment_methods_user_will_consider
        and Decimal("0") < certificate.amount_safe_to_pay < certificate.requested_amount
    ):
        earliest_date = certificate.earliest_date_for_full_payment
        if earliest_date is None or earliest_date > request.desired_completion_date:
            if request.desired_completion_date > request.request_date:
                p1_amt = certificate.amount_safe_to_pay
                p2_amt = certificate.requested_amount - p1_amt
                p1_ev = SimulatedEvent(
                    event_id=f"part_p1_{request.request_id}_{request.request_date.strftime('%Y%m%d')}",
                    user_id=request.user_id,
                    effective_date=request.request_date,
                    direction=Direction.OUTFLOW,
                    amount=p1_amt,
                    currency=profile.home_currency,
                    category=request.request_type,
                    event_type="expense",
                    description=f"Partial payment 1 for {request.request_id}",
                    cash_impact_type=CashImpactType.SETTLED_OUTFLOW,
                    flexibility="fixed",
                    is_protected=False,
                    source_type="action",
                    is_forecast=False,
                )
                p2_ev = SimulatedEvent(
                    event_id=f"part_p2_{request.request_id}_{request.desired_completion_date.strftime('%Y%m%d')}",
                    user_id=request.user_id,
                    effective_date=request.desired_completion_date,
                    direction=Direction.OUTFLOW,
                    amount=p2_amt,
                    currency=profile.home_currency,
                    category=request.request_type,
                    event_type="expense",
                    description=f"Partial payment 2 for {request.request_id}",
                    cash_impact_type=CashImpactType.SETTLED_OUTFLOW,
                    flexibility="fixed",
                    is_protected=False,
                    source_type="action",
                    is_forecast=False,
                )
                best_scenario = optimize_spending_changes_for_candidate(
                    user_id=request.user_id,
                    simulation_start=sim_start,
                    simulation_end=sim_end,
                    profile=profile,
                    canonical_events=canonical_events,
                    future_events=future_events,
                    scenarios=scenarios,
                    candidate_events=(p1_ev, p2_ev),
                )
                if best_scenario is not None:
                    p1 = CandidatePayment(request.request_date, p1_amt, 1)
                    p2 = CandidatePayment(request.desired_completion_date, p2_amt, 2)
                    rescued.append(
                        Candidate(
                            candidate_id=_make_candidate_id(request.request_id, CandidateType.PARTIAL_PAYMENT, "two_payment_with_spending_changes"),
                            request_id=request.request_id,
                            candidate_type=CandidateType.PARTIAL_PAYMENT,
                            payment_method=CandidateType.PARTIAL_PAYMENT.value,
                            total_amount_paid=certificate.requested_amount,
                            first_payment_date=request.request_date,
                            final_payment_date=request.desired_completion_date,
                            number_of_payments=2,
                            payment_schedule=(p1, p2),
                            financing_fee=Decimal("0"),
                            spending_changes=tuple(c.to_action_string() for c in best_scenario.changes),
                            source_payment_option_id=None,
                            is_safe=True,
                            completion_date=request.desired_completion_date,
                            feasibility_reason=None,
                            status=CandidateStatus.ELIGIBLE_AND_SAFE,
                            provenance=CandidateProvenance(
                                source="spending_change_optimization",
                                source_payment_option_id=None,
                                safe_to_pay_certificate_id=certificate.request_id,
                                earliest_full_payment_date=request.desired_completion_date,
                            ),
                        )
                    )

    # 4. WAIT rescue
    if (
        "full_payment" in profile.payment_methods_user_will_consider
        and not certificate.is_full_payment_safe_today
    ):
        earliest_date = certificate.earliest_date_for_full_payment
        if earliest_date is None or earliest_date > request.desired_completion_date:
            if request.desired_completion_date > request.request_date:
                wait_event = SimulatedEvent(
                    event_id=f"candidate_wait_{request.request_id}_{request.desired_completion_date.strftime('%Y%m%d')}_{str(certificate.requested_amount).replace('.', '_')}",
                    user_id=request.user_id,
                    effective_date=request.desired_completion_date,
                    direction=Direction.OUTFLOW,
                    amount=certificate.requested_amount,
                    currency=profile.home_currency,
                    category=request.request_type,
                    event_type="expense",
                    description=f"Candidate wait purchase for request {request.request_id}",
                    cash_impact_type=CashImpactType.SETTLED_OUTFLOW,
                    flexibility="fixed",
                    is_protected=False,
                    source_type="action",
                    is_forecast=False,
                )
                best_scenario = optimize_spending_changes_for_candidate(
                    user_id=request.user_id,
                    simulation_start=sim_start,
                    simulation_end=sim_end,
                    profile=profile,
                    canonical_events=canonical_events,
                    future_events=future_events,
                    scenarios=scenarios,
                    candidate_events=(wait_event,),
                )
                if best_scenario is not None:
                    payment = CandidatePayment(
                        payment_date=request.desired_completion_date,
                        amount=certificate.requested_amount,
                        sequence_number=1,
                    )
                    rescued.append(
                        Candidate(
                            candidate_id=_make_candidate_id(
                                request.request_id, CandidateType.WAIT, f"{request.desired_completion_date.isoformat()}_with_spending_changes"
                            ),
                            request_id=request.request_id,
                            candidate_type=CandidateType.WAIT,
                            payment_method=CandidateType.WAIT.value,
                            total_amount_paid=certificate.requested_amount,
                            first_payment_date=request.desired_completion_date,
                            final_payment_date=request.desired_completion_date,
                            number_of_payments=1,
                            payment_schedule=(payment,),
                            financing_fee=Decimal("0"),
                            spending_changes=tuple(c.to_action_string() for c in best_scenario.changes),
                            source_payment_option_id=None,
                            is_safe=True,
                            completion_date=request.desired_completion_date,
                            feasibility_reason=None,
                            status=CandidateStatus.ELIGIBLE_AND_SAFE,
                            provenance=CandidateProvenance(
                                source="spending_change_optimization",
                                source_payment_option_id=None,
                                safe_to_pay_certificate_id=certificate.request_id,
                                earliest_full_payment_date=request.desired_completion_date,
                            ),
                        )
                    )

    return rescued


# ---------------------------------------------------------------------------
# Master Candidate Generation Function
# ---------------------------------------------------------------------------

def generate_candidates(
    request: FinancialRequest,
    profile: FinancialProfile,
    certificate: SafeToPayCertificate,
    payment_option_feasibilities: Sequence[PaymentPlanFeasibility],
    payment_options_by_id: Dict[str, PaymentOption],
    canonical_events: Sequence[CanonicalEvent] = (),
    future_events: Sequence[FutureEvent] = (),
    recurrence_series: Sequence[RecurrenceSeries] = (),
) -> CandidateSet:
    """Generate the complete deterministic candidate set for a single request.

    This is the main entry point for the candidate generation layer.

    CONTRACT:
    - Does NOT rank candidates.
    - Does NOT select a best candidate.
    - Does NOT modify any upstream state.
    - All money is Decimal.
    - No randomness, no wall clock, no network.
    - Calling this function twice with the same inputs produces identical CandidateSet.

    Args:
        request:                     The financial request being evaluated.
        profile:                     The user's financial profile.
        certificate:                 SafeToPayCertificate from the frozen safe_to_pay engine.
        payment_option_feasibilities: All feasibility results from the frozen payment_plan engine.
        payment_options_by_id:       Dict mapping payment_option_id -> PaymentOption for lineage.
        canonical_events:            User's canonical events for spending change simulation.
        future_events:               User's future recurring events for spending change simulation.
        recurrence_series:           User's recurring series for spending change policy evaluation.

    Returns:
        CandidateSet with immutable, deterministically ordered candidates.
    """
    candidates: List[Candidate] = []

    # =========================================================================
    # 1. FULL_PAYMENT candidate (always constructed; s4)
    # =========================================================================
    full_payment_cand = build_full_payment_candidate(request, certificate, profile)
    candidates.append(full_payment_cand)

    # =========================================================================
    # 2. INSTALLMENT_PLAN candidates (one per installment feasibility result; s5)
    # =========================================================================
    for feasibility in payment_option_feasibilities:
        option = payment_options_by_id.get(feasibility.payment_option_id)
        if option is None:
            continue
        if option.payment_method != "installments":
            # full_payment options from the CSV are not used to build INSTALLMENT_PLAN candidates.
            # The canonical FULL_PAYMENT candidate is built from the request itself (step 1).
            continue
        cand = build_installment_candidate(request, feasibility, option)
        candidates.append(cand)

    # =========================================================================
    # 3. PARTIAL_PAYMENT candidate (s6)
    # =========================================================================
    partial_cand = build_partial_payment_candidate(request, certificate, profile)
    if partial_cand is not None:
        candidates.append(partial_cand)

    # =========================================================================
    # 4. WAIT candidate (s7)
    # =========================================================================
    wait_cand = build_wait_candidate(request, certificate, profile)
    if wait_cand is not None:
        candidates.append(wait_cand)

    # =========================================================================
    # 5. SPENDING-CHANGE RESCUE CANDIDATES (Phases 9 & 10)
    # =========================================================================
    spending_rescued = generate_spending_change_candidates(
        request=request,
        profile=profile,
        certificate=certificate,
        payment_option_feasibilities=payment_option_feasibilities,
        payment_options_by_id=payment_options_by_id,
        canonical_events=canonical_events,
        future_events=future_events,
        recurrence_series=recurrence_series,
    )
    candidates.extend(spending_rescued)

    # =========================================================================
    # 6. Deterministic ordering (s13 - reproducibility only, NOT ranking)
    # =========================================================================
    candidates.sort(key=_candidates_sort_key)

    return CandidateSet(
        request_id=request.request_id,
        candidates=tuple(candidates),
    )
