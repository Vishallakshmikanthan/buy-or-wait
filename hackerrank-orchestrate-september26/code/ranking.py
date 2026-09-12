"""Deterministic Specification-Faithful Ranking Engine for Buy or Wait?

Specification Sources:
  1. problem_statement.md lines 187-197 ("Choosing Between Safe Plans"):
     "When more than one eligible plan is safe, rank the plans in this order:
      1. Complete the full request by desired_completion_date.
      2. Require no spending changes.
      3. Minimize the total amount paid.
      4. Start payment earlier.
      5. Use fewer payments.
      6. Use the lowest payment_option_id as the final tie-breaker."
  2. AGENTS.md line 215 (§6.3 Financial Decision Rules):
     "Prefer a plan that completes the request by its deadline, avoids spending
      changes, minimizes total payment cost, starts earlier, and uses fewer payments."

WHAT THIS MODULE DOES:
- Receives ONLY rankable candidates (candidate.is_rankable == True).
- Hard gate: raises ValueError if any non-rankable candidate enters ranking.
- Evaluates candidates according to the exact 6 contest criteria via a deterministic
  lexicographic comparator.
- Handles non-option candidates (full_payment, partial_payment, wait) at Criterion 6:
  payment_option_id is compared when both candidates have a payment_option_id;
  non-option candidates tie at Criterion 6 at the business ranking level.
- Final technical tie-breaker (Criterion 7): candidate.candidate_id (lexicographic string
  comparison) guarantees total, permutation-invariant, reproducible ordering without
  injecting arbitrary business preference.
- Emits both a strict deterministic rank (1, 2, 3, ...) and a business rank (1, 1, 3, ...)
  that documents true business equivalence.
- Provides full criteria trace for auditability and explainability.

WHAT THIS MODULE DOES NOT DO:
- No floating-point scores.
- No weighted linear combinations (w1*x1 + w2*x2 + ...).
- No heuristic penalties.
- No modification of Candidate, CandidateSet, or any upstream data structures.
- No recommendation generation, output.csv writing, or LLM explanation calls.
- No modification of any frozen upstream module.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional, Sequence, Tuple

from code.candidate_generation import Candidate, CandidateSet, is_rankable


# ---------------------------------------------------------------------------
# Ranking Criteria Trace
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankingCriteriaTrace:
    """Full criteria trace for one candidate evaluation against a request deadline.

    Records the exact values of all 6 specification criteria plus the technical tie-breaker.
    All monetary values use Decimal. No float.
    """
    candidate_id: str
    criterion_1_deadline_met: bool
    criterion_2_no_spending_changes: bool
    criterion_3_total_amount_paid: Decimal
    criterion_4_first_payment_date: date
    criterion_5_number_of_payments: int
    criterion_6_payment_option_id: Optional[str]
    technical_tie_breaker: str  # candidate_id


# ---------------------------------------------------------------------------
# Individual Criterion Evaluators
# ---------------------------------------------------------------------------

def eval_criterion_1_deadline_met(candidate: Candidate, desired_completion_date: date) -> bool:
    """Criterion 1: Complete the full request by desired_completion_date.

    Source: problem_statement.md line 191; AGENTS.md line 215.
    Semantics: completion_date <= desired_completion_date (inclusive).
    True ranks before False.
    """
    if candidate.completion_date is None:
        return False
    return candidate.completion_date <= desired_completion_date


def eval_criterion_2_no_spending_changes(candidate: Candidate) -> bool:
    """Criterion 2: Require no spending changes.

    Source: problem_statement.md line 192; AGENTS.md line 215.
    Semantics: len(candidate.spending_changes) == 0.
    True ranks before False (zero spending changes ranks before >0 changes).
    """
    return len(candidate.spending_changes) == 0


def eval_criterion_3_total_amount_paid(candidate: Candidate) -> Decimal:
    """Criterion 3: Minimize the total amount paid.

    Source: problem_statement.md line 193; AGENTS.md line 215 ("minimizes total payment cost").
    Semantics: candidate.total_amount_paid (Decimal).
    Lower amount ranks before higher amount. Includes financing fees for installments.
    """
    return candidate.total_amount_paid


def eval_criterion_4_first_payment_date(candidate: Candidate) -> date:
    """Criterion 4: Start payment earlier.

    Source: problem_statement.md line 194; AGENTS.md line 215 ("starts earlier").
    Semantics: candidate.first_payment_date.
    Earlier date ranks before later date.
    For rankable candidates, first_payment_date is guaranteed to be a valid date.
    """
    if candidate.first_payment_date is None:
        raise ValueError(
            f"Candidate {candidate.candidate_id} has no first_payment_date; cannot evaluate Criterion 4."
        )
    return candidate.first_payment_date


def eval_criterion_5_number_of_payments(candidate: Candidate) -> int:
    """Criterion 5: Use fewer payments.

    Source: problem_statement.md line 195; AGENTS.md line 215 ("uses fewer payments").
    Semantics: candidate.number_of_payments (int).
    Fewer payments ranks before more payments (1 for full_payment, 1 for wait, 2 for partial, N for installments).
    """
    return candidate.number_of_payments


def eval_criterion_6_payment_option_id(candidate: Candidate) -> Optional[str]:
    """Criterion 6: Use the lowest payment_option_id as the final tie-breaker.

    Source: problem_statement.md line 196.
    Semantics: candidate.source_payment_option_id.
    Populated for installment options from request_payment_options.csv.
    None for full_payment, partial_payment, and wait candidates.
    """
    return candidate.source_payment_option_id


def build_criteria_trace(candidate: Candidate, desired_completion_date: date) -> RankingCriteriaTrace:
    """Build the complete audit trace of all 6 criteria for a candidate."""
    return RankingCriteriaTrace(
        candidate_id=candidate.candidate_id,
        criterion_1_deadline_met=eval_criterion_1_deadline_met(candidate, desired_completion_date),
        criterion_2_no_spending_changes=eval_criterion_2_no_spending_changes(candidate),
        criterion_3_total_amount_paid=eval_criterion_3_total_amount_paid(candidate),
        criterion_4_first_payment_date=eval_criterion_4_first_payment_date(candidate),
        criterion_5_number_of_payments=eval_criterion_5_number_of_payments(candidate),
        criterion_6_payment_option_id=eval_criterion_6_payment_option_id(candidate),
        technical_tie_breaker=candidate.candidate_id,
    )


# ---------------------------------------------------------------------------
# Business and Deterministic Comparators
# ---------------------------------------------------------------------------

def compare_candidates_business(c1: Candidate, c2: Candidate, desired_completion_date: date) -> int:
    """Compare two candidates strictly on the 6 specification business criteria.

    Returns:
      -1 if c1 is strictly preferred over c2 under business criteria
       1 if c2 is strictly preferred over c1 under business criteria
       0 if c1 and c2 tie across all applicable business criteria

    Exact Priority Order:
      1. Deadline met (True > False)
      2. No spending changes (True > False)
      3. Total amount paid (Decimal, lower is better)
      4. Start payment earlier (date, earlier is better)
      5. Fewer payments (int, fewer is better)
      6. Lowest payment_option_id (when both have payment_option_id; string ascending)

    Non-option candidates (full_payment, partial_payment, wait) have no payment_option_id;
    if one or both candidates lack payment_option_id, Criterion 6 does not separate them,
    yielding a business tie (0).
    """
    # Criterion 1: Complete the full request by desired_completion_date
    m1 = eval_criterion_1_deadline_met(c1, desired_completion_date)
    m2 = eval_criterion_1_deadline_met(c2, desired_completion_date)
    if m1 != m2:
        return -1 if m1 else 1

    # Criterion 2: Require no spending changes
    sc1 = eval_criterion_2_no_spending_changes(c1)
    sc2 = eval_criterion_2_no_spending_changes(c2)
    if sc1 != sc2:
        return -1 if sc1 else 1

    # Criterion 3: Minimize total amount paid
    t1 = eval_criterion_3_total_amount_paid(c1)
    t2 = eval_criterion_3_total_amount_paid(c2)
    if t1 != t2:
        return -1 if t1 < t2 else 1

    # Criterion 4: Start payment earlier
    d1 = eval_criterion_4_first_payment_date(c1)
    d2 = eval_criterion_4_first_payment_date(c2)
    if d1 != d2:
        return -1 if d1 < d2 else 1

    # Criterion 5: Use fewer payments
    p1 = eval_criterion_5_number_of_payments(c1)
    p2 = eval_criterion_5_number_of_payments(c2)
    if p1 != p2:
        return -1 if p1 < p2 else 1

    # Criterion 6: Lowest payment_option_id
    opt1 = eval_criterion_6_payment_option_id(c1)
    opt2 = eval_criterion_6_payment_option_id(c2)
    if opt1 is not None and opt2 is not None:
        if opt1 != opt2:
            return -1 if opt1 < opt2 else 1

    # Tied on all applicable business criteria
    return 0


def compare_candidates_deterministic(c1: Candidate, c2: Candidate, desired_completion_date: date) -> int:
    """Full deterministic comparator combining business criteria + technical tie-breaker.

    Guarantees:
      - Strictly obeys all 6 business criteria.
      - If and only if business criteria tie, applies Criterion 7 (candidate_id ascending).
      - Strict weak ordering on business rules; strict total ordering overall.
      - 100% permutation-invariant and deterministic.
    """
    biz = compare_candidates_business(c1, c2, desired_completion_date)
    if biz != 0:
        return biz

    # Business criteria tied: apply technical tie-breaker (candidate_id)
    if c1.candidate_id != c2.candidate_id:
        return -1 if c1.candidate_id < c2.candidate_id else 1

    return 0


def deterministic_ranking_key(candidate: Candidate, desired_completion_date: date) -> Tuple:
    """Lexicographic sort key corresponding to the deterministic ordering.

    Useful for inspection and direct sort key operations.
    Order:
      1. (0 if deadline_met else 1)
      2. (0 if no_spending_changes else 1)
      3. total_amount_paid (Decimal)
      4. first_payment_date (date)
      5. number_of_payments (int)
      6. payment_option_id tie-breaker:
         - (0, source_payment_option_id) if option present
         - (1, "") if non-option (business tie-broken technically)
      7. candidate_id (str)
    """
    d_met = 0 if eval_criterion_1_deadline_met(candidate, desired_completion_date) else 1
    no_sc = 0 if eval_criterion_2_no_spending_changes(candidate) else 1
    t_amt = eval_criterion_3_total_amount_paid(candidate)
    f_date = eval_criterion_4_first_payment_date(candidate)
    n_pay = eval_criterion_5_number_of_payments(candidate)
    opt_id = eval_criterion_6_payment_option_id(candidate)
    opt_key = (0, opt_id) if opt_id is not None else (1, "")
    return (d_met, no_sc, t_amt, f_date, n_pay, opt_key, candidate.candidate_id)


# ---------------------------------------------------------------------------
# Ranked Candidate Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankedCandidate:
    """An evaluated and ranked candidate. Immutable."""
    candidate: Candidate
    rank: int  # 1-indexed deterministic total rank (1, 2, 3, ...)
    business_rank: int  # 1-indexed business rank with ties (1, 1, 3, ...)
    trace: RankingCriteriaTrace


@dataclass(frozen=True)
class RankedCandidateSet:
    """Immutable collection of ranked candidates for a request.

    Contains candidates sorted strictly in ascending deterministic rank order.
    """
    request_id: str
    ranked_candidates: Tuple[RankedCandidate, ...]

    @property
    def top_candidate(self) -> Optional[Candidate]:
        """Rank #1 candidate according to deterministic comparator, or None if empty."""
        if not self.ranked_candidates:
            return None
        return self.ranked_candidates[0].candidate

    @property
    def top_ranked_candidate(self) -> Optional[RankedCandidate]:
        """Top RankedCandidate with rank, business_rank, and criteria trace."""
        if not self.ranked_candidates:
            return None
        return self.ranked_candidates[0]

    @property
    def is_empty(self) -> bool:
        return len(self.ranked_candidates) == 0

    @property
    def count(self) -> int:
        return len(self.ranked_candidates)

    @property
    def business_tied_at_rank_1(self) -> bool:
        """True if more than one candidate shares business_rank == 1."""
        if len(self.ranked_candidates) < 2:
            return False
        return self.ranked_candidates[0].business_rank == self.ranked_candidates[1].business_rank


# ---------------------------------------------------------------------------
# Hard Gate and Main Ranking Functions
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: Sequence[Candidate],
    desired_completion_date: date,
    request_id: str = "",
) -> RankedCandidateSet:
    """Rank a sequence of rankable candidates according to the contest specification.

    HARD GATE (SAFETY INVARIANT):
    Every candidate MUST satisfy candidate.is_rankable == True.
    Raises ValueError if any unsafe, contractually ineligible, or structurally
    invalid candidate is supplied.

    DOES NOT:
    - Mutate candidate objects.
    - Call an LLM.
    - Write output.csv.
    - Invent scores or arbitrary weights.
    """
    # 1. Hard gate: strictly reject any non-rankable candidate
    for c in candidates:
        if not c.is_rankable or not is_rankable(c):
            raise ValueError(
                f"Candidate {c.candidate_id} rejected by ranking hard gate: "
                f"is_rankable={c.is_rankable}, status={c.status.value}, is_safe={c.is_safe}."
            )

    if not candidates:
        return RankedCandidateSet(request_id=request_id, ranked_candidates=())

    # 2. Sort deterministically using the specification comparator
    cmp_fn = functools.partial(compare_candidates_deterministic, desired_completion_date=desired_completion_date)
    sorted_candidates = sorted(candidates, key=functools.cmp_to_key(cmp_fn))

    # 3. Assign deterministic rank (1..N), business rank (with ties), and build traces
    ranked_list = []
    current_business_rank = 1
    for idx, c in enumerate(sorted_candidates):
        det_rank = idx + 1
        if idx > 0:
            # Check if c ties with previous candidate on business criteria
            prev = sorted_candidates[idx - 1]
            if compare_candidates_business(c, prev, desired_completion_date) != 0:
                current_business_rank = det_rank

        trace = build_criteria_trace(c, desired_completion_date)
        ranked_list.append(
            RankedCandidate(
                candidate=c,
                rank=det_rank,
                business_rank=current_business_rank,
                trace=trace,
            )
        )

    req_id = request_id or (candidates[0].request_id if candidates else "")
    return RankedCandidateSet(
        request_id=req_id,
        ranked_candidates=tuple(ranked_list),
    )


def rank_candidate_set(
    candidate_set: CandidateSet,
    desired_completion_date: date,
) -> RankedCandidateSet:
    """Convenience entry point: extracts rankable_candidates and ranks them.

    Guarantees non-rankable candidates in CandidateSet never enter ranking.
    """
    return rank_candidates(
        candidates=candidate_set.rankable_candidates,
        desired_completion_date=desired_completion_date,
        request_id=candidate_set.request_id,
    )
