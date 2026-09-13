"""Deterministic Spending-Change Optimization Layer for Buy or Wait?

Specification Sources:
  1. problem_statement.md lines 148-161 ("Allowed values / spending_changes_needed"):
     - spending_changes_needed may contain up to three changes separated by |:
         stop:<event_id>
         reduce_to:<event_id>:<new_amount>
     - Only recurring expenses marked as flexible may be changed.
     - Stopping and reducing the same financial event are mutually exclusive.
     - If both types of spending change are required, they must reference different events.
  2. AGENTS.md §6.2 ("Required Output") and §6.3 ("Financial Decision Rules"):
     - spending_changes_needed is none or up to three stop:<event_id> and
       reduce_to:<event_id>:<new_amount> actions.
     - Only non-protected, flexible events in a category the user permits may be changed.
     - Criterion 2 evaluates: "Require no spending changes".

WHAT THIS MODULE DOES:
- Identifies eligible flexible expenses strictly according to dataset/profile policy gates.
- Constructs immutable, typed SpendingChange actions.
- Explores combinations up to MAX_SPENDING_CHANGES (3) deterministically without exponential explosion.
- Applies temporary immutable overlays to future events and canonical events (zero state mutation).
- Re-simulates candidate actions through the sole source-of-truth simulator (simulate_user).
- Minimizes disruption: prefers fewer changes, smaller total spending reduction, fewer stopped
  obligations, earlier impact, and deterministic tie-breaking.
- Evaluates spending-change variants for full_payment, installments, partial_payment, and wait.

WHAT THIS MODULE DOES NOT DO:
- No LLM calls.
- No heuristic float scoring.
- No mutation of canonical ledger, recurrence series, or simulator state.
- No fabrication of amounts, events, or categories.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from enum import Enum
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

from code.canonical import CanonicalEvent, CashImpactType, Direction
from code.models import FinancialProfile, FinancialRequest
from code.recurrence import FutureEvent, RecurrenceSeries
from code.simulator import SimulatedEvent, SimulationResult, simulate_user

MAX_SPENDING_CHANGES: int = 3


class SpendingActionType(str, Enum):
    """Allowed deterministic spending change actions."""
    STOP = "stop"
    REDUCE_TO = "reduce_to"


def format_change_amount(amt: Decimal) -> str:
    """Format monetary amount for contest-compliant reduce_to string.

    Integer amounts (e.g. IDR 665950) are formatted without trailing decimal point.
    Fractional amounts (e.g. USD 23.50) are formatted to 2 decimal places.
    Matches dataset/sample_requests.csv exactly.
    """
    if amt == amt.to_integral():
        return str(int(amt))
    return f"{amt:.2f}"


@dataclass(frozen=True)
class SpendingChange:
    """Immutable representation of a single deterministic spending change action.

    Attributes:
        event_id: Authoritative anchor event ID from the dataset (e.g. 'event_476').
        series_id: Optional recurring series ID for provenance.
        action_type: SpendingActionType.STOP or SpendingActionType.REDUCE_TO.
        original_amount: Pre-change amount (forecast or scheduled).
        modified_amount: Post-change amount (0 for STOP, minimum_allowed_amount for REDUCE_TO).
        effective_date: Earliest date in simulation window affected by this change.
        category: Expense category from the dataset.
        disruption_score: Monetary reduction amount (original_amount - modified_amount).
        human_description: Concise description for auditability and explanation grounding.
    """
    event_id: str
    series_id: Optional[str]
    action_type: SpendingActionType
    original_amount: Decimal
    modified_amount: Decimal
    effective_date: date
    category: str
    disruption_score: Decimal
    human_description: str

    def __post_init__(self) -> None:
        if not isinstance(self.original_amount, Decimal):
            raise TypeError(f"original_amount must be Decimal, got {type(self.original_amount)}")
        if not isinstance(self.modified_amount, Decimal):
            raise TypeError(f"modified_amount must be Decimal, got {type(self.modified_amount)}")
        if not isinstance(self.disruption_score, Decimal):
            raise TypeError(f"disruption_score must be Decimal, got {type(self.disruption_score)}")
        if self.modified_amount < Decimal("0"):
            raise ValueError(f"modified_amount cannot be negative: {self.modified_amount}")
        if self.modified_amount > self.original_amount:
            raise ValueError(
                f"modified_amount ({self.modified_amount}) cannot exceed original_amount ({self.original_amount})"
            )
        if self.action_type == SpendingActionType.STOP and self.modified_amount != Decimal("0"):
            raise ValueError(f"STOP action must have modified_amount=0, got {self.modified_amount}")

    @property
    def amount_saved(self) -> Decimal:
        """Monetary reduction achieved per occurrence."""
        return max(Decimal("0"), self.original_amount - self.modified_amount)

    def to_action_string(self) -> str:
        """Format to official contest specification: stop:<event_id> or reduce_to:<event_id>:<amount>."""
        if self.action_type == SpendingActionType.STOP:
            return f"stop:{self.event_id}"
        elif self.action_type == SpendingActionType.REDUCE_TO:
            amt_str = format_change_amount(self.modified_amount)
            return f"reduce_to:{self.event_id}:{amt_str}"
        raise ValueError(f"Unknown action type: {self.action_type}")


@dataclass(frozen=True)
class SpendingChangeScenario:
    """Immutable combination of 1 to MAX_SPENDING_CHANGES spending change actions.

    Orders actions deterministically and precomputes multi-attribute disruption metrics.
    """
    changes: Tuple[SpendingChange, ...]
    num_changes: int
    total_reduction: Decimal
    num_stopped: int
    earliest_date: date

    @property
    def disruption_tuple(self) -> Tuple[int, Decimal, int, date, Tuple[str, ...]]:
        """Multi-attribute disruption objective for deterministic comparison.

        Priority:
          1. Fewer changed obligations (len(changes) ascending: 1, then 2, then 3).
          2. Smaller total reduction in spending (total_reduction ascending).
          3. Fewer stopped obligations (num_stopped ascending).
          4. Earlier date of impact (earliest_date ascending).
          5. Deterministic tie-breaker (lexicographic action strings).
        """
        action_keys = tuple(c.to_action_string() for c in self.changes)
        return (
            self.num_changes,
            self.total_reduction,
            self.num_stopped,
            self.earliest_date,
            action_keys,
        )

    @property
    def action_string(self) -> str:
        """Contest-compliant pipe-separated string (e.g. 'stop:event_1815|reduce_to:event_1816:23.50')."""
        return "|".join(c.to_action_string() for c in self.changes)


# ---------------------------------------------------------------------------
# Phase 2: Policy Eligibility Gate
# ---------------------------------------------------------------------------

def identify_eligible_spending_actions(
    user_id: str,
    profile: FinancialProfile,
    series_list: Sequence[RecurrenceSeries],
    future_events: Sequence[FutureEvent],
    canonical_events: Sequence[CanonicalEvent],
    request_date: date,
    simulation_end: date,
) -> Tuple[SpendingChange, ...]:
    """Deterministically identify all eligible spending-change options for a user.

    Strict eligibility policy gates (Phase 2):
      - Must belong to user_id.
      - Must be an outflow / expense obligation.
      - Must have active occurrences within the forecast window [request_date, simulation_end].
      - Must NOT be in profile.expense_categories_to_protect.
      - Must NOT be marked is_protected.
      - Must NOT be cancelled (is_cancelled == False).
      - Must have numerically usable positive amount (> 0).
      - Must have authoritative flexibility in ('stoppable', 'reducible', 'reducible_or_stoppable')
        (never inferred from text).
      - Category must be explicitly permitted by user profile:
          * STOP: category in profile.expense_categories_user_is_willing_to_stop AND
            flexibility in ('stoppable', 'reducible_or_stoppable').
          * REDUCE_TO: category in profile.expense_categories_user_is_willing_to_reduce AND
            flexibility in ('reducible', 'reducible_or_stoppable') AND
            minimum_allowed_amount is not None AND 0 <= minimum_allowed_amount < original_amount.

    Returns:
        Immutable tuple of eligible SpendingChange actions, sorted deterministically.
    """
    actions: List[SpendingChange] = []
    seen_event_action_pairs: Set[Tuple[str, SpendingActionType]] = set()

    protected_cats = set(profile.expense_categories_to_protect)
    stop_cats = set(profile.expense_categories_user_is_willing_to_stop)
    reduce_cats = set(profile.expense_categories_user_is_willing_to_reduce)

    # 1. Evaluate recurring series with active occurrences in horizon
    for s in series_list:
        if s.user_id != user_id or s.is_cancelled or s.is_protected:
            continue
        if s.direction != Direction.OUTFLOW:
            continue
        if s.category in protected_cats:
            continue
        if s.forecast_amount <= Decimal("0"):
            continue

        flex = s.flexibility.strip().lower() if s.flexibility else "fixed"
        if flex not in ("stoppable", "reducible", "reducible_or_stoppable"):
            continue

        # Check for active occurrences in [request_date, simulation_end]
        matching_future_dates = [
            fe.effective_date
            for fe in future_events
            if fe.user_id == user_id
            and fe.series_id == s.series_id
            and request_date <= fe.effective_date <= simulation_end
        ]
        if not matching_future_dates:
            continue
        first_date = min(matching_future_dates)

        target_eid = s.anchor_event_id or s.series_id

        # Candidate Action A: REDUCE_TO
        can_reduce = (
            s.category in reduce_cats
            and flex in ("reducible", "reducible_or_stoppable")
            and s.minimum_allowed_amount is not None
            and Decimal("0") <= s.minimum_allowed_amount < s.forecast_amount
        )
        if can_reduce:
            key = (target_eid, SpendingActionType.REDUCE_TO)
            if key not in seen_event_action_pairs:
                seen_event_action_pairs.add(key)
                min_amt = s.minimum_allowed_amount
                saving = s.forecast_amount - min_amt
                actions.append(SpendingChange(
                    event_id=target_eid,
                    series_id=s.series_id,
                    action_type=SpendingActionType.REDUCE_TO,
                    original_amount=s.forecast_amount,
                    modified_amount=min_amt,
                    effective_date=first_date,
                    category=s.category,
                    disruption_score=saving,
                    human_description=f"Reduce {s.description or s.category} to {format_change_amount(min_amt)}",
                ))

        # Candidate Action B: STOP
        can_stop = (
            s.category in stop_cats
            and flex in ("stoppable", "reducible_or_stoppable")
        )
        if can_stop:
            key = (target_eid, SpendingActionType.STOP)
            if key not in seen_event_action_pairs:
                seen_event_action_pairs.add(key)
                actions.append(SpendingChange(
                    event_id=target_eid,
                    series_id=s.series_id,
                    action_type=SpendingActionType.STOP,
                    original_amount=s.forecast_amount,
                    modified_amount=Decimal("0"),
                    effective_date=first_date,
                    category=s.category,
                    disruption_score=s.forecast_amount,
                    human_description=f"Stop {s.description or s.category}",
                ))

    # 2. Evaluate explicit scheduled events in canonical ledger (if any future flexible ones exist)
    for ce in canonical_events:
        if ce.user_id != user_id:
            continue
        if ce.effective_date < request_date or ce.effective_date > simulation_end:
            continue
        if not ce.is_numerically_usable_cash_event or ce.direction != Direction.OUTFLOW:
            continue
        if ce.status != "scheduled":
            continue
        if ce.category in protected_cats:
            continue

        ce_amt = ce.get_usable_cash_amount()
        if ce_amt <= Decimal("0"):
            continue

        flex = ce.flexibility.strip().lower() if ce.flexibility else "fixed"
        if flex not in ("stoppable", "reducible", "reducible_or_stoppable"):
            continue

        can_reduce = (
            ce.category in reduce_cats
            and flex in ("reducible", "reducible_or_stoppable")
            and ce.minimum_allowed_amount_home is not None
            and Decimal("0") <= ce.minimum_allowed_amount_home < ce_amt
        )
        if can_reduce:
            key = (ce.event_id, SpendingActionType.REDUCE_TO)
            if key not in seen_event_action_pairs:
                seen_event_action_pairs.add(key)
                min_amt = ce.minimum_allowed_amount_home
                saving = ce_amt - min_amt
                actions.append(SpendingChange(
                    event_id=ce.event_id,
                    series_id=None,
                    action_type=SpendingActionType.REDUCE_TO,
                    original_amount=ce_amt,
                    modified_amount=min_amt,
                    effective_date=ce.effective_date,
                    category=ce.category,
                    disruption_score=saving,
                    human_description=f"Reduce {ce.description or ce.category} to {format_change_amount(min_amt)}",
                ))

        can_stop = (
            ce.category in stop_cats
            and flex in ("stoppable", "reducible_or_stoppable")
        )
        if can_stop:
            key = (ce.event_id, SpendingActionType.STOP)
            if key not in seen_event_action_pairs:
                seen_event_action_pairs.add(key)
                actions.append(SpendingChange(
                    event_id=ce.event_id,
                    series_id=None,
                    action_type=SpendingActionType.STOP,
                    original_amount=ce_amt,
                    modified_amount=Decimal("0"),
                    effective_date=ce.effective_date,
                    category=ce.category,
                    disruption_score=ce_amt,
                    human_description=f"Stop {ce.description or ce.category}",
                ))

    # Stable deterministic ordering: (effective_date, event_id, action_type, modified_amount)
    actions.sort(key=lambda a: (a.effective_date, a.event_id, a.action_type.value, a.modified_amount))
    return tuple(actions)


# ---------------------------------------------------------------------------
# Phase 4 & 5: Combinatorial Scenario Generation
# ---------------------------------------------------------------------------

def generate_spending_change_scenarios(
    eligible_actions: Sequence[SpendingChange],
    max_changes: int = MAX_SPENDING_CHANGES,
) -> List[SpendingChangeScenario]:
    """Generate all valid spending-change combinations up to max_changes.

    Guarantees:
      - Size bounded to [1, max_changes] (max 3 per contest rule).
      - Mutual exclusivity on event_id: no combination targets the same event twice.
      - Deterministic ordering sorted by the multi-attribute disruption objective:
          1. Fewer changed obligations.
          2. Smaller total reduction in spending.
          3. Fewer stopped obligations.
          4. Earlier date of impact.
          5. Lexicographic action strings tie-breaker.
    """
    scenarios: List[SpendingChangeScenario] = []

    for k in range(1, max_changes + 1):
        for combo in combinations(eligible_actions, k):
            # Enforce mutual exclusivity on event_id
            event_ids = set(a.event_id for a in combo)
            if len(event_ids) != len(combo):
                continue

            # Deterministic sorting within combination: chronological, then event_id
            sorted_combo = tuple(sorted(combo, key=lambda a: (a.effective_date, a.event_id, a.action_type.value)))

            num_changes = len(sorted_combo)
            total_reduction = sum((a.amount_saved for a in sorted_combo), Decimal("0"))
            num_stopped = sum(1 for a in sorted_combo if a.action_type == SpendingActionType.STOP)
            earliest_date = min(a.effective_date for a in sorted_combo)

            scenarios.append(SpendingChangeScenario(
                changes=sorted_combo,
                num_changes=num_changes,
                total_reduction=total_reduction,
                num_stopped=num_stopped,
                earliest_date=earliest_date,
            ))

    # Deterministic sort across all scenarios
    scenarios.sort(key=lambda s: s.disruption_tuple)
    return scenarios


# ---------------------------------------------------------------------------
# Phase 6: Temporary Scenario Overlay Application
# ---------------------------------------------------------------------------

def apply_spending_changes_overlay(
    canonical_events: Sequence[CanonicalEvent],
    future_events: Sequence[FutureEvent],
    changes: Sequence[SpendingChange],
) -> Tuple[Tuple[CanonicalEvent, ...], Tuple[FutureEvent, ...]]:
    """Apply an immutable temporary scenario overlay to canonical and future events.

    Pure functional transformation:
      - Does NOT mutate any original CanonicalEvent or FutureEvent.
      - Does NOT mutate the CanonicalLedger or Recurrence models.
      - Omits stopped obligations from the simulation timeline.
      - Modifies amount_home on reduced obligations.
      - Preserves all untouched events bit-for-bit.
    """
    if not changes:
        return tuple(canonical_events), tuple(future_events)

    stop_event_ids: Set[str] = {c.event_id for c in changes if c.action_type == SpendingActionType.STOP}
    stop_series_ids: Set[str] = {c.series_id for c in changes if c.action_type == SpendingActionType.STOP and c.series_id}

    reduce_by_event: Dict[str, Decimal] = {
        c.event_id: c.modified_amount
        for c in changes
        if c.action_type == SpendingActionType.REDUCE_TO
    }

    # Overlay FutureEvent occurrences
    overlaid_future: List[FutureEvent] = []
    for fe in future_events:
        # Check if stopped
        if (
            fe.anchor_event_id in stop_event_ids
            or fe.event_id in stop_event_ids
            or (fe.series_id and fe.series_id in stop_series_ids)
        ):
            continue

        # Check if reduced
        matched_reduced_amt = None
        if fe.anchor_event_id and fe.anchor_event_id in reduce_by_event:
            matched_reduced_amt = reduce_by_event[fe.anchor_event_id]
        elif fe.event_id in reduce_by_event:
            matched_reduced_amt = reduce_by_event[fe.event_id]

        if matched_reduced_amt is not None:
            overlaid_future.append(replace(fe, amount_home=matched_reduced_amt))
        else:
            overlaid_future.append(fe)

    # Overlay CanonicalEvent occurrences
    overlaid_canonical: List[CanonicalEvent] = []
    for ce in canonical_events:
        # Check if stopped
        if ce.event_id in stop_event_ids:
            continue

        # Check if reduced
        if ce.event_id in reduce_by_event:
            reduced_amt = reduce_by_event[ce.event_id]
            overlaid_canonical.append(replace(ce, amount_home=reduced_amt))
        else:
            overlaid_canonical.append(ce)

    return tuple(overlaid_canonical), tuple(overlaid_future)


# ---------------------------------------------------------------------------
# Phase 7 & 8: Re-Simulation & Minimal Disruption Optimization
# ---------------------------------------------------------------------------

def optimize_spending_changes_for_candidate(
    user_id: str,
    simulation_start: date,
    simulation_end: date,
    profile: FinancialProfile,
    canonical_events: Sequence[CanonicalEvent],
    future_events: Sequence[FutureEvent],
    scenarios: Sequence[SpendingChangeScenario],
    candidate_events: Sequence[SimulatedEvent],
) -> Optional[SpendingChangeScenario]:
    """Find the minimal-disruption spending-change scenario that makes candidate_events safe.

    Execution:
      - Iterates through scenarios in strict disruption-ascending order.
      - Applies temporary overlay without mutating base data.
      - Simulates trajectory with simulate_user.
      - Enforces the same safety floor, overdraft rules, and scheduled obligations.
      - Returns the first (lowest disruption) scenario that passes all safety checks.
      - Returns None if no valid scenario makes the candidate safe.
    """
    safety_floor = profile.minimum_balance_to_keep

    for scenario in scenarios:
        overlaid_can, overlaid_fut = apply_spending_changes_overlay(
            canonical_events=canonical_events,
            future_events=future_events,
            changes=scenario.changes,
        )

        sim_result: SimulationResult = simulate_user(
            user_id=user_id,
            simulation_start=simulation_start,
            simulation_end=simulation_end,
            canonical_events=overlaid_can,
            future_events=overlaid_fut,
            profile=profile,
            additional_events=candidate_events,
        )

        # Enforce all safety invariants:
        # 1. Not breached according to simulator flag
        # 2. Minimum cash >= safety_floor throughout window
        # 3. No overdraft breaches
        if (
            not sim_result.is_safety_floor_breached
            and sim_result.minimum_projected_available_cash >= safety_floor
            and len(sim_result.obligation_breaches) == 0
        ):
            return scenario

    return None
