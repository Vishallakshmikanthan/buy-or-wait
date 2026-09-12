"""Comprehensive Unit and Property Tests for Deterministic Ranking Engine.

Covers:
  - Prompt 11 requirements 1-26
  - Hard gate and non-rankable candidate rejection
  - Individual criteria (1 through 6)
  - Strict lexicographic precedence
  - Synthetic cases A through J
  - Critical edge cases from Section 13 and Section 14
  - Properties 1 through 7
  - Permutation invariance and repeated determinism
  - Immutability and zero mutation
  - Traceability completeness
"""

import copy
import random
import unittest
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Optional, Tuple

from code.candidate_generation import (
    Candidate,
    CandidatePayment,
    CandidateProvenance,
    CandidateSet,
    CandidateStatus,
    CandidateType,
    is_rankable,
)
from code.ranking import (
    RankingCriteriaTrace,
    RankedCandidate,
    RankedCandidateSet,
    build_criteria_trace,
    compare_candidates_business,
    compare_candidates_deterministic,
    deterministic_ranking_key,
    eval_criterion_1_deadline_met,
    eval_criterion_2_no_spending_changes,
    eval_criterion_3_total_amount_paid,
    eval_criterion_4_first_payment_date,
    eval_criterion_5_number_of_payments,
    eval_criterion_6_payment_option_id,
    rank_candidates,
    rank_candidate_set,
)


def _make_candidate(
    candidate_id: str = "c_test",
    request_id: str = "req_test",
    candidate_type: CandidateType = CandidateType.INSTALLMENT_PLAN,
    total_amount_paid: Decimal = Decimal("100.00"),
    first_payment_date: Optional[date] = date(2026, 3, 1),
    number_of_payments: int = 2,
    completion_date: Optional[date] = date(2026, 4, 1),
    spending_changes: Tuple[str, ...] = (),
    source_payment_option_id: Optional[str] = "payment_option_01",
    is_safe: bool = True,
    status: CandidateStatus = CandidateStatus.ELIGIBLE_AND_SAFE,
    financing_fee: Decimal = Decimal("0.00"),
    payment_method: Optional[str] = None,
) -> Candidate:
    """Helper to build a valid, strictly well-formed Candidate for ranking tests."""
    schedule: List[CandidatePayment] = []
    if status == CandidateStatus.ELIGIBLE_AND_SAFE and is_safe:
        # Construct valid schedule matching total and count
        if number_of_payments == 1:
            schedule.append(
                CandidatePayment(
                    payment_date=first_payment_date or date(2026, 3, 1),
                    amount=total_amount_paid,
                    sequence_number=1,
                )
            )
        else:
            part = (total_amount_paid / Decimal(number_of_payments)).quantize(Decimal("0.01"))
            running = Decimal("0")
            for i in range(1, number_of_payments):
                pay_d = (first_payment_date or date(2026, 3, 1)) + timedelta(days=30 * (i - 1))
                schedule.append(CandidatePayment(payment_date=pay_d, amount=part, sequence_number=i))
                running += part
            last_part = total_amount_paid - running
            schedule.append(
                CandidatePayment(
                    payment_date=completion_date or date(2026, 4, 1),
                    amount=last_part,
                    sequence_number=number_of_payments,
                )
            )

    return Candidate(
        candidate_id=candidate_id,
        request_id=request_id,
        candidate_type=candidate_type,
        payment_method=payment_method or candidate_type.value,
        total_amount_paid=total_amount_paid,
        first_payment_date=first_payment_date,
        final_payment_date=completion_date,
        number_of_payments=number_of_payments if schedule else 0,
        payment_schedule=tuple(schedule),
        financing_fee=financing_fee,
        spending_changes=spending_changes,
        source_payment_option_id=source_payment_option_id,
        is_safe=is_safe,
        completion_date=completion_date,
        feasibility_reason=None if is_safe else "Unsafe in simulation",
        status=status,
        provenance=CandidateProvenance(
            source="test_provenance",
            source_payment_option_id=source_payment_option_id,
            safe_to_pay_certificate_id=None,
            earliest_full_payment_date=completion_date,
        ),
    )


# ===========================================================================
# 1. Hard Gate Tests
# ===========================================================================

class TestHardGate(unittest.TestCase):
    """Hard gate verifies only is_rankable==True candidates can be ranked."""

    def test_reject_structurally_invalid_candidate(self) -> None:
        c = Candidate(
            candidate_id="inv_1",
            request_id="req_1",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            payment_method="installments",
            total_amount_paid=Decimal("100"),
            first_payment_date=date(2026, 1, 1),
            final_payment_date=None,
            number_of_payments=2,
            payment_schedule=(),
            financing_fee=Decimal("0"),
            spending_changes=(),
            source_payment_option_id="opt_bad",
            is_safe=False,
            completion_date=None,
            feasibility_reason="Schema mismatch",
            status=CandidateStatus.STRUCTURALLY_INVALID,
            provenance=CandidateProvenance("test", "opt_bad", None, None),
        )
        with self.assertRaises(ValueError) as ctx:
            rank_candidates([c], desired_completion_date=date(2026, 6, 1))
        self.assertIn("rejected by ranking hard gate", str(ctx.exception))

    def test_reject_contractually_ineligible_candidate(self) -> None:
        c = Candidate(
            candidate_id="inelig_1",
            request_id="req_1",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            payment_method="installments",
            total_amount_paid=Decimal("100"),
            first_payment_date=date(2026, 1, 1),
            final_payment_date=None,
            number_of_payments=2,
            payment_schedule=(),
            financing_fee=Decimal("0"),
            spending_changes=(),
            source_payment_option_id="opt_pref",
            is_safe=False,
            completion_date=None,
            feasibility_reason="User preferences exclude installments",
            status=CandidateStatus.CONTRACTUALLY_INELIGIBLE,
            provenance=CandidateProvenance("test", "opt_pref", None, None),
        )
        with self.assertRaises(ValueError) as ctx:
            rank_candidates([c], desired_completion_date=date(2026, 6, 1))
        self.assertIn("rejected by ranking hard gate", str(ctx.exception))

    def test_reject_structurally_valid_unsafe_candidate(self) -> None:
        c = Candidate(
            candidate_id="unsafe_1",
            request_id="req_1",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            payment_method="installments",
            total_amount_paid=Decimal("100"),
            first_payment_date=date(2026, 1, 1),
            final_payment_date=None,
            number_of_payments=2,
            payment_schedule=(),
            financing_fee=Decimal("0"),
            spending_changes=(),
            source_payment_option_id="opt_unsafe",
            is_safe=False,
            completion_date=None,
            feasibility_reason="Breaches minimum balance floor",
            status=CandidateStatus.STRUCTURALLY_VALID_UNSAFE,
            provenance=CandidateProvenance("test", "opt_unsafe", None, None),
        )
        with self.assertRaises(ValueError) as ctx:
            rank_candidates([c], desired_completion_date=date(2026, 6, 1))
        self.assertIn("rejected by ranking hard gate", str(ctx.exception))

    def test_empty_candidates_returns_empty_ranked_set(self) -> None:
        res = rank_candidates([], desired_completion_date=date(2026, 6, 1))
        self.assertTrue(res.is_empty)
        self.assertEqual(res.count, 0)
        self.assertIsNone(res.top_candidate)

    def test_rank_candidate_set_filters_to_rankable_only(self) -> None:
        c_safe = _make_candidate("c_safe", is_safe=True, status=CandidateStatus.ELIGIBLE_AND_SAFE)
        c_unsafe = Candidate(
            candidate_id="c_unsafe",
            request_id="req_test",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            payment_method="installments",
            total_amount_paid=Decimal("100"),
            first_payment_date=date(2026, 1, 1),
            final_payment_date=None,
            number_of_payments=2,
            payment_schedule=(),
            financing_fee=Decimal("0"),
            spending_changes=(),
            source_payment_option_id="opt_u",
            is_safe=False,
            completion_date=None,
            feasibility_reason="unsafe",
            status=CandidateStatus.STRUCTURALLY_VALID_UNSAFE,
            provenance=CandidateProvenance("test", "opt_u", None, None),
        )
        cset = CandidateSet(request_id="req_test", candidates=(c_safe, c_unsafe))
        res = rank_candidate_set(cset, desired_completion_date=date(2026, 6, 1))
        self.assertEqual(res.count, 1)
        self.assertEqual(res.top_candidate.candidate_id, "c_safe")


# ===========================================================================
# 2. Individual Criteria and Boundary Tests
# ===========================================================================

class TestIndividualCriteria(unittest.TestCase):
    """Test individual criteria evaluators and boundary semantics."""

    def test_criterion_1_deadline_inclusive_boundary(self) -> None:
        deadline = date(2026, 4, 15)
        c_before = _make_candidate("c_before", completion_date=date(2026, 4, 14))
        c_exact = _make_candidate("c_exact", completion_date=date(2026, 4, 15))
        c_after = _make_candidate("c_after", completion_date=date(2026, 4, 16))

        self.assertTrue(eval_criterion_1_deadline_met(c_before, deadline))
        self.assertTrue(eval_criterion_1_deadline_met(c_exact, deadline))  # INCLUSIVE
        self.assertFalse(eval_criterion_1_deadline_met(c_after, deadline))

        res = rank_candidates([c_after, c_exact, c_before], desired_completion_date=deadline)
        self.assertIn(res.ranked_candidates[0].candidate.candidate_id, ("c_before", "c_exact"))
        self.assertEqual(res.ranked_candidates[2].candidate.candidate_id, "c_after")

    def test_criterion_2_spending_changes_semantics(self) -> None:
        c_no_sc = _make_candidate("c_none", spending_changes=())
        c_one_sc = _make_candidate("c_one", spending_changes=("stop:ev_01",))
        c_two_sc = _make_candidate("c_two", spending_changes=("stop:ev_01", "reduce_to:ev_02:100"))

        self.assertTrue(eval_criterion_2_no_spending_changes(c_no_sc))
        self.assertFalse(eval_criterion_2_no_spending_changes(c_one_sc))
        self.assertFalse(eval_criterion_2_no_spending_changes(c_two_sc))

        res = rank_candidates([c_one_sc, c_no_sc], desired_completion_date=date(2026, 6, 1))
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_none")

    def test_criterion_3_total_amount_paid_decimal_exact(self) -> None:
        c1 = _make_candidate("c_cheap", total_amount_paid=Decimal("100.00"))
        c2 = _make_candidate("c_exp", total_amount_paid=Decimal("100.01"))

        res = rank_candidates([c2, c1], desired_completion_date=date(2026, 6, 1))
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_cheap")
        self.assertEqual(res.ranked_candidates[1].candidate.candidate_id, "c_exp")

    def test_criterion_4_start_earlier_semantics(self) -> None:
        c_early = _make_candidate("c_early", first_payment_date=date(2026, 3, 1))
        c_late = _make_candidate("c_late", first_payment_date=date(2026, 3, 15))

        res = rank_candidates([c_late, c_early], desired_completion_date=date(2026, 6, 1))
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_early")

    def test_criterion_5_fewer_payments_semantics(self) -> None:
        c_few = _make_candidate("c_few", number_of_payments=2)
        c_many = _make_candidate("c_many", number_of_payments=6)

        res = rank_candidates([c_many, c_few], desired_completion_date=date(2026, 6, 1))
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_few")

    def test_criterion_6_payment_option_id_tie_breaker(self) -> None:
        c_opt1 = _make_candidate(
            "c_opt1",
            source_payment_option_id="payment_option_01",
            total_amount_paid=Decimal("100"),
            first_payment_date=date(2026, 3, 1),
            number_of_payments=2,
            completion_date=date(2026, 4, 1),
        )
        c_opt2 = _make_candidate(
            "c_opt2",
            source_payment_option_id="payment_option_02",
            total_amount_paid=Decimal("100"),
            first_payment_date=date(2026, 3, 1),
            number_of_payments=2,
            completion_date=date(2026, 4, 1),
        )

        res = rank_candidates([c_opt2, c_opt1], desired_completion_date=date(2026, 6, 1))
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_opt1")
        self.assertEqual(res.ranked_candidates[1].candidate.candidate_id, "c_opt2")


# ===========================================================================
# 3. Critical Edge Cases (Sections 13 and 14)
# ===========================================================================

class TestCriticalEdgeCases(unittest.TestCase):
    """Section 13 and Section 14 critical edge cases."""

    def test_section_13_criterion_2_strictly_precedes_criterion_3(self) -> None:
        """Section 13:
        Candidate A: deadline met, no spending changes, total paid = 100, first payment = D, payments = 2
        Candidate B: deadline met, spending changes required, total paid = 90, first payment = D, payments = 1
        Candidate A MUST outrank B if criterion 2 precedes criterion 3.
        Lower total cost must NOT override spending changes!
        """
        D = date(2026, 3, 1)
        cand_a = _make_candidate(
            candidate_id="cand_A",
            total_amount_paid=Decimal("100.00"),
            spending_changes=(),  # no spending changes
            first_payment_date=D,
            number_of_payments=2,
            completion_date=D + timedelta(days=30),
        )
        cand_b = _make_candidate(
            candidate_id="cand_B",
            total_amount_paid=Decimal("90.00"),  # cheaper!
            spending_changes=("stop:event_99",),  # but requires changes!
            first_payment_date=D,
            number_of_payments=1,
            completion_date=D,
        )
        res = rank_candidates([cand_b, cand_a], desired_completion_date=date(2026, 6, 1))
        self.assertEqual(
            res.ranked_candidates[0].candidate.candidate_id,
            "cand_A",
            "Candidate A (no changes, higher cost) MUST outrank Candidate B (changes, lower cost).",
        )

    def test_section_14_criterion_4_strictly_precedes_criterion_5(self) -> None:
        """Section 14:
        Candidate A: deadline met, no spending changes, total = 100, first payment = D+10, payments = 1
        Candidate B: deadline met, no spending changes, total = 100, first payment = D, payments = 2
        B MUST outrank A because earlier start is criterion 4 and occurs before number of payments criterion 5.
        """
        D = date(2026, 3, 1)
        cand_a = _make_candidate(
            candidate_id="cand_A",
            total_amount_paid=Decimal("100.00"),
            spending_changes=(),
            first_payment_date=D + timedelta(days=10),  # starts later (D+10)
            number_of_payments=1,  # fewer payments!
            completion_date=D + timedelta(days=10),
        )
        cand_b = _make_candidate(
            candidate_id="cand_B",
            total_amount_paid=Decimal("100.00"),
            spending_changes=(),
            first_payment_date=D,  # starts earlier (D)
            number_of_payments=2,  # more payments!
            completion_date=D + timedelta(days=30),
        )
        res = rank_candidates([cand_a, cand_b], desired_completion_date=date(2026, 6, 1))
        self.assertEqual(
            res.ranked_candidates[0].candidate.candidate_id,
            "cand_B",
            "Candidate B (earlier start, more payments) MUST outrank Candidate A (later start, fewer payments).",
        )


# ===========================================================================
# 4. Synthetic Cases A through J (Section 12)
# ===========================================================================

class TestSyntheticCasesAThroughJ(unittest.TestCase):
    """All 10 required synthetic scenarios from Section 12."""

    def setUp(self) -> None:
        self.req_d = date(2026, 3, 1)
        self.deadline = date(2026, 5, 1)

    def test_case_a_full_payment_vs_installment(self) -> None:
        """A. Full payment (1 payment, req_date) vs installment (2 payments, req_date, 0 fee).
        Full payment wins on Criterion 5 (fewer payments: 1 < 2).
        """
        c_full = _make_candidate(
            "full",
            candidate_type=CandidateType.FULL_PAYMENT,
            total_amount_paid=Decimal("100"),
            first_payment_date=self.req_d,
            number_of_payments=1,
            completion_date=self.req_d,
            source_payment_option_id=None,
        )
        c_inst = _make_candidate(
            "inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("100"),
            first_payment_date=self.req_d,
            number_of_payments=2,
            completion_date=self.req_d + timedelta(days=30),
            source_payment_option_id="opt_1",
        )
        res = rank_candidates([c_inst, c_full], desired_completion_date=self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "full")

    def test_case_b_full_payment_vs_wait(self) -> None:
        """B. Full payment (starts req_date) vs wait (starts later).
        Full payment wins on Criterion 4 (starts earlier).
        """
        c_full = _make_candidate(
            "full",
            candidate_type=CandidateType.FULL_PAYMENT,
            total_amount_paid=Decimal("100"),
            first_payment_date=self.req_d,
            number_of_payments=1,
            completion_date=self.req_d,
            source_payment_option_id=None,
        )
        c_wait = _make_candidate(
            "wait",
            candidate_type=CandidateType.WAIT,
            total_amount_paid=Decimal("100"),
            first_payment_date=self.req_d + timedelta(days=20),
            number_of_payments=1,
            completion_date=self.req_d + timedelta(days=20),
            source_payment_option_id=None,
        )
        res = rank_candidates([c_wait, c_full], desired_completion_date=self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "full")

    def test_case_c_wait_vs_installment(self) -> None:
        """C. Wait (no fee, starts D+20) vs installment with financing fee (fee=10, starts req_date).
        Wait wins on Criterion 3 (lower total amount paid: 100 < 110).
        """
        c_wait = _make_candidate(
            "wait",
            candidate_type=CandidateType.WAIT,
            total_amount_paid=Decimal("100"),
            first_payment_date=self.req_d + timedelta(days=20),
            number_of_payments=1,
            completion_date=self.req_d + timedelta(days=20),
            source_payment_option_id=None,
        )
        c_inst = _make_candidate(
            "inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("110"),  # higher cost
            first_payment_date=self.req_d,
            number_of_payments=3,
            completion_date=self.req_d + timedelta(days=60),
            source_payment_option_id="opt_fee",
        )
        res = rank_candidates([c_inst, c_wait], desired_completion_date=self.deadline + timedelta(days=30))
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "wait")

    def test_case_d_partial_vs_installment(self) -> None:
        """D. Partial (2 payments, 0 fee) vs installment (3 payments, 0 fee).
        Partial payment wins on Criterion 5 (fewer payments: 2 < 3).
        """
        c_part = _make_candidate(
            "partial",
            candidate_type=CandidateType.PARTIAL_PAYMENT,
            total_amount_paid=Decimal("100"),
            first_payment_date=self.req_d,
            number_of_payments=2,
            completion_date=self.req_d + timedelta(days=30),
            source_payment_option_id=None,
        )
        c_inst = _make_candidate(
            "inst",
            candidate_type=CandidateType.INSTALLMENT_PLAN,
            total_amount_paid=Decimal("100"),
            first_payment_date=self.req_d,
            number_of_payments=3,
            completion_date=self.req_d + timedelta(days=30),
            source_payment_option_id="opt_3p",
        )
        res = rank_candidates([c_inst, c_part], desired_completion_date=self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "partial")

    def test_case_e_two_installment_options(self) -> None:
        """E. Two installment options differing in total cost."""
        c1 = _make_candidate("opt_cheap", total_amount_paid=Decimal("100"), source_payment_option_id="opt_cheap")
        c2 = _make_candidate("opt_exp", total_amount_paid=Decimal("105"), source_payment_option_id="opt_exp")
        res = rank_candidates([c2, c1], desired_completion_date=self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "opt_cheap")

    def test_case_f_two_identical_schedules_different_option_id(self) -> None:
        """F. Two identical schedules: lowest payment_option_id wins."""
        c1 = _make_candidate("c_alpha", source_payment_option_id="payment_option_01")
        c2 = _make_candidate("c_beta", source_payment_option_id="payment_option_02")
        res = rank_candidates([c2, c1], desired_completion_date=self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_alpha")
        self.assertEqual(res.ranked_candidates[1].candidate.candidate_id, "c_beta")

    def test_case_g_same_payment_date_different_payment_count(self) -> None:
        """G. Same payment date, different payment count: fewer payments wins (Criterion 5)."""
        c1 = _make_candidate("c_2pay", number_of_payments=2, first_payment_date=self.req_d)
        c2 = _make_candidate("c_4pay", number_of_payments=4, first_payment_date=self.req_d)
        res = rank_candidates([c2, c1], desired_completion_date=self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_2pay")

    def test_case_h_same_total_cost_different_first_payment_date(self) -> None:
        """H. Same total cost, different first payment date: earlier start wins (Criterion 4)."""
        c1 = _make_candidate("c_early", first_payment_date=self.req_d)
        c2 = _make_candidate("c_later", first_payment_date=self.req_d + timedelta(days=7))
        res = rank_candidates([c2, c1], desired_completion_date=self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_early")

    def test_case_i_deadline_met_vs_deadline_missed(self) -> None:
        """I. Deadline-met vs deadline-missed: deadline-met wins (Criterion 1)."""
        c_on_time = _make_candidate("c_ontime", completion_date=self.deadline)
        c_late = _make_candidate("c_late", completion_date=self.deadline + timedelta(days=1))
        res = rank_candidates([c_late, c_on_time], desired_completion_date=self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_ontime")

    def test_case_j_spending_change_vs_no_spending_change(self) -> None:
        """J. Spending-change vs no-spending-change: no-change wins (Criterion 2)."""
        c_no = _make_candidate("c_no_sc", spending_changes=())
        c_sc = _make_candidate("c_sc", spending_changes=("reduce_to:dining:50",))
        res = rank_candidates([c_sc, c_no], desired_completion_date=self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c_no_sc")


# ===========================================================================
# 5. Seven Property Tests (Section 22)
# ===========================================================================

class TestRankingProperties(unittest.TestCase):
    """Verification of all 7 mandatory mathematical properties from Section 22."""

    def setUp(self) -> None:
        self.req_d = date(2026, 1, 1)
        self.deadline = date(2026, 4, 1)

    def test_property_1_input_permutation_invariance(self) -> None:
        """PROPERTY 1: Input permutation does not change ranking."""
        cands = [
            _make_candidate(f"c_{i}", total_amount_paid=Decimal(str(100 + i * 5)), first_payment_date=self.req_d + timedelta(days=i))
            for i in range(10)
        ]
        baseline = rank_candidates(cands, self.deadline)
        baseline_order = [rc.candidate.candidate_id for rc in baseline.ranked_candidates]

        rng = random.Random(42)
        for _ in range(20):
            shuffled = list(cands)
            rng.shuffle(shuffled)
            res = rank_candidates(shuffled, self.deadline)
            perm_order = [rc.candidate.candidate_id for rc in res.ranked_candidates]
            self.assertEqual(baseline_order, perm_order)

    def test_property_2_improving_criterion_cannot_worsen_rank(self) -> None:
        """PROPERTY 2: Improving criterion k while holding higher equal cannot worsen rank."""
        c1 = _make_candidate("c1", total_amount_paid=Decimal("150.00"), first_payment_date=self.req_d + timedelta(days=5))
        c2 = _make_candidate("c2", total_amount_paid=Decimal("120.00"), first_payment_date=self.req_d + timedelta(days=2))
        c3 = _make_candidate("c3", total_amount_paid=Decimal("110.00"), first_payment_date=self.req_d)

        initial = rank_candidates([c1, c2, c3], self.deadline)
        initial_rank_c1 = next(rc.rank for rc in initial.ranked_candidates if rc.candidate.candidate_id == "c1")

        # Improve c1 on Criterion 3 (lower total cost from 150 to 115)
        c1_improved = _make_candidate("c1", total_amount_paid=Decimal("115.00"), first_payment_date=self.req_d + timedelta(days=5))
        improved = rank_candidates([c1_improved, c2, c3], self.deadline)
        new_rank_c1 = next(rc.rank for rc in improved.ranked_candidates if rc.candidate.candidate_id == "c1")

        self.assertLessEqual(new_rank_c1, initial_rank_c1)

    def test_property_3_worsening_criterion_cannot_improve_rank(self) -> None:
        """PROPERTY 3: Worsening criterion k while holding higher equal cannot improve rank."""
        c1 = _make_candidate("c1", total_amount_paid=Decimal("100.00"), number_of_payments=2)
        c2 = _make_candidate("c2", total_amount_paid=Decimal("100.00"), number_of_payments=3)

        initial = rank_candidates([c1, c2], self.deadline)
        initial_rank_c1 = next(rc.rank for rc in initial.ranked_candidates if rc.candidate.candidate_id == "c1")

        # Worsen c1 on Criterion 5 (payments from 2 to 4)
        c1_worse = _make_candidate("c1", total_amount_paid=Decimal("100.00"), number_of_payments=4)
        worsened = rank_candidates([c1_worse, c2], self.deadline)
        new_rank_c1 = next(rc.rank for rc in worsened.ranked_candidates if rc.candidate.candidate_id == "c1")

        self.assertGreaterEqual(new_rank_c1, initial_rank_c1)

    def test_property_4_non_rankable_cannot_appear_in_ranked_output(self) -> None:
        """PROPERTY 4: A non-rankable candidate can never appear in ranked output."""
        c_safe = _make_candidate("c_safe", is_safe=True)
        cset = CandidateSet(request_id="req_1", candidates=(c_safe,))
        res = rank_candidate_set(cset, self.deadline)
        for rc in res.ranked_candidates:
            self.assertTrue(rc.candidate.is_rankable)

    def test_property_5_repeated_ranking_produces_identical_ordering(self) -> None:
        """PROPERTY 5: Repeated ranking produces identical ordering."""
        cands = [
            _make_candidate(f"c_{i}", total_amount_paid=Decimal(str(200 - i * 10)), first_payment_date=self.req_d)
            for i in range(5)
        ]
        first = rank_candidates(cands, self.deadline)
        for _ in range(10):
            repeated = rank_candidates(cands, self.deadline)
            self.assertEqual(
                [rc.candidate.candidate_id for rc in first.ranked_candidates],
                [rc.candidate.candidate_id for rc in repeated.ranked_candidates],
            )
            self.assertEqual(
                [rc.rank for rc in first.ranked_candidates],
                [rc.rank for rc in repeated.ranked_candidates],
            )

    def test_property_6_decimal_total_cost_comparison_is_exact(self) -> None:
        """PROPERTY 6: Decimal total-cost comparison is exact."""
        # 0.1 + 0.2 in float != 0.3, but in Decimal it is exact
        c1 = _make_candidate("c1", total_amount_paid=Decimal("0.1") + Decimal("0.2"))  # 0.3
        c2 = _make_candidate("c2", total_amount_paid=Decimal("0.30000000000000004"))
        res = rank_candidates([c2, c1], self.deadline)
        self.assertEqual(res.ranked_candidates[0].candidate.candidate_id, "c1")

    def test_property_7_changing_lower_criterion_cannot_change_higher_difference(self) -> None:
        """PROPERTY 7: Changing a lower-priority criterion cannot change ordering when a higher differs."""
        # c1 meets deadline (crit 1), c2 misses deadline (crit 1)
        # c2 has 0 cost (crit 3), starts today (crit 4), 1 payment (crit 5), opt_00 (crit 6)
        c1 = _make_candidate(
            "c1_on_time",
            completion_date=self.deadline,
            total_amount_paid=Decimal("99999.00"),
            first_payment_date=self.deadline,
            number_of_payments=20,
            source_payment_option_id="opt_99",
        )
        c2 = _make_candidate(
            "c2_misses",
            completion_date=self.deadline + timedelta(days=1),
            total_amount_paid=Decimal("1.00"),
            first_payment_date=self.req_d,
            number_of_payments=1,
            source_payment_option_id="opt_01",
        )
        res = rank_candidates([c2, c1], self.deadline)
        self.assertEqual(
            res.ranked_candidates[0].candidate.candidate_id,
            "c1_on_time",
            "Deadline-met candidate MUST outrank deadline-missed candidate regardless of lower criteria.",
        )


# ===========================================================================
# 6. Immutability and Traceability Tests
# ===========================================================================

class TestImmutabilityAndTraceability(unittest.TestCase):
    """Verify objects are not mutated and criteria traces are complete."""

    def test_immutability_of_candidates_after_ranking(self) -> None:
        c1 = _make_candidate("c1", total_amount_paid=Decimal("100.00"))
        c2 = _make_candidate("c2", total_amount_paid=Decimal("200.00"))
        c1_snapshot = copy.deepcopy(c1)
        c2_snapshot = copy.deepcopy(c2)

        _ = rank_candidates([c1, c2], date(2026, 6, 1))

        self.assertEqual(c1, c1_snapshot)
        self.assertEqual(c2, c2_snapshot)

    def test_traceability_fields_populated_correctly(self) -> None:
        c = _make_candidate(
            candidate_id="c_trace",
            total_amount_paid=Decimal("123.45"),
            first_payment_date=date(2026, 3, 1),
            number_of_payments=2,
            completion_date=date(2026, 4, 1),
            spending_changes=(),
            source_payment_option_id="opt_trace_01",
        )
        deadline = date(2026, 5, 1)
        trace = build_criteria_trace(c, deadline)

        self.assertEqual(trace.candidate_id, "c_trace")
        self.assertTrue(trace.criterion_1_deadline_met)
        self.assertTrue(trace.criterion_2_no_spending_changes)
        self.assertEqual(trace.criterion_3_total_amount_paid, Decimal("123.45"))
        self.assertEqual(trace.criterion_4_first_payment_date, date(2026, 3, 1))
        self.assertEqual(trace.criterion_5_number_of_payments, 2)
        self.assertEqual(trace.criterion_6_payment_option_id, "opt_trace_01")
        self.assertEqual(trace.technical_tie_breaker, "c_trace")


if __name__ == "__main__":
    unittest.main()
