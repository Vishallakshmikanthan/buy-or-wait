import csv, hashlib, json, os, re, subprocess, sys, time
from datetime import date
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from code.decision_certificate import (DecisionCertificate, EvidenceRef, FinancialEvidence,
    RankingTraceEvidence, SafetyCheckEvidence, StateEvidence)
from code.explanation import (ClaimType, ExplanationValidator, GroundedExplanationInput,
    build_grounded_explanation_input, generate_grounded_explanation)
from code.nemotron import (DEFAULT_MODEL, GroundedFactPack, NemotronAdapter, NemotronConfig,
    NemotronTelemetry, StructuredClaim, build_grounded_fact_pack)
from code.output import generate_all_outputs

SEP = "=" * 72
DECISION_FIELDS = ["amount_safe_to_pay","affordability_status","recommended_payment_method",
    "payment_plan","earliest_date_for_full_payment","spending_changes_needed"]
FROZEN = {
    "code/canonical.py":    "b1bd5a7d8dc73ca1c52f37e88e49e5064eb185b64c5256504b0d7af9678d6a9f",
    "code/recurrence.py":   "ddcd866761e7683fc78ab942618f761520804d9e401185679c9cf52cb7665b32",
    "code/simulator.py":    "820235d4068af75c45e7909717301fa00e60210a38440bd152b92dc5dc23ac6d",
    "code/safe_to_pay.py":  "b1cb9c995e776a4fe0900a198fdfd3b247577e035f92cb788d2f5427595d3aa2",
    "code/user_state.py":   "f2340e6285a72f0904b67855292ffc5a9329ef2eaa285f256dd9f120ee271f81",
    "code/affordability.py":"6cda8c8b2524f1f71304cdeb43ceecb32d838b3c11195ee2bcb1d7b391c6bee9",
    "code/payment_plan.py": "2af320858fe6ae3d91987bb7d2a1a85dbea719c3c58544994fbfb622f997f20b",
    "code/ranking.py":      "ff57d55a7a3b3f607c79a943f50a239f2d12c048ceb4f5924e8abf3f56899775",
}
FAILS = []

def ok(m): print(f"  PASS  {m}")
def nok(m): print(f"  FAIL  {m}"); FAILS.append(m)
def hdr(n, t): print(f"\n{SEP}\nSECTION {n}: {t}\n{SEP}")

def make_cert(status="affordable_now", method="full_payment",
              req=Decimal("500"), safe=Decimal("500"), floor=Decimal("200"),
              des=date(2026,9,30), ear=date(2026,9,1), plan="none", sc="none", mc=Decimal("350")):
    fe = FinancialEvidence(safety_floor=floor, requested_amount=req, desired_completion_date=des,
        baseline_minimum_available_cash=mc, post_action_minimum_available_cash=mc,
        limiting_date=None, financing_fee=Decimal("0"), total_amount_paid=req,
        completion_date=des, payment_count=1)
    se = StateEvidence()
    lin = (EvidenceRef("fd","FD","r1","amount_safe_to_pay"),)
    return DecisionCertificate(request_id="r1",user_id="u1",certificate_version="1.0.0",
        selected_candidate_id="c1",affordability_status=status,recommended_payment_method=method,
        amount_safe_to_pay=safe,earliest_date_for_full_payment=ear,payment_plan=plan,
        spending_changes_needed=sc,selected_rank=1,total_rankable_candidates=2,
        ordered_candidate_ids=("c1",),ranking_criteria_trace=RankingTraceEvidence(
            candidate_id="c1",deadline_met=True,no_spending_changes=True,total_amount_paid=req,
            first_payment_date=ear,number_of_payments=1),
        competing_candidates=(),final_safety_gate_passed=True,
        selected_candidate_safety_checks=(SafetyCheckEvidence("floor",True,"PASS","ok"),),
        selected_candidate_reason_codes=(),rejected_candidates=(),
        financial_evidence=fe,state_evidence=se,evidence_lineage=lin)
