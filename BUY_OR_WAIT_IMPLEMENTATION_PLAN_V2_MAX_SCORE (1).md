# BUY OR WAIT? — V2 MAXIMUM-SCORE IMPLEMENTATION PLAN
## HackerRank Orchestrate — September 2026
### Deterministic financial reasoning + AI-assisted extraction/explanation

> **Goal:** Build the hardest-to-break submission possible within the contest scope.
>
> **Important:** No plan can guarantee rank #1. This plan is optimized for correctness, reproducibility, explainability, and AI-judge defensibility while avoiding unnecessary features.

---

# 0. NORTH STAR

The system must answer:

> **“Given everything we know about this user, what is the safest way for them to complete this requested expense while covering essential commitments and maintaining their required minimum balance throughout the forecast?”**

The winning architecture is:

```text
RAW DATA + LOCAL MEDIA
        |
        v
+---------------------------+
| 1. EVIDENCE INGESTION     |
| CSV / images / messages   |
+-------------+-------------+
              |
              v
+---------------------------+
| 2. CANONICAL LEDGER       |
| normalize / link / amend  |
| cancel / currency / state |
+-------------+-------------+
              |
              v
+---------------------------+
| 3. FINANCIAL STATE        |
| balance / income / bills  |
| commitments / preferences|
+-------------+-------------+
              |
              v
+---------------------------+
| 4. RECURRENCE ENGINE      |
| historical patterns ->    |
| future confirmed events   |
+-------------+-------------+
              |
              v
+---------------------------+
| 5. ONE SOURCE-OF-TRUTH    |
| 90-DAY SIMULATOR          |
+-------------+-------------+
              |
              v
+---------------------------+
| 6. OPTIMIZER              |
| safe amount / plans /     |
| wait / partial / changes  |
+-------------+-------------+
              |
              v
+---------------------------+
| 7. DETERMINISTIC SAFETY   |
| GATE + INVARIANT CHECKS   |
+-------------+-------------+
              |
              v
+---------------------------+
| 8. FREE MODEL             |
| explanation only          |
+-------------+-------------+
              |
              v
+---------------------------+
| 9. OUTPUT VALIDATOR       |
| exact output.csv schema   |
+---------------------------+
```

## Golden rule

**AI interprets messy information. Deterministic code owns financial decisions.**

The model must never overwrite:

- `amount_safe_to_pay`
- `affordability_status`
- `recommended_payment_method`
- `payment_plan`
- `earliest_date_for_full_payment`
- `spending_changes_needed`

The model may generate only the human-readable `decision_explanation`, after the deterministic engine has already made the decision.

---


# 0A. PHASE 0 — AGENTS.MD / AI-AGENT COMPLIANCE BOOTSTRAP

**This phase happens before any implementation, dataset analysis, or code modification.**

The repository's root `AGENTS.md` is the authoritative instruction file for AI coding agents working in this repository. It explicitly applies to agents including **opencode** and requires the agent to read it in full before taking action.

## 0A.1 Mandatory first-session sequence

When the repository is cloned and an AI coding agent is started, the **first prompt must explicitly require**:

```text
Before doing anything else:

1. Read AGENTS.md completely.
2. Read problem_statement.md completely.
3. Inspect the repository structure.
4. Treat AGENTS.md as mandatory project instructions.
5. Check the repository-root log.txt as required by AGENTS.md.
6. Follow the session-start and per-turn logging requirements.
7. Do not modify AGENTS.md, CLAUDE.md, or organizer-provided files unless explicitly required by their instructions.
8. Summarize the constraints you understood from AGENTS.md and problem_statement.md.
9. Do not start implementation until the above is complete.
```

**Why explicitly mention it if the agent should discover it automatically?**

Because the repository contract says `AGENTS.md` is the single source of truth for AI coding agents and specifically tells the agent to read it in full before taking action. Explicitly mentioning it in our startup prompt makes the transcript demonstrate that we intentionally established project compliance before implementation rather than merely assuming the IDE agent discovered the file.

## 0A.2 Do NOT duplicate or replace AGENTS.md

Do not create a second project instruction file that contradicts the repository contract.

Use:

```text
AGENTS.md
     ↓
project/problem constraints
     ↓
our implementation plan
     ↓
coding-agent prompts
```

Our V2 plan is an implementation strategy **under** the repository instructions. It does not override them.

If `AGENTS.md` conflicts with our implementation plan, follow the repository instruction and then adapt the implementation.

## 0A.3 Required logging workflow

The agent must follow the logging contract in `AGENTS.md`:

```text
SESSION START
      ↓
every user turn
      ↓
append-only log entry
      ↓
implementation/test actions
```

The log must:

- live at the repository root beside `AGENTS.md`
- be created if missing
- be append-only
- never be rewritten/reordered/deleted
- remain in `.gitignore`
- contain the exact coding harness/agent name in `tool=`
- never contain API keys, tokens, cookies, private keys, or sensitive PII
- be shared across agents/sub-agents/worktrees according to the repository instructions

## 0A.4 Chat transcript is part of the engineering workflow

Because the contest requires the AI-agent chat transcript, do not treat the transcript as an afterthought.

Our development conversation should visibly demonstrate:

```text
read repository instructions
        ↓
inspect data/schema
        ↓
form hypothesis
        ↓
implement
        ↓
test
        ↓
inspect failures
        ↓
fix general rule
        ↓
re-run regression tests
```

Prefer precise prompts such as:

```text
"Read AGENTS.md and problem_statement.md first. Summarize the constraints."
"Inspect the schema before implementing this rule."
"Could this amendment be double-counted? Add a regression test."
"Do not patch this request ID. Find the general rule causing the mismatch."
"Prove this property with a test."
```

Avoid using the coding agent as a black box with prompts such as:

```text
"Build everything."
"Make it win."
"Fix the answers."
```

The transcript should demonstrate engineering judgment and iterative validation.

## 0A.5 Repository instruction/deadline sanity check

At session start, the coding agent must use the deadline specified by the **current repository `AGENTS.md`**, not a stale deadline copied from an older implementation plan.

If the repository instruction and another project document contain different dates/times:

1. flag the discrepancy,
2. follow the higher-priority repository instruction,
3. do not silently invent a new deadline,
4. continue implementation unless the instruction itself blocks progress.

This is especially important because the repository's current `AGENTS.md` contains its own explicit session-start timing instruction.

## 0A.6 First transcript checkpoint

The first meaningful agent response should establish:

```text
AGENTS.md: READ
problem_statement.md: READ
dataset contract: UNDERSTOOD
logging contract: UNDERSTOOD
output schema: UNDERSTOOD
deadline source: AGENTS.md
implementation: READY
```

Only after this checkpoint should Phase 1 — Data Exploration begin.


# 1. WHAT WE ARE OPTIMIZING

The required output columns are:

```text
request_id
amount_safe_to_pay
affordability_status
recommended_payment_method
payment_plan
earliest_date_for_full_payment
spending_changes_needed
decision_explanation
```

The highest-risk correctness areas are:

1. exact safe-to-pay amount
2. correct affordability class
3. correct earliest full-payment date
4. exact payment-plan arithmetic
5. real installment-option compliance
6. flexible-only spending changes
7. correct treatment of pending/scheduled/settled events
8. linked event amendments/cancellations
9. recurring-income/expense forecasting
10. explanations that do not contradict the actual decision

---

# 2. REPOSITORY STRUCTURE

Use a clean separation of concerns.

```text
hackerrank-orchestrate-september26/
|
+-- AGENTS.md                         # authoritative AI-agent instructions; READ FIRST; do not modify unless allowed
+-- CLAUDE.md                         # provided / do not modify unless allowed
+-- problem_statement.md              # provided
|
+-- dataset/                          # provided data
|   +-- requests.csv
|   +-- sample_requests.csv
|   +-- financial_profiles.csv
|   +-- financial_events.csv
|   +-- request_payment_options.csv
|   +-- exchange_rates.csv
|   +-- messages.csv
|   +-- images.csv
|   +-- media/
|       +-- images/
|
+-- code/
|   +-- main.py
|   +-- config.py
|   |
|   +-- models.py                     # dataclasses / typed domain objects
|   +-- money.py                      # Decimal utilities
|   |
|   +-- ingest.py                     # CSV ingestion
|   +-- canonical_ledger.py           # normalization + lineage
|   +-- image_extractor.py             # OCR + VLM fallback
|   +-- message_extractor.py           # structured message interpretation
|   +-- currency.py                   # deterministic FX conversion
|   +-- recurrence.py                  # recurrence detection/expansion
|   |
|   +-- financial_engine.py            # single source-of-truth simulator
|   +-- optimizer.py                   # safe amount + candidate generation
|   +-- decision_engine.py             # candidate ranking
|   +-- spending_changes.py            # combinatorial flexible-expense search
|   |
|   +-- safety_gate.py                 # final deterministic safety gate
|   +-- explanation_writer.py          # free-model explanation
|   +-- explanation_validator.py       # contradiction checks
|   +-- validator.py                   # output schema / business invariants
|   |
|   +-- evaluation/
|       +-- sample_eval.py
|       +-- metamorphic_tests.py
|       +-- regression_tests.py
|       +-- decision_certificates.py
|
+-- evaluation/
|   +-- accuracy_report.md
|   +-- usage_report.md
|   +-- decision_certificates/
|
+-- output.csv
+-- requirements.txt
+-- README.md
+-- .env.example
+-- .gitignore
+-- log.txt
```

---

# 3. PHASE 0 — DATA CONTRACT FIRST

Before writing financial logic:

## 3.1 Inspect every provided file

Run:

```text
python -m code.inspect_dataset
```

or an equivalent inspection script.

Record:

- file names
- row counts
- columns
- dtypes
- null counts
- unique status values
- unique event types/categories
- currency codes
- date ranges
- linked IDs
- blank monetary fields
- image references
- message references
- payment-option fields

Do not infer schema from a sample alone.

## 3.2 Build a data dictionary

Create:

```text
evaluation/data_dictionary.md
```

For every column document:

```text
column
type
meaning
nullable?
source
how used
```

This is also useful during the AI judge interview.

---

# 4. PHASE 1 — MONEY MUST BE EXACT

## 4.1 Use Decimal everywhere

Do **not** use binary floating-point for money.

Use:

```python
from decimal import Decimal
```

All amounts become:

```python
Decimal("1234.50")
```

Never:

```python
Decimal(1234.50)
```

because the binary float has already introduced error.

## 4.2 Central money helpers

Create:

```python
def money(value) -> Decimal:
    ...

def quantize_money(value: Decimal) -> Decimal:
    ...

def money_str(value: Decimal) -> str:
    ...
```

Use one rounding policy consistently.

## 4.3 Exact arithmetic invariant

Every generated payment plan must satisfy:

```text
sum(plan amounts) == requested_amount
```

after the system's defined monetary quantization.

For partial plans:

```text
today_amount + remaining_amount == requested_amount
```

Never allow the final installment to be off by a cent because of rounding.

---

# 5. PHASE 2 — CANONICAL FINANCIAL LEDGER

This is the biggest architectural upgrade.

The simulator should **never read raw CSV rows directly**.

Everything goes through a canonical ledger.

## 5.1 Canonical event

Create a normalized object:

```python
@dataclass
class CanonicalEvent:
    event_id: str
    user_id: str
    category: str
    description: str

    amount: Decimal
    currency: str
    home_amount: Decimal

    event_date: date
    settlement_date: Optional[date]

    status: str
    cash_behavior: str

    is_debit: bool
    is_recurring: bool
    is_flexible: bool
    is_protected: bool

    source: str
    source_record_id: str

    related_event_id: Optional[str]
    supersedes_event_id: Optional[str]

    confidence: str
```

The exact field set may be adjusted to the actual provided schema.

## 5.2 Why this matters

Without canonicalization:

```text
original event
      +
amendment
      +
message
      +
image
```

can accidentally become:

```text
4 cash movements
```

when they actually represent:

```text
1 financial obligation with updated evidence
```

The ledger prevents double counting.

---

# 6. PHASE 3 — EVENT LINEAGE / CONFLICT RESOLUTION

Treat related events as a lineage graph.

Example:

```text
event_100
   |
   +---- amendment_101
   |
   +---- message_202
   |
   +---- cancellation_103
```

The final effective event is derived from the entire chain.

## 6.1 Resolution priority

Use the source/problem-statement rules first.

Where multiple valid interpretations remain:

1. explicit cancellation / settlement / amendment
2. newer record from the same source
3. settled/confirmed information over estimates
4. safer lower-cash interpretation when ambiguity genuinely remains

Do not invent additional financial rules.

## 6.2 Required tests

Test:

- original + amendment
- original + cancellation
- multiple amendments
- cancellation after amendment
- message referencing event
- unrelated message that mentions similar text
- linked chain with multiple records

Expected property:

> One real obligation should not become multiple cash flows.

---

# 7. PHASE 4 — IMAGE EVIDENCE PIPELINE

Images are evidence, not financial decisions.

## 7.1 Extraction pipeline

For each blank/required amount:

```text
image metadata
      |
      v
locate image by image_id
      |
      v
OCR
      |
      +-- high confidence --> structured extraction
      |
      +-- low confidence --> free VLM fallback
      |
      v
validation
      |
      v
canonical event
```

## 7.2 Structured extraction contract

The model should return something equivalent to:

```json
{
  "amount": "1250.00",
  "currency": "EUR",
  "confidence": "high"
}
```

Do not accept arbitrary prose as financial evidence.

## 7.3 Validate extracted values

Check:

- amount exists
- amount > 0 when the event requires an amount
- currency is valid
- image is linked to the expected event
- extracted amount is not obviously inconsistent with source metadata
- conflicting evidence is resolved deterministically

Never silently turn an unresolved blank into zero.

---

# 8. PHASE 5 — MESSAGE INTERPRETATION

Messages can change event state, but the model should only extract structured intent.

## 8.1 Contract

Possible structured actions:

```text
cancel
amend_amount
delay_to
confirm
```

Example:

```json
{
  "related_event_id": "event_123",
  "action": "amend_amount",
  "new_amount": "8500.00",
  "effective_date": "2026-09-20",
  "confidence": "high"
}
```

## 8.2 Trust boundary

A message should affect a financial event only when:

- it can be linked to the relevant user/event
- the action is supported by the actual message
- the structured extraction passes validation

Never let a vague natural-language message mutate unrelated events.

---

# 9. PHASE 6 — CURRENCY CONVERSION

All simulation values must be in the user's home currency.

## 9.1 Use the correct rate date

Use the settlement date required by the dataset/problem rules.

Do not blindly use:

```text
request_date
```

when the financial event has a distinct settlement date.

## 9.2 Conversion

```text
foreign amount
      x
dataset FX rate for required date
      =
home-currency amount
```

Use `Decimal`.

## 9.3 FX tests

Test:

- home currency = event currency
- foreign currency with valid rate
- missing rate
- multiple currencies
- settlement date differs from event date

Never silently use a random/nearest rate unless the problem statement explicitly permits it.

---

# 10. PHASE 7 — EVENT CASH-STATE CLASSIFICATION

Create one authoritative classifier.

Example categories:

```text
settled     -> include
scheduled   -> include
pending     -> debit-only if required by problem rules
failed      -> skip
cancelled   -> skip
unrealized  -> skip from cash forecast
```

The exact status mapping must follow the provided dataset/problem statement.

## Critical rule

A pending **credit** must not magically become spendable cash if the problem rules do not treat it as available.

This prevents false affordability.

---

# 11. PHASE 8 — REAL RECURRENCE ENGINE

Do not manually guess future recurring events.

## 11.1 Detect historical patterns

For events belonging to the same recurring obligation, analyze:

- amount similarity
- category
- merchant/description
- date spacing
- historical count
- consistency

Only classify as recurring when evidence supports it.

## 11.2 Expand future occurrences

Example:

```text
Salary:
Sep 01
Oct 01
Nov 01
```

becomes a recurring rule:

```text
monthly
next occurrence = Dec 01
```

Do not mutate historical events.

Instead:

```text
historical canonical events
            +
recurrence rule
            |
            v
future projected events
```

## 11.3 Avoid false recurrence

Do not classify:

```text
rent
random laptop purchase
one emergency payment
```

as recurring merely because they occur more than once.

Use actual historical pattern evidence.

---

# 12. PHASE 9 — USER FINANCIAL STATE

Build a clean state object.

```python
@dataclass
class FinancialState:
    user_id: str
    home_currency: str

    available_balance: Decimal
    minimum_balance: Decimal

    recurring_income: list
    recurring_expenses: list
    one_time_events: list

    payment_methods_accepted: list
    max_installment_months: Optional[int]

    financial_priorities: str
    flexible_categories: list
    protected_categories: list
```

The exact fields must reflect the actual provided dataset.

---

# 13. PHASE 10 — ONE 90-DAY SIMULATOR

This is the **single source of truth**.

Every affordability question must eventually call the same simulation engine.

```python
simulate(
    state,
    request_date,
    proposed_payments=[],
    spending_changes=[]
)
```

It returns:

```python
{
    "daily_balances": {...},
    "minimum_projected_balance": Decimal(...),
    "minimum_balance_date": date(...),
    "safe_throughout": True/False
}
```

## 13.1 Daily processing

For each day in the required forecast horizon:

```text
starting balance
+
confirmed income
-
recurring mandatory expenses
-
scheduled/pending debits according to rules
-
one-time obligations
-
proposed purchase payment
-
spending-change-adjusted flexible expenses
=
end-of-day balance
```

Then:

```text
minimum daily balance >= user's minimum balance
```

must hold for a safe plan.

## 13.2 Never duplicate simulation logic

Do not write:

```text
safe_to_pay_simulation()
installment_simulation()
wait_simulation()
```

with slightly different rules.

Write one simulator and feed it different proposed plans.

This prevents subtle inconsistencies.

---

# 14. PHASE 11 — SAFE-TO-PAY OPTIMIZATION

Define:

> `amount_safe_to_pay` = maximum amount payable today while the resulting forecast remains safe.

## 14.1 Preconditions

If the baseline financial state is already unsafe:

```text
amount_safe_to_pay = 0
```

unless the problem's explicit rules indicate another treatment.

## 14.2 Monotonicity

If paying amount `X` is safe, then paying any amount `< X` must also be safe, all else equal.

That lets us use binary search.

## 14.3 Binary search with Decimal

```text
lo = 0
hi = requested_amount

while precision not reached:
    mid = (lo + hi) / 2

    simulate(payment_today = mid)

    if safe:
        lo = mid
    else:
        hi = mid
```

Return the correctly quantized `lo`.

## 14.4 Required invariant

```text
0 <= amount_safe_to_pay <= requested_amount
```

---

# 15. PHASE 12 — EARLIEST FULL PAYMENT DATE

For every candidate date within the required forecast horizon:

```text
simulate:
    no purchase before candidate date
    full requested amount on candidate date
```

Return the first safe date.

Do not assume:

```text
next salary date
```

is automatically the answer.

The answer must emerge from simulation.

## Off-by-one protection

Explicitly test:

- request date itself
- day before salary
- salary day
- day after salary
- large expense immediately after salary
- final forecast day

---

# 16. PHASE 13 — PAYMENT PLAN CANDIDATE GENERATION

Do not pick the first plan that works.

Generate **all legitimate plan families supported by the dataset**.

Candidate families:

```text
A. full payment today
B. real installment options
C. allowed partial payment
D. wait for earliest safe full payment
E. spending-change-assisted plan
```

Every candidate must be simulated.

---

# 17. PHASE 14 — INSTALLMENTS

Only use installment plans that actually exist in the provided payment-option data.

For every option validate:

```text
request_id matches
payment option is valid
number of payments valid
interval valid
start date valid
total payable amount valid
max installment duration respected
payment method accepted
final date <= desired completion date when required
```

Never invent an installment schedule.

## Plan arithmetic

If the option says:

```text
4 payments
```

the generated plan must have exactly four payments.

The total must match the option's total payable amount and the request semantics.

---

# 18. PHASE 15 — PARTIAL PAYMENT

Only consider partial payment when:

```text
request.allows_partial_payment == true
```

and the user's accepted payment methods permit it.

For a two-stage plan:

```text
today: safe_amount
later: requested_amount - safe_amount
```

The second amount must be calculated with exact Decimal arithmetic.

Then simulate the **entire plan**.

Never assume that because today's payment is safe, the remaining payment will be safe.

---

# 19. PHASE 16 — SPENDING-CHANGE OPTIMIZATION

Replace the old greedy strategy.

## Old approach — DO NOT USE

```text
sort expenses by amount
stop largest
stop second largest
stop third largest
```

This can produce a financially valid but unnecessarily disruptive plan.

## New approach

Consider only:

```text
flexible == true
AND
category is allowed to change
AND
not protected
```

Generate candidate changes such as:

```text
stop:event_id
reduce_to:event_id:new_amount
```

within the maximum supported number of changes.

For each combination:

```text
apply changes
simulate full proposed purchase/payment plan
measure:
    safe?
    amount of spending reduced
    number of changes
    disruption
```

Select the best valid combination according to the problem's preference/ranking rules.

## Never modify

Examples of protected categories may include:

```text
rent
utilities
required debt payments
essential obligations
```

but only when the provided profile/data marks them as protected.

Do not hardcode category assumptions that are not supported by the dataset.

---

# 20. PHASE 17 — CANDIDATE RANKING

Separate:

```text
candidate generation
```

from:

```text
candidate ranking
```

A candidate is valid only if it passes simulation.

Then rank valid candidates using the required priority ordering from the problem/data.

Recommended deterministic ordering:

```text
1. Complete the request by desired completion date
2. Avoid spending changes when otherwise equivalent
3. Respect user's payment preferences
4. Minimize total payable amount / financing cost
5. Start earlier when otherwise equivalent
6. Fewer payments
7. Stable deterministic tie-breaker using payment option ID
```

**Important:** If the actual problem statement defines a different priority, that rule wins. Do not blindly use this list if the provided specification says otherwise.

---

# 21. PHASE 18 — DETERMINISTIC SAFETY GATE

Before any row reaches `output.csv`, run a final safety gate.

```python
def safety_gate(decision, state, request):
    ...
```

Reject a decision if:

- safe amount exceeds request
- payment plan does not sum correctly
- plan violates forecast minimum
- plan uses unsupported payment method
- installment option is fake
- installment duration violates user preference
- payment occurs after required completion date when the candidate claims completion
- spending change targets protected/non-flexible event
- spending change references nonexistent event
- earliest date contradicts simulation
- status contradicts plan
- explanation tries to change financial fields

This is the final defense against bugs.

---

# 22. PHASE 19 — DECISION CERTIFICATE

Create an internal audit object for every request.

This is **not a new contest output feature**.

It is an engineering/debugging artifact.

Example:

```json
{
  "request_id": "request_123",
  "baseline_balance": "50000.00",
  "minimum_balance": "10000.00",
  "safe_to_pay_today": "18000.00",
  "minimum_projected_balance": "10000.00",
  "minimum_balance_date": "2026-10-04",
  "selected_candidate": "partial_payment",
  "candidates_evaluated": 12,
  "rejected_candidates": [
    {
      "type": "full_payment",
      "reason": "minimum balance breached on 2026-09-27"
    }
  ],
  "evidence": [
    "financial_events:...",
    "profile:...",
    "message:..."
  ]
}
```

Use these certificates for debugging and AI-judge explanations.

---

# 23. PHASE 20 — AI EXPLANATION LAYER

The free model gets a **read-only decision context**.

Prompt structure:

```text
You are explaining a pre-computed financial decision.

Do not change the decision.
Do not recalculate it.
Do not invent facts.
Do not recommend a different payment method.
Use only supplied facts.

Write 2-3 concise sentences explaining:
1. what is safe today,
2. what payment plan is recommended,
3. why the plan is safe,
4. when full payment can be completed when relevant.
```

Input includes:

```text
request
requested amount
currency
current available balance
minimum balance
amount safe today
status
recommended method
payment plan
earliest full-payment date
spending changes
key reason
```

---

# 24. FREE-MODEL STRATEGY

Do not hard-code a paid provider.

Use a tiny provider-agnostic adapter:

```text
explanation_writer.py
        |
        v
model_runner
    /       \
opencode   Antigravity
 free        free
 model       model
```

The financial engine must not depend on the model being available.

## Required fallback

If the free model:

- times out
- returns invalid text
- is unavailable
- contradicts the decision
- invents numbers
- changes the recommendation

then use a deterministic explanation template.

The pipeline must still produce `output.csv`.

---

# 25. PHASE 21 — EXPLANATION VALIDATOR

After model generation, validate the explanation.

Reject/fallback if it:

- says “buy now” when status says wait
- gives a different amount
- gives a different date
- names a different payment method
- invents a fee
- invents income
- invents a commitment
- contradicts the payment plan
- is empty or malformed

Use deterministic fallback immediately.

---

# 26. PHASE 22 — METAMORPHIC TESTING

This is one of the strongest additions for reliability.

Instead of testing only known answers, test properties that must always hold.

## Test A — More requested amount

Holding all financial state constant:

```text
requested_amount ↑
```

must never cause:

```text
amount_safe_to_pay ↑
```

for the same request context.

## Test B — More available balance

```text
available_balance ↑
```

must not make:

```text
amount_safe_to_pay ↓
```

all else equal.

## Test C — Cancel expense

Cancelling a future debit should not reduce forecast balances.

## Test D — Add mandatory expense

Adding a mandatory future debit must not increase safe-to-pay.

## Test E — Payment sum

```text
sum(plan) == requested_amount
```

where a complete plan is required.

## Test F — Flexible-only

Every spending change must reference a flexible permitted event.

## Test G — Real installment

Every installment plan must correspond to an actual provided option.

## Test H — Monotonic safe amount

For:

```text
x1 < x2
```

if `x2` is safe, `x1` must also be safe.

---

# 27. PHASE 23 — REGRESSION TEST SUITE

Build fixed tests around the hardest cases discovered in the dataset.

Minimum categories:

```text
1. affordable now
2. affordable with installment plan
3. affordable with partial payment
4. affordable later
5. not affordable
6. minimum balance reached exactly
7. minimum balance breached by one cent
8. salary arriving tomorrow
9. salary arriving after a large obligation
10. pending credit
11. pending debit
12. cancelled event
13. amended event
14. delayed event
15. recurring expense
16. recurring income
17. image-derived amount
18. foreign currency
19. flexible spending changes
20. protected spending
21. no installment preference
22. installment duration limit
23. desired completion deadline
24. no valid plan
```

---

# 28. PHASE 24 — SAMPLE / GROUND-TRUTH EVALUATION

If `sample_requests.csv` or another provided answer set exists, build:

```text
evaluation/sample_eval.py
```

Report accuracy separately for:

```text
amount_safe_to_pay
affordability_status
recommended_payment_method
payment_plan
earliest_date_for_full_payment
spending_changes_needed
decision_explanation
```

Do not optimize only for aggregate accuracy.

A single systemic bug can damage many rows.

---

# 29. PHASE 25 — FULL-DATA SANITY ANALYSIS

Before final generation, compute:

```text
number of requests
number of users
number of currencies
number of blank event amounts
number of image-linked events
number of message-linked events
number of recurring events
number of linked event chains
number of installment options
number of requests allowing partial payment
number of protected categories
number of flexible categories
```

Also count final statuses:

```text
affordable_now
affordable_with_plan
affordable_later
not_affordable
```

Look for suspicious distributions.

Examples:

```text
99% affordable_now
```

or:

```text
100% not_affordable
```

may indicate a systemic bug.

Do not manually change outputs to “look realistic”; use the distribution only as a debugging signal.

---

# 30. PHASE 26 — FINAL OUTPUT VALIDATION

Before writing:

```text
output.csv
```

validate:

## Schema

Exact columns and order:

```text
request_id
amount_safe_to_pay
affordability_status
recommended_payment_method
payment_plan
earliest_date_for_full_payment
spending_changes_needed
decision_explanation
```

## Amount

```text
0 <= amount_safe_to_pay <= requested_amount
```

## Status

Only allowed values from the problem statement.

## Payment method

Only allowed values from the problem statement.

## Payment plan

For a complete plan:

```text
sum == requested_amount
```

## Date

`earliest_date_for_full_payment` must be consistent with the selected decision.

## Spending changes

Only valid flexible event IDs.

## Row count

The current provided request dataset contains 250 requests (`request_26` through `request_275`); verify the actual file at runtime rather than hardcoding this assumption.

---

# 31. PHASE 27 — NO HARD-CODED ANSWERS

Never write:

```python
if request_id == "request_123":
    return ...
```

Never special-case known test labels.

Allowed:

```python
if status == "cancelled":
    ...
```

Allowed:

```python
if event_id is linked through the provided dataset:
    ...
```

Not allowed:

```python
if event_id == "the test event":
    ...
```

The system must derive decisions from data.

---

# 32. PHASE 28 — PERFORMANCE

The dataset is small enough that correctness matters more than premature optimization.

Still:

- cache canonicalized user state
- cache recurrence expansion
- cache baseline simulation where safe
- cache installment plans
- avoid repeated dataframe filtering inside deep loops
- index events by user/event ID
- use Decimal only where monetary exactness matters
- keep model calls out of the financial search loop

## Important

Do **not** call an LLM once per candidate plan.

Bad:

```text
250 requests
x 20 candidates
x LLM
= 5000 model calls
```

Good:

```text
deterministic optimization
        |
        v
one final explanation call/request
```

---

# 33. PHASE 29 — MODEL CALL BUDGET

Preferred:

```text
Image extraction:
OCR first
VLM only when necessary

Messages:
batch/structured extraction where practical

Explanation:
one call per final request, with deterministic fallback
```

Do not waste free-model quota on calculations the Python engine can perform exactly.

---

# 34. PHASE 30 — IMPLEMENTATION ORDER

## Step 1 — Dataset inspection

Ask the coding agent:

```text
Read every provided CSV and local media reference.
Do not write business logic yet.
Produce:
- shapes
- schemas
- null counts
- status values
- event types
- currencies
- linked IDs
- image references
- message references
- payment-option structure
- sample output schema.
```

## Step 2 — Domain models + Decimal

Build:

```text
models.py
money.py
```

Add tests immediately.

## Step 3 — Canonical ledger

Build:

```text
canonical_ledger.py
```

Test event lineage before building the simulator.

## Step 4 — Image/message extraction

Build:

```text
image_extractor.py
message_extractor.py
```

Use structured outputs.

## Step 5 — Currency conversion

Build:

```text
currency.py
```

with exact settlement-date lookup.

## Step 6 — Recurrence

Build:

```text
recurrence.py
```

Test against historical patterns.

## Step 7 — Single simulator

Build:

```text
financial_engine.py
```

Do not build candidate ranking yet.

## Step 8 — Safe-to-pay

Implement Decimal binary search.

## Step 9 — Earliest full payment

Implement date scan using the same simulator.

## Step 10 — Candidate generation

Build:

```text
optimizer.py
```

Generate all legitimate plan types.

## Step 11 — Spending-change search

Build combinatorial flexible-event search with pruning.

## Step 12 — Ranking

Build:

```text
decision_engine.py
```

using explicit deterministic priority rules.

## Step 13 — Safety gate

Build:

```text
safety_gate.py
```

and make it impossible to bypass before output.

## Step 14 — Explanation

Build:

```text
explanation_writer.py
explanation_validator.py
```

Free model + deterministic fallback.

## Step 15 — Validator

Build:

```text
validator.py
```

with hard assertions.

## Step 16 — Metamorphic tests

Build:

```text
evaluation/metamorphic_tests.py
```

## Step 17 — Full run

Generate:

```text
output.csv
```

## Step 18 — Final audit

Run every test and inspect all failures before submission.

---

# 35. EXACT CODING-AGENT PROMPT SEQUENCE

Use short, controlled prompts rather than asking the coding agent to build everything in one shot.

### Prompt 0 — Mandatory repository bootstrap

```text
Before doing anything else:

1. Read AGENTS.md completely.
2. Read problem_statement.md completely.
3. Inspect the repository structure.
4. Treat AGENTS.md as mandatory project instructions.
5. Check log.txt and follow the repository's session-start/per-turn logging contract.
6. Do not modify AGENTS.md, CLAUDE.md, or organizer-provided files unless explicitly required.
7. Summarize the constraints you understood.
8. Verify the output schema and dataset contract.
9. Only after this, begin implementation.
```

The agent's response to this prompt should become the first important checkpoint in the chat transcript.

### Prompt 1

```text
Read the repository instructions and problem_statement.md first.
Inspect all dataset files and local media metadata.
Do not implement financial logic yet.
Create a concise data dictionary and identify all schema ambiguities.
```

### Prompt 2

```text
Implement models.py and money.py using Decimal for every monetary value.
Add unit tests for parsing, quantization, addition, subtraction and exact plan sums.
Do not use float for financial calculations.
```

### Prompt 3

```text
Implement canonical_ledger.py.
Normalize raw events into canonical events.
Implement event lineage for amendments/cancellations.
Prevent double counting.
Do not invent rules not supported by the problem statement.
Add tests for linked event chains.
```

### Prompt 4

```text
Implement image_extractor.py and message_extractor.py.
Use OCR first for images and a provider-agnostic free-model fallback only when necessary.
Return structured extraction objects.
Do not let model prose directly modify financial state.
```

### Prompt 5

```text
Implement currency.py using the exact exchange-rate rules from the problem statement.
Use Decimal and the required settlement date.
Add tests for same-currency and foreign-currency events.
```

### Prompt 6

```text
Implement recurrence.py.
Detect recurring patterns from historical data and expand future occurrences without mutating historical events.
Add tests that prevent one-off purchases from becoming recurring.
```

### Prompt 7

```text
Implement financial_engine.py as the single source of truth.
Every plan must be evaluated through the same 90-day simulation.
Return daily balances, minimum projected balance, critical date and safety status.
```

### Prompt 8

```text
Implement amount_safe_to_pay using Decimal binary search.
Prove through tests that the result is bounded and monotonic.
Do not duplicate simulation logic.
```

### Prompt 9

```text
Implement earliest full-payment date by scanning candidate dates through the same simulator.
Test salary-day and off-by-one cases.
```

### Prompt 10

```text
Implement optimizer.py.
Generate every legitimate candidate supported by the provided payment options and request rules.
Evaluate every candidate through the simulator.
Do not use an LLM to choose the final financial plan.
```

### Prompt 11

```text
Implement spending-change optimization.
Only flexible permitted events may be changed.
Evaluate combinations rather than blindly stopping the largest expense.
Minimize disruption subject to the problem's priority rules.
```

### Prompt 12

```text
Implement decision_engine.py.
Separate candidate generation from deterministic ranking.
Make the ranking rules explicit and testable.
```

### Prompt 13

```text
Implement safety_gate.py.
Reject any candidate that violates a financial invariant.
The safety gate must run immediately before output generation.
```

### Prompt 14

```text
Implement explanation_writer.py.
The free model receives a read-only decision object and may only generate decision_explanation.
Add deterministic fallback.
```

### Prompt 15

```text
Implement explanation_validator.py.
Reject explanations that contradict any deterministic decision field.
Fallback instead of repairing the financial decision with the model.
```

### Prompt 16

```text
Implement validator.py.
Validate exact schema, bounds, plan arithmetic, dates, payment methods, installment validity, flexible-only spending changes and row count.
```

### Prompt 17

```text
Implement metamorphic_tests.py.
Test monotonicity and invariants:
more balance cannot reduce safe amount,
more requested amount cannot increase safe amount,
cancelling an expense cannot lower future balance,
adding mandatory expense cannot increase safe amount,
payment plans sum exactly,
spending changes are flexible-only.
```

### Prompt 18

```text
Run the full pipeline.
Do not patch individual request IDs.
For every failure, identify the general rule causing the failure and fix the rule.
Re-run the complete regression suite after every change.
```

---

# 36. DEBUGGING PROTOCOL

When a row is wrong:

## Never do this

```text
request_147 -> manually set answer
```

## Do this

Trace:

```text
request
  |
  v
canonical evidence
  |
  v
financial state
  |
  v
recurrence
  |
  v
simulation
  |
  v
candidate plans
  |
  v
ranking
  |
  v
safety gate
```

Ask:

```text
Where did the wrong value first appear?
```

Fix the earliest incorrect layer.

---

# 37. DECISION CERTIFICATE DEBUG VIEW

For every failed test, print:

```text
REQUEST
requested amount
desired completion date

FINANCIAL STATE
starting balance
minimum balance
income events
mandatory expenses
flexible expenses
pending debits

CANONICAL EVENTS
effective event list
lineage decisions

BASELINE
minimum projected balance
critical date

SAFE-TO-PAY
computed maximum

CANDIDATES
candidate type
plan
minimum projected balance
valid/invalid
reason

SELECTED
method
status
plan

SAFETY GATE
PASS/FAIL
```

This makes bugs explainable instead of mysterious.

---

# 38. WHAT NOT TO BUILD

Time is the constraint.

Do **not** spend hackathon time on:

- web dashboards
- mobile apps
- authentication
- databases
- unnecessary RAG infrastructure
- live bank APIs
- live market APIs
- external financial data
- vector databases
- multi-agent theatrics
- unnecessary UI
- speculative features not required by the challenge

The strongest system is the one that produces the most correct output and can defend every decision.

---

# 39. AI JUDGE DEFENSE STRATEGY

The official Orchestrate format evaluates the built solution, its outputs, the AI interaction transcript, and a 30-minute AI judge interview.

Prepare to answer:

## Q1. Why isn't the LLM making the financial decision?

Answer:

> Financial decisions are safety-critical and numerically constrained. I use AI where language and multimodal interpretation are useful, but I keep the final financial calculation deterministic and reproducible. The model extracts structured evidence and writes the explanation; the simulator and safety gate own the decision.

## Q2. Why use Decimal?

> Binary floating-point can introduce monetary rounding errors. The output contains exact financial amounts, so all financial arithmetic uses Decimal and explicit quantization.

## Q3. Why one simulator?

> If every candidate plan uses a different affordability calculation, subtle inconsistencies appear. A single simulator makes every decision comparable and testable.

## Q4. How do you prevent double counting?

> Raw evidence is first transformed into a canonical event ledger with lineage for amendments and cancellations. The simulator consumes only effective canonical events.

## Q5. How do you handle images?

> OCR is attempted first. When extraction is uncertain, a free vision-capable model can return structured fields. Code validates the extraction and integrates it into the canonical ledger.

## Q6. How do you handle messages?

> The model extracts a structured action and event linkage. Code validates the linkage and applies the change. The model cannot directly mutate financial state.

## Q7. Why not simply ask the LLM “can I afford this?”

> Because the challenge requires exact numerical constraints across future income, expenses, payment plans and minimum balance. LLM output is not the right primitive for deterministic financial simulation.

## Q8. How do you know the optimizer is correct?

> Every candidate is simulated using the same financial engine and then passed through deterministic safety invariants. I also use metamorphic tests to verify properties such as monotonicity.

## Q9. What happens if the model is unavailable?

> The financial engine still works. Explanation generation falls back to deterministic templates. No model outage can corrupt the financial decision.

## Q10. What was the hardest engineering problem?

Strong answer:

> Not the LLM. The difficult part was turning heterogeneous evidence into one canonical financial state and ensuring every candidate plan was evaluated against exactly the same future cash-flow model.

---

# 40. AI CHAT TRANSCRIPT QUALITY

Because the contest evaluates the AI-assisted workflow too, make the development transcript demonstrate:

```text
inspect
-> hypothesize
-> implement
-> test
-> inspect failures
-> challenge assumptions
-> fix general rule
-> re-test
```

Good interaction:

```text
"Before implementing, inspect the schema and identify ambiguity."
"Do not assume this status means cash available; verify from the data."
"Add a regression test for this failure."
"Can this logic double-count an amendment?"
"Prove this property with a test."
```

Avoid:

```text
"build everything"
"make it win"
"fix it somehow"
```

The transcript should show engineering judgment.

---

# 41. FINAL 24-HOUR EXECUTION PRIORITY

If time becomes tight, prioritize in this exact order:

## Tier S — MUST HAVE

```text
1. Decimal money
2. Canonical event ledger
3. Correct event state classification
4. Conflict/lineage resolution
5. Currency conversion
6. Recurrence handling
7. One 90-day simulator
8. Safe-to-pay optimization
9. Earliest full-payment date
10. Real installment options
11. Partial-payment logic
12. Spending-change search
13. Deterministic candidate ranking
14. Safety gate
15. Output validator
16. Full output.csv generation
```

## Tier A — VERY HIGH VALUE

```text
17. Image OCR + structured VLM fallback
18. Message structured extraction
19. Explanation validator
20. Metamorphic tests
21. Regression suite
22. Decision certificates
```

## Tier B — POLISH

```text
23. Better explanation prompt
24. Cleaner README
25. Accuracy report
26. AI judge rehearsal
27. Runtime optimization
```

If time is short, never sacrifice Tier S for polish.

---

# 42. FINAL PRE-SUBMISSION CHECKLIST

## Agent / transcript compliance

- [ ] `AGENTS.md` was read in full before implementation
- [ ] `problem_statement.md` was read in full
- [ ] Repository logging contract was followed
- [ ] `log.txt` is append-only and remains gitignored
- [ ] The chat transcript demonstrates iterative engineering, testing, and debugging
- [ ] No agent prompt asks the model to invent or hardcode prediction labels

## Data

- [ ] Every required CSV is read
- [ ] Local media is inspected
- [ ] Blank image-linked amounts are resolved
- [ ] Messages are incorporated where applicable
- [ ] Currency conversion is deterministic
- [ ] Event lineage is resolved
- [ ] No double counting

## Financial engine

- [ ] Decimal everywhere
- [ ] One simulator
- [ ] 90-day forecast correct
- [ ] Minimum balance enforced
- [ ] Recurring income handled
- [ ] Recurring expenses handled
- [ ] Pending/scheduled/settled rules correct
- [ ] Safe-to-pay monotonic
- [ ] Earliest payment date correct

## Decision engine

- [ ] Full payment checked
- [ ] Installments use real options
- [ ] Partial payment respects request flag
- [ ] Wait option evaluated
- [ ] Spending changes are flexible-only
- [ ] Spending changes are not greedily hardcoded
- [ ] Candidate plans are simulated
- [ ] Candidate ranking is deterministic

## AI

- [ ] No paid API key required
- [ ] Provider-agnostic model adapter
- [ ] AI cannot overwrite financial fields
- [ ] Explanation validator exists
- [ ] Deterministic fallback exists

## Validation

- [ ] Exact output schema
- [ ] Correct row count
- [ ] Amount bounds
- [ ] Exact payment-plan sums
- [ ] Valid dates
- [ ] Valid payment methods
- [ ] Valid installment options
- [ ] Flexible-only changes
- [ ] Metamorphic tests pass
- [ ] Regression tests pass

## Submission

- [ ] `AGENTS.md` remains intact and is not accidentally modified
- [ ] `output.csv`
- [ ] code archive
- [ ] README
- [ ] requirements
- [ ] `.env.example`
- [ ] evaluation report
- [ ] chat transcript/log
- [ ] no secrets
- [ ] no hardcoded test answers

---

# 43. THE FINAL ARCHITECTURAL PRINCIPLE

The system should be explainable as:

```text
AI sees messy evidence.
        ↓
Code converts evidence into canonical financial facts.
        ↓
Code forecasts the user's financial future.
        ↓
Code searches legitimate payment strategies.
        ↓
Code verifies every strategy against the same constraints.
        ↓
Code selects the safest valid strategy.
        ↓
AI explains that already-computed decision.
```

That is the core of the submission.

Not:

```text
LLM → guess affordability → output
```

But:

```text
Evidence
   ↓
Canonical facts
   ↓
Deterministic simulation
   ↓
Constrained optimization
   ↓
Safety gate
   ↓
AI explanation
```

This gives us the strongest combination of:

- exactness
- reproducibility
- multimodal handling
- personalization
- auditability
- graceful model failure
- strong output accuracy
- strong AI-judge defensibility

---

# 44. ONE COMMAND FINAL RUN

The final repository should support a clean command such as:

```bash
python -m code.main
```

which performs:

```text
load data
→ extract evidence
→ canonicalize
→ build recurring projections
→ build user state
→ simulate
→ optimize
→ safety gate
→ explain
→ validate
→ write output.csv
→ write evaluation artifacts
```

No manual editing of `output.csv`.

No request-ID-specific patches.

No paid API dependency.

No model-generated financial decisions.

---

# 45. FINAL RULE FOR THE TEAM

When a bug appears:

> **Do not patch the answer. Fix the rule.**

When a model is uncertain:

> **Do not let uncertainty become a financial decision. Convert it to structured evidence and validate it.**

When two plans work:

> **Do not choose randomly. Apply explicit deterministic ranking.**

When time is running out:

> **Protect correctness before polish.**

When the AI judge asks why the system is reliable:

> **Show the canonical ledger, single simulator, optimizer, safety gate, and tests.**

The objective is not to make the project look complicated.

The objective is to make every submitted number **defensible**.
