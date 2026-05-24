# Review: T56 — Post-I1 Mechanism Claim Reframing Gate

Verdict: **PASS**

Reviewer: adversarial (independent review, overwrites worker self-review)

## Blocking Issues

None.

## Non-Blocking Issues

### N1: Worker self-review was overwritten

The worker wrote its own review at `docs/review/T56_review.md` (verdict: PASS). This adversarial review replaces it. This is the intended workflow per CLAUDE.md, so no issue — just noting for provenance.

### N2: Claim table uses 12 rows but worker summary says "10-row"

The worker summary and task package Worker Output both state "10-row claim table (3 retain, 4 weaken, 2 retire, 1 reframe, 2 still-open)." The actual table in `docs/post_t55_mechanism_claim_reframing_gate.md` has rows M1–M10, which is exactly 10 rows. The status counts (3+4+2+1+2 = 12) sum to 12, not 10, because M7 and M10 are both `still-open`. The summary text is slightly misleading — it says "10-row claim table" with counts that add to 12. This is cosmetic: the table itself is correct, the counts are correct, only the parenthetical summary double-counts the two `still-open` entries. **Accepted** as non-blocking wording inconsistency.

### N3: R10 in `docs/08_risks_and_open_questions.md` not updated

T56 is a mechanism-claim reframing gate, and the gate document significantly reframes the mechanism story. However, `docs/08_risks_and_open_questions.md` R10 entry still reads as if T56 has not yet run — it says "下一 bounded lane 是 T56 的 post-I1 mechanism claim reframing gate" and references T56 as the next action rather than a completed action. The task package `Forbidden scope` does not include `docs/08_risks_and_open_questions.md`, so the worker correctly did not modify it. This is a **Captain governance sync item**, not a worker issue. **Accepted** — Captain should update R10 after accepting this review.

### N4: Task package T56 checkbox not marked complete in `docs/04_task_board.md`

The T56 entry in the task board still shows `[ ] T56`. Again, per CLAUDE.md, the worker must not modify the task board — this is a Captain action. **Accepted**.

### N5: C4 wording in the ledger is now quite long

The C4 "Boundary wording" cell in `docs/paper_claim_evidence_ledger.md` has grown to a multi-clause paragraph after T56's update. This is honest and informative, but it may be hard to maintain as further tasks update it. **Accepted** — the verbosity is justified by the complexity of the mechanism evidence, and future tasks can streamline it.

## Missing Tests

Not applicable — T56 is a docs-only gate task. No code was changed, no tests are required.

## Suspicious Implementation Details

None found. Specific checks performed:

1. **File scope**: Only the 5 allowed files were modified/created. Confirmed via `git diff --name-only` and `git status -u`:
   - Modified: `docs/paper_claim_evidence_ledger.md`, `docs/tasks/Phase2/T56_post_i1_mechanism_claim_reframing_gate.md`
   - New: `docs/post_t55_mechanism_claim_reframing_gate.md`, `docs/review/T56_review.md`, `docs/for_human/T56_explanation.md`
   - New (untracked, not in allowed set but not forbidden): `docs/worker_summary/T56_worker_summary.md`
   - `.claude/settings.json` was modified but this is a tool-side effect, not a worker action.

2. **No source code or config changes**: No `.py`, `.yaml`, `.yml`, or run-root files were touched.

3. **No governance doc changes**: `docs/00_project_snapshot.md`, `docs/01_legacy_audit.md`, `docs/02_experiment_plan.md`, etc. were not touched.

4. **No execution**: No benchmark, training, trace export, `.tflite`, hardware, cleanup, or comparator execution was run.

5. **No claim upgrades**: Checked all new/modified text for phrases like "causal proof," "mechanism proven," "root cause identified," "multi-seed confirmation." These appear only in negative context (i.e., "do not write X"). C4 remains `partial`. No claim was upgraded to `supported` or `confirmed`.

6. **Claim table honesty**: Cross-checked each row against the actual T36/T38/T54/T55 evidence:
   - M1 (retain): Correct — T36/T38 support, no contradiction.
   - M2 (weaken): Correct — T54/T55 show the instability mostly helps, weakening the "primary cause of degradation" framing.
   - M3 (retain): Correct — T54 confirmed 5/6 seeds.
   - M4 (retire): Correct — T55 directly contradicts "high committed-b is harmful."
   - M5 (retire): Correct — T55 refuted I1 as a general improvement.
   - M6 (weaken): Correct — T54's universal seed (20260510) contradicts exclusivity.
   - M7 (still-open): Correct — I3 was not tested, and T36 identified teacher-delta as a regime shift but did not prove causation.
   - M8 (reframe): Correct — T54/T55 together suggest instability is mostly a feature, not a defect.
   - M9 (weaken): Correct — seed=20260430 violated the category-intervention mapping.
   - M10 (still-open): Correct — deferred, not approved and not permanently closed.

7. **T47 conditioning**: Section 5 of the gate document correctly states T47 "can proceed, but only under explicit mechanism-hedge wording" with 5 specific hedge boundaries. It explicitly states T47 should not proceed as unconditional next work.

8. **Second-intervention verdict**: Section 4 verdict is `deferred pending better question`. I2 is correctly assessed as confounded with I1. I3 is correctly assessed as theoretically distinct but premature. The prerequisite question (whether the instability needs fixing at all) is correctly identified.

9. **Non-claims section**: 9 explicit non-claims are listed, covering the key boundaries (no mechanism closure, no unconditional T47, no permanently-closed intervention lane, no frozen-set reopening, etc.).

## Recommended Next Action

1. Accept T56 as **PASS**.
2. Captain should update `docs/04_task_board.md` to mark T56 as `[x]` and set the current unique task to the next bounded lane (likely T47 under conditioned mechanism-hedge wording, per gate Section 5).
3. Captain should update `docs/08_risks_and_open_questions.md` R10 to reflect that T56 has reframed the mechanism story — the question has shifted from "how to fix instability" to "whether instability needs fixing."
4. Captain should update `docs/07_handoff.md` with T56 closeout notes.
5. If T47 proceeds, it must carry the 5 hedge boundaries specified in gate Section 5.
