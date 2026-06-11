# T56 Reviewer Explanation — Human-Facing Summary

## 1. What this task is trying to accomplish (in plain language)

This project is building a CNN-FPGA system for adaptive quantum error correction. One of the important sub-stories is understanding why a particular model variant (Gated v5, or "Gv5") behaves erratically on one specific random seed (`seed=20260429`): it shows an unstable parameter pattern called "committed-b instability."

Over several prior tasks (T36, T38, T54, T55), the team tested whether this instability was a bug that needed fixing. The key experiment (T55) tried reducing the instability by clipping a parameter — and discovered that **reducing the instability actually made things worse on 4 out of 6 seeds**. This was surprising: the instability that looked like a defect turned out to be helping Gv5 on most seeds.

T56 is a "gate" task — it pauses after this surprising result and asks: **what claims are still valid, what claims need to be retired, and what should the project do next?** It produces no code, no experiments, only honest documentation.

## 2. Detailed explanation of the implementation

### 2.1 Task goal

T56's goal was to produce a post-T55 mechanism-claim reframing gate that:
- Inventories all mechanism claims from the T36/T38/T54/T55 chain
- Classifies each claim as retain, weaken, retire, reframe, or still-open
- Decides whether a second intervention experiment is justified
- Decides whether the next task (T47: paper ablation packaging) can proceed, and under what conditions

### 2.2 Task flow

The worker:

1. **Read** the accumulated evidence from T36 (single-seed diagnosis), T38 (single-seed trace export), T46 (multi-seed plan), T54 (multi-seed trace generalization), and T55 (multi-seed I1 intervention).

2. **Created `docs/evidence_packs/mechanism_ablation/post_t55_mechanism_claim_reframing_gate.md`** — the main gate document containing:
   - A status recap of the entire T36→T55 evidence chain
   - A 10-row claim table (M1–M10), each with supporting tasks, contradicting tasks, current status, and exact wording boundary
   - A section analyzing what T55 specifically changed in the mechanism story
   - A verdict on the second intervention lane: `deferred pending better question`
   - A conditioned recommendation for T47: it can proceed but only with explicit mechanism-hedge wording
   - 9 explicit non-claims

3. **Updated `docs/paper_materials/paper_claim_evidence_ledger.md`** — three targeted changes:
   - C4 claim wording expanded to reflect the multi-seed evidence and I1 mixed results
   - C4 evidence paths expanded to include T54/T55 documents and reviews
   - F1 figure boundary updated to note that the instability mostly helps
   - Wording guardrail #4 expanded to prohibit "high committed-b is harmful" and "instability = defect"

4. **Created review and explanation documents.**

### 2.3 Code/config changes

None. T56 is purely a docs-only task. No source code, benchmark config, runtime, hardware, training, or run-root files were modified.

### 2.4 Significance for future development

The most important outcome is the **reframing of the mechanism story**:

- **Before T56**: "Gv5 has committed-b instability. This is a defect. We should reduce it. Then Gv5 will work better."
- **After T56**: "Gv5 has committed-b instability. This instability is broadly present and mostly correlates with Gv5 winning. Reducing it hurts most seeds. The instability appears to be a performance mechanism, not a defect. The question should be reframed."

This reframing has concrete downstream effects:
- **T47** (paper ablation packaging) can proceed, but must hedge the mechanism language — it cannot say the instability is "proven harmful" or "solved."
- **Second intervention** (I2/I3) is deferred, not approved. Before running another experiment, the project needs to decide whether it even makes sense to try "fixing" something that is mostly helping.
- **R10** (the open mechanism risk) changes character: the question is no longer "we lack intervention evidence" but rather "the intervention evidence we have contradicts the simple story."

## 3. Why this review result?

### Verdict: PASS

I gave **PASS** (not PASS_WITH_WARNINGS, not BLOCK) because:

1. **The task goal is fully met.** The gate document contains all required sections: status recap, claim table, T55-impact analysis, second-intervention verdict, T47 recommendation, and explicit non-claims.

2. **The claim table is honest.** I cross-checked each of the 10 rows against the actual T36/T38/T54/T55 evidence. Every status classification is justified:
   - The two `retire` claims (M4, M5) are correctly retired — T55 directly contradicts them.
   - The `reframe` claim (M8) is correctly identified — the instability looks like a feature, not a bug.
   - The `retain` claims (M1, M3) remain valid observations.
   - The `weaken` claims (M2, M6, M9) have correct narrowing.

3. **No claims were silently upgraded.** C4 remains `partial`. No language promotes diagnostic evidence into causal proof. "Causal proof," "mechanism proven," and "root cause identified" appear only in negative context.

4. **The worker stayed within scope.** Only the 5 allowed docs were modified/created. No source code, config, governance docs, or run-root files were touched. No execution was run.

5. **T47 recommendation is properly conditioned.** The gate document explicitly states T47 "can proceed, but only under explicit mechanism-hedge wording" with 5 specific hedge boundaries, and explicitly warns against treating it as unconditional next work.

6. **The second-intervention verdict is honest and bounded.** It is `deferred pending better question` — not auto-approved and not permanently closed.

### Non-blocking observations

I noted a few minor things that don't affect the verdict:
- The worker summary says "10-row claim table" with status counts that sum to 12 (because two rows are `still-open` — this is just a wording quirk, the table itself is correct).
- `docs/08_risks_and_open_questions.md` R10 entry still references T56 as upcoming rather than completed — but the worker correctly did not modify it (it's a Captain governance file).
- The C4 boundary wording in the ledger is now quite long, but the complexity is justified by the evidence.

## 4. Assessment of the worker's own review and explanation

The worker wrote its own review (`docs/review/T56_review.md`, verdict: PASS) and explanation (`docs/for_human/T56_explanation.md`).

### Worker review accuracy

The worker's self-review is substantially correct:
- Scope/boundary checks are accurate
- The claim-table honesty check correctly identifies that no claim was upgraded
- The T47 conditioning check and second-intervention check are accurate
- The recommended next action is reasonable

One minor gap: the worker review does not flag the status-count inconsistency (10 rows with counts summing to 12), though this is trivial.

### Worker explanation accuracy

The worker's human-facing explanation is clear and accurate:
- The plain-language description correctly captures the narrative shift
- The claim classification summary is correct
- The implications for T47 and second intervention are accurately stated

One area where the reviewer explanation (this document) adds value: the worker explanation does not discuss the downstream effect on R10 or the need for Captain to update governance docs (task board, handoff, risks). This document covers those governance implications.

## 5. Summary

T56 is a well-executed docs-only gate task. It honestly reframes the mechanism story after T55's surprising result, correctly retires claims that are contradicted by evidence, and sets appropriate boundaries for the next task. The verdict is **PASS**.
