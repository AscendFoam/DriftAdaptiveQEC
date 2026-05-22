# T46: Multi-seed mechanism/intervention plan and trace pack

## Status

- Proposed by Captain on `2026-05-19`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: docs-only planning gate

## Why This Task Exists

`T36` and `T38` together give a stronger explanation for `seed=20260429`, but that explanation is still single-seed and diagnostic.

`T44` froze the paper-facing truth boundary, `T53` added a bounded theory walkthrough for the current mainline, and `T45` locked the benchmark-expansion lane so that mechanism work does not silently turn into benchmark-scope expansion.

This task therefore exists to define the smallest credible next step for mechanism evidence: a bounded multi-seed and intervention-oriented trace plan that could later test the current hypothesis without overbuilding the task.

## Goal

Produce a docs-only mechanism plan that answers, in writing:

1. what mechanism statement is safe today
2. what stronger mechanism statement is still unsupported
3. what minimal multi-seed or intervention evidence would strengthen the current hypothesis
4. which seeds, traces, summary rows, and comparison fields would be minimally required
5. what should count as diagnostic evidence versus causal evidence
6. how to keep the future execution scope small enough that it remains believable and bounded

## Allowed Files

Worker may modify only:

- `docs/seed_mechanism_multi_seed_plan.md`
- `docs/review/T46_review.md`
- `docs/for_human/T46_explanation.md`
- `docs/tasks/Phase2/T46_multi_seed_mechanism_intervention_plan_and_trace_pack.md`

## Docs To Update

This task should update only:

1. `docs/seed_mechanism_multi_seed_plan.md`
2. `docs/review/T46_review.md`
3. `docs/for_human/T46_explanation.md`
4. this task package itself, only to append Worker output and verification notes after completion

## Forbidden Scope

This task must not:

1. edit any source code, benchmark code, config, test, runtime, hardware, or training file
2. edit `runs/`, `artifacts/`, or create new run/result directories
3. edit governance docs other than this task package itself, including `docs/00_project_snapshot.md`, `docs/04_task_board.md`, `docs/05_decision_log.md`, `docs/07_handoff.md`, `docs/08_risks_and_open_questions.md`, or `docs/02_experiment_plan.md`
4. run benchmark, training, `.tflite`, hardware, cleanup, or any other evidence-producing command
5. rewrite `T36` or `T38` single-seed evidence as if it were already multi-seed confirmation
6. turn `T46` into `T47`, benchmark expansion, deployment validation, mitigation implementation, or a new broad mechanism program

## Required Inputs

Read at minimum:

- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/reality_recovery/00_freeze_snapshot.md`
- `docs/reality_recovery/01_claim_evidence_table.md`
- `docs/reality_recovery/02_code_truth_audit.md`
- `docs/reality_recovery/03_experiment_reproducibility_audit.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`
- `docs/seed20260429_failure_diagnosis.md`
- `docs/seed20260429_trace_export_diagnosis.md`
- `docs/mainline_theory_analysis.md`
- `docs/paper_benchmark_expansion_protocol.md`
- `docs/review/T44_review.md`
- `docs/review/T45_review.md`
- `docs/review/T53_review.md`
- `docs/review/T36_review.md`
- `docs/review/T38_review.md`
- `docs/paper_reviewer_risk_audit.md`

## Minimum Questions T46 Must Answer

1. What is the narrowest mechanism statement that current artifacts can already support?
2. What stronger paper-facing mechanism statement remains unsupported today?
3. If we want to move beyond one seed, what is the minimal seed-selection logic and why?
4. What trace fields and summary rows are strictly required before any later execution task can claim progress?
5. Which candidate intervention ideas are true mechanism tests, and which are actually future mitigation or model-design work?
6. What should be the explicit go/no-go rule for a later bounded execution task?

## Planning Boundaries

This task must stay docs-only.

It may define only:

1. minimal seed-selection logic
2. minimal trace-field requirements
3. minimal summary/comparison pack requirements
4. intervention or counterfactual ideas as future mechanism tests
5. acceptance thresholds for diagnostic versus causal language
6. what must stay out of scope until a later task

The plan should follow these constraints:

1. Anchor the plan on the existing `seed=20260429` diagnosis lane first.
2. If additional seeds are proposed, state the selection logic explicitly instead of turning the plan into a new benchmark run list.
3. Prefer a very small contrast set over a broad sweep; the worker should state an upper-bound philosophy for any future seed pack.
4. Keep frozen-set benchmark semantics separate from mechanism planning; T46 must not reopen T45.
5. Treat intervention ideas as future mechanism-test ideas, not as implemented fixes or new model directions.
6. Keep diagnostic evidence and causal evidence separate; the task may describe a causal path, but it must not claim causal proof.
7. Make explicit which comparison rows are minimally needed, such as per-scenario, per-mode, per-repeat trace rows, cross-seed summary rows, and diagnostic-versus-intervention comparison notes.

## Expected Output

Create `docs/seed_mechanism_multi_seed_plan.md` with:

1. status and scope
2. current supported mechanism statement
3. stronger unsupported claims and remaining evidence gap
4. minimal seed-selection logic
5. minimal trace schema and required file/field inventory
6. minimal future comparison pack
7. intervention or counterfactual matrix
8. diagnostic versus causal evidence boundary
9. go / no-go recommendation for a later execution task
10. explicit non-claims

The plan must contain at least the following concrete tables:

1. a claim-boundary table
2. a seed-selection table
3. a trace-field inventory table
4. a future execution-pack table

Create `docs/review/T46_review.md` with:

1. scope and boundary confirmation
2. whether the current mechanism gap is still represented honestly as single-seed today
3. whether the proposed seed pack, trace pack, and intervention pack are still small enough to execute later
4. whether the plan stays separate from benchmark expansion and deployment validation
5. recommended next bounded task, if any, and why

Create `docs/for_human/T46_explanation.md` with a short human-facing summary.

## Verification

Required verification is documentation-only:

1. confirm only the allowed files changed
2. confirm no source, config, test, `runs/`, or `artifacts` files were modified
3. confirm no benchmark, training, `.tflite`, hardware, cleanup, or other evidence-producing command was started
4. confirm the plan does not claim multi-seed confirmation or causal proof already exists
5. confirm the task remains a stepwise plan, not a broad mechanism project
6. confirm the plan keeps `T46` separate from `T47`, `T48`, and `T49`

## Review No-Go Triggers

Review should be treated as `BLOCK` if any of the following happen:

1. the output upgrades existing single-seed evidence into multi-seed confirmation or causal proof
2. the output silently widens into benchmark expansion, deployment validation, or mitigation implementation
3. the worker edits files outside the allowed list
4. the proposed future execution scope is so broad that it is effectively a new benchmark milestone rather than a bounded mechanism task

## Captain Notes

T46 should answer one practical question:

- what is the smallest believable next step that can move `seed=20260429` from a single-seed diagnosis toward a real mechanism story without pretending that the story is already proven?

The right outcome is not to solve the mechanism inside T46. The right outcome is to freeze a future mechanism-evidence pack that is small, honest, and executable.

## Worker Output

- Worker completed T46 on `2026-05-22`.
- All required inputs read and cross-checked: docs/04_task_board.md, docs/07_handoff.md, docs/08_risks_and_open_questions.md, docs/reality_recovery/00_freeze_snapshot.md, docs/reality_recovery/01_claim_evidence_table.md, docs/reality_recovery/02_code_truth_audit.md, docs/reality_recovery/03_experiment_reproducibility_audit.md, docs/reality_recovery/04_figure_and_result_ledger.md, docs/reality_recovery/05_paper_claim_risk_table.md, docs/seed20260429_failure_diagnosis.md, docs/seed20260429_trace_export_diagnosis.md, docs/mainline_theory_analysis.md, docs/paper_benchmark_expansion_protocol.md, docs/review/T44_review.md, docs/review/T45_review.md, docs/review/T53_review.md, docs/review/T36_review.md, docs/review/T38_review.md, docs/paper_reviewer_risk_audit.md.
- `docs/seed_mechanism_multi_seed_plan.md` produced: 10-section mechanism plan with 4 required tables (claim-boundary, seed-selection, trace-field inventory, execution-pack), intervention matrix, diagnostic-vs-causal evidence boundary, go/no-go rules, and explicit non-claims.
- `docs/review/T46_review.md` produced with verdict = `PASS`, no blocking issues, 4 non-blocking issues (3-seed sample size, intervention clip factor, existing seed reuse, I3 implementation feasibility).
- `docs/for_human/T46_explanation.md` produced with Chinese human-facing explanation.
- Single-seed evidence not upgraded to multi-seed confirmation or causal proof anywhere in the plan.
- Frozen-set benchmark boundary not reopened; plan uses existing frozen four scenarios only.
- No code, config, `runs/`, or `artifacts` changes were introduced.
- No benchmark, training, `.tflite`, hardware, cleanup, or other evidence-producing command was started.

## Verification Record

1. Only allowed files changed: `docs/seed_mechanism_multi_seed_plan.md` (new), `docs/review/T46_review.md` (new), `docs/for_human/T46_explanation.md` (new), and this task package (modified for Worker Output / Verification Record). No other files created or modified.
2. No source code, config, test, `runs/`, or `artifacts` files were modified.
3. No benchmark, training, `.tflite`, hardware, cleanup, or other evidence-producing command was started.
4. The plan does not claim multi-seed confirmation or causal proof already exists. All multi-seed claims are explicitly marked as `unsupported`.
5. The task remains a stepwise plan, not a broad mechanism project. The seed pack is bounded at 6 total, interventions at 1-2 variants, and scenarios at the existing frozen four.
6. The plan keeps T46 separate from T47 (ablation result-pack), T48 (TFLite runtime), and T49 (real-board smoke).
7. The plan correctly preserves C4 as `partial` — single-seed trace-supported diagnostic evidence — and does not silently upgrade it.
8. The plan references and respects the T45 frozen-set benchmark expansion protocol: no scenario expansion, no new baselines, no benchmark code changes.
9. The intervention matrix correctly separates mechanism tests (I1, I2, I3) from model-design work (I4, I5, I6). Recommended intervention (I1) is config-only.
10. Diagnostic vs causal evidence boundary is explicit throughout: current evidence is diagnostic only; causal evidence requires intervention testing across multiple seeds.
