# T53: Mainline theory analysis document for the full GKP correction loop

## Status

- Created by Captain on `2026-05-19`
- Current phase: `Phase 2: Controlled Development`
- Decision state: `Go`
- Task type: docs-only theory/explanation task

## Verification Record

Worker completed on `2026-05-19`.

### Output shape check

1. `docs/mainline_theory_analysis.md` — exists; explains the current mainline from approximate GKP definition to fast/slow-loop closed correction with formulas and implementation anchors.
2. `docs/review/T53_review.md` — exists; verdict `PASS`; blocking issues none.
3. `docs/for_human/T53_explanation.md` — exists; provides a short human-facing explanation of what the theory document is and is not.

### Boundary check

1. No source, config, test, `runs/`, or `artifacts` files were modified: yes.
2. No benchmark, training, `.tflite`, or hardware execution was started: yes.
3. The theory document explicitly separates theory, code-truth, evidence anchors, and blocked claims: yes.
4. The document keeps current blocked boundaries blocked: yes.

### Mainline alignment check

The theory document is aligned to the current repository mainline on the following points:

1. approximate GKP lattice constant and modulo-syndrome semantics
2. noisy syndrome measurement and logical-error accumulation
3. `ParamMapper` covariance-to-`(K, b)` mapping
4. teacher baselines as slow-loop predictors
5. runtime-consistent feature construction and `hybrid_residual_b`
6. `LinearRuntime` fixed-point fast path
7. `ParamBank` stage/commit switching
8. AXI register-map I/O contract

## Why This Task Exists

The user requested a project-specific theory analysis document, similar in spirit to `docs/reference/延伸改进思路.md`, but grounded in the current repository mainline rather than in extension ideas.

This task exists to produce a rigorous, readable explanation of the complete correction loop:

- approximate GKP code definition
- encoding and logical structure
- displacement error and syndrome generation
- teacher input/output semantics
- CNN input/output semantics
- `ParamMapper` mapping into runtime `(K, b)`
- FPGA fast-loop I/O and decode execution
- the complete slow-loop/fast-loop closed correction cycle

It is an explanation task, not a benchmark task and not a paper-claim upgrade task.

## Goal

Write one human-facing mainline theory document that:

1. starts from the approximate GKP code and lattice picture
2. derives the linear-decoding view used by the repository
3. explains why the project uses a dual-loop design
4. explains the teacher-guided residual-b mainline in formula form
5. maps every major stage to the actual repo implementation
6. uses already documented project numbers only as explanatory anchors
7. clearly separates:
   - theory
   - implementation contract
   - currently supported evidence
   - blocked / not-yet-validated deployment claims

## Allowed Files

Worker may modify only:

- `docs/tasks/Phase2/T53_mainline_theory_analysis_document.md`
- `docs/mainline_theory_analysis.md`
- `docs/review/T53_review.md`
- `docs/for_human/T53_explanation.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`

## Required Inputs

Read at minimum:

- `README.md`
- `AGENTS.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/reference/延伸改进思路.md`
- `docs/reference/CNN_FPGA_GKP_工程化实验方案.md`
- `docs/reality_recovery/00_freeze_snapshot.md`
- `docs/reality_recovery/01_claim_evidence_table.md`
- `docs/paper_claim_evidence_ledger.md`
- `physics/README.md`
- `cnn_fpga/data/README.md`
- `cnn_fpga/decoder/README.md`
- `cnn_fpga/runtime/README.md`
- `cnn_fpga/hwio/README.md`
- `cnn_fpga/model/README.md`
- `cnn_fpga/benchmark/README.md`

Also inspect the mainline implementation files needed to keep the theory aligned:

- `physics/gkp_state.py`
- `physics/syndrome_measurement.py`
- `physics/error_correction.py`
- `physics/logical_tracking.py`
- `cnn_fpga/decoder/param_mapper.py`
- `cnn_fpga/decoder/linear_runtime.py`
- `cnn_fpga/decoder/window_baseline.py`
- `cnn_fpga/runtime/feature_builder.py`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/runtime/fast_loop_emulator.py`
- `cnn_fpga/runtime/param_bank.py`
- `cnn_fpga/runtime/inference_service.py`
- `cnn_fpga/runtime/noise_bridge.py`
- `cnn_fpga/model/tiny_cnn.py`
- `cnn_fpga/hwio/axi_map.py`
- `cnn_fpga/data/runtime_dataset_builder.py`

## Boundary Rules

This task must stay docs-only.

It may:

- explain theory
- restate formulas already implemented in the repository
- explain I/O contracts
- use already documented result numbers as explanatory anchors
- explicitly point out where historical stronger claims are not current mainline verified truth

It must not:

- run benchmark
- run training
- run `.tflite`
- call hardware
- modify source code, config, tests, `runs/`, or `artifacts`
- silently upgrade evidence levels
- treat historical stronger docs as current validated deployment truth
- rewrite blocked evidence as completed

## Expected Output

Create `docs/mainline_theory_analysis.md` with the following shape:

1. scope and non-claims
2. approximate GKP code definition and lattice picture
3. encoded logical information and modulo-syndrome measurement
4. displacement error, measurement noise, and logical-error accumulation
5. why the repository uses linear runtime decoding `Δ = K s + b`
6. teacher prediction semantics and classical baseline role
7. runtime-consistent CNN input/output and residual-b semantics
8. `ParamMapper` mapping from `(sigma, mu_q, mu_p, theta)` to `(K, b)`
9. fast-loop fixed-point decode, AXI register contract, and parameter-bank switching
10. complete closed-loop correction cycle with formulas and implementation anchors
11. how current project data/results support the interpretation
12. boundaries, unresolved gaps, and how to read this document safely

Create `docs/review/T53_review.md` with:

1. scope confirmation
2. code-vs-doc alignment check
3. evidence-boundary honesty check
4. any non-blocking issues
5. final verdict

Create `docs/for_human/T53_explanation.md` with a short human-facing summary.

## Verification

Required verification is documentation-only:

1. confirm no source, config, `runs/`, or `artifacts` files were modified
2. confirm no benchmark, training, `.tflite`, or hardware execution was started
3. confirm the theory document distinguishes mainline truth from historical or blocked evidence
4. confirm formulas and I/O descriptions align with current code paths

## Reviewer Type

Adversarial.

Focus areas:

- whether the theory text overstates current evidence
- whether formulas match current implementation rather than extension ideas
- whether teacher/CNN/FPGA I/O semantics are stated precisely
- whether blocked `.tflite` / real-board / expanded benchmark claims remain blocked

## Captain Notes

This task is intentionally for explanation, not for evidence expansion.

The best output is not a grand narrative.  
It is a precise technical walkthrough that lets a human understand:

- what the project is mathematically doing
- what is actually implemented
- why the current mainline became `teacher-guided residual-b`
- how to connect the formulas to the current repository and current evidence boundary
