# T25: P4 Formal Evidence Gate Review And Result-Boundary Update

Task ID: `T25`

Goal: 对 `T24` 的 frozen-set formal software revalidation evidence pack 做只读 gate review，明确 T24 结果可以支持的 claim、不能支持的 claim、仍需 deferred 的机制/部署缺口，并推荐下一唯一任务但不执行。

Why now: `docs/review/T24_review.md` verdict = `PASS_WITH_WARNINGS`，blocking issues 为无。Captain 已接受 T24 完成，但 T24 仍留下 `correction_saturation_rate_mean` structural zero 与 teacher diagnostics header-only 两个机制证据缺口，需要在继续推进前做 result-boundary 收口。

Allowed files:

- `docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md`
- `docs/review/T25_p4_formal_evidence_gate_review.md`
- `docs/for_human/T25_explanation.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Inputs to read:

- `docs/02_experiment_plan.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/review/T24_review.md`
- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/teacher_scalar_diagnostics.csv`
- `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/report.md`

Expected output:

- A gate review document at `docs/review/T25_p4_formal_evidence_gate_review.md` containing:
  - Verdict: `PASS` / `PASS_WITH_WARNINGS` / `BLOCK`
  - T24 evidence completeness check
  - T24 metric availability and zero-metric interpretation check
  - Result-boundary statement
  - Accepted / deferred / rejected classification for T24 warnings
  - Recommended next unique task
- A human-readable explanation at `docs/for_human/T25_explanation.md`.
- Update this task package with Worker output:
  - changed files
  - commands / read-only checks run
  - verification result
  - remaining risks

Forbidden scope:

- Do not run new benchmark commands.
- Do not modify source code.
- Do not modify config semantics.
- Do not modify benchmark protocol, baseline set, scenario set, seed semantics, or ParamMapper mainline semantics.
- Do not run training.
- Do not run `.tflite` export/runtime.
- Do not call hardware commands, and do not run `backend=board` HIL.
- Do not execute cleanup.
- Do not edit `runs/` or `artifacts/`; they are read-only evidence inputs for this task.
- Do not add `statcalib`, soft-information / correlation-aware comparator, extra drift family, CI-driven stopping rule, or teacher-representation long run.
- Do not write T24 as `.tflite` runtime evidence, `real_board` validation, or paper-grade expanded benchmark.

Verification:

1. Confirm T25 review cites the T24 run dir exactly: `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`.
2. Confirm T25 review states whether T24 can be treated as completed frozen-set formal software revalidation.
3. Confirm T25 review explicitly states T24 is still mock-backed software HIL only.
4. Confirm T25 review classifies:
   - `correction_saturation_rate_mean` structural zero
   - `teacher_scalar_diagnostics.csv` header-only / teacher diagnostics all-zero
   - T24 task-board environment-note warning
5. Confirm any deferred issue is present in `docs/08_risks_and_open_questions.md`.
6. Confirm governance docs name one next unique task and do not mark it complete.
7. Confirm no new run directory is created.

Docs to update:

- `docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md`
- `docs/review/T25_p4_formal_evidence_gate_review.md`
- `docs/for_human/T25_explanation.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Reviewer type: `adversarial`

## Worker Output

### Changed files

- `docs/tasks/Phase2/T25_p4_formal_evidence_gate_review.md` — added this Worker output section
- `docs/review/T25_p4_formal_evidence_gate_review.md` — added adversarial gate-review draft
- `docs/for_human/T25_explanation.md` — added human-readable explanation
- `docs/04_task_board.md` — added T25 draft-review status note without marking T25 complete
- `docs/05_decision_log.md` — added T25 draft-state decision note
- `docs/07_handoff.md` — added T25 draft conclusion and next-task recommendation note
- `docs/08_risks_and_open_questions.md` — added T25 draft-status note and open-question updates

### Commands / read-only checks run

1. Read task inputs and governance docs:
   - `README.md`
   - `AGENTS.md`
   - `docs/02_experiment_plan.md`
   - `docs/04_task_board.md`
   - `docs/07_handoff.md`
   - `docs/08_risks_and_open_questions.md`
   - `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
   - `docs/review/T24_review.md`
   - `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
2. Read T24 evidence files:
   - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json`
   - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`
   - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/teacher_scalar_diagnostics.csv`
   - `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/report.md`
3. Read-only grep/checks:
   - verified T25-related anchors in `docs/04_task_board.md`
   - verified deferred risk anchors `R10` and `R20`
   - verified no pre-existing T25 review/explanation files with conflicting content

No benchmark, training, `.tflite`, hardware, cleanup, or source-code command was run.

### Verification result

1. T25 review cites the exact T24 run dir `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`: **PASS**
2. T25 review states T24 can be treated as completed frozen-set formal software revalidation: **PASS**
3. T25 review explicitly states T24 is still mock-backed software HIL only: **PASS**
4. T25 review classifies:
   - `correction_saturation_rate_mean` structural zero: **PASS**
   - `teacher_scalar_diagnostics.csv` header-only / teacher diagnostics all-zero: **PASS**
   - T24 task-board environment-note warning: **PASS**
5. Deferred issues present in `docs/08_risks_and_open_questions.md`:
   - `R10`
   - `R20`
   Result: **PASS**
6. Governance docs name one next unique task and do not mark it complete:
   - recommended next task is `T27`
   - `Current Unique Task` remains `T25`
   Result: **PASS**
7. No new run directory created: **PASS**

### Remaining risks

1. `T24` still has no mechanism-level closure for teacher diagnostics; R10 remains open.
2. `correction_saturation_rate_mean` remains structurally zero across all 20 rows; R20 remains open.
3. Even after T25 draft acceptance, the benchmark boundary must stay at mock-backed software HIL and must not be rewritten as `.tflite`, `real_board`, or paper-grade evidence.
4. This turn intentionally does not mark `T25` complete; Captain/manual review still needs to decide whether to accept the draft and switch the next unique task.
