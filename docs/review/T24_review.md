# T24 Adversarial Review

**Task**: T24: P4 bounded formal software revalidation execution
**Reviewer type**: adversarial
**Date**: 2026-05-11

---

## 1. Scope verification

T24 task package (`docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`) defined:

- Goal: execute frozen-set formal software revalidation (4 scenarios x 5 modes x repeats=2)
- Allowed files: `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`, `docs/P4_benchmark_formal_protocol.md`, `docs/04_task_board.md`, `docs/07_handoff.md`, `docs/08_risks_and_open_questions.md`, `runs/p4_benchmark/T24_formal_software_revalidation_*`
- Forbidden: source code edits, config semantic edits, benchmark semantic edits, training, .tflite runtime, hardware commands, cleanup, statcalib, soft-information, extra drift families, CI stopping

**Scope check**: Worker modified only allowed files. No source code changes. No config semantic changes. No training, .tflite, hardware, cleanup, or scope expansion detected. **PASS**.

## 2. Execution verification

### 2.1 CLI shape compliance

The task package specified repeat-chunked execution with full scenario/mode selection in each invocation. The Worker used:

1. Chunk 1: `--repeat-start 0 --repeat-stop 1` with all 4 scenarios and all 5 modes
2. Chunk 2: `--repeat-start 1 --repeat-stop 2` with all 4 scenarios and all 5 modes
3. Final: `--resume-only`

This matches the task package's exact CLI shape. Seed semantics preserved (no single-scenario splitting). **PASS**.

### 2.2 Evidence pack completeness

Verified from run dir `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`:

| File | Exists | Verified |
| --- | --- | --- |
| `summary.json` | Yes | Yes |
| `comparison.csv` | Yes (21 lines = header + 20 rows) | Yes |
| `launch_plan.json` | Yes | Yes |
| `progress.jsonl` | Yes | Yes |
| `delta.csv` | Yes | Yes |
| `teacher_scalar_diagnostics.csv` | Yes (header only, no data) | Yes |
| `report.md` | Yes | Yes |

`launch_plan.json` records `repeats = 2`, `paired_seeds = true`. **PASS**.

### 2.3 Coverage verification

From `summary.json`:
- `missing_runs = []`
- `raw_rows` count = 40
- `comparison.csv`: 20 data rows (4 scenarios x 5 modes)
- All rows have `completed_repeats = 2`, `expected_repeats = 2`, `coverage = 1.0`
- Repeat indices: 0 and 1

**PASS**.

### 2.4 Cross-validation with T15

T15 ran the same config (`p4_multiscenario_strong_baselines.yaml`) on `static_bias_theta` and `linear_ramp` with the same paired seeds. Comparing the overlapping 20 repeat-runs:

| Scenario | Repeat | T15 final_ler | T24 final_ler | Match |
| --- | --- | --- | --- | --- |
| `static_bias_theta` / `hybrid_residual_b` | 0 | 0.812090 | 0.812090 | exact |
| `static_bias_theta` / `hybrid_residual_b` | 1 | 0.809713 | 0.809713 | exact |
| `linear_ramp` / `hybrid_residual_b` | 0 | 0.788194 | 0.788194 | exact |
| `linear_ramp` / `hybrid_residual_b` | 1 | 0.787316 | 0.787316 | exact |

The overlapping runs are bit-for-bit identical, confirming seed consistency and no silent semantic drift between T15 and T24. **PASS**.

### 2.5 Metric availability

From `comparison.csv` header, all required metrics are present:

- `final_ler_mean`, `final_ler_std` — present
- `overflow_rate_mean` — present
- `histogram_input_saturation_rate_mean` — present, equals `overflow_rate_mean`
- `correction_saturation_rate_mean` — present, all 0.0
- `aggressive_param_rate_mean` — present, all 0.0
- `n_commits_applied_mean` — present
- `slow_update_violation_rate_mean` — present, all 0.0
- `fast_cycle_violation_rate_mean` — present, ~1.5e-05
- `teacher_contribution_l2_mean_mean` — present, all 0.0
- `teacher_scalar_abs_mean_mean` — present, all 0.0
- `teacher_gate_mean_mean` — present, all 0.0
- `teacher_gate_std_mean` — present, all 0.0

All requested metrics are present in the CSV. Zero-valued metrics are explicitly reported as evidence gaps, not silently omitted. **PASS**.

### 2.6 Backend verification

All 40 `raw_rows` report `backend = ""` (the runner does not emit a non-empty backend field for mock runs, but the protocol confirms mock-backed execution). The `launch_plan.json` and runner config confirm software HIL with no `board_backend` involvement. **PASS**.

### 2.7 Config hash and git commit

- `config_hash`: `b82874392710` — recorded in summary.json
- `git_commit`: `0c82ee1` — recorded in summary.json, matches the commit that created the T24 task package

**PASS**.

## 3. Boundary statement audit

The Worker's output contains explicit boundary statements in:

- `docs/P4_benchmark_formal_protocol.md` Section 15.7: "mock-backed software HIL formal software revalidation only"
- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md` Worker Output: "Mock-backed software HIL only"
- `docs/08_risks_and_open_questions.md` R5: updated to reflect T24 completion but maintained "mock-backed software HIL" qualifier

No instance of `.tflite runtime`, `real_board`, or `paper-grade` being claimed as validated. **PASS**.

## 4. Risk update audit

R5: Updated from "中高" to "中", accurately reflecting that T24 completed the formal frozen-set revalidation but is still mock-backed. Description and mitigation are updated appropriately.

R9: Narrowed to reflect T24 completion, correctly states that further expansion still needs new task packages.

R19: Marked as "已收口" (closed), correctly states that T24 fixed the CLI shape and reported metric availability.

**PASS**.

## 5. Governance document consistency

### 5.1 `docs/04_task_board.md`

- T24 marked as `[x]`
- T24 result summary line items match verified data
- Current Unique Task updated to T25 (pending)
- Captain Output updated to reflect T24 completion

### 5.2 `docs/07_handoff.md`

- Date updated to `2026-05-11`
- Item 39 added with accurate T24 execution summary
- Section 4 judgment list updated (items 26-28)
- Section 6 updated from pending task to completion summary
- Section 7 next-step updated to T25
- Section 8 hold items preserved

### 5.3 `docs/08_risks_and_open_questions.md`

- R5, R9, R19 updated as noted above
- Current T24 status note updated
- R13/R14 preserved unchanged (still valid)

**PASS**.

## 6. Pseudo-implementation / mock / stub / hardcode check

No pseudo-implementation detected. The Worker:

1. Ran actual benchmark commands (not mocked or stubbed)
2. Recorded actual CLI commands with actual output
3. Reported actual metric values from `comparison.csv` and `summary.json`
4. The zero-valued metrics (`correction_saturation_rate_mean`, teacher diagnostics) are actual runner outputs, not fabricated — they match T15 observations and are reported as evidence gaps

**PASS**.

## 7. Over-engineering check

The changes are proportional to the task:

- Protocol document: one new section (Section 15) recording execution results
- Task package: one new section (Worker Output) recording commands, verification, and results
- Governance docs: standard status updates

No unnecessary abstractions, no speculative content, no scope creep. **PASS**.

## 8. What T24 did NOT do (correctly)

T24 correctly did not:

1. Modify source code
2. Modify config semantics
3. Run training
4. Run `.tflite` runtime
5. Run hardware commands
6. Execute cleanup
7. Add `statcalib` or soft-information comparators
8. Add extra drift families
9. Implement CI-driven stopping
10. Claim results as paper-grade, `.tflite` runtime, or `real_board` evidence

## 9. Per-scenario result consistency check

The four scenarios show consistent ranking:

1. `hybrid_residual_b` wins all four
2. `ukf` is runner-up in all four
3. Gaps are 0.014-0.023 — consistent with T15 (0.014-0.023 for the overlapping two scenarios)

LER values are in the expected range (~0.78-0.84). No outliers or suspicious values. **PASS**.

## 10. Remaining evidence gaps (correctly reported by Worker)

1. **Teacher diagnostics all-zero**: `teacher_scalar_diagnostics.csv` has only a header row. `teacher_contribution_l2_mean`, `teacher_scalar_abs_mean`, `teacher_gate_mean`, `teacher_gate_std` are all 0.0 for all modes. This is consistent with T15 and is a known mechanism-analysis gap. Non-blocking for LER ranking but limits mechanistic interpretability of `hybrid_residual_b`.

2. **`correction_saturation_rate_mean` = 0.0 for all rows**: Either a metric collection limitation or genuine absence of correction saturation. The Worker correctly reports this rather than silently omitting it.

3. **`repeats=2` only**: Sufficient for formal revalidation of the historical frozen set, but does not provide confidence intervals.

4. **No `.tflite` runtime, no `real_board`, no paper-grade expansion**: All correctly deferred.

## Verdict

**PASS_WITH_WARNINGS**

### Blocking issues

None.

### Non-blocking issues

N1: `correction_saturation_rate_mean` is structurally zero for all 20 rows across all modes and scenarios. While the Worker correctly reports this as a gap, the root cause (metric collection bug vs. genuine zero) is still unresolved. A future mechanism-audit task should trace whether the runner's correction saturation detection logic is wired correctly for the current parameter regime. If the code path for computing this metric has a dead branch for these parameter settings, the metric's structural zero should be documented as "not applicable for this regime" rather than "0.0 observed".

N2: The `04_task_board.md` diff adds a one-line tip: `全局建议：运行代码可以使用conda的DLEnv环境(重环境)，也可以直接使用conda的默认python环境(轻环境)。` This is a minor out-of-pattern addition — it is helpful context but does not belong to the T24 task's allowed scope (it is not a T24 execution result). It should be treated as a minor governance sync note and does not affect the T24 verdict.

N3: `teacher_scalar_diagnostics.csv` has a header but zero data rows. This has been consistent since T15 and has been deferred through T16, T23, and now T24. The deferred chain is now four tasks long. While it remains non-blocking for LER ranking, the growing deferred chain increases the risk that this gap will be forgotten. Recommend explicitly scheduling a mechanism-evidence audit (T27 or equivalent) as the next priority after T25 gate review.

### Missing validation

None beyond what the task package specified. All 9 verification checks pass.

### Suspicious implementation details

None. The cross-validation with T15 shows bit-for-bit identical overlapping runs, confirming reproducibility and seed integrity.

### Recommended next action

1. Captain accepts T24 as `PASS_WITH_WARNINGS`.
2. Create T25 task package for adversarial gate review of T24 evidence.
3. After T25, prioritize the teacher-diagnostics mechanism audit (T27 or equivalent) to break the four-task deferred chain on R10/N3.
4. Maintain the current boundary: T24 formal revalidation is mock-backed software HIL only — not `.tflite`, not `real_board`, not paper-grade.
