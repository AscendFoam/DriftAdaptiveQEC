# T24: P4 Bounded Formal Software Revalidation

Task ID: `T24`

Goal: 执行 `T23` 已锁定的 P4 frozen-set formal software revalidation，补齐四场景、五模式、`repeats=2` 的 evidence pack。该任务只验证 mock-backed software HIL formal benchmark，不验证 `.tflite` runtime，不验证真板。

Why now: `docs/review/T23_review.md` verdict = `PASS_WITH_WARNINGS`，blocking issues 为无。Captain 接受 `T23` gate：`GO_FOR_BOUNDED_FORMAL_SOFTWARE_REVALIDATION` + `NO_GO_FOR_SCOPE_EXPANSION_INSIDE_T24`。

Allowed files:

- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_*`

Inputs to read:

- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/review/T23_review.md`
- `docs/protocols/benchmark/P4_benchmark_development_protocol.md`
- `docs/review/T15_frozen_smoke_review.md`
- `docs/review/T16_p4_evidence_gate_review.md`
- `docs/08_risks_and_open_questions.md`
- `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

Exact CLI shape:

Use `C:\ProgramData\anaconda3\python.exe` and one fixed run directory. The run directory name may add a timestamp suffix, but it must start with `runs/p4_benchmark/T24_formal_software_revalidation_`.

Primary repeat-chunked shape:

```powershell
$runDir = "runs/p4_benchmark/T24_formal_software_revalidation_YYYYMMDD_HHMMSS"

& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark `
  --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml `
  --scenario static_bias_theta `
  --scenario linear_ramp `
  --scenario step_sigma_theta `
  --scenario periodic_drift `
  --mode ekf `
  --mode ukf `
  --mode constant_residual_mu `
  --mode rls_residual_b `
  --mode hybrid_residual_b `
  --paired-seeds `
  --repeats 2 `
  --run-dir $runDir `
  --repeat-start 0 `
  --repeat-stop 1

& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark `
  --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml `
  --scenario static_bias_theta `
  --scenario linear_ramp `
  --scenario step_sigma_theta `
  --scenario periodic_drift `
  --mode ekf `
  --mode ukf `
  --mode constant_residual_mu `
  --mode rls_residual_b `
  --mode hybrid_residual_b `
  --paired-seeds `
  --repeats 2 `
  --run-dir $runDir `
  --repeat-start 1 `
  --repeat-stop 2

& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark `
  --config cnn_fpga/config/p4_multiscenario_strong_baselines.yaml `
  --scenario static_bias_theta `
  --scenario linear_ramp `
  --scenario step_sigma_theta `
  --scenario periodic_drift `
  --mode ekf `
  --mode ukf `
  --mode constant_residual_mu `
  --mode rls_residual_b `
  --mode hybrid_residual_b `
  --paired-seeds `
  --repeats 2 `
  --run-dir $runDir `
  --resume-only
```

Important seed rule:

- Do not split execution by a single `--scenario` at a time. In the current runner, filtering to one scenario changes the local `scenario_idx` used in seed construction. Repeat-based chunking preserves the full scenario order and seed semantics.

Expected output:

- One fixed run directory under `runs/p4_benchmark/T24_formal_software_revalidation_*`
- Required evidence pack:
  - `launch_plan.json`
  - `progress.jsonl`
  - `summary.json`
  - `comparison.csv`
  - `delta.csv`
  - `teacher_scalar_diagnostics.csv`
  - `report.md`
  - each repeat directory's `hil_summary.json`
  - each repeat directory's `repeat_status.json`
- Update `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` with a T24 execution record:
  - exact commands or exact chunking shape
  - run dir
  - config path
  - config hash
  - git commit from `summary.json`
  - `missing_runs`
  - coverage per scenario/mode
  - per-scenario winners and runner-up gaps
  - metric availability table
  - explicit boundary statement: mock-backed software HIL only
- Update this task package with Worker output:
  - changed files
  - commands run
  - verification results
  - remaining risks

Forbidden scope:

- 不改源码。
- 不改 config 文件语义。
- 不改 benchmark 口径、baseline 集合、scenario 集合、seed 语义或 ParamMapper 主线语义。
- 不运行训练。
- 不运行 `.tflite` export/runtime。
- 不调用硬件命令，不运行 `backend=board` HIL。
- 不执行 cleanup，不改写历史 `runs/` / `artifacts/` 作为新事实来源。
- 不新增 `statcalib`、soft-information / correlation-aware comparator、额外 drift family、CI-driven stopping rule 或 teacher-representation 长跑。
- 不把本任务结果写成 `.tflite` runtime、`real_board` validation 或 paper-grade expanded benchmark。

Verification:

1. Confirm `summary.json` exists.
2. Confirm `missing_runs = []`.
3. Confirm `comparison.csv` has exactly `4 scenarios x 5 modes = 20` scenario/mode rows.
4. Confirm every row has `completed_repeats = 2`, `expected_repeats = 2`, and `coverage = 1.0`.
5. Confirm `raw_rows` in `summary.json` corresponds to `40` repeat-runs, if that field is present.
6. Confirm `launch_plan.json` records `repeats = 2` and `paired_seeds = true`.
7. Confirm `comparison.csv` contains or explicitly lacks:
   - `histogram_input_saturation_rate_mean`
   - `correction_saturation_rate_mean`
   - `fast_cycle_violation_rate_mean`
8. If any requested metric is absent, report it as a T24 evidence gap instead of silently omitting it.
9. Confirm the final writeup states that this is still mock-backed software HIL evidence, not `.tflite` runtime evidence and not `real_board` evidence.

Failure handling:

- If a command times out but partial repeat outputs exist, continue only with the same `$runDir`.
- If any repeat remains missing after the final `--resume-only`, report `T24 incomplete` and do not claim formal revalidation completion.
- If the runner errors due to environment or code behavior, stop and report the blocking issue; do not edit source in this task.

Docs to update:

- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md`
- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Reviewer type: `adversarial`

## Worker Output

### Changed files

- `docs/protocols/benchmark/P4_benchmark_formal_protocol.md` — added T24 execution record (Section 15)
- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md` — this Worker output section
- `docs/04_task_board.md` — T24 status update
- `docs/07_handoff.md` — T24 completion record
- `docs/08_risks_and_open_questions.md` — risk updates

### Commands run

All commands used `C:\ProgramData\anaconda3\python.exe` with config `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` and fixed run dir `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`.

1. **Chunk 1 (repeat 0)**: `--scenario static_bias_theta --scenario linear_ramp --scenario step_sigma_theta --scenario periodic_drift --mode ekf --mode ukf --mode constant_residual_mu --mode rls_residual_b --mode hybrid_residual_b --paired-seeds --repeats 2 --run-dir <run_dir> --repeat-start 0 --repeat-stop 1`
   - Completed: 20/20 repeat-runs (repeat 0 for all scenario/mode pairs)
   - Exit code: 0

2. **Chunk 2 (repeat 1)**: Same CLI with `--repeat-start 1 --repeat-stop 2`
   - Completed: 20/20 repeat-runs (repeat 1 for all scenario/mode pairs)
   - Exit code: 0

3. **Resume-only**: Same CLI with `--resume-only`
   - All 40 repeat-runs resumed from existing output
   - Exit code: 0

Total wall-clock: approximately 20 hours (started 2026-05-10 20:08, completed 2026-05-11 ~15:51).

### Verification results

1. `summary.json` exists: **PASS**
2. `missing_runs = []`: **PASS**
3. `comparison.csv` has exactly 20 scenario/mode rows: **PASS**
4. All rows have `completed_repeats = 2`, `expected_repeats = 2`, `coverage = 1.0`: **PASS**
5. `raw_rows` count = 40: **PASS**
6. `launch_plan.json` records `repeats = 2` and `paired_seeds = true`: **PASS**
7. Metric availability:
   - `histogram_input_saturation_rate_mean`: present, non-zero (equals overflow_rate)
   - `correction_saturation_rate_mean`: present, all 0.0
   - `fast_cycle_violation_rate_mean`: present, non-zero but very small (~1.5e-05)
8. Evidence gaps reported (not silently omitted):
   - `correction_saturation_rate_mean` = 0.0 for all rows
   - `teacher_scalar_diagnostics.csv` has header only (no data rows)
   - All teacher diagnostic metrics = 0.0 (consistent with T15/T16 deferred gap)
   - `delta.csv` = all null (expected: strong-baseline config excludes `static_linear`/`cnn_fpga`)
9. Boundary statement: mock-backed software HIL only: **PASS**

### Per-scenario results

| Scenario | Winner | Winner LER | Runner-Up | Gap |
| --- | --- | ---: | ---: | ---: |
| `static_bias_theta` | `hybrid_residual_b` | 0.810902 | `ukf` | 0.014469 |
| `linear_ramp` | `hybrid_residual_b` | 0.787755 | `ukf` | 0.023446 |
| `step_sigma_theta` | `hybrid_residual_b` | 0.788800 | `ukf` | 0.022748 |
| `periodic_drift` | `hybrid_residual_b` | 0.806392 | `ukf` | 0.015166 |

### Remaining risks

1. Teacher diagnostics (`hybrid_residual_b`) remain all-zero — mechanism analysis gap deferred from T15/T16, non-blocking for LER ranking.
2. `correction_saturation_rate_mean` is always 0.0 — may indicate metric collection limitation or genuine absence of correction saturation in these parameter regimes.
3. This run is mock-backed software HIL only — not `.tflite` runtime, not `real_board`, not paper-grade expanded benchmark.
4. `repeats=2` with fixed paired seeds is sufficient for formal revalidation of the historical frozen set, but does not provide confidence intervals or CI-driven stopping.
5. `statcalib`, soft-information comparators, extra drift families, true `.tflite` runtime and real-board smoke remain deferred to later tasks.
