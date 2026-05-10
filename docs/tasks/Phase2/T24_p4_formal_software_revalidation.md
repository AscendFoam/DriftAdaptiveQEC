# T24: P4 Bounded Formal Software Revalidation

Task ID: `T24`

Goal: 执行 `T23` 已锁定的 P4 frozen-set formal software revalidation，补齐四场景、五模式、`repeats=2` 的 evidence pack。该任务只验证 mock-backed software HIL formal benchmark，不验证 `.tflite` runtime，不验证真板。

Why now: `docs/review/T23_review.md` verdict = `PASS_WITH_WARNINGS`，blocking issues 为无。Captain 接受 `T23` gate：`GO_FOR_BOUNDED_FORMAL_SOFTWARE_REVALIDATION` + `NO_GO_FOR_SCOPE_EXPANSION_INSIDE_T24`。

Allowed files:

- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
- `docs/P4_benchmark_formal_protocol.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `runs/p4_benchmark/T24_formal_software_revalidation_*`

Inputs to read:

- `docs/P4_benchmark_formal_protocol.md`
- `docs/review/T23_review.md`
- `docs/P4_benchmark_development_protocol.md`
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
- Update `docs/P4_benchmark_formal_protocol.md` with a T24 execution record:
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

- `docs/P4_benchmark_formal_protocol.md`
- `docs/tasks/Phase2/T24_p4_formal_software_revalidation.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Reviewer type: `adversarial`
