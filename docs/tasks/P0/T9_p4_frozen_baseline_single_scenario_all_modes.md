# Task Package: T9

Task ID:
`T9`

Goal:
重新验收一个恢复期 `P4 frozen baseline` 单场景全模式 smoke path，确认 `run_p4_multiscenario_benchmark.py` 能在当前机器上针对正式冻结的四类 baseline `static_linear / window_variance / ekf / cnn_fpga` 完成一次新的单场景复验，并把 backend、artifact type、过滤条件与新 run 证据写回治理文档。

Why now:
`T7` 已经证明当前机器上可以重新跑通一条最小 `P4` recovery smoke path，但当时只覆盖了 `static_linear + cnn_fpga` 两种 mode。`T8` 的 gate review 已经明确指出，这还不足以支撑项目从 `Repair` 进入 `Go`。因此下一步最合理的收口任务，就是在不扩大到正式多场景长跑的前提下，把单场景的 mode 集扩到正式冻结的四类 baseline。

Allowed files:
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`

Forbidden scope:
- `cnn_fpga/`、`physics/` 核心逻辑修改
- 正式 benchmark 主线语义改写
- 真板设备联调
- `.tflite` 真导出主线扩写
- teacher-representation 新长跑
- `runs/`、`artifacts/`、`__pycache__/` 的清理或 untrack

Inputs to read:
- `docs/03_hil_p4_boundary_audit.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/06_repo_noise_governance.md`
- `docs/review/T8_gate_review.md`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `cnn_fpga/config/p4_multiscenario.yaml`
- `cnn_fpga/config/p4_multiscenario_smoke.yaml`
- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/summary.json`
- `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/comparison.csv`

Expected output:
1. 一次新的 `P4 frozen baseline` 单场景全模式 smoke 复验结果
2. 明确写出的 backend / slow-loop mode / inference mode / artifact type / artifact path / scenario / mode 过滤条件
3. 更新后的 task board、decision log、handoff、legacy audit、risk 与 P4 bootstrap 文档
4. 若复验失败，输出可复现的阻塞证据，而不是扩大修改范围

Verification:
- `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds`
- `Get-Content -Raw -Encoding UTF8 "<run_dir>/summary.json"`
- `Get-Content -Raw -Encoding UTF8 "<run_dir>/comparison.csv"`
- `Get-Content -Raw -Encoding UTF8 "<run_dir>/delta.csv"`
- `Get-Content -Raw -Encoding UTF8 "<run_dir>/report.md"`

Docs to update:
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`

Reviewer type:
`milestone`
