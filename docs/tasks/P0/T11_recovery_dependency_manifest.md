# Task Package: T11

Task ID:
`T11`

Goal:
在仓库根目录补一份恢复期最小依赖 manifest，并把它正式接到 `P0/P3/P4 recovery smoke` 的 bootstrap 与治理文档里，避免后续会话继续完全依赖“本机解释器记忆”。

Why now:
`T10` 的 gate review 已明确给出 `Continue Repair`，其中最直接、最可收口的缺口就是：根目录仍没有一份作用域诚实的最小依赖说明。继续缺失这份 manifest，会让当前已经恢复出来的 `P0/P3/P4 recovery smoke` 路径仍然停留在“只能在熟悉这台机器的人手里复用”的状态。

Allowed files:
- `README.md`
- `requirements-recovery.txt`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/tasks/P0/T11_recovery_dependency_manifest.md`

Forbidden scope:
- `cnn_fpga/`、`physics/` 核心逻辑修改
- 正式 benchmark 主线语义改写
- `frozen_baseline_set`、ParamMapper 或统计口径变更
- 真板设备联调
- `.tflite` 真导出主线扩写
- 新训练、新 benchmark 长跑、teacher-representation 新扩展
- `runs/`、`artifacts/`、`__pycache__/` 的清理或 untrack

Inputs to read:
- `README.md`
- `AGENTS.md`
- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/02_experiment_plan.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
- `docs/review/T10_gate_review.md`
- `benchmark/compare_full_vs_simplified_ler.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `cnn_fpga/runtime/inference_service.py`
- `cnn_fpga/utils/config.py`

Expected output:
1. 一份根目录下的 recovery-scoped 最小依赖 manifest
2. 明确写出的覆盖范围与不覆盖范围
3. 更新后的 README 与 P0/P3/P4 bootstrap 文档
4. 更新后的 task board、decision log、handoff、legacy audit 与风险文档

Verification:
- `Get-Content -Raw -Encoding UTF8 "requirements-recovery.txt"`
- `& 'C:\ProgramData\anaconda3\python.exe' -c "import numpy, yaml; print(numpy.__version__); print(yaml.__version__)"`
- `& 'C:\ProgramData\anaconda3\python.exe' benchmark/compare_full_vs_simplified_ler.py --help`
- `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --help`
- `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --help`

Docs to update:
- `README.md`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/recovery_bootstrap/P0_smoke_bootstrap.md`
- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`

Reviewer type:
`normal`
