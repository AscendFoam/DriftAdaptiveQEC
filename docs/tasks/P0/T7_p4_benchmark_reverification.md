# Task Package: T7

Task ID:
`T7`

Goal:
重新验收一条恢复期最小 P4 benchmark 路径，确认 `run_p4_multiscenario_benchmark.py` 能在当前机器上继承 `T6` 已复验的软件 HIL 入口，并把 backend、artifact type、最小 benchmark 过滤条件与新 run 证据写回治理文档。

Why now:
`T6` 已经把恢复期最小 software HIL 路径重新验收到“可复验”状态，但 `P4 benchmark` 入口还没有在同一套 `mock + model_artifact + artifact_npz + inproc` 边界下完成最小复验。如果继续停在 `T6`，后续就仍然只能说“P4 脚本存在”，不能说“当前机器上已再次确认一条最小 P4 benchmark 路径”。

Allowed files:
- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/P4_benchmark_recovery_bootstrap.md`
- `docs/tasks/P0/T7_p4_benchmark_reverification.md`

Forbidden scope:
- `cnn_fpga/`、`physics/` 核心逻辑修改
- 正式长跑配置语义改写
- 真板设备联调
- `.tflite` 真导出主线扩写
- teacher-representation 新长跑
- `runs/`、`artifacts/`、`__pycache__/` 的清理或 untrack

Inputs to read:
- `docs/03_hil_p4_boundary_audit.md`
- `docs/P3_software_hil_bootstrap.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/06_repo_noise_governance.md`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `cnn_fpga/config/p4_multiscenario.yaml`
- `cnn_fpga/config/p4_multiscenario_smoke.yaml`
- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`

Expected output:
1. 一份恢复期专用的最小 P4 benchmark 配置
2. 一次新的 P4 benchmark 最小复验结果
3. 明确写出的 backend / slow-loop mode / inference mode / artifact type / artifact path / scenario / mode 过滤条件
4. 更新后的 task board、decision log、handoff、legacy audit、risk 与 P4 bootstrap 文档
5. 若复验失败，输出可复现的阻塞证据

Verification:
- `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds`
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
- `docs/P4_benchmark_recovery_bootstrap.md`

Reviewer type:
`milestone`
