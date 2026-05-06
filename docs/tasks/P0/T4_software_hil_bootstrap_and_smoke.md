# Task Package: T4

Task ID:
`T4`

Goal:
补出一条无真板歧义、可在当前机器上直接复用的软件 HIL 最小 bootstrap / smoke path，并把 backend、artifact type、推荐解释器与验证命令固定写入文档。

Why now:
`T3` 已经把 HIL / P4 的真实性边界澄清清楚。下一步如果没有一条实际可跑的软件 HIL 最小路径，这些边界仍然只是“说清楚了”，还没有被恢复为可复验入口。

Allowed files:
- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- `docs/P3_software_hil_bootstrap.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/01_legacy_audit.md`
- `docs/tasks/P0/T4_software_hil_bootstrap_and_smoke.md`

Forbidden scope:
- `cnn_fpga/` 核心逻辑代码修改
- 真板设备访问
- 正式 P4 benchmark 长跑
- teacher-representation 分支扩展

Inputs to read:
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/config/hardware_hil.yaml`
- `cnn_fpga/runtime/slow_loop_runtime.py`
- `cnn_fpga/runtime/inference_service.py`
- `cnn_fpga/model/tiny_cnn.py`
- `artifacts/models/static_theta_v2/`

Expected output:
1. 一份可直接复用的软件 HIL bootstrap 文档
2. 一份最小 software HIL smoke 配置
3. 一条已执行或已明确阻塞的软件 HIL 最小命令
4. backend / artifact type / interpreter 的固定口径

Verification:
- 配置复核：
  - `Get-Content -Raw -Encoding UTF8 "cnn_fpga/config/hardware_hil_recovery_smoke.yaml"`
- 运行命令：
  - `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 结果复核：
  - `Get-Content -Raw -Encoding UTF8 "<run_dir>/hil_summary.json"`
  - `Get-Content -Raw -Encoding UTF8 "<run_dir>/hil_events.json"`

Docs to update:
- `docs/P3_software_hil_bootstrap.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/01_legacy_audit.md`

Reviewer type:
`milestone`
