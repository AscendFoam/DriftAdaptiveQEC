# Task Package: T3

Task ID:
`T3`

Goal:
审计 HIL / P4 链路中的 `mock`、`stub`、`placeholder` 与真实路径边界，并把这些边界固定写入治理文档，避免后续会话或报告误写项目完成度。

Why now:
`T2` 已经把 P0 smoke 收口为可复用闭环。下一步最容易造成误判的，不是 P0 环境，而是 P3/P4 链路里哪些环节是真实软件实现、哪些只是工程近似、回退路径或真板占位骨架。

Allowed files:
- `docs/01_legacy_audit.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/tasks/P0/T3_hil_p4_boundary_audit.md`

Forbidden scope:
- `cnn_fpga/` 代码逻辑修改
- benchmark 配置与正式结果口径调整
- teacher-representation 分支扩展
- 真板设备访问与硬件验证

Inputs to read:
- `docs/01_legacy_audit.md`
- `docs/07_handoff.md`
- `docs/CNN_FPGA_GKP_阶段结论.md`
- `docs/CNN_FPGA_GKP_后续仿真与工程补强计划.md`
- `cnn_fpga/benchmark/run_hil_suite.py`
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
- `cnn_fpga/hwio/fpga_driver.py`
- `cnn_fpga/hwio/mock_fpga.py`
- `cnn_fpga/hwio/board_backend.py`
- `cnn_fpga/model/export.py`
- `cnn_fpga/runtime/inference_service.py`

Expected output:
1. 一份明确的 HIL / P4 边界审计文档
2. 一张覆盖 `mock` / `stub` / `placeholder` / real path 的边界矩阵
3. 一组后续文档必须遵守的口径规则
4. 对 `T4/T6/T7` 的前置约束说明

Verification:
- 代码证据复核：
  - `Get-Content -Raw -Encoding UTF8 "cnn_fpga/benchmark/run_hil_suite.py"`
  - `Get-Content -Raw -Encoding UTF8 "cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py"`
  - `Get-Content -Raw -Encoding UTF8 "cnn_fpga/hwio/fpga_driver.py"`
  - `Get-Content -Raw -Encoding UTF8 "cnn_fpga/hwio/mock_fpga.py"`
  - `Get-Content -Raw -Encoding UTF8 "cnn_fpga/hwio/board_backend.py"`
  - `Get-Content -Raw -Encoding UTF8 "cnn_fpga/model/export.py"`
  - `Get-Content -Raw -Encoding UTF8 "cnn_fpga/runtime/inference_service.py"`
- 关键关键词复核：
  - `rg -n "backend_name|run_hil_session|Placeholder real-board backend|tflite_stub_v1|tflite_stub_service|tflite_service" cnn_fpga`

Docs to update:
- `docs/01_legacy_audit.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/05_decision_log.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Reviewer type:
`milestone`
