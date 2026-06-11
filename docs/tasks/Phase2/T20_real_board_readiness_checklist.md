# T20: Real-Board Readiness Checklist

Task ID: `T20`

Goal: 为真板 HIL 路径补 readiness checklist 与缺口清单，不实现真板 backend，也不改写 placeholder 语义。

Why now: 真板路径是项目高风险边界。只有先固定设备、地址表、权限、RTL/driver 证据需求，后续才能安全推进真实板级任务。

Allowed files:

- `docs/tasks/Phase2/T20_real_board_readiness_checklist.md`
- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Forbidden scope:

- 不改 `cnn_fpga/hwio/board_backend.py`
- 不改 `cnn_fpga/hwio/fpga_driver.py`
- 不写真实板级完成
- 不运行需要硬件设备的命令
- 不把 mock backend 证据外推到 real board

Inputs to read:

- `docs/03_hil_p4_boundary_audit.md`
- `cnn_fpga/hwio/board_backend.py`
- `cnn_fpga/hwio/fpga_driver.py`
- `cnn_fpga/benchmark/run_hil_suite.py`

Expected output:

- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- 包含：
  - 当前 placeholder 证据
  - 真板任务前置条件
  - 所需设备/地址/权限/日志
  - 最小 smoke 验收标准
  - 禁止表述

Verification:

- 只读审计，无硬件调用。

Docs to update:

- `docs/03_hil_p4_boundary_audit.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/04_task_board.md`（Captain 整合阶段）

Reviewer type: `adversarial`

## Worker Output Summary

- Output type: `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- Read-only audit confirmed:
  - `board_backend.py` remains placeholder-oriented
  - `fpga_driver.py` still frames `board/real` as future-facing integration
  - `run_hil_suite.py` does not upgrade mock-backed evidence into real-board evidence
- Hardware commands were not executed
- Board-side code was not modified
- Updated docs:
  - `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
  - `docs/03_hil_p4_boundary_audit.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
