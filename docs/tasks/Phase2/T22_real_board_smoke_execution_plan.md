# T22: Real-Board Smoke Execution Plan

Task ID: `T22`

Goal: 为后续真板 smoke 制定可执行计划，补齐平台确认、AXI/register map 审计路径与量化验收阈值；本任务仍不调用硬件、不实现真板 backend。

Why now: `T21` milestone gate verdict = `Conditional`，建议下一步优先收口 `R13/R14`。`T20` 已有 readiness checklist，但 reviewer 指出寄存器名来源、量化阈值和权限模型仍需具体化。直接执行真板 smoke 风险过高，先补 execution plan。

Allowed files:

- `docs/tasks/Phase2/T22_real_board_smoke_execution_plan.md`
- `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`
- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Forbidden scope:

- 不改 `cnn_fpga/hwio/board_backend.py`
- 不改 `cnn_fpga/hwio/fpga_driver.py`
- 不改 `cnn_fpga/benchmark/run_hil_suite.py`
- 不调用硬件命令
- 不运行 `backend=board` 的 HIL
- 不写 real-board HIL 已完成
- 不改 benchmark 口径

Inputs to read:

- `docs/review/T20_review.md`
- `docs/review/T21_phase2_milestone_review.md`
- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- `docs/03_hil_p4_boundary_audit.md`
- `cnn_fpga/hwio/board_backend.py`
- `cnn_fpga/hwio/fpga_driver.py`
- `cnn_fpga/benchmark/run_hil_suite.py`
- 如存在，仅读审计：`cnn_fpga/hwio/axi_map.py`、`cnn_fpga/hwio/dma_client.py`

Expected output:

- `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`
- 必须包含：
  - target platform decision points: Linux / Windows / WSL / remote board host
  - AXI/register map audit checklist
  - DMA buffer audit checklist
  - quantitative acceptance thresholds draft
  - minimum command plan, marked as not executed
  - required evidence pack for future hardware run
  - prohibited wording

Verification:

- 只读审计。
- 可使用 `rg` / `Get-Content` 查看源码与文档。
- 不执行硬件命令、不运行 benchmark、不写 run dir。

Docs to update:

- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/04_task_board.md`（Captain 整合阶段）

Reviewer type: `adversarial`

## Worker Output Summary

本轮 Worker 已产出以下 plan-only 输出，供后续 reviewer / Captain 审核：

- 新增 `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`
  - 明确 Linux / Windows / WSL / remote board host 四种宿主决策点
  - 直接基于 `cnn_fpga/hwio/axi_map.py` 补 AXI/register map 审计清单
  - 直接基于 `cnn_fpga/hwio/dma_client.py` 补 DMA buffer 审计清单
  - 补 Layer A-D 量化阈值草案、fail-fast budget、minimum command plan（marked not executed）、future evidence pack 和 prohibited wording
- 更新 `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
  - 补充后续真板任务必须先决定宿主模型，不得默认 `/dev/uio*` 就是目标执行模型
- 更新 `docs/03_hil_p4_boundary_audit.md`
  - 明确 `T22` 只可产出 execution plan，不可写成 hardware validated
- 更新 `docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md`
  - 同步 `T22` plan-only worker output 和“待审核，不等于真板已完成”的状态

本轮未调用硬件命令，未运行 `backend=board` HIL，未修改 `board_backend.py` / `fpga_driver.py` / `run_hil_suite.py`。

## Captain Closeout

- Review: `docs/review/T22_review.md`
- Verdict accepted by Captain: `PASS_WITH_WARNINGS`
- Blocking issues: none
- Warning classification:
  - N1 out-of-scope governance files: `accepted`
    - Captain confirms these were governance synchronization edits from the T21/T22 integration stage, not a T22 Worker scope violation.
  - N2 `AXI_REGISTER_MAP` preflight prints dataclass repr: `deferred`
    - Future hardware execution tasks should format the register map explicitly.
  - N3 `byte_count = 4096` depends on `32 x 32 float32`: `deferred`
    - Future hardware execution tasks must confirm actual bitstream / DMA contract before treating this as validated.
- Next unique task after Project Manager clarification: `T23: P4 formal benchmark protocol lock and evidence gap audit`
