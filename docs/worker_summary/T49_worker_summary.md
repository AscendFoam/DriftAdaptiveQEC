# T49 Worker Summary

## 改了什么

本轮只改了 `T49` 允许路径：

- 新增 helper：`cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
- 新增 focused tests：`tests/test_t49_real_board_smoke_execution_gate.py`
- 写入 task-scoped artifacts：`artifacts/t49_real_board_smoke_execution_gate/`
- 新增主报告：`docs/t49_real_board_smoke_execution_gate.md`
- 新增 review 草稿：`docs/review/T49_review.md`
- 新增人类解释：`docs/for_human/T49_explanation.md`
- 新增本文件：`docs/worker_summary/T49_worker_summary.md`

## 如何验证

实际执行并记录了：

1. 真实 host fact probe
   - `artifacts/t49_real_board_smoke_execution_gate/host_fact_manifest.json`
2. 真实 device-path read-only probe
   - `artifacts/t49_real_board_smoke_execution_gate/device_path_probe.json`
3. 真实 code-side AXI/DMA/placeholder audit
   - `artifacts/t49_real_board_smoke_execution_gate/code_side_audit.json`
4. `python -m py_compile cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
5. `python -m unittest tests.test_t49_real_board_smoke_execution_gate`
6. helper 真执行
   - `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`
7. 边界检查
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

## 关键结果

- host probe 解释器：`C:\ProgramData\anaconda3\python.exe`
- Python：`3.12.7`
- 当前宿主：Windows 11，`cmd /c ver = Microsoft Windows [Version 10.0.26200.8457]`
- 设备路径事实：
  - `/dev/uio0` 不存在
  - `/dev/uio1` 不存在
  - `\\\\.\\XilinxDMA` 不存在
  - `\\\\.\\XDMA` 不存在
  - `\\\\.\\uio0` 不存在
  - `\\\\.\\uio1` 不存在
- 设备/驱动线索：
  - `pnputil /enum-devices /connected` 未匹配到 `Xilinx|AMD|FPGA|XDMA|UIO`
  - 只枚举到弱线索服务名：`Ndisuio`、`RDMANDK`、`uiomap`
- bitstream / contract 证据：
  - 仅存在 config-level 记录：`bitstream_version = fpga_linear_v1`
  - 当前仓库 `fpga/` 下未发现 bitstream 文件
  - RTL 地址表、DMA contract、Q4.20 板侧一致性均未确认
- `board_backend.py` 当前仍属于 placeholder-only execution path
- 最终 gate verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

## 当前最强可支持的表述

当前只支持：

“仓库存在 board-path scaffolding，且代码侧 AXI/DMA 事实可被审计；但当前这台 Windows 主机上没有可读打开的真板设备路径，因此还不具备进入 bounded real-board smoke 的最小前提。”

## 剩余风险

1. 当前 `NO_GO` 由 host/device 层先触发，但 bitstream / RTL / DMA contract 和 repo placeholder 也都还没闭环；后续不能因为补齐单一层就提前升级口径。
2. `service_name_clues` 里的 `Ndisuio / RDMANDK / uiomap` 不是板卡存在证据，只能算弱线索。
3. 当前探测在现有权限下无法使用 `Get-CimInstance` / `Get-PnpDevice` / `systeminfo`，因此 host fact 主要来自 `platform`、注册表和 `pnputil`；若未来需要更强宿主证据，必须在更合适权限或目标板卡宿主上补 probe。
