# T49 Review

## Verdict

`PASS_WITH_WARNINGS`

`T49` 的有界目标已经完成：worker 交付了 task-scoped helper、focused tests、四份 task-scoped artifacts、主报告和面向人的说明；当前结论也保持了边界诚实，明确收口为：

`NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

我做了不重跑长实验的轻量复核，结果与 worker 报告一致：

1. `python -m py_compile cnn_fpga/hwio/build_t49_real_board_smoke_gate.py` 成功
2. `python -m unittest tests.test_t49_real_board_smoke_execution_gate` 成功，`Ran 4 tests`, `OK`
3. `C:\ProgramData\anaconda3\python.exe --version` 为 `Python 3.12.7`
4. `cmd /c ver` 为 `Microsoft Windows [Version 10.0.26200.8457]`
5. `/dev/uio0`、`/dev/uio1`、`\\\\.\\XilinxDMA`、`\\\\.\\XDMA`、`\\\\.\\uio0`、`\\\\.\\uio1` 当前都不存在
6. `pnputil /enum-devices /connected | Select-String 'Xilinx|AMD|FPGA|XDMA|UIO'` 无命中
7. forbidden 边界 diff 均为空：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`
   - `git diff --name-only -- requirements-recovery.txt requirements-train-cpu-win-py312.txt requirements-tflite-win-py311.txt`

没有发现把只读 probe 写成“真板 smoke 已成功”、没有发现写侧 MMIO/DMA/寄存器动作、也没有发现越界修改 `board_backend.py` / `axi_map.py` / `dma_client.py` 或治理文档。

## Blocking issues

- 无。

## Non-blocking issues

- `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py:129-149` 的 `device_path_truth` 判定只检查 `read_only_openable` 的数量是否 `>= 2`，没有利用 probe 里已经存在的 `role` 字段区分 `mmio` 和 `dma`。这意味着未来如果出现“两条可打开路径其实都指向同一类角色”的情况，helper 可能把 device 层过早判成 `ready`。当前提交的真实 artifact 是 `openable_count = 0`，因此这不影响本次 `NO_GO` 结论，但逻辑上仍偏松。

## Missing tests

- 补一个 `device_path_truth` 角色回归测试：至少要求 `1` 条 `mmio` 路径和 `1` 条 `dma` 路径都可只读打开，才允许进入 host/device ready，而不是只按“可打开路径总数”计数。
- 补一个“读取当前 checked-in artifacts” 的回归测试：直接喂给 helper `artifacts/t49_real_board_smoke_execution_gate/*.json`，断言最终 verdict 仍是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。这样后续若有人改 helper 逻辑或 probe schema，可以及时发现与当前事实包脱节。

## Suspicious implementation details

- `artifacts/t49_real_board_smoke_execution_gate/host_fact_manifest.json`、`device_path_probe.json`、`code_side_audit.json` 是本次最核心的证据文件，但 diff 里没有对应的 checked-in 生成脚本；仓库里交付的是“结果文件 + gate helper”，而不是“完整 probe 生成链”。这不等于伪实现，因为我已经对其中关键事实做了独立交叉验证：
  - 宿主版本、解释器版本与当前环境一致
  - 候选设备路径不存在与当前环境一致
  - `board_backend.py` placeholder 证据、`axi_map.py` 地址表、`dma_client.py` 结构约定与源码一致
  但它确实意味着：当前事实包的再生路径仍偏手工 / 临时命令式，而不是完全由 repo 内单一入口重建。
- `service_name_clues = [Ndisuio, RDMANDK, uiomap]` 只能算弱线索，不能当作板卡存在或驱动可用的正证据。worker 文档里已经基本保持了这个边界，后续引用时仍要继续压住这条口径。

## Recommended next action

- 接受 `T49` 在其有界范围内完成，并把当前主结论固定为：“仓库存在可审计的 board-path scaffolding 与 AXI/DMA 代码事实，但当前这台 Windows 宿主没有可读打开的真板设备路径，因此当前只能给出诚实的 `NO_GO`。”
- 如果这条 lane 还要继续复用，建议单开一个很小的后续任务，只做两件事：
  1. 把 `device_path_truth` 改成 role-aware 判定，并补相应测试；
  2. 为 host/device/code-side probe 补一个 checked-in 的只读生成入口，提升事实包再生性。
- 真板执行类工作不要在 `T49` 上继续追加。只有当目标宿主真实暴露出可读设备节点，并且 bitstream / RTL / DMA contract 也能绑定到当前主机时，才应单开新的 bounded real-board execution task。
