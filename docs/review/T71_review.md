# T71 Review

## Verdict

`PASS_WITH_WARNINGS`

`T71` 的核心目标已经完成，而且完成方式基本符合任务包边界：

- `build_t49_real_board_smoke_gate.py` 的 `device_path_truth` 已从“只按 openable path 总数计数”改成 role-aware 的 `mmio + dma` 双角色判定
- 新增了 checked-in 的只读 collector：`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
- 新增 / 更新了 focused tests：
  - `tests/test_t49_real_board_smoke_execution_gate.py`
  - `tests/test_t71_real_board_gate_regeneration_pack.py`
- 当前宿主 regeneration 与 `T49` checked-in artifact replay 的最终 verdict 一致，仍然是：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- 没有发现 write-side MMIO / DMA / register 操作
- 没有发现越界修改 `board_backend.py`、`axi_map.py`、`dma_client.py`、`runs/` 或治理文档
- 文档没有把本任务写成真板执行成功、真板 ready、`P3 real-board HIL complete` 或 deployment closure

我做了不重跑长实验的轻量复核，结果如下：

1. `python -m unittest tests.test_t49_real_board_smoke_execution_gate`
   - `Ran 6 tests`, `OK`
2. `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
   - `Ran 2 tests`, `OK`
3. 用临时 `.pyc` 路径验证了两份脚本可编译：
   - `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
   - `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
4. 在临时目录真实执行了一次 collector + gate helper
   - 结果仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
5. 对 `artifacts/t49_real_board_smoke_execution_gate/*.json` 做了一次 replay
   - 结果仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
6. 边界 diff 为空：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

## Blocking issues

- 无。

## Non-blocking issues

- `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py:278-283` 把 Windows `probe_limitations` 直接写成固定的 “access denied” 文案，但 collector 本身并没有实际执行这些 `Get-CimInstance` / `Get-PnpDevice` / `systeminfo` 探针。这不会改变当前 gate verdict，因为 helper 根本不依赖这些字段，但它会把“旧任务里观察到的限制”写成“当前 / future-host 重新探测出来的事实”，削弱 host-transfer pack 的 provenance 严谨度。
- `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py:367-370` 与 `:473` 仍保留了面向当前默认配置的硬编码描述：
  - `source_records` 固定写成 `cnn_fpga/config/hardware_hil.yaml: hil.board=ZCU111` 和 `hil.bitstream_version=fpga_linear_v1`
  - `expected_byte_count_basis` 固定写成 `32 x 32 float32 histogram -> 4096 bytes under current config defaults`
  但该脚本又同时暴露了 `--config`、`--mmio-path`、`--dma-path` 作为 future-host 迁移入口。也就是说，一旦 future-host 用的是非默认 config 或不同板卡配置，artifact 中的部分 provenance 文本可能变成“字段值是新的，说明文字还是旧的”。这不影响当前主结论，但与 `T71` 的 future-host portability 目标相比还不够干净。

## Missing tests

- 缺一个 collector 回归测试，显式覆盖 `--config` 非默认值时的 provenance 字段同步问题。
  - 现在的测试只验证 replay / regeneration verdict 一致，没有验证 `source_records`、`repo_board_defaults`、`expected_byte_count_basis` 是否随 config 真正变化。
- 缺一个 collector 回归测试，验证 `--mmio-path` / `--dma-path` override 不仅影响 probe path，也不会让说明性 provenance 字段继续保留默认宿主 / 默认配置的话术。
- 缺一个“probe limitations 来源”回归测试。
  - 当前没有测试能防止以后继续把未实际探测的限制文案写成事实。

## Suspicious implementation details

- `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py` 的 collector 是真实可运行的，不是 mock，也不是 stub；但它内部确实混合了“真实探测结果”和“继承自 T49 当前宿主的固定说明文字”。其中最明显的是：
  - `probe_limitations` 固定列表：`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py:278-283`
  - `bitstream_evidence.source_records` 固定列表：`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py:367-370`
  - `dma_contract.expected_byte_count_basis` 固定说明：`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py:473`
  这些都属于“不会改 verdict、但会让 provenance 说明变松”的实现细节。
- collector 通过 `from cnn_fpga.hwio.board_backend import BoardFPGAConfig` 读取 board-side 默认配置：`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py:18,520`。这本身没有越界修改 `board_backend.py`，但它意味着 future-host 迁移包仍然依赖 placeholder backend 模块可导入，而不是一个完全独立的轻量 collector。当前环境下这没有出问题，但从 host-transfer 工具成熟度来看，仍然偏“复用现有模块”而不是“最小依赖入口”。

## Recommended next action

- 接受 `T71` 在当前有界范围内完成，并把它视为：
  - `T49` current-host honest `NO_GO` 的代码化再生成加固
  - 不是 `T37` 解锁
  - 更不是任何真板执行成功或真板 ready 结论
- 如果要把这套 pack 当成“未来候选宿主上的标准迁移入口”长期复用，建议单开一个极小后续任务，只做两件事：
  1. 去掉 collector 中未探测即写死的 provenance 文案，让 `probe_limitations`、`source_records`、`expected_byte_count_basis` 都真正从当前执行上下文推导出来；
  2. 补上 `--config` / `--mmio-path` / `--dma-path` 的 collector regression tests。
- 在这些 warning 收口前，`T37` 仍应保持 blocked；当前最强表述仍然只是：
  - 仓库已经有一个 checked-in、只读、role-aware、可 replay / regeneration 的 real-board gate 包
  - 当前宿主重放后仍然是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
