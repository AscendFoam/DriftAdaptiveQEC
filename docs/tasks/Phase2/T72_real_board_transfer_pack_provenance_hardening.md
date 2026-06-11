# T72：真板 transfer-pack provenance 去硬编码与 override 回归加固

## 状态

- 由 Captain 在 `2026-06-11` 基于 `T71` closeout 提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：有界 deployment-boundary provenance hardening 任务

## 为什么现在做这个任务

`T71` 已经把 `T49` 的 current-host honest `NO_GO` 收敛成了一个 checked-in、只读、role-aware、可 replay / regeneration 的 real-board gate 包：

- `device_path_truth` 已改为 `mmio + dma` role-aware 判定
- 已有仓库内 collector：`cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
- `T49` checked-in artifact replay 与 current-host regeneration verdict 已验证一致

但 `T71 review` 同时指出，当前 transfer-pack 的剩余问题已经不在主 verdict，而在 provenance 严谨度：

1. `probe_limitations` 有未实际探测却写成当前事实的固定文案
2. `source_records` 与 `expected_byte_count_basis` 仍偏默认 config 写死
3. `--config` / `--mmio-path` / `--dma-path` 的 provenance/override focused regression 不足

因此，`T72` 的目标不是继续靠近真板执行，而是把 `T71` 这套 pack 再收紧一层，让 future-host transfer-pack 的文字说明也和执行上下文保持一致。

## 目标

在不改变 `T71` 主 verdict 的前提下，完成以下四件事：

1. 去掉 collector 中“未探测即写死”的 provenance 文案
2. 让 config/path provenance 变成 execution-derived / override-aware
3. 补齐 `--config` / `--mmio-path` / `--dma-path` 的 focused regression
4. 产出一份新的 hardened current-host pack 与报告，明确说明：
   - 哪些字段现在变成动态推导
   - 哪些字段仍只是 code-side / config-side 线索
   - 为什么 `T37` 仍然 blocked

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T72_real_board_transfer_pack_provenance_hardening.md`
- `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
- `docs/review/T72_review.md`
- `docs/for_human/T72_explanation.md`
- `docs/worker_summary/T72_worker_summary.md`
- `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
- `tests/test_t71_real_board_gate_regeneration_pack.py`
- `tests/test_t72_real_board_transfer_pack_provenance_hardening.py`
- `artifacts/t72_real_board_transfer_pack_provenance_hardening/`

说明：

- `T72` 允许读取 `T49` / `T71` 的 checked-in artifacts，但不得改写它们
- `T72` 不需要也不得修改 `build_t49_real_board_smoke_gate.py`，除非出现无法避免且可证明必要的 bug；若出现这种情况，应在 review 中明确解释
- `T72` 允许在测试里创建临时 YAML / 临时目录，但不得向仓库新增新的 canonical config 文件

## Docs To Update

Worker 必须更新：

- `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
- `docs/review/T72_review.md`
- `docs/for_human/T72_explanation.md`
- `docs/worker_summary/T72_worker_summary.md`

Worker 不得更新治理文档；治理同步由 Captain 在 review 后统一处理。

## Forbidden Scope

Worker 不得：

- 修改 `docs/02_experiment_plan.md`
- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `runs/` 下文件
- 修改 `cnn_fpga/hwio/board_backend.py`
- 修改 `cnn_fpga/hwio/axi_map.py`
- 修改 `cnn_fpga/hwio/dma_client.py`
- 修改任何 canonical config（包括但不限于 `cnn_fpga/config/hardware_hil.yaml`）
- 修改 `T49` / `T71` 的历史 artifact、历史 report 或历史 review verdict
- 发起任何 write-side MMIO / DMA / register 操作
- 运行 real-board smoke、real-board benchmark、P3/P4 benchmark、sidecar 实验
- 混入 theory 分支工作，或把 main 分支任务输出写到 theory 分支语义里
- 把 `T72` 写成真板 ready、真板执行成功、`P3 real-board HIL complete`、`hardware_validated` 或 `deployment closure`

## 必须复用的输入

Worker 必须复用以下既有输入，而不是改写历史事实：

- `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
- `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`
- `docs/review/T49_review.md`
- `docs/review/T71_review.md`
- `docs/worker_summary/T71_worker_summary.md`
- `artifacts/t49_real_board_smoke_execution_gate/host_fact_manifest.json`
- `artifacts/t49_real_board_smoke_execution_gate/device_path_probe.json`
- `artifacts/t49_real_board_smoke_execution_gate/code_side_audit.json`
- `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`
- `artifacts/t71_real_board_gate_regeneration_pack/host_fact_manifest.json`
- `artifacts/t71_real_board_gate_regeneration_pack/device_path_probe.json`
- `artifacts/t71_real_board_gate_regeneration_pack/code_side_audit.json`
- `artifacts/t71_real_board_gate_regeneration_pack/current_host_regenerated_gate.json`
- `artifacts/t71_real_board_gate_regeneration_pack/t49_checked_in_replay_gate.json`
- `artifacts/t71_real_board_gate_regeneration_pack/replay_vs_regeneration_comparison.json`

## 固定边界

- 主线分支：当前 `main` experiment branch only
- 宿主边界：当前机器 only，用于 read-only collector / regeneration / replay 验证
- 证据边界：real-board host / device / bitstream / AXI / DMA / repo-path truth only
- 输出边界：只允许写入 `artifacts/t72_real_board_transfer_pack_provenance_hardening/`
- 非目标边界：不是 board execution 任务，不是 benchmark 任务，不是 `.tflite` 任务，不是 paper 任务，不是 theory 任务

## 任务要求

### A. `probe_limitations` 必须改成“执行来源可区分”的 provenance

Worker 必须修正 collector，使 reviewer 能分辨：

1. 哪些 probe 确实执行过
2. 哪些 probe 执行失败
3. 哪些 probe 在当前平台上根本未尝试

最低要求：

- 不允许继续把未执行的探测写成固定的“access denied”事实
- 若命令实际执行失败，应记录命令、returncode、stderr 或等价失败信息
- 若某探测当前平台未执行，应显式标成 `not_probed` / `not_applicable` / 等价状态，而不是伪装成失败事实

### B. config/path provenance 必须改成动态推导

Worker 必须让以下说明字段跟随实际执行上下文变化，而不是继续写死默认值：

1. `source_records`
2. `repo_board_defaults` 里与 config/path 直接相关的字段说明
3. `expected_byte_count_basis`

最低要求：

- `--config` 指向非默认 config 时，artifact 中的 config/path provenance 必须反映该文件，而不是继续写死 `hardware_hil.yaml`
- `--mmio-path` / `--dma-path` override 时，artifact 中的相关 provenance 必须反映 override，而不是继续保留默认宿主叙事
- `expected_byte_count_basis` 必须由当前 `histogram_shape` / `dtype` / `buffer_bytes` 推导，而不是固定一句 `32 x 32 float32 -> 4096 bytes under current config defaults`

### C. focused regression 必须覆盖 override 与 provenance integrity

至少补齐以下测试：

1. 一个 `--config` override regression
   - 验证 provenance 随 config 变化
2. 一个 `--mmio-path` / `--dma-path` override regression
   - 验证 probe 与 provenance 同时反映 override
3. 一个 `probe_limitations` integrity regression
   - 验证未执行的探测不会再被写成“当前事实”
4. 一个 verdict stability regression
   - 确认 hardened provenance 改动不会让 `T49` replay / current-host regeneration 的 verdict 漂移

### D. 最终文档必须回答的问题

`docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md` 至少要回答：

1. `T72` 是否改变了 `T71` / `T49` 的 current-host final verdict
2. 哪些 provenance 字段从“写死文案”改成了“执行导出”
3. `--config` / `--mmio-path` / `--dma-path` 的 override 现在如何被反映到 artifact 中
4. `T37` 为什么仍然 blocked
5. 这次 hardening 后，transfer-pack 现在最强能支持什么说法、仍然不能支持什么说法

### E. strongest supported claim

`T72` 最终文档只能支持类似如下口径：

“仓库当前已有一套 checked-in、只读、role-aware、可 replay / regeneration 的 real-board gate 包；`T72` 进一步把其中的 provenance 从默认文案推进为更贴近实际执行上下文的动态导出，并补齐了 override 相关回归。当前 Windows 宿主再生成后的 verdict 仍然是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。这说明 mainline 进一步收紧了 future-host transfer-pack 的严谨度，但这仍不等于真板执行成功、real-board validation、`P3 real-board HIL complete` 或 deployment closure。”

### F. 不允许写出的结论

`T72` 不得写出：

- `real-board smoke executed successfully`
- `T37 ready to execute on current host`
- `real-board host-transfer pack fully production ready`
- `hardware_validated`
- `P3 real-board HIL complete`
- `deployment closure`

## 预期输出

Worker 必须产出：

- `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
- `docs/review/T72_review.md`
- `docs/for_human/T72_explanation.md`
- `docs/worker_summary/T72_worker_summary.md`
- `tests/test_t72_real_board_transfer_pack_provenance_hardening.py`
- 必要时更新：
  - `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
  - `tests/test_t71_real_board_gate_regeneration_pack.py`
- `artifacts/t72_real_board_transfer_pack_provenance_hardening/`

## 验证

Worker 必须实际执行并报告：

1. `python -m py_compile cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
2. `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
3. `python -m unittest tests.test_t72_real_board_transfer_pack_provenance_hardening`
4. 一次 current-host 只读 artifact regeneration，输出到 `artifacts/t72_real_board_transfer_pack_provenance_hardening/`
5. 一次用 `T72` artifact 驱动 gate helper 的真实执行
6. 一次用 `T49` checked-in artifact 驱动 gate helper 的 replay 执行
7. 至少一次 `--config` 或 path override 的 focused regression 证据（可通过单测或 task-scoped temp execution 完成，但必须写清如何验证）
8. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- cnn_fpga/config`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

Worker 还必须显式报告：

1. hardened 前后 final verdict 是否变化
2. replay verdict 与 current-host regeneration verdict 是否仍一致
3. 哪些 provenance 字段改成了动态推导
4. 哪些 future-host 缺口仍然没有关闭

## Review No-Go Triggers

Reviewer 在以下任一情况应返回 `BLOCK`：

1. Worker 让 `T49` replay 或 current-host regeneration 的 verdict 漂移，且没有强证据解释为什么这是正确修复
2. Worker 仍把未执行的 probe 限制写成当前事实
3. Worker 越界修改 `board_backend.py` / `axi_map.py` / `dma_client.py` / canonical config / 治理文档
4. Worker 发起任何 write-side MMIO/DMA/register 操作
5. Worker 把 `T72` 写成真板 ready / 真板执行成功 / `hardware_validated`
6. Worker 把 main 分支实验主线任务与 theory 分支工作混在一起

## Captain 备注

- `T72` 是 `T71` 的 provenance 收口任务，不是 `T37` 解锁任务。
- `T72` 的价值不在于更接近跑真板，而在于避免未来把“默认文案”误当成“当前执行事实”。
- theory 分支继续独立存在；本任务只在 main 分支实验主线内推进。

## Worker Output

### 改了什么

- 更新 `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
  - 新增结构化 `probe_execution_records`
  - `probe_limitations` 改成 execution-derived 的结构化记录，不再写死固定 `access denied` 文案
  - `repo_board_defaults` 新增 `config_path`、`config_argument_kind`、`board_source_record`、`bitstream_version_source_record`、`candidate_mmio_path_record`、`candidate_dma_path_record`
  - `bitstream_evidence.source_records` 改为动态记录当前实际 config 路径与字段值
  - `dma_contract.expected_byte_count_basis` 改为按 `histogram_shape`、`dtype`、`buffer_bytes` 动态推导
- 新增 `tests/test_t72_real_board_transfer_pack_provenance_hardening.py`
  - 覆盖 `--config` override provenance
  - 覆盖 `--mmio-path` / `--dma-path` override provenance
  - 覆盖 `probe_limitations` integrity
  - 覆盖 hardening 后 verdict 稳定性
- 写入新的 task-scoped outputs：
  - `artifacts/t72_real_board_transfer_pack_provenance_hardening/host_fact_manifest.json`
  - `artifacts/t72_real_board_transfer_pack_provenance_hardening/device_path_probe.json`
  - `artifacts/t72_real_board_transfer_pack_provenance_hardening/code_side_audit.json`
  - `artifacts/t72_real_board_transfer_pack_provenance_hardening/current_host_regenerated_gate.json`
  - `artifacts/t72_real_board_transfer_pack_provenance_hardening/t49_checked_in_replay_gate.json`
  - `artifacts/t72_real_board_transfer_pack_provenance_hardening/replay_vs_regeneration_comparison.json`
- 新增/更新文档：
  - `docs/evidence_packs/deployment_boundary/t72_real_board_transfer_pack_provenance_hardening.md`
  - `docs/review/T72_review.md`
  - `docs/for_human/T72_explanation.md`
  - `docs/worker_summary/T72_worker_summary.md`

### 如何验证

实际执行了：

1. `python -m py_compile cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
2. `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
3. `python -m unittest tests.test_t72_real_board_transfer_pack_provenance_hardening`
4. `python -m cnn_fpga.hwio.collect_t71_real_board_gate_artifacts --output-dir artifacts/t72_real_board_transfer_pack_provenance_hardening`
5. 一次用 `T72` artifact 驱动 gate helper 的真实执行
6. 一次用 `T49` checked-in artifact 驱动 gate helper 的 replay 执行
7. `--config` / `--mmio-path` / `--dma-path` 的 focused regression 证据，通过 `tests/test_t72_real_board_transfer_pack_provenance_hardening.py` 完成
8. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- cnn_fpga/config`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

### 关键结果

- `T49` replay verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- `T72` current-host regeneration verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- `replay_vs_regeneration_comparison.json` 显示：
  - `verdict_match = true`
  - `strongest_statement_match = true`
  - `device_path_truth_status_match = true`
  - `bitstream_truth_status_match = true`
  - `repo_execution_path_truth_status_match = true`

### 剩余风险

- `T72` 只收紧了 transfer-pack provenance，不等于当前或 future-host 已具备真板执行前提。
- 当前仍缺：
  - 可只读打开的真实 `mmio` / `dma` 设备路径
  - bitstream / RTL / DMA / fixed-point contract 绑定事实
  - 非-placeholder 的 board execution path
- 因此 `T37` 继续 blocked 仍是正确结果。
