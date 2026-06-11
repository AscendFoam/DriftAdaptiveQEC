# T49：真实板级 smoke execution gate 与当前宿主硬件事实包

## 状态

- 由 Captain 于 `2026-06-10` 在 `T48` closeout 后提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：有界真板前提 gate 任务，要求当前宿主事实探测、AXI/DMA/bitstream 前提核对、repo 执行路径边界审计、task-scoped helper、focused tests 和显式 gate verdict

## 为什么现在做这个任务

`T48` 已经把当前机器上的软件侧 `.tflite` runtime 真值收窄到一个明确结论：

1. 当前机器存在一个 isolated `tensorflow==2.21.0` 解释器环境；
2. preserved `static_theta_v2` float / int8 true `.tflite` 都能真实加载、真实执行，并做 source-vs-`.tflite` 一致性校验；
3. 但这个结论仍然不等于 HIL closure、real-board validation 或 deployment closure。

因此，主线下一个最小且更诚实的问题已经不再是“`.tflite` 能不能跑”，而是：

1. 当前主线宿主上是否真的具备进入真板 smoke 的前提；
2. `bitstream / AXI / DMA / device-path / permissions / repo board path` 哪一层仍然缺口最大；
3. 仓库现在是否可以合理地再开一个“真实板级执行任务”，还是应该明确 `NO_GO`。

这个任务的价值不在于“硬上真板”，而在于把当前宿主的真实硬件前提收成一个代码驱动、可复核、不会过度乐观的 gate 包。

## 目标

产出一份真实板级 smoke gate 包，显式收口到以下四种结论之一：

1. `GO_REAL_BOARD_SMOKE_EXECUTION_PRECONDITIONS_READY`
2. `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
3. `NO_GO_REAL_BOARD_BITSTREAM_OR_AXI_DMA_CONTRACT_UNCONFIRMED`
4. `NO_GO_REAL_BOARD_REPO_EXECUTION_PATH_PLACEHOLDER_ONLY`

说明：

- 这里的 `GO / NO_GO` 是任务内部 gate verdict，不是 reviewer verdict。
- 即便最终结论是 `NO_GO`，只要证据真实、边界诚实、缺口定位清楚，本任务仍然可以完成并通过 review。
- `GO` 也只表示“可以考虑再开一个后续真板执行任务”，不表示真板已验证。

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T49_real_board_smoke_execution_gate.md`
- `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
- `docs/review/T49_review.md`
- `docs/for_human/T49_explanation.md`
- `docs/worker_summary/T49_worker_summary.md`
- `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
- `tests/test_t49_real_board_smoke_execution_gate.py`
- `artifacts/t49_real_board_smoke_execution_gate/`

说明：

- `artifacts/t49_real_board_smoke_execution_gate/` 是本任务唯一允许写入的 artifact 输出根目录。
- 本任务不新增 `runs/` run root。

## Docs To Update

Worker 必须更新：

- `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
- `docs/review/T49_review.md`
- `docs/for_human/T49_explanation.md`
- `docs/worker_summary/T49_worker_summary.md`

Worker 不得更新治理文档；Captain 会在 review 后统一更新。

## Forbidden Scope

Worker 不得：

- 修改 `docs/02_experiment_plan.md`
- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `runs/` 下文件
- 修改 `cnn_fpga/hwio/board_backend.py`
- 修改 `cnn_fpga/hwio/axi_map.py`
- 修改 `cnn_fpga/hwio/dma_client.py`
- 修改任何 benchmark、HIL、runtime、decoder、训练或 `.tflite` 主线语义文件
- 运行 benchmark、P3/P4 HIL、real-board benchmark 或 sidecar 实验
- 发起任何 MMIO 写、DMA 写、寄存器写、commit/ack 写入型探测
- 把只读 host probe、device-path probe 或 plan-level 代码审计写成“真板 smoke 已执行成功”
- 混入 theory-only branch 或 sidecar branch 材料

允许的硬件探测只限：

1. 当前宿主只读事实收集；
2. 设备路径是否存在 / 可枚举 / 可读打开的无副作用检查；
3. 代码侧 `AXI_REGISTER_MAP`、DMA config 约定和 placeholder 边界的只读审计。

## 必须复用的输入

Worker 必须复用以下既有输入，而不是重写历史事实：

- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`
- `docs/review/T20_review.md`
- `docs/review/T22_review.md`
- `docs/evidence_packs/deployment_boundary/t48_true_tflite_runtime_gate.md`
- `docs/review/T48_review.md`
- `cnn_fpga/hwio/board_backend.py`
- `cnn_fpga/hwio/axi_map.py`
- `cnn_fpga/hwio/dma_client.py`

## 固定边界

- 主线分支：当前 `main` experiment branch only
- 宿主边界：当前机器 only
- 证据边界：real-board host / device / bitstream / AXI / DMA / repo-path truth only
- 输出边界：只允许写 `artifacts/t49_real_board_smoke_execution_gate/`
- 非目标边界：不是 benchmark 任务，不是真板性能任务，不是 HIL promotion 任务，不是论文 prose 任务，不是 sidecar 任务

## 任务要求

### A. 当前宿主事实探测

Worker 必须明确记录当前宿主事实，至少包括：

1. OS / kernel / build / shell 环境；
2. 当前用于只读探测的解释器路径与 Python 版本；
3. 是否存在可识别的 board model / board host / driver / device path 线索；
4. 是否存在可识别的 bitstream 文件、bitstream 版本或其来源记录；
5. 若仓库或本机没有足够事实，必须明确写 `missing`，不得猜测。

需要产出：

- `artifacts/t49_real_board_smoke_execution_gate/host_fact_manifest.json`

### B. device-path / permission 只读 probe

Worker 必须对当前宿主做一次有界只读 probe：

1. 枚举并记录与真板路径最相关的候选 device path / driver clue；
2. 对明确存在的候选路径，可做：
   - `exists`
   - `type`
   - `readable/openable_in_read_only_mode`
3. 若当前宿主根本没有这些路径，也必须显式记录 `not_found`，不得把“未探测到”模糊写成“待后续可能存在”。

需要产出：

- `artifacts/t49_real_board_smoke_execution_gate/device_path_probe.json`

### C. AXI / DMA / placeholder 边界审计包

Worker 必须从代码侧提炼一个结构化审计包，至少覆盖：

1. `AXI_REGISTER_MAP` 中的关键寄存器名与地址偏移；
2. `dma_client.py` 中 DMA path、buffer、payload shape / byte-count 相关约定；
3. `board_backend.py` 当前哪些路径仍是 placeholder / future real-board integration；
4. 哪些是“代码已知事实”，哪些仍依赖“当前宿主 + bitstream + RTL/DMA contract”的外部事实。

需要产出：

- `artifacts/t49_real_board_smoke_execution_gate/code_side_audit.json`

### D. task-scoped helper

新增一个 task-scoped helper：

- `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`

它至少要完成以下工作：

1. 读取 host fact manifest
2. 读取 device-path probe
3. 读取 code-side audit JSON
4. 生成 layer-by-layer gate 结果：
   - `host_environment`
   - `device_path_truth`
   - `bitstream_and_contract_truth`
   - `repo_execution_path_truth`
5. 生成最终 gate verdict
6. 明确列出：
   - 已满足前提
   - 缺失前提
   - 当前支持的最强表述
   - 当前不支持的 real-board claims

### E. focused tests

新增：

- `tests/test_t49_real_board_smoke_execution_gate.py`

测试至少覆盖：

1. host/device 缺失时给出 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
2. host/device 存在但 bitstream / AXI / DMA contract 缺失时给出 `NO_GO_REAL_BOARD_BITSTREAM_OR_AXI_DMA_CONTRACT_UNCONFIRMED`
3. host/device/contract 都齐但 repo execution path 仍是 placeholder-only 时给出 `NO_GO_REAL_BOARD_REPO_EXECUTION_PATH_PLACEHOLDER_ONLY`
4. 所有前提满足时给出 `GO_REAL_BOARD_SMOKE_EXECUTION_PRECONDITIONS_READY`

### F. 最终文档必须回答的问题

`docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md` 至少要回答：

1. 当前宿主是否真的具备进入真板 smoke 的基本前提
2. 当前宿主上到底缺的是 host/device、bitstream/contract，还是 repo execution path
3. `AXI_REGISTER_MAP` 与 DMA 约定有哪些代码侧已知事实
4. 当前最强可支持的 real-board 表述是什么
5. 当前仍然不能支持哪些 real-board / HIL / deployment claims
6. 最终 gate verdict 是什么

文档中必须包含一个紧凑表格，至少区分：

- `host_environment`
- `device_path_truth`
- `bitstream_and_contract_truth`
- `repo_execution_path_truth`
- `supported_claims`
- `unsupported_claims`

## 预期输出

Worker 必须产出：

- `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
- `docs/review/T49_review.md`
- `docs/for_human/T49_explanation.md`
- `docs/worker_summary/T49_worker_summary.md`
- `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
- `tests/test_t49_real_board_smoke_execution_gate.py`
- `artifacts/t49_real_board_smoke_execution_gate/`

## 验证

Worker 必须实际执行并报告：

1. `python -m py_compile cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
2. `python -m unittest tests.test_t49_real_board_smoke_execution_gate`
3. 一次真实 host fact probe
4. 一次真实 device-path read-only probe
5. helper 的一次真实执行
6. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

Worker 还必须显式报告：

1. host probe 使用的解释器路径
2. 当前宿主最关键的 OS / driver / device-path 事实
3. 是否存在可引用的 bitstream / contract 证据
4. `board_backend.py` 当前是否仍属于 placeholder-only execution path
5. 最终 gate verdict
6. 当前支持的最强 real-board 表述
7. 当前不支持的 real-board / HIL / deployment claims

## Review No-Go Triggers

Reviewer 在以下任一情况应返回 `BLOCK`：

1. worker 把只读 host/device/code 审计写成“真板 smoke 已执行成功”
2. worker 实际执行了 MMIO 写、DMA 写、寄存器写、commit/ack 写入型探测
3. worker 越界修改 `board_backend.py`、`axi_map.py`、`dma_client.py` 或任何治理文档
4. worker 未给出显式 gate verdict
5. worker 把本任务写成 HIL closure、real-board validated 或 deployment closure

## Captain 备注

- `T49` 之所以放在 `T48` 之后，是因为 `T48` 已经把当前机器的软件侧 `.tflite` 真值收清楚了；现在主线剩余的 deployment-boundary 大缺口是“当前宿主到底能不能诚实地进入真板 smoke 前提”。
- 这不是一个 docs-only 任务。它要求当前宿主事实探测、只读 device-path probe、代码侧 AXI/DMA/placeholder 审计、task-scoped helper、focused tests 和显式 gate verdict 一起收口。
- 若最终结论是 `NO_GO`，只要缺口定位清楚、边界真实、没有过度乐观，这仍然是有效完成，而不是失败。

## Worker Output

### 本轮产物

- helper：`cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
- focused tests：`tests/test_t49_real_board_smoke_execution_gate.py`
- artifacts：
  - `artifacts/t49_real_board_smoke_execution_gate/host_fact_manifest.json`
  - `artifacts/t49_real_board_smoke_execution_gate/device_path_probe.json`
  - `artifacts/t49_real_board_smoke_execution_gate/code_side_audit.json`
  - `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`
- docs：
  - `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
  - `docs/review/T49_review.md`
  - `docs/for_human/T49_explanation.md`
  - `docs/worker_summary/T49_worker_summary.md`

### 本轮关键结论

- host probe 解释器：`C:\ProgramData\anaconda3\python.exe`
- 当前宿主：Windows 11，`cmd /c ver = Microsoft Windows [Version 10.0.26200.8457]`
- repo config 仍记录：`board=ZCU111`、`bitstream_version=fpga_linear_v1`
- 当前机器上未找到可读打开的真板候选设备路径：
  - `/dev/uio0`
  - `/dev/uio1`
  - `\\\\.\\XilinxDMA`
  - `\\\\.\\XDMA`
  - `\\\\.\\uio0`
  - `\\\\.\\uio1`
- `pnputil /enum-devices /connected` 没有匹配到 `Xilinx|AMD|FPGA|XDMA|UIO` 已连接设备
- `board_backend.py` 仍属于 placeholder-only execution path
- 最终 gate verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`

### 已执行验证

1. `python -m py_compile cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
2. `python -m unittest tests.test_t49_real_board_smoke_execution_gate`
3. 一次真实 host fact probe
4. 一次真实 device-path read-only probe
5. helper 的一次真实执行
6. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

### 剩余风险

- 当前 `NO_GO` 首先由 host/device 层触发，但 bitstream / RTL / DMA contract 也仍未确认。
- 当前宿主上的 `board` 路径依然只能写成 placeholder scaffolding，不能升级成真板 smoke 已可执行。
