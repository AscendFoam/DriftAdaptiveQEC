# T71：真板 gate 再生成加固与宿主迁移包

## 状态

- 由 Captain 于 `2026-06-10` 在 `T49` closeout 后提出
- 当前阶段：`Phase 2: Controlled Development`
- 决策状态：`Go`
- 任务类型：有界 mainline deployment-boundary hardening 任务

## 为什么现在做这个任务

`T49` 已经把当前机器上的真板入口问题收成了一个诚实结论：

- 当前宿主 gate verdict = `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- 没有执行任何真板 smoke
- 真实缺口已经不再是“当前宿主到底有没有被检查过”，而是：
  1. 未来如果换到候选真板宿主，仓库里是否已经有一个单一、checked-in、只读、可再生成的 gate 入口
  2. `device_path_truth` 的 ready 判定是否足够严格，能避免把“两条同角色路径”误判成 `mmio + dma` 双前提都已满足

因此，`T71` 的目标不是继续伪装推进真板执行，而是把 `T49` 的 current-host honest `NO_GO` 包，加固成一个 future-host 也可复用的、边界清楚的 read-only gate 再生成入口。

## 目标

在不触碰真板执行的前提下，补齐以下四件事：

1. 把 `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py` 的 `device_path_truth` 判定改成 role-aware：
   - 至少需要 `1` 条可只读打开的 `mmio`
   - 且至少需要 `1` 条可只读打开的 `dma`
   - 不能只按 openable path 总数计数
2. 增加一个 checked-in 的只读 artifact 收集入口，用于在任意候选宿主再生成：
   - `host_fact_manifest.json`
   - `device_path_probe.json`
   - `code_side_audit.json`
3. 增加 focused regression：
   - “两条同角色 openable path 不得误判 ready”
   - “`T49` checked-in artifacts 回放后 verdict 仍为 current-host `NO_GO`”
4. 形成一个可直接交给未来候选真板宿主使用的 host-transfer pack：
   - 当前宿主重放结果
   - future-host 执行命令
   - 仍未闭环的 `bitstream / RTL / DMA / placeholder` 边界

## Allowed Files

Worker 只可修改或新增以下路径：

- `docs/tasks/Phase2/T71_real_board_gate_regeneration_and_host_transfer_pack.md`
- `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`
- `docs/review/T71_review.md`
- `docs/for_human/T71_explanation.md`
- `docs/worker_summary/T71_worker_summary.md`
- `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
- `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
- `tests/test_t49_real_board_smoke_execution_gate.py`
- `tests/test_t71_real_board_gate_regeneration_pack.py`
- `artifacts/t71_real_board_gate_regeneration_pack/`

说明：

- `T71` 允许复用和读取 `T49` 的 checked-in artifacts，但不得改写它们
- `T71` 只允许把新输出写到 `artifacts/t71_real_board_gate_regeneration_pack/`
- 本任务不创建 `runs/` run root

## Docs To Update

Worker 必须更新：

- `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`
- `docs/review/T71_review.md`
- `docs/for_human/T71_explanation.md`
- `docs/worker_summary/T71_worker_summary.md`

Worker 不得更新治理文档；Captain 会在 review 后统一更新。

## Forbidden Scope

Worker 不得：

- 修改 `docs/02_experiment_plan.md`
- 修改任何治理文档 `docs/00_*` 到 `docs/08_*`
- 修改任何 `runs/` 下文件
- 修改 `cnn_fpga/hwio/board_backend.py`
- 修改 `cnn_fpga/hwio/axi_map.py`
- 修改 `cnn_fpga/hwio/dma_client.py`
- 修改任何 benchmark / HIL / training / `.tflite` / paper reopen / theory branch 相关主线语义
- 发起任何 MMIO 写、DMA 写、寄存器写、commit/ack 写入型探测
- 运行 real-board smoke、real-board benchmark、P3/P4 benchmark 或 sidecar 实验
- 改写 `T49` 的历史结论、历史 artifact 或历史 review verdict
- 把 current-host 的 read-only gate 再生成写成“真板已经 ready / 已验证 / 已执行成功”

## 必须复用的输入

Worker 必须复用以下既有输入，而不是重写历史事实：

- `docs/evidence_packs/deployment_boundary/t49_real_board_smoke_execution_gate.md`
- `docs/review/T49_review.md`
- `docs/worker_summary/T49_worker_summary.md`
- `artifacts/t49_real_board_smoke_execution_gate/host_fact_manifest.json`
- `artifacts/t49_real_board_smoke_execution_gate/device_path_probe.json`
- `artifacts/t49_real_board_smoke_execution_gate/code_side_audit.json`
- `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`
- `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`
- `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`

## 固定边界

- 主线分支：当前 `main` experiment branch only
- 宿主边界：当前机器 only，用于做 read-only regeneration replay
- 证据边界：real-board host / device / bitstream / AXI / DMA / repo-path truth only
- 输出边界：只允许写入 `artifacts/t71_real_board_gate_regeneration_pack/`
- 非目标边界：不是 board execution 任务，不是 benchmark 任务，不是 paper 任务，不是 sidecar 任务

## 任务要求

### A. role-aware gate 逻辑加固

Worker 必须把 `device_path_truth` 的 ready 条件改成 role-aware：

1. 至少 `1` 条 `role = mmio` 且 `read_only_openable = true`
2. 至少 `1` 条 `role = dma` 且 `read_only_openable = true`
3. 不允许仅以 `openable_count >= 2` 作为 ready 判定

同时必须保证：

- 使用 `T49` checked-in artifacts 回放时，verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- 不改变 `T49` 当前宿主 strongest supported claim

### B. checked-in read-only artifact 收集入口

新增：

- `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`

它必须能在当前宿主只读生成：

1. `host_fact_manifest.json`
2. `device_path_probe.json`
3. `code_side_audit.json`

要求：

- 不调用任何 write-side MMIO/DMA/register 操作
- 不依赖 `runs/`
- 不要求 benchmark、训练、`.tflite`、真板执行环境
- 必须把输出收敛到 `artifacts/t71_real_board_gate_regeneration_pack/`

### C. focused regression

至少补以下回归：

1. 两条 openable path 若同属 `dma` 或同属 `mmio`，不得判定 `device_path_truth.ready`
2. 一条 `mmio` + 一条 `dma` 的 openable path 组合，才允许判定 ready
3. 直接回放 `artifacts/t49_real_board_smoke_execution_gate/*.json` 时，最终 verdict 仍为 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
4. `T71` 当前宿主 read-only 再生成 artifact 通过 gate helper 后，最终 verdict 必须与 `T49` 当前宿主结论一致

### D. 最终文档必须回答的问题

`docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md` 至少要回答：

1. `T71` 是否改变了 `T49` 当前宿主的最终 gate verdict
2. `T49` checked-in artifact 回放结果与 `T71` 当前宿主再生成结果是否一致；若不一致，只允许出现时间戳/环境噪声级差异，不允许出现 verdict 漂移
3. future-host 若要继续 real-board lane，现在应使用哪一个 checked-in 入口和哪一组命令
4. 当前仍未闭环的是哪几层：
   - `device_path_truth`
   - `bitstream_and_contract_truth`
   - `repo_execution_path_truth`
5. 为什么 `T37` 现在仍然不能开工

### E. 文档中的 strongest supported claim

`T71` 文档最终只能支持类似以下口径：

“仓库现在有一个 checked-in、只读、可在候选宿主再生成的 real-board gate 入口；当前这台 Windows 宿主重放后仍然是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`。这说明 mainline 已经把 current-host 真板前提检查从一次性 task 结果，提升成了可再生成的 gate 包，但这仍不等于真板执行成功、real-board validation、P3 real-board HIL complete 或 deployment closure。”

### F. 不允许写出的结论

`T71` 不得写出：

- `real-board smoke executed successfully`
- `T37 ready to execute on current host`
- `P3 real-board HIL complete`
- `board backend validated`
- `hardware_validated`
- `deployment closure`

## 预期输出

Worker 必须产出：

- `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`
- `docs/review/T71_review.md`
- `docs/for_human/T71_explanation.md`
- `docs/worker_summary/T71_worker_summary.md`
- `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
- `tests/test_t71_real_board_gate_regeneration_pack.py`
- 必要时更新：
  - `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
  - `tests/test_t49_real_board_smoke_execution_gate.py`
- `artifacts/t71_real_board_gate_regeneration_pack/`

## 验证

Worker 必须实际执行并报告：

1. `python -m py_compile cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
2. `python -m py_compile cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
3. `python -m unittest tests.test_t49_real_board_smoke_execution_gate`
4. `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
5. 一次当前宿主只读 artifact 再生成
6. 一次用 `T71` 再生成 artifact 驱动 gate helper 的真实执行
7. 一次用 `T49` checked-in artifact 驱动 gate helper 的 replay 执行
8. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

Worker 还必须显式报告：

1. role-aware 逻辑加固后，当前宿主 verdict 是否变化
2. `T49` 回放 verdict 与 `T71` 当前宿主再生成 verdict 是否一致
3. future-host 推荐入口命令
4. `T37` 当前仍被哪些层阻塞

## Review No-Go Triggers

Reviewer 在以下任一情况应返回 `BLOCK`：

1. Worker 把 `T71` 写成真板执行、真板 ready、或真实 HIL 完成
2. Worker 发起任何 write-side MMIO/DMA/register 操作
3. Worker 越界修改 `board_backend.py` / `axi_map.py` / `dma_client.py` / 治理文档
4. role-aware 逻辑没有真正防住“两条同角色路径误判 ready”
5. `T49` checked-in artifact replay verdict 被无解释地改坏
6. Worker 没有给出 future-host 可执行的 checked-in read-only regeneration 入口

## Captain 备注

- `T49` 已经给出一个诚实 current-host `NO_GO`，所以主线现在不该假装进入 `T37`
- `T71` 的价值不在于“更接近跑真板”，而在于把未来任何候选宿主上的真板前提核查，变成一个代码化、可复核、可迁移、不会轻易过度乐观的 gate 包
- 若 `T71` 完成后仍没有目标宿主暴露真实设备节点、bitstream 绑定与非-placeholder repo path 证据，则 `T37` 继续保持 blocked 是正确结果，而不是失败

## Worker Output

### 本轮产物

- 更新 helper：
  - `cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
- 新增 collector：
  - `cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
- 更新/新增 tests：
  - `tests/test_t49_real_board_smoke_execution_gate.py`
  - `tests/test_t71_real_board_gate_regeneration_pack.py`
- artifacts：
  - `artifacts/t71_real_board_gate_regeneration_pack/host_fact_manifest.json`
  - `artifacts/t71_real_board_gate_regeneration_pack/device_path_probe.json`
  - `artifacts/t71_real_board_gate_regeneration_pack/code_side_audit.json`
  - `artifacts/t71_real_board_gate_regeneration_pack/current_host_regenerated_gate.json`
  - `artifacts/t71_real_board_gate_regeneration_pack/t49_checked_in_replay_gate.json`
  - `artifacts/t71_real_board_gate_regeneration_pack/replay_vs_regeneration_comparison.json`
- docs：
  - `docs/evidence_packs/deployment_boundary/t71_real_board_gate_regeneration_pack.md`
  - `docs/review/T71_review.md`
  - `docs/for_human/T71_explanation.md`
  - `docs/worker_summary/T71_worker_summary.md`

### 本轮关键结论

- role-aware 加固后，当前宿主 verdict 没有变化
- `T49` checked-in artifact replay verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- `T71` current-host regeneration verdict：
  - `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`
- replay 与 regeneration 一致：
  - `verdict_match = true`
  - `strongest_statement_match = true`
  - `device_path_truth_status_match = true`
- future-host 推荐入口命令：
  1. `python -m cnn_fpga.hwio.collect_t71_real_board_gate_artifacts --output-dir <future_host_pack_dir> --mmio-path <MMIO_PATH> --dma-path <DMA_PATH>`
  2. `python -m cnn_fpga.hwio.build_t49_real_board_smoke_gate --host-fact-manifest-json <future_host_pack_dir>/host_fact_manifest.json --device-path-probe-json <future_host_pack_dir>/device_path_probe.json --code-side-audit-json <future_host_pack_dir>/code_side_audit.json --output-json <future_host_pack_dir>/real_board_gate.json`
- `T37` 当前仍被以下层阻塞：
  - `device_path_truth`
  - `bitstream_and_contract_truth`
  - `repo_execution_path_truth`

### 已执行验证

1. `python -m py_compile cnn_fpga/hwio/build_t49_real_board_smoke_gate.py`
2. `python -m py_compile cnn_fpga/hwio/collect_t71_real_board_gate_artifacts.py`
3. `python -m unittest tests.test_t49_real_board_smoke_execution_gate`
4. `python -m unittest tests.test_t71_real_board_gate_regeneration_pack`
5. 一次当前宿主只读 artifact 再生成
6. 一次用 `T71` 再生成 artifact 驱动 gate helper 的真实执行
7. 一次用 `T49` checked-in artifact 驱动 gate helper 的 replay 执行
8. 边界检查：
   - `git diff --name-only -- runs`
   - `git diff --name-only -- cnn_fpga/hwio/board_backend.py cnn_fpga/hwio/axi_map.py cnn_fpga/hwio/dma_client.py`
   - `git diff --name-only -- docs/00_project_snapshot.md docs/01_legacy_audit.md docs/03_hil_p4_boundary_audit.md docs/04_task_board.md docs/05_decision_log.md docs/06_repo_noise_governance.md docs/07_handoff.md docs/08_risks_and_open_questions.md`

### 剩余风险

- `T71` 只加固了 gate 的再生成性，不等于当前或 future-host 已经具备真实板级执行前提。
- 如果 future-host 没有真实暴露 `mmio + dma` 路径、bitstream 绑定和非-placeholder repo path，`T37` 仍应继续 blocked。
