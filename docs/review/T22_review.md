# T22 Adversarial Review: Real-Board Smoke Execution Plan

**Task ID**: `T22`
**Reviewer**: Claude Code Reviewer (adversarial)
**Date**: 2026-05-10

---

## Verdict: PASS_WITH_WARNINGS

---

## 1. Task Completion Check (Adversarial Mode)

T22 目标：为后续真板 smoke 制定可执行计划，补齐平台确认、AXI/register map 审计路径与量化验收阈值；不调用硬件、不实现真板 backend。

逐条验收：

| Done Criteria | 状态 | 说明 |
|---------------|------|------|
| 产出 real-board smoke execution plan | 完成 | `docs/real_board_smoke_execution_plan.md` 新增，10 节，323 行 |
| 明确目标平台待决项：Linux / Windows / WSL / remote board host | 完成 | 第 3 节列出 4 种宿主模型，每种都有进入前必须确认的检查项和排除条件 |
| 明确 AXI/register map 审计路径 | 完成 | 第 4 节列出全部 14 个寄存器地址 + 7 个 bit mask + fixed-point contract + 4 条退出条件 |
| 明确 DMA buffer 审计路径 | 完成 | 第 5 节列出 config 冻结项 + runtime 语义冻结项 + byte/shape 审计 + 4 条退出条件 |
| 给出量化验收阈值草案 | 完成 | 第 6 节 Layer A-D 各有具体数值，第 6.5 节有 fail-fast budget |
| 给出 future hardware run evidence pack | 完成 | 第 8 节列出 7 类必须证据 |
| 给出禁止表述 | 完成 | 第 9 节列出 5 条禁止 + 3 条允许 |
| 不调用硬件、不改源码、不运行 benchmark | 确认无违规 | 所有命令标注 `NOT EXECUTED IN T22` |

## 2. Scope Compliance

### Allowed files — 全部合规

T22 允许的核心 diff 文件：

- `docs/real_board_smoke_execution_plan.md`（新增） — allowed
- `docs/tasks/Phase2/T22_real_board_smoke_execution_plan.md` — allowed（追加 Worker Output Summary）
- `docs/real_board_hil_readiness.md` — allowed
- `docs/03_hil_p4_boundary_audit.md` — allowed
- `docs/04_task_board.md` — allowed
- `docs/07_handoff.md` — allowed
- `docs/08_risks_and_open_questions.md` — allowed

### Out-of-scope files modified — WARNING

以下 5 个文件在 diff 中被修改，但**不在** T22 的 Allowed files 列表中：

1. `docs/00_project_snapshot.md`
2. `docs/01_legacy_audit.md`
3. `docs/05_decision_log.md`
4. `docs/06_repo_noise_governance.md`
5. `docs/tasks/Phase2/T21_phase2_milestone_review.md`

这些文件的修改内容是同步更新任务引用（T21→T22）和补充 T21 milestone gate 的 decision log。从内容性质看，这些修改属于 Captain 整合阶段的治理文件同步（T21 gate + T22 初始化），不是 Worker 在执行任务时擅自越界。它们不涉及代码变更、benchmark 口径或硬件调用。

**但严格按 Allowed files 规则，这些文件确实不在 T22 的允许列表中。** 这可能是因为 Captain 在 T22 Worker 执行前/后做了额外同步，而 Worker 报告中没有提及这些文件。

### Forbidden scope — 无违规

- `cnn_fpga/hwio/board_backend.py` — 未改动
- `cnn_fpga/hwio/fpga_driver.py` — 未改动
- `cnn_fpga/benchmark/run_hil_suite.py` — 未改动
- 无硬件命令调用
- 无 `backend=board` HIL 运行
- 无 real-board HIL 已完成表述
- 无 benchmark 口径变更

## 3. AXI/Register Map Accuracy (Independent Verification)

我用 `Read` 直接读取了 `cnn_fpga/hwio/axi_map.py`（215 行），逐项交叉验证执行计划第 4 节的每个地址和 mask：

| Plan Claim | Source Code (`axi_map.py`) | Match |
|------------|--------------------------|-------|
| `ctrl_addr = 0x00` | Line 64: `ctrl_addr: int = 0x00` | Yes |
| `status_addr = 0x04` | Line 65: `status_addr: int = 0x04` | Yes |
| `hist_meta_addr = 0x08` | Line 66: `hist_meta_addr: int = 0x08` | Yes |
| `overflow_count_addr = 0x0C` | Line 67: `overflow_count_addr: int = 0x0C` | Yes |
| `k11_addr = 0x10` | Line 68: `k11_addr: int = 0x10` | Yes |
| `k12_addr = 0x14` | Line 69: `k12_addr: int = 0x14` | Yes |
| `k21_addr = 0x18` | Line 70: `k21_addr: int = 0x18` | Yes |
| `k22_addr = 0x1C` | Line 71: `k22_addr: int = 0x1C` | Yes |
| `b1_addr = 0x20` | Line 72: `b1_addr: int = 0x20` | Yes |
| `b2_addr = 0x24` | Line 73: `b2_addr: int = 0x24` | Yes |
| `active_bank_addr = 0x30` | Line 74: `active_bank_addr: int = 0x30` | Yes |
| `epoch_id_addr = 0x34` | Line 75: `epoch_id_addr: int = 0x34` | Yes |
| `commit_epoch_addr = 0x38` | Line 76: `commit_epoch_addr: int = 0x38` | Yes |
| `hist_seq_addr = 0x3C` | Line 77: `hist_seq_addr: int = 0x3C` | Yes |
| `ctrl_start_mask` | Line 79: `ctrl_start_mask: int = 1 << 0` | Yes |
| `ctrl_reset_hist_mask` | Line 80: `ctrl_reset_hist_mask: int = 1 << 1` | Yes |
| `ctrl_commit_bank_mask` | Line 81: `ctrl_commit_bank_mask: int = 1 << 2` | Yes |
| `status_ready_mask` | Line 83: `status_ready_mask: int = 1 << 0` | Yes |
| `status_hist_ready_mask` | Line 84: `status_hist_ready_mask: int = 1 << 1` | Yes |
| `status_commit_ack_mask` | Line 85: `status_commit_ack_mask: int = 1 << 2` | Yes |
| `status_overflow_alert_mask` | Line 86: `status_overflow_alert_mask: int = 1 << 3` | Yes |
| `Q4.20` | Line 63: `fixed_point_spec: str = "Q4.20"` | Yes |
| `A -> 0, B -> 1` bank encoding | Lines 203-211: `encode_bank`/`decode_bank` | Yes |

**全部 23 项寄存器/mask/编码审计对象与源码完全吻合。** 这解决了 T20 review 中 N1（寄存器名来源不透明）的问题。

## 4. DMA Audit Accuracy (Independent Verification)

执行计划第 5 节引用了 `dma_client.py` 的结构：

| Plan Claim | Source Code (`dma_client.py`) | Match |
|------------|-----------------------------|-------|
| `MemoryMappedDMAConfig` class | Line 86: `class MemoryMappedDMAConfig` | Yes |
| `buffer_bytes` field | Line 90: `buffer_bytes: int` | Yes |
| `buffer_count` field | Line 91: `buffer_count: int = 2` | Yes |
| `DMAReadout` class | Line 17: `class DMAReadout` | Yes |
| `path` config item | Implicit from constructor pattern | Yes |
| `read_histogram()` method | Lines 58, 75 | Yes |
| `buffer_id` / `byte_count` / `metadata` in DMAReadout | From dataclass fields | Yes |

**DMA 审计清单与源码吻合。**

## 5. Adversarial Probes

### 5.1 是否有把量化阈值伪装成已验证事实？

第 6 节标题明确写"Quantitative Acceptance **Threshold Draft**"。第 1 节写"它不是硬件执行记录"。第 7 节每个命令块都标注 `# NOT EXECUTED IN T22`。

**未发现伪装。**

### 5.2 量化阈值是否合理（不过度也不遗漏）？

- Layer A：`1/1` 设备存在率 + `5s` 超时 — 最小必要条件
- Layer B：`3` 次状态读回 + `1s` start-to-ready + `5s` epoch 变化窗口 — 合理
- Layer C：`3` 个连续 readout + `100%` byte_count + `buffer_id` 范围检查 — 合理
- Layer D：`3` 轮 stage+commit + `1s` ack 超时 + `2` 次切换 — 合理
- Fail-fast budget：`15 min` 总预算 + `0` 容忍 DMA/commit 失败 — 严格但不过度

**阈值草案合理，且明确标注为"后续硬件任务可以细化，但不得弱化为纯定性描述"。**

### 5.3 平台决策点是否完整覆盖？

四种宿主模型（Linux local / Windows local / WSL / remote）覆盖了常见的真板部署拓扑。每种都有"进入前必须确认"和"如果不确定则不得进入"的排除条件。

**覆盖合理。**

### 5.4 是否有未覆盖的审计对象？

Worker 审计了 `axi_map.py`（完整）和 `dma_client.py`（结构级）。遗漏检查：

- `board_backend.py` 中 `hist_meta` 解码（`build_hist_meta_word` / `decode_hist_meta_word`）— 已在 `axi_map.py` 第 188-200 行中定义，plan 第 5.2 节引用了 `DMAReadout.metadata`。未遗漏。
- `param_bank.py` 的 `DecoderRuntimeParams` — 被 `axi_map.py` 的 `pack_params` / `unpack_params` 引用。Plan 第 4.2 节列出了 `K` 和 `b` 参数寄存器。未遗漏。

### 5.5 是否有把计划写成事实的隐蔽表述？

逐段扫描 execution plan：

- 第 1 节："不是硬件执行记录" — 明确
- 第 2 节："本计划不覆盖...board_backend.py 的实现改写" — 明确
- 第 7 节所有命令标注 `NOT EXECUTED IN T22` — 明确
- 第 9 节 5 条禁止表述 — 严格

治理文件同步：
- `03_hil_p4_boundary_audit.md`：新增"即使 execution plan 已存在，也只能写成 execution plan exists, but it has not been executed" — 诚实
- `real_board_hil_readiness.md`：新增"不得默认 `/dev/uio*` 权限模型就是目标执行模型" — 解决了 T20 review N3

**未发现计划写成事实。**

## 6. Blocking Issues

**无。**

## 7. Non-blocking Issues

### N1: 5 个 out-of-scope 文件被修改（WARNING）

以下文件被修改但不在 T22 Allowed files 中：`00_project_snapshot.md`、`01_legacy_audit.md`、`05_decision_log.md`、`06_repo_noise_governance.md`、`T21_phase2_milestone_review.md`。

内容属于 Captain 整合阶段的治理文件同步（T21 gate decision log + T22 任务引用更新），不是 Worker 越界。但这严格违反了"Worker 只改 Allowed files"的规则。

建议：Captain 在整合阶段应明确声明这些文件是自己修改的，而非 Worker 输出。或者在后续任务包中把治理文件的引用同步也加入 Allowed files。

### N2: 第 7.1 节 preflight 命令中 `AXI_REGISTER_MAP` 引用

第 7.1 节写 `from cnn_fpga.hwio.axi_map import AXI_REGISTER_MAP as m; print(m)`。但 `AXI_REGISTER_MAP` 是 `AxiRegisterMap()` 的实例（`axi_map.py:214`），直接 `print` 一个 `dataclass` 实例会给出 `AxiRegisterMap(...)` 的 repr，不是格式化的地址表。如果后续执行任务需要更可读的输出，可能需要额外格式化。

不阻塞，但后续执行任务应注意。

### N3: 第 5.3 节 `byte_count = 4096` 的假设

Plan 写"如果当前设计沿用 `32 x 32 float32` histogram，则期望 `byte_count = 4096`"。计算验证：`32 * 32 * 4 bytes = 4096`。正确。但这个假设需要后续执行任务确认实际 bitstream 是否真的使用 `32x32 float32`。

## 8. Missing Tests / Validation

验证方式为只读代码审计，这是任务包 `Verification` 中明确要求的。所有 23 项 AXI 寄存器/mask 和 DMA 结构经独立交叉验证全部准确。

**无缺失验证。**

## 9. Suspicious Implementation Details

**无。** 本任务是只读文档任务，没有实现细节。

## 10. Recommended Next Action

1. T22 adversarial review 通过（PASS_WITH_WARNINGS），可由 Captain 标记完成并提交 git。
2. N1（out-of-scope 文件）需要 Captain 确认这些修改是否属于 Captain 整合阶段而非 Worker 越界。
3. T22 是 Phase 2 Milestone 2F 的唯一任务。完成后应考虑是否做一次 Phase 2 extended milestone review，或者直接进入 Phase 3。
4. 真板 smoke 的下一步应是实际执行任务（需要有硬件环境），不是继续写计划文档。
5. 物理 cache cleanup（T19 manifest 的执行）仍待 Captain 批准。

---

**Verdict: PASS_WITH_WARNINGS**
**Blocking issues: 无**
**Non-blocking issues: N1（5 个 out-of-scope 文件被修改，需 Captain 确认归属），N2（preflight 命令输出格式），N3（`byte_count=4096` 假设需后续确认）**
**Suspicious implementation details: 无**
