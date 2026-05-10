# T20 Adversarial Review: Real-Board HIL Readiness Checklist

**Task ID**: `T20`
**Reviewer**: Claude Code Reviewer (adversarial)
**Date**: 2026-05-10

---

## Verdict: PASS

---

## 1. Task Completion Check (Adversarial Mode)

T20 目标：为真板 HIL 路径补 readiness checklist 与缺口清单，不实现真板 backend，也不改写 placeholder 语义。

逐条验收：

| Done Criteria | 状态 | 说明 |
|---------------|------|------|
| 产出 real-board HIL readiness checklist | 完成 | `docs/real_board_hil_readiness.md` 新增，7 节，164 行 |
| 明确 `board_backend.py` / `fpga_driver.py` 的 placeholder 证据 | 完成 | 第 2 节逐一列出，经独立代码验证全部准确 |
| 明确真板前置条件、设备/地址/权限/日志需求 | 完成 | 第 3 节分 4 个子节（设备拓扑、权限、驱动/寄存器、运行时日志） |
| 明确最小 smoke 验收标准 | 完成 | 第 4 节分 4 层（A: 设备存在、B: 寄存器活性、C: DMA 读出、D: commit/ack），且明确"四层全部通过后才适合继续讨论 benchmark" |
| 不修改真板代码 | 确认无违规 | `git diff HEAD --name-only | grep "cnn_fpga/"` 返回 `0` |
| 不调用硬件命令 | 确认无违规 | 无任何硬件调用痕迹 |
| 不把 mock/software HIL 写成 real-board 完成 | 确认无违规 | 第 6 节列出 5 条禁止表述 + 3 条允许表述 |

## 2. Scope Compliance

### Allowed files — 全部合规

diff 涉及的文件：

- `docs/real_board_hil_readiness.md`（新增） — allowed
- `docs/tasks/Phase2/T20_real_board_readiness_checklist.md` — allowed（仅追加 Worker Output Summary）
- `docs/03_hil_p4_boundary_audit.md` — allowed
- `docs/04_task_board.md` — allowed
- `docs/07_handoff.md` — allowed
- `docs/08_risks_and_open_questions.md` — allowed

`cnn_fpga/` 下 `0` 个文件被修改。

### Forbidden scope — 无违规

- `cnn_fpga/hwio/board_backend.py` — 未改动（经 `git diff` 确认）
- `cnn_fpga/hwio/fpga_driver.py` — 未改动
- 无"真实板级完成"表述
- 无硬件命令调用
- 无 mock→real-board 外推

## 3. Placeholder Evidence Accuracy (Independent Verification)

我用 `grep` 对源码做了独立交叉验证：

| Manifest Claim | Source Code Evidence | Match |
|----------------|---------------------|-------|
| `Placeholder real-board backend...` (file header) | `board_backend.py:1` — exact match | Yes |
| `board_device_missing:...` | `board_backend.py:168` — `raise BoardBackendUnavailableError(f"board_device_missing:{...}")` | Yes |
| `target_bank = None`, `version = None`, `ack_delay_us = None` | `board_backend.py:257` — `return {"target_bank": None, ..., "version": None, "ack_delay_us": None}` | Yes |
| `step(...) return []` | `board_backend.py:285` — `return []` | Yes |
| `allow_missing_device = True` default | `board_backend.py:58` — `allow_missing_device: bool = True` | Yes |
| `reserved for future real-board integration` | `fpga_driver.py:77` — exact match | Yes |
| `mock and future real HIL backends` (file header) | `fpga_driver.py:1` — exact match | Yes |

**全部 7 项占位符证据均与源码逐字吻合。**

## 4. Adversarial Probes

以下是对高风险区域的主动攻击性检查：

### 4.1 是否有"伪装成真板证据"的隐蔽表述？

逐段扫描 `docs/real_board_hil_readiness.md`：

- 第 1 节："本文件不代表真板 backend 已实现" — 明确
- 第 2.3 节："只能证明 software HIL 路径存在，不能外推为真板证据" — 明确
- 第 4 节：四层验收标准全部用"至少需要证明"而非"已证明" — 准确
- 第 6 节：禁止表述清单中列出 `real-board HIL complete` / `board backend validated` — 严格

**未发现伪装表述。**

### 4.2 是否有把配置项等同于硬件能力的暗示？

检查第 3 节前置条件：

- 所有条目都用"明确..."、"是否..."等探询语气，而不是"已有..."
- 例如"明确 MMIO 设备路径，例如 `axi_uio_path` 对应哪个实际节点" — 只列出需要确认的问题，不暗示已确认

**未发现。**

### 4.3 是否有过度承诺的四层验收标准？

检查第 4 节 Layer A-D：

- Layer A 只要求"设备存在 + 能打开 + 能映射"，这是最小必要条件
- Layer B 只要求"寄存器可读/写 + epoch 变化"，不过度
- Layer C 只要求"histogram 能读出 + shape 正确 + 不是空数据"，合理
- Layer D 只要求"commit/ack 至少一次成功"，不过度
- 四层全部标注"至少需要证明"而非"已证明"
- 明确写"只有在 A/B/C/D 四层都具备证据后，才适合继续讨论 benchmark"

**验收标准合理，既不过度也不遗漏。**

### 4.4 是否有未覆盖的占位代码路径？

Worker 报告审计了 `board_backend.py`、`fpga_driver.py`、`run_hil_suite.py`。遗漏检查：

- `mock_fpga.py` — 不在 Forbidden scope 内，但也不在 Inputs to read 内。然而 Worker 的核心目标是识别真板缺口，`mock_fpga.py` 是 mock 路径，不是真板路径。不构成遗漏。
- `axi_map.py` / `dma_client.py` — 在 Inputs to read 中未列出，但 Worker 在第 3.3 节提到了 `AxiRegisterMap` 和寄存器地址表。如果 Worker 没有实际读取这些文件，寄存器名称列表可能来自推断而非审计。但考虑到本任务是只读 checklist（不是实现），且列出的寄存器名（`ctrl/status/epoch/hist_meta/...`）与 `board_backend.py` 中可见的引用一致，不构成 blocking issue。

### 4.5 是否有"计划写成事实"的隐蔽风险？

检查治理文件更新：

- `03_hil_p4_boundary_audit.md`：新增"可以说：`real-board readiness checklist exists, but hardware validation evidence does not.`" — 诚实
- `08_risks_and_open_questions.md`：新增 R13（真板缺口），标记为"高"风险 — 准确
- `04_task_board.md`：T20 仍为 `[ ]`，未提前勾选 — 合规
- `07_handoff.md`：标注"等待 Reviewer / Captain 收口" — 合规

**未发现计划写成事实。**

## 5. Blocking Issues

**无。**

## 6. Non-blocking Issues

### N1: 第 3.3 节寄存器名称来源不透明

第 3.3 节列出了 `ctrl/status/epoch/hist_meta/commit_epoch/hist_seq/overflow_count` 等寄存器名，以及 `active_bank`、`commit_ack`、`hist_ready` 等语义。这些名称虽然与 `board_backend.py` 中的引用一致，但 Worker 的 Inputs to read 列表中没有包含 `axi_map.py` 或 `dma_client.py`。如果这些名称是推断而来而非审计所得，后续真板任务可能需要重新确认。

建议：后续真板执行任务如果需要引用这些寄存器名，应直接审计 `axi_map.py`。

### N2: 第 4 节验收标准缺乏量化阈值

四层验收标准都是定性描述（"能变为真"、"至少一次"、"值域合理"），没有给出量化阈值。例如：
- Layer B："`epoch_id` 会变化" — 多快？每秒？每毫秒？
- Layer C："与配置一致 shape 的 histogram" — 具体什么 shape？
- Layer D："至少一次成功返回" — 超时阈值是多少？

这不是 blocking issue，因为量化阈值取决于实际硬件配置，在当前没有真板时强行指定阈值反而是伪精确。但后续真板任务应注意补充。

### N3: 第 3.2 节权限描述偏 Linux

权限描述（`/dev/uio*`、`sudo`、udev、IOMMU）是 Linux 特有的。当前项目运行在 Windows 11 上，如果真板也连在 Windows 机器上，权限模型会完全不同（如 WinDriver / libwdi / Xilinx DMA drivers）。

这不阻塞当前 checklist 的有效性，但后续执行任务应确认目标平台是 Linux 还是 Windows，并相应调整权限描述。

## 7. Missing Tests / Validation

验证方式为只读代码审计，这是任务包 `Verification` 中明确要求的。所有占位符证据经独立交叉验证全部准确。

**无缺失验证。**

## 8. Suspicious Implementation Details

**无。** 本任务是只读文档任务，没有实现细节。

## 9. Recommended Next Action

1. T20 adversarial review 通过，可由 Captain 标记完成并提交 git。
2. T20 是 Phase 2 最后一个有界任务（Milestone 2D）。完成后应考虑：
   - 是否做一次 Phase 2 milestone review（类似 T16 对 Milestone 2A 的 gate review）
   - 是否进入 Phase 3 或继续扩展 Phase 2 任务队列
3. 真板路径的下一步应是"真板 smoke execution plan"（如 readiness 第 7 节所建议），且必须单开独立任务，不能借任何其他任务顺手实现。
4. 物理 cache cleanup（T19 manifest 的执行）仍待 Captain 批准。

---

**Verdict: PASS**
**Blocking issues: 无**
**Non-blocking issues: N1（寄存器名来源不透明），N2（验收标准缺量化阈值），N3（权限描述偏 Linux）**
**Suspicious implementation details: 无**
