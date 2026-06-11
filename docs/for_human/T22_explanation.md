# T22: Real-Board Smoke Execution Plan — 通俗解释与 Review 说明

## 一、这个 Task 在做什么？（通俗版）

### 背景

T20 已经写了一份"真板准备清单"——列出了真板需要补齐的前置条件。但那份清单有几个模糊的地方：

1. **不知道在什么操作系统上跑**：当前开发机是 Windows，但代码里写的是 Linux 的 `/dev/uio*` 设备路径。到底在哪跑？
2. **不知道寄存器地址对不对**：文档里提到了一些寄存器名，但没说这些名字是从哪里来的、地址值是多少。
3. **不知道"成功"的标准是什么**：验收标准都是定性的（"能读出"、"至少一次"），没有具体数字。

### T22 做了什么

T22 的任务是：**把 T20 的模糊点全部具体化，写一份可以直接拿去执行的施工图。** 具体来说：

- 补了 4 种可能的运行平台选择（Linux 本机、Windows 本机、WSL、远端主机），每种都列出了"进入前必须确认"和"不确定就不能进入"的条件
- 直接读了 `axi_map.py` 源码，把 14 个寄存器地址、7 个 bit mask、定点数格式、bank 编码规则全部抄出来，形成审计清单
- 直接读了 `dma_client.py` 源码，把 DMA 缓冲区的配置项和运行时语义全部列出来
- 给 4 层验收标准补了具体数字（比如"3 次连续读回"、"1 秒超时"、"0 容忍 DMA 失败"）
- 写了命令骨架（但标注"未执行"）

用一句话总结：**T22 把"真板需要什么"从定性清单升级成了带有具体地址、具体数字、具体命令的执行计划。**

## 二、任务实现详细解释

### 2.1 任务目标

为后续真板 smoke 制定可执行计划，补齐目标平台决策点、AXI/register map 审计路径、DMA buffer 审计路径与量化验收阈值草案。本任务仍不调用硬件、不实现真板 backend。

### 2.2 任务流程

1. **读取 T20 review 和 T21 milestone review**：了解前置任务的 review 结论（T20 的 N1/N2/N3 问题）和 milestone gate 判断。

2. **审计 `axi_map.py`（215 行）**：
   - 提取 `AxiRegisterMap` dataclass 中的全部 14 个寄存器地址（`ctrl`、`status`、`hist_meta`、`overflow_count`、`k11`-`b2`、`active_bank`、`epoch_id`、`commit_epoch`、`hist_seq`）
   - 提取 7 个 bit mask（`ctrl_start`、`ctrl_reset_hist`、`ctrl_commit_bank`、`status_ready`、`status_hist_ready`、`status_commit_ack`、`status_overflow_alert`）
   - 提取 `FixedPointFormat`（`Q4.20`）和 bank 编码（`A→0, B→1`）

3. **审计 `dma_client.py`**：
   - 提取 `MemoryMappedDMAConfig`（`path`、`buffer_bytes`、`buffer_count`）
   - 提取 `DMAReadout` 结构（`buffer_id`、`byte_count`、`metadata`）
   - 推导 `32×32 float32` → `byte_count = 4096`

4. **撰写 execution plan**：产出 `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`，10 节，323 行。

5. **更新治理文档**：同步更新 `real_board_hil_readiness.md`、`03_hil_p4_boundary_audit.md`、`04_task_board.md`、`07_handoff.md`、`08_risks_and_open_questions.md`。

### 2.3 代码/配置文件的变化

**没有代码变更。** 没有配置文件变更。所有改动都在 `docs/` 目录内。

核心新增文件：

| 文件 | 变化类型 | 内容 |
|------|----------|------|
| `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md` | 新增 | execution plan 全文（10 节，323 行） |

修改的允许范围文件：

| 文件 | 修改内容 |
|------|----------|
| `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md` | 追加 T22 平台决策点引用 |
| `docs/03_hil_p4_boundary_audit.md` | 追加 T22 约束和推荐表述 |
| `docs/04_task_board.md` | 追加 Worker 只读产出 |
| `docs/07_handoff.md` | 追加 T22 Worker 只读结果 |
| `docs/08_risks_and_open_questions.md` | 同步 T22 结论 |

### 2.4 对后续开发的意义

1. **解决了 T20 review 的三个 N 问题**：
   - N1（寄存器名来源不透明）→ 第 4 节逐项从 `axi_map.py` 映射，23 项全部有源码对应
   - N2（验收标准缺量化阈值）→ 第 6 节补了 Layer A-D 的具体数字和 fail-fast budget
   - N3（权限描述偏 Linux）→ 第 3 节列出 4 种宿主模型的决策树，每种都有排除条件

2. **形成可执行的审计清单**：后续真板任务不需要再重新翻源码，直接按第 4/5 节的清单逐项对齐 RTL/bitstream 文档即可。

3. **与 T20 readiness 形成互补**：T20 回答"为什么当前不能写真板完成"，T22 回答"如果要写真板完成，必须按什么顺序执行哪些检查"。后续真板任务应同时引用两份文档。

4. **Phase 2 Milestone 2F 的唯一任务**：T22 完成后，Phase 2 的全部六个 Milestone（A: 证据增强、B: 环境清单、C: 仓库清理、D: 硬件准备、E: Phase 2 gate、F: 真板规划）都已完成至少一个有界任务。

## 三、为什么给出 PASS_WITH_WARNINGS 的 Review 结果？

### 审查逻辑（Adversarial Mode）

本次是 adversarial review，我从 6 个维度检查，额外做了 5 项攻击性检查：

1. **任务是否真的完成** — 是。T22 目标是制定 execution plan，plan 已产出，覆盖了任务包要求的全部 7 项内容。所有 AXI 寄存器地址（14 个）和 bit mask（7 个）经独立读取 `axi_map.py` 全文验证，23 项全部与源码吻合。

2. **是否有伪实现** — 无。所有命令标注 `NOT EXECUTED IN T22`。第 1 节写明"不是硬件执行记录"。第 6 节标题是"Threshold Draft"。

3. **是否缺测试或验证** — 不缺。任务包限定只读审计。AXI/DMA 交叉验证全部通过。

4. **是否过度工程** — 否。10 节内容服务于单一目标（把 T20 的模糊点具体化），没有多余抽象。量化阈值是合理的最小必要条件。

5. **是否破坏已有功能** — 否。无代码变更。

6. **文档是否把计划写成事实** — 否。多处显式区分 plan 与 execution。

### 为什么是 PASS_WITH_WARNINGS 而不是 PASS

有一个 WARNING：

**N1: 5 个 out-of-scope 文件被修改。** `00_project_snapshot.md`、`01_legacy_audit.md`、`05_decision_log.md`、`06_repo_noise_governance.md`、`T21_phase2_milestone_review.md` 不在 T22 的 Allowed files 列表中。内容上看属于 Captain 整合阶段的治理文件同步（T21 gate + T22 引用更新），不是 Worker 越界。但严格按规则，这些文件不应出现在 T22 Worker 的 diff 中。

这不构成 BLOCK，因为：
- 没有代码变更
- 没有 benchmark 口径变更
- 没有硬件调用
- 内容属于治理文件的任务引用同步

但需要 Captain 确认这些修改的归属。

### 另外两个非阻塞提醒

- **N2**：第 7.1 节 preflight 命令中 `print(AXI_REGISTER_MAP)` 会输出 dataclass repr，不是格式化地址表。后续执行任务可能需要更可读的输出格式。
- **N3**：第 5.3 节假设 `byte_count = 4096`（基于 `32×32 float32`），需要后续执行任务确认实际 bitstream 是否匹配。

### 结论

Worker 完成了 T22 的全部目标，AXI/DMA 审计清单与源码完全吻合，量化阈值合理，边界表述诚实。唯一的 WARNING 是 out-of-scope 文件修改，需要 Captain 确认归属。**PASS_WITH_WARNINGS**。
