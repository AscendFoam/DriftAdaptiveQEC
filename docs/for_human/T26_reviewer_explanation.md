# T26 Reviewer 人话版说明

## 1. 这个 Task 在做什么（通俗版）

想象你在造一辆自动驾驶汽车，目前已经完成了基础测试——在各种路况下跑了一圈，记录了成绩（这就是 T24 的 frozen benchmark）。现在你想给车加一个新的"传感器校准模块"（statcalib），但你不确定能不能安全地加、加在哪里、会不会影响已有的测试成绩。

T26 就是做这件事：**不是去造那个模块，而是先画一张蓝图，写清楚"如果要加，应该怎么加、加在哪里、不能碰什么"。**

最终结论是：可以做，但必须作为一个独立的比较对象（单独一条赛道），不能偷偷塞进已有的成绩排行榜里。

## 2. 任务实现详解

### 2.1 任务目标

T26 的目标是产出一份**可行性判断文档**（feasibility gate），回答以下问题：

1. 在当前代码和基准测试的基础上，能不能安全地加入一个 calibration/statcalib 比较器？
2. 如果能，最小设计是什么样的？
3. 有哪些东西绝对不能动？

### 2.2 任务流程

Worker 按照任务包的要求，执行了以下只读步骤：

1. **阅读现有文档**：读了正式基准协议（`P4_benchmark_formal_protocol.md`）、前序任务 review（T24-T29）、风险清单、以及核心代码文件（`param_mapper.py`、`run_p4_multiscenario_benchmark.py`）。
2. **判断可行性**：基于上述只读审计，判断 statcalib 作为独立比较器是可行的，但不能合并到已冻结的 T24 基准集。
3. **产出设计文档**：写了 gate 文档、review 文档和人话版说明，在任务包中补充了 Worker Output 和 Verification Record。

### 2.3 文件变化

| 文件 | 变化类型 | 说明 |
|------|----------|------|
| `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md` | 新建 | 主文档：可行性判断、设计分类、最小接口、验证计划、非声明 |
| `docs/review/T26_statcalib_feasibility_gate.md` | 新建 | Worker 自审：只读范围确认、检查文件清单、可行性结论 |
| `docs/for_human/T26_explanation.md` | 新建 | 人话版说明 |
| `docs/tasks/Phase2/T26_statcalib_feasibility_gate.md` | 追加 | Worker Output 和 Verification Record |

**没有任何源代码、配置文件、运行目录或 artifact 被修改。** 这是纯文档任务。

### 2.4 对后续开发的意义

1. **划清边界**：gate 文档明确说 statcalib 只能是独立 comparator lane，不能修改已冻结的 T24 基准。这防止了后续实现任务不小心"污染"已有证据。
2. **定义最小接口**：`StatCalibInput` / `StatCalibOutput` 的概念性定义，为后续实现任务提供了设计起点。
3. **列出先决条件**：6 条 prerequisite checklist 确保后续实现任务不会跳过必要步骤。
4. **约束未来任务包**：gate 文档要求任何后续实现任务必须包含 Allowed files / Forbidden scope / Verification / Docs to update 四个字段。

在项目整体路线图（[docs/04_task_board.md](../04_task_board.md)）中，T26 属于 Milestone 2I: Mechanism Evidence Hardening。它的前置任务是 T27-T29（修复 teacher diagnostics 和报告格式），后续可能是 T30（paper-inspired statcalib 实现）或其他 statcalib 实现任务。T36（seed 失败机理诊断）是并行的独立优先级。

## 3. 为什么给出 PASS 的 Review 结果

### 3.1 任务确实完成了

对照任务包的"Expected Output"逐项检查：

- `docs/evidence_packs/statcalib_fr8/statcalib_feasibility_gate.md` 存在且包含全部 7 个必要章节（current boundary、objective、prerequisite、adopted/deferred/rejected、interface、metrics、non-claims）
- `docs/review/T26_statcalib_feasibility_gate.md` 存在且包含 scope confirmation、files inspected、verdict、warnings
- `docs/for_human/T26_explanation.md` 存在且提供了非技术说明
- 任务包已补充 Worker Output 和 Verification Record

### 3.2 没有伪实现、mock、stub 或 hardcode

T26 没有写任何代码。gate 文档在"Explicit Non-Claims"中明确写了：

- "This document does not claim statcalib exists."
- "This document does not claim statcalib has been validated."

没有任何把"计划"写成"已完成事实"的情况。

### 3.3 没有破坏已有功能

- `git status` 确认只有 4 个文档文件变更，全部在 Allowed files 范围内。
- 没有新的 `runs/` 目录。
- 没有修改 frozen benchmark protocol、baseline 集合、scenario 集合、seed/repeat policy 或 metric definitions。

### 3.4 没有过度工程

gate 文档的内容量适中——有设计分类（adopted/deferred/rejected）、有最小接口概念、有验证顺序，但没有预实现任何代码或写过多的架构设计。对于一个 feasibility gate 来说，这个分寸是合适的。

### 3.5 非阻塞备注

Review 中标注了 3 个非阻塞项（N1-N3），主要是说 Worker 自审文档可以更详细、人话版说明可以稍长、接口定义在后续任务中需要具体化。这些都不影响 T26 本身的完成质量，只是对后续任务的提醒。
