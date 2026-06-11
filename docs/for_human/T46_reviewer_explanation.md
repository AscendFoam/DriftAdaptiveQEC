# T46 Reviewer Explanation — 给人类的说明

## 1. T46 通俗解释

### 1.1 这个项目在做什么

项目研究的是一种"量子纠错解码器"——用于 GKP 量子码的自适应纠错系统。系统由两个回路组成：

- **快回路**（微秒级）：每一周期执行线性解码，就像一个快速但不太聪明的计算器
- **慢回路**（毫秒级）：从统计信息中估计噪声漂移，更新快回路的参数，就像一个较慢但更聪明的分析师

当前主方案叫 `Hybrid Residual-B`：让经典算法（teacher）做主要工作，CNN 只学习对控制偏置 `b` 的残差修正。这个方案在正式 benchmark 中稳定优于所有经典基线。

### 1.2 当前遇到了什么问题

在之前的工作中（T36、T38），项目用 `seed=20260429` 这一组随机种子做了一次深入诊断，发现：

- Gated v5（一种更先进的 CNN 方案）在大多数 seed 上明显优于主方案
- 但在 `seed=20260429` 上，Gated v5 和主方案几乎持平甚至略差
- 诊断发现原因是"combined committed-b instability"：teacher 输出的 b 值幅度太大，CNN 输出的残差 delta_b 幅度也太大，两者叠加后导致 committed b 不稳定

但这个发现只基于**一个 seed**，而且只是**观察性诊断**，不是因果证明。

### 1.3 T46 做了什么

T46 没有运行任何新实验。它只是写了一份**计划文件**，回答了一个关键问题：

> 如何把"一个 seed 上的诊断发现"推进到"更可信的机制故事"——而不假装这个故事已经被证明了？

具体来说，T46 的计划文件定义了：

1. **当前能安全说什么**：seed=20260429 上存在 trace-supported 的 combined committed-b 不稳定性诊断证据
2. **什么还说不出来**：不稳定性是否在其他 seed 上复现；干预是否能稳定改善
3. **最小 seed 选择逻辑**：现有 3 个 seed + 新增 3 个，总共不超过 6 个
4. **最小 trace 字段**：17 个字段（与 T38 一致），核心是 teacher_b、delta_b、committed_b 和 window_ler
5. **干预矩阵**：识别了 3 个真正的机制测试和 3 个不属于机制测试的方向
6. **诊断 vs 因果边界**：当前只有诊断证据，因果证据需要干预实验在多 seed 上一致改善
7. **未来执行任务的 go/no-go 规则**

## 2. T46 实现详解

### 2.1 任务目标

T46 的目标是"冻结一份未来机制证据执行包"——定义未来执行任务的最小范围、trace 字段、seed 选择和干预方案，使后续执行任务不需要从头设计就能直接开始。

### 2.2 任务流程

1. Worker 阅读 19 份输入文档（包括 freeze snapshot、claim/evidence table、code truth audit、T36/T38/T44/T45/T53 review 等）
2. 基于 T36+T38 的单 seed 诊断，构建多 seed 扩展计划
3. 定义 10 节计划文件、4 个必要表格
4. 写 review 和 human-facing explanation
5. 更新 task package 的 Worker Output 和 Verification Record

### 2.3 文件变化

| 文件 | 变化类型 | 说明 |
| --- | --- | --- |
| `docs/evidence_packs/mechanism_ablation/seed_mechanism_multi_seed_plan.md` | 新建 | 主计划文件，10 节 + 4 表 + 干预矩阵 + go/no-go |
| `docs/review/T46_review.md` | 新建→被 reviewer 覆写 | Worker 自审后被对抗审查覆写 |
| `docs/for_human/T46_explanation.md` | 新建 | 中文人类说明 |
| `docs/tasks/Phase2/T46_...md` | 修改 | 追加 Worker Output + Verification Record |

**没有变化的文件类型**：源码、config、benchmark runner、runs/、artifacts/、tests/、物理仿真代码。

**治理文件变化**（`docs/05_decision_log.md`、`docs/07_handoff.md`）不是 Worker 产出，而是 T46 开始前的 Captain 治理同步（D-2026-05-22-01 决策记录），属于任务包收紧而非任务执行。

### 2.4 对后续开发的意义

1. **Phase A 执行任务可以立即开始设计**：seed 列表、trace 字段、输出格式、比较维度都已在计划中锁定，不需要额外设计工作
2. **执行范围有明确上限**：≤6 seed、冻结 4 场景、≤1 个干预变体、不改变 benchmark 语义——这使得后续执行任务可以保持在 bounded task 范围内
3. **诊断与因果的边界有明确语言规则**：后续执行任务的产出文档不会意外地把观察性证据写成因果证明
4. **go/no-go 规则为 Captain 提供了明确的判断标准**：如果 Phase A 显示模式不可复现，项目可以直接决定保持 C4 partial 状态，用诊断措辞完成论文，而不是无止境地追求因果证明

### 2.5 与项目整体路线的关系

T46 是 Milestone 2P（Mainline Evidence Hardening）的一部分。它的上游是 T36（单 seed 失败诊断）和 T38（单 seed trace 导出），下游是建议的 Phase A 执行任务（多 seed trace 探针）和可能的 Phase B（干预实验）。

T46 完成后，项目状态不变：
- 仍然是 `Phase 2: Controlled Development / Research Reality Recovery Mode`
- R10（teacher mechanism evidence 缺口）仍未关闭
- C4 仍为 `partial`
- 论文 prose 仍然暂停

T46 不改变任何事实状态，它只冻结了一份未来执行计划。

## 3. 为什么给出了 PASS 的 review 结果

### 3.1 任务目标完全达成

任务包要求产出 10 节计划文件、4 个必要表格、review 文档和 human-facing explanation。所有文件均已产出，结构完整，内容准确。

### 3.2 没有禁止范围内的越界

- 没有修改源码、config、benchmark runner、tests、runs/、artifacts/
- 没有运行任何 benchmark、training、.tflite、硬件或 cleanup 命令
- 没有修改治理文件（`05_decision_log.md` 和 `07_handoff.md` 的变化是 T46 开始前的 Captain 同步，不是 Worker 产出）
- 没有把单 seed 诊断升级成多 seed 确认或因果证明

### 3.3 证据等级诚实

Grep 确认所有"multi-seed confirmation"、"causal proof"、"mechanism proven"、"root cause identified"的出现均在否定上下文中（如 "does not claim"、"unsupported"、"unsafe wording"）。C4 正确保持为 `partial`。

### 3.4 计划保持 bounded

- Seed pack ≤ 6
- 仅用冻结 4 场景
- 干预变体 ≤ 1-2
- 不改变 benchmark 语义
- 明确与 T47/T48/T49 分离

### 3.5 Worker 自审准确

Worker 自审识别了 4 个非阻塞问题（3-seed 样本量、clip 降幅、现有 seed 复用、I3 实现可行性），全部合理且分类正确。没有发现自审中有错误或遗漏。

### 3.6 非阻塞问题总结

| # | 问题 | 为什么不阻塞 |
| --- | --- | --- |
| N1 | seed 20260430 与 20260429 相邻 | 计划已显式说明选择理由（测试模式是否延续到相邻 seed），是有效的设计选择 |
| N2 | 治理文件变化在工作区中与 T46 输出共存 | 变化是 Captain 治理同步，不是 Worker 越界；Captain 可在整合时分别处理 |
| N3 | 计划未提供 Phase A 执行时间估计 | 时间估计属于执行任务责任，不属于计划 gate |
| N4 | 计划未直接回引 `docs/paper_materials/paper_claim_evidence_ledger.md` | C4 标签正确，不需要显式交叉引用 |
| N5 | Worker 自审准确无遗漏 | 纯信息性说明 |

## 4. Worker 已有 review 和 explanation 的补充说明

### 4.1 Worker 自审（原 `docs/review/T46_review.md`）

Worker 自审的 4 个非阻塞问题全部合理：

1. **N1 3-seed 样本量**：确实，用 3 个 seed 定义"正常"与"异常"不够稳健。Phase A 的 6-seed 设计是对此的正确回应。
2. **N2 clip 降幅**：0.12→0.06 确实步幅较大。计划正确地将具体值锁定推迟到执行任务，而非在计划 gate 中承诺。
3. **N3 现有 seed 复用**：20260427/20260428 的现有 artifacts 是否需要重新运行，是执行层面的决策。T38 已验证 trace-export 可从现有 `hil_events.json` 工作。
4. **N4 I3 实现可行性**：teacher-delta 衰减可能需要代码修改。计划正确地将 I1（config-only）排在更高优先级。

**补充**：Worker 自审没有讨论 seed 20260430 与 20260429 的相邻性问题（见我的 N1），也没有讨论治理文件在工作区中的共存问题（见我的 N2）。这两个观察已在我的对抗审查中补充。

### 4.2 Worker explanation（`docs/for_human/T46_explanation.md`）

Worker 的中文说明简洁准确，覆盖了 4 个要点：做了什么、没改变什么、review 结论、下一步。

**补充**：

1. Worker explanation 没有讨论 T46 在项目整体路线中的位置（上游 T36/T38、下游 Phase A/B 执行任务），这在我的 explanation 第 2.5 节中补充。
2. Worker explanation 的"N2 干预 I1 的 clip 降幅"说明中说"具体值由执行任务锁定"——这是正确的，但可以更明确地补充：如果执行任务选择更保守的步幅（如 0.12→0.09），计划的 go/no-go 规则仍然适用。
3. Worker explanation 的第 4 节列出了 Phase A 和 Phase B 的建议，但没有提到"如果 Phase A 模式不可复现"的退路。计划文件 Section 9.3 和我的 review 都明确了这个退路：C4 保持 partial，论文用诊断措辞。
