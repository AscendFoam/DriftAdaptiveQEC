# T54 Reviewer Explanation — 给人类的说明

## 1. T54 通俗解释

### 1.1 这个项目在做什么

项目研究的是一种"量子纠错解码器"——用于 GKP 量子码的自适应纠错系统。系统由两个回路组成：

- **快回路**（微秒级）：每一周期执行线性解码，就像一个快速但不太聪明的计算器
- **慢回路**（毫秒级）：从统计信息中估计噪声漂移，更新快回路的参数，就像一个较慢但更聪明的分析师

当前主方案叫 `Hybrid Residual-B`：让经典算法（teacher）做主要工作，CNN 只学习对控制偏置 `b` 的残差修正。

另一个更先进的 CNN 方案叫 **Gated v5**——它在大多数随机种子上性能优于主方案，但之前在 `seed=20260429` 上出现了异常（接近持平甚至略差）。

### 1.2 T36/T38 的发现

T36 和 T38 对 `seed=20260429` 做了深入诊断，发现了一个"combined committed-b 不稳定性"现象：Gated v5 的 teacher 输出的 b 值幅度很大，CNN 输出的残差 delta_b 幅度也很大，两者叠加后导致最终使用的 committed b 不稳定。

但这个发现只基于**一个种子**，不知道这个问题是 `seed=20260429` 独有的，还是普遍存在的。

### 1.3 T54 做了什么

T54 是 T46 计划推荐的第一步执行任务（Phase A）：**用 6 个种子做多种子 trace-only 泛化探针**，判断 committed-b 不稳定性是否在 `seed=20260429` 之外也出现。

T54 复用了 3 个已有种子的数据（`20260427`、`20260428`、`20260429`），新增跑了 3 个新种子（`20260425`、`20260430`、`20260510`），总共分析了 **57,586 条 trace 行**。

关键发现是：

- committed-b 不稳定性**不是 `seed=20260429` 独有的**——5/6 个种子都存在
- 但模式比之前想的更复杂，6 个种子分为三类：
  - **安静型（1/6）**：两种模式都稳定，无 instability
  - **经典型（4/6）**：Full 稳定，Gated v5 不稳定，但 Gv5 性能更好
  - **普遍不稳定型（1/6）**：两种模式都不稳定
- 不稳定性在大多数种子上反而**帮助了** Gated v5 的性能，而非损害

T54 没有运行任何干预实验，没有修改代码/config，没有升级 trace 证据为因果证明。

---

## 2. T54 实现详解

### 2.1 任务目标

T54 的目标是执行 T46 计划中的 Phase A——做一个有界的多种子 trace-only generalized 探针，回答：

1. committed-b 不稳定性是否在 `seed=20260429` 之外也出现？
2. 模式是孤立的、部分重复的、还是广泛重复的？
3. 后续干预实验（Phase B）是否有理论基础？

任务包限制严格：仅限 `Full` vs `Gated v5`、4 个已冻结场景、最多 6 个种子、trace-only（不允许干预变体）、不修改源码/config。

### 2.2 任务流程

Worker 的执行流程：

1. **读取输入**：T46 计划、T36/T38 诊断报告、T45 benchmark 边界、claim/evidence ledger、相关源码和 config
2. **Preflight 已有种子**：检查 20260427 和 20260428 的已有 artifact 是否包含所需的 trace 字段。确认 19 个字段全部存在，无需重跑
3. **复用已有 trace**：
   - 20260429：直接从 T38 trace export 复制
   - 20260427/20260428：从已有 V5 chunked pair benchmark artifacts 导出
4. **新增种子全流程**：对 20260425、20260430、20260510，使用 DLEnv 环境执行完整 pipeline（数据集构建 + 训练 + benchmark + trace 导出）。3 个新种子总计约 14 小时
5. **Cross-seed 分析**：用 `cross_seed_analysis.py` 读取所有 6 个种子的 trace CSV，产出 3 个 cross-seed 汇总 CSV
6. **产出文档**：8 节报告 + 4 张表格 + review + 中文说明

### 2.3 文件变化

| 文件 | 变化类型 | 说明 |
| --- | --- | --- |
| `docs/multi_seed_trace_generalization_probe.md` | 新建 | 主报告文件，8 节 + 4 张必需表格 |
| `docs/review/T54_review.md` | 新建→被 reviewer 覆写 | Worker 自审后被对抗审查覆写 |
| `docs/for_human/T54_explanation.md` | 新建 | 中文人类说明 |
| `docs/tasks/Phase2/T54_multi_seed_trace_only_generalization_probe.md` | 修改 | 追加 Worker Output + Verification Record |
| `runs/T54_multi_seed_trace_phase_a_20260522/` | 新建 run root | 包含 6 个 seed 的 trace exports、3 个 cross-seed CSV、seed reuse manifest |

**没有变化的文件类型**：源码、config、benchmark runner、governance 文档（00–08）、test 文件。

**新增 benchmark 运行目录**（`runs/teachrepr/p4_benchmark/trp604{25,30,510}_resume/`）：这些是 paired runner 默认输出路径创建的新目录，不是对历史运行目录的覆盖或修改。

### 2.4 对后续开发的意义

1. **机制叙事需要更新**：原来 T36/T38 说"seed=20260429 上有 committed-b 不稳定性导致 Gv5 受损"——现在知道这个 instability 是广泛存在的，而且大多数情况下**帮助了** Gv5 性能。`seed=20260429` 不是最典型的反而是一个边界情况（唯一在 static_bias_theta 上 Gv5 略差）。

2. **Phase B 干预设计需要更精细**：不能简单假设"降低残差幅度 = 性能改善"。干预必须在所有 6 个种子上测试，特别关注：
   - 20260425（安静型）：干预可能无效
   - 20260510（普遍不稳定型）：干预可能影响 Full 和 Gv5 两个模式

3. **C4 必须保持 `partial`**：mechanism story 比以前更复杂，不是简单的高 committed-b = 坏。论文措辞必须包含这些限定条件。

4. **三个种子类别不是固定的**：它们只反映 6 个种子的样本，可能随着更多种子或不同训练条件而变化。论文中不能把这三个类别写成穷尽分类。

### 2.5 与项目整体路线的关系

T54 是 Milestone 2P（Mainline Evidence Hardening）的执行任务。它的上游是：

- **T46**（multi-seed mechanism plan）——定义了 Phase A 的 seed 选择、trace 字段和执行边界
- **T36/T38**（单种子诊断和 trace 导出）——提供了诊断方法和 trace 导出路径

下游是：

- **可能的 Phase B（干预实验）**——如果项目决定推进 I1（降低 residual clip），需要在所有 6 个种子上测试
- **T47（paper ablation result-pack）**——需要在更清楚的机制证据基础上再做

T54 完成后，项目状态不变：
- 仍然是 `Phase 2: Controlled Development / Research Reality Recovery Mode`
- R10（teacher mechanism evidence 缺口）仍未关闭
- C4 仍为 `partial`
- 论文 prose 仍然暂停

T54 不改变任何 claim 状态，它只提供了更完整的诊断证据——并且这个证据显示机制故事比以前想的更复杂。

---

## 3. 为什么给出了 PASS 的 review 结果

### 3.1 任务目标完全达成

任务包要求产出 8 节报告、4 张必需表格、review 文档和中文说明。所有文件均已产出，结构完整，内容准确。3 个 cross-seed CSV 和 seed reuse manifest 均已生成。

### 3.2 没有禁止范围内的越界

- 没有修改源码、config、benchmark runner、tests、governance 文档
- 没有运行任何干预变体（Phase B 未执行）
- 没有覆盖历史 `runs/` 或 `artifacts/` 路径
- 使用 T46 锁定的 seed pack（6 个 seed），未扩展
- 使用已冻结的 4 个场景和 2 个模式（Full vs Gated v5）
- 没有扩 benchmark 边界、没有添加新 baseline

### 3.3 证据等级诚实

Grep 确认所有"causal proof"、"mechanism proven"、"root cause identified"、"multi-seed confirmation"的出现均在否定上下文中（non-claims 段或"unsupported"标记）。C4 正确保持为 `partial`。

### 3.4 发现本身有价值且表述诚实

Worker 没有因为"5/6 种子都出现 instability"就简化结论。相反，报告诚实地指出了三个种子类别、instability 在大多数情况下帮助而非损害性能、以及 20260510 上 Full 本身也不稳定的事实。这种"广泛复现但有重要差异"的表述是科学诚实的。

### 3.5 边界问题全部可接受

对抗审查补充了一个 Worker 自审未涉及的边界问题：**种子复用清单（seed_reuse_manifest.json）对新种子的行数和字段可用性记录为 null**。这不影响最终报告的正确性（报告正文已给出完整数据），但使 manifest 本身不够自包含。

我的审查中识别了与非阻塞问题（N2：seed reuse manifest 对新种子元数据不完整），与 Worker 自审的 5 个问题合并，共 5 个非阻塞问题，全部归为 `accepted`。

### 3.6 非阻塞问题总结

| # | 问题 | 为什么不阻塞 |
| --- | --- | --- |
| N1 | Benchmark 输出目录在 T54 根之外（`runs/teachrepr/` 下） | 是新目录非覆盖；trace 导出和分析 CSV 都在 T54 根内 |
| N2 | Seed reuse manifest 对新种子的行数/字段为 null | 最终报告已给出完整数据；manifest 记录的是 pre-rerun 决策状态 |
| N3 | 两个不同的 paired-seed 约定被混用 | 分析按实际 seed 值分组，不受配对约定影响 |
| N4 | `cross_seed_analysis.py` 是脚本而非 CSV/JSON 摘要 | 在 run root 内、不影响源码路径的辅助脚本 |
| N5 | 机制结论从"广泛复现"细化为"广泛复现但有重要差异" | 细化的结论更诚实，增加了科学价值 |

---

## 4. Worker 已有 review 和 explanation 的补充说明

### 4.1 Worker 自审（原 `docs/review/T54_review.md`）

Worker 自审的 5 个非阻塞问题全部合理：

1. **N1 benchmark 输出在 T54 根外**：确实，paired runner 的默认输出路径将 benchmark 目录创建在 `runs/teachrepr/` 下。Worker 正确地将 trace export 和所有分析 CSV 收敛到 T54 根内。
2. **N2 cross-seed 使用混合 paired-seed 数据**：确实，不同来源的种子使用了不同的配对约定。分析按实际 seed 值分组是正确的做法。
3. **N3 新种子 pipeline 执行时间**：确实，3 个新种子需要完整的数据集构建 + 训练 + benchmark，14 小时是合理的。
4. **N4 cross_seed_analysis.py 创建在 T54 run root**：辅助脚本在 run root 内，不修改源码路径。
5. **N5 机制结论细化**：从"broadly repeated"到"broadly repeated with qualifications"是诚实且有价值的细化。

**补充**：Worker 自审没有发现 seed reuse manifest 中新种子元数据不完整的问题（见我的 N2）。这个观察已在我的对抗审查中补充。

### 4.2 Worker explanation（`docs/for_human/T54_explanation.md`）

Worker 的中文说明简洁准确，覆盖了 5 个要点：做了什么、核心发现（三个种子类别）、机制泛化结论、对后续任务的建议、没有改变什么。

**补充**：

1. Worker explanation 没有讨论 T54 在项目整体路线中的位置（上游 T46/T36/T38、下游 Phase B/T47），这在我的 explanation 第 2.5 节中补充。

2. Worker explanation 的第 4 节建议了"Phase B 干预仍然有理由做"——这是正确的，但更关键的额外信息是：**干预实验需要调低预期**，因为 instability 在大多数种子上是帮助而非损害。计划中的干预 I1（降低 residual clip）可能改善 20260429 上的性能，但也可能在 20260427/20260428/20260430 上损害 Gv5 的已有优势。

3. Worker explanation 没有提到**20260510 的普遍不稳定性**可能是一个独立的问题——它可能是一个 Full 模式的问题（或训练 artifact），恰好与 Gv5 的 committed-b instability 重叠。这个区分对于后续机制分析很重要。

4. Worker explanation 可以更明确地说明**三个种子类别不是固定的**——它们只反映 6 个种子的样本，不能穷尽所有可能的模式。
