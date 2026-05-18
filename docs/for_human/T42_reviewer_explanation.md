# T42 Reviewer 审查说明

## 1. T42 是什么（通俗解释）

T42 是一个"论文结构补全与定位校准"任务。它的作用是在 Milestone 2K（论文组装工具箱就绪）之后、正文撰写之前，把论文骨架中缺失的 Background / Related Work 章节补上，并校准论文的整体定位。

具体来说，T42 回答三个问题：

1. 论文骨架里缺了什么结构？——缺 Background / Related Work 章节。
2. 论文标题应该怎么取？——保守定位 vs 方法向前定位，哪个更安全？
3. Introduction 里的贡献点应该怎么写？——哪些 claim 能写进贡献、哪些不能？

T42 不写论文正文、不跑实验、不改代码、不升级任何证据等级。它只修改文档骨架和产出定位校准笔记。

## 2. T42 的具体实现

### 2.1 任务目标

T42 的目标是产出四个文档：

1. **`docs/paper_draft_skeleton.md`**（修改）——在现有骨架中新增 Background / Related Work 章节（6 个子节），重新组织标题候选为保守组和方法向前组，校准 Introduction 贡献点。
2. **`docs/paper_method_positioning_calibration.md`**（新增）——定位校准笔记，对比保守定位与方法向前定位的利弊，推荐安全选择，列出 8 类禁用措辞。
3. **`docs/review/T42_review.md`**（新增）——Worker 自审报告。
4. **`docs/for_human/T42_explanation.md`**（新增）——中文人类说明。

### 2.2 任务流程

Worker 的工作流程是：

1. 读取 T42 任务包要求的全部输入文档（包括 README、AGENTS.md、项目快照、legacy 审计、实验计划、HIL 边界审计、任务板、决策日志、交接文档、风险清单、claim/evidence ledger、paper skeleton、reviewer risk audit、T34/T35/T41 review、Milestone 2K review、工程方案、阶段结论、后续计划、paper-inspired 草案等）。
2. 在 `docs/paper_draft_skeleton.md` 中插入 Background / Related Work 章节，包含 6 个子标题、允许的 claim/figure/table 映射、blocked claim 限制和起草说明。
3. 重新组织标题候选：将原来的 4 个标题分为"保守组"（2 个）和"方法向前组"（2 个），并标注推荐方案。
4. 校准 Introduction 贡献点：每条贡献明确绑定 claim ID，blocked claims 被显式排除。
5. 产出独立的定位校准笔记，详细对比两种定位的优劣势、推荐方案和禁用措辞。

### 2.3 文件变化

T42 修改/新增了五个文档：

| 文件 | 类型 | 内容 |
|------|------|------|
| `docs/paper_draft_skeleton.md` | 修改 | 新增 Background / Related Work 章节（6 个子节），重新组织标题候选，校准贡献点 |
| `docs/paper_method_positioning_calibration.md` | 新增 | 定位校准笔记 |
| `docs/review/T42_review.md` | 新增 | Worker 自审 |
| `docs/for_human/T42_explanation.md` | 新增 | 中文人类说明 |
| `docs/tasks/Phase2/T42_paper_background_related_work_and_positioning.md` | 修改 | 追加 Verification Record |

没有代码、配置、`runs/`、`artifacts/` 或其他治理文档的变化。

### 2.4 核心结论

- **Background / Related Work 章节规划了 6 个子节**：GKP QEC 问题框架、快回路/慢回路分离、CNN 辅助 QEC 解码文献、经典自适应估计器、teacher-guided residual 定位、benchmark/deployment 证据边界。
- **推荐定位**：方法向前的标题（"A Dual-Loop Teacher-Guided Residual Decoding Framework for Real-Time GKP Error Correction"），配合证据受限的正文。
- **贡献点校准**：5 条贡献分别绑定 C1、C2+C3、C4(partial)、C5、C9，所有 blocked claims（C6/C7/C8/C10/C11）被排除。
- **禁用措辞**：8 类措辞被明确禁止，因为它们会静默升级 blocked claims。

### 2.5 对后续开发的意义

T42 的意义在于为论文正文撰写建立了一个安全的起跑线：

1. **Background / Related Work 补齐后，论文骨架不再缺少关键章节。** 这意味着后续正文撰写可以从任意章节开始，而不需要中途停下来补结构。特别是第 5 子节（"为什么 teacher + CNN residual 不同于 CNN 替代一切"）直接回应了 T35 reviewer risk audit 中的 novelty challenge N1，为方法贡献的叙事建立了框架。

2. **定位校准确认了"方法向前标题 + 证据受限正文"的策略。** 这与实验计划 Section 10.1 的推荐标题一致，也回应了 T35 review N1（标题偏保守）和 Milestone 2K review 的建议。但最终决策权在 Captain/人类——标题选择是策略决策，不是技术决策。

3. **贡献点校准把 C1-C11 的状态直接映射到了论文叙事。** 后续每写一段贡献点，都可以直接对照 claim ledger 确认措辞是否越界，而不需要再从零判断"这句话能不能写"。

4. **禁用措辞清单为后续正文撰写提供了明确的红线。** 例如，"hardware validated"会被替换为"mock-backed software HIL revalidation"，"reproducible training pipeline"会被替换为"one clean-environment CPU-only smoke"。这些替换规则直接对应了 claim ledger 的 Wording Guardrails 和 risk audit 的 Overclaim Wording Traps。

5. **T42 确认了当前证据边界没有被绕过。** 所有 blocked claims 仍然 blocked，所有 risk IDs 仍然有效，没有因论文结构扩展而引入新的风险。

## 3. 为什么我给出了 PASS 的审查结果

### 3.1 任务完成度

T42 的所有 Required Output Shape 条件都已满足：

- `docs/paper_draft_skeleton.md` 已更新：Background / Related Work 章节存在且包含 6 个子标题、allowed evidence map、blocked claim 限制和 drafting notes。标题候选已重新组织。贡献点已校准。
- `docs/paper_method_positioning_calibration.md` 已产出：包含保守定位选项、方法向前定位选项、推荐安全定位、贡献点校准表和禁用措辞清单。
- Verification Record 已追加到任务包。

### 3.2 没有伪实现、mock、stub 或 hardcode

T42 是纯文档任务，没有代码、没有可执行逻辑、没有测试。所有产出都是文档骨架、校准笔记和说明文本。没有把计划写成事实。

### 3.3 没有破坏已有功能

T42 只修改了 `docs/paper_draft_skeleton.md`（在已有骨架中插入新章节和更新标题/贡献点）和任务包（追加 Verification Record）。没有修改任何源码、配置、benchmark protocol、`runs/`、`artifacts/` 或治理结论文档。项目当前的功能状态完全不受影响。

### 3.4 没有过度工程

T42 的产出严格限制在任务包要求的范围内。Worker 没有借机修改治理文档、没有额外添加 claim、没有升级证据等级、没有创建新实验计划、没有开始论文正文撰写。Background / Related Work 是骨架级大纲（子标题 + 起草说明），不是论文正文段落。

### 3.5 发现的非阻塞问题

我在审查中发现了三个小问题：

1. **N1**: Background 子节 6（"量子系统论文中的 benchmark 和部署证据边界"）的 drafting notes 把"对证据边界的显式披露"描述为"对可复现量子工程实践的贡献"。这个措辞可能被审稿人读成"我们的坦诚本身就是新颖性"，而不是"这是好的工程实践"。建议在正文撰写时把这个子节定位为文献综述（其他量子系统论文如何处理软件/硬件证据差距），而不是自我贡献声明。

2. **N2**: 校准笔记推荐了方法向前标题，但标题选择最终是 Captain/人类的策略决策。骨架正确保留了所有四个标题选项，校准笔记正确标注为"推荐"，但建议在 prose expansion 开始前由 Captain/人类显式确认。

3. **N3**: Worker 自审被对抗性审查覆盖——这是 T34/T35/T36/T38/T41 以来的标准模式，不是问题。

这三个问题都不足以阻止 T42 通过，因此最终裁决为 **PASS**。
