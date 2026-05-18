# T43 Reviewer 审查说明

## 1. T43 是什么（通俗解释）

T43 是一个"论文 Background / Related Work 正文撰写"任务。它的作用是在 T42（骨架补全与定位校准）之后，把论文骨架中 Background / Related Work 的 6 个子节从骨架大纲变成真正的段落级学术正文。

具体来说，T43 回答一个问题：

> 论文的 Background / Related Work 章节，用真实的学术语言写出来，应该是什么样？

T43 不写论文的其他章节（Abstract、Introduction、Method、Results、Conclusion），不跑实验，不改代码，不升级任何证据等级。它只产出 Background / Related Work 的有界正文草稿。

## 2. T43 的具体实现

### 2.1 任务目标

T43 的目标是产出三个文档：

1. **`docs/paper_background_related_work_draft.md`**（新增）——Background / Related Work 的有界正文草稿，包含 6 个子节的段落级学术正文。
2. **`docs/review/T43_review.md`**（新增）——Worker 自审报告。
3. **`docs/for_human/T43_explanation.md`**（新增）——中文人类说明。

另外，在任务包 `docs/tasks/Phase2/T43_paper_background_related_work_prose_draft.md` 中追加了 Verification Record。

### 2.2 正文的 6 个子节

Worker 为 Background / Related Work 撰写了 6 个子节的完整段落级正文：

1. **GKP 量子纠错与自适应解码问题**：介绍 GKP 编码的基本原理、syndrome 测量、线性解码规则 $\Delta = Ks + b$，以及噪声参数漂移导致固定解码器失配的工程动机。这是纯背景知识，不涉及项目特有的 claim。

2. **双回路时间尺度分离**：解释快回路（~5μs 确定性线性解码）和慢回路（10–100ms 统计估计与参数更新）的架构设计，以及参数 bank 切换的原子性约束。这也是纯背景/架构描述。

3. **机器学习辅助 QEC 解码**：综述 ML-based 解码在 surface code 等离散变量码上的进展（引用 [3]--[7]），指出 GKP 连续变量场景的差异，以及本工作用 Tiny-CNN 作为慢回路估计器的定位。关键措辞是"不让 CNN 替代整个快回路解码规则，只学习慢回路参数"——这正是 method-forward 定位的核心。

4. **经典自适应漂移跟踪方法**：描述五个经典基线（EKF、UKF、Window Variance、RLS Residual-B、Constant Residual-Mu），明确 UKF 是最强经典基线。引用了 supported claims C2/C3。使用了正确的术语"frozen-set formal benchmark"和"mock-backed software HIL revalidation"。

5. **Teacher-Guided 残差修正定位**（核心方法定位子节）：解释"为什么不让 CNN 直接回归绝对参数"——因为"离线训练改善 ≠ formal HIL 改善"（稳定结论 9.1 第 7 条）。阐述 teacher-guided residual-b 的两个机制：teacher 提供稳定锚点，residual 降维简化学习目标。引用了 supported claim C3 和稳定结论。这是论文 novelty defense 的核心段落。

6. **量子系统验证中的证据边界**：简短的中立段落，讨论量子系统研究中软件模拟与硬件验证之间的证据落差。正确标注 C5 为 supported、C6/C7/C8 为 blocked。

### 2.3 文件变化

T43 新增/修改了四个文档：

| 文件 | 类型 | 内容 |
|------|------|------|
| `docs/paper_background_related_work_draft.md` | 新增 | Background / Related Work 6 子节的段落级正文 |
| `docs/review/T43_review.md` | 新增 | Worker 自审（后由对抗性审查覆盖） |
| `docs/for_human/T43_explanation.md` | 新增 | 中文人类说明 |
| `docs/tasks/Phase2/T43_paper_background_related_work_prose_draft.md` | 修改 | 追加 Verification Record |

没有代码、配置、`runs/`、`artifacts/` 或其他治理文档的变化。

### 2.4 核心结论

- **正文草稿包含了 6 个子节的完整段落**，从纯背景（GKP 编码原理）到方法定位（为什么 teacher + residual 不同于 CNN 替代一切），再到证据边界讨论。
- **所有 blocked claims 保持 blocked**：C6（训练复现性）、C7（.tflite runtime）、C8（真板 HIL）在正文中被明确标注为"未验证"或"不可用"，没有被隐式升级。
- **没有使用任何禁用短语**：审查中对照 calibration note 的 8 类禁用短语逐一搜索，未发现违规。
- **method-forward 定位保持正确**：正文将 teacher-guided residual 定位为方法贡献，但没有声称"state-of-the-art"或"全面优于所有经典方法"。
- **subsection 5 的 novelty claim 合理**：使用了"to the best of our knowledge"限定词，且 novelty claim 范围窄化为"teacher + residual 应用于实时 GKP 解码"，而非泛化的"我们的方法全面创新"。

### 2.5 对后续开发的意义

T43 的意义在于论文从"骨架"进入了"正文撰写"阶段：

1. **Background / Related Work 正文草稿是后续所有章节的叙事基础。** Introduction 中的方法定位、Method 章节的系统描述、Experiment 章节的实验协议，都需要与 Background 中的术语和叙事保持一致。例如，subsection 2 中定义的"快回路/慢回路时间尺度分离"是 Method 章节系统描述的前置框架；subsection 5 中阐述的"teacher + residual vs. CNN 替代一切"是整个论文 novelty defense 的核心论点。

2. **subsection 5 直接回应了 reviewer risk audit 的 novelty challenge N1。** T35 识别出的最大审稿风险是"论文没有明确说明 teacher-guided residual 与直接 CNN 回归有什么区别"。subsection 5 用两个完整的段落解释了这个区别：离线训练改善不等于 HIL 改善（稳定结论 9.1 第 7 条），teacher 提供稳定锚点，residual 降维简化学习目标。这为后续 Introduction 的 contribution bullets 提供了叙事支撑。

3. **证据边界纪律已经在正文中建立。** subsection 6 正确标注了所有 blocked claims，后续章节如果引用相同的概念，可以直接复用这里的措辞模板（"mock-backed software HIL revalidation"、"one clean-environment CPU-only training smoke"等）。

4. **引用标记 [1]--[7] 为后续文献管理建立了起点。** 后续章节可以扩展这个引用体系，并在论文组装时统一映射到具体文献。

5. **T43 确认了 method-forward framing lock 在正文层面的可行性。** T42 确定了 method-forward 标题 + evidence-bounded 正文的定位策略，T43 在实际正文中验证了这一定位是否可行——结论是可行的，正文可以自然地以方法贡献为中心展开叙事，同时保持证据边界诚实。

## 3. 为什么我给出了 PASS 的审查结果

### 3.1 任务完成度

T43 的所有 Required Output Shape 条件都已满足：

- `docs/paper_background_related_work_draft.md` 已产出：包含 6 个子节的段落级正文，不是要点列表或骨架大纲。
- 正文范围严格限定在 Background / Related Work：没有越界到 Abstract、Introduction、Method、Results、Conclusion 或 Appendix。
- subsection 6 保持简短且中立，符合任务包的裁量权条款。
- Worker 自审已产出。

### 3.2 没有伪实现、mock、stub 或 hardcode

T43 是纯文档任务，没有代码、没有可执行逻辑、没有测试。所有产出都是段落级正文、审查报告和说明文本。正文中的引用标记 [1]--[7] 是学术写作的占位符，不是 hardcode 或假结果。

### 3.3 没有破坏已有功能

T43 只新增了 `docs/paper_background_related_work_draft.md`、`docs/review/T43_review.md`、`docs/for_human/T43_explanation.md`，并在任务包中追加了 Verification Record。没有修改任何源码、配置、benchmark protocol、`runs/`、`artifacts/` 或治理结论文档。项目当前的功能状态完全不受影响。

### 3.4 没有过度工程

T43 的产出严格限制在任务包要求的范围内。Worker 没有借机修改治理文档、没有额外添加 claim、没有升级证据等级、没有创建新实验计划、没有开始其他章节的正文撰写。正文中的内部标注（如 `[supported claim C3]`、`[stable conclusion 9.1 item 7]`）是草稿阶段的合理辅助工具。

### 3.5 发现的非阻塞问题

我在审查中发现了四个小问题：

1. **N1**: Subsection 6 的第二段从文献综述转向了自我引用（"The present work encounters this boundary directly"）。虽然所有事实内容都正确、blocked claims 被正确标注，但这种自我引用可能被审稿人读成"我们的坦诚本身就是贡献"。T42 review N1 已提出过类似关注。建议在后续正文撰写时观察这一段落是否自然，如果不自然，可以并入 Limitations 章节。

2. **N2**: 引用标记 [1]--[7] 尚未对应具体文献列表。这在有界草稿阶段是可接受的，但建议在下一个正文撰写任务开始前建立共享文献文件。

3. **N3**: 正文中保留了内部草稿标注（如 `[stable conclusion 9.1 item 7]`、`[supported claims C2, C3]`），这些需要在论文组装时统一清理为自然语言。

4. **N4**: 内部 claim reference 的格式不完全统一（"claims" vs "claim" 的单复数），纯排版问题。

这四个问题都不足以阻止 T43 通过，因此最终裁决为 **PASS**。
