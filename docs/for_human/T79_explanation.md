# T79 任务与评审说明

## 1. 先用通俗的话解释这个任务

`T79` 不是写论文正文，也不是补新实验。它做的是一道“闸门判断”：

> 现在仓库里已经整理好的论文材料，够不够支撑下一轮受控的 prose reopen？

这里的关键不是“能不能继续写”，而是“能不能**在不越界**的前提下继续写”。

因为到 `T78` 为止，主线已经有了：

- 锁好的结果表和 figure/caption 路由；
- 能直接落笔的 Results 句胚；
- 渲染 QA 和 callout；
- 同步进 note 的结果层；
- 经校准的非结果层 wording、`statcalib` 降权和版面 warning 收口。

所以 `T79` 要回答的不是“材料还缺不缺”，而是“这些材料是否已经够支撑下一轮只写特定章节的 prose，而不用先补更多基础材料”。

## 2. 这个任务具体做了什么

### 2.1 任务目标

从 [docs/02_experiment_plan.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/02_experiment_plan.md:158)、[docs/04_task_board.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/04_task_board.md:528) 和 [docs/07_handoff.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/07_handoff.md:9) 来看，`T79` 是 `T78` 之后的当前唯一任务，目的非常明确：

- 不直接 reopen manuscript；
- 不顺手扩写 prose；
- 不跑实验；
- 只做一张 docs-only 的 reopen/readiness gate。

它需要给出：

1. 一个唯一 verdict；
2. 一个 section-level readiness matrix；
3. 一个 gap-to-action matrix；
4. 一个唯一的后续任务建议。

这意味着 `T79` 更像“论文材料是否已经可进入下一阶段”的评审，而不是“下一阶段本身”。

### 2.2 这轮实现的核心产物

这次 worker 主要交付了三类东西。

#### A. gate 报告

核心文件是 [paper_reopen_gate_and_prose_readiness_review.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_reopen_gate_and_prose_readiness_review.md:1)。

它包含了四个关键部分：

1. `Gate Verdict`
2. `Strongest Supported Truth`
3. `Section-Level Readiness Matrix`
4. `Single Recommended Next Task`

其中最重要的是 verdict：

- `GO_FOR_BOUNDED_PROSE_REOPEN`

这不是说“全文 ready”，而是说：

- 对下一轮**有界** prose reopen 来说，当前材料已经够用了；
- 但方法章、expanded benchmark、机制闭环、`.tflite` default-env 和 real-board success 这些更强叙事仍然不够。

#### B. gap matrix

第二个关键文件是 [paper_reopen_gap_matrix.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/paper_reopen_gap_matrix.md:1)。

它把当前剩余缺口拆成结构化表格，并回答几个很实际的问题：

- 这个缺口到底是什么；
- 它限制的是下一轮 bounded reopen，还是更强范围的全文 reopen；
- 现有证据在哪里；
- 需要的动作是什么；
- 是否能通过一张后续 bounded task 解决。

这里有一个重要设计：它没有把所有 open risk 都说成“现在不能写”。相反，它把缺口分成两类：

1. 会阻止 full-manuscript 或更强叙事的缺口；
2. 只限制 claim ceiling，但不阻止下一轮 bounded prose 的缺口。

这正是 `T79` 这类 gate 任务应该做的事。

#### C. README 入口同步

worker 还同步了 [docs/paper_materials/README.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/README.md:31)，把 `T79` 的两个新入口写进去：

- `paper_reopen_gate_and_prose_readiness_review.md`
- `paper_reopen_gap_matrix.md`

并且在 [README.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/paper_materials/README.md:86) 明确写出：

- `T79` 只回答“当前材料栈是否足够支持下一轮 bounded prose reopen”；
- 它不是 prose reopen 本身；
- 即使 verdict 是 `GO_FOR_BOUNDED_PROSE_REOPEN`，后续也仍然需要单独的新任务包。

这一步很重要，因为它直接防止了“把 gate 文件误读成已经进入写作阶段”的常见问题。

### 2.3 它为什么能给 `GO`

这个 `GO` 不是空口给的，而是建立在前面几轮材料都已经收口的基础上。

#### 结果层和图表链已经够写

`T74` 到 `T76` 已经把这些关键东西准备好了：

- stable ID 结果表；
- caption 与 insertion route；
- 最终成图资产；
- rendered QA；
- Results-section assembly；
- paragraph-level callout。

这意味着下一轮如果只是写 Results / Discussion / Conclusion 一带，已经不是“材料不够”，而是“如何按现有材料组织 prose”。

#### note 的关键 narrative section 已经校准

`T77` 把结果层同步进 note，`T78` 又把：

- 标题；
- 引言；
- Related Work；
- Discussion；
- Conclusion；
- `statcalib` 的视觉层级；
- LaTeX warning

收紧到了当前证据边界内。

所以，下一轮若只 reopen 已校准的 narrative / result-facing 区域，已经不需要再补一轮前置 note 校准。

#### claim / risk / deployment 护栏已经在

现在 repo 里已经有这些关键护栏：

- `paper_claim_evidence_ledger.md`
- `paper_claim_risk_table.md`
- `paper_results_callout_sheet.md`
- `paper_authoring_do_not_write_list.md`

它们已经明确限制：

- `T24` 不能写成 expanded benchmark；
- `T48` 不能写成 deployment closure；
- `T49/T71/T72` 不能写成 real-board success；
- `T64-T70` 不能写成 mature `statcalib` comparator promotion。

所以 `T79` 的判断是：这些边界虽然还没有被“解决”，但已经足够在下一轮 prose reopen 中充当 guardrail，而不是 blocker。

## 3. 为什么这次 review 结果是 `PASS`

我给 `PASS`，主要基于三点。

### 3.1 任务包要求的四件核心事，worker 都做到了

`T79` 任务包要求：

1. 唯一 verdict；
2. readiness matrix；
3. gap matrix；
4. 唯一后续任务建议。

当前交付里都能直接找到：

- 唯一 verdict：`GO_FOR_BOUNDED_PROSE_REOPEN`
- 14 项最小 area 覆盖的 readiness matrix
- 结构化 gap matrix
- 唯一下一任务：`T80: 主线校准段落的 bounded prose reopen`

这说明不是“写了个状态说明就算交付”，而是真完成了 gate 任务的核心结构。

### 3.2 没有把 gate 写成 prose reopen 本身

这是这类任务里最容易出问题的地方。

如果 worker 在文档里直接写成：

- “已经可以恢复全文扩写”
- “paper ready”
- “可以对外讲 deployment / real-board”

那就会直接越界。

但当前文档没有这么做。相反，它反复强调：

- 只允许下一轮 bounded prose reopen；
- 方法章仍是 `defer_out_of_scope`；
- expanded benchmark、机制闭环、`.tflite` default-env、real-board success 仍被阻塞；
- 后续仍需要一张单独的 `T80` 任务包。

这说明它把 gate 的边界守住了。

### 3.3 没有把计划写成事实

这一点我也特别检查了。

`T79` 的写法没有把：

- future benchmark
- future hardware host
- future deployment closure
- future full-manuscript reopen

说成已经存在的事实。

`paper_reopen_gap_matrix.md` 也把“当前能做的动作”和“当前仍然 blocked 的范围”分开写清楚了，没有拿未来条件倒推当前结论。

因此，从 reviewer 视角，这次不是“大胆乐观”，而是“在现有证据边界内给出了一张严格受限的 GO”。

## 4. 为什么这不是更强或更弱的 verdict

### 4.1 为什么不是更强

它不是 “full-manuscript ready”，也不是 “paper-ready submission”，因为：

- 方法章节还没有做全文级 reopen-ready 校准；
- `T24` 之外没有 expanded benchmark 证据；
- 机制证据仍是 descriptive，不是 causal closure；
- `.tflite` 仍不是 default-env/deployment closure；
- real-board 仍是 `NO_GO` gate/provenance，而不是 execution success。

所以更强 verdict 会直接 overclaim。

### 4.2 为什么也不是更弱

它也不该是 `CONDITIONAL_GO_WITH_PRE_REOPEN_FIXES` 或 `NO_GO_NEED_MORE_MATERIALS`，因为：

- 当前已经不是“材料没同步”；
- 也不是“note 还会直接误导”；
- 结果层写作材料、callout、claim/risk 护栏都已经具备；
- open 风险主要限制的是更高的 claim ceiling，而不是下一轮有界 prose 本身。

换句话说：如果这里还不给 `GO`，主线就会重新掉回“无限期继续整理材料”的状态，和 `T79` 这张 gate 的设计目标相违背。

## 5. Worker 已写的 review / explanation 文档怎么看

### 5.1 对 worker explanation 的看法

worker 原来的 [T79_explanation.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/for_human/T79_explanation.md:1) 总体方向是对的：

- 它正确区分了 gate 与 prose reopen；
- 也正确说明了为什么 `GO` 仍然是有界的。

我这次补充的重点主要是：

- 把它和 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md` 的主线衔接讲得更清楚；
- 明确指出 reviewer 为什么给 `PASS`，而不是只重复 worker 的自我解释。

### 5.2 对 worker review 的看法

worker 预写在 [T79_review.md](/D:/Codes/Quantum/DriftAdaptiveQEC/docs/review/T79_review.md:1) 里的内容更像自评式 reviewer 草稿。方向基本正确，但正式结论不应直接沿用 worker 自判，而应以后续 reviewer 覆盖版本为准。

这不算伪实现，但在治理上必须区分：

- worker 的自检；
- reviewer 的正式 verdict。

## 6. 结论

`T79` 本质上完成了一张“是否可以继续写”的受控闸门任务。

它的价值不在于新增了什么实验，而在于把当前主线状态说清楚了：

- 现在已经足够进入下一轮**有界** prose reopen；
- 但还远没到 full-manuscript ready，更没到 deployment / real-board / expanded benchmark 可以放大的阶段。

所以这次 review 给 `PASS` 是合理的。下一步最合适的是开一张单独的 `T80` bounded prose reopen 任务，而不是直接把 gate 当成写作许可无限放大。
