# T78 任务与评审说明

## 1. 先用通俗的话解释这个任务

`T78` 不是做新实验，也不是写新结论，而是在给现有论文 note 做一次“收口”。

更具体地说，`T77` 已经把结果表、结果段和图表 trace 链同步进 note 了，但还留着四个尾巴：

1. note 里一些非结果层文字还没有按当前证据边界重新校准；
2. `statcalib` 虽然文字已经写得很保守，但在版面结构上还是显得太像主结果；
3. note 能编译，但日志里还有不少 `Underfull \hbox` 这类排版 warning；
4. 缺一份更机械、便于 reviewer 复核的“这轮到底改了哪些 section”说明。

所以 `T78` 的本质是：让这份 note 更诚实、更清楚、更不容易让人误解，而不是让项目突然多出新的实验成果。

## 2. 这个任务具体做了什么

### 2.1 任务目标

从 `docs/04_task_board.md` 和 `docs/07_handoff.md` 看，`T78` 是 `T77` 之后的当前唯一任务，定位非常明确：

- 只做 docs-only 的 note 校准；
- 只做非结果层 wording、`statcalib` 层级降权和排版 warning 收口；
- 不碰源码、测试、`runs/`、`artifacts/`、治理文档；
- 不把这轮工作写成 benchmark、`.tflite`、real-board 或 mature `statcalib comparator` 的升级。

这和 `docs/02_experiment_plan.md` 当前的主线也一致：项目在论文材料优先阶段，先把结果材料、边界表述和 note 质量收紧，再决定是否进入下一张 paper reopen gate。

### 2.2 实现流程

这轮改动大致分成四块。

#### A. 校准标题和非结果层表述

在 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 里，worker 调整了：

- `Title`
- `Abstract`
- `Introduction`
- `Summary of Contributions`
- `Relationship to Existing Work`
- `Discussion`
- `Conclusion`

其中最明显的一步是标题从带有“teacher + statistical calibration 并列主线”暗示的写法，改成更聚焦主线的 `Teacher-Anchored Residual Calibration` 表述，见 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:11`。

同时，在引言和 related-work 一带加入了更明确的 evidence hierarchy 提示。例如：

- 主结果层仍然是锁定的 software-HIL benchmark；
- deployment-facing 材料仍只是分层边界；
- `statcalib` 仍是 supplement-side extension lane，而不是被提升后的主线 comparator。

这些调整的目的，不是重写论文，而是把 note 的“读者第一印象”压回当前治理文档允许的证据边界。

#### B. 把 `statcalib` 从版面结构上降权

这是 `T78` 最关键的改动之一。

在 `T77` 之后，`statcalib` 的文字边界已经比较保守，但它在 `Numerical Results` 里还是以三个与主结果段几乎同等级的小节出现，视觉上容易让人把它读成并列主结果。

这轮做了两件事：

1. 在三段 `statcalib` 说明前加了一条 bridge 句，明确后面只是 supplement-side extension-lane 记录；
2. 把三个小节从 `\subsection` 全部降成 `\subsubsection`。

对应位置在 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex:735` 开始。

这里很重要的一点是：worker 没有改这些表格里的数字，也没有发明新的结果，只是调整读者阅读顺序和结构强度。这正好符合任务包“降权但不升格证据”的要求。

#### C. 收掉可修的 LaTeX warning

`T77` 留下的排版问题主要是 `Underfull \hbox`。`T78` 里实际采取了两类最小修复：

1. 把 `Metric-level advantages` 表的列格式改成 `raggedright`，减少窄列强制两端对齐；
2. 把 `Discussion` 里一条 real-board boundary 长句拆得更容易断行。

结果是：

- `HEAD` 基线：`Underfull \hbox = 32`
- 当前工作树：`Underfull \hbox = 0`

这说明它不是“说自己修了 warning”，而是实际把 warning 数量压下来了。

#### D. 增加可复核的 closeout 文档和入口说明

除了 `.tex` 本体，这轮还同步了几类文档：

- `docs/paper_materials/paper_note_alignment_and_layout_closeout.md`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_note_results_sync_manifest.md`
- `docs/paper_materials/paper_results_section_assembly_pack.md`
- `docs/paper_notes/README.md`

其中最关键的是 `paper_note_alignment_and_layout_closeout.md`。它把下面几件事放到了一处：

- 这轮到底改了哪些 section；
- 每个 section 的改动目标是什么；
- 哪些 section 明确没改；
- `statcalib` 的层级如何 before/after 变化；
- warning 的 before/after 是什么；
- `% T78-SCOPE` 注释覆盖了哪些 section。

这就把 `T77` 时“主要依赖 manifest 和人工解释”的 section-scope 证明，再往前推进了一步。

### 2.3 这轮没有改什么

这一点同样重要。

`T78` 没有改：

- `cnn_fpga/`、`physics/`、`benchmark/`、`tests/`
- `runs/`
- `artifacts/`
- `docs/00_*` 到 `docs/08_*` 治理文档
- 任何 benchmark、训练、`.tflite`、real-board 执行结果

也就是说，这轮的价值不是“多做了什么实验”，而是“把现有论文材料的表达和边界整理得更可靠”。

## 3. 这些变化对后续开发/写作有什么意义

这轮工作的意义主要有三层。

### 3.1 它让 note 更接近“可继续写”，但还没到“直接恢复全文扩写”

`T78` 解决的是 paper-facing 质量问题，不是证据等级问题。

它做完以后，可以更有把握地说：

- 当前 note 的非结果层表述，已经更接近现有证据栈；
- `statcalib` 不再那么容易被误看成主线成熟 comparator；
- note 的编译 warning 已经明显收口；
- reviewer 现在有一份更机械的 section-scope closeout 文档可查。

但它并不等于：

- full-manuscript reopen 已自动获准；
- 论文主文已经可以无条件扩写；
- `.tflite`、real-board、`statcalib` 的证据边界已经升级。

### 3.2 它是在保护 `T24` / `T48` / `T49/T71/T72` / `T64-T70` 的边界不被写作冲掉

这个仓库的硬规则一直是：写作不能反过来篡改证据层级。

`T78` 的一个核心意义，就是把这些常见误读点继续钉住：

- `T24` 仍是历史主锚点，不是 expanded benchmark；
- `T48` 仍只是 isolated current-host true `.tflite` runtime；
- `T49/T71/T72` 仍不是 real-board execution success；
- `T64-T70` 仍只是 `statcalib` extension lane，不是 mature comparator promotion。

这对后续任何论文写作任务都很重要，因为一旦 note 的结构和措辞失控，很容易把这些边界 silently 写没。

### 3.3 它为下一张 gate 类任务做准备

从治理文档看，`T78` 之后更合理的下一步不是“继续顺手写正文”，而是一张很窄的 gate：

- 判断当前 note、results pack、claim/evidence ledger、risk table 是否已经足以支撑下一轮 prose 扩展；
- 如果还不够，就继续开补缺任务，而不是提前恢复 full-manuscript。

所以 `T78` 更像一张“材料质量闸前的最后收口包”，而不是论文工作流的终点。

## 4. 为什么这次 review 给的是 `PASS`

我给 `PASS`，原因很直接：`T78` 的目标本来就不是做大，而是把 `T77` 留下的四类 warning 收掉；从实际 diff 看，这四类问题都得到了对应处理。

### 4.1 `T77` 遗留的四类问题，这次都有直接对应

`T77` 留下的是：

1. 非结果层未校准；
2. `statcalib` 视觉层级偏高；
3. `Underfull \hbox` 还很多；
4. 缺少更机械的 section-scope 审计。

`T78` 的对应收口分别是：

1. `Title`、`Abstract`、`Introduction`、`Summary of Contributions`、`Relationship to Existing Work`、`Discussion`、`Conclusion` 做了受控校准，并加了 `% T78-SCOPE`；
2. `Numerical Results` 里的三段 `statcalib` 从 `\subsection` 降为 `\subsubsection`，且加了 bridge 句；
3. `Underfull \hbox` 从 32 降到 0；
4. 新增 `paper_note_alignment_and_layout_closeout.md`，明确写出改动 section、未校准 section 和 warning before/after。

这意味着任务包要求的收口动作，不是停留在 summary 里，而是都能在实际文件里找到。

### 4.2 没看到伪实现、mock/stub、hardcode 伪装成交付

这轮是 docs-only 任务，所以这里最常见的问题其实不是代码 bug，而是“把计划写成事实”。

我没有看到这种情况。相反，相关文档一直在重复强调：

- `T78` 只是 note 质量收口；
- 不等于 full-manuscript reopen；
- 不升级任何实验或部署证据；
- 未校准 section 仍然明确列出。

这说明 worker 这次的写法是收紧边界，而不是偷升边界。

### 4.3 验证强度对 docs-only 任务来说已经够了

这类任务不需要代码测试矩阵，但需要三类证据：

1. diff 范围证据；
2. note 内容证据；
3. 编译/版面证据。

当前都具备：

- `git diff` 范围没有越界到源码、`runs/`、`artifacts/`、治理文档；
- `.tex` 改动点确实落在任务包允许 section；
- `statcalib` 降权和 `T78-SCOPE` 注释都能直接看到；
- `Underfull \hbox` 计数真实下降到 0。

所以从 reviewer 角度，这已经足以支撑 `PASS`。

## 5. Worker 已写的 review / explanation 文档怎么看

### 5.1 对 worker explanation 的看法

worker 已写的 `docs/for_human/T78_explanation.md` 总体方向是对的：

- 对任务目标的理解基本准确；
- 没有把这轮工作写成新实验；
- 对 `statcalib` 降权、warning 收口、closeout 文档的描述也基本属实。

我这次补充的重点主要是 reviewer 视角下的两点：

1. 把它和 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md` 的主线关系说得更清楚；
2. 更明确地区分“这次为什么能给 `PASS`”与“这次没有做什么”。

### 5.2 对 worker review 的看法

worker 已写的 `docs/review/T78_review.md` 不是正式 adversarial review，而是一个自检占位件。它本身也写明了 `WORKER_SELF_CHECK_PENDING_EXTERNAL_REVIEW`。

这不算伪实现，但也不能直接拿它当最终 reviewer verdict。正式结论应当以后续 reviewer 覆盖版本为准，也就是本次写入的 `docs/review/T78_review.md`。

## 6. 结论

`T78` 完成的是一张很典型的“论文材料边界收口任务”：

- 不做新实验；
- 不改代码；
- 不升级证据；
- 只把现有 note 的表达、结构层级和版面质量收紧。

从实际 diff 和编译产物看，它确实完成了这件事，所以这次 review 给 `PASS` 是合理的。真正的下一步不应自动变成“继续大写正文”，而应先由 Captain 决定是否开启下一张受控的 paper reopen gate。
