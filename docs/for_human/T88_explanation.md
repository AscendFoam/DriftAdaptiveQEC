# T88 说明

## 1. 通俗解释：这轮任务到底在做什么

`T88` 不是去做新实验，也不是把论文直接推到“已经可以投稿”的完成态。

它做的是一件更窄、但很关键的事：把 `T87` 已经批准的那批“只允许人工做的小修小补”真正落地，然后把当前主线论文材料冻结下来，避免后面再一边手改一边把边界改丢。

可以把它理解成：

- `T87` 回答的是：“现在能不能进入有界手工收口阶段？”
- `T88` 回答的是：“既然允许进入，那这批手工收口到底做了什么，哪些 surface 现在固定了，哪些 blocked surface 仍然不能越界？”

所以，`T88` 是一次 `manual finish + surface freeze` 收口，不是 `submission-ready completed`。

## 2. 详细解释：任务目标、流程、变更和意义

### 2.1 任务在主线中的位置

从 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md` 可以看出，仓库主线在 `T74-T87` 这一串任务里，已经逐步完成了：

- 结果表、图、caption、traceability 材料整理；
- note 的 results 层同步和非结果层校准；
- full-note consistency sweep；
- final polish；
- submission-readiness preflight；
- submission-facing assembly / exclusion；
- author-final QA。

到 `T87` 为止，主线得到的不是“投稿包已完成”，而是一个更窄的结论：

`GO_FOR_BOUNDED_AUTHOR_MANUAL_FINISH_ONLY`

这意味着只能继续做被批准的人工终修动作，不能扩大为：

- 新 prose reopen；
- 新 benchmark；
- `.tflite` portability closure；
- real-board success retelling；
- `statcalib` promotion；
- submission-ready completed。

`T88` 的任务，就是把这个“只允许有限人工终修”的许可真正执行完，并冻结为单一可审计答案。

### 2.2 这轮实际改了什么

这轮没有改源码、配置、测试、`runs/`、`artifacts/`，只改了 paper note 和 paper materials。

核心新增了 5 份文档：

- `paper_manual_finish_execution_log.md`
  - 逐条记录 `MF01-MF05` 到底执行了什么。
- `paper_mainline_surface_freeze_manifest.md`
  - 冻结当前 main text / appendix / supplement 的 surface 选择。
- `paper_author_edit_decision_register.md`
  - 记录真实的编辑决策，而不是抽象原则。
- `paper_blocked_surface_disclaimer_table.md`
  - 明确哪些 blocked surface 的免责声明必须继续保留。
- `paper_frozen_mainline_handoff_gate.md`
  - 给出本轮唯一 handoff verdict。

同时，`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 做了最小必要的人工收口，主要集中在：

- `Numerical Results`
- `Mechanism probe for residual-b behavior`
- `Discussion`
- `Conclusion`

并且给这些真实触碰的 section 加上了：

- `% T88-MANUAL: Numerical Results`
- `% T88-MANUAL: Mechanism probe for residual-b behavior`
- `% T88-MANUAL: Discussion`
- `% T88-MANUAL: Conclusion`

这使得后续审查可以直接把“文档台账中的动作”回链到 note 里的具体修改位置。

### 2.3 这轮做出的关键编辑决策

从内容上看，`T88` 真正冻结了几件事：

1. 主结果在当前 note 中以 `T74-TBL-01 / Table~\ref{tab:five-mode-benchmark}` 作为主呈现。
2. 机制层仍然只保留保守、描述性的解释，不升级为因果闭环。
3. appendix 和 supplement 的职责被再次收紧并固定：
   - appendix 承接 ablation、training/material provenance、isolated true runtime；
   - supplement 承接 `statcalib` extension lane、real-board gate/provenance、exclusion notes。
4. `MF04` 没有强行在当前 note 里再造一套 boundary schematic caption，而是明确 `left_as_is`，继续沿用 `T74/T75` 已锁定的外部 caption/placement 文案。
5. 结尾不再写成“后面还要继续 assembly/manual finish”，而改成“当前主线已冻结，可 handoff，但仍不是 completed submission state”。

### 2.4 对后续开发/写作的意义

这轮的价值，不是新增证据，而是防止证据叙事继续漂移。

它为后续带来的正面意义是：

- 后续人类作者或 Captain 再看主线材料时，知道现在固定采用哪种 surface 组织方式；
- blocked surface 不能再被“顺手润色”成更强 claim；
- 当前 note/material 可以围绕 frozen-mainline 继续维护，而不必重新解释哪些内容还在 appendix、哪些还在 supplement、哪些仍然必须 blocked。

它没有带来的东西也必须说清楚：

- 没有新增 benchmark 证据；
- 没有新增训练复现证据；
- 没有关闭 default-env / cross-host `.tflite` portability；
- 没有让 real-board 从 `NO_GO` 变成成功；
- 没有让 `FR8/statcalib` 变成 mature comparator；
- 没有让主线进入 submission-ready completed。

## 3. 为什么我给出 `PASS`

我给 `PASS` 的原因很直接：`T88` 任务包要求的交付物和边界，基本都被真实满足了。

### 3.1 任务确实完成了

我核对后确认：

- 5 份新增台账/hand off 文档都已存在；
- `MF01-MF05` 全部在执行日志里有对应状态；
- freeze manifest 覆盖了任务包要求的主文、appendix、supplement 和 blocked surface；
- decision register 记录了真实编辑决策，不是空泛口号；
- README 已完成 `T88` 入口登记；
- note 中保留了 `% T80` 到 `% T87` 标记，并新增 `% T88-MANUAL` 标记；
- handoff gate 只有一个 verdict，且是允许值之一：
  - `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`

### 3.2 没发现伪实现、mock、stub、硬编码冒充完成

这轮是 docs-only 任务，本身没有代码逻辑可 fake。

我重点检查的是“有没有只写台账、但 note 本体其实没落地”，结论是没有：

- note 里确实有对应 section 的实改；
- 这些实改能回链到 execution log / freeze manifest / decision register；
- `MF04` 虽然没有执行 caption 重写，但它是显式 `left_as_is`，并且给了边界理由，所以是诚实的不执行，不是伪完成。

### 3.3 验证足够，且没有乱跑长实验

我只做了轻量验证，没有重跑任何长实验：

- allowlist 范围核对；
- `git diff --check`；
- marker 保留检查；
- red-flag 词扫描；
- gate/verdict 唯一性检查；
- README 登记检查；
- `.log` 关键字扫描。

这些验证足以支撑一个 docs-only / manual-finish / surface-freeze 任务的通过。

### 3.4 没有把计划写成事实

这是我最重点看的点之一。

结论是：当前文档虽然用了很多像 `submission-ready completion`、`hardware-ready finalization`、`deployment closure` 这样的词，但它们仍然只出现在：

- negative guardrail；
- blocked disclaimer；
- exclusion/hand off/gate 语境。

也就是说，这些词是在说“不能这么写”，不是在说“已经完成了”。

这也是为什么我没有给 `BLOCK`。

## 4. 我对现有 review / explanation 的看法和补充

Worker 已经写了 `docs/review/T88_review.md` 和 `docs/for_human/T88_explanation.md`。我检查后认为，它们的大方向结论基本正确，没有明显把 `T88` 写成越界完成态。

我这次补充的重点有三点：

1. 明确指出当前 worktree 仍是 dirty，`T88` 的判断必须继续依赖 allowlist + marker + 当前文件内容，不能机械看 whole-file diff。
2. 明确解释 `MF04 = left_as_is` 为什么是可接受的边界内决策，而不是漏做。
3. 明确强调 `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY` 只代表 frozen handoff，不代表 submission-ready completed。

## 5. 总结

`T88` 的真实作用，是把 `T87` 允许的最后一小批人工终修动作落地，并把当前主线论文材料冻结成一个不容易继续漂移的写法。

因此我给 `PASS`，但这个 `PASS` 只能被解读为：

- 主线材料已经完成一次有界 manual finish；
- 当前写作 surface 可以 frozen handoff；
- blocked surface 仍然 blocked。

不能把它解读成：

- 投稿包已完成；
- 证据等级提升；
- 部署边界关闭；
- 硬件路径打通。
