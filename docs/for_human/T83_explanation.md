# T83 说明

## 1. 这个任务在做什么

`T83` 做的不是“再补一块材料”，而是对当前整份主线 note 做一次全文级的一致性总检查。

如果把前面的任务串起来看：

- `T80` 解决的是 ready sections 的 prose reopen；
- `T81` 解决的是 `Summary of Contributions` 和 methods 的校准；
- `T82` 解决的是 supporting materials 该放主文/附录/补充/blocked 哪一层；

那么 `T83` 要解决的就是更高一层的问题：

现在这整份 note，放在一起时，到底有没有前后打架、层级混淆、把 blocked surface 写成既成事实、或者把 follow-up lane 写成已接受主结果？

所以，`T83` 的本质是：

做一次全文 consistency sweep，并据此给出一个明确的 closeout gate。

## 2. 这次实现到底做了什么

### 2.1 任务目标

从 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md` 和 `docs/08_risks_and_open_questions.md` 可以看出，当前主线任务已经不是“补哪个局部 section”，而是“当前全文是否已经对齐到 strongest supported truth，以及后续是否只应进入 bounded final polish”。

`T83` 因此被定义成一张更强的 docs-only 主线任务，它要同时做：

- 全文 section-by-section consistency sweep；
- 必要但受控的 wording 收口；
- 一份 section-to-evidence crosswalk；
- 一份唯一的 closeout gate / blocker register。

注意，这仍然不是 full-manuscript closeout，更不是 submission-ready pack。

### 2.2 实际 sweep 覆盖了哪些 section

本轮 crosswalk 覆盖了当前 note 的核心 section：

- `Title`
- `Abstract`
- `Summary of Contributions`
- `Introduction`
- `Relationship to Existing Work`
- `Brief Review of the GKP Code`
- `Noise and Drift Model`
- `Model Architecture`
- `Experimental Setup`
- `Numerical Results`
- `Discussion`
- `Conclusion`

此外，`T83` 还把原来结果层里残留的一块 follow-up/sidecar 区，显式收口成：

- `Bounded follow-up lanes outside the accepted result layer`

这一步很关键，因为它把“未来可能开的 lane”从“当前结果叙事的一部分”重新降回成“边界登记区”。

### 2.3 本轮真正改了哪些正文位置

虽然 `T83` 做的是全文 sweep，但实际正文改动只集中在 4 处，并都加上了 `% T83-CLOSEOUT: ...` 注释：

1. `Numerical Results`
2. `Bounded follow-up lanes outside the accepted result layer`
3. `Discussion`
4. `Conclusion`

这说明 `T83` 不是重新大修整篇 note，而是先做全文审计，再只对发现仍有 wording drift 或层级混淆的地方做最小修正。

### 2.4 两个核心新文档分别做了什么

#### A. `paper_fullnote_consistency_crosswalk.md`

这份文档是全文一致性的“总索引”。

它逐 section 记录：

- 当前最强可支持事实是什么；
- 主要 evidence anchors 是什么；
- 哪些 retelling 明确禁止；
- 如果未来还要推进，允许的最小后续动作是什么。

它的作用不是替 note 说话，而是给 Reviewer / Captain / 作者一个统一的“全文边界导航图”。

#### B. `paper_closeout_gate_and_blocker_register.md`

这份文档是 `T83` 最关键的 gate 产物。

它给出了唯一 verdict：

- `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`

并把 blocker 分成两层：

1. 当前 manuscript route 内部还剩哪些 final-polish 类问题
   - 比如内部 provenance/task 术语还偏作者内部口径；
   - 结果层、appendix、supplement 还可做结构压缩；
   - 这些问题不需要新实验，只需要作者向 final polish。

2. 明确仍然 blocked、并且不属于当前 route 的 surface
   - real-board execution / board timing / resource
   - default-env `.tflite` portability
   - full training reproducibility
   - paper-grade expanded benchmark

这层区分非常重要，因为它防止后续有人把“可以做 final polish”误听成“所有 blocked surface 都已经解锁”。

### 2.5 note 和 README 的变化意味着什么

这次除了 note 局部收口，还更新了：

- `docs/paper_notes/README.md`
- `docs/paper_materials/README.md`

它们新增了 `T83` 的入口，明确说：

- `paper_fullnote_consistency_crosswalk.md`
- `paper_closeout_gate_and_blocker_register.md`
- `% T83-CLOSEOUT: ...`

只用于回答“全文是否已经对齐到当前 strongest supported truth，以及后续是否只能进入 bounded final polish”。

这条链路不等于：

- submission-ready pack
- deployment closure
- real-board success
- full-manuscript closeout 已完成

### 2.6 对后续开发的意义

`T83` 的意义，不是新添实验，而是把“当前这整份 mainline note 到底还能不能继续往前走，以及能走到哪一步”这件事变成了可审计事实。

它的价值主要体现在三点：

- 全文口径终于有了统一 crosswalk，而不是靠分散的 `T80/T81/T82` 局部产物拼起来理解。
- Captain 现在可以基于显式 gate，而不是基于印象，决定下一步是否只开 bounded final polish。
- blocked surface 被单独登记出来，减少后续在 final polish 阶段被“顺手写高”的风险。

## 3. 为什么这次 review 给出 PASS

我给 `PASS`，因为这次交付完成了任务包要求的关键工作，而且没有把边界写坏。

### 3.1 任务确实完成了

`T83` 要求的主要产物都已经在：

- `paper_fullnote_consistency_crosswalk.md` 已生成；
- `paper_closeout_gate_and_blocker_register.md` 已生成；
- note 中实际修改的 section 都有 `% T83-CLOSEOUT: ...` 注释；
- `T80/T81/T82` 的旧标记仍然保留；
- README 入口已登记；
- note 已重新编译，`.log` 关键字扫描没有发现明显 warning。

### 3.2 没有伪实现、mock、stub、hardcode

这轮不是代码功能任务，所以关键不在“代码是不是假实现”，而在“文档有没有把 blocked 事实写成完成态”。

我没有看到以下问题：

- 没有把 `FR8/statcalib` 写成 promoted comparator；
- 没有把 training/material 写成 full reproducibility；
- 没有把 `.tflite` 写成 default-env / deployment closure；
- 没有把 real-board 写成 execution success；
- 没有把无 `Linux + FPGA` host 的 blocked surface 偷偷改写成“其实只差最后一步”。

### 3.3 为什么 `GO_FOR_BOUNDED_FINAL_POLISH_ONLY` 是可以接受的

这轮最关键的判断不是“论文是不是已经完全 ready”，而是：

当前剩下的问题，还属于不属于“final polish”范畴。

从 `paper_closeout_gate_and_blocker_register.md` 看，当前 route 内剩下的主要是：

- reader-facing terminology translation；
- Results / appendix / supplement 的结构压缩；
- 内部 route 语言向读者语言的翻译。

而那些真正需要新证据的 surface，例如：

- real-board execution
- `.tflite` portability
- full training reproducibility
- expanded benchmark

都仍被单独列成 route 外的 blocked 项。

这说明 gate 的 “GO” 是受限且诚实的，不是泛化的 “都 ready 了”。

### 3.4 为什么不是 PASS_WITH_WARNINGS 或 BLOCK

我保留了几条非阻断提醒：

- 当前 worktree 本来就有一批与 `T83` 无关的额外 diff，所以 review 必须用 allowlist-scoped diff；
- `paper_materials/README.md` 里仍有一个轻微标题不一致：章节名还是 `T74-T82`，但正文已加入 `T83`；
- compile 结论依赖当前主机可用的 `TeX Live 2024 + latexmk`，bundled `tectonic` 仍没完全恢复。

但这些都不影响 `T83` 的核心交付真实性，所以不足以升级为 `BLOCK`，也没有严重到必须压成 `PASS_WITH_WARNINGS`。

## 4. 对已有 worker 文档的看法和补充

仓库里已有的 `docs/review/T83_review.md` 和 `docs/for_human/T83_explanation.md` 草稿，方向基本是对的，已经抓住了核心：

- `T83` 比 `T82` 更强，因为它是全文 sweep；
- `T83` 的 gate 结论是 `GO_FOR_BOUNDED_FINAL_POLISH_ONLY`；
- 这仍不等于 full-manuscript closeout。

我这次补充的重点主要有三类：

- 把 “为什么这个 GO 仍然是保守 GO” 讲清楚。
  - 不是因为 blocked surface 消失了，而是因为剩余 route 内问题已经降到了 final-polish 范围。

- 把当前 worktree 的额外脏 diff 单独指出。
  - 避免后续有人误把 00~08 治理变更或 `T82` 未提交产物算成 `T83` 本身的越界。

- 补了一个轻微文档一致性提醒。
  - `paper_materials/README.md` 的章节标题仍写 `T74-T82`，应在后续顺手修掉。

## 5. 一句话结论

`T83` 的价值，不是宣布论文已经彻底收口，而是把“当前全文已经自洽到什么程度、还剩哪些只属于 final polish 的问题、哪些仍然明确 blocked”说成了可审计事实；它完成得合格，所以 review 结论是 `PASS`。
