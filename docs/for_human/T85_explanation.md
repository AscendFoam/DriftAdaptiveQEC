# T85 任务与 Review 说明

## 1. 先用通俗的话解释这个任务

`T85` 不是补实验，也不是把论文直接做成投稿包。

它更像一次“投稿前诚实度预检”：

- 先检查当前主线 note 里，还有没有把“已经做完的工作”写成“后面再做”的旧句子；
- 再判断：在不升级任何证据等级的前提下，当前这套主线材料是否已经足够干净，可以继续开下一张 submission-facing assembly 任务；
- 同时把当前仍然不能写强、必须继续排除的 surface 单独列出来。

所以 `T85` 的目标不是宣布“现在就能投稿”，而是回答：

现在能不能诚实地进入下一张“受边界约束的投稿装配任务”。

## 2. 这个任务具体做了什么

结合 `docs/02_experiment_plan.md`、`docs/04_task_board.md` 和 `docs/07_handoff.md`，`T85` 是在 `T84` 之后被打开的当前唯一任务。

`T84` 已经做完的事情是：

- 主线 note 的 bounded reader-facing final polish；
- 内部术语到读者表述的第一轮受控翻译；
- main text / appendix / supplement / blocked 的读者化装配整理。

但 `T84` 的 review 还留下了一个真实 warning：

- `Conclusion` 里仍有一句旧口径，把已经完成的 reader-facing polish 写成“后续还要做的工作”。

因此 `T85` 的任务就变成三层：

1. 清掉 residual wording-lag。
2. 做 submission-readiness preflight gate。
3. 明确下一步若继续推进，哪些内容仍然只能作为 blocker / exclusion 保留。

本轮实际改动主要落在这些文件：

- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_notes/README.md`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_submission_readiness_preflight_gate.md`
- `docs/paper_materials/paper_submission_blocker_matrix.md`
- `docs/paper_materials/paper_residual_state_lag_sweep.md`

其中正文只最小修改了两个 section：

1. `Discussion`
2. `Conclusion`

并新增了对应的 `% T85-PREFLIGHT: ...` 注释。

### 2.1 note 里实际改了什么

这轮不是重写整篇 note，而是只做了 residual wording-lag 清扫。

最关键的变化有两处：

- `Discussion` 不再把 manuscript-side 的剩余工作写成“还要继续 reader-facing condensation / route cleanup”，而是改成：后续若继续推进，剩下的是在现有分层基础上做 bounded submission-facing assembly。
- `Conclusion` 不再把 `T84` 已经完成的 reader-facing polish 写成未来待办，而是明确：
  - 当前主线已经把层级翻译清楚；
  - 下一步若继续推进，只是围绕这套既有层级做 bounded submission-facing assembly；
  - 这仍然不等于 deployment closure、submission-ready completion 或 hardware-ready finalization。

### 2.2 三份新文档分别解决什么问题

#### `paper_residual_state_lag_sweep.md`

这份文档专门记录：

- 哪些位置还有状态滞后句；
- 为什么它们是滞后；
- 本轮如何处理；
- 边界是否保持不变。

它的作用是防止以后再次把“上一轮已完成的润色”写回成“下一轮待做”。

#### `paper_submission_blocker_matrix.md`

这份矩阵回答的是：

如果下一步真的开 submission-facing assembly，哪些 surface 仍然不能被装进去，或者只能被明确排除。

它列出的 blocker 包括：

- 板级 execution / timing / resource
- default-env `.tflite` / deployment portability
- full training reproducibility
- `statcalib` promoted comparator retelling
- expanded benchmark story
- submission-facing pack 本身尚未真正装配

这份矩阵的意义不在“说项目不行”，而在“防止下一轮把不该升格的内容顺手写强”。

#### `paper_submission_readiness_preflight_gate.md`

这是本轮最关键的输出。

它给出的唯一 verdict 是：

- `GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY`

这个 `GO` 的正确含义非常窄：

- 当前主线材料已经足够诚实，可以允许再开一张“受边界约束的投稿装配任务”；
- 不是说 submission-ready pack 已完成；
- 不是说所有 blocker 都消失了；
- 也不是说任何 evidence surface 被升级了。

## 3. 这个实现对后续开发意味着什么

从项目路线看，`T85` 的意义在于把主线从“读者化润色阶段”推进到“是否值得开投稿装配任务的 gate 阶段”。

也就是说，当前主线不再主要缺：

- 全文一致性；
- ready section prose；
- methods / contribution 校准；
- supporting-material 四层收口；
- reader-facing translation / assembly。

现在主线缺的是更窄的一件事：

- 在不扩证据、不重开 benchmark、不碰 `.tflite` portability、不碰 real-board execution 的前提下，判断当前材料是否已经干净到可以进入 submission-facing assembly。

`T85` 给出的答案是：可以，但只能以**有边界的下一张装配任务**形式继续，而不是直接宣称“已经 ready to submit”。

这对后续开发的直接意义是：

- 如果 Captain 继续推进，下一张任务应该是 docs-only、mainline-only、assembly-only；
- 该任务只能整理现有材料、同步 claim/evidence/risk 三本账、压缩 submission-facing exclusion 说明；
- 不能借这个 `GO` 去补写实验、补开 benchmark、补写 portability、补写 real-board 成功，或把 `statcalib` 晋升成主结果 comparator。

## 4. 为什么我的 review 结果是 PASS

我给 `PASS`，是因为 `T85` 的任务目标已经被真实完成，而且没有越界。

主要依据有这些：

1. 任务要求的 3 份核心文档都已经创建：
   - `paper_submission_readiness_preflight_gate.md`
   - `paper_submission_blocker_matrix.md`
   - `paper_residual_state_lag_sweep.md`
2. note 中 `T84_review` 指出的残余状态滞后句已经被处理掉。
3. note 中保留了旧标记链：
   - `% T80-REOPEN`
   - `% T81-CALIBRATION`
   - `% T82-SUPPORT`
   - `% T83-CLOSEOUT`
   - `% T84-POLISH`
4. 本轮新改动的 section 也有 `% T85-PREFLIGHT: ...` 标记。
5. 两个 README 都已经登记 `T85` 入口。
6. 本地 `latexmk -g` 编译成功，日志中也没有扫出常见 warning 关键字。
7. 最重要的是：`GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY` 被写成了一个**受边界约束的下一步许可**，而不是“当前仓库已经具备 submission-ready pack”。

## 5. 为什么我没有给 PASS_WITH_WARNINGS 或 BLOCK

我没有给 `BLOCK`，因为没有发现以下问题：

- 没有伪实现、mock、stub、hardcode；
- 没有把 blocked surface 写强；
- 没有把计划写成已经完成的事实；
- 没有越出 allowlist 去改治理文档、源码、实验产物或历史事实。

我也没有给 `PASS_WITH_WARNINGS`，因为 `T84` 留下的那条真实 warning 正是 `T85` 这轮要解决的核心内容，而它确实已经被解决了。

本轮仍有一些过程性提醒，但还不构成降级 verdict 的理由：

- 当前 worktree 本身较脏，导致直接对 `HEAD` 的 diff 会混入一部分前序 `T84` 文本变化；
- 因此这轮审查需要继续靠 allowlist-scoped 核对、当前文件内容、标记链和新台账来确认范围；
- 编译成功也仍然只是当前主机事实，不是普适环境事实。

这些都值得写进 review 作为提醒，但不影响 `T85` 本身完成。

## 6. Worker 已写的 review / explanation 是否有问题

整体上，Worker 已写的 `T85_review.md` 和 `T85_explanation.md` 方向是对的。

我和 Worker 的核心判断一致：

- `T85` 不是 submission-ready completion；
- 它只是一次更强的 preflight gate；
- 即使给了 `GO_FOR_BOUNDED_SUBMISSION_PACK_ASSEMBLY`，也仍然必须保留大量 blocker / exclusion。

我这里主要补充了两点：

1. 更明确地点出当前 diff 边界为什么需要谨慎解释。
原因是工作区里仍混有前序 `T84` 的未提交内容，所以单纯看 `git diff` 相对 `HEAD` 的结果，会把一部分 `T84` 文本变化和 `T85` 混在一起。

2. 更明确地区分 “GO for next bounded task” 和 “ready to submit now”。
这是这轮最容易被读歪的地方，所以需要在解释文档里单独强调。

## 7. 推荐的下一步

如果 Captain 接受 `T85`，合理的下一步只应该是：

- 新开一张 docs-only、mainline-only、assembly-only 的 bounded submission-pack assembly 任务。

这张下一任务应当做的是：

- 组织已有 main text / appendix / supplement / exclusion 材料；
- 保持 claim/evidence/risk 三本账同步；
- 明确 submission-facing 的排除项说明。

它不应做的是：

- 重开 benchmark；
- 升级 `.tflite` portability；
- 升级 real-board execution；
- 把 `FR8/statcalib` 写成成熟主线 comparator；
- 把当前主线直接回述成 submission-ready pack 已完成。
