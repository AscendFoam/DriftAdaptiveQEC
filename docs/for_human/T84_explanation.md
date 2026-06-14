# T84 任务与 Review 说明

## 1. 先用通俗的话解释这个任务

`T84` 不是补实验，也不是补代码。

它做的是一件更像“论文写作整理”的事：在不增加任何新结论的前提下，把已经被接受的主线事实，翻译成更像正式读者稿的表达方式。可以把它理解成：

- 不改“证据等级”；
- 只改“怎么说更像给读者看”；
- 同时把哪些内容该进主文、哪些该进附录、哪些只能留在 supplement、哪些目前仍 blocked 这件事讲清楚。

所以，`T84` 的目标不是把论文直接做成投稿终包，而是把主线 note 从“内部 closeout 版本”往“读者可读版本”再推进一小步。

## 2. 这个任务具体做了什么

结合 `docs/04_task_board.md` 与 `docs/07_handoff.md`，`T84` 是在 `T83` 通过后被明确打开的当前唯一任务。`T83` 已经完成的是：

- 全 note 的一致性 sweep；
- 明确唯一 gate：`GO_FOR_BOUNDED_FINAL_POLISH_ONLY`。

这意味着仓库当时已经不缺“是否自洽”的判断，而是缺一轮有边界的读者化润色与装配。因此 `T84` 的工作重点是三类：

1. 对主线 note 的 6 个 section 做 reader-facing polish。
2. 产出 3 份配套台账，明确“哪些内部术语怎么翻”“哪些材料该放在哪一层”。
3. 更新两个 README，让后续作者或 agent 能快速找到这轮入口。

本轮实际改动主要在这些文件：

- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_notes/README.md`
- `docs/paper_materials/README.md`
- `docs/paper_materials/paper_bounded_final_polish_change_map.md`
- `docs/paper_materials/paper_reader_facing_term_translation_table.md`
- `docs/paper_materials/paper_appendix_supplement_reader_assembly_map.md`

其中最关键的是 note 源文件。Worker 实际修改了 6 个 section，并加上了 `% T84-POLISH: ...` 注释：

1. `Summary of Contributions`
2. `Experimental Setup`
3. `Numerical Results`
4. `Follow-up routes that remain outside the accepted result layer`
5. `Discussion`
6. `Conclusion`

这轮改写的核心方向，不是新增 claim，而是把内部口径压成更适合读者的语言。例如：

- 把 `T24` 这类内部锚点压成 `frozen reference benchmark` 一类的读者表述；
- 把 `FR8/statcalib` 压成单独标记的 calibration-extension supplement lane，而不是主线 comparator promotion；
- 把 real-board / `.tflite` / training/material 这些边界，继续压在 `supporting / supplement / blocked` 层，而不是写成“已经完成部署”。

3 份新增台账分别承担不同职责：

- `paper_bounded_final_polish_change_map.md`
  - 记录这轮到底改了哪些 section、为什么改、改完后还保留了哪些 strongest supported truth，以及哪些边界完全没碰。
- `paper_reader_facing_term_translation_table.md`
  - 把最容易写歪的内部术语单独列出来，明确允许怎么翻、禁止怎么讲。
- `paper_appendix_supplement_reader_assembly_map.md`
  - 明确 main text / appendix / supplement / blocked 的装配分层，降低后续把不同证据层混写成一层的风险。

从后续开发和写作的意义看，这轮工作很重要，但意义是“收口写法”，不是“升级事实”：

- 它让后续作者在继续润色时更不容易把内部 task/provenance 语言直接端上正文；
- 它让主结果层、支持解释层、补充扩展层、blocked 硬件层的层级关系更清楚；
- 它给后续是否进入更强的 manuscript assembly 或 submission-side 工作提供了更干净的起点；
- 但它没有改变 `docs/02_experiment_plan.md` 中那些核心边界：主结果仍是冻结四场景 mock-backed software-HIL，`.tflite` 仍是 isolated current-host runtime，real-board 仍没有变成 execution success。

## 3. 为什么我的 review 结果是 PASS_WITH_WARNINGS

我没有给 `BLOCK`，因为 `T84` 的主体任务其实已经完成了：

- 3 份必需台账都已创建；
- 两个 README 都已登记 `T84` 入口；
- note 的 6 个目标 section 都确实被改动，并有 `% T84-POLISH: ...` 标记；
- `T80/T81/T82/T83` 的旧标记链都还在；
- 没有越过 allowlist；
- 没有引入新实验、没有改代码、没有改治理文档；
- 本地 `TeX Live 2024 + latexmk` 编译成功，`.log` 也没有扫出常见 warning 关键字。

我没有给纯 `PASS`，是因为存在一个真实但不至于阻塞的问题：

- 在 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的 `Conclusion` 里，仍然保留着一句：
  - `The remaining writing work is to translate these internal layers into a final reader-facing polish pass ...`

这句话的问题在于：`T84` 自己就是这轮 reader-facing final polish。也就是说，正文里仍有一处把“本轮已经做完的事”写成“后面还要做的事”。

这个问题为什么不是 block：

- 它没有造成越界；
- 没有把 blocked surface 写强；
- 没有把计划写成事实；
- 没有破坏编译或证据边界；
- 也没有让 3 份配套台账失效。

但它为什么值得降成 `PASS_WITH_WARNINGS`：

- 因为它会让 note 内部状态和 `worker_summary`、`change_map`、review closeout 口径之间出现轻微不同步；
- 后续作者如果只看 note，不看台账，可能会误以为“reader-facing polish 这一步还没开始做”。

所以，这更像一个“小的收口状态滞后”，而不是任务失败。

## 4. Worker 已写的 review / explanation 是否有问题

方向上，Worker 已写的 `T84_review.md` 和 `T84_explanation.md` 大体是对的：

- 它们正确描述了 `T84` 是 docs-only reader-facing final polish；
- 也正确保留了 `.tflite`、real-board、`statcalib`、training/material 等边界；
- 对 README 登记、marker 链、LaTeX 编译的描述也基本准确。

我这里补充和修正的地方主要有两点：

1. 我把 verdict 从 `PASS` 调整为 `PASS_WITH_WARNINGS`。
原因不是任务没做，而是正文 `Conclusion` 里还残留一处把 `T84` 本轮工作写成“后续待做”的句子。

2. 我把这个 warning 的性质说得更明确。
它不是证据问题，不是实验问题，也不是越界问题，而是“文稿状态同步还差最后一小步”的问题。

## 5. 推荐怎么处理这个 warning

最合理的后续动作不是重开大任务，也不是补实验，而是：

- 如果 Captain 觉得值得收口，就只开一个极小的 docs-only cleanup；
- 只改 `Conclusion` 里那句仍把 final polish 写成未来工作的表述；
- 不要顺手扩大成 submission-ready pack、部署闭环、硬件完成态或新 benchmark 叙事。

换句话说，`T84` 已经把主线 note 推到了“更像读者稿”的状态，但还差一处小的状态同步清理，才能让文内表述和 closeout 结论完全一致。
