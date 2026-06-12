# T80 说明

## 1. 这个任务在做什么

`T80` 可以用一句话概括：

把当前论文 note 里“已经被前序任务证明可以安全重写”的那 8 个主线章节，重新写得更像正式论文正文，但不借机扩写 methods、不补新实验、也不升级任何证据等级。

它不是重新做实验，也不是宣布整篇论文已经 ready，更不是把 `.tflite`、real-board 或 `statcalib` 的边界往上提。它只是一次非常受约束的 prose reopen。

## 2. 这次实现到底做了什么

### 2.1 任务目标

从 `docs/04_task_board.md` 和 `docs/07_handoff.md` 可以看出，`T80` 是在 `T79` 给出 `GO_FOR_BOUNDED_PROSE_REOPEN` 之后才被打开的当前唯一任务。结合 `docs/02_experiment_plan.md` 的总原则，这一轮工作的目标不是“把论文写得更激进”，而是：

- 只整理已经有证据支撑的主线 prose；
- 继续维持当前仓库在 Phase 2 的边界纪律；
- 让论文 note 的主线叙事与前面 `T74-T79` 收拢出来的 claim/evidence 层级一致。

### 2.2 实际改动范围

本轮真正被改写的只有 8 个 ready sections，对应 note 源文件中的 `% T80-REOPEN` 标记：

- `Title`
- `Abstract`
- `Introduction`
- `Relationship to Existing Work`
- `Experimental Setup`
- `Numerical Results`
- `Discussion`
- `Conclusion`

这些标记在 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex` 的第 11、20、61、536、643、687、993、1056 行附近出现，便于后续做范围审计。

### 2.3 明确没有改什么

以下内容本轮保持不动：

- `Summary of Contributions`
- `Brief Review of the GKP Code`
- `Noise and Drift Model`
- `Model Architecture`

这点很重要。因为 `T80` 的任务包明确要求“只做 bounded prose reopen”，不能把 methods calibration 偷带进来。也就是说，这次工作完成以后，当前 note 仍然不是 full-manuscript reopen，只是“部分 ready sections 已经被重新整理过”。

### 2.4 新增和更新的文档

这次除了改 note 本身，还做了三类配套文档工作：

1. 新增 `docs/paper_materials/paper_bounded_prose_reopen_manifest.md`
   - 这是本轮最关键的 traceability 文档。
   - 它逐节登记哪些 section 改了、绑定哪些 evidence anchors、保留了哪些 guardrails、compile 状态是什么、哪些 out-of-scope sections 保持 untouched。

2. 更新 `docs/paper_notes/README.md`
   - 把 `T80` 的入口登记进去。
   - 明确 `% T80-REOPEN` 只覆盖 8 个 ready sections，不包含 `Summary of Contributions` 和三章 methods。

3. 更新 `docs/paper_materials/README.md`
   - 把新的 prose reopen manifest 注册到 paper materials 索引里。
   - 明确说明这个 manifest 只代表“有界 prose reopen 已完成”，不代表 full-manuscript reopen 获批。

### 2.5 是否涉及代码或配置变化

没有。

本轮是纯文档/LaTeX note 任务，没有修改 `cnn_fpga/`、`benchmark/`、`physics/`、`tests/` 或任何运行配置，因此不存在“功能被改坏”的代码回归风险。这里的风险主要是叙事风险，也就是会不会把边界说过头。

### 2.6 对后续开发的意义

这轮工作的价值，不在于新增了实验事实，而在于把现有事实的叙述层整理得更可用：

- 对论文主线来说，当前最稳的那层叙事已经不再分散在多个补丁式材料里，而是收回到了 note 正文中。
- 对后续任务来说，`paper_bounded_prose_reopen_manifest.md` 给出了一个很清楚的“哪些地方已经安全改写，哪些地方不能顺手推进”的边界图。
- 对治理来说，它延续了 `T79` 的 reopen gate 逻辑：允许写作层收口，但不允许把写作收口冒充成实验闭环、部署闭环或 comparator promotion。

## 3. 为什么这次 review 给出 PASS

我给 `PASS`，理由是这次交付满足了任务包要求，而且没有发现越界或伪完成：

- 任务确实完成了。
  - 8 个允许 section 已重写。
  - `% T80-REOPEN` 标记已加入。
  - manifest 已新增。
  - 两份 README 已登记。
  - note 编译产物已刷新，`.log` 里没有看到明显的 warning 关键字。

- 没有伪实现、mock、stub、hardcode 问题。
  - 因为这不是代码功能任务，而是 prose/manifest 收口任务。
  - 重点不是“功能真假”，而是“叙事有没有把边界说假”。这一点本轮没有发现。

- 没有把计划写成事实。
  - `T24` 仍然只被写成主线主锚点。
  - `FR6/FR7` 仍然只是 descriptive support。
  - `FR8/statcalib` 仍然保持 extension lane / no-promotion。
  - `.tflite` 没被写成 default-env/deployment closure。
  - real-board 没被写成 execution success。

- 没有破坏已有功能。
  - 因为没有改源码，也没有改 benchmark/protocol。

之所以不是 `PASS_WITH_WARNINGS` 或 `BLOCK`，是因为我没有看到会阻止接收本轮交付的缺口。剩下的限制，例如“整篇 note 仍未 full reopen”“methods 仍未校准”，本来就是任务边界本身，不是本轮实现失败。

## 4. 对已有 worker 文档的看法和补充

仓库里已有的 `docs/review/T80_review.md` 和 `docs/for_human/T80_explanation.md` 草稿，方向基本是对的：都抓住了“这是一轮 bounded prose reopen，而不是 full-manuscript reopen”这个核心。

我这次补充的重点主要有三点：

- 把判断依据再落得更具体一些。
  - 明确指出 `% T80-REOPEN` 出现在哪些 section；
  - 明确指出哪些章节仍 untouched；
  - 明确指出 manifest 与 README 是如何共同限制外推的。

- 把 compile 口径说得更严谨一些。
  - 这里只能说“当前 note 已在当前可用工具链上成功刷新”，不能泛化成“所有 LaTeX 路径都已完全稳定”。

- 把“为什么是 PASS”说得更像正式 review。
  - 不只是说“看起来没问题”，而是逐项对应任务完成度、边界一致性、验证覆盖和 residual boundary。

## 5. 一句话结论

`T80` 的价值，是把已经 ready 的主线章节写得更像论文正文，同时继续守住证据边界；它完成得合格，所以 review 结论是 `PASS`，但它绝不等于整篇论文已经全面 reopen。
