# T82 说明

## 1. 这个任务在做什么

`T82` 可以简单理解成：

把当前论文材料里那些“能说明边界、但不能当主结果”的 supporting materials，统一整理成一条更清楚的 manuscript-facing 路线。

这轮不是再写一轮大 prose，也不是补实验，更不是宣布论文已经 full-manuscript closeout。它做的是另一种收口：把哪些内容该放主文、哪些该放 appendix、哪些只能放 supplement、哪些目前必须老老实实标成 blocked，集中写清楚。

## 2. 这次实现到底做了什么

### 2.1 任务背景

从 `docs/02_experiment_plan.md`、`docs/04_task_board.md` 和 `docs/07_handoff.md` 可以看出，当前项目仍处在 `Phase 2: Controlled Development / Go`，paper 主线工作还在 `Research Reality Recovery Mode` 下受控推进。

`T79` 给了 bounded prose reopen 的 gate，`T80` 重写了 8 个 ready sections，`T81` 又把 `Summary of Contributions` 和三章 methods 压回当前证据边界。到这一步，主文叙事和方法叙事已经基本对齐，但 supporting-boundary 材料还比较分散：

- `FR8/statcalib` 的 supplement-side boundary
- training/material 的 reproducibility boundary
- isolated current-host true `.tflite` runtime boundary
- real-board 的 read-only gate / provenance with current-host `NO_GO`
- 以及更高层的 blocked hardware-dependent surface

`T82` 的目标，就是把这些分散边界整理成一条更清楚的 manuscript-facing closeout route。

### 2.2 实际改动范围

本轮新增了两份核心文档：

1. `docs/paper_materials/paper_supporting_material_closeout_pack.md`
2. `docs/paper_materials/paper_manuscript_closeout_readiness_matrix.md`

同时，只回写了 note 里 4 处与 supporting-boundary 直接相关的段落，并加上了 4 条 `% T82-SUPPORT: ...` 标记：

- `Runtime, quantization, and fixed-point degradation`
- `Embedded runtime and board-level validation`
- `Discussion` 中的 deployment/support boundary 段落
- `Conclusion` 中的 remaining technical gap 段落

除此之外，还更新了：

- `docs/paper_notes/README.md`
- `docs/paper_materials/README.md`
- `docs/review/T82_review.md`
- `docs/for_human/T82_explanation.md`
- `docs/worker_summary/T82_worker_summary.md`

### 2.3 两个新增文档分别解决什么问题

#### A. `paper_supporting_material_closeout_pack.md`

这份文件做的是“路由整合”。

它把当前 supporting surfaces 按四层拆开：

- `main text`
- `appendix`
- `supplement`
- `blocked`

并且对每一层都写清楚：

- 证据锚点是什么；
- 最安全的 manuscript 用法是什么；
- 哪些 forbidden claims 绝对不能写。

它给出的核心收口是：

- `T24` frozen benchmark anchor 仍然留在主文主结果层；
- `FR6/FR7` 作为保守解释层，可进主文加附录；
- training/material 与 isolated current-host true `.tflite` 留在 appendix；
- `FR8/statcalib` 与 real-board gate/provenance 留在 supplement；
- 任何依赖 `Linux + FPGA` 宿主、device path、board timing/resource 的 surface 继续明确写成 `blocked`。

#### B. `paper_manuscript_closeout_readiness_matrix.md`

这份文件做的是“状态分类”。

它不回答“项目还差多少工作才算全部完成”，而是回答：

当前各个 manuscript-facing surface 到底是：

- `ready`
- `support-only`
- `blocked`

并为每一类补齐：

- `blocker_type`
- `evidence anchors`
- `forbidden claims`
- `next bounded action`

这意味着 `T82` 不是单纯地说“这些材料可以写”，而是进一步说明“能写到什么层、不能写到什么层、若要升级下一步该开什么新任务”。

### 2.4 note 里的局部改动在做什么

`T82` 回写 note 的 4 处 supporting-boundary 段落，核心不是增补新事实，而是把 supporting surfaces 的层级说得更清楚。

具体来说：

1. `Runtime, quantization, and fixed-point degradation`
   - 明确当前最强 runtime-facing 事实，只是“选定保留的 float/int8 `.tflite` artifacts 在一个隔离 current-host 环境中被真实执行过”。
   - 明确更广义的 fixed-point/runtime degradation 问题仍未关闭。

2. `Embedded runtime and board-level validation`
   - 明确把 `.tflite` current-host runtime 和 real-board gate/provenance 分成两层。
   - 明确因为当前没有可用的 `Linux + FPGA` host 和 openable device paths，所以硬件执行面仍然是 blocked。

3. `Discussion` 中的 deployment/support boundary 段落
   - 把整个 supporting route 组织成 `main text / appendix / supplement / blocked` 四层，而不是混成一个 deployment story。

4. `Conclusion` 中的 remaining technical gap 段落
   - 明确这轮完成的是 supporting-material integration。
   - 明确没有完成 deployment closure，也没有完成 full-manuscript finalization。

### 2.5 有没有代码或配置变化

没有。

本轮完全是 docs/LaTeX 任务，没有修改 `cnn_fpga/`、`benchmark/`、`physics/`、`tests/`、`runs/`、`artifacts/` 或治理文档，也没有启动任何 benchmark、训练、`.tflite` smoke 或 real-board 执行。

所以这次 review 的重点不是代码正确性，而是叙事边界有没有被写高。

### 2.6 对后续开发的意义

`T82` 的意义，不在于新增实验，而在于把“论文里哪些 supporting materials 能安全出现、应出现在哪一层、哪些仍必须 blocked”这件事收清楚了。

它的价值主要有三点：

- 主文、附录、补充材料和 blocked surface 的边界更清楚了。
  - 后续作者更不容易把 support-only 材料误写成主结果或部署完成态。

- 后续 Captain 更容易做下一步 gate 决策。
  - 因为现在不只是 prose 被校准了，连 supporting surfaces 的 manuscript-facing readiness 也有了统一矩阵。

- 它继续维持了项目在 `Phase 2` 的治理纪律。
  - 允许 paper-facing 材料整合；
  - 不允许把整合材料冒充成 hardware success、deployment closure 或 full-manuscript closeout。

## 3. 为什么这次 review 给出 PASS

我给 `PASS`，原因是这次交付满足了任务包要求，而且没有发现会阻止接收的越界问题。

### 3.1 任务确实完成了

任务包要求的关键项都在：

- supporting-material closeout pack 已新增；
- manuscript closeout readiness matrix 已新增；
- note 里确实出现了 4 条 `% T82-SUPPORT` 标记；
- `T81` 的 4 条 `% T81-CALIBRATION` 与 `T80` 的 8 条 `% T80-REOPEN` 仍保留；
- 两份 README 已登记；
- LaTeX 产物已刷新，`.log` 关键字扫描没看到明显 warning。

### 3.2 没有伪实现、mock、stub、hardcode

这轮不是代码功能任务，所以要看的不是“代码有没有假实现”，而是“文档有没有把边界写假”。

我没有看到以下问题：

- 没有把 `statcalib` 写成 promoted comparator；
- 没有把 training/material 写成 full reproducibility；
- 没有把 `.tflite` 写成 default-env / deployment closure；
- 没有把 real-board 写成 execution success；
- 没有把 blocked hardware-dependent surface 写成“其实已经具备，只差文案”。

### 3.3 对这类任务来说，验证已经够用

对 docs-only supporting-material closeout 任务，关键验证不在于重跑实验，而在于：

- diff 范围是否越界；
- 4 处 `% T82-SUPPORT` 标记是否齐全；
- `T80/T81` 的既有边界标记是否仍保留；
- 两个新增文档是否真的把 route/readiness 结构写完整；
- 编译是否真实通过，日志是否干净。

这些点本轮都能从现有 diff 和产物中直接核对。

### 3.4 为什么不是 PASS_WITH_WARNINGS 或 BLOCK

我保留了两条非阻断提醒：

- 这轮完成的是 supporting-boundary route closeout，不是 full-manuscript closeout；
- compile 结论依赖当前主机可用的 `TeX Live 2024 + latexmk`，bundled `tectonic` 仍没有完全恢复。

但这两点都没有破坏任务目标，也没有让 `T82` 的主要交付失真，所以不足以升级成 `BLOCK`，也没有严重到必须压成 `PASS_WITH_WARNINGS`。

## 4. 对已有 worker 文档的看法和补充

仓库里已有的 `docs/review/T82_review.md` 和 `docs/for_human/T82_explanation.md` 草稿，方向基本是对的，已经抓住了两个关键点：

- `T82` 做的是 supporting-material closeout；
- `T82` 完成后仍然不是 full-manuscript closeout。

我这次补充的重点主要有三类：

- 把两个新增文档的分工说得更清楚。
  - 一个负责 route 整合；
  - 一个负责 readiness 分类。

- 把 note 中 4 处局部改写的作用说具体。
  - 不只是“改了 supporting 段落”，而是解释它们分别在澄清什么边界。

- 把为什么能给 `PASS` 说得更完整。
  - 逐项对应任务完成度、边界守恒、验证覆盖，以及为什么剩余 limitation 仍只是边界，而不是本轮失败。

## 5. 一句话结论

`T82` 的价值，不是让论文 suddenly ready，而是把 supporting materials 的 manuscript-facing 路由和 blocked surface 交代清楚；它完成得合格，所以 review 结论是 `PASS`，但它依然不等于 full-manuscript closeout。
