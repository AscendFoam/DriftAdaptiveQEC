# T73 任务解释与本次 Review 说明

## 1. 先用通俗的话解释这个 task

`T73` 不是去跑新实验，也不是继续写论文正文。

它做的事情更像“给论文材料做总账”：

- 哪些说法现在真的能写
- 哪些图表现在真的能引用
- 哪些地方最容易越界

前面几个任务已经各自补了不少新证据：

- `T48` 补了 isolated true `.tflite` runtime
- `T50` 补了 training/material regeneration pack
- `T57/T58` 补了 FR7/FR6 论文材料
- `T70` 补了 `statcalib` 的 bounded closure pack
- `T72` 补了 real-board gate/transfer-pack provenance

问题是，这些证据当时分散在不同 evidence pack 和 review 里。`T73` 的作用，就是把它们统一回写到 paper-facing 的三本主台账里，避免后面写 paper claim、挑图表、写风险说明时还要到处翻。

## 2. 这个任务具体实现了什么

从 `docs/02_experiment_plan.md`、`docs/04_task_board.md` 和 `docs/07_handoff.md` 看，`T73` 的定位很明确：

- 它是 `post-T72` 的 mainline docs-only 台账刷新任务
- 它不是 benchmark、`.tflite`、real-board execution 或 paper prose reopen
- 它完成后，主线应该转去 `T74`，而不是重新打开真板执行准备

这次实现主要做了五件事。

第一，刷新了 `docs/paper_materials/paper_claim_evidence_ledger.md`。

这本账回答“现在论文里哪些 claim 能写”。本轮最关键的变化是把以下四类近期事实吸收进去了：

- `T48`：可以写 isolated current-host true `.tflite` runtime，但不能写 default-env / HIL / deployment closure
- `T50`：可以写 canonical material chain intact + clean CPU-only bounded rerun，但不能写 full reproducibility
- `T70`：可以写 `statcalib` extension lane closure + no-promotion gate，但不能写 mature comparator
- `T72`：可以写 checked-in read-only real-board gate / regeneration / provenance boundary，但不能写 execution success

第二，新建了 `docs/paper_materials/paper_result_figure_ledger.md`。

这本账回答“现在有哪些图、表、result-pack 可以往论文里带”。它把 paper-facing 项目按 `ready / partial / blocked` 分开，并且给每一项都绑定了来源：

- 对应 evidence pack
- 对应 review
- 对应 run root / artifact / figure asset / table.csv
- 以及不能外推的边界

其中最低要求覆盖的几类都已经出现了：

- `T24` frozen-set formal software revalidation table
- `FR6` mechanism/intervention figure pack
- `FR7` feature/teacher ablation table
- `FR8` statcalib bounded closure pack
- training/material boundary table
- deployment boundary table

第三，新建了 `docs/paper_materials/paper_claim_risk_table.md`。

这本账回答“哪些 claim area 最容易写过头”。它把 paper-facing 风险重新映射到了当前治理口径：

- `R31` 已由 `T72` 收口
- `R32` 是更窄的 future-host provenance 剩余风险
- `R33` 是当前没有 `Linux + FPGA` 硬件宿主带来的主线现实约束
- `R24` 仍然约束 `statcalib`
- `R15/R16` 仍然约束 paper-grade expanded benchmark / prose overclaim

并且每项都给了“最安全写法”和“禁止写法”。

第四，同步了 `docs/paper_materials/README.md`。

现在这个目录的入口关系更清楚了：claim ledger、result ledger、risk table、ablation pack 各自是什么，应该先看哪一份，以及 post-`T72` 之后必须一起保留的边界是什么。

第五，同步了 `docs/paper_materials/paper_ablation_result_pack.md`。

这里最关键的是把 `FR8` 从旧的“missing”状态改成了更准确的“partial extension-lane closure / no-promotion material”。这不是新结果，只是把 `T70` 已经建立的现实边界正确写回了材料账本。

## 3. 这对后续开发有什么意义

`T73` 的意义不在于新增实验结果，而在于把“论文材料到底能怎么用”这件事统一了。

它的直接价值有三点：

1. 后面写 paper claim 时，不容易把 bounded evidence 写成更强完成态。
2. 后面挑图、挑表时，不容易把还只是 `partial` 的材料误当成 `ready`。
3. 后面做 `T74` 这类论文可直接复用的 simulation result / figure pack 时，有了统一入口，不用再先手工拼 claim、result、risk 三层语义。

换句话说，`T73` 解决的是“材料已经有了，但总账还没统一”的问题。它是 paper-facing 主线的清账动作，不是能力升级动作。

## 4. 为什么这次 review 给的是 `PASS`

我给 `PASS`，而不是 `PASS_WITH_WARNINGS` 或 `BLOCK`，原因是这次工作满足了 task package 的完成标准，而且没有发现新的边界问题。

我重点核对了六件事。

第一，是否真的完成任务。

结论是完成了。task package 要求的四个核心目标都已落地：

- claim ledger 已刷新
- result/figure ledger 已新建
- paper claim risk table 已新建
- README 与 `paper_ablation_result_pack.md` 已同步

第二，是否有伪实现、mock、stub、hardcode。

这轮是 docs-only 任务，所以“伪实现”的典型风险是：只是写了名字，但实际没有对应证据路径；或者把旧计划换个说法写成新事实。我的抽查结果是：

- 台账里列出的关键 `T48/T50/T70/T72` evidence / review / artifact / run 路径都存在
- 我对几份新文档里出现的 repo 路径做了存在性检查，没有发现悬空路径
- 文档也没有把 blocked/partial/extension-lane/no-promotion 静默升级

第三，是否缺测试或验证。

对这类 docs-only 任务，没有源码测试需求。关键验证是：

- 有没有越界改动
- 路径是否真实存在
- 口径是否和现有 review / evidence pack 一致

我复核了 `git diff` 的几个边界命令，确认 `runs/`、`artifacts/`、源码/测试目录、治理文档都没有被改动。

第四，是否过度工程。

没有。新增的两本台账和一份风险表，都是 task package 明确要求的；README 和 ablation pack 的同步也都是必需动作，不是额外扩写。

第五，是否破坏已有功能。

没有。因为这轮没有碰源码、测试、运行产物或治理文档，影响范围严格限制在 paper-facing 材料目录和 review/explanation/summary。

第六，文档是否把计划写成事实。

没有发现这种问题。特别是几条高风险边界都还保留得很清楚：

- `T48` 没被写成 default-env / deployment closure
- `T49/T71/T72` 没被写成 real-board success
- `T70` 没被写成 mature comparator promotion
- `FR6/FR7` 没被写成 causal proof

所以，从 reviewer 角度看，`T73` 是一次真实完成、边界诚实、范围受控的 docs-only 主线收敛任务，应该给 `PASS`。

## 5. Worker 已写的 review / explanation 文档有没有问题

总体方向是对的，没有发现实质性错误。

Worker 已有文档里最重要的几条口径都保持正确：

- `T73` 是台账刷新，不是新实验
- `T48/T50/T70/T72` 的边界没有被升级
- `R31` / `R32` / `R33` 的关系写对了
- `FR8` 被同步为 extension-lane closure / no-promotion material，而不是 promoted comparator

我这次补充的地方主要有两类：

1. 把“为什么应该给 PASS”写得更明确。
   - 原先 worker 的 `docs/review/T73_review.md` 更像自检记录，不是最终 reviewer verdict。

2. 把这个任务放回主线节奏里解释清楚。
   - 也就是：为什么 `T73` 不是简单文档整理，而是 `post-T72` 到 `T74` 之间必须经过的一次主台账统一动作。

## 6. 一句话总结

`T73` 不是让项目更“强”了，而是让项目当前到底“能说到哪一步”这件事，终于有了一套统一、可回查、不会乱升级的 paper-facing 总账。
