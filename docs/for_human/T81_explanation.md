# T81 说明

## 1. 这个任务在做什么

`T81` 的意思可以简单理解成：

在 `T80` 已经把 8 个“可以先改写的主线章节”收口之后，再把之前故意没动的 4 个剩余核心章节也校准到同一套证据边界上。

这 4 个章节是：

- `Summary of Contributions`
- `Brief Review of the GKP Code`
- `Noise and Drift Model`
- `Model Architecture`

所以，`T81` 不是做新实验，也不是宣布论文整篇已经 ready，而是把 note 里最后一块“旧口径残留区”补齐，让整份 mainline note 至少在贡献概括和方法描述上不再前后失配。

## 2. 这次实现到底做了什么

### 2.1 任务背景

从 `docs/02_experiment_plan.md`、`docs/04_task_board.md` 和 `docs/07_handoff.md` 可以看出，当前项目仍处在 `Phase 2: Controlled Development / Go`，而 paper 相关工作仍被严格限制在“主线 docs/material 收口”这个层级。

`T79` 先做了 reopen gate，结论是允许一轮有界 prose reopen；`T80` 随后完成了 8 个 ready sections 的重写。但 `T80` 是故意留下 4 个区域不动的，因为那 4 个区域更容易一不小心把“方法解释”写成“证据升级”。

所以 `T81` 的目标不是继续扩写，而是专门处理这 4 个还没校准的 section。

### 2.2 实际改动范围

本轮真正改写的目标只有 4 个 section，并在 note 源码里加上了 4 条 `% T81-CALIBRATION: ...` 标记，便于后续 scope 审计。

对应关系是：

- `% T81-CALIBRATION: Summary of Contributions`
- `% T81-CALIBRATION: Brief Review of the GKP Code`
- `% T81-CALIBRATION: Noise and Drift Model`
- `% T81-CALIBRATION: Model Architecture`

同时，`T80` 的 8 条 `% T80-REOPEN` 标记仍然保留，说明这轮没有把 `T81` 偷偷扩成对前 8 个 ready sections 的重新大改。

### 2.3 这些改写具体在校准什么

这轮最核心的动作，其实不是“写得更好看”，而是把几类最容易写过头的叙事压回到当前最强、但仍受限的 evidence stack 上。

具体来说：

1. `Summary of Contributions`
   - 不再把贡献写成“论文已经全面闭环”。
   - 明确主线最强结果仍然是 `T24` 锁定协议下的 frozen-set mock-backed software-HIL 排名。
   - 明确 `FR6/FR7` 仍只是 descriptive support。
   - 明确 `FR8/statcalib` 仍是 separately labeled extension lane，不是 promoted comparator。
   - 明确 training/material、`.tflite`、real-board 仍只是 layered boundary evidence，而不是并列主结果。

2. `Brief Review of the GKP Code`
   - 把它压回“物理背景 + 局部 affine 近似为什么合理”的角色。
   - 不把这一章写成“已经得到 exact decoder closure”。

3. `Noise and Drift Model`
   - 明确这是 effective model / control-oriented abstraction。
   - 不把当前四场景 effective drift 建模写成 full circuit-level、exhaustive 或 hardware-validated noise closure。

4. `Model Architecture`
   - 明确 mainline 仍是 teacher-anchored residual path。
   - 明确 `statcalib` 只是 FR8 extension lane。
   - 明确 `.tflite` 和 real-board 仍处于 supporting / boundary 层，而不是部署闭环。

### 2.4 新增和更新的文档

除了 note 本身，这次还更新了三类配套文档：

1. `docs/paper_materials/paper_methods_and_contribution_calibration_manifest.md`
   - 这是 `T81` 最关键的新文档。
   - 它记录了这 4 个 section 哪些改了、各自绑定哪些 evidence anchors、必须保留哪些 non-claims / guardrails、compile 状态是什么、`T80` 的 8 个 ready sections 如何保持 untouched。

2. `docs/paper_notes/README.md`
   - 把 `T81` 链路登记进去。
   - 明确 `% T81-CALIBRATION` 只覆盖这 4 个 target sections，不代表 full-manuscript reopen。

3. `docs/paper_materials/README.md`
   - 把 `paper_methods_and_contribution_calibration_manifest.md` 注册进 paper-material 索引。
   - 继续强调 `T74-T81` 都是 paper-material / note-calibration 链路，不是新实验事实来源。

### 2.5 有没有代码或配置变化

没有。

这一轮完全是 docs/LaTeX 任务，没有改 `cnn_fpga/`、`benchmark/`、`physics/`、`tests/`、`runs/`、`artifacts/` 或治理文档，也没有启动任何新 benchmark、训练、`.tflite` smoke 或 real-board 执行。

所以这次 review 的重点，不是代码正确性，而是叙事边界是否被偷偷抬高。

### 2.6 对后续开发的意义

`T81` 的意义，是把当前 mainline note 里最后一块最敏感的“贡献 + 方法章”区域，也压到了和 `T74-T80` 一致的真实边界上。

这有三个直接价值：

- 主线 note 的内部一致性更高了。
  - 不会出现前半部分已经很克制，后半部分 methods 还在沿用更老、更容易 overclaim 的说法。

- 后续如果 Captain 要决定是否进入更大范围的 manuscript closeout，会有更清楚的基础。
  - 因为现在不只是结果段、讨论段、结论段被校准了，连 `Summary of Contributions` 和 methods 也已经被限定到当前 strongest supported truth。

- 它继续维持了项目在 `Phase 2` 的治理纪律。
  - 允许写作层收口；
  - 不允许把写作收口伪装成 benchmark 扩张、deployment closure、real-board success 或 `statcalib` promotion。

## 3. 为什么这次 review 给出 PASS

我给 `PASS`，主要因为这次交付满足了任务包要求，而且没有看到会阻止接收的越界问题。

### 3.1 任务确实完成了

任务包要求的几件核心事情都做到了：

- 只改了 4 个 target sections；
- note 源码里出现了 4 条 `% T81-CALIBRATION` 标记；
- `paper_methods_and_contribution_calibration_manifest.md` 已生成；
- 两份 README 已登记；
- 现有 LaTeX 产物已刷新，`.log` 关键字扫描没有看到明显 warning。

### 3.2 没有伪实现、mock、stub、hardcode

这轮本来就不是代码功能任务，所以这里要看的不是“代码是不是假实现”，而是“文字有没有把边界写假”。

我没有看到以下几类问题：

- 没有把 `statcalib` 写成成熟主比较器；
- 没有把 `.tflite` 写成 default-env / deployment closure；
- 没有把 real-board 写成 execution success；
- 没有把 methods 章写成新的硬件验证或更强实验事实。

### 3.3 验证对这类任务来说已经足够

对 docs-only note calibration 任务，最重要的验证不是重跑实验，而是：

- diff 范围有没有越界；
- target section 标记是否齐全；
- `T80` 的旧边界是否被保留；
- manifest 是否把本轮范围和 guardrail 记清楚；
- 编译是否真实可过，日志是否干净。

这些点本轮都能从现有 diff 和产物里核对到。

### 3.4 为什么不是 PASS_WITH_WARNINGS 或 BLOCK

我确实保留了两条非阻断提醒：

- `paper_materials/README.md` 的整理略宽于最低需要；
- compile 仍依赖当前可用的 `TeX Live 2024 + latexmk`，bundled `tectonic` 并没有完全恢复。

但这两点都没有破坏任务目标，也没有让 `T81` 的主要交付失真，所以不足以升级成 `BLOCK`。同时，它们也没有严重到需要把 verdict 压成 `PASS_WITH_WARNINGS`。

## 4. 对已有 worker 文档的看法和补充

仓库里已有的 `docs/review/T81_review.md` 和 `docs/for_human/T81_explanation.md` 草稿，方向基本是对的，尤其抓住了两个核心点：

- `T81` 是 bounded contribution/methods calibration；
- `T81` 完成后仍然不是 full-manuscript reopen。

我这次补充的重点主要有三类：

- 把依据再落得更细一些。
  - 明确本轮只改了哪 4 个 section；
  - 明确 `T80` 的 8 个 section 仍然保留；
  - 明确新文本里的关键句子能回链到哪些现有 paper materials / review 事实。

- 把“为什么能判 PASS”说得更正式一些。
  - 不只是复述任务完成了，而是逐项说明为什么没有越界、为什么验证够用、为什么 residual limitation 仍只是边界而不是缺陷。

- 增加了两条非阻断提醒。
  - README 整理范围略宽；
  - compile 结论不能被夸大成“所有 LaTeX 工具链问题都解决了”。

## 5. 一句话结论

`T81` 的价值，不是新增了什么实验，而是把主线 note 里最后 4 个最容易 overclaim 的 section 也压回了当前 strongest supported truth；它完成得合格，所以 review 结论是 `PASS`，但它依然不等于 full-manuscript reopen。
