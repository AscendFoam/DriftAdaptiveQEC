# T58 说明：FR6 多 seed 机制 / intervention 图包

## 1. 这个 task 在做什么

T58 不是新实验，也不是新 benchmark。

它做的事情很具体：把前面已经完成的三段证据

- `T54` 的 6-seed trace generalization
- `T55` 的 I1 intervention probe
- `T56` 的机制表述收口

整理成一个可以给论文直接引用的 `FR6` 图包。

通俗地说，T58 的目标不是“证明机制已经搞清楚了”，而是“把已经有的证据画成一张诚实、可追溯、不会乱夸的图”。

这张图至少要回答两件事：

1. 多个 seed 下，Gated v5 相对 Full 的表现差异是什么样。
2. 把 I1 clip-reduction intervention 加进去以后，每个 seed 是变好还是变坏。

## 2. 这次实现具体做了什么

### 2.1 产物层

Worker 新建了一个 task-scoped 图资产目录：

- `docs/figure_assets/T58_fr6_multi_seed_mechanism_intervention/`

里面有：

- `build_figure.py`
- `fr6_multi_seed_mechanism_intervention.svg`
- `fr6_multi_seed_mechanism_intervention.png`
- `figure_data.csv`
- `figure_manifest.json`
- `caption.md`

此外还补了主说明文档：

- `docs/fr6_multi_seed_mechanism_intervention_figure_pack.md`

并更新了论文侧 ledger：

- `docs/paper_ablation_result_pack.md`
- `docs/paper_claim_evidence_ledger.md`
- `docs/reality_recovery/04_figure_and_result_ledger.md`
- `docs/reality_recovery/05_paper_claim_risk_table.md`

核心变化是：把 `FR6` 从“还缺图”推进到“图包资产 ready，但解释边界仍然很严”的状态。

### 2.2 数据来源层

这次图包没有新跑数据，而是直接复用已有结果：

- Panel A 来自 `runs/T54_multi_seed_trace_phase_a_20260522/cross_seed_comparison.csv`
- Panel B 来自 `runs/T55_multi_seed_i1_probe_20260523/analysis/intervention_summary.csv`

我核对过 `figure_data.csv` 里的关键数值，它们和源 CSV 是对得上的：

- 6 个 seed 的 `mean(Gated v5) - mean(Full)` 数值一致
- 6 个 seed 的 `mean(I1) - mean(Gated v5 baseline)` 数值一致
- `harmful / mixed_or_no_clear_effect / helpful` verdict 也一致

### 2.3 图本身在表达什么

这张图是一个两 panel 图：

1. Panel A
   - 画的是每个 seed 的 baseline gap
   - 负值表示 `Gated v5` 比 `Full` 更好
   - 同时给 seed 打上 `quiet / classic / universal` 标签

2. Panel B
   - 画的是 I1 intervention 相对原始 Gated v5 baseline 的变化
   - 正值表示 intervention 变差
   - 负值表示 intervention 变好

这个设计和 T58 task package 规定的默认安全方案是对齐的。

### 2.4 对后续开发 / 论文收口的意义

结合 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md` 来看，T58 的意义主要有三点：

1. 它补的是 paper-material gap，不是 experiment gap。
   - 也就是说，之前缺的不是“有没有证据”，而是“有没有一个正式、可引用、可追溯的图包”。

2. 它让 `FR6` 可以从 narrative-only 变成 asset-backed。
   - 以后引用 FR6，不需要再靠散落在 T54/T55/T56 文档里的文字描述拼图。

3. 它没有改变主结论边界。
   - `C4` 仍然是 `partial`
   - 不能说 causal proof
   - 不能说机制闭环
   - 不能把 “high committed-b is harmful” 这条已被 T55/T56 削弱甚至退休的旧叙事重新写活

所以，T58 的价值是“让已有证据更好地被论文使用”，不是“让机制故事更强”。

## 3. 为什么我的 review 结果改成了 PASS_WITH_WARNINGS

我的结论分两层：

### 3.1 内容层面：大体完成，而且没有明显乱写事实

如果只看 FR6 本身，我认为它基本完成了 task：

- 图资产齐全
- 数据来源清楚
- caption 和 ledger 没有明显越界
- 没把计划写成事实
- 没把图写成 causal proof

这一层我没有发现伪实现、mock 冒充真结果、或者把 T54/T55/T56 之外的数据偷偷塞进来。

### 3.2 diff 层面：原来的阻塞点在你澄清后不再成立

我最初给 `BLOCK` 的原因不在 FR6 图本身，而在这次待审 diff 的边界。

T58 task package 明确禁止修改这些治理文件：

- `docs/00_project_snapshot.md`
- `docs/01_legacy_audit.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

当时从 `git diff / git status` 看，这些文件和 T58 产物一起出现在同一份待审改动里。

这会带来两个问题：

1. Worker 没有满足 “只改 Allowed files” 这条硬约束。
2. Worker 自己写的 `T58_review.md` 和 `T58_worker_summary.md` 里声称“only allowed docs changed”，但从当前 diff 状态看，这句话站不住。

你现在补充说明了一个关键前提：

- 这些治理文档改动是 Captain 在 Worker 开始前就做的
- 只是当时忘记先提交
- 所以它们不应算作 Worker 在 T58 中的越界修改

在这个前提下，我会把原来的 scope blocker 降级为非阻塞警告：

1. Worker 本身可以视为没有违反 Allowed files。
2. 但这次审查也暴露了一个协作问题：
   - 如果 Captain 的治理同步和 Worker 的任务产出混在同一个未提交工作区里
   - reviewer 仅靠 git 状态就很难机械地区分“谁改了什么”

所以更新后的 verdict 更适合写成 `PASS_WITH_WARNINGS`：

- `PASS`
  因为 FR6 图包内容本身是成立的，且没有明显越界夸大
- `WITH_WARNINGS`
  因为这次边界判断依赖了你后补的人工说明，而不是从提交状态里一眼可证

## 4. 我对 Worker 已写 review / explanation 的看法

### 4.1 对 Worker 原始 `docs/review/T58_review.md` 的看法

它更像 self-checklist，不像 adversarial review。

它的问题主要有两个：

1. 它只检查了 FR6 内容是否存在，没有处理“工作区里混入未提交 Captain 改动时 reviewer 应如何判边界”这个问题。
2. 它没有指出 `build_figure.py` 里存在一层 task-local 派生逻辑：
   - `quiet / classic / universal` 是通过硬编码阈值重新推出来的
   - 这不算伪造结果，但 provenance 比“直接引用一个冻结好的类别表”更弱

所以原来的 worker review 不能作为最终 acceptance review，但它作为 delivery checklist 仍然有参考价值。

### 4.2 对现有 `docs/for_human/T58_explanation.md` 的看法

原文方向是对的，但过于短，只说了：

- T58 没有新实验
- 图有两 panel
- 边界没有变

它缺少几件对人真正有帮助的事情：

1. 没解释为什么项目在 `T57` 之后要先补 `FR6`。
2. 没解释图的数据到底来自哪两个 frozen source。
3. 没解释这项工作的真正价值是“补 paper asset”，不是“升级机制证据”。
4. 没解释为什么这次 review 虽然肯定了大部分内容，却仍然给了 `BLOCK`。

这份新说明文档就是在把这些缺的上下文补全。

## 5. 接下来最合理的处理方式

最合理的下一步仍然不是重做 FR6 图，而是把协作边界做得更清楚：

1. 保留这次 T58 的 FR6 图包和 paper ledger 更新。
2. 后续尽量把 Captain 治理同步先提交，或者至少和 Worker 任务产物分开。
3. 这样之后的 Reviewer 不需要依赖聊天解释来判断 scope。

换句话说，这次 T58 的主要问题已经不再是“内容不合格”，而是“工作流边界不够自证”。在你补充说明后，我认为它更适合按 `PASS_WITH_WARNINGS` 处理。
