# T89 说明

## 1. 通俗解释

`T89` 不是去“继续写论文”，也不是去“把证据补强”。它更像是把已经冻结好的主线论文材料装进一个带说明书的档案袋里，并在袋子外面贴好规则：

- 现在该先看哪几份文档
- 哪些内容是当前唯一可信入口
- 哪些结论绝对不能顺手写强
- 以后如果还想改，什么情况能直接改，什么情况必须重新开任务

如果说 `T88` 回答的是“当前主线能不能冻结并交接”，那 `T89` 回答的就是“冻结以后，别人该怎么安全接手，而不把边界写坏”。

## 2. 详细解释：任务目标、流程、文档变化和意义

### 2.1 任务目标

从 `docs/04_task_board.md` 和 `docs/07_handoff.md` 来看，`T89` 是当前唯一任务；从 `docs/02_experiment_plan.md` 的大方向看，仓库仍处在 `Controlled Development` 阶段，重点是保护已经验证过的主线证据，不把 blocked surface、`.tflite`、real-board、training reproducibility、`FR8/statcalib` 等边界偷偷写强。

因此，`T89` 的目标不是新增主线事实，而是把 `T88` 已经得到的 `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY` 收口成一套更稳的维护协议：

- 给 frozen mainline 指定单一 handoff 入口
- 把当前 authoritative source 集中列出来
- 把 post-freeze 变更分级成明确规则
- 把 blocked surface 以后重开的最低证据门槛写清楚

### 2.2 实际任务流程

这轮实现是纯文档型，没有代码变更、没有配置变更、没有实验重跑。

新增了 4 份 paper-material 文档：

- `paper_frozen_mainline_handoff_packet.md`
  - 定义“后续维护当前 frozen mainline 时，先看什么、允许引用什么、哪些 non-claims 必须继续保留”。
- `paper_frozen_mainline_source_of_truth_map.md`
  - 把 `FZ01-FZ05` 和 `BD01-BD06` 统一映射到 authoritative source，避免后续维护者从旧 review、旧任务包或零散 prose 自己拼事实版本。
- `paper_postfreeze_change_control.md`
  - 定义 `L0/L1/L2/L3` 四级变更控制，并给出 `CCR-01` 到 `CCR-10` 的具体规则。
- `paper_blocked_surface_reentry_conditions.md`
  - 定义 `RE01` 到 `RE07` 的 blocked surface 重开条件，覆盖 real-board、`.tflite` portability、training reproducibility、`FR8/statcalib`、expanded benchmark、theory mergeback、deployment-closure route。

同时更新了 2 个 README：

- `docs/paper_materials/README.md`
- `docs/paper_notes/README.md`

这两个 README 的作用，是把 `T89` 新增文档正式登记进阅读路径，并明确说明：这些文档只服务 handoff 和 change-control，不升级任何证据等级。

### 2.3 对后续开发的意义

`T89` 对后续开发的价值，主要在“防误写”和“防越界”。

它把未来维护者最容易犯的几类错误提前堵住了：

1. 不再需要从旧任务包、旧 review、旧 caption pack 里拼凑“当前该信哪个版本”。
2. 不再容易把 `BD01-BD06` 这类 blocked disclaimer 在人工润色时悄悄写弱。
3. 不再容易把 “看起来已经差不多” 误写成 “已经 submission-ready completed” 或 “已经 deployment closure”。
4. 不再容易把 theory 分支内容直接 mergeback 到当前 mainline。

从治理角度看，这正好承接了 `T88` 的 freeze gate，并把它转成一套可长期维护的 change-control 纪律。这对后续 Captain/Worker/Reviewer 都有意义：

- Captain 可以据此快速判断后续请求属于 `L0`、`L1`、`L2` 还是 `L3`
- Worker 会更清楚哪些改动根本不能在当前 main 上顺手做
- Reviewer 也有了统一的 source-of-truth，不必每次从头追溯全部 paper-note 历史

## 3. 为什么我给出 `PASS`

我给 `PASS`，原因不是“文档写得多”，而是“任务包要求的事情确实做到了，而且没有越界”。

具体看：

- 任务包要求的 4 份核心文档都已存在，而且不是空壳。
- `paper_frozen_mainline_handoff_packet.md` 明确保留了唯一 verdict `GO_FOR_FROZEN_MAINLINE_HANDOFF_ONLY`，没有把 handoff 写成 submission-ready completion。
- `paper_frozen_mainline_source_of_truth_map.md` 覆盖了 `FZ01-FZ05`、`BD01-BD06`，还把 theory mergeback 额外登记为 `TH01`。
- `paper_postfreeze_change_control.md` 不只是概念性分层，而是实打实写出了 `L0/L1/L2/L3` 和 10 条具体规则。
- `paper_blocked_surface_reentry_conditions.md` 覆盖了任务包要求的 blocked surface 重开条件，包含 theory mergeback 和 deployment-closure route，不只是实验面。
- 两个 README 都完成了 `T89` 登记，并明确“不升级证据等级”。

同时，我做了 allowlist-scoped 的只读检查：

- `git diff` 只显示 README 的 tracked 修改；4 份核心文档、review、explanation、worker summary 都是当前新增文件
- `git diff --check -- <allowlist>` 没有内容级错误
- 针对 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`、常见编译产物、`runs/`、`artifacts/`、`docs/evidence_packs/` 的 targeted `git status` 没有返回路径项

这说明 `T89` 的真实状态是：完成了手头文档收口任务，但没有伪造新证据，也没有把计划或 blocked surface 写成既成事实。因此它应当是 `PASS`，而不是 `PASS_WITH_WARNINGS` 或 `BLOCK`。

## 4. 对 Worker 已写 review / explanation 的判断与补充

Worker 原先写的 `docs/review/T89_review.md` 和 `docs/for_human/T89_explanation.md`，总体方向是对的：

- 结论方向对：`T89` 应是 `PASS`
- 边界判断对：它是 handoff / change-control，不是 submission-ready completion
- 覆盖面也基本够：确实提到了 `FZ01-FZ05`、`BD01-BD06`、`L0-L3` 和 blocked surface re-entry

我这里主要补了两点：

1. 把 review 表述收紧成“allowlist-scoped 事实”。
   - 因为当前 worktree 不是全仓绝对干净，所以 review 必须说明：我们判断的是 `T89` 允许范围内的变更，而不是拿 whole-repo 脏状态直接给 `T89` 定性。
2. 把与治理文档的关系说得更清楚。
   - 也就是：`T89` 为什么是 `T88` 之后必须做的一步、它和 `docs/02_experiment_plan.md` / `docs/04_task_board.md` / `docs/07_handoff.md` 的关系是什么、它对后续 Captain/Worker/Reviewer 各自意味着什么。

## 5. 一句话总结

`T89` 真正完成的，不是“把论文变得更强”，而是“把已经冻结的主线变得更好交接、更难被误改、更容易守住边界”。
