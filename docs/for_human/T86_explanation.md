# T86 任务与 Review 说明

## 1. 先用通俗的话解释这个任务

`T86` 不是补实验，也不是把论文直接做成“已经可以投稿”的最终包。

它做的是一件更靠近投稿前整理纪律的事：

- 把当前主线材料里哪些能进 submission-facing package 说清楚；
- 把哪些只能进 appendix / supplement 说清楚；
- 把哪些现在必须继续排除、绝不能顺手写强的 surface 单独登记出来；
- 再给后续作者一份 handoff，说明下一步人工终修时能做什么、不能做什么。

所以，`T86` 的真正作用不是“新增结论”，而是“把当前已有结论如何被诚实地装配起来”写成可审计事实。

## 2. 这个任务具体做了什么

结合 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md`，`T86` 是在 `T85` 完成 submission-readiness preflight 后被打开的当前唯一任务。

`T85` 已经回答了：

- 当前主线 note 的 residual wording-lag 是否清掉了；
- 是否允许打开下一张 bounded submission-pack assembly 任务；
- 哪些 blocker 仍然存在。

因此 `T86` 不再重复做 preflight，而是继续向前推进一小步：把已经允许的 submission-facing assembly 真正组织成一套装配规则。

本轮实际新增了 4 份台账：

1. `paper_submission_pack_assembly_manifest.md`
2. `paper_submission_surface_route_map.md`
3. `paper_submission_exclusion_register.md`
4. `paper_submission_author_handoff.md`

同时，主 note 只做了非常小范围的装配导向刷新，实际触达的 section 只有：

1. `Numerical Results`
2. `Discussion`
3. `Conclusion`

并加上了：

- `% T86-ASSEMBLY: Numerical Results`
- `% T86-ASSEMBLY: Discussion`
- `% T86-ASSEMBLY: Conclusion`

### 2.1 四份新文档分别在解决什么问题

#### `paper_submission_pack_assembly_manifest.md`

这份清单回答的是：

- 当前 submission-facing package 里到底有哪些 surface 可以进；
- 每个 surface 的角色是什么；
- 它来自哪份已有材料；
- 证据锚点在哪里；
- 作者后续对它只允许做什么人工装配动作。

换句话说，它不是“成果清单”，而是“可装配 surface 清单”。

#### `paper_submission_surface_route_map.md`

这份路由图回答的是：

- 哪个 claim 或 section 应该放在 main text；
- 哪些只能放 appendix；
- 哪些只能放 supplement；
- 哪些必须明确写成 exclusion。

它的价值很直接：防止后续写作时把本来只能作为 supporting / supplement / blocked 的东西抬进主文。

#### `paper_submission_exclusion_register.md`

这份表把当前必须继续排除的 surface 单独拎出来，例如：

- real-board execution / timing / resource
- default-env / cross-host `.tflite` portability
- full training reproducibility
- `statcalib` mature comparator promotion
- expanded benchmark story
- unified portability/deployment closure figure or prose

每一项都同时写了：

- 为什么现在不能进；
- 绝对不能怎么写；
- 未来如果真要解锁，需要开什么新任务。

#### `paper_submission_author_handoff.md`

这份文档是给后续作者看的操作说明。

它集中回答四件事：

1. 当前 package 已经具备哪些内容；
2. 哪些 surface 还必须继续排除；
3. 作者下一步仍可以做哪些 bounded manual editorial action；
4. 哪些 claim 绝对不能写强。

这比单纯给一份 manifest 更重要，因为它把“写作自由度”也一起约束住了。

### 2.2 note 里具体改了什么

这轮不是重写正文，而是只增强了 route / exclusion 过渡句。

几个关键变化是：

- `Numerical Results` 更明确地区分：
  - 锁定 benchmark 主结果层；
  - 保守解释层；
  - appendix / supplement / excluded surfaces 的分流边界。
- `Discussion` 更明确地告诉读者：
  - submission-facing assembly 里主文只能保留冻结结果和保守解释；
  - supporting tables 应该通过显式 route 下沉到 appendix 或 supplement；
  - blocked hardware / portability surfaces 继续留在 package claim 之外。
- `Conclusion` 更明确地把下一步定义成：
  - bounded submission-facing assembly；
  - 而不是 deployment closure、submission-ready completion 或 hardware-ready finalization。

## 3. 这个实现对后续开发意味着什么

从项目路线看，`T86` 的意义是把主线从“是否允许进入 submission-facing assembly”推进到“assembly 应该怎么做、哪里必须停下”。

也就是说，当前主线已经不再主要缺：

- ready sections prose；
- methods / contribution calibration；
- supporting-boundary closeout；
- full-note consistency gate；
- reader-facing final polish；
- submission-readiness preflight。

当前主线缺的是更细的一层 discipline：

- 现有材料在投稿导向的包里如何被放置；
- 哪些边界必须显式排除；
- 作者手工继续修改时，哪些动作属于合法装配，哪些动作会越界。

这对后续开发和写作的直接意义是：

- 如果 Captain 继续推进，下一张任务不应该再是“扩大写作范围”，而应该是更小的作者终检 / 投稿前 QA；
- 后续任何装配工作都应以这 4 份台账为准，而不是靠作者临场判断；
- real-board、`.tflite` portability、training full reproducibility、`statcalib` promotion 这些内容，仍然必须作为 exclusion / blocker 被显式保留。

## 4. 为什么我的 review 结果是 PASS

我给 `PASS`，是因为 `T86` 的任务目标已经被真实完成，而且没有越界。

主要依据是：

1. 任务包要求的 4 份核心台账都已创建：
   - `paper_submission_pack_assembly_manifest.md`
   - `paper_submission_surface_route_map.md`
   - `paper_submission_exclusion_register.md`
   - `paper_submission_author_handoff.md`
2. note 中实际修改的 3 个 section 都有 `% T86-ASSEMBLY: ...` 标记。
3. `% T80-REOPEN` 到 `% T85-PREFLIGHT` 的旧标记链都仍然保留。
4. 两个 README 都登记了 `T86` 的入口和使用边界。
5. 本地 `latexmk -g` 编译成功，日志也没有扫出常见 warning 关键字。
6. 最关键的是：所有新增文本都把 `T86` 写成“submission-facing assembly / exclusion 收口”，没有把它写成“submission-ready pack 已完成”。

## 5. 为什么我没有给 PASS_WITH_WARNINGS 或 BLOCK

我没有给 `BLOCK`，因为没有发现这些问题：

- 没有伪实现、mock、stub、hardcode；
- 没有把 blocked surface 写强；
- 没有把计划写成已经完成的事实；
- 没有越出 allowlist 去改治理文档、源码、历史证据或实验产物。

我也没有把 verdict 降成 `PASS_WITH_WARNINGS`，因为本轮虽然存在过程性噪声，但没有形成内容级问题：

- 当前 worktree 较脏，导致相对 `HEAD` 的 diff 仍混有部分前序 `T84/T85` 未提交内容；
- `git status` / `git diff --check` 仍会出现 Windows 下的 `LF -> CRLF` 提示和 `git/ignore` 读取告警；
- 这些都是宿主机和工作区状态噪声，不是 `T86` 文档本身的错误。

因此，这些更适合被记录成 non-blocking operational reminders，而不是下调任务结论。

## 6. Worker 已写的 review / explanation 是否有问题

整体上，Worker 已写的 `T86_review.md` 和 `T86_explanation.md` 判断方向是对的：

- 它们正确把 `T86` 定义成 docs-only、mainline-only、assembly-only；
- 也正确保留了 `.tflite`、real-board、training/material、`statcalib` 等边界；
- 对 4 份新台账的功能描述基本准确。

我这里主要补充了两点：

1. 更明确地区分 “assembly 完成” 和 “submission-ready 完成”。
这是这轮最容易被读歪的地方，所以我在 review 和 explanation 里都单独强调了。

2. 更明确地点出当前 diff 边界为什么需要谨慎解释。
因为当前工作区里仍混有前序 `T84/T85` 的未提交内容，所以这轮审查不能机械地把相对 `HEAD` 的整段 diff 当成纯 `T86` 范围，而必须结合当前文件内容和 `% T86-ASSEMBLY` 标记链判断。

## 7. 推荐的下一步

如果 Captain 接受 `T86`，合理的下一步不应该是“直接宣布可投稿”，而应该是两种更小的选择之一：

1. 开一张更小的作者终检 / 投稿前 QA 任务。
2. 如果 Captain 认为当前仍不值得继续推进，则继续保持 `NO_GO_SUBMISSION_READY_COMPLETION`。

无论选哪种，都不应再做这些事情：

- 重开 benchmark；
- 升级 `.tflite` portability；
- 升级 real-board execution；
- 把 `FR8/statcalib` 写成成熟主线 comparator；
- 把当前主线回述成 submission-ready pack 已完成。

换句话说，`T86` 已经把“如何诚实装配当前论文材料”这件事固定下来了，但它并没有把仓库带到“可直接投稿”的完成态。
