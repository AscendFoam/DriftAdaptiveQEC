# T62 解释文档

## 1. 这个任务在做什么

可以把 `T62` 理解成一次“把上次实验重新跑对”的任务。

前一轮 `T61` 的问题不是结果不好，而是“证据链不干净”。`T61` 虽然也看到了 `statcalib` 在两个锁定场景里继续领先，但它运行过程中发生了分支切换，导致：

- 启动时看到的 commit
- 结束时所在的 commit
- `summary.json` 里记录的 commit

这三个锚点对不上，所以那次结果不能作为“干净可追溯”的证据。

`T62` 的目标很窄：不改代码、不扩 benchmark、不做新结论，只把 `T61` 那个有 provenance 缺口的小型 rerun，在更严格的条件下再做一遍，确认这次证据链是闭合的。

## 2. 这次实现到底做了什么

### 任务目标

结合 `docs/04_task_board.md` 和 `docs/07_handoff.md`，`T62` 的真实目标不是“证明 statcalib 已经正式成立”，而是：

1. 复用 `T59/T61` 已经锁定的极小 smoke matrix。
2. 在 clean committed `main` 上启动。
3. 整个运行期间不切分支、不 resume、不二次启动。
4. 让 launch / finish / `summary.json` 三个 commit 锚点完全一致。
5. 在这个前提下，再看 `statcalib` 的优势是否还存在。

### 任务流程

从现有产物看，Worker 实际完成的是：

1. 在 `main` 上做 preflight 检查，记录干净启动状态。
2. 运行一个新的、唯一的 T62 run root：
   - `runs/p4_benchmark/T62_statcalib_provenance_isolated_20260527_122943`
3. 使用的矩阵保持严格锁定，没有扩范围：
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ukf`, `hybrid_residual_b`, `statcalib`
   - `--paired-seeds`
   - `--repeats 2`
4. 运行完成后核对：
   - finish branch 还是 `main`
   - finish `HEAD` 还是 `e2773d3`
   - `summary.json git_commit` 也是 `e2773d3`
   - `progress.jsonl` 没有同一 repeat key 的重复 `running`
5. 最后把结果写入这几份 T62 文档：
   - `docs/statcalib_provenance_isolated_fairness_rerun.md`
   - `docs/worker_summary/T62_worker_summary.md`
   - `docs/review/T62_review.md`
   - `docs/for_human/T62_explanation.md`

### 代码 / 配置有没有变化

没有。

这一点很关键。`T62` 是一个 execution + audit task，不允许借着“修 provenance”去偷偷改：

- 源码
- 测试逻辑
- `cnn_fpga/config/` 正式配置语义
- benchmark 范围

我核对当前 diff 时，看到的改动都还在 T62 允许的文档边界内；运行产物则落在唯一允许新增的 T62 run root 里。这符合任务包要求。

### 结果说明了什么

这次最重要的不是“数值又赢了”，而是“这次能证明这些数值确实来自同一个干净 commit”。

从 `summary.json` 和运行日志看：

- `statcalib` 在两个场景里仍然是 winner
- `statcalib_status=generated`
- `statcalib_reason=statcalib_params_emitted`
- `statcalib_generated_windows_mean=600.0`
- T62 的聚合结果和 T61 数值一致

这说明两件事同时成立：

1. `T61` 看到的强信号没有因为 provenance 收紧而消失。
2. 这次终于把 `T61` 被卡住的 provenance blocker 关掉了。

### 对后续开发的意义

这一步对后续最重要的意义，不是“可以宣布 statcalib 成功”，而是项目状态终于从：

- “结果看起来不错，但证据链不干净”

推进到：

- “至少在这个有界 smoke matrix 里，结果和 provenance 现在都能对上”

所以它能支撑的下一步只是：

- 讨论是否要开一个单独的 `FR8` gate task

它**不能**直接支撑的结论包括：

- `FR8` 已经完成
- statcalib 已经成为正式 comparator
- `.tflite` 路径已验证
- 真板 HIL 已验证

这和仓库当前的硬规则是一致的：不能把有界 software-HIL 证据写成更高级别的既成事实。

## 3. 为什么这次 review 给出 PASS

我给 `PASS`，是因为 `T62` 的任务目标本来就非常单一，而它确实完成了这个单一目标。

我主要看了四类证据：

1. 范围证据
   - 当前 diff 没有越出 T62 allowed files，也没有源码或 config 语义改动。
2. 运行边界证据
   - 只有一个 T62 run root，没有第二个同类 rerun。
3. provenance 证据
   - launch / finish / `summary.json` 的 commit 全部是 `e2773d3`。
   - `git reflog` 里也没有出现 T61 那种中途 branch movement。
4. 结果证据
   - `statcalib` 的 bounded 优势继续存在，而且产物完整，没有 `missing_runs`，没有重复 `running` 噪声。

换句话说，`T62` 不是“证明了更大的东西”，而是“把它承诺要修的那个 blocker 修掉了”。对于这样的任务，这就足够 `PASS`。

## 4. Worker 已写文档有没有问题，是否需要补充

### 对 Worker review 的判断

Worker 已写的 `docs/review/T62_review.md` 方向基本是对的。

它抓住了最关键的判断点：

- 这次不是看谁赢，而是看 provenance 是否闭合
- `T62` 确实把 `T61` 的 blocker 关掉了
- 这仍然不是 `FR8`

我补充的部分主要是两点：

1. 把“为什么这次可以从 `T61=BLOCK` 变成 `T62=PASS`”写得更明确。
2. 把“下一步只能是 `FR8` gate discussion，不是自动升级结论”写得更硬一些，避免后续误读。

### 对 Worker explanation 的判断

Worker 已写的 `docs/for_human/T62_explanation.md` 没有明显错误，但太短。

它能概括结论，却不够回答下面这些人类读者真正关心的问题：

- `T62` 到底是为了解决什么历史问题？
- 为什么这次不改代码反而是对的？
- 为什么结果和 T61 一样反而不是问题？
- 为什么 `PASS` 以后仍然不能直接写成 `FR8`？

所以我把它扩成了这份更完整的解释文档。

## 5. 一句话总结

`T62` 的价值不在于“又跑出一次 statcalib 很强”，而在于“这次终于把这份结果和一个单一、干净、可追溯的 commit 绑在了一起”。这就是它能通过 review、但又仍然不能被写成正式 comparator 结论的原因。
