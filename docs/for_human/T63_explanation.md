# T63 解释文档

## 1. 这个任务在做什么

`T63` 不是去跑一个新的 `statcalib` 实验，而是做一个“能不能开下一个实验”的闸门判断。

更具体地说，它要回答的问题是：

- 经过 `T59` 到 `T62` 这一串工作后，`statcalib` 这条 comparator lane 的前置阻塞项是不是已经清干净了？
- 如果已经清干净，那么下一步能不能开一个真正的 `FR8` 结果表任务？
- 如果还没清干净，那么是不是还要先补一个更小的 prerequisite？

所以，`T63` 的本质是一个 docs-only gate review，不是 benchmark task，也不是 paper claim 升级任务。

## 2. 这个任务的实现细节

### 任务目标

结合 `docs/04_task_board.md`、`docs/07_handoff.md` 和 `docs/08_risks_and_open_questions.md`，`T63` 的目标很明确：

1. 复用已有证据，不跑任何新实验。
2. 明确哪些事情已经被 `T59/T60/T61/T62` 证实了。
3. 明确哪些事情还没有被证实，尤其是不能越界写成“正式 comparator 结论”的部分。
4. 对 `R27` 做最终判断。
5. 在 `GO_FOR_BOUNDED_FR8_TASK` 和 `NO_GO_NEEDS_ONE_MORE_PREREQUISITE` 两个结论中，给出且只给出一个。

### 实际做了什么

从当前 diff 看，Worker 这次只做了文档工作：

1. 写了 gate report：
   - `docs/fr8_statcalib_comparator_gate_review.md`
2. 写了 review：
   - `docs/review/T63_review.md`
3. 写了给人的说明文档：
   - `docs/for_human/T63_explanation.md`
4. 写了 worker summary：
   - `docs/worker_summary/T63_worker_summary.md`
5. 把 Worker Output 补回任务包：
   - `docs/tasks/Phase2/T63_fr8_statcalib_comparator_gate_review.md`

这符合任务包的边界要求，因为 `T63` 明确禁止：

- 改源码
- 改测试
- 改 `cnn_fpga/config/`
- 新建或改写 `runs/` / `artifacts/`
- 跑 benchmark、smoke、training、`.tflite`、real-board、cleanup

### 代码变化 / 配置变化

没有代码变化，也没有配置变化。

这不是缺点，反而正是 `T63` 应该做到的事。因为它的职责不是“实现 statcalib”，也不是“跑 FR8”，而是先判断仓库有没有资格去开那个下一步任务。

### 它参考了哪些历史事实

这次 gate review 依赖的关键链条是：

1. `T26`
   - 证明 `statcalib` 可以作为 separate comparator lane 被讨论，但只能是条件性可行，不可静默塞进主线 benchmark。
2. `T30`
   - 只完成 interface contract 和 focused tests，不等于 integrated comparator evidence。
3. `T59`
   - 第一次把 `statcalib` 接进独立 smoke lane，但当时有 dirty-worktree provenance 弱点，而且结果强得不够放心。
4. `T60`
   - 修掉 cross-mode leakage 和回归硬化问题，关闭 `R26`。
5. `T61`
   - 试图做 clean-provenance rerun，但因为运行期间 branch movement 导致失败，被判 `BLOCK`。
6. `T62`
   - 在 clean `main` 上做了一次 provenance-clean bounded rerun，最终关闭 `R27`。

于是 `T63` 要判断的就是：

- 现在是不是已经从“前置 blocker 还没清理完”推进到了“可以正式开一个 bounded FR8 extension-lane task”。

### 为什么它给出的是 GO

`T63` 最重要的结论是：

- `GO_FOR_BOUNDED_FR8_TASK`

这里的 `GO` 不是说：

- `FR8` 已经完成
- `statcalib` 已经被证明是正式 comparator
- 当前仓库已经有 paper-grade comparator evidence

这里的 `GO` 只表示：

- 开启下一步 `FR8` 任务的前置 blocker 已经被清掉了
- 剩下的问题不再是“要不要先补前置条件”，而是“去做那个 bounded FR8 task 本身”

这是一个很重要的区别。

## 3. 对后续开发的意义

`T63` 的意义是把项目状态从：

- “还在修 `statcalib` lane 的基本可信度问题”

推进到：

- “基本可信度问题已经收口，可以考虑做正式但仍有界的 extension-lane comparator result-table task”

从 `docs/P4_benchmark_formal_protocol.md` 来看，下一步最小安全范围已经相当清楚：

1. 保持四个 frozen scenarios：
   - `static_bias_theta`
   - `linear_ramp`
   - `step_sigma_theta`
   - `periodic_drift`
2. 保持五个 frozen ranked modes 不变：
   - `ekf`
   - `ukf`
   - `constant_residual_mu`
   - `rls_residual_b`
   - `hybrid_residual_b`
3. `statcalib` 只能作为 separately labeled extension lane 加进去。
4. 保持 `paired-seeds` 和 `repeats=2`。
5. 继续要求 provenance-clean run。
6. 继续明确写清：
   - mock-backed software-HIL only
   - not `.tflite`
   - not real-board

也就是说，`T63` 的作用不是把边界放松，而是把“下一步允许做多大”定义得更明确。

## 4. 为什么这次 review 给出 PASS

我给 `PASS`，因为在 T63 这个任务自己的边界内，它完成得是成立的。

我主要看了三件事：

1. 它有没有越界
   - 没有。当前 diff 只落在 T63 allowed files 里，没有源码、配置、运行产物修改。
2. 它有没有把计划写成事实
   - 没有。它明确写了 `T63` 不是 `FR8`，当前证据仍然只是 mock-backed software-HIL，`.tflite` 和 real-board 仍然不在当前证据内。
3. 它的 gate reasoning 是否和仓库里的既有证据一致
   - 一致。`R27` 已经被 `T62` 关闭；`R24` 仍然开放；因此当前最合理的下一步不是再开一个抽象前置任务，而是开一个边界更清楚的 FR8 extension-lane task。

换句话说，`T63` 的结论并不激进，它只是把“是否可以进入下一步”这件事说清楚了。

## 5. Worker 已有文档有没有问题

### 对 Worker review 的判断

Worker 已有的 `docs/review/T63_review.md` 方向是对的：

- 它正确判断了这是 docs-only task。
- 它正确判断了 `R27` 已被 `T62` 关闭。
- 它正确保留了 `R24` 和 software-HIL 边界。

我补充的主要是结构和完整性：

- 按你要求补齐更标准的 review 输出结构
- 把 `Missing tests`、`Suspicious implementation details`、`Recommended next action` 明确写出来
- 把“为什么是 PASS，而不是已经完成 FR8”说得更清楚

### 对 Worker explanation 的判断

Worker 原有的 `docs/for_human/T63_explanation.md` 没有明显事实错误，但偏短。

它能说明结论，却还不够解释：

- 为什么这个任务不需要改代码
- 为什么 `GO` 不等于 `FR8 已完成`
- 为什么 `R24` 还开着却仍然可以进入下一步
- 下一步的 FR8 task 到底应该被限制在什么边界内

所以我把这部分补成了更完整的说明。

## 6. 一句话总结

`T63` 通过 review，不是因为它已经产出了 `FR8`，而是因为它诚实地证明了：`statcalib` 这条 lane 的前置 blocker 已经清到足以开启一个有界的 `FR8` extension-lane task，但还远远没有到可以把 `FR8` 当成既成事实的程度。
