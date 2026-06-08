# T68 任务与评审说明

## 1. 这个任务在做什么

`T68` 要回答的是一个很具体的问题：

在前面的 `T64/T66/T67` 里，`statcalib` 这条 extension lane 已经显示出它有潜力，也已经证明它不只是依赖 `teacher_mode=ukf`。但还有一个关键疑问没有彻底回答：

能不能在一个预先写死的小网格里，找到一种 `statcalib` 候选，它同时满足两件事：

1. 在四个锁定场景里都赢过两个冻结锚点：
   - `ukf`
   - `hybrid_residual_b`
2. 在四个场景里都保持 `statcalib_status = generated`，也就是没有 `mixed` 行。

通俗地说，`T68` 不是在问“`statcalib` 强不强”，而是在问：

“在当前这套受控边界里，能不能找到真正干净、四场景全 generated 的赢家？”

这次结果给出的答案是：能，而且不止一个。

## 2. 这个任务是怎么实现的

### 2.1 任务目标

结合 `docs/04_task_board.md`、`docs/07_handoff.md` 和 `docs/08_risks_and_open_questions.md`，`T68` 的位置很清楚：

1. `T64` 先证明了 extension lane 里有 clean provenance 的 bounded 胜利。
2. `T66` 先证明这种结果不是某个单点 threshold 偶然撞出来的。
3. `T67` 再证明这种结果不主要依赖 `ukf teacher`。
4. `T68` 接着问：既然非 `ukf` teacher 也成立，那在小范围 threshold 网格里，是否存在真正全 generated 的 clean winner。

所以，`T68` 不是新算法开发任务，也不是大规模 benchmark 扩张任务。它是一个非常窄、非常有针对性的验证任务。

### 2.2 实现流程

这次实现主要分成四块。

第一块是新增一个 task-scoped config：

- `cnn_fpga/config/p4_multiscenario_statcalib_generated_only.yaml`

这个配置固定了整套矩阵：

1. 场景仍然是 4 个锁定场景：
   - `static_bias_theta`
   - `linear_ramp`
   - `step_sigma_theta`
   - `periodic_drift`
2. 冻结锚点仍然是 2 个：
   - `ukf`
   - `hybrid_residual_b`
3. `statcalib` 候选固定成 8 条：
   - `window_variance` teacher 下 4 个 threshold
   - `ekf` teacher 下 4 个 threshold
4. `repeats=2`
5. `paired_seeds=true`

总矩阵就是：

- `4 scenarios x 10 modes x 2 repeats = 80 repeat-runs`

第二块是真实 benchmark 运行。

Worker 没有改：

1. `cnn_fpga/decoder/statcalib.py`
2. `cnn_fpga/runtime/slow_loop_runtime.py`
3. `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

也就是说，这次不是靠改语义“做出结果”，而是用现有主线语义，在任务包允许的范围内加一个新矩阵去验证。

为了满足“从 clean committed main 启动”的要求，这次不是直接在当前工作区里跑，而是从一个干净短路径 clone 启动：

- `C:\t68cf2b`

最后结果被写入唯一一个新的 run root：

- `runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`

第三块是新增一个 task-scoped summary helper：

- `cnn_fpga/benchmark/summarize_statcalib_generated_only.py`

它的作用不是做通用 benchmark 基建，而是专门服务 T68。这一点很重要。

它会检查：

1. `missing_runs` 是否为空
2. `requested_scenarios` 是否正好等于任务包锁定的 4 个场景
3. `requested_modes` 是否正好等于任务包锁定的 10 个 mode
4. `paired_seeds` 是否保持开启
5. `repeats` 是否仍然是 2
6. `coverage` 是否都是 `1.0`
7. `completed_repeats` 是否都是 `2`
8. 是否存在重复 comparison row

然后它会生成：

1. 每个场景下的最佳候选
2. 每个候选的平均 LER、最差场景 LER、最佳场景 LER
3. 每个候选的 `generated` 行数和 `mixed` 行数
4. 每个候选在多少个场景里同时击败两个冻结锚点
5. teacher-anchor 与 threshold 的分组比较
6. 显式 tie 表示
7. Pareto 风格汇总
8. 如果没有 full generated-only winner，则给出 nearest miss

第四块是 focused test：

- `tests/test_statcalib_generated_only_summary.py`

这些测试覆盖了：

1. grouped outputs 是否生成
2. tie 是否被显式表达
3. full generated-only winner 是否被识别
4. teacher-anchor 内部的 threshold 排序与 monotonicity
5. no-full-winner 时 near miss 是否正确给出
6. incomplete matrix rejection
7. wrong mode set rejection

### 2.3 这次结果说明了什么

这次最重要的结果是：

确实存在 full generated-only winner。

而且不是只有一个，而是有 4 个：

1. `statcalib_window_variance_t001`
2. `statcalib_window_variance_t003`
3. `statcalib_window_variance_t005`
4. `statcalib_ekf_t001`

它们都满足：

1. 四个场景全是 `generated`
2. 四个场景都击败 `ukf`
3. 四个场景都击败 `hybrid_residual_b`

其中最强的一组 clean winner 是：

1. `statcalib_window_variance_t001`
2. `statcalib_window_variance_t003`
3. `statcalib_window_variance_t005`

这三条在 mean LER 上完全打平，并且都优于 `ekf_t001`。

如果换成 worst-case 视角，那么还要再加上一条：

4. `statcalib_window_variance_t010`

但这条虽然 worst-case 不差，仍然有 `mixed` 行，所以不能把它和前三条混成同一种“最干净赢家”。

### 2.4 对后续开发意味着什么

`T68` 对后续最重要的意义有三点。

第一，`R24` 又被收窄了一步。

在 `T67` 之前，`R24` 里还有一种担心是：

“也许整个 bounded statcalib lane 根本就找不到全 generated 的 clean winner。”

`T68` 现在把这个怀疑排除了。也就是说，后面再讨论 `R24`，就不能再说“也许一个 clean winner 都没有”。

第二，它保护了 `T24` 的主表权威。

这次虽然结果很正面，但它没有动 `T24` 的冻结五模式主表，也没有把 `statcalib` 静默塞回主表。这个边界非常关键，因为当前 repo 的核心纪律就是：

1. 历史冻结表不能被后续 extension lane 偷偷改写
2. 新 evidence 必须带着自己的 scope label

第三，它给后续决策留下了更明确的问题形状。

现在后续真正的问题不再是“有没有 generated-only winner”，而更像是：

1. 在已有多个 clean winner 的前提下，要不要继续做唯一阈值选择
2. 如果要选，标准是 mean-best、worst-case-best，还是别的稳定性标准
3. 这些工作值不值得继续做，还是已经足够作为 bounded extension-lane 结论停在这里

## 3. 为什么我的 review 结果是 PASS_WITH_WARNINGS

### 3.1 为什么不是 BLOCK

因为任务主体确实完成了，而且证据链是完整的。

我实际复核到的关键点包括：

1. 只有一个新的 T68 run root
2. `launch HEAD == finish HEAD == summary.json["git_commit"] == bda8f2b`
3. `comparison.csv` 有 40 行，正好对应 `4 x 10`
4. `progress.jsonl` 中：
   - `running = 80`
   - `completed = 80`
   - duplicate `running = 0`
   - duplicate `completed = 0`
5. 我实际重跑了轻量验证：
   - `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_generated_only.py`
   - `python -m unittest tests.test_statcalib_generated_only_summary`
   - `python -m cnn_fpga.benchmark.summarize_statcalib_generated_only --run-dir runs/p4_benchmark/T68_statcalib_generated_only_20260605_205723`
6. 没有发现对 `statcalib`、runtime、benchmark runner 主语义的越界修改。

这些都说明：

1. 不是伪实现
2. 不是手工填表
3. 不是 mock summary 冒充真实运行
4. 也不是靠改 benchmark 语义把结论“做出来”

### 3.2 为什么不是无保留的 PASS

因为这次虽然完成得不错，但仍然有三类 warning 要保留。

第一类 warning：结果不是唯一 winner，而是 tie set。

这次最强 clean winner 不是一个单点，而是：

1. `window_variance_t001`
2. `window_variance_t003`
3. `window_variance_t005`

并列最优。

这意味着后续如果引用 T68，不能说成：

“已经找到唯一最佳 threshold”

最多只能说：

“在当前 bounded grid 里，最强 clean winners 是一个 tie set”

第二类 warning：虽然存在 clean winners，但整个候选网格并不全干净。

依然存在 `mixed` 的候选：

1. `statcalib_window_variance_t010`
2. `statcalib_ekf_t003`
3. `statcalib_ekf_t005`
4. `statcalib_ekf_t010`

所以，T68 回答的是“存在性问题”，不是“整个参数网格都已经 clean”。

第三类 warning：证据等级没有升级。

这次的边界仍然是：

1. mock-backed software-HIL
2. extension lane
3. separate comparator label

它仍然不是：

1. `.tflite`
2. real-board
3. mature calibration comparator
4. paper-grade expanded benchmark

所以最合适的 verdict 仍然是 `PASS_WITH_WARNINGS`，而不是把它写成“已经万事俱备”。

## 4. 对 Worker 已有 review 和 explanation 的复核

Worker 已经写了 review 和 explanation 草稿，整体方向基本是对的。

我这次主要做了三件补充。

第一，我把 reviewer 口径写得更清楚。

尤其是把下面三点明确落成 warning：

1. tie set 不能被压扁成唯一 winner
2. mixed candidates 仍然存在
3. 证据等级仍然只是 mock-backed software-HIL extension lane

第二，我把任务在整条链上的位置讲清楚了。

也就是把 `T64 -> T66 -> T67 -> T68` 这条线重新串起来，让人一眼看出：

1. T64 解决了什么
2. T66 解决了什么
3. T67 解决了什么
4. T68 又往前推进了哪一个更细的问题

第三，我补强了“后续开发意义”的部分。

因为 T68 的价值不只是“又多了一次 benchmark”，而是它改变了后续问题的形状：

后面如果还想继续推进，就不该再问“有没有 clean winner”，而该问“多个 clean winners 里还要不要继续分辨，为什么分辨，边界在哪里”。

## 5. 一句话总结

`T68` 的核心贡献不是把 `statcalib` 升级成主线赢家，而是把一个重要疑问排除了：

在当前 bounded、mock-backed software-HIL 的 extension lane 里，确实存在四场景全 generated、并且同时赢过两个冻结锚点的 clean winner，而且最强 clean winners 来自 `window_variance` teacher-anchor 下的 `t001/t003/t005` tie set。
