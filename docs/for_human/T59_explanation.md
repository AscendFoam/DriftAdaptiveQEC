# T59 解释文档

## 1. 这个 task 通俗上是在做什么

T59 可以把它理解成一句话：

把一个叫 `statcalib` 的新比较器，先以“单独车道”的方式接进现有慢环运行时和 benchmark 流程里，并做一次很小、很受控的冒烟验证，确认它真的能跑通，但暂时不把它写成正式结论。

这里最重要的不是“证明 statcalib 已经赢了”，而是先回答下面几个基础问题：

- 它是不是一个独立 mode，而不是偷偷改写老的 `T24` / `hybrid_residual_b` 语义。
- 它能不能从 runtime 一路把自己的 `status / reason / provenance` 带到最终 benchmark 输出。
- 它有没有最基本的 smoke 证据，证明不是只写了接口和字段，却没有真正跑起来。

所以，T59 的定位是“把 comparator lane 接通”，不是“给论文定输赢”。

## 2. 这次实现具体做了什么

### 2.1 任务目标

结合 `docs/02_experiment_plan.md`、`docs/04_task_board.md` 和 `docs/07_handoff.md`，T59 的真实目标很明确：

- 延续 T30 的 `statcalib` 接口工作。
- 但不满足于“只有 contract、没有 runtime 集成”。
- 在不破坏主线已冻结 benchmark 语义的前提下，把 `statcalib` 作为独立 comparator lane 接入。
- 做一次严格受限的 smoke，补齐“接口存在”到“端到端可运行”之间的证据。

这一步对后续开发的意义是：

- 后面如果要做更正式的 `FR8` comparator 结果表，仓库里终于有一个真正可运行、可统计、可输出状态字段的 `statcalib` lane。
- 以后讨论 `statcalib` 时，不再只是“理论上可以接”，而是“已经按受控范围接入过一次，而且 smoke 跑过”。

### 2.2 代码变化

这次核心变化集中在 4 个地方。

1. `cnn_fpga/decoder/statcalib.py`

- 保留并扩展了 `StatCalibInput` / `StatCalibOutput` 合同。
- 新增 `summarize_histogram()`，把原始 histogram 压缩成适合写入 metadata 的摘要。
- 新增 `run_statcalib_estimator()`。

这个估计器不是神秘模型，也不是假的空实现。它做的是一个非常朴素但明确的动作：

- 读取窗口统计里的 `mean_syndrome_q` 和 `mean_syndrome_p`。
- 形成一个二维信号。
- 乘以 `residual_scale_b`。
- 再按 `residual_clip_b` 裁剪。
- 把这个 `delta_b` 加到 teacher 映射出来的参数上。

如果窗口无效、histogram 没质量、或者信号太小，就不产出参数，而是返回 `not_generated`。

2. `cnn_fpga/runtime/slow_loop_runtime.py`

- 增加了新的 `slow_loop.mode = statcalib` 分支。
- 运行时会先走 teacher 分支，得到 `teacher_prediction` 和 `teacher_params`。
- 然后把窗口 histogram、窗口统计、窗口诊断信息，组装成 `StatCalibInput`。
- 再调用 `run_statcalib_estimator()`。

如果 `statcalib` 产出成功：

- 返回新的 `DecoderRuntimeParams`。
- metadata 里会带上 `statcalib_status`、`statcalib_reason`、`statcalib_provenance`、`applied_delta_b`、`statcalib_input`、`statcalib_output` 等信息。

如果没有产出：

- 回退到 `teacher_params`。
- 同时明确写出 `statcalib_fallback = teacher_params`。

这意味着 runtime 不只是“知道 statcalib 存在”，而是已经有独立的成功路径、失败路径和回退路径。

3. `cnn_fpga/benchmark/run_hil_suite.py`

- 新增 `statcalib` 诊断聚合逻辑。
- 会统计每个 run 里 `generated / not_generated / not_applicable / diagnostic_error` 这类状态的出现情况。
- 最终写入 `hil_summary.json` 的 `statcalib_diagnostics` 字段。

4. `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`

- 把 `statcalib_status`、`statcalib_reason`、`statcalib_generated_windows`、`statcalib_signal_norm_mean` 等字段接到 per-repeat、comparison、summary、report 输出链路中。
- 因此 `comparison.csv`、`summary.json`、`report.md` 里现在都能直接看到 `statcalib` 的状态，而不是只知道它“跑过一个 mode 名字”。

### 2.3 配置和验证变化

新增配置：

- `cnn_fpga/config/p4_multiscenario_statcalib_smoke.yaml`

这个配置做的是 task-scoped 小矩阵：

- scenarios: `static_bias_theta`, `linear_ramp`
- modes: `ukf`, `hybrid_residual_b`, `statcalib`
- repeats: `1`
- `--paired-seeds`

新增测试：

- `tests/test_statcalib_runtime_smoke.py`

它至少覆盖了两条关键路径：

- `statcalib` 成功产出参数时，metadata 和 `applied_delta_b` 是否正确。
- `statcalib` 信号太弱时，是否按预期回退到 `teacher_params`。

我额外复核并实跑了 Worker 声称的最小验证：

- `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_interface tests.test_statcalib_runtime_smoke`
- `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/decoder/statcalib.py cnn_fpga/runtime/slow_loop_runtime.py cnn_fpga/benchmark/run_hil_suite.py cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py tests/test_statcalib_interface.py tests/test_statcalib_runtime_smoke.py`

这两项都通过了。

### 2.4 这次 smoke 说明了什么

run root 是：

- `runs/p4_benchmark/t59statc_20260526_211532_3a3d00_23740`

我复核到的事实包括：

- `launch_plan.json` 的矩阵和 task 包要求一致，没有偷偷扩大范围。
- `comparison.csv` 里确实有单独的 `mode=statcalib` 行。
- 两个 scenario 的 `coverage` 都是 `1.0`。
- `summary.json` 和各自的 `hil_summary.json` 里都能看到 `statcalib_status` / `statcalib_reason`。
- 对非 statcalib mode，状态是 `not_applicable`，说明它没有反向污染旧 mode 的输出语义。

但这里也有一个很重要的解读边界：

这次 smoke 里 `statcalib` 的结果明显强于 `ukf` 和 `hybrid_residual_b`，而且是一个很小的、规则很硬的 lane。这个现象不能直接被当成正式结论，它更像是在提醒我们：

- 这个 lane 的信号定义可能非常强。
- 它的比较口径可能还需要更严格的公平性审查。
- 在做正式 `FR8` 之前，必须先搞清楚它为什么在小 smoke 上会强成这样。

## 3. 为什么我的 review 结论是 PASS_WITH_WARNINGS

我没有给 `BLOCK`，因为 T59 的任务目标基本都完成了：

- `statcalib` 的确被作为独立 lane 集成进 runtime。
- 没有把旧 `T24` 语义直接重写掉。
- 有 focused tests。
- 有 task-scoped config。
- 有真实 bounded smoke。
- 状态字段确实端到端落到了最终产物里。

我也没有给纯 `PASS`，因为还存在几条 reviewer 应该明确保留的警告。

第一类警告是“代码边界还不够干净”：

- `SlowLoopRuntimeConfig.from_config()` 里，`statcalib.teacher_mode` 现在被放进了通用 `teacher_mode` 回退链。T59 自己这次没被这个问题咬到，但它确实增加了未来混合配置时的跨 mode 耦合风险。

第二类警告是“证据已经有了，但解释还不能放松”：

- 这次实现不是 stub，也不是 mock 字段冒充完成态。
- 但它的估计器是一个非常直接的硬规则：用窗口均值 syndrome 直接生成 `delta_b`。
- 在小 smoke 中，它 600/600 个窗口都进入 `generated`，而且最终 snapshot 里已经能看到 clip-bound 的迹象。
- 这不构成否定，但足以说明正式 comparator 结论之前必须先做公平性和稳健性审查。

第三类警告是“文档和产物细节还需要更严谨”：

- `docs/statcalib_comparator_lane_smoke.md` 对 `hil_summary.json` 里的字段名写得不够准确。
- 当前 run artifact 记录了 `git_commit = a40adca`，但因为这次评审看的是未提交 diff，这个 commit hash 本身还不能单独唯一标识 T59 代码状态，产物 provenance 还不算最强。

所以，`PASS_WITH_WARNINGS` 的含义是：

T59 作为“受控接入 + bounded smoke”任务可以接受；
但它还没有把 `statcalib` 送到可以直接写正式结果表的状态。

## 4. 如果 Worker 已经写了 review / explanation 文档，有没有问题

我检查了仓库里已有的两个草稿。

1. `docs/review/T59_review.md`

- 方向基本对。
- 它正确看到了这是一个 integration-complete 但还不是 formal evidence 的任务。
- 但它漏掉了两点更细的风险：
  - `teacher_mode` 回退链的跨 mode 耦合风险。
  - smoke 文档字段名和真实 JSON 字段不完全一致。

所以我保留了它的大方向，但补强了 reviewer 应该留下的技术边界。

2. `docs/for_human/T59_explanation.md`

- 现有草稿有明显编码损坏，属于 mojibake，已经不适合作为给人的解释文档。
- 我这次等于重写了一版可读、可追溯、和当前 task 边界一致的说明。

## 5. 我建议的后续动作

最合理的下一步不是直接跳去更大的 formal comparator 任务，而是先做一个有界 follow-up：

1. 把 `teacher_mode` 的回退链重新收紧，避免 `statcalib` 配置对其他 mode 产生隐式影响。
2. 增补更直接的单元测试，特别是 `not_generated`、clip 边界、聚合字段回归。
3. 做一次小范围 fairness / robustness sanity check，先解释清楚为什么这个极简 lane 在小 smoke 上会明显领先。

只有这一步完成之后，再进入更正式的 `FR8` 比较任务，证据链才会更稳。
