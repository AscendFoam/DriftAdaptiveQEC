# T69 任务与这次 Review 的解释

## 1. 先用大白话解释 T69 在做什么

`T68` 已经回答了一个问题：在一组预先锁死、不允许事后挑数的 statcalib 候选里，确实存在“全程都是 generated、而且比两个冻结锚点更好”的 clean winner。

但 `T68` 没有回答完另一个更细的问题：最强的 clean winner 不是一个点，而是三个候选打平：

1. `statcalib_window_variance_t001`
2. `statcalib_window_variance_t003`
3. `statcalib_window_variance_t005`

所以 `T69` 的任务不是继续扩大搜索，也不是去做 `.tflite`、真板、训练链，而是只问一个很窄的问题：

- 把重复次数从较弱预算提高到 `repeats=4` 以后，这个三路平局会不会自己塌缩成一个唯一赢家？

如果会，仓库就能诚实地说“在这个有界条件下，唯一 clean reference point 出现了”。

如果不会，也要诚实地说“最强 clean answer 仍然是 tie set，不能硬说成唯一阈值”。

T69 的结果是后者：平局没有塌缩。

## 2. 这次实现具体做了什么

### 2.1 任务目标

T69 的目标很明确：

1. 保持 `T24` 主表冻结不动
2. 保持 `statcalib` 只是 extension lane，不去改主线 comparator 语义
3. 只比较 `T68` 留下的四个 full generated-only 候选
4. 只把 repeat budget 提高到 `4`
5. 在这个更强但仍然有界的矩阵里，看 tie set 是否还存在

也就是说，T69 不是“做一个更大 benchmark”，而是“用最小额外成本把剩下的一个关键歧义问清楚”。

### 2.2 改了哪些代码和配置

这次新增的任务范围内实现主要是三部分：

1. `cnn_fpga/config/p4_multiscenario_statcalib_clean_winner_tiebreak.yaml`
   - 这是 T69 专用配置
   - 它锁死了四个场景、两个 frozen anchors、四个候选 statcalib 模式，以及 `paired_seeds=true`、`repeats=4`
   - 它没有去改历史 config，也没有碰主线 runner 语义

2. `cnn_fpga/benchmark/summarize_statcalib_clean_winner_tiebreak.py`
   - 这是 T69 专用 summary helper
   - 它读取 T69 run root 里的 `summary.json`、`launch_plan.json`、`comparison.csv`
   - 它还会读取保存下来的 `T68` summary pack，但只用于比较“原来的 clean tie-set 现在是 persist / reduce / collapse”
   - 它最终输出：
     - 各候选的 mean LER
     - 各候选的 worst-case / best-case LER
     - `generated` / `mixed` 行数
     - 各候选是否四个场景都击败两个 frozen anchors
     - 三个 `window_variance` 候选之间的 tie 情况
     - `window_variance` clean set 与 `ekf_t001` 的对照
     - pairwise head-to-head 表
     - 最终分类：`unique_clean_reference_point` / `reduced_clean_tie_set` / `persistent_clean_tie_set`

3. `tests/test_statcalib_clean_winner_tiebreak_summary.py`
   - 这是针对 helper 的聚焦测试
   - 它至少覆盖了：
     - persistent tie 检出
     - unique clean point 检出
     - reduced tie set 检出
     - mixed 候选不应被误判为 full generated-only
     - 缺行矩阵要报错

### 2.3 没有改什么

这点很重要，因为它决定了这是不是一次“受控开发”而不是偷改语义：

1. 没有改 `cnn_fpga/decoder/statcalib.py`
2. 没有改 `cnn_fpga/runtime/slow_loop_runtime.py`
3. 没有改 `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
4. 没有重写 `T24/T64/T66/T67/T68` 历史结果
5. 没有把范围扩到 `.tflite`、真板或更大规模搜索

所以这次不是“换一套算法再宣布结果”，而是“沿用既有主线语义，对一个剩余疑问做有界复核”。

### 2.4 benchmark 是怎么跑的

任务包要求 clean provenance。由于主工作区当时存在无关治理改动，worker 没有直接在当前工作树里跑，而是从干净短路径 clone 发起：

- clean clone: `C:\t69c_1dbfbc3`

真正的 benchmark 只产生了一个 T69 run root：

- `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_20260608_160358`

矩阵大小是：

- `4 scenarios x 6 modes x 4 repeats = 96 repeat-runs`

这里的 6 个 mode 是：

1. `ukf`
2. `hybrid_residual_b`
3. `statcalib_window_variance_t001`
4. `statcalib_window_variance_t003`
5. `statcalib_window_variance_t005`
6. `statcalib_ekf_t001`

这满足了任务包要求的“一个 run root、一个完整矩阵、不按 mode chunk、不按 scenario chunk”的限制。

### 2.5 跑出来的结果是什么

核心结果可以压缩成四句话：

1. 四个 statcalib 候选全部仍然是 full `generated`
2. 四个候选在四个场景里都继续优于两个 frozen anchors
3. `statcalib_ekf_t001` 仍然是 clean winner，但弱于三个 `window_variance` 候选
4. `window_variance_t001 = t003 = t005` 的三路平局完整保留了下来

更细一点看：

- 四个场景里的最佳 statcalib 集合都完全一样，都是 `t001 = t003 = t005`
- 这三个候选在 mean-best 上打平
- 它们在 worst-case-best 上也继续打平
- `ekf_t001` 在四个场景中都输给这三个 `window_variance` 候选

最终 helper 给出的正式分类是：

- `persistent_clean_tie_set`

也就是：`T69` 没有产生唯一 clean reference point。

## 3. 这对后续开发意味着什么

T69 的意义不是“把 statcalib 推成主线”，而是把一个悬而未决的边界问题收干净了。

它对后续开发的意义主要有三点：

1. 主线现在可以更诚实地表述结果
   - 不必再说“也许进一步重复后会出现唯一 clean winner”
   - 在当前有界证据下，更诚实的说法是“三个 `window_variance` 阈值构成稳定 tie set”

2. 它帮助关闭 `R24` 这一类“clean winner 到底是不是唯一”的歧义
   - 至少在当前 mainline bounded protocol 里，这个问题已经被更强 repeats=4 复核过

3. 它避免后续开发误走方向
   - 如果有人还想硬选一个唯一阈值，那应该另开任务，先定义新的选择规则
   - 不能拿 T69 现有结果直接编造成“唯一阈值已经确定”

换句话说，T69 的价值是“把不确定性画清楚”，而不是“强行制造一个更好看的结论”。

## 4. 为什么我的 Review 结论是 `PASS_WITH_WARNINGS`

### 4.1 为什么不是 `BLOCK`

因为我没有发现会推翻任务完成度的实质问题。

我实际核对到的关键证据包括：

1. 只存在一个 T69 run root
2. `launch_head == finish HEAD == summary.json["git_commit"] == 1dbfbc3`
3. `launch_plan.json` 中：
   - 场景就是锁定的 4 个
   - mode 就是锁定的 6 个
   - `paired_seeds = true`
   - `repeats = 4`
4. T69 summary helper、单测、helper 对真实 run root 的总结都能重新跑通
5. `progress.jsonl` 可复核出：
   - `running = 96`
   - `completed = 96`
   - duplicate `running = 0`
   - duplicate `completed = 0`
6. 没有发现主线语义文件被改动
7. 没有发现历史 run root 被重写

所以这不是伪实现，也不是只写文档不落地。

### 4.2 为什么不是无保留的 `PASS`

因为 `PASS_WITH_WARNINGS` 里的 warning 不是说“worker 做坏了”，而是说“这个结果很容易被后续人误讲”。

最大的误讲风险有两个：

1. 把 persistent tie set 讲成唯一阈值
2. 把 mock-backed software-HIL extension-lane 证据讲成 `.tflite`、真板或成熟 comparator 证据

所以我保留 warning，是为了给后续治理和阶段文档一个清晰提醒：

- T69 的正确价值是“证明 tie 仍然存在”
- 不是“证明唯一阈值已经诞生”

## 5. Worker 已有 review / explanation 文档有没有问题

有两点判断：

1. 核心方向是对的
   - worker 已经抓住了最重要的结论：T69 的答案是 persistent tie，不是 unique clean reference point

2. 仍然值得补充和重写
   - 我这次补强了 provenance 证据链
   - 把“为什么不是 BLOCK、为什么也不是无保留 PASS”讲得更清楚
   - 也把“这不是主表改写、不是 `.tflite`、不是实板、不是成熟 comparator”这些边界写得更明确

因此，这份文档更像是在 worker 原有方向上做了一次 reviewer 视角的收口，而不是推翻 worker 的主结论。
