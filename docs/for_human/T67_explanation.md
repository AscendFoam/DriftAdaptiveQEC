# T67 任务与评审说明

## 1. 这个任务在做什么

`T67` 想回答一个很具体的问题：

之前 `statcalib` 在 `FR8` extension lane 里表现不错，这个优势会不会只是因为它选了 `teacher_mode=ukf` 这个老师锚点？如果把老师锚点换成别的，比如 `window_variance` 或 `ekf`，它是不是就不行了？

所以，`T67` 不是在发明新算法，也不是在开新长跑，而是在已经锁定的 `FR8` 边界里做一次受控复核：

1. 场景仍然只看那 4 个锁定场景。
2. repeat 仍然只做 `2` 次。
3. 仍然用 `paired seeds`。
4. 仍然拿冻结锚点 `ukf` 和 `hybrid_residual_b` 来比。
5. 只是把 `statcalib` 的 teacher-anchor 从原来的 `ukf` 扩到 `window_variance`、`ekf`，并保留 `default / high_threshold` 两个参数点。

通俗地说，这次是在检查：

“`statcalib` 的好成绩，到底是它本身有点东西，还是只是搭上了 `ukf teacher` 这辆车？”

`T67` 给出的有界答案是：在当前这套 bounded、mock-backed software-HIL 证据里，它并不主要依赖 `ukf teacher`。

## 2. 这次实现具体做了什么

### 2.1 任务目标

结合 `docs/04_task_board.md`、`docs/07_handoff.md` 和 `docs/08_risks_and_open_questions.md`，`T67` 的作用很明确：

1. 接着 `T64/T65/T66` 往前走。
2. 把 `R24` 从“参数点是否脆弱”进一步收紧到“teacher-anchor 依赖是否存在”。
3. 继续保持 `statcalib` 只是 extension lane，而不是去改写主 benchmark 结论。

也就是说，这个任务的意义不是“宣布 `statcalib` 已经成为正式主线赢家”，而是“把一个具体怀疑点查清楚”。

### 2.2 任务流程

这次 Worker 做的事情主要分成四块。

第一块是配置矩阵。

新增了 `cnn_fpga/config/p4_multiscenario_statcalib_teacher_anchor.yaml`，把这次 benchmark 固定成 8 条 lane：

1. 冻结锚点：
   - `ukf`
   - `hybrid_residual_b`
2. `statcalib` 6 个变体：
   - `statcalib_default_teacher_ukf`
   - `statcalib_default_teacher_window_variance`
   - `statcalib_default_teacher_ekf`
   - `statcalib_high_threshold_teacher_ukf`
   - `statcalib_high_threshold_teacher_window_variance`
   - `statcalib_high_threshold_teacher_ekf`

第二块是真实运行。

Worker 没有去改 runtime、benchmark runner 或 `statcalib` 主语义，而是保留语义不动，只发起了一次新的 bounded benchmark。由于源工作区里当时有一个与 T67 无关的 PDF 改动，为了满足“从 clean committed main 启动”的要求，Worker 选择从一个干净短路径 clone `C:\t67c` 发起 detached host launch，把结果写回唯一的 T67 run root：

- `runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`

这一点很重要，因为它说明 Worker 没有靠“改代码绕过去”，而是靠隔离运行环境来满足任务包约束。

第三块是汇总工具。

新增了 `cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py`。这个 helper 的职责不是通用化 benchmark 基建，而是专门读取本次 T67 run root 下的：

1. `summary.json`
2. `launch_plan.json`
3. `comparison.csv`

然后检查：

1. 场景集合对不对。
2. mode 集合对不对。
3. `paired_seeds` 是否为真。
4. `repeats` 是否仍然是 `2`。
5. 是否有缺失 comparison row。
6. 每个 comparison row 的 `coverage` 是否为 `1.0`。
7. 每个 comparison row 的 `completed_repeats` 是否为 `2`。

在这些检查通过后，它再生成按场景、按 teacher-anchor、按参数点聚合的结果总结。

第四块是最小测试与文档。

新增了 `tests/test_statcalib_teacher_anchor_summary.py`，用合成的小矩阵去测 helper 的基本行为；同时补了：

1. `docs/evidence_packs/statcalib_fr8/statcalib_teacher_anchor_bounded_benchmark.md`
2. `docs/review/T67_review.md`
3. `docs/for_human/T67_explanation.md`
4. `docs/worker_summary/T67_worker_summary.md`

### 2.3 结果说明了什么

从保留下来的 T67 run root 看，这次 bounded benchmark 的主要事实是：

1. 只存在一个新的 T67 run root。
2. `launch HEAD`、`finish HEAD`、`summary.json["git_commit"]` 都锁在 `84f4468`。
3. `comparison.csv` 一共 `32` 行，正好对应 `4 scenarios x 8 modes`。
4. `progress.jsonl` 是 `running=64`、`completed=64`、duplicate `running=0`，说明没有像 `T66` 那样出现重复 full-matrix 重发。

更重要的是结论本身：

1. 所有 6 条 `statcalib teacher-anchor` 变体，都在 4 个锁定场景里同时优于两个冻结锚点：
   - `ukf`
   - `hybrid_residual_b`
2. 从按参数点聚合的排序看，非 `ukf` teacher-anchor 并没有垮掉，反而在当前 bounded 证据里整体排在 `ukf teacher` 前面。
3. 最强结果主要集中在 `window_variance` 和 `ekf` teacher-anchor 上。

这就是为什么说，`T67` 解决的是“teacher-anchor 依赖怀疑”这个问题，而不是别的问题。

### 2.4 对后续开发的意义

这一步对后续开发的意义主要有三点。

第一，它让 `R24` 的讨论更具体了。

在 `T66` 之后，我们已经知道 `statcalib` 结果并不只是单个参数点的偶然产物。`T67` 再往前推进了一步，说明它也不只是“绑定 `ukf teacher` 才有效”。这样后面如果继续谈 `statcalib` 的 extension-lane 价值，论证会更扎实。

第二，它保护了主线 benchmark 口径。

`T67` 没有去改 `T24` 的冻结五模式主表，也没有静默改 baseline 集合。这一点很重要，因为 Phase 2 的前提就是“在不破坏已恢复可信度的前提下继续受控开发”。

第三，它为后续任务留下了更清楚的边界。

后面如果还要推进，也应该继续围绕：

1. extension lane 的 bounded 解释
2. 风险闭环
3. 更明确的证据标签

而不是直接跳到：

1. `.tflite`
2. real-board
3. paper-grade expanded benchmark
4. 主表改写

## 3. 为什么我的 review 结论是 PASS_WITH_WARNINGS

我给 `PASS_WITH_WARNINGS`，不是 `BLOCK`，也不是完全无保留的 `PASS`。

### 3.1 为什么不是 BLOCK

因为任务主体确实完成了。

我复核到的关键证据包括：

1. T67 只新增了一个 run root。
2. `py_compile` 通过：
   - `python -m py_compile cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py`
3. `unittest` 通过：
   - `python -m unittest tests.test_statcalib_teacher_anchor_summary`
   - `Ran 6 tests`, `OK`
4. summary helper 复核通过：
   - `python -m cnn_fpga.benchmark.summarize_statcalib_teacher_anchor --run-dir runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`
5. run root 内部完整性成立：
   - `comparison.csv` 完整
   - `missing_runs = []`
   - `progress.jsonl` 没有 duplicate `running`
6. 没有发现对 `statcalib.py`、runtime、benchmark runner 主语义的越界修改。

这些都说明它不是伪实现、不是手工拼表、也不是靠改 benchmark 语义“做出来”的结果。

### 3.2 为什么不是完全无保留的 PASS

因为还有三类 warning 需要保留。

第一类 warning 是 mixed worktree attribution。

当前工作区里确实还挂着一个 T67 范围外的 PDF 改动：`docs/汇报用/5月汇报材料/note-draft逐段口头汇报解释.pdf`。我没有把它当 blocker，是因为 T67 报告和 `host_launch_meta.json` 都一致说明 Worker 是从干净 clone 启动 benchmark，没有去碰那个无关文件。但从 reviewer 角度说，这仍然会削弱“只看当前 diff 就能完全归因”的清晰度。

第二类 warning 是 helper 的 tie 表达不够精确。

在 `cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py` 里，`better_parameter_point_by_mean_ler` 这个字段没有显式表示“平手”。当 `default` 和 `high_threshold` 的 mean LER 相等时，它会落到 `"high_threshold"`。而 T67 的实际 summary pack 里，`ekf` 那一组恰好就是平手，所以机器产物和文档 prose 之间有一个小的不一致。

这不是大 bug，但它确实说明：

1. 结果是真的；
2. 只是 machine-readable 标签比 prose 粗糙了一点；
3. 后续引用时不能把“平手”说成“已经明确找到更优参数点”。

第三类 warning 是 provenance 仍不算完全“纯 generated-only”。

T67 里还有两条 comparison row 是 `mixed`：

1. `static_bias_theta / statcalib_high_threshold_teacher_window_variance`
2. `step_sigma_theta / statcalib_high_threshold_teacher_ukf`

所以 T67 可以支撑“teacher-anchor 依赖不是主因”这个 bounded 结论，但不能被偷换成“所有相关 provenance 都已经最理想化”。

## 4. 对 Worker 已有 review / explanation 的复核

Worker 已经写了 review 和 explanation 草稿，方向大体是对的，我这次主要做了两件事。

第一，我补了 reviewer 口径里更该显式写出来的 warning：

1. 当前 worktree 仍有 scope-external PDF 改动，虽然我最终没有把它当 blocker。
2. helper 对 tie 的 machine-readable 表达不够精确。

第二，我把 explanation 重写得更完整一些。

Worker 原稿已经抓到了任务主旨，也就是“这次是在验证 `statcalib` 是否过度依赖 `ukf teacher`”。但我补充了下面这些更适合后续 handoff 和归档的内容：

1. 它和 `T64/T65/T66` 的承接关系。
2. 它为什么能回答 `R24` 的一个更具体子问题。
3. 它为什么不能升级成 `.tflite`、real-board 或主表改写结论。
4. 为什么这次 verdict 是 `PASS_WITH_WARNINGS` 而不是别的标签。

## 5. 一句话总结

`T67` 的价值，不是“证明 `statcalib` 已经完全定型”，而是把一个关键怀疑点排除了：

在当前这套 bounded、mock-backed software-HIL 证据里，`statcalib` 的优势并不是靠死绑 `teacher_mode=ukf` 才成立。
