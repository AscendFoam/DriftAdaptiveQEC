# T64 解释文档

## 1. 这个任务在做什么

通俗地说，`T64` 要回答的问题不是“`statcalib` 是不是已经正式成为主线冠军”，而是一个更小、更谨慎的问题：

能不能在不动历史正式结果 `T24` 的前提下，把 `statcalib` 作为一个额外的第六条 lane，加进同一套四场景 benchmark 里，做一次干净、可追溯、范围受控的比较？

如果把 `T24` 想成已经冻结存档的“五名正式选手成绩单”，那么 `T64` 做的事情更像是：

- 不改那张旧成绩单
- 另外让第六名候选选手 `statcalib` 参加同样的赛道
- 单独记录它和旧冠军之间的差距

所以 `T64` 的重点不是“重排主榜单”，而是“安全地增加一条扩展观察 lane”。

## 2. 这个任务为什么会出现在现在

从 `docs/04_task_board.md` 和 `docs/07_handoff.md` 的链条看，`T64` 不是突然冒出来的。

- `T59` 先把 `statcalib` lane 接进 runtime 和 benchmark，但那只是 bounded smoke，不是 FR8。
- `T60` 修掉了 `teacher_mode` 隔离问题，补了回归保护。
- `T61` 想做 provenance-clean rerun，但 run 期间 commit 身份漂移，所以被 reviewer block。
- `T62` 重做了一次 provenance-isolated rerun，真正把 `R27` 关掉。
- `T63` 再做 gate review，结论是：可以开“恰好一个” bounded FR8 extension-lane task，但不能改写 `T24`，也不能把 `R24` 当成已经关闭。

所以 `T64` 的历史角色很明确：

- 它不是 `statcalib` 的第一次出现
- 它也不是最终部署验证
- 它是 gate 之后唯一允许开的那次“正式边界内扩展 lane benchmark”

## 3. Worker 具体做了什么

这次实现的关键变化其实很克制，核心只有三部分。

第一部分是新建任务专用配置：

- `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml`

这个配置的意义不是“发明新 benchmark 规则”，而是为了在不改 `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` 的情况下，保留原来的：

- 四个冻结场景
  - `static_bias_theta`
  - `linear_ramp`
  - `step_sigma_theta`
  - `periodic_drift`
- 五个冻结模式顺序
  - `ekf`
  - `ukf`
  - `constant_residual_mu`
  - `rls_residual_b`
  - `hybrid_residual_b`

然后只把 `statcalib` 追加成第六个模式。

为什么要单开这个 config？因为这个 runner 的配置合并对 list 更接近“整段替换”，不是“自动追加”。如果直接改历史强基线 config，就会污染 `T24` 的冻结语义；而单开一个 task-scoped derived config，正好符合任务包允许的最小改动。

第二部分是执行一轮 bounded benchmark：

- run root: `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`

这轮 benchmark 保持了：

- `paired_seeds`
- `repeats=2`
- 固定四场景
- 固定冻结五模式顺序
- `statcalib` 作为第六 lane

我复核到的结果是：

- `comparison_rows_count=24`
- `raw_rows_count=48`
- `missing_runs=[]`
- 所有 comparison rows 都是 `coverage=1.0`
- 所有 comparison rows 都是 `completed_repeats=2`
- `progress.jsonl` 没有同一 repeat key 的重复 `running`

第三部分是把结果整理成文档：

- `docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md`
- `docs/review/T64_review.md`
- `docs/for_human/T64_explanation.md`
- `docs/worker_summary/T64_worker_summary.md`

其中最重要的实质性结论有两个：

1. `T64` 里的冻结五模式子表，与 `T24` 的 20 个冻结 comparison rows 完全一致，没有偷偷改历史结果。
2. `statcalib` 作为第六条 extension lane，在这四个场景里都优于冻结 winner `hybrid_residual_b`。

## 4. 这对后续开发有什么意义

这一步的价值，不在于“终于可以宣布 `statcalib` 已经统治一切”，而在于它把后续讨论方式锁得更清楚了。

`T64` 之后，团队可以诚实地说：

- 我们手里现在有一份干净 provenance 的、四场景的、mock-backed software-HIL 的 `statcalib` extension-lane result pack。
- 这份 result pack 和历史 `T24` 冻结主表可以并存。
- 后续如果要讨论 FR8，只能在“主表不改写、扩展 lane 单独标注”的边界里讨论。

它对后续开发的现实意义主要有三点：

1. 它给了后续 FR8 文档、图表或 gate 一个可以引用的 bounded result pack，而不需要再借用较弱的 smoke 结果。
2. 它证明仓库已经具备“在 frozen benchmark 外围加一条独立 lane 而不破坏历史锚点”的操作能力。
3. 它继续把 truth boundary 钉死在 mock-backed software-HIL，避免团队把局部成功误写成 `.tflite` 或 real-board 事实。

换句话说，`T64` 更像是“报告边界和证据边界的升级”，不是“部署边界的升级”。

## 5. 为什么我的 review 结果不是 `PASS`，而是 `PASS_WITH_WARNINGS`

我没有给 `BLOCK`，因为任务包要求的核心事情确实做到了，而且没有触发 review no-go：

- 没改 source/test
- 没改历史强基线 config
- 没改写历史 run roots
- `launch HEAD`、`finish HEAD`、`summary.json["git_commit"]` 一致
- 冻结五模式子表和 `T24` 完全一致
- `statcalib` 是单独标注的第六 lane，而不是把 `T24` 直接重排

但我也没有给纯 `PASS`，原因是我看到两类不该忽略的问题。

第一类是文档准确性问题。

`docs/evidence_packs/statcalib_fr8/fr8_statcalib_extension_lane_benchmark.md` 里有两处不够严谨：

1. 它把执行形态写成 `one detached one-shot invocation only`。
   - 但任务包明确接受的说法是 `one foreground invocation`，或者按 repeat-range chunking。
   - 从 run 产物看，这次执行确实是单次、单 run root、无 resume，我不认为这是结果失真；但报告文本没有严格贴合任务包语言。

2. 它把 `2026-05-29 12:01:16 +08:00` 写成 `finish timestamp from summary.json`。
   - 我实际核对 `summary.json` 后确认，文件里没有这个字段。
   - 这个时间对应的是 `summary.json` 文件的 LastWriteTime。
   - 这说明 provenance 描述里混入了“从文件属性读出来的时间”与“JSON 内显式记录的字段”之间的界限不清。

第二类是结果解释风险。

`statcalib` 在四个场景里领先幅度很大，这当然值得关注；但它仍然只是：

- mock-backed software-HIL
- bounded FR8 extension lane
- 沿用前面 T59/T60/T62 路线里那个最小 comparator 语义

所以如果直接给纯 `PASS`，容易让后续读者忽略“这是一份强结果，但仍然是受限结果”的事实。`PASS_WITH_WARNINGS` 更符合这个任务的真实状态。

## 6. 有没有伪实现、mock、stub、hardcode

这是这轮 review 的重点之一。我的结论是：

- 没有发现把不存在的执行结果伪装成已完成 benchmark 的情况。
- 也没有发现通过改写旧 run root 或篡改 `T24` 主表来“制造胜利”的情况。

但有两个必须说清的边界：

1. `statcalib` 不是一个“凭空伪造出来的 lane”，它确实贯通了 runtime、aggregation 和最终结果表。
2. 但它也不是一个已经被充分证明的成熟 calibration 系统。它仍然是此前任务链里那个 bounded comparator lane，参数也沿用了 smoke lane 的最小配置思路。

因此，它不是 fake implementation，但它依然带有“最小实现、边界严格受限”的属性。把这类最小实现写成“已经完成完整 comparator 研究”仍然是不诚实的，而 T64 文档主体暂时没有越过这条线。

## 7. 我对 Worker 现有 review / explanation 的看法

Worker 已经写了两个文档草稿，但都还有可补强的地方。

先说原来的 `docs/review/T64_review.md`。

- 优点：方向基本对，抓住了“冻结子表没被改写”和“extension lane 独立存在”这两个核心点。
- 不足：
  - 直接给了 `PASS`，我认为略宽。
  - 没有完整覆盖你要求的所有栏目，尤其缺少 `Suspicious implementation details`。
  - 没把结果文档里的两处精确表述问题单独拎出来。

再说原来的 `docs/for_human/T64_explanation.md`。

- 它没有明显写错核心事实。
- 但内容太短，更像一句结论摘要，不足以帮助不了解前情的人理解：
  - 为什么 T64 会在 T63 后面出现
  - 它和 T59/T60/T61/T62 的关系是什么
  - 为什么“赢了四个场景”仍然不等于“主榜单改写”或“部署验证完成”

所以我这次不是简单打补丁，而是把两份文档都补成了更完整的版本。

## 8. 最后应该怎样理解 T64

最稳妥的理解方式是：

- `T64` 成功了
- 但它成功的是“bounded extension-lane benchmark”
- 不是“主线正式榜单改写”
- 不是“.tflite 已验证”
- 不是“real-board 已验证”
- 也不是“从此 `statcalib` 已经完成 paper-grade comparator 定论”

如果用一句话总结：

`T64` 让仓库第一次拥有了一份可以单独引用的、干净 provenance 的 `statcalib` 四场景 extension-lane result pack；同时，它也再次证明这个项目现在最重要的不是把好结果说得更大，而是把边界说得更准。
