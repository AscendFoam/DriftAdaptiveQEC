# T70 Review

## Verdict

`PASS`

我按 T70 任务包要求对本次 diff 做了只读审查，并额外复跑了本轮要求的轻量验证：

1. `python -m py_compile cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py`
2. `python -m unittest tests.test_fr8_statcalib_bounded_closure_pack`
3. `python -m cnn_fpga.benchmark.build_fr8_statcalib_bounded_closure_pack`
4. `@(Get-ChildItem -LiteralPath 'runs\\p4_benchmark' -Directory | Where-Object { $_.Name -like 'T70*' }).Count`
5. `git diff --name-only -- runs`

上述检查均与任务包预期一致：

- 没有新建 `T70` run root
- 没有重跑 benchmark
- `git diff --name-only -- runs` 为空
- helper 能稳定重建 `T24/T64/T66/T67/T68/T69` 的 closure/gate 结果
- closure pack 文档与 helper 输出一致，没有把计划写成事实，也没有把 extension-lane 证据写成成熟 comparator / `.tflite` / 真板 / paper-grade 结论

## Blocking issues

- None.

## Non-blocking issues

1. helper 中部分“只读完整性”字段是任务内声明值，真正的可信检查仍然依赖外部验证命令。
   - `cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py:581-583` 直接写入了：
     - `no_new_run_root_created = True`
     - `historical_runs_modified = False`
     - `sidecar_outputs_used = False`
   - 这不构成阻断，因为任务包本身也要求额外执行：
     - `T70*` run-root 计数检查
     - `git diff --name-only -- runs`
     - helper 实际运行输出检查
   - 但需要明确：这三项里，真正证明“没有写 run / 没有新 run root”的证据主要来自外部验证，不是来自 helper 自身推理。

2. helper 是明确的 task-scoped consolidator，不应被误当成通用基础设施。
   - 它硬编码了 `T24/T64/T66/T67/T68/T69` 的固定输入路径、固定 review chain 和固定 gate 逻辑。
   - 对 `T70` 这是合理设计，不属于伪实现；但如果将来要做别的 closure/gate 任务，不应直接把它当 generic promotion framework 复用。

## Missing tests

没有发现会影响 `T70` 结论成立的阻断性缺测。

如果后续继续扩展这类 closure helper，建议补以下负向测试：

1. `T24` 不再满足“四场景 winner 全为 `hybrid_residual_b`、runner-up 全为 `ukf`”时 helper 应拒绝通过
2. `T64` 冻结五模式子集不再与 `T24` exact-match 时 helper 应拒绝通过
3. review chain 中任一历史任务 verdict 不再是已接受链（`PASS` / `PASS_WITH_WARNINGS`）时 helper 应拒绝通过
4. 输入路径若落到 `.wt/` 或 `runs/sidecar/` 时 helper 应有显式拒绝测试，而不是只依赖当前实装逻辑

## Suspicious implementation details

1. `cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py` 使用了大量固定路径和固定值。
   - 这看起来像 hardcode，但这里是任务包明确要求的 task-scoped closure helper。
   - 它的职责不是做通用 benchmark 框架，而是把 `T24/T64/T66/T67/T68/T69` 这条已接受 FR8 证据链安全收口。

2. helper 同时把 review docs 和 historical run summaries 当作输入。
   - 这是合理的，因为 `T70` 的目标之一就是把“实验事实”和“已接受审查链”一起固定下来，避免后续治理文档自由发挥。

3. `docs/fr8_statcalib_bounded_closure_pack.md` 中的两个 gate 结论是保守的，但它们与现有证据一致：
   - `No-Promotion Gate = no_promotion_keep_extension_lane_only`
   - `Unique-Threshold Gate = future_selection_task_required`
   这不是缺陷，而是对当前证据边界的正确表达。

## Recommended next action

接受 `T70` 为 `PASS`。

后续建议：

1. Captain 可以把 `docs/fr8_statcalib_bounded_closure_pack.md` 当作 FR8 主线 closure artifact 使用，而不必再把 `T64/T66/T67/T68/T69` 拆开自由转述。
2. 后续所有治理或 paper-material 文档若引用 FR8，应保持三条边界不变：
   - `T24` 仍是 authoritative frozen ranked table
   - `statcalib` 仍是 separately labeled extension lane
   - 当前不支持唯一阈值结论，也不支持 promotion
3. 如果未来业务上真的需要：
   - 选一个唯一阈值
   - 讨论 promotion
   - 升级到 `.tflite` / 真板 / 更强验证面
   则必须另开新的预声明任务，不能从 `T70` 当前 closure pack 静默外推
