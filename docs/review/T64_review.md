# T64 Review

- Verdict: `PASS_WITH_WARNINGS`

`T64` 的主体目标已经完成。当前证据足以支持这样一个有界结论：在不改写 `T24` 冻结五模式历史结果的前提下，仓库已经新增了一次 provenance-clean、四场景、`repeats=2`、`paired_seeds` 的 `statcalib` 第六 lane 扩展基准；而且 `T64` 中的冻结五模式子表与 `T24` 的 20 个冻结 comparison rows 完全一致。

我本次审查只基于现有 diff、任务包、run 产物、`summary.json`、`launch_plan.json`、`progress.jsonl`、`git reflog` 和相关文档做只读核查，没有重跑长实验。

## Blocking issues

- None.

没有触发任务包里的 review no-go 条件。我核对到的事实包括：

- `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` 没有被修改。
- 没有 source/test 文件改动。
- `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml` 保持了冻结四场景和冻结五模式顺序，并把 `statcalib` 追加为第六 lane。
- 只存在一个 T64 run root：`runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658`。
- `launch HEAD`、`finish HEAD`、`summary.json["git_commit"]` 都是 `1e59f24`。
- `comparison_rows_count=24`，`raw_rows_count=48`，`missing_runs=[]`，所有 comparison rows 都是 `coverage=1.0` 且 `completed_repeats=2`。
- `progress.jsonl` 没有同一 `(scenario, mode, repeat)` 的重复 `running` 记录。
- `T64` 的冻结五模式子集与 `T24` 在 20 个冻结 rows 上 `final_ler_mean` 和 `overflow_rate_mean` 的最大绝对差都是 `0`。

## Non-blocking issues

1. `docs/fr8_statcalib_extension_lane_benchmark.md` 把执行形态写成了 `one detached one-shot invocation only`。但任务包 `docs/tasks/Phase2/T64_fr8_statcalib_extension_lane_bounded_benchmark.md` 明确接受的只有两种形态：
   - one foreground invocation across the full matrix
   - repeat-range chunking under one fixed run root
   现有 run 产物确实表现为单次、单 run root、无 resume 的 clean 执行，所以我不把它升格为 blocker；但报告文字没有严格贴住任务包原文，后续复用时应该收紧表述。

2. 同一份结果文档把 `2026-05-29 12:01:16 +08:00` 写成了 `finish timestamp from summary.json`。我核对后确认：
   - `runs/p4_benchmark/T64_fr8_statcalib_extension_lane_20260527_221658/summary.json` 本身没有 finish timestamp 字段
   - 这个时间实际上对应 `summary.json` 文件的 LastWriteTime
   这不是结果造假，但属于 provenance 文案不准确。

3. `statcalib` 在四个冻结场景里对冻结 winner / runner-up 都给出了异常大的优势。当前证据仍然只能说明：
   - 这是一个 mock-backed software-HIL extension lane 结果
   - `statcalib` 仍然是沿用 T59/T60/T62 链条里那个最小 comparator 语义
   不能把这个结果自动外推成更成熟的 calibration comparator 或部署结论。

## Missing tests

1. 这次任务本身没有源码改动，所以不存在“必须新增单元测试却没写”的直接 blocker。

2. 但仓库里仍缺一个轻量级的一致性检查，来自动比对：
   - `docs/fr8_statcalib_extension_lane_benchmark.md`
   - `summary.json`
   - `launch_plan.json`
   - `progress.jsonl`
   当前 timestamp 误归因之所以能滑过去，本质上就是因为结果文档仍靠人工整理，没有 report-to-artifact consistency guard。

3. 同样缺一个自动化 guard，专门断言 extension-lane 任务不会改写 `T24` 冻结子表。T64 这次是通过人工审查确认“冻结子集完全一致”，不是通过现成测试或校验脚本自动兜底。

## Suspicious implementation details

1. `cnn_fpga/config/p4_multiscenario_statcalib_extension_lane.yaml` 里的 `statcalib` 参数块，本质上是把 earlier smoke lane 的最小参数集带进 formal-bounded extension lane，而不是基于新的 calibration search 或独立推导重新得到的配置。

2. `statcalib` 不是 mock/stub 意义上的伪实现，它确实贯通了 runtime、aggregation 和最终 `comparison.csv`；但它也不是一个已经被充分证明的“成熟比较器”。这一点在后续任何 FR8 文案里都必须继续写清。

3. 所有抽查的 `hil_summary.json` 仍然显示 `backend: mock`。这与任务边界一致，但也说明任何进一步的 `.tflite` / real-board 含义都还不存在。

## Recommended next action

1. Captain 可以把 `T64` 按 `PASS_WITH_WARNINGS` 接受为“有界 extension-lane benchmark 已完成”。

2. 在任何后续文档、图表或 gate 里复用 `T64` 前，先做一个很小的 docs-only 收口，把 `docs/fr8_statcalib_extension_lane_benchmark.md` 里的两处表述修正掉：
   - `detached` 执行形态说法
   - `finish timestamp from summary.json` 的误归因

3. 后续如果继续推进 FR8 相关工作，必须继续保持三条边界不变：
   - `T24` 冻结五模式历史表单独保留，不被重写
   - `statcalib` 只作为 separately labeled extension lane 报告
   - 证据等级仍然是 mock-backed software-HIL，不是 `.tflite`，不是 real-board，也不是自动升级后的 paper-grade expanded benchmark
