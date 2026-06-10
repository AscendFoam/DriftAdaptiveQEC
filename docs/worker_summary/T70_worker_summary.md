# T70 Worker Summary

## 改了什么

本轮只改了 `T70` 允许路径：

1. 新增 task-scoped helper：`cnn_fpga/benchmark/build_fr8_statcalib_bounded_closure_pack.py`
2. 新增 focused tests：`tests/test_fr8_statcalib_bounded_closure_pack.py`
3. 新增 closure pack 文档：`docs/fr8_statcalib_bounded_closure_pack.md`
4. 新增 review 文档：`docs/review/T70_review.md`
5. 新增人类解释文档：`docs/for_human/T70_explanation.md`
6. 新增本 worker summary：`docs/worker_summary/T70_worker_summary.md`
7. 在任务包 `docs/tasks/Phase2/T70_fr8_statcalib_bounded_closure_pack_and_promotion_gate.md` 末尾追加本轮 Worker Output

helper 的职责是只读复用：

- `docs/P4_benchmark_formal_protocol.md`
- `T64/T65/T66/T67/T68/T69` 报告与 review
- `runs/p4_benchmark/T24/T64/T66/T67/T68/T69` 的 preserved historical artifacts

然后输出一个统一 closure pack，其中显式给出：

- `T24` frozen anchor verdict
- `T64/T66/T67/T68/T69` 的 bounded evidence chain
- `final_strongest_clean_answer_set_after_t69`
- `unique_clean_reference_point_exists`
- `promotion_gate`
- `unique_threshold_gate`
- 后续若真要选单一阈值时的最小前提

## 如何验证

执行过的验证命令：

1. `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_fr8_statcalib_bounded_closure_pack`
2. `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga\benchmark\build_fr8_statcalib_bounded_closure_pack.py`
3. `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.build_fr8_statcalib_bounded_closure_pack`
4. `@(Get-ChildItem -LiteralPath 'runs\\p4_benchmark' -Directory | Where-Object { $_.Name -like 'T70*' }).Count`
5. `git diff --name-only -- runs`

验证结果：

- 单测通过：`Ran 2 tests`, `OK`
- `py_compile` 通过
- helper 单次执行成功，并输出完整 closure pack JSON
- `T70*` run-root 计数 = `0`
- `git diff --name-only -- runs` 为空
- helper 输出里显式记录：
  - `no_new_run_root_created = true`
  - `historical_runs_modified = false`
  - `sidecar_outputs_used = false`

## 核心结果

最终最强 clean answer set after `T69`：

- `statcalib_window_variance_t001`
- `statcalib_window_variance_t003`
- `statcalib_window_variance_t005`

是否存在 unique clean reference point：

- `False`

最终 promotion gate verdict：

- `no_promotion_keep_extension_lane_only`

最终 unique-threshold gate verdict：

- `future_selection_task_required`

如果未来还要做单一阈值选择任务，最小前提至少包括：

1. 先预声明一个当前 `T69` tie-set 之外的 selection criterion
2. 先锁定 candidate set 和 decision rule
3. 在新 gate 前继续保持 `T24` frozen、`statcalib` extension lane only
4. 若目标 claim 超出 mock-backed software-HIL 边界，则必须另开新验证任务，不得复用 `T64-T69` 直接升格

## 剩余风险

1. `R24` 还没有被“主线升格意义”上关闭；它只是被 `T70` 收缩成更明确的 overclaim/promotion boundary。
2. 当前 `FR8` 结果仍然只是 mock-backed software-HIL extension-lane evidence。
3. broader predeclared grid 仍然不是 uniformly clean closure story，因此当前不支持 promotion。
4. 若后续有人在口头或文档里把 `T69` 说成“已经找到唯一阈值”，那会直接违背 `T70` 的 gate 结论。
