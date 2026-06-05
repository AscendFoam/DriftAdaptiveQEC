# T67 Worker Summary

## 改了什么

- 新增了 task-scoped config：[p4_multiscenario_statcalib_teacher_anchor.yaml](D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/config/p4_multiscenario_statcalib_teacher_anchor.yaml)
- 新增了 task-scoped summary helper：[summarize_statcalib_teacher_anchor.py](D:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py)
- 新增了 focused unit test：[test_statcalib_teacher_anchor_summary.py](D:/Codes/Quantum/DriftAdaptiveQEC/tests/test_statcalib_teacher_anchor_summary.py)
- 完成了唯一允许的 T67 run root：[T67_statcalib_teacher_anchor_20260601_225718](D:/Codes/Quantum/DriftAdaptiveQEC/runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718)
- 补齐了 T67 文档：[statcalib_teacher_anchor_bounded_benchmark.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/statcalib_teacher_anchor_bounded_benchmark.md)、[T67_review.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/review/T67_review.md)、[T67_explanation.md](D:/Codes/Quantum/DriftAdaptiveQEC/docs/for_human/T67_explanation.md)

## 如何验证

- `C:\ProgramData\anaconda3\python.exe -m py_compile cnn_fpga/benchmark/summarize_statcalib_teacher_anchor.py`
- `C:\ProgramData\anaconda3\python.exe -m unittest tests.test_statcalib_teacher_anchor_summary`
- `C:\ProgramData\anaconda3\python.exe -m cnn_fpga.benchmark.summarize_statcalib_teacher_anchor --run-dir runs/p4_benchmark/T67_statcalib_teacher_anchor_20260601_225718`
- provenance 闭合：
  - launch `HEAD = 84f4468`
  - finish `HEAD = 84f4468`
  - `summary.json git_commit = 84f4468`
- 结果完整性：
  - `comparison.csv` 共 `32` 行
  - `missing_runs = []`
  - 所有 comparison rows 都是 `coverage=1.0`、`completed_repeats=2`
  - `progress.jsonl` 里 `running=64`、`completed=64`
  - duplicate `running` 记录数是 `0`
- 范围与历史结果保护：
  - 只有一个 `T67` run root
  - `T24/T64/T66` 的目录最后写入时间没有变化

## 剩余风险

- `T67` 的正面结果很明确，但边界仍然只是 mock-backed software-HIL，不是 `.tflite`，不是真板。
- 这次不能拿来改写 `T24` 冻结主表；`statcalib` 仍然只能作为单独 extension lane 报告。
- 有两条 comparison row 仍然是 `mixed`：
  - `static_bias_theta / statcalib_high_threshold_teacher_window_variance`
  - `step_sigma_theta / statcalib_high_threshold_teacher_ukf`
- 因此最强 aggregate lane 不是完全 generated-only 的“干净闭环”结果，后续引用时必须保留这个 provenance caveat。
