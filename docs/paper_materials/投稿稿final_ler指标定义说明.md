# 投稿稿 final_ler 指标定义说明

日期：2026-07-02

对象：`docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

性质：投稿稿 metric definition 写作材料；不新增实验，不重跑 benchmark，不升级 HIL、`.tflite`、real-board、statcalib 或统计证据等级。

## 一句话结论

`final_ler` 应写成当前 software-HIL fast-loop emulator 末端输出的协议定义逻辑错误代理指标。它来自 `LogicalErrorTracker` 的累计 X/Z 边界越界计数除以 fast-loop 轮数；它不是真实硬件 logical error rate，不是 logical-channel tomography，也不是 confidence interval 或显著性统计。

## 代码锚点

| 层级 | 当前锚点 | 稿件可写内容 | 禁止外推 |
| --- | --- | --- | --- |
| 逻辑错误追踪 | `physics/logical_tracking.py` 中 `get_total_logical_errors()` 返回 `logical_x_errors + logical_z_errors`，`get_logical_error_rate()` 返回该总数除以 `total_rounds` | 每轮平均的协议定义逻辑错误代理 | 真实硬件 logical failure rate、outer-code threshold 或 channel tomography |
| 边界判定 | `LogicalErrorTracker.update()` 用 q/p 累积残差越过 `±LATTICE_CONST/2` 分别计 X/Z error，并 wrap 回基本区间 | 指标由半晶格残差边界判定产生 | 完整 finite-energy GKP physical model 或 wrapped-posterior decoder |
| fast-loop 输出 | `cnn_fpga/runtime/fast_loop_emulator.py` 每轮追加 `tracker.get_logical_error_rate()` 到 `_cumulative_ler_curve`，最终输出 `final_logical_error_rate` | run 结束时的累计 LER proxy | fallback-free runtime proof 或 hardware measurement |
| source data | `docs/figure_assets/submission_draft_python_figures/source_data_fig02_main_results.csv` 记录 `final_ler_mean`、`final_ler_sd`、`n_repeats=2` | Fig. 2/Table 1 使用 mean + descriptive SD | CI、standard error、p-value、statistical significance |

## 稿件定义建议

可在正文中使用：

> `final_ler` is the final value of the cumulative logical-error proxy reported by the software-HIL fast-loop emulator.

更完整版本：

> For one run, `final_ler` is computed as \((N_X + N_Z)/T\), where \(N_X\) and \(N_Z\) are the numbers of q- and p-quadrature residual-boundary crossings counted by `LogicalErrorTracker`, and \(T\) is the number of fast-loop rounds in the run.

必须紧跟边界句：

> This metric is a protocol-defined software-HIL proxy. It is not a hardware logical-error measurement, a logical-channel estimate, a confidence interval, or a significance test.

## 当前 source-data 解释

- `final_ler_mean`：同一 scenario/mode 下两个 repeats 的 `final_ler` 均值。
- `final_ler_sd`：同一 scenario/mode 下两个 repeats 的 descriptive standard deviation。
- `n_repeats=2`：当前统计样本量太小，只能作为 descriptive uncertainty marker。
- `source`：当前均来自 `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`。

## 审稿人会追问的问题

1. **这个指标是否等于真实 logical error rate？**

   不等于。它使用 GKP 半晶格逻辑边界作为 software-HIL 协议内的判定规则，但当前没有 real-board measurement、logical-channel tomography 或 outer-code failure evidence。

2. **Fig. 2 误差条能否支持显著性？**

   不能。`n_repeats=2` 的 `final_ler_sd` 只能作为 descriptive SD。投稿前若要写统计强结论，需要 paired CI、bootstrap 或更高 repeats。

3. **为什么仍可作为主结果指标？**

   因为该指标与当前方法主张一致：比较不同 slow-loop calibration policy 在同一 fixed software-HIL protocol 下对 affine fast path 的影响。它适合 frozen-set ranking，不适合 broad distributional claim。

## 后续补强项

1. 把定义框中的公式与 `LogicalErrorTracker` 行为纳入轻量单元测试或 CI preflight。
2. 在 source-data manifest 中为每行补 `metric_definition_version`、run root、config hash、commit、runner version 和 artifact hash。
3. 增加 paired CI / bootstrap 或更高 repeats。
4. 增加 holdout drift family 和 stronger baselines，避免主结果只停留在 frozen-set ranking。
