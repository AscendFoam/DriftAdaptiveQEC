# 投稿稿 Phase A `linear_ramp` formal repeat 01 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_01_02_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 1 --repeat-stop 2`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 1 | 0.8093405555555555 | 0.0024747222222222223 | 0.0024747222222222223 | 900 | 0.00001361111111111111 |
| `hybrid_residual_b` | 1 | 0.787055 | 0.0024694444444444442 | 0.0024694444444444442 | 900 | 0.00001361111111111111 |

UKF-minus-Hybrid final-LER delta: `0.02228555555555556`.

After this run, the cumulative `linear_ramp` formal partial source data are:

- paired repeats: `2/12`
- repeat indices: `0,1`
- mean UKF-minus-Hybrid delta: `0.021540138889`
- positive pairs: `2/2`

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_01_02_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_01_02_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_01_02_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_01_02_20260706/progress.jsonl`

## 汇总同步

本次运行已通过 `build_submission_draft_phase_a_repeat_summary.py` 纳入：

- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`
- `投稿稿phase_a_repeat_summary记录.md`

`run_phase_a_paired_interval_analysis.py` 仍只输出 `static_bias_theta`，因为 `linear_ramp` 尚未达到 `12/12` completed scenario-row 条件。

## 可写边界

- 可以写：`linear_ramp` 的第二个 formal-length paired repeat 中，Hybrid Residual-B 的 final LER 低于 UKF，paired delta 为 `0.02228555555555556`。
- 可以写：截至本次运行记录生成时，`linear_ramp` 有 `2/12` formal partial source rows，两个 observed pairs 均为 positive UKF-minus-Hybrid delta；当前 cumulative partial 状态以最新 repeat 记录和 `submission_draft_phase_a_repeat_summary.csv/json` 为准。
- 不能写：`linear_ramp` 已完成 formal scenario row。
- 不能写：该结果提供 paired interval、confidence interval、p-value、all-scenario repeat-expanded advantage、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
