# 投稿稿 Phase A `linear_ramp` formal repeat 00 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_00_01_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 0 --repeat-stop 1`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 0 | 0.8081319444444445 | 0.0024716666666666667 | 0.0024716666666666667 | 899 | 0.00001638888888888889 |
| `hybrid_residual_b` | 0 | 0.7873372222222222 | 0.0024213888888888887 | 0.0024213888888888887 | 900 | 0.00001638888888888889 |

UKF-minus-Hybrid final-LER delta: `0.02079472222222223`.

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_00_01_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_00_01_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_00_01_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_00_01_20260706/progress.jsonl`

## 汇总同步

本次运行已通过 `build_submission_draft_phase_a_repeat_summary.py` 纳入：

- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`
- `投稿稿phase_a_repeat_summary记录.md`

该 run 形成 `linear_ramp` 的第一个 formal partial source row。该记录保留 repeat `0` 之后的历史状态；当前 cumulative partial 状态以最新 repeat 记录和 `submission_draft_phase_a_repeat_summary.csv/json` 为准。`run_phase_a_paired_interval_analysis.py` 仍只输出 `static_bias_theta`，因为 `linear_ramp` 尚未达到 `12/12` completed scenario-row 条件。

## 可写边界

- 可以写：`linear_ramp` 的第一个 formal-length paired repeat 中，Hybrid Residual-B 的 final LER 低于 UKF，paired delta 为 `0.02079472222222223`。
- 可以写：该结果是后续补齐 `linear_ramp` scenario-row 的 source-data 起点之一。
- 不能写：`linear_ramp` 已完成 formal scenario row。
- 不能写：该结果提供 paired interval、confidence interval、p-value、all-scenario repeat-expanded advantage、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
