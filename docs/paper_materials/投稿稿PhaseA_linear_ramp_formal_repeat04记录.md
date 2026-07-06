# 投稿稿 Phase A `linear_ramp` formal repeat 04 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_04_05_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 4 --repeat-stop 5`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 4 | 0.8122875 | 0.0024852777777777777 | 0.0024852777777777777 | 900 | 0.000018333333333333333 |
| `hybrid_residual_b` | 4 | 0.7873411111111112 | 0.0024594444444444447 | 0.0024594444444444447 | 900 | 0.000018333333333333333 |

UKF-minus-Hybrid final-LER delta: `0.024946388889`.

After this run, the cumulative `linear_ramp` formal partial source data were:

- paired repeats: `5/12`
- repeat indices: `0,1,2,3,4`
- mean UKF-minus-Hybrid delta: `0.022919888889`
- sample SD of paired deltas: `0.001773372337`
- positive pairs: `5/5`

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_04_05_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_04_05_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_04_05_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_04_05_20260706/progress.jsonl`

## 汇总同步

本次运行已通过 `build_submission_draft_phase_a_repeat_summary.py` 纳入：

- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`
- `投稿稿phase_a_repeat_summary记录.md`

`run_phase_a_paired_interval_analysis.py` 仍只输出 `static_bias_theta`，因为 `linear_ramp` 尚未达到 `12/12` completed scenario-row 条件。

## 可写边界

- 可以写：`linear_ramp` 的第五个 formal-length paired repeat 中，Hybrid Residual-B 的 final LER 低于 UKF，paired delta 为 `0.024946388889`。
- 可以写：该 run 后 `linear_ramp` 有 `5/12` formal partial source rows，五个 observed pairs 均为 positive UKF-minus-Hybrid delta；后续当前累计状态见 repeat05 记录。
- 不能写：`linear_ramp` 已完成 formal scenario row。
- 不能写：该结果提供 paired interval、confidence interval、p-value、all-scenario repeat-expanded advantage、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
