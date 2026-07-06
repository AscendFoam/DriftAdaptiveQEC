# 投稿稿 Phase A `linear_ramp` formal repeat 02 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_02_03_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 2 --repeat-stop 3`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 2 | 0.810135 | 0.0024666666666666665 | 0.0024666666666666665 | 900 | 0.00001611111111111111 |
| `hybrid_residual_b` | 2 | 0.7881302777777778 | 0.002403611111111111 | 0.002403611111111111 | 900 | 0.00001611111111111111 |

UKF-minus-Hybrid final-LER delta: `0.022004722222`.

After this run, the cumulative `linear_ramp` formal partial source data at that point were:

- paired repeats: `3/12`
- repeat indices: `0,1,2`
- mean UKF-minus-Hybrid delta: `0.021695000000`
- positive pairs: `3/3`

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_02_03_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_02_03_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_02_03_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_02_03_20260706/progress.jsonl`

## 汇总同步

本次运行已通过 `build_submission_draft_phase_a_repeat_summary.py` 纳入：

- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`
- `投稿稿phase_a_repeat_summary记录.md`

`run_phase_a_paired_interval_analysis.py` 仍只输出 `static_bias_theta`，因为 `linear_ramp` 尚未达到 `12/12` completed scenario-row 条件。

## 可写边界

- 可以写：`linear_ramp` 的第三个 formal-length paired repeat 中，Hybrid Residual-B 的 final LER 低于 UKF，paired delta 为 `0.022004722222`。
- 可以写：截至本次运行记录生成时，`linear_ramp` 有 `3/12` formal partial source rows，三个 observed pairs 均为 positive UKF-minus-Hybrid delta；当前 cumulative partial 状态以最新 repeat 记录和 `submission_draft_phase_a_repeat_summary.csv/json` 为准。
- 不能写：`linear_ramp` 已完成 formal scenario row。
- 不能写：该结果提供 paired interval、confidence interval、p-value、all-scenario repeat-expanded advantage、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
