# 投稿稿 Phase A `linear_ramp` formal repeat 05 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_05_06_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 5 --repeat-stop 6`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 5 | 0.8092636111111111 | 0.0024491666666666668 | 0.0024491666666666668 | 900 | 0.000016944444444444446 |
| `hybrid_residual_b` | 5 | 0.7893130555555555 | 0.0024780555555555554 | 0.0024780555555555554 | 900 | 0.000016944444444444446 |

UKF-minus-Hybrid final-LER delta: `0.019950555556`.

After this run, the cumulative `linear_ramp` formal partial source data were:

- paired repeats: `6/12`
- repeat indices: `0,1,2,3,4,5`
- mean UKF-minus-Hybrid delta: `0.022425000000`
- sample SD of paired deltas: `0.001996339057`
- positive pairs: `6/6`

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_05_06_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_05_06_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_05_06_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_05_06_20260706/progress.jsonl`

Summary SHA-256: `2d04586b393cc2a1386374c899e4c2531e3f5d0ccd05b736b59fafd2c8e344e8`.

## 汇总同步

本次运行已通过 `build_submission_draft_phase_a_repeat_summary.py` 纳入：

- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`
- `投稿稿phase_a_repeat_summary记录.md`

`run_phase_a_paired_interval_analysis.py` 仍只输出 `static_bias_theta`，因为 `linear_ramp` 尚未达到 `12/12` completed scenario-row 条件。

## 可写边界

- 可以写：`linear_ramp` 的第六个 formal-length paired repeat 中，Hybrid Residual-B 的 final LER 低于 UKF，paired delta 为 `0.019950555556`。
- 可以写：该 run 后 `linear_ramp` 有 `6/12` formal partial source rows，六个 observed pairs 均为 positive UKF-minus-Hybrid delta；后续当前累计状态见 repeat06 记录。
- 不能写：`linear_ramp` 已完成 formal scenario row。
- 不能写：该结果提供 paired interval、confidence interval、p-value、all-scenario repeat-expanded advantage、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
