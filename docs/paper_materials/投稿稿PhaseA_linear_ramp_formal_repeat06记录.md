# 投稿稿 Phase A `linear_ramp` formal repeat 06 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_06_07_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 6 --repeat-stop 7`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 6 | 0.8136075 | 0.0024380555555555557 | 0.0024380555555555557 | 900 | 0.000016666666666666667 |
| `hybrid_residual_b` | 6 | 0.7893744444444445 | 0.002416666666666667 | 0.002416666666666667 | 900 | 0.000016666666666666667 |

UKF-minus-Hybrid final-LER delta: `0.024233055556`.

After this run, the cumulative `linear_ramp` formal partial source data are:

- paired repeats: `7/12`
- repeat indices: `0,1,2,3,4,5,6`
- mean UKF-minus-Hybrid delta: `0.022683293651`
- sample SD of paired deltas: `0.001946317196`
- positive pairs: `7/7`

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_06_07_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_06_07_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_06_07_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_06_07_20260706/progress.jsonl`

Summary SHA-256: `15c8a184d5c4acddfe2589b10a3d9958431bad660b9f5f9d7bc789450accc41a`.

## 汇总同步

本次运行已通过 `build_submission_draft_phase_a_repeat_summary.py` 纳入：

- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`
- `投稿稿phase_a_repeat_summary记录.md`

`run_phase_a_paired_interval_analysis.py` 仍只输出 `static_bias_theta`，因为 `linear_ramp` 尚未达到 `12/12` completed scenario-row 条件。

## 可写边界

- 可以写：`linear_ramp` 的第七个 formal-length paired repeat 中，Hybrid Residual-B 的 final LER 低于 UKF，paired delta 为 `0.024233055556`。
- 可以写：该 run 后 `linear_ramp` 有 `7/12` formal partial source rows，七个 observed pairs 均为 positive UKF-minus-Hybrid delta；后续当前累计状态见 repeat08 记录。
- 不能写：`linear_ramp` 已完成 formal scenario row。
- 不能写：该结果提供 paired interval、confidence interval、p-value、all-scenario repeat-expanded advantage、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
