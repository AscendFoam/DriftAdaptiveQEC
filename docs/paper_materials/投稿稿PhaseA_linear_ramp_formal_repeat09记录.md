# 投稿稿 Phase A `linear_ramp` formal repeat 09 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_09_10_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 9 --repeat-stop 10`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 9 | 0.8079730555555555 | 0.002406111111111111 | 0.002406111111111111 | 900 | 0.000016666666666666667 |
| `hybrid_residual_b` | 9 | 0.7877236111111111 | 0.0023880555555555556 | 0.0023880555555555556 | 900 | 0.000016666666666666667 |

UKF-minus-Hybrid final-LER delta: `0.020249444444`.

After this run, the cumulative `linear_ramp` formal source data are:

- paired repeats: `10/12`
- repeat indices: `0,1,2,3,4,5,6,7,8,9`
- mean UKF-minus-Hybrid delta: `0.022407722222`
- sample SD of paired deltas: `0.001765260606`
- positive pairs: `10/10`

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_09_10_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_09_10_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_09_10_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_09_10_20260706/progress.jsonl`

Summary SHA-256: `ae75eafd78bfb0edc07bf46656181ef619ce1c458b4e62f88853414655545635`.

## 汇总同步

本次运行已通过 `build_submission_draft_phase_a_repeat_summary.py` 纳入：

- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`

`linear_ramp` 在本 run 后仍未达到 `12/12` completed scenario-row 条件，不能单独写成 paired interval 或 all-scenario gate。

## 可写边界

- 可以写：`linear_ramp` 的第十个 formal-length paired repeat 中，Hybrid Residual-B 的 final LER 低于 UKF，paired delta 为 `0.020249444444`。
- 可以写：`linear_ramp` 在本 run 后累计 `10/12` formal source rows，十个 observed pairs 均为 positive UKF-minus-Hybrid delta。
- 不能写：`linear_ramp` 在本 run 后已经完成 formal scenario row。
- 不能写：该单次 run 提供 confidence interval、p-value、all-scenario repeat-expanded advantage、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
