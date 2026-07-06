# 投稿稿 Phase A `linear_ramp` formal repeat 10 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_10_11_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 10 --repeat-stop 11`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 10 | 0.8129452777777778 | 0.002428888888888889 | 0.002428888888888889 | 900 | 0.00001638888888888889 |
| `hybrid_residual_b` | 10 | 0.7881052777777777 | 0.002436111111111111 | 0.002436111111111111 | 900 | 0.00001638888888888889 |

UKF-minus-Hybrid final-LER delta: `0.024840000000`.

After this run, the cumulative `linear_ramp` formal source data are:

- paired repeats: `11/12`
- repeat indices: `0,1,2,3,4,5,6,7,8,9,10`
- mean UKF-minus-Hybrid delta: `0.022628838384`
- sample SD of paired deltas: `0.001828208531`
- positive pairs: `11/11`

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_10_11_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_10_11_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_10_11_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_10_11_20260706/progress.jsonl`

Summary SHA-256: `db4e550abddd417c77575a508e1449c5d45f5d1a57bdde7d24e9b911108f5f41`.

## 汇总同步

本次运行已通过 `build_submission_draft_phase_a_repeat_summary.py` 纳入：

- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`

`linear_ramp` 在本 run 后仍未达到 `12/12` completed scenario-row 条件，不能单独写成 paired interval 或 all-scenario gate。

## 可写边界

- 可以写：`linear_ramp` 的第十一个 formal-length paired repeat 中，Hybrid Residual-B 的 final LER 低于 UKF，paired delta 为 `0.024840000000`。
- 可以写：`linear_ramp` 在本 run 后累计 `11/12` formal source rows，十一个 observed pairs 均为 positive UKF-minus-Hybrid delta。
- 不能写：`linear_ramp` 在本 run 后已经完成 formal scenario row。
- 不能写：该单次 run 提供 confidence interval、p-value、all-scenario repeat-expanded advantage、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
