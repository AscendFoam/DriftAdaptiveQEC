# 投稿稿 Phase A `linear_ramp` formal repeat 07 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_07_08_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 7 --repeat-stop 8`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 7 | 0.8091938888888889 | 0.0024580555555555553 | 0.0024580555555555553 | 900 | 0.000012777777777777777 |
| `hybrid_residual_b` | 7 | 0.7864475 | 0.0024630555555555556 | 0.0024630555555555556 | 900 | 0.000012777777777777777 |

UKF-minus-Hybrid final-LER delta: `0.022746388889`.

After this run, the cumulative `linear_ramp` formal partial source data are:

- paired repeats: `8/12`
- repeat indices: `0,1,2,3,4,5,6,7`
- mean UKF-minus-Hybrid delta: `0.022691180556`
- sample SD of paired deltas: `0.001802077656`
- positive pairs: `8/8`

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_07_08_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_07_08_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_07_08_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_07_08_20260706/progress.jsonl`

Summary SHA-256: `238d4b952d7e7fe51cce642ad7e3c927ac11281e6e737616a68f5c5375c00f85`.

## 汇总同步

本次运行已通过 `build_submission_draft_phase_a_repeat_summary.py` 纳入：

- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`
- `投稿稿phase_a_repeat_summary记录.md`

`run_phase_a_paired_interval_analysis.py` 仍只输出 `static_bias_theta`，因为 `linear_ramp` 尚未达到 `12/12` completed scenario-row 条件。

## 可写边界

- 可以写：`linear_ramp` 的第八个 formal-length paired repeat 中，Hybrid Residual-B 的 final LER 低于 UKF，paired delta 为 `0.022746388889`。
- 可以写：该 run 后 `linear_ramp` 有 `8/12` formal partial source rows，八个 observed pairs 均为 positive UKF-minus-Hybrid delta；后续当前累计状态见 repeat08 记录。
- 不能写：`linear_ramp` 已完成 formal scenario row。
- 不能写：该结果提供 paired interval、confidence interval、p-value、all-scenario repeat-expanded advantage、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
