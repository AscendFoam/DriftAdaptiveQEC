# 投稿稿 Phase A `linear_ramp` formal repeat 11 记录

日期：2026-07-06

## 运行对象

- Run directory: `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_11_12_20260706`
- Config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- Scenario: `linear_ramp`
- Modes: `ukf`, `hybrid_residual_b`
- Repeat filter: `--repeat-start 11 --repeat-stop 12`
- Paired seeds: `true`
- Expected scenario repeats: `12`

## 运行结果

| Mode | Repeat | Final LER | Overflow rate | Histogram-input saturation rate | Commit count | Fast-cycle violation rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ukf` | 11 | 0.8084191666666667 | 0.002459722222222222 | 0.002459722222222222 | 900 | 0.0000175 |
| `hybrid_residual_b` | 11 | 0.7883347222222222 | 0.0024444444444444444 | 0.0024444444444444444 | 900 | 0.0000175 |

UKF-minus-Hybrid final-LER delta: `0.020084444444`.

After this run, the cumulative `linear_ramp` formal source data are:

- paired repeats: `12/12`
- repeat indices: `0,1,2,3,4,5,6,7,8,9,10,11`
- mean UKF-minus-Hybrid delta: `0.022416805556`
- sample SD of paired deltas: `0.001891558035`
- positive pairs: `12/12`
- paired-\(t\) 95% interval: `[0.021214967006, 0.023618644106]`
- bootstrap 95% interval: `[0.021404645833, 0.023439867477]`

## Source files

- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_11_12_20260706/summary.json`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_11_12_20260706/comparison.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_11_12_20260706/delta.csv`
- `runs/paper_submission_phase_a/formal_probe_linear_ramp_ukf_hybrid_r12_11_12_20260706/progress.jsonl`
- `docs/paper_materials/submission_draft_phase_a_paired_interval_analysis.csv`
- `docs/paper_materials/submission_draft_phase_a_paired_interval_analysis.json`

Summary SHA-256: `bc4169064d9028887202da5c7c692b181aa558b675d0adba84a53a1ad5bf85d0`.

## 汇总同步

本次运行已通过以下脚本和派生文件同步：

- `build_submission_draft_phase_a_repeat_summary.py`
- `run_phase_a_paired_interval_analysis.py`
- `build_submission_draft_phase_a_upgrade_gate.py`
- `submission_draft_phase_a_repeat_summary.csv`
- `submission_draft_phase_a_repeat_summary.json`
- `submission_draft_phase_a_paired_interval_analysis.csv`
- `submission_draft_phase_a_paired_interval_analysis.json`
- `submission_draft_phase_a_upgrade_gate.csv`
- `submission_draft_phase_a_upgrade_gate.json`

`linear_ramp` 现在达到 `12/12` completed scenario-row 条件，并进入 paired interval source data。该状态只支持本 scenario 的 positive paired-interval check，不支持 all-scenario repeat-expanded advantage、pooled analysis、holdout robustness 或 hardware wording。

## 可写边界

- 可以写：`linear_ramp` formal-length `12/12` paired repeats 中，UKF-minus-Hybrid final-LER delta 均为 positive，mean delta 为 `0.022416805556`。
- 可以写：`linear_ramp` 的 paired-\(t\) 和 bootstrap 95% interval lower bounds 均为 positive。
- 不能写：Phase A all-scenario gate 已通过。
- 不能写：该 scenario-row 提供 p-value、pooled all-scenario inference、holdout robustness、finite-energy logical-channel fidelity、FPGA latency/resource/source-vs-board agreement 或 hardware validation。
