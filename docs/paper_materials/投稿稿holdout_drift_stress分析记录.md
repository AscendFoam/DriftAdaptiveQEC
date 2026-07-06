# 投稿稿 holdout drift stress 分析记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它补充三类未见漂移压力测试：random-walk drift、burst/reset drift 和 faster-than-window oscillatory drift。

## 生成文件

- `docs\paper_materials\submission_draft_holdout_drift_stress_analysis.csv`
- `docs\paper_materials\submission_draft_holdout_drift_stress_analysis.json`

## 协议

- random seed: `20260744`
- sequences per scenario: `384`
- steps per sequence: `512`
- lagged affine commit interval: `64` steps
- lagged affine commit lag: `32` steps
- methods: fixed affine, lagged affine, oracle affine, wrapped-Gaussian posterior mean, wrapped-Gaussian MAP
- metric: per-sequence half-lattice residual-boundary crossing proxy plus residual MSE

## 结果摘要

| Scenario | Fixed LER proxy | Lagged LER proxy | Oracle LER proxy | Wrapped mean LER proxy | Wrapped MAP LER proxy | Oracle F_avg_surr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random_walk_drift | 0.000127 | 0.000127 | 0.000127 | 0.000804 | 0.002268 | 0.999915 |
| burst_reset_drift | 0.000331 | 0.000331 | 0.000331 | 0.000921 | 0.002167 | 0.999780 |
| faster_than_window_oscillation | 0.000046 | 0.000046 | 0.000046 | 0.000336 | 0.001165 | 0.999969 |

## 可写边界

- 可以写：该分析提供了三类未见漂移压力测试，缓解但不完全关闭 holdout-drift 缺口。
- 可以写：`pauli_surrogate_average_fidelity` 是由 residual-boundary crossing rate 派生的 Pauli-channel-style surrogate，不是 finite-energy logical-channel fidelity。
- 可以写：`lagged_affine` 是慢提交 known-state 参数压力参考，不是当前 CNN residual branch。
- 不能写：该分析完成了正式 expanded benchmark、CI/p-value、真实硬件、完整 logical-channel fidelity 或 trained model generalization proof。
