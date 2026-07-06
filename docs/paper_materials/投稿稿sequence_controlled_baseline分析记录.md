# 投稿稿 sequence-controlled baseline 分析记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它把 one-step oracle-affine / wrapped-Gaussian sanity check 扩展到短序列 controlled local-Gaussian drift setting。

它不是正式 P4 benchmark、不是 holdout drift、不是 confidence interval run、不是硬件测量，也不改变已有主结果证据等级。

## 生成文件

- `docs\paper_materials\submission_draft_sequence_controlled_baseline_analysis.csv`
- `docs\paper_materials\submission_draft_sequence_controlled_baseline_analysis.json`

## 协议

- random seed: `20260703`
- sequences per scenario: `384`
- steps per sequence: `512`
- scenarios: static bias/theta, linear ramp, step sigma/theta, periodic drift
- methods: nearest syndrome, fixed affine, oracle affine, wrapped-Gaussian posterior mean, wrapped-Gaussian MAP
- metric: sequence mean of half-lattice residual-boundary crossing proxy plus residual MSE

## 结果摘要

| Scenario | Nearest syndrome | Fixed | Oracle | Wrapped mean | Wrapped MAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_bias_theta | 0.000214 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| linear_ramp | 0.000264 | 0.000015 | 0.000015 | 0.000158 | 0.000575 |
| step_sigma_theta | 0.000580 | 0.000310 | 0.000310 | 0.001221 | 0.003220 |
| periodic_drift | 0.000229 | 0.000005 | 0.000005 | 0.000127 | 0.000519 |

## 可写边界

- 可以写：在 controlled sequence setting 中，oracle affine 通常改善 fixed affine；wrapped-Gaussian posterior mean/MAP 并未稳定支配 oracle affine。
- 可以写：该结果比 one-step sanity check 更接近 sequence-level baseline，但仍然只是受控 local-Gaussian positioning analysis。
- 不能写：该分析已经补齐正式 benchmark、CI、holdout drift、logical-channel fidelity 或硬件证据。
