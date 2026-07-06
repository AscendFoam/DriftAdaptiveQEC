# 投稿稿 oracle-affine 受控分析记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的投稿稿补强。分析目标是补一个小规模、可复现的 oracle-affine / wrapped-Gaussian sanity check，用于区分 affine fast-path 的局部模型上限、slow-loop estimator 误差和 wrapped posterior 参照。它不是正式 P4 benchmark、不是 sequence-level wrapped decoder、不是 holdout drift、不是硬件测量，也不改变已有主结果的证据等级。

## 生成文件

- `docs\paper_materials\submission_draft_controlled_oracle_affine_analysis.csv`
- `docs\paper_materials\submission_draft_controlled_oracle_affine_analysis.json`

## 协议

- random seed: `20260703`
- samples per scenario: `120000`
- scenarios: static bias/theta, linear-ramp midpoint, post-step state, periodic high-phase state
- methods: no correction, nearest-syndrome, fixed nominal affine, oracle affine, wrapped-Gaussian posterior mean, wrapped-Gaussian MAP
- metric: one-step residual MSE and half-lattice residual-boundary crossing rate

## 结果摘要

| Scenario | Fixed MSE | Oracle MSE | Wrapped mean MSE | Wrapped MAP MSE |
| --- | ---: | ---: | ---: | ---: |
| static_bias_theta | 0.049199 | 0.048095 | 0.047955 | 0.048203 |
| linear_ramp_midpoint | 0.061417 | 0.060251 | 0.062341 | 0.063105 |
| step_after_jump | 0.108216 | 0.092541 | 0.105815 | 0.110339 |
| periodic_high_phase | 0.081757 | 0.076028 | 0.082349 | 0.084002 |

## 可写边界

- 可以写：在受控 local-Gaussian setting 中，oracle affine 参数通常降低 residual MSE，说明 affine fast path 本身有可解释的局部上限。
- 可以写：wrapped-Gaussian posterior mean 在本受控一步设置中给出混合结果；它只在 static state 略优，其他状态并未优于 oracle affine，说明正式 wrapped baseline 需要独立协议和调参。
- 不能写：该分析已经补齐正式 benchmark、统计显著性、holdout drift、logical-channel fidelity 或硬件证据。
