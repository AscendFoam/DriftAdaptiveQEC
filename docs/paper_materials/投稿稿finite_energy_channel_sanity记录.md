# 投稿稿 finite-energy channel sanity 记录

日期：2026-07-06

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它在同一组 controlled local-Gaussian states 上加入一个 finite-squeezing measurement-channel toy sanity check，用来缩小 residual-boundary surrogate 与 finite-energy logical-channel 之间的解释缺口。

## 生成文件

- `docs\paper_materials\submission_draft_finite_energy_channel_sanity.csv`
- `docs\paper_materials\submission_draft_finite_energy_channel_sanity.json`

## 协议

- random seed: `20260706`
- samples per state: `60000`
- finite-energy delta values: `0.18, 0.26, 0.34`
- syndrome model: `wrap(error) + Normal(0, sqrt(delta^2 + (1-eta)/(2 eta)))`
- compared methods: hard nearest-syndrome, fixed affine, oracle affine
- metric: q/p half-lattice logical-event probability plus surrogate average-fidelity readout

## 聚合结果

| delta | method | mean p_any | worst state | worst p_any | mean F_avg^surr |
| ---: | --- | ---: | --- | ---: | ---: |
| 0.18 | Hard nearest-syndrome | 0.000148 | `step_after_jump` | 0.000517 | 0.999901 |
| 0.18 | Fixed affine | 0.000148 | `step_after_jump` | 0.000517 | 0.999901 |
| 0.18 | Oracle affine | 0.000148 | `step_after_jump` | 0.000517 | 0.999901 |
| 0.26 | Hard nearest-syndrome | 0.000160 | `step_after_jump` | 0.000475 | 0.999893 |
| 0.26 | Fixed affine | 0.000129 | `step_after_jump` | 0.000458 | 0.999914 |
| 0.26 | Oracle affine | 0.000129 | `step_after_jump` | 0.000458 | 0.999914 |
| 0.34 | Hard nearest-syndrome | 0.001002 | `step_after_jump` | 0.001500 | 0.999332 |
| 0.34 | Fixed affine | 0.000162 | `step_after_jump` | 0.000617 | 0.999892 |
| 0.34 | Oracle affine | 0.000165 | `step_after_jump` | 0.000625 | 0.999890 |

## 可写边界

- 可以写：该 sanity check 在一个简化 finite-squeezing measurement channel 下比较了 hard nearest-syndrome、fixed affine 和 oracle affine 的 half-lattice logical-event probability。
- 可以写：该表比纯 residual-boundary surrogate 更接近 approximate-GKP channel language，但仍然只是 toy-channel bridge。
- 不能写：该表完成了 finite-energy GKP logical-channel tomography、process fidelity、真实物理器件 calibration、正式 software-HIL benchmark、统计显著性或硬件验证。
