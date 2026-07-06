# 投稿稿 fixed-point 一致性分析记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它用 `LinearRuntime` 的 Q4.20 fixed-point emulation 对比 floating-point affine fast path，检查固定点量化是否在受控 local-Gaussian 样本上引入可见退化。

它不是 FPGA synthesis、不是 timing closure、不是 resource/power 测量、不是 real-board execution，也不是 logical-channel fidelity 分析。

## 生成文件

- `docs\paper_materials\submission_draft_fixed_point_parity_analysis.csv`
- `docs\paper_materials\submission_draft_fixed_point_parity_analysis.json`

## 协议

- random seed: `20260703`
- samples per scenario: `80000`
- scenarios: static bias/theta, linear-ramp midpoint, post-step state, periodic high-phase state
- runtime pair: floating-point affine fast path vs Q4.20 fixed-point emulation

## 结果摘要

| Scenario | Max abs diff | p99 abs diff | MSE delta | Crossing delta | Quant sat. |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_bias_theta | 0.000001011 | 0.000000778 | 1.428313232e-10 | 0.000000000e+00 | 0.000000 |
| linear_ramp_midpoint | 0.000000993 | 0.000000797 | -2.211748999e-09 | 0.000000000e+00 | 0.000000 |
| step_after_jump | 0.000001605 | 0.000001130 | 1.156396812e-09 | 0.000000000e+00 | 0.000000 |
| periodic_high_phase | 0.000001335 | 0.000000951 | 9.640101495e-10 | 0.000000000e+00 | 0.000000 |

## 可写边界

- 可以写：在受控样本和当前 affine fast-path 参数范围内，Q4.20 emulation 与 floating-point 输出的 correction 差异处于约一个 quantization step 的量级，未产生可见的 residual-boundary crossing 退化。
- 可以写：该结果支持 fixed-point feasibility 的软件数值一致性动机。
- 不能写：该结果已经证明 FPGA timing closure、LUT/FF/DSP/BRAM、power、source-vs-board agreement、real-board latency 或 logical-channel fidelity。
