# 投稿稿 GKP boundary-sensitivity 分析记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它用 zero-mean Gaussian residual 的 half-lattice crossing 公式，连接 residual scale、boundary-crossing probability 和 Pauli-channel-style surrogate fidelity 语言。

它不是 finite-energy GKP logical-channel simulation，不是 process tomography，不是硬件测量，也不重跑 benchmark。

## 生成文件

- `docs\paper_materials\submission_draft_gkp_boundary_sensitivity.csv`
- `docs\paper_materials\submission_draft_gkp_boundary_sensitivity.json`

## 结果摘要

| sigma | squeezing dB | one-quadrature crossing | any q/p crossing | surrogate infidelity |
| ---: | ---: | ---: | ---: | ---: |
| 0.15 | 16.48 | 0.000000 | 0.000000 | 0.000000 |
| 0.20 | 13.98 | 0.000000 | 0.000000 | 0.000000 |
| 0.25 | 12.04 | 0.000001 | 0.000001 | 0.000001 |
| 0.30 | 10.46 | 0.000029 | 0.000059 | 0.000039 |
| 0.35 | 9.12 | 0.000342 | 0.000685 | 0.000456 |
| 0.40 | 7.96 | 0.001729 | 0.003454 | 0.002303 |

## 可写边界

- 可以写：在 Gaussian residual approximation 下，residual scale 接近 half-lattice boundary 时 crossing probability 和 surrogate infidelity 会迅速上升。
- 可以写：该表解释为什么降低 residual MSE 与降低 \\finalLER{} proxy 相关，但二者不是同一个指标。
- 不能写：该表估计了 finite-energy logical-channel fidelity、process fidelity、outer-code LER 或硬件 logical error rate。
