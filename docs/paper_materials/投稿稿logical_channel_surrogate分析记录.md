# 投稿稿 logical-channel surrogate 分析记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它从受控 local-Gaussian 样本的 q/p residual half-lattice crossing 率构造一个 Pauli-channel-style surrogate，用于把 `final_ler` 类型的 residual-boundary proxy 和 channel 语言之间的关系讲清楚。

## 生成文件

- `docs\paper_materials\submission_draft_logical_channel_surrogate_analysis.csv`
- `docs\paper_materials\submission_draft_logical_channel_surrogate_analysis.json`

## 口径

- random seed: `20260703`
- samples per scenario: `120000`
- 输入：受控 oracle-affine / wrapped-Gaussian local-Gaussian 分析中的 q/p boundary crossing rate。
- 分解：`p_I=1-p_any`、`p_q_only=p_q-p_both`、`p_p_only=p_p-p_both`、`p_both=p_q+p_p-p_any`。
- 可选解释量：若把该分解仅当作 Pauli-channel-style surrogate，则 `F_avg_surr=(1+2 p_I)/3`。

## 结果摘要

| Scenario | Fixed p_any | Oracle p_any | Wrapped mean p_any | Wrapped MAP p_any | Oracle surrogate average fidelity |
| --- | ---: | ---: | ---: | ---: | ---: |
| static_bias_theta | 0.000000 | 0.000000 | 0.000000 | 0.000017 | 1.000000 |
| linear_ramp_midpoint | 0.000000 | 0.000000 | 0.000042 | 0.000183 | 1.000000 |
| step_after_jump | 0.000467 | 0.000467 | 0.002417 | 0.006450 | 0.999689 |
| periodic_high_phase | 0.000042 | 0.000042 | 0.000617 | 0.001867 | 0.999972 |

## 可写边界

- 可以写：该 surrogate 把 residual-boundary crossing 显式拆成 q-only、p-only、both 和 identity-like 分量，使 `final_ler` proxy 与 channel 语言之间的关系更透明。
- 可以写：surrogate average fidelity 只是在 Pauli-channel-style surrogate 下由 identity-like 分量诱导的解释量，不能与有限能量 GKP logical-channel fidelity 等同。
- 不能写：本分析完成了 logical-channel tomography、process fidelity estimation、finite-energy GKP channel simulation、硬件保真度测量或统计显著性证明。
