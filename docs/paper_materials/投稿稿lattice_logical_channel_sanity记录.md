# 投稿稿 lattice logical-channel sanity 记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它只把已有 residual-boundary Pauli-style surrogate 按方法聚合成 lattice-level sanity summary，用于给审稿人一个更清楚的 `p_any` / `F_avg^surr` 读数入口。

## 生成文件

- `docs\paper_materials\submission_draft_lattice_logical_channel_sanity.csv`
- `docs\paper_materials\submission_draft_lattice_logical_channel_sanity.json`

## 结果摘要

| Method | Mean p_any | Worst state | Worst p_any | Mean F_avg^surr | Worst F_avg^surr |
| --- | ---: | --- | ---: | ---: | ---: |
| Fixed affine | 0.000127 | `step_after_jump` | 0.000467 | 0.999915 | 0.999689 |
| Oracle affine | 0.000127 | `step_after_jump` | 0.000467 | 0.999915 | 0.999689 |
| Wrapped mean | 0.000769 | `step_after_jump` | 0.002417 | 0.999487 | 0.998389 |
| Wrapped MAP | 0.002129 | `step_after_jump` | 0.006450 | 0.998581 | 0.995700 |

## 可写边界

- 可以写：该表把四个 controlled local-Gaussian states 上的 residual-boundary surrogate 聚合成方法级 mean / worst-state sanity summary。
- 可以写：`F_avg^surr` 只是由 `p_I^surr=1-p_any` 推出的 Pauli-style surrogate readout，用于审稿可追溯性。
- 不能写：该表完成了 finite-energy GKP logical-channel fidelity、process tomography、hardware fidelity、outer-code logical-error estimate 或正式 expanded benchmark。
