# 投稿稿 fast-path cost 模型记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它给出 per-shot analytical operation-count，用于支撑稿件中 deterministic affine fast path 的低复杂度定位。

它不是 FPGA synthesis、不是 timing closure、不是 power/resource 测量，也不是 hardware result。所有真实硬件 latency/resource 仍必须由后续板级实验给出。

## 生成文件

- `docs\paper_materials\submission_draft_fast_path_cost_model.csv`
- `docs\paper_materials\submission_draft_fast_path_cost_model.json`

## 结果摘要

| Decoder | Branches | Mult. | Add. | Nonlinear ops | Stored scalars |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affine fast path | 1 | 4 | 4 | 0 | 6 |
| Wrapped MAP, 3x3 branches | 9 | 49 | 40 | 0 | 33 |
| Wrapped posterior mean, 3x3 branches | 9 | 99 | 98 | 18 | 33 |

## 可写边界

- 可以写：affine fast path 的 per-shot arithmetic 和 state footprint 明显小于 3x3 branch wrapped-Gaussian posterior references。
- 可以写：该表支持低延迟/低资源的工程动机，但不等于真实 FPGA timing 或 resource measurement。
- 不能写：已经完成 hardware latency、LUT/FF/DSP/BRAM、power、source-vs-board agreement 或 timing closure。
