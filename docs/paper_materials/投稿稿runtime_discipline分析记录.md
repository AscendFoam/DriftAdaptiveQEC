# 投稿稿 runtime-discipline 分析记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它从 preserved comparison CSV 派生 software-in-the-loop runtime counters，用于说明稿件中 stage-and-commit / saturation / cycle-violation 口径的当前可写边界。

## 生成文件

- `docs\paper_materials\submission_draft_runtime_discipline_summary.csv`
- `docs\paper_materials\submission_draft_runtime_discipline_summary.json`

## 结果摘要

| Mode | Commits applied | Slow-update violation | Fast-cycle violation | Overflow | Correction saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant Residual-Mu | 899.8 | 0 | 1.57639e-05 | 0.00258243 | 0 |
| EKF | 899.8 | 0 | 1.57639e-05 | 0.00255882 | 0 |
| Hybrid Residual-B | 899.9 | 0 | 1.57639e-05 | 0.00253604 | 0 |
| RLS Residual-B | 899.8 | 0 | 1.57639e-05 | 0.00253788 | 0 |
| UKF | 899.8 | 0 | 1.57639e-05 | 0.00257146 | 0 |

## 可写边界

- 可以写：preserved software-in-the-loop comparison rows include runtime counters for commit activity, slow-update violation, fast-cycle violation, overflow and saturation.
- 可以写：这些 counters 支持 stage-and-commit contract 在软件协议中的可观测性。
- 不能写：这些 counters 是 board commit latency、hardware reliability、rollback proof、source-vs-board agreement 或 FPGA timing/resource evidence。
