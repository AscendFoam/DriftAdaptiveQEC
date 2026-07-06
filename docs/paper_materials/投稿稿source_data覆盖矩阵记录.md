# 投稿稿 source-data 覆盖矩阵记录

日期：2026-07-03

本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`，用于把 source-data coverage 从叙述性说明转为可审计矩阵。

## 生成文件

- `docs\paper_materials\submission_draft_source_data_coverage_matrix.csv`
- `docs\paper_materials\submission_draft_source_data_coverage_matrix.json`

## 覆盖矩阵

| Coverage group | Status | Boundary |
| --- | --- | --- |
| `main_performance_tables` | mechanically_checked_against_source_csv | not CI/p-value/significance; not expanded benchmark |
| `controlled_diagnostics` | mechanically_checked_against_generated_csv | not tuned nearest-lattice decoder; nearest-syndrome rows are sanity references only; not trained-branch holdout proof; not calibrated finite-energy logical-channel tomography; not hardware latency |
| `implementation_feasibility_tables` | mechanically_checked_against_generated_csv | not FPGA synthesis/timing/resource/power; not source-vs-board agreement |
| `source_and_literature_maps` | mechanically_checked_for_row_hash_and_citation_consistency | not a leaderboard; not recursive historical-run hash closure; not full reproducibility proof |
| `formal_phase_a_interval` | generated_from_completed_formal_phase_a_summary | not all-scenario repeat-expanded evidence; not p-value evidence; not holdout robustness; not hardware validation |
| `planning_and_boundary_tables` | bounded_planning_or_boundary_surface | not current result evidence; not hardware validation; not submission-completeness proof |

## 可写边界

- 可以写：当前稿件的 selected tables/source files 有机械一致性审计和 file-level manifest。
- 可以写：availability、hardware plan、runtime-artifact scope 等表格是 boundary/planning surfaces，不是结果证据。
- 不能写：该矩阵完成了 full reproducibility、recursive run-directory hash closure、CI/p-value、expanded benchmark 或 hardware validation。
