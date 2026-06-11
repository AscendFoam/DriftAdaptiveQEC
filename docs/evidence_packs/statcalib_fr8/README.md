# StatCalib / FR8 Evidence Packs

本目录保存 `statcalib` separate extension lane 及 FR8 相关任务产物。

## 文件清单

| 文件 | 来源任务 | 用途 |
| --- | --- | --- |
| `statcalib_feasibility_gate.md` | `T26` | 判断 `statcalib` 只能作为 separate comparator lane 的 feasibility gate |
| `statcalib_comparator_lane_smoke.md` | `T59` | separate lane integration + bounded smoke 产物 |
| `statcalib_lane_isolation_and_regression_hardening.md` | `T60` | cross-mode semantics isolation 与 regression hardening 总结 |
| `statcalib_fairness_sanity.md` | `T61` | provenance 有缺陷的 fairness sanity 记录，保留作历史边界 |
| `statcalib_provenance_isolated_fairness_rerun.md` | `T62` | provenance-isolated fairness rerun 记录 |
| `fr8_statcalib_comparator_gate_review.md` | `T63` | 是否打开 bounded FR8 task 的 gate review |
| `fr8_statcalib_extension_lane_benchmark.md` | `T64` | bounded extension-lane benchmark 结果 |
| `fr8_statcalib_extension_lane_consistency_audit.md` | `T65` | T64 report/artifact consistency audit |
| `statcalib_sensitivity_bounded_benchmark.md` | `T66` | local heuristic sensitivity grid |
| `statcalib_teacher_anchor_bounded_benchmark.md` | `T67` | teacher-anchor dependence bounded benchmark |
| `statcalib_generated_only_robustness_bounded_benchmark.md` | `T68` | generated-only robustness bounded benchmark |
| `statcalib_clean_winner_tiebreak_bounded_benchmark.md` | `T69` | clean-winner tie-break bounded benchmark |
| `fr8_statcalib_bounded_closure_pack.md` | `T70` | bounded closure pack and promotion/no-promotion gate |

## 边界

`T64-T70` 只能支持 `statcalib` 作为 separately labeled extension lane。当前权威结论仍是 `T70` 的 no-promotion / no-unique-threshold closure，不得把本目录改写成正式主线 comparator promotion。
