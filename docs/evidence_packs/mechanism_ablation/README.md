# Mechanism And Ablation Evidence Packs

本目录保存机制诊断、multi-seed trace/intervention、FR6/FR7 ablation 相关任务产物。

## 文件清单

| 文件 | 来源任务 | 用途 |
| --- | --- | --- |
| `seed20260429_failure_diagnosis.md` | `T36` | `seed=20260429` failure mechanism diagnosis |
| `seed20260429_trace_export_diagnosis.md` | `T38` | single-seed trace export diagnosis |
| `seed_mechanism_multi_seed_plan.md` | `T46` | multi-seed / intervention evidence plan |
| `multi_seed_trace_generalization_probe.md` | `T54` | multi-seed trace-only generalization probe |
| `multi_seed_i1_intervention_probe.md` | `T55` | I1 residual-clip intervention probe |
| `post_t55_mechanism_claim_reframing_gate.md` | `T56` | post-I1 mechanism claim reframing gate |
| `fr7_feature_teacher_ablation_reexecution.md` | `T57` | FR7 feature/teacher ablation re-execution |
| `fr6_multi_seed_mechanism_intervention_figure_pack.md` | `T58` | FR6 figure pack and caption material |

## 边界

这些文档可以支撑 bounded mechanism / ablation 叙事，但不能写成完整 causal proof。尤其 `T55/T56` 已经削弱了简单的 high committed-`b` harmful 叙事，后续论文表述必须保留 hedge。
