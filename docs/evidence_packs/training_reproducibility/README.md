# Training Reproducibility Evidence Packs

本目录保存训练链依赖、clean-environment bootstrap、minimal training smoke 和材料再生证据包。

## 文件清单

| 文件 | 来源任务 | 用途 |
| --- | --- | --- |
| `training_chain_bootstrap.md` | `T17` | training entrypoint and local bootstrap boundary |
| `training_chain_portable_dependency_lock_plan.md` | `T31` | portable dependency-lock plan and interpreter inventory |
| `training_chain_cpu_cleanenv_bootstrap.md` | `T39` | CPU-only clean-env draft lock and import/dry-run bootstrap |
| `training_chain_cpu_cleanenv_train_smoke.md` | `T40` | CPU-only clean-env minimal real-training smoke |
| `training_reproducibility_and_material_regeneration_pack.md` | `T50` | canonical material references and bounded train/eval rerun pack |

## 边界

这些文档支持 bounded training/material reproducibility story，但不证明 full training reproducibility、repeated-run stability、cross-host portability、GPU/CUDA portability、Linux portability、`.tflite` correctness 或 deployment closure。
