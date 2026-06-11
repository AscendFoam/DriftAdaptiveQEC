# Deployment Boundary Evidence Packs

本目录保存 `.tflite`、real-board gate、readiness checklist、transfer-pack provenance 相关文档。

## 文件清单

| 文件 | 来源任务 | 用途 |
| --- | --- | --- |
| `TFLite_runtime_bootstrap.md` | `T18` | `.tflite` export/runtime manifest and smoke plan |
| `t48_true_tflite_runtime_gate.md` | `T48` | current-host isolated true `.tflite` runtime gate |
| `real_board_hil_readiness.md` | `T20` | real-board HIL readiness checklist |
| `real_board_smoke_execution_plan.md` | `T22` | real-board smoke execution plan and AXI/DMA audit checklist |
| `t49_real_board_smoke_execution_gate.md` | `T49` | current-host real-board smoke execution gate; verdict remains `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE` |
| `t71_real_board_gate_regeneration_pack.md` | `T71` | checked-in read-only gate regeneration and host-transfer pack |
| `t72_real_board_transfer_pack_provenance_hardening.md` | `T72` | transfer-pack provenance hardening output |

## 边界

本目录不证明 real-board execution success。`T48` 只证明 current-host isolated true `.tflite` runtime；`T49/T71/T72` 只证明 gate/provenance/read-only transfer-pack 边界，不能升级为 HIL closure、deployment closure 或 true board validation。
