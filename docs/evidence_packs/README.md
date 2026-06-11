# Evidence Packs

本目录统一存放已经完成的任务产物、结果包、gate 输出和证据边界说明。它用于减少 `docs/` 根目录堆积，不改变任何实验结论。

## 子目录

| 子目录 | 内容 | 边界 |
| --- | --- | --- |
| `statcalib_fr8/` | `T26/T59-T70` 的 `statcalib` / FR8 extension-lane 证据链 | 仅是 mock-backed software-HIL extension lane；不得写成 mature comparator、`.tflite` 或 real-board evidence |
| `mechanism_ablation/` | `T36/T38/T46/T54-T58` 的机制诊断、multi-seed、intervention、FR6/FR7 ablation 材料 | 只能支持 bounded diagnostic / ablation wording；不得写成 causal proof |
| `deployment_boundary/` | `.tflite`、real-board gate、transfer-pack provenance、board-readiness 相关证据包 | 不等于 HIL closure、deployment closure 或 real-board execution success |
| `training_reproducibility/` | 训练链依赖、clean-env smoke、材料再生证据包 | 不等于 full training reproducibility、GPU/CUDA/Linux portability |
| `repo_hygiene/` | tracked cache / repo noise cleanup 相关 manifest | 不覆盖 `runs/` / `artifacts/` 历史结果清理 |

## 使用规则

1. 这里的文档是证据材料或任务产物，不是当前唯一计划入口。
2. 当前唯一任务仍以 `docs/04_task_board.md` 和 `docs/07_handoff.md` 为准。
3. 论文或计划引用这里的内容时，必须同时保留对应 task、review、run root、artifact 或 helper 边界。
4. 不要把本目录中的 bounded smoke、mock-backed result、gate/readiness pack 外推成更高证据等级。
