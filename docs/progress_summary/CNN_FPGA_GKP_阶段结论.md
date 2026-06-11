# CNN-FPGA-GKP 阶段结论（已退役索引）

**状态**：已退役  
**退役日期**：2026-06-11  
**当前替代入口**：`docs/02_experiment_plan.md` Part I  
**历史全文归档**：`docs/legacy_context/progress_summary_retired_2026-06-11/CNN_FPGA_GKP_阶段结论_历史全文.md`

## 为什么退役

原阶段结论文档创建于项目早期，主要记录 2026 年 3-4 月 P1-P4 的阶段性结果、参数调优、早期 `.tflite` 与 P3/P4 叙事。后续从 2026-05 起，项目进入恢复治理和受控开发流程，关键结论已经被任务包、review、证据包和 00-08 治理文档多次 supersede。

继续维护本文件会带来两个问题：

1. 它会与 `docs/02_experiment_plan.md` Part I 形成两份阶段结论入口。
2. 它包含较多旧路径、旧验收口径和旧主线判断，容易被误读为当前事实。

## 当前读取方式

- 当前阶段结论、高层时间线和 P0-P4 / T 系列关键转折，请读 `docs/02_experiment_plan.md` Part I。
- 当前后续开发计划，请读 `docs/02_experiment_plan.md` Part II。
- 当前唯一任务，请读 `docs/04_task_board.md` 和 `docs/07_handoff.md`。
- 历史全文只作为早期 P1-P4 叙事、汇报材料或旧实验细节的参考，不再作为当前事实入口。

## 维护规则

1. 不再向本文件追加新的阶段结论。
2. 不再把本文件列为当前研究背景或阶段结论的必读入口。
3. 如需恢复其中某个历史结论，必须先在 `docs/02_experiment_plan.md` 中按当前证据边界重写，再拆成任务包或论文材料。
4. 不得仅凭历史全文中的旧结论改写当前 `.tflite`、real-board、statcalib、paper-grade benchmark 或 HIL 状态。
