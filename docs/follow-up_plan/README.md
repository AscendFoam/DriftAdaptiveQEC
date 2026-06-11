# follow-up_plan 目录说明（已退役）

自 2026-06-11 起，后续开发计划的唯一维护入口改为：

- `docs/02_experiment_plan.md` 的 **Part II：后续开发计划**

本目录不再承担“后续计划 README”的职责，也不再接收新的路线、实验计划、论文提纲或任务候选池内容。旧 `follow-up_plan/README.md` 中已融合的内容已经吸收到 `docs/02_experiment_plan.md`，包括：

- 原四份后续计划文档的可复用内容
- reference 归档文档中的有效建议摘要
- `GPT-Pro` 深度调研报告中可转化为后续任务的建议
- `.tflite`、real-board、statcalib、paper、sidecar extension 等后续路线边界

## 维护规则

1. 新的后续计划、任务候选池和投稿路线更新，统一写入 `docs/02_experiment_plan.md`。
2. 论文 note、LaTeX 草稿和保留的编译产物，统一放在 `docs/paper_notes/`。
3. 深度研究报告统一放在 `docs/deep_research_reports/`，并由该目录的 README 标记可复用内容与过时内容。
4. 历史任务包或 review 中提到 `docs/follow-up_plan/**` 的 forbidden scope / legacy path 时，可继续按历史语境理解，不代表本目录仍是当前计划入口。
5. 如需恢复旧建议，必须先同步到 `docs/02_experiment_plan.md` 的后续计划部分，再拆成带 `Allowed files`、`Forbidden scope`、`Verification` 的独立任务包。
