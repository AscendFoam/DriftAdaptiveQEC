# 人类文档目录

这里是 `docs/` 顶层兼容文档与 `reports/` 的人类可读视图。高绑定原文继续保留旧路径，以免破坏生成脚本、测试、冻结 JSON/CSV、manifest 和历史引用；本目录按 current / phase / risk / legacy 重新组织阅读入口。

## 推荐阅读顺序

1. 从“当前入口”确认任务板、风险和实验计划。
2. 按 Phase 打开主题报告；表格中的 Task 是与任务板对齐的主任务 ID。
3. 需要机器证据时转到 [`../evidence_catalog/README.md`](../evidence_catalog/README.md)。
4. `legacy` 只用于追溯，不可覆盖当前状态源。

## 分类

| 分类 | 文档数 | 用途 |
| --- | ---: | --- |
| [当前入口](current/README.md) | 4 | 当前任务、风险、实验计划和文档导航。 |
| [规划来源](planning/README.md) | 1 | 冻结的原始规划或设计来源，不作为实时状态板。 |
| [Phase 0](phase0/README.md) | 2 | 研究范围、文献矩阵与问题定义。 |
| [Phase 1](phase1/README.md) | 5 | 主张、术语、参数和系统边界合同。 |
| [Phase 2](phase2/README.md) | 24 | 物理模型、协议、仿真和硬件边界。 |
| [Phase 3](phase3/README.md) | 16 | 解码 baseline、memory 与 oracle 对照。 |
| [Phase 4](phase4/README.md) | 17 | 混合慢/快回路、teacher/student 与故障恢复。 |
| [Phase 5](phase5/README.md) | 27 | 统一验证、因果消融、logical channel 与硬件 Pareto。 |
| [Phase 6](phase6/README.md) | 47 | Route-A、外部复现、RTL 和多证据 lane。 |
| [Phase 7](phase7/README.md) | 16 | 论文图表/章节合同与 reviewer response。 |
| [Phase 9](phase9/README.md) | 8 | 高保真双后端、raw-IQ 和三 lane 资格协议。 |
| [风险任务](risk/README.md) | 1 | 插入风险任务直接产生的顶层报告。 |
| [旧治理链](legacy/README.md) | 9 | 00—08 旧状态链，仅用于历史追溯和兼容引用。 |

## 专题目录

| 目录 | Markdown 数 | 角色 |
| --- | ---: | --- |
| [new_tasks/](../new_tasks/README.md) | 203 | 当前逐 task 完成记录 |
| [reports/](../reports/README.md) | 19 | 按 Phase 保存的新式独立报告 |
| [review/](../review/README.md) | 81 | 历史任务与里程碑 Review |
| [protocols/](../protocols/README.md) | 5 | 可执行协议和 benchmark 合同 |
| [paper_notes/](../paper_notes/README.md) | 5 | 论文草稿与装配入口 |
| [paper_materials/](../paper_materials/README.md) | 146 | 论文证据、表格和素材 |
| [paper_readers/](../paper_readers/README.md) | 3 | 离线论文阅读副本与翻译笔记 |
| [figures/](../figures/README.md) | 1 | 图件生成脚本与渲染产物 |
| [figure_assets/](../figure_assets/README.md) | 7 | 图件源数据与可编辑资产 |
| [for_human/](../for_human/README.md) | 83 | 通俗解释、答辩和审阅说明 |
| [deep_research_reports/](../deep_research_reports/README.md) | 7 | 深度调研报告 |
| [codebase_overview/](../codebase_overview/README.md) | 4 | 代码库结构说明 |
| [reference/](../reference/README.md) | 6 | 外部调研和工具参考 |
| [worker_summary/](../worker_summary/README.md) | 39 | 旧任务交接摘要 |
| [任务版改进记录/](../任务版改进记录/README.md) | 5 | 用于强化 task 设计的论文笔记 |
| [汇报用/](../汇报用/README.md) | 8 | 汇报材料 |
| [legacy_context/](../legacy_context/README.md) | 27 | 明确退役或迁移的历史材料 |
| [tasks/](../tasks/README.md) | 277 | 旧任务记录与兼容镜像 |

## 路径政策

- 未迁移的顶层 Markdown 是当前兼容层，不再作为人工浏览入口。
- 新 task 完成记录写入 `docs/new_tasks/`；新的独立人类报告写入 `docs/reports/phaseN/`。
- 只有解除代码、测试、机器证据和哈希绑定后，才物理迁移旧报告。
- 文档新增或状态变化后运行 `python scripts/build_document_catalog.py` 刷新本目录。
