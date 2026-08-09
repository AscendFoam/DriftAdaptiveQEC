# 文档导航

`docs/` 同时保存项目治理、任务记录、论文材料和可复核实验资产。部分顶层 JSON/CSV 被代码、测试、LFS 规则或证据哈希按路径绑定，因此本目录采用“稳定路径 + 明确入口”的方式整理；不要仅为了目录外观批量搬动这些文件。

## 从这里开始

| 目的 | 入口 | 说明 |
| --- | --- | --- |
| 查看当前任务 | [`new_task_board.md`](new_task_board.md) | 新任务序列的状态源与当前推荐任务 |
| 查看当前风险 | [`new_risks.md`](new_risks.md) | 风险等级、插入任务与阻塞条件 |
| 查看实验计划 | [`experiment_plan.md`](experiment_plan.md) | 新任务序列的实验计划与后续修订 |
| 查看任务完成记录 | [`new_tasks/`](new_tasks/) | 当前任务体系的逐任务记录 |
| 按主题浏览人类文档 | [`document_catalog/README.md`](document_catalog/README.md) | 汇合顶层报告、完成记录、Review 和机器证据 |
| 浏览机器证据 | [`evidence_catalog/README.md`](evidence_catalog/README.md) | 按 Phase、Milestone 和风险任务分类的 275 个 JSON/CSV 索引 |
| 查看论文材料 | [`paper_notes/README.md`](paper_notes/README.md) | 论文 note、写作边界和装配入口 |
| 查看阶段快照 | [`00_project_snapshot.md`](00_project_snapshot.md) | 旧治理链的项目快照；用于历史接力 |
| 查看历史材料 | [`legacy_context/README.md`](legacy_context/README.md) | 已退役计划、早期分析和迁移快照 |

## 目录地图

### 当前执行与治理

- `new_task_board.md`、`new_risks.md`、`experiment_plan.md`：当前新任务序列的状态、风险和计划。
- `new_tasks/`：当前任务体系的完成记录；新增记录只放这里。
- `review/`：审阅记录与结论。
- `protocols/`：benchmark 和执行协议。
- `00_project_snapshot.md`—`08_risks_and_open_questions.md`：较早的治理链，保留用于追溯。

### 论文与人类可读材料

- `document_catalog/`：顶层 Markdown 的主阅读入口，使用 `CURRENT`、`REFERENCE`、`FROZEN`、`LEGACY` 四态区分权威性。
- `reports/`：后续独立报告的规范位置；首批低绑定 Phase 4/6 报告已迁入。
- `paper_notes/`、`paper_materials/`、`paper_readers/`：论文草稿、证据合同和读者材料。
- `for_human/`、`deep_research_reports/`、`汇报用/`：解释性材料、深度调研和汇报材料。
- `专利/`：专利相关材料。

### 证据与图表

- `evidence_catalog/`：顶层冻结 JSON/CSV 的人类可读目录；优先从这里找证据，不要浏览顶层文件名。
- `evidence/`：后续任务的新机器证据目录，按 `phaseN/milestone_N_M/` 分层。
- `evidence_packs/`：按证据类型整理的完成包和 gate 输出。
- `figure_assets/`、`figures/`：可编辑图资产、生成脚本和渲染结果。
- `t*.json`、`t*.csv`：既有冻结机器合同、验证摘要或 Source Data；物理路径暂时保留，通过 `evidence_catalog/` 阅读。

### 历史与兼容目录

- `legacy_context/`：明确退役的计划、规则和快照。
- `tasks/`：旧任务记录与兼容引用；不再新增文件。
- `reference/`、`reality_recovery/`、`recovery_bootstrap/`：历史工作流和恢复期材料。

## 文件放置规则

1. 新任务记录写入 `new_tasks/`，不要同时复制到 `tasks/`。
2. 新的临时 smoke/探针文件不要写入 `docs/`；使用系统临时目录或被忽略的 `docs/_tmp*`。
3. 新机器证据写入 `evidence/phaseN/milestone_N_M/`，不要继续增加顶层 JSON/CSV；命名和迁移门槛见 [`evidence/README.md`](evidence/README.md)。
4. 新的独立任务报告写入 `reports/phaseN/`，不要继续增加顶层 Markdown；与 task 对应的完成记录写入 `new_tasks/`。
5. 移动既有顶层 JSON/CSV 前，必须同时检查代码默认路径、测试、`.gitattributes`、LFS、receipt/manifest 和哈希绑定，并重新封存相关证据。
6. 一次性迁移清单、旧规则和过期计划放入 `legacy_context/`，不要留在仓库根目录。

## 编辑器视图

VS Code 默认隐藏 `docs/` 顶层的小写 Markdown、旧 `00`—`08` 治理链和 JSON/CSV；`README.md` 与各分类目录保持可见。隐藏只影响文件树展示，不删除文件；所有材料仍可从本页、`document_catalog/` 和 `evidence_catalog/` 打开。
