# Figure Assets

本目录存放论文、汇报和审稿材料使用的图件资产。图件资产必须能回指来源数据、任务包或证据边界说明；不能把可视化本身当作新的实验事实。

## 当前登记

| 目录 | 用途 | 维护边界 |
| --- | --- | --- |
| `T58_fr6_multi_seed_mechanism_intervention/` | FR6 多 seed 机制/干预图件 | descriptive mechanism only，不支持 causal closure |
| `T74_paper_ready_simulation_result_pack/` | paper-ready simulation result pack | T74 stable-ID 结果包，不产生新实验事实 |
| `T75_maintext_results_authoring_pack/` | 主文结果 authoring 图件包 | T74 结果的 publication-facing 重排 |
| `T76_rendered_figure_qa_pack/` | T75 渲染 QA 预览 | presentation-only QA，不改变证据等级 |
| `submission_draft_python_figures/` | 当前独立投稿稿 Python 图件包 | 只可视化当前稿件表格、paired descriptive source data、validation-contract source summaries 和方法契约；不运行新实验 |

## 维护规则

- 新图件包应包含 README、来源映射或 manifest。
- 图件 caption 必须说明 validation boundary，尤其是 simulation、extension lane、planned hardware measurement、real-board `NO_GO` 等边界。
- 不得将历史 `runs/` 或 `artifacts/` 改写为新事实来源。
