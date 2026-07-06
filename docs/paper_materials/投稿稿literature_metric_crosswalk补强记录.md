# 投稿稿 literature metric crosswalk 补强记录

日期：2026-07-03

本文档记录 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的外部比较可追溯性补强。本轮不新增实验、不改变主结果数字、不升级硬件或统计证据等级。

## 本轮补强内容

1. 新增 `build_submission_draft_literature_metric_crosswalk.py`，从本地 `zotero_literature_review_cards.md` 已整理的指标锚点中生成投稿稿外部比较交叉表。
2. 新增 `submission_draft_literature_metric_crosswalk.csv` 与 `.json`，覆盖六个 comparison axis：
   - analog GKP information；
   - calibration-aware QEC；
   - logical-error and overhead targets；
   - logical-channel fidelity and infidelity；
   - learned QEC modules；
   - real-time FPGA decoders。
3. 每行记录 active citation key、文献报告的指标锚点、稿件用途和不可外推边界。
4. 将该 crosswalk 加入 `submission_draft_source_data_manifest.csv/json`，并纳入 source-data 机械审计。
5. 在投稿稿外部比较表 caption 和 supporting-material availability 表中登记该 crosswalk 的作用域。

## 可写边界

当前稿件可以写：

- 外部比较表有机器可读的 literature-metric crosswalk 支撑；
- crosswalk 记录的是相邻工作报告的 metric anchors 与本文不可直接比较的边界；
- 这些文献指标用于定位本文的层级和缺口。

当前稿件不能写：

- 这些文献值是本文实验 baseline；
- 本文完成了跨 code family 的 normalized leaderboard；
- 本文已达到 surface-GKP overhead、finite-energy logical-channel fidelity、real-time FPGA latency/resource 或 closed-loop hardware evidence 的标准；
- literature crosswalk 能替代 benchmark expansion、硬件测量包或统计推断包。

## 验证口径

已执行：

- `python docs\paper_materials\build_submission_draft_literature_metric_crosswalk.py`

后续应与 source-data manifest、source-data audit、symbol-boundary audit、TeX forbidden-word scan 和 LaTeX compile 一起验证。
