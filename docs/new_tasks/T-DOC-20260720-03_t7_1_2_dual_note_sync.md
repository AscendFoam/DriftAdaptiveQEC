# T-DOC-20260720-03：T7.1.1--T7.1.2 双 note 与正式主图同步

- Task ID：`T-DOC-20260720-03`
- 标题：将 claim matrix 与 evidence-bounded Fig.1--2 同步到新旧 note
- 日期：2026-07-20
- 状态：Done

## 输入材料

- `docs/t7_1_1_claim_evidence_boundary_matrix.json`、29-row Source Data 与人读矩阵。
- `docs/t7_1_2_main_figure_contract.json`、38-row Source Data、图合同与多格式图包。
- T6.7--T6.19.3 的 V4、V5 early-stop、Phase 6C、CXXRTL/P\&R estimate 和 board-null 父证据。
- 既有两份 note、PPT 双回路图和 teacher--student 有效保真寿命图。

## 实际完成内容

1. 将新 note 标题收紧为 evidence-gated、fail-closed 的 restricted pre-board contract，证据截止点推进至 T7.1.2。
2. 按 T7.1.1 placement 重写新 note 摘要：只保留合同系统集成、相对 locked EWMA 的 smooth paired 结果和六周期预板架构三条允许主张；将 multimode CPD、CNOT ML、V5 早停、board/GQF 等结果放回其允许的 Results、Conclusion 或 Supplement 边界。
3. 新增 canonical terminology ledger 同步、29-claim placement 汇总表、T7.1.1/T7.1.2 terminal gate 与 repository evidence map。
4. 将 T7.1.2 的正式 Fig.1--2 嵌入新 note：Fig.1 展示 host/FPGA 双回路、evidence ladder 和 timing ownership；Fig.2 展示 typed action、atomic A/B/LKG 事务及 Dropped V5/Blocked board 边界。调整浮动顺序后两图连续编号，原 PPT 架构图作为后续历史图保留。
5. 对旧 note 仅做中等同步：保留 teacher-anchored affine/CNN 主体和历史实验，新增 T7.1.1 claim-placement 覆盖层、当前 Fig.1、T7.1.2 状态与结论约束；未将旧正文重写为新架构论文。
6. 修复旧 note 目录/引用的红色 PDF 边框，统一为可读彩色链接，不改变结果内容。
7. 同步根 README 与 `docs/paper_notes/README.md` 的文档入口、证据截点、页数和 QA 状态。

## 产物路径

- `docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/paper_notes/README.md`
- `README.md`
- 本记录文件。

正式图、Source Data 和生成器由 T7.1.2 冻结在：

- `docs/figures/t7_1_2_main_figures/`
- `docs/figures/make_t7_1_2_main_figures.py`
- `docs/t7_1_2_main_figure_source_data.csv`
- `docs/t7_1_2_main_figure_contract.json`

## 验证方式和结果

- 使用 LaTeX compile 工作流和 TeX Live/latexmk 编译两份 TeX：新 note 26 页，旧 note 29 页，均成功退出。
- 两份最终日志均无 overfull、LaTeX warning、未定义引用、重复标签、`Float too large` 或 fatal error。
- 使用 Poppler 将 26+29=55 页全部渲染为 PNG，并检查整份 contact sheet；未发现裁切、遮挡、黑块、断表、错误浮动或不可读页面。
- 对正式 Fig.1/2 页面做原尺寸复核：连续编号正确，字体与图注可读；V5/board 边界无生产入边，P\&R estimate 与 board measurement 明确分层。
- 旧 note 超链接红框已消失，正文与历史图表分页未产生回归。

## 风险复核

- R-N142、R-N143 继续为 `Mitigated / High / Monitor`：本轮直接消费 29-claim matrix 和 38-element figure contract，没有复活 V5、隐藏 negative、把 estimate 写成 measurement 或给 blocked 模块绘制生产箭头。
- T6.9.2 仍为 Blocked，42 个 board-measured 字段继续为 null。
- T7.1.3 正在冻结主图 3--4；本轮不提前引用其尚未 terminal 的输出，后续单独同步。

## 是否需要插入新 task

不插入。T7.1.3--T7.1.4 已承接后续主图和 Supplement，T7.2/T7.4 已承接正文与最终 claim/provenance 审计；新增 post-hoc performance rescue 会违反 V5 early-stop 和只读证据边界。

## 对任务板的同步

- 在 `docs/new_task_board.md` 进展记录新增 `T-DOC-20260720-03 User request -> Done`。
- 当前推荐任务保持 `T7.1.3 In Progress`，不改变另一 Codex 对话的执行指针。
- `docs/new_risks.md` 增加本轮“不插入”复核记录。
