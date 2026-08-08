# T-DOC-20260720-02 双 note 证据与图表同步

- Task ID：`T-DOC-20260720-02`
- 标题：将 T6.19.3 截止证据同步为正式论文式新 note，并保守同步旧 CNN+FPGA note
- 日期：2026-07-20
- 状态：Done

## 输入材料

- `docs/new_task_board.md`：低频轮询 T6.19.3 从 In Progress 到 Done，并冻结截止点为 `PASS_AUX_COMPARISON_INTEGRITY`；当前推荐任务保持 `T7.1.1 In Progress`。
- `docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex` 与 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`。
- V4/V5/Phase 6C 的人读报告、JSON、Source Data、P&R/CXXRTL 产物与 T6.19.3 六-lane atlas。
- `docs/figures/ppt_summary_20260716/` 中的双回路架构图与 teacher--student 有效保真寿命图。

## 实际完成内容

1. 大幅重写新 note 为正式论文式证据门结构：相关工作、合同方法、预注册统计、V4 受限结果、V5 因果 headroom 早停、Phase 6C 独立 lane、复现性、终态门、讨论与结论。
2. 把论文主张改为结果驱动：明确 V4 只在 locked-EWMA/periodic 口径存在窄优势，static/Window 与 calibration tail 构成反例；V5 因 `0.02549% < 12%` action-space headroom 提前停止。
3. 新增三张 Python 主文图：V4 evidence boundary、V5 early stop、auxiliary matched evidence；每张均从当前 JSON/CSV 读取并输出 SVG/PDF/PNG/600-dpi LZW-TIFF。
4. 新增 58-row plotted Source Data 与 SHA-256 manifest，绑定 7 个输入源和 13 个输出；图中显式保留 CI、seed cluster、N/A/null、estimate/measured 与 no-global-ranking 边界。
5. 复用两张 PPT 图：架构图改作接口图，student 降为可替换扩展；寿命图限定为十周期 finite-model 指标，不与 LER、板上时间或官方 GQF 混写。
6. 将 T6.19.3 206-cell、24/24 gates/mutations、162 联合回归的 atlas 作为补充比较图纳入新 note，并明确它不升级 V5 NO-GO 或 T6.9.2 Blocked。
7. 对旧 note 做中等同步：保留 teacher-anchored affine/CNN 主体和旧数表，更新标题、摘要、当前证据状态、metric table、Results 入口、Discussion、Conclusion，并嵌入 PPT/主文图。

## 产物路径

- `docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/figures/make_route_a_manuscript_figures.py`
- `docs/figures/route_a_manuscript_20260720/`
- `docs/paper_notes/README.md`
- `README.md`
- 本记录文件

## 验证方式和结果

- 图生成脚本：`python -m py_compile` 通过；真实 schema 的 fail-closed 检查在首次错误字段时拒绝运行，修正为当前字段后生成 58 条 plotted rows。
- 图视觉 QA：三张新主文图逐张检查；修复 tail 图例遮挡、pre-board 注释重叠与 TIFF 未压缩问题；最终 SVG 保留 editable text，TIFF 使用 LZW。
- 新 note：TeX Live `latexmk` 编译成功，22 页；日志无 overfull、未定义引用、重复标签、Float too large 或 LaTeX warning；22/22 页 Poppler 渲染与 contact-sheet 视觉 QA 通过。
- 旧 note：TeX Live `latexmk` 编译成功，28 页；同类日志扫描为 clean；28/28 页视觉 QA 通过。
- T6.19.3 截止复核：任务板为 Done，verdict=`PASS_AUX_COMPARISON_INTEGRITY`；T6.15.5 仍 `NO_GO_V5_EARLY_HEADROOM_STOP`，T6.9.2 仍 Blocked。

## 风险复核

- 复用 R-N118/R-N119/R-N128/R-N139--R-N141：static/Window 反例、tail 高干预、V5 NO-GO、multimode model-family alignment、pre-board estimate 和 external same-task=0 均在摘要、图注、表格与结论显式保留。
- 未新增阻塞当前文档同步的新风险。T7.1.1 正在建立正式 claim--evidence--boundary matrix，后续 note 轮次需继续低频同步，但不构成本轮插入任务理由。

## 是否需要插入新 task

不插入。当前 `T7.1.1`、T7.3/T7.4 与 Blocked 的 T6.9.2 已覆盖 claim 冻结、审稿边界、可复现发布和真板缺口；为获得正结论而插入 post-hoc rescue task 会破坏预注册边界。

## 对任务板的同步说明

在 `docs/new_task_board.md` 进度日志新增 `T-DOC-20260720-02` Done 记录；不改变当前推荐任务 `T7.1.1` 的状态或顺序。
