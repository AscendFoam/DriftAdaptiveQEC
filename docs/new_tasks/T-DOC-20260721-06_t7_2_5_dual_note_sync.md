# T-DOC-20260721-06：T7.2.5 双 note 同步

- Task ID：`T-DOC-20260721-06`
- 标题：完整 Supplementary 合同、PPT 图复用与旧 note 保守同步
- 日期：2026-07-21
- 状态：Done

## 输入材料

- T7.2.5 提交 `43e9707`：46-row 八态 Supplementary 合同、24/24 gates/mutations、66 项联合回归、双百万周期 RTL 资格分层和 Phase 6C source locator。
- `docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex` 与 `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`。
- `docs/figures/ppt_summary_20260716/` 的双回路架构图、teacher--student 有效保真寿命图，以及正式 Fig.1--4、Supplement S1--S5。

## 实际完成内容

1. 复核新 note 的 Supplementary 已按定义、冻结参数、完整 comparator/statistics、失败域、RTL/工具链/长序列和 Phase 6C locator 组织，且没有形成第二张跨任务排行榜。
2. 在旧 note 增加 T7.2.5 证据表项、五合同 formal-paper overlay、46-row 状态语义、双百万周期试验非混并和 206-cell non-ranking 说明。
3. 保留旧 CNN/affine/teacher--student/FPGA 主体、历史实验数字和既有 PPT 图片；没有为 T7.2.5 重复新增图片，因为 S1--S5、架构图、寿命图和 auxiliary atlas 已覆盖所需视觉证据。
4. 明确 T7.2.5 只增加可复现性与审计深度，不修复 Route-A 相对 static/Window 的不利 LER 排名，也不产生 V5、真实板测或 physical break-even 证据。
5. 独立干净构建发现 T7.2.5 完成记录将新 note 误记为 50 页、1,217,875 bytes；实际 PDF 为 51 页、1,005,134 bytes，末页含参考文献 [29]--[30]，并非空白页。已同步纠正 README、任务板、风险表和双份 T7.2.5 记录。

## 当前可声称优势

- V4 只在锁定 EWMA smooth 对照上有 `+2.14%`；static 与 Window 仍是更强主 benchmark baseline。
- multimode posterior-weighted CPD 的 task-local absolute LER gain 为 `0.064668 [0.064413, 0.064926]`，32/32 clusters 同向。
- matched two-GKP-CNOT analog-ML reduction 为 `31.160%--67.982%`。
- 四状态/95 参数 student 在 cutoff 12/16 的十周期有效保真寿命为固定参数的 `2.36x/1.58x`，至少保留 `98.15%` teacher gain，参数量约缩小 `767x`；它是扩展证据，不是当前 LER winner。
- six-cycle、II=1 与两类百万周期 CXXRTL 资格支持确定性 pre-board contract；不支持 measured board latency、jitter、power、fastest 或 SOTA。

## 产物路径

- `docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `docs/new_tasks/T-DOC-20260721-06_t7_2_5_dual_note_sync.md`
- `README.md`
- `docs/paper_notes/README.md`
- `docs/new_risks.md`
- `docs/new_task_board.md`
- `docs/new_tasks/T7.2.5_supplementary_evidence.md`
- `docs/tasks/T7.2.5_supplementary_evidence.md`

## 验证方式和结果

- 五份合同按依赖顺序重建后分别通过 18/18、18/18、20/20、22/22、24/24 gates/mutations；独立扩大联合回归 72/72 通过。
- 新 note：TeX Live/XeLaTeX/BibTeX 干净构建 51 页、1,005,134 bytes；日志 0 overfull、LaTeX/package warning、undefined reference/citation、float-too-large 或 fatal error。
- 旧 note：TeX Live/XeLaTeX 最终干净构建 33 页、675,079 bytes；同类日志扫描为 0。
- 两稿共 84/84 页渲染检查通过；重点放大检查旧稿 overlay 第 11 页、新稿 Supplementary 第 40/44/46/48 页和参考文献第 51 页，无裁切、重叠、乱码或伪空白页。
- `git diff --check` 作为提交前格式检查。

## 风险复核

- R-N110、R-N142、R-N145 继续由 T7.3--T7.4、T6.9.2 与 Phase 8 承接。
- N/A、null、failed、negative、blocked、ineligible 仍是不同科学状态，不得填零或合并评分。
- 不插入新 task；新增求正、V5 rescue、跨任务总榜或伪板测会违反现有合同。

## 对任务板的同步

- 进度日志登记 `T-DOC-20260721-06 User request -> Done`。
- 不改变 `T7.3.1 In Progress` 的当前推荐状态与顺序。
