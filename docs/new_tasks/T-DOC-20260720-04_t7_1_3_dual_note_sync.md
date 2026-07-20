# T-DOC-20260720-04：T7.1.3 主图 3--4 双 note 同步

- Task ID：`T-DOC-20260720-04`
- 标题：将 T7.1.3 主结果/预板证据图同步到新旧 note
- 日期：2026-07-20
- 状态：Done

## 输入材料

- `docs/new_task_board.md` 中已完成的 T7.1.3 终态与当前 T7.1.4 入口。
- T7.1.3 稳定提交 `e96ec9d`。
- `docs/t7_1_3_main_result_figure_contract.json`、55-row Source Data、图包 manifest 与双轨完成记录。
- `docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex`。
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`。
- 已复用的 PPT 架构图和 teacher--student 有效保真寿命图。

## 实际完成内容

1. 新 Route-A note 的证据截止点更新到 T7.1.3；在 cutoff、terminal gate、reproducibility、conclusion 和 repository evidence map 中加入 55-row、16/16 gates/mutations、153 项父链回归与人工视觉 QA 边界。
2. 用 T7.1.3 正式 Figure 3 替换早期自制 V4 汇总图。图注同时保留 locked-EWMA 窄正结果、Window strongest、static/oracle-gap 负结果、六类 abrupt/OOD non-inferiority 和高 fallback/recovery cost，未把 Phase 6C 结果提升到主排名。
3. 新增正式 Figure 4，分离百万周期 CXXRTL 零 mismatch/undefined/overflow、6-cycle/II=1 clock model、两 profile×三 seed P&R estimate 与 42 个 board-null/V5-Dropped 字段。
4. 为保持 T7.1.2--T7.1.3 正式主图编号连续，将复用的 PPT 双回路架构图改为不占 figure counter 的历史示意；图仍留在 Method 中，teacher--student PPT 图仍留在扩展证据段。
5. 旧 CNN+FPGA note 保留 teacher-anchored affine/CNN 主体与 PPT 架构图，只将早期 V4 汇总图替换为正式 Figure 3、增加正式 Figure 4，并更新当前证据状态与结论覆盖层。
6. 两份 note 均继续禁止把 P&R/222.222 ns 写成板测，把 tail non-inferiority 写成 tail advantage，或把历史 CNN/teacher/student 与 Phase 6C task-local 结果用于升级 V4/V5 主结论。

## 产物路径

- `docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex`
- `docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- `output/pdf/contract_note_t7_1_3/Contract_Centric_Regime_Aware_GKP_note_draft.pdf`
- `output/pdf/old_note_t7_1_3/CNN_FPGA_GKP_theory_note_draft.pdf`
- 本记录文件。

## 验证方式和结果

- T7.1.3 live contract：`PASS_MAIN_FIGURES_3_4_RESTRICTED_PREBOARD_RESULTS`，55 records，16/16 gates，16/16 substantive mutations。
- T7.1.3 专项复核：`8 passed in 7.50s`；父任务稳定记录为 153 项父链联合回归通过。
- 两张正式图：Python/matplotlib only；SVG 保留 160 个 text node；600-dpi TIFF 最短边 3236/3000 px；人工视觉 QA=`PASS`。
- TeX Live 中间与终态编译均成功：新 note 27 页，旧 note 30 页。
- 编译日志未发现 `Overfull`、`LaTeX Warning`、未定义引用、重复定义、过大 float、emergency stop 或 fatal error。
- Poppler 渲染 57/57 页并检查完整 contact sheet；另以原始分辨率检查正式 Figure 3--4 页面。未发现裁切、重叠、空白页、断裂图注或不可读文字。
- 主图编号核对：新 note 的证据冻结主图连续为 Figure 1--4；复用 PPT 架构图为不编号历史示意。旧 note 中 PPT 架构仍为 Figure 2，T7.1.3 结果/预板图为 Figure 3--4。

## 风险复核

- R-N142 保持 `Mitigated / High / Monitor`：正文仍直接消费 29-claim placement contract，摘要未加入 Phase 6C、V5 或板测主张。
- R-N143 保持 `Mitigated / High / Monitor`：fast/slow/learning/physical 层与 board-null 继续由正式图合同约束。
- R-N144 保持 `Mitigated / Critical / Monitor`：正式 Figure 3 同时呈现正、负和成本结果；Figure 4 保留 clock-model/P&R-estimate/board-null/V5-Dropped 分层。
- T6.9.2 仍为 Blocked，42 个 measured 字段保持 null；T6.15.5 仍为 `NO_GO_V5_EARLY_HEADROOM_STOP`。

## 是否需要插入新 task

不插入。T7.1.4 已承接 Supplement figure contract，T7.2--T7.4 承接正文、补充材料和最终 claim audit；T6.9.2 承接真实板测升级条件。新增求正、伪板测或 V5 复活任务会违反已冻结边界。

## 对 `docs/new_task_board.md` 的同步说明

- 不改变 T7.1.3=`Done`、T7.1.4=`In Progress` 或当前推荐任务。
- 在进度日志登记本轮 T7.1.3 双 note、PPT 复用、57 页渲染 QA 与本地提交。
