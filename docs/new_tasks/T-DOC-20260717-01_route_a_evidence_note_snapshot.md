# T-DOC-20260717-01 Route-A 证据同步 note

- 日期：2026-07-17
- 状态：Done
- 类型：用户请求的 bounded docs snapshot；不替代或推进当前 `T6.7.3 In Progress`

## 输入材料

- 旧架构任务板：`docs/04_task_board.md`
- 新架构任务板：`docs/new_task_board.md`
- 旧架构 note：`docs/paper_notes/CNN_FPGA_GKP_theory_note_draft.tex`
- 旧/新任务记录、风险登记及 T24、T4.4.4、T5.1.4、T5.4.2、T5.5.2--T5.5.4、T6.2.2、T6.5--T6.7.2 的人读/机器证据。
- PPT 图源：`docs/figures/ppt_summary_20260716/drift_adaptive_architecture_cn.pdf`、`effective_fidelity_lifetime_cn.pdf`、对应生成脚本和 T4.4.4 source data。

## 实际完成内容

1. 将旧 CNN-centric 双回路与 Route-A contract-centric 架构按“可复用内容、被证否主张、当前主线”重新整理。
2. 新建独立英文 LaTeX note，固定术语、证据截止点、执行合同、方法、formal protocol、结果、硬件证据、负结果、降级路径和 repository evidence map。
3. 建立 metric-level advantage matrix，明确区分：
   - 已支持：joint MAP 的模型级相关噪声优势；Route-A 相对 pilot-locked EWMA 的 smooth primary 与 periodic family 优势；六周期确定性 fast path；预板目标器件可行性；teacher/student 压缩与 bit-exact 组件证据。
   - 有条件：旧 T24 Hybrid Residual-B 相对 UKF 的 1.75%--2.89% software-HIL 优势、smooth action balance，以及 T6.7.2 相对 locked EWMA 的 abrupt/OOD 与 nominal non-inferiority。
   - 未建立/反证：相对 static/Window 的普适 LER 优势、tail LER 优势、CNN universal winner、真板测速/资源和外部 SOTA。
4. 将 PPT 的双回路架构矢量图放入 Method，将有效保真寿命矢量图放入 teacher/student Results；不直接复用 PPT 强叙事，而用英文图注明确四状态 student 是可选扩展、主 LER expert 是 Window/EWMA、寿命不是 LER/FPGA wall-clock、旧成熟度徽章是 2026-07-16 历史快照。
5. 保留旧 note 不变；新 note 以 `docs/new_task_board.md` 为当前状态源，快照同步到 T6.7.2，并显式标注 T6.7.3 正在执行。

## 产物路径

- `docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex`
- `docs/figures/ppt_summary_20260716/drift_adaptive_architecture_cn.pdf`
- `docs/figures/ppt_summary_20260716/effective_fidelity_lifetime_cn.pdf`
- `docs/paper_notes/README.md`
- 本记录：`docs/new_tasks/T-DOC-20260717-01_route_a_evidence_note_snapshot.md`

## 验证方式和结果

- TeX Live `latexmk -pdf -interaction=nonstopmode -halt-on-error`：通过，15 页。
- 编译日志扫描：无 overfull box、undefined reference/citation、fatal error。
- Poppler 逐页渲染与视觉检查：15/15 页检查，无裁切、重叠、表格越界或空白异常；两张中文矢量图和英文边界图注均清晰。
- `git diff --check`：本 task 文件通过。
- 关键数字逐项回查 T6.7.1、T6.7.2、T6.2.2、T5.5.2/T5.5.3、T4.4.4 和旧 T24 的机器/源数据；note 同时写入有利与不利 comparator 结果。

## 风险复核

- 复用现有 `R-N118`：T6.7.1 的优势只允许写为 locked-EWMA / periodic-qualified，不能升级为 static/Window 或普适优势。
- 复用 `R-N098`：旧 fallback aggregate OOD 收益不能覆盖 compound 与 nominal 反例。
- 复用 `R-N119`：T6.7.2 所有门通过主要来自 Route-A≈locked EWMA；static calibration 更强且 tail fallback/false-update 很高，不能写成 tail 性能优势。
- PPT 架构图的底部成熟度徽章是历史快照；图注已说明 student 后续具备 fixed-point RTL/pre-board 证据，但真板仍未完成，避免旧图升级当前 claim。
- 新 note 是 T6.7.2 截止快照；若 T6.7.3 或 T6.7.4 产生新结论，必须按 promotion/downgrade gate 更新，不能静默沿用。

## 是否需要插入新 task

不需要。现有 T6.7.3、T6.7.4、T6.8.1、T6.9 已覆盖集成 RTL、独立复核、强 comparator 和真板缺口。

## 任务板同步说明

已在 `docs/new_task_board.md` 变更记录登记首次 docs snapshot 及本次图/证据同步；当前推荐任务保持 `T6.7.3 In Progress`，未修改其实现、协议、阈值、seed/cell 或实验顺序。
