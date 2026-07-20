# T-RISK-20260713-02：六篇补充论文驱动的 v2.2 任务板补强

- Task ID：`T-RISK-20260713-02`
- 标题：根据六篇补充论文小幅补强任务板
- 日期：2026-07-13
- 状态：`Done`

## 输入材料

- arXiv:2605.08009，`Error Correction of Beamsplitter-Generated Entangled GKP States`
- arXiv:2604.08247，`Optimized Gottesman-Kitaev-Preskill Error Correction via Tunable Preprocessing`
- arXiv:2411.05262，`Noise Transfer Approach to GKP Quantum Circuits`
- arXiv:2505.14775，`Performance analysis of GKP error correction`
- arXiv:2401.02022，`The Near-optimal Performance of Quantum Error Correction Codes`
- arXiv:2510.06531，`Approximate maximum-likelihood decoding with K minimum weight matchings`
- `docs/task_board.md` v2.1
- `docs/experiment_plan.md` 第 14—15 节
- `docs/risks.md`

## 文献核查范围

按用户要求执行轻量核查：阅读六篇论文的标题、摘要、实验或数值结果、结论，并检查关键结果图。只有 2605.08009 是真实 trapped-ion GKP QEC 实验；其余主要为理论、解析或数值工作。因此，本任务不把六篇工作统一称为“实验论文”。

## 补强原则

1. 不重排 v2.1 阶段；
2. 只新增能直接闭合当前证据缺口的任务；
3. single-mode approximate GKP 主范围不变；
4. 约 300 元 FPGA 仍只承担数字控制平面、定点算法和 HIL；
5. secondary protocol 不进入 sBs 主排名；
6. 每个新上界、代理和 baseline 必须有明确证否边界。

## 实际完成内容

### 1. 三项主线补强

- 增加 Heisenberg noise-transfer 中保真度代理及其高/低 squeezing 有效域验证。
- 增加 QEC-matrix/Petz channel-recovery bound，并与 decoder oracle、control oracle 分开。
- 增加 single-mode top-K lattice-coset truncated MAP baseline 和 K—精度—资源 Pareto。

### 2. Secondary evidence

- Knill/qunaught、Steane/ME-Steane、P-Steane 仅进入协议趋势复现与 Supplement。
- 2605.08009 的双模 trapped-ion 实验只用于补强逐 Pauli、QEC on/off、wall-clock、reset/recoil 和并行开销叙事。
- surface-GKP K-MWM 不直接实现；只借鉴 top-K likelihood accumulation。

### 3. 风险闭环

- `R-024`：noise-transfer 代理超出有效域；
- `R-025`：Petz/channel-recovery bound 与实际 decoder/controller 混称；
- `R-026`：surface-GKP、多模或量子 preprocessing 引发范围和 FPGA claim 扩散。

## 产物路径

- `docs/task_board.md`
- `docs/experiment_plan.md`
- `docs/risks.md`
- `docs/tasks/T-RISK-20260713-02_six_paper_v22_board_strengthening.md`

## 验证方式和结果

- 任务 ID 结构检查：通过；正式任务表共 144 个 ID，全部唯一。
- 新任务检查：通过；`T2.3.8`、`T3.1.5`、`T5.3.5` 和治理任务均已登记。
- 状态检查：通过；既有 13 个主线 `Done` 状态未改变。
- 当前指针检查：通过；仍为 `T1.3.3`。
- 风险路由检查：通过；`R-024`—`R-026` 均关联到明确任务和证据门。
- 实验计划检查：通过；第 16 节存在且六篇来源可追溯。
- 冻结文件检查：通过；`docs/rough_plan.md` 未修改。
- 文档差异检查：通过；`git diff --check` 返回 0。
- 本任务只修改治理与实验设计文档，未执行代码测试。

## 是否需要插入新 task

当前无需额外插入。R-024—R-026 已由 v2.2 正常 task 和增强后的 evidence gate 覆盖；只有执行时发现现有任务无法形成可信结论，才新增后续 `T-RISK-*`。

## 对任务板的同步说明

`T-RISK-20260713-02` 已登记为 `Done`，并追加 `Proposed -> In Progress -> Done` 进度日志；当前推荐任务保持 `T1.3.3`，未提前启动任何新增实验任务。
