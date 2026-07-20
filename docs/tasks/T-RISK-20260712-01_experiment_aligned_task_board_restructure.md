# T-RISK-20260712-01 实验对齐的任务板重构

- **Task ID**: T-RISK-20260712-01
- **标题**: 按 GKP 实验协议、低成本 FPGA 边界和论文证据链重构任务板
- **日期**: 2026-07-12
- **状态**: Done

## 输入材料

- `docs/task_board.md`：v1 当前任务状态源；当前推荐任务为 `T1.3.3`。
- `docs/experiment_plan.md`：原始 phase/milestone/task 计划及论文主张。
- `docs/rough_plan.md`：冻结粗规划，仅作为历史背景读取，未修改。
- `docs/risks.md`：现有科学、工程、写作和治理风险。
- `docs/relative_papers/Quantum_error_correction_of_a_qubit_encoded_in_grid_states_of_an_oscillator.md`：2020 年 GKP FPGA/measurement-feedback 实验参考。
- `docs/relative_papers/Advances_in_Bosonic_Quantum_Error_Correction_with_Gottesman–Kitaev–Preskill_Codes_Theory_Engineering_and_Applications.md`：重点核查 Noise Models 和 Quantum Engineering with GKP codes。
- `docs/relative_papers/Real-time_quantum_error_correction_beyond_break-even.md`：重点核查主文、Methods、sBs/Kraus、周期时序、RL workflow、syndrome/leakage、长期稳定性、平均通道保真度和全部图片。

## 触发原因和证据

本任务由 R-014—R-018 触发。文献核查对原计划产生了以下实质影响：

1. 实际 FPGA fast path 与 PPO/学习优化属于不同时间尺度。逐周期路径负责 measurement processing、branching、frame/phase update；PPO 的训练和 FPGA 重新编译位于秒级慢路径。
2. sBs 的严格 syndrome 语义是 `gg/ge/eg/ee` 成对 Kraus outcome 和 error-space transition，不能只用独立连续 residual 或单个 g/e 代表完整物理状态。
3. 长时非平稳性的重要来源是 leakage streak 和离散 regime switching；去除持续多周期 leakage 后，长相关尾显著消失，因此只建模平滑 Gaussian drift 不足。
4. 约 300 元纯数字 FPGA 可提供真实的经典控制平面、定点、时序和 HIL 证据，但不能替代微波 ADC/DAC、cavity/transmon 或真实 beyond-break-even 实验。
5. 原 v1 把论文冻结放在真实 FPGA 之前，会导致主图和硬件结论在板级证据产生前过早锁定。

## 实际完成内容

### 1. 保留已有证据与当前指针

- 保留 `T0.1.1`—`T1.3.2` 的状态、产物和验证。
- 当前推荐任务保持 `T1.3.3`，不因治理重构跳过当前理论任务。
- 原 v1 Phase 2 以后均未开始，因此只重排 Todo，不删除完成记录。

### 2. 新增 claim/protocol/board gate

- 新增 M1.4：claim ladder、实际板卡边界、两个计算域/三个时间尺度接口、paper-parameter registry。
- 新增板级 provisional claim contract，区分 software simulation、synthesis estimate、board measurement、HIL 和 quantum experiment。

### 3. 重构仿真和算法阶段

- Phase 2 改为 sBs-first、sharpen–trim-secondary 的实验式多保真数字孪生。
- 新增 sBs Kraus/error hierarchy、g/e/leakage observation、reset、cycle state machine、故障注入趋势和 syndrome occupancy/correlation 交叉验证。
- 将 drift 扩展为连续参数 + 离散 regime/recovery/leakage 混合状态。
- Phase 3 增加 run-length FSM、parameter-bank、HMM/change-point baseline；post-selection 降为诊断上界并要求核算 rejection cost。
- Phase 4 改为两个计算域、三个时间尺度控制器；offline teacher、host estimator 和 FPGA fast path 分离。

### 4. 将结果阶段改为 evidence gates

- 协议/数字孪生可信度；
- 算法/oracle-gap；
- 因果故障注入/syndrome diagnosis；
- logical channel/simulation-derived coherence gain/cost；
- OOD/消融/负结果；
- hardware design freeze。

### 5. 交换 FPGA 与论文顺序

- Phase 6 改为实际低成本 FPGA bring-up、RTL fast path、host-to-FPGA HIL、长序列和失败实验。
- Phase 7 改为 claim-evidence 主图冻结、论文、审稿风险和可复现发布。
- Phase 8 保留可选真实 GKP 数据和量子控制链路接入，不阻塞第一篇论文。

### 6. 低频更新实验计划

在 `docs/experiment_plan.md` 追加第 14 节，记录触发材料、修订后的边界、阶段顺序、证据门和禁止主张。原计划正文保留为历史规划，`docs/task_board.md` 继续作为唯一当前执行顺序和状态源。

## 产物路径

- `docs/task_board.md`
- `docs/experiment_plan.md`
- `docs/risks.md`
- `docs/tasks/T-RISK-20260712-01_experiment_aligned_task_board_restructure.md`

## 验证方式和结果

已执行以下文档验证：

```powershell
rg -n "当前推荐任务|T1.3.3|Phase 2：实验式|Phase 6：低成本|Phase 7：论文|Phase 8：可选" docs/task_board.md
rg -n "^# 14\.|^## 14\." docs/experiment_plan.md
rg -n "R-014|R-015|R-016|R-017|R-018|T-RISK-20260712-01" docs/risks.md docs/task_board.md
git diff --check
```

通过标准：

- `T1.3.3` 仍为当前推荐任务；
- 已完成任务状态数量和 ID 不因重构减少；
- Phase 6 位于 Phase 7 之前，真实量子接入只出现在 Phase 8；
- 新任务均有单一问题、产物/通过标准和来源；
- `docs/rough_plan.md` 无 diff；
- Markdown 无尾随空格或 patch 结构错误。

验证结果：

- 共识别 113 个正式 task ID，无重复 ID；
- 当前推荐任务唯一且仍为 `T1.3.3`；
- 原有 13 个 `T0.*`—`T1.3.2` Done task 全部保留；
- Phase 6、7、8 的起始行依次为低成本 FPGA、论文、可选真实硬件，顺序正确；
- `R-014`—`R-018` 和 `T-RISK-20260712-01` 已同时出现在风险表与任务板；
- `git diff --check` 通过；
- `git diff --name-only -- docs/rough_plan.md` 无输出，确认冻结文件未修改；
- 本任务仅修改文档，没有运行代码测试套件。

## 风险复核

- 新增 `R-014`：实验协议与 syndrome 语义错配。
- 新增 `R-015`：把 leakage/regime 非平稳性误写成连续漂移。
- 新增 `R-016`：低成本数字开发板被误写成真实量子控制硬件。
- 新增 `R-017`：post-selection 增益忽略 rejection cost。
- 新增 `R-018`：同一文献参数用于校准和验证造成循环证据。
- 调整 `R-005`、`R-007`、`R-012`、`R-013` 的关联 task 和缓解措施以匹配 v2。

## 是否需要插入新 task

本治理任务本身已作为 `T-RISK-20260712-01` 插入并完成。R-014—R-018 的实际实验工作已由 M1.4、M2.0、Phase 5—7 覆盖，当前不再插入额外风险 task。

## 对 task_board 的同步说明

- `T-RISK-20260712-01` 状态为 `Done`。
- 当前推荐任务保持 `T1.3.3`。
- 进度日志登记本次 `Proposed -> In Progress -> Done`。

## 对实验计划的影响

文献核查实质改变了 protocol reference、phase 顺序、核心 hardware claim 和 evidence gate，满足 `AGENTS.md` 的低频更新条件。因此在 `docs/experiment_plan.md` 追加第 14 节；没有改写历史正文，也没有修改冻结的 `docs/rough_plan.md`。
