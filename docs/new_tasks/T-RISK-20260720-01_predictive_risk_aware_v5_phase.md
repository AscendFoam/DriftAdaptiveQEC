# T-RISK-20260720-01：插入 posterior-predictive risk-aware GKP MAP V5 Phase

- **Task ID：** T-RISK-20260720-01
- **标题：** 在 Route-A V4 负结果后插入软件优先的 V5 性能恢复与预板闭环
- **日期：** 2026-07-20
- **状态：** Done
- **来源风险：** R-N130、R-N131、R-N132
- **执行位置：** Phase 6A 后、Phase 7 前

## 输入材料

- 用户要求：暂不等待 FPGA 真板，优先通过仿真实验实现 observed-only、posterior-predictive、risk-aware GKP MAP；在未见漂移上相对最强 deployable baseline 至少降低 10% LER，显著改善 calibration/telegraph worst-window，并保持 atomic A/B、LKG rollback、6-cycle、II=1、fail-closed；
- T6.7.1/T6.8.1：V4 Route-A 只优于锁定 EWMA，仍差于 Window/static，static-to-oracle gap closure 为负；
- T6.7.2：V4 tail 大多等于 EWMA，calibration worst 为 181/512、static 为 32/512，fallback/false-update 代价高；
- T6.7.3、T6.2.1—T6.2.2：已有 production RTL、fixed-point/CXXRTL、atomic bank 和百万周期资格验证；
- T6.8.2/T6.9.1—T6.9.3：外部 budget、P&R estimate、真板 blocker 与旧完整论文 NO-GO；
- 当前 `cnn_fpga/runtime/regime_aware_safe_policy.py` 仅允许 Window/EWMA candidate，属于 policy/integration layer，不是新 decoder。

## 执行前方案

1. 不修改、重开或重新解释任何 T6.5—T6.9 Done task；把旧 formal 固定为 prior-informed diagnosis；
2. 使用 Phase 6B 和 Milestone 6.10—6.15，保持 Phase 7/8 编号稳定；
3. 先做 causal headroom/action-value NO-GO，再冻结 V5 contract 和全新四分割，避免目标驱动重构；
4. 在 host slow loop 实现 multiscale wrapped features、IMM/BOCPD、activation prediction、posterior-predictive MAP 与 LER/CVaR risk gate；
5. typed experts 必须映射到真实 two-bank residency/event action，不能用 Python 瞬时切换冒充部署；
6. formal 主排名使用 production-format quantized action，对全部 eligible baselines 做 simultaneous comparison；
7. 算法通过后再做独立 integer golden、long CXXRTL、actual-module formal properties 和 multi-seed P&R；
8. 真板任务 T6.9.2 保持 Blocked，只阻塞 measured hardware，不阻塞 simulation/pre-board paper gate。

## 实际完成内容

- 在 `docs/new_task_board.md` 插入 Phase 6B，共 6 个 Milestone、22 个 Todo task：
  - M6.10：causal headroom、V5 claim/execution contract、全新预注册；
  - M6.11：multiscale observed-only features、continuous IMM、BOCPD/fault posterior、activation-horizon prediction；
  - M6.12：统一 expert library、uncertainty-marginalized MAP-LUT、LER/CVaR risk gate、typed two-bank policy；
  - M6.13：train/calibration、matched pilot、formal-entry hash lock；
  - M6.14：全新 untouched smooth/tail formal 与 algorithm promotion/falsification；
  - M6.15：independent fixed-point golden、million-cycle CXXRTL、actual-module formal properties、multi-seed P&R、simulation/pre-board final gate；
- 冻结主性能门：
  - 对全部 eligible deployable baselines 的最小相对 LER 降幅至少 10%；
  - 每个 paired absolute improvement 的 simultaneous 95% 下界大于 0；
  - static-to-oracle gap closure 的点估计和 95% 下界大于 0；
  - step/calibration 与 telegraph 的 per-trajectory worst-window endpoint 相对 V4 至少下降 50%，并相对最强 tail baseline non-inferior；
- 冻结非简化实现要求：严格 observed-only、全新 split、effect-blind power/扩样、真实 production LUT projection、实际 two-bank residency、production-format quantized formal action、全域/高精度数值对照、actual parameterized RTL formal proof、每 family 长序列与 mutation；
- 将当前推荐任务从局部 Blocked 的 T6.9.2 切换为 T6.10.1，T6.9.2 状态和 42 项 measured null contract 不变；
- 修改 Phase 7 为双证据门：T6.15.5=GO_SIM_PREBOARD 才能冻结算法/预板论文；measured latency/jitter/deadline/power 仍单独依赖 T6.9.2；
- 在 `docs/new_risks.md` 新增 R-N130—R-N132，并把 R-N127/R-N129 的 P&R/UART 边界扩展到 V5；
- 更新任务板治理测试，使当前推荐、Phase 6B task 集、主门、formal 隔离、真板局部阻塞和 Phase 7 双门可机器检查。

## 产物路径

- `docs/new_task_board.md`
- `docs/new_risks.md`
- `docs/t6_9_2_route_a_board_measurement_blocker.json`
- `docs/t6_9_3_route_a_final_evidence_gate.json`
- `tests/test_new_task_board_governance.py`
- `docs/new_tasks/T-RISK-20260720-01_predictive_risk_aware_v5_phase.md`
- `docs/tasks/T-RISK-20260720-01_predictive_risk_aware_v5_phase.md`

## 验证方式和结果

- 检查 T6.10.1—T6.15.5 是否连续、唯一，数量是否为 22；
- 检查所有新 task 初始状态为 Todo，T6.9.2 仍为 Blocked，当前推荐为 T6.10.1；
- 检查旧 T6.7/T6.8 formal 只作诊断、新 formal 四分割/hash lock、10% LER/tail/observed-only/quantized/CXXRTL/formal/P&R 门均写入；
- 检查 Markdown 表格列数、任务依赖 ID、Done task completion record 和 whitespace；
- 运行 `python -m pytest tests/test_new_task_board_governance.py -q`。

实际结果：

- `python -m pytest tests/test_new_task_board_governance.py -q`：`7 passed`；
- 任务板扩展使 T6.9.2/T6.9.3 的旧哈希绑定按预期 fail-closed；使用各自生成器重签当前任务板绑定后，T6.9.2 仍为 `BLOCKED...ALL_MEASURED_FIELDS_NULL`（11/11 gates、42 measured fields 全为 null），T6.9.3 仍为 `NO_GO...RESTRICTED_PREBOARD_DRAFT_ONLY`（17/17 gates），没有把预板结果升格为真板证据；
- 相邻治理、claim、final evidence、hardware Pareto、board blocker 与 protocol hierarchy 回归：`71 passed`；
- Phase 6B 主任务行：`22`，全部为 `Todo`，Markdown 列数错误 `0`；
- Phase 6B 共解析 50 个既有/新增 task 引用，缺失引用 `0`；
- R-N130—R-N132 三行齐全，Markdown 列数错误 `0`；
- 两份完成记录 SHA-256 一致；
- `git diff --check` 无 whitespace error（仅提示工作区现有 LF/CRLF 转换策略）。

## 非简化实现复核

本次不是只增加“实现 IMM、跑 benchmark”两行任务。每个核心 claim 均拆为 headroom NO-GO、contract、互斥数据、数值实现、部署映射、pilot lock、untouched formal、独立重算、定点保持、actual-RTL formal、长序列和最终撤销门。特别禁止四种 demo 路径：

- 使用已打开的 T6.7 formal 调参或作为 V5 formal；
- 用 posterior mean plug-in 冒充 posterior-predictive risk integration；
- 用 Python 多专家瞬时切换冒充 two-bank hardware residency；
- 用 directed CXXRTL/P&R estimate 冒充 atomic 全状态证明或真板测量。

## 风险复核

- **R-N130 / High / Immediate / Open：** 现有相对 Window 的诊断 headroom 仅约 11.8%，必须先证明 causal/action-space 余量；
- **R-N131 / Critical / Immediate / Open：** posterior-mixture/full-joint/typed policy 可能无法映射到现有 phase-LUT 与 two-bank；
- **R-N132 / Critical / Immediate / Open：** sampled CXXRTL 不等于全状态 atomic proof，且更快 commit 可能耗尽 uint16 version；
- **R-N127、R-N129：** 继续 Open；V5 P&R 仍是 estimate，UART 仍是 control/diagnostic plane。

## 是否需要继续插入 task

不需要。R-N130—R-N132 已由 T6.10—T6.15 正常顺序完整承接；若 T6.10.1 headroom 或 T6.13.2 pilot NO-GO，应按失败分支停止 V5 formal，而不是再插入为通过主张服务的旁路实验。

## 对任务板的同步

- 插入任务区新增 T-RISK-20260720-01 并标记 Done；
- Phase 6B 新增 T6.10.1—T6.15.5，共 22 个 Todo task；
- 当前推荐任务更新为 T6.10.1；
- T6.9.2 保持 Blocked；
- Phase 7 的 algorithm/pre-board 入口改为 T6.15.5=GO_SIM_PREBOARD，measured hardware 仍依赖 T6.9.2；
- 进度日志新增 User request -> In Progress -> Done。
