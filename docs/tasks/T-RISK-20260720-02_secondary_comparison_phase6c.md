# T-RISK-20260720-02：插入异构方法非主要比较 Phase 6C

- **Task ID：** T-RISK-20260720-02
- **标题：** 在 Phase 6B 后插入 CI/ML、NN/RL、AQEC、CPD 与 FPGA 的分赛道补充比较
- **日期：** 2026-07-20
- **状态：** Done
- **来源风险：** R-N133、R-N134、R-N135、R-N136
- **执行位置：** Phase 6B 完成后、Phase 7 前

## 输入材料

- 用户提供的两张方法比较图，包含 CI、ML、神经网络/RL、AQEC、CPD、Hybrid CNN–FPGA，以及逻辑错误、threshold、寿命、漂移适应、latency 和硬件开销等列；
- 用户要求：这些比较可以加入任务板，但只作非主要排名，并且必须在原 Phase 6B 完成后执行；
- 既有 T6.8.3—T6.8.7：Puviani GQF exact 复现资格、外部 drift lane、FPGA QEC 规范化和 claim matrix；
- 既有 T3.2.8：AQEC 与 measurement-feedback 在共同 wall-clock 下出现 per-cycle/per-us 排序反转；
- Phase 6B 的 V5 formal、10% LER、tail、fixed-point/CXXRTL/formal/P&R 与最终 GO/NO-GO 合同；
- 前序文献核查：两 GKP-qubit gate 的 CI/analog-ML failure、full surface–GKP threshold、structured-lattice CPD 和外部 FPGA latency 的任务/分母边界。

## 执行前方案

1. 不直接把截图改写成排行榜，先按 decision object、protocol、code family、metric denominator 和 timing boundary 拆 lane；
2. Phase 6C 只在 `T6.15.5=Done` 后启动，不要求其 verdict 为 GO，因此 V5 失败也不会阻塞诚实的补充比较；
3. 保存 Phase 6B live hashes，禁止 Phase 6C 重选 comparator、调阈值、改主图权重或进入 V5 的 10% LER denominator；
4. 对可执行且有直接意义的对象安排真实复现：single-mode CPD/CI 等价、两-GKP gate CI/ML、已有 learned model replay、AQEC wall-clock、official CPD、预板 RTL/P&R；
5. 对需要完整外部物理系统或来源缺失的对象设计 partial/null/NOT_RUN 分支，禁止用简化 simulator 或自写近似网络冒充 exact reproduction；
6. 最终只生成分面 comparison atlas；完整性门允许负结果、不可比和 null，不以“必须取得优势”为通过条件。

## 实际完成内容

- 在 `docs/new_task_board.md` 的 Phase 6B 与 Phase 7 之间插入 Phase 6C，共 4 个 Milestone、12 个 Todo task：
  - M6.16：一手来源审计、metric/timing/resource ontology、只读 secondary preregistration；
  - M6.17：single-mode CPD/CI 等价边界、两-GKP gate CI/ML 复现、Direct NN/adaptive NN/RL eligibility；
  - M6.18：AQEC common-wallclock replay、official structured-lattice CPD threshold reproduction、条件式 multimode drift extension；
  - M6.19：项目内同任务预板 profile、外部 FPGA 规范化、六 lane comparison atlas 与完整性门；
- 冻结六条 evidence lane：single-mode decoder、surface–GKP gate/outer-code、multimode CPD、controller/RL/NMF、AQEC protocol、FPGA implementation；
- 冻结证据等级：`LITERATURE_ONLY`、`OFFICIAL_CODE_REPRODUCTION`、`PROJECT_NATIVE_MATCHED`、`INELIGIBLE`、`BLOCKED`、`NEGATIVE`；
- 明确 `N/A/null/failed` 语义，禁止 global score、跨分母换算、跨 code family raw latency 排名和 literature value 冒充项目 run；
- 对截图中 `9.9 dB`、约50%、`<50 ns`、零延迟、约20%、`10--100 us` 等定量或定性结论设置一手 locator 门；无直接证据时必须为 null；
- 将 Phase 7 改为双门：V5 主张仍由 `T6.15.5=GO_SIM_PREBOARD` 控制，补充比较由 `T6.19.3=PASS_AUX_COMPARISON_INTEGRITY` 控制；后者不能挽救前者；
- 在 `docs/new_risks.md` 新增 R-N133—R-N136，并记录插入判断；
- 更新任务板治理测试，锁定 Phase 6B/6C 切分、12 个 task、只读/非混排/失败分支、真板边界和 Phase 7 双门。

## 产物路径

- `docs/new_task_board.md`
- `docs/new_risks.md`
- `tests/test_new_task_board_governance.py`
- `docs/new_tasks/T-RISK-20260720-02_secondary_comparison_phase6c.md`
- `docs/tasks/T-RISK-20260720-02_secondary_comparison_phase6c.md`

## 验证方式和结果

- 检查 T6.16.1—T6.19.3 是否连续、唯一、数量为 12 且初始状态均为 Todo；
- 检查当前推荐仍为 T6.10.1，T6.9.2 仍为 Blocked；
- 检查 Phase 6C 是否明确位于 T6.15.5 后，且禁止回写 Phase 6B formal；
- 检查六 lane、证据等级、N/A/null/failed、复现失败分支、hardware evidence boundary 和 final integrity gate；
- 检查 Markdown 表格、task 引用、Done completion record、派生 evidence hash 和 whitespace；
- 运行任务板治理及相邻 Route-A evidence tests。

实际结果：

- `python -m pytest tests/test_new_task_board_governance.py -q`：`8 passed`；
- 治理、claim matrix、final evidence gate、hardware Pareto、board blocker 与 protocol hierarchy 相邻回归：`72 passed`；
- Phase 6C 主任务行 `12`、唯一 task ID `12`、全部为 Todo，5 列 Markdown 表格错误 `0`；
- R-N133—R-N136 共 `4` 行，8 列风险表结构错误 `0`；
- 当前推荐仍为 T6.10.1，T6.9.2 仍为 Blocked；
- 重新生成 board-measurement blocker：`BLOCKED_T6_9_2_NO_PHYSICAL_BOARD_EVIDENCE_ALL_MEASURED_FIELDS_NULL`，`11/11` gates，42 个 measured 字段仍为 null；
- 重新生成 final evidence gate：`NO_GO_FULL_HIGH_LEVEL_PAPER_RESTRICTED_PREBOARD_DRAFT_ONLY`，`17/17` gates；任务板扩展没有升级 V4/真板主张；
- 两份完成记录内容一致；`git diff --check` 无 whitespace error（仅有仓库既有 LF/CRLF 策略提示）。

## 非简化实现复核

本次没有把五类方法压成“各跑一个 demo”。每类实验都带有 task signature、正式输入输出、共同预算、统计停止规则、来源 commit、exact/reference 对照和失败分支：

- single-mode CPD 必须解析证明并在完整 10-bit 域或百万样本上与 CI 0 mismatch；
- gate-level CI/ML 必须用 common random numbers、raw failure count、paired CI 和 brute-force likelihood audit；
- learned model 必须过 observed-only、checkpoint、warm-up/cadence、precision 与 budget 门；
- AQEC 必须按共同物理 wall-clock 重放，不能用 cycle 数直接声称寿命优势；
- official CPD 必须通过 upstream/exact-CVP/finite-size checks，入口失败时保持 partial/NOT_RUN；
- FPGA 只比较真实 RTL，slow-loop 与 6-cycle fast path分开，P&R 不得写成 measured。

## 风险复核

- **R-N133 / Critical / Immediate / Open：** 跨协议 global leaderboard 会制造伪 SOTA；
- **R-N134 / High / Immediate / Open：** 截图中的无任务/分母数字可能被当成通用事实；
- **R-N135 / Critical / Immediate / Open：** post-formal secondary 工作可能污染 V5 主门；
- **R-N136 / High / Immediate / Open：** 近似外部实现、少距离交点或缩小物理栈可能冒充 exact reproduction；
- **R-N125、R-N129：** 继续负责外部 FPGA latency boundary 和 pre-board/measured 区分。

## 是否需要继续插入 task

不需要。R-N133—R-N136 已由 T6.16—T6.19 正常顺序覆盖；T6.18.3 明确允许 `NOT_RUN_SCOPE_GATE`，T6.19.3 允许 negative/null/failed 但不允许证据混写，因此不应再为取得正排名插入旁路任务。

## 对任务板的同步

- 插入任务区新增 T-RISK-20260720-02 并标记 Done；
- Phase 6C 新增 T6.16.1—T6.19.3，共 12 个 Todo task；
- 当前推荐任务保持 T6.10.1；
- T6.9.2 保持 Blocked；
- Phase 7 增加 T6.19.3 secondary-integrity 前置，但 V5 主门仍由 T6.15.5 控制；
- 进度日志新增 User request -> In Progress -> Done。
