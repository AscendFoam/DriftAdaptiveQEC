# T-RISK-20260722-01：插入 performance-first 单模 GKP Phase 9

- **Task ID：** T-RISK-20260722-01
- **标题：** 在 Puviani 官方资产不可得时建立可独立推进的单模多速率非马尔可夫双回路实验程序
- **日期：** 2026-07-22
- **状态：** Done
- **来源风险：** R-N162—R-N168

## 输入材料

- 用户指定的执行顺序：IQ/leakage/reset/action-conditioned 数字孪生、trusted recovery codebook、GRU/TCN/SSM/Transformer/Bayesian tournament、observed-only posterior、A/B/CRC/LKG/六周期集成、六态长序列 lifetime、高速板 HIL、最后决定论文拓扑；
- 用户明确的现实约束：暂时无法获得 Puviani official checkpoint、20-agent seeds、selection ledger 和 six-state evaluator；
- `docs/deep_research_reports/非马尔可夫的双回路GKP纠错.md`；
- T6.8.4/T6.17.3 的 GQF/Puviani reproduction 状态、T6.20.4 的 multimode v1 headroom NO-GO；
- T6.25.2—T6.25.4 已完成的 unique converged top、百万周期 CXXRTL 与三 seed pre-board P&R；
- T7.3.2—T7.3.4 已冻结的 learning、experiment-related、postselection/break-even 证据边界；
- `docs/experiment_plan.md §18--§19` 与 `docs/new_risks.md`。

## 执行前方案

1. 不把缺少 official Puviani artifact 解释为所有后续工作都不可做；把它限制为 official-exact/surpass claim 的局部 blocker。
2. 为项目自建 Puviani lane 固定 `PAPER_CONSTRAINED_REIMPLEMENTATION`，要求项目自有 20-agent seeds、全部 checkpoint、validation-only selection ledger、six-state evaluator 和 deviation ledger。
3. 先冻结单轮 LER、六态 lifetime、数字链 HIL latency 三个 task signature，防止跨分母总榜或系统优势补算法门。
4. 先建立两个独立 physics backend，再做 codebook 和模型；controller 不得只在训练 surrogate 上自评。
5. 把 slow AI 的动作权限限制为 observed-only posterior/risk 与 trusted codebook 选择；FPGA fast path 保留 bounded action、A/B/CRC/version/LKG、6-cycle/II=1。
6. 对 GRU/TCN/SSM/Transformer/Bayesian 使用同输入、同样本、同 action、同 compute，允许简单模型胜出并淘汰复杂模型。
7. 六态 formal 使用完整分母、无 postselection、长序列和独立重算；真实板 HIL 只阻塞 measured hardware 证据。
8. 最后按三项独立 verdict 决定单篇、拆篇、单 lane 或 NO-GO，不预设必须合并。

## 实际完成内容

- 在 `docs/new_task_board.md` 新增 v2.4 迁移说明和 Phase 9，共 8 个 Milestone、34 个 task：
  - M9.1：三 task signature、official/paper-constrained Puviani 双 lane、baseline/power contract；
  - M9.2：IQ/leakage/reset/action-conditioned 双后端数字孪生与 multi-fidelity surrogate；
  - M9.3：器件网格、offline optimized trusted codebook、量化编译与 coverage/fallback；
  - M9.4：不可变 split、Bayesian/classical baseline、CNN/GRU/TCN/SSM/causal Transformer、observed posterior 和一次性 tournament；
  - M9.5：posterior-risk selector、shadow/OOD/fallback、bank package compiler 和 unique-top formal；
  - M9.6：六态 per-round LER、`10^4`-cycle lifetime、tail/OOD 与独立 promotion gate；
  - M9.7：百万周期 integrated CXXRTL、硬件边界/平台冻结和真实高速板 IQ HIL；
  - M9.8：三项独立 verdict、论文真值表和可复现交付。
- `T9.1.2` 标记 Blocked，只等待 official Puviani 四类资产；`T9.1.3` 明确不依赖它。
- `T9.7.3--T9.7.4` 标记 Blocked，只等待高速实物板；T9.1.1—T9.7.2 的软件、预板和设计工作可继续。
- 冻结 SOTA candidate 门：对每个 eligible strongest deployable baseline，LER relative point `>=15%`、simultaneous paired 95% LCB `>=10%`；六态 lifetime minimum point `>=15%`、simultaneous LCB `>0`，并保留 reset/control/postselection 成本。
- 在 `docs/experiment_plan.md` 新增 §20，说明非阻塞证据等级、双后端独立性、codebook/模型权限、多速率系统、formal/HIL 门和论文拓扑。
- 在 `docs/new_risks.md` 新增 R-N162—R-N168，并在交叉检查、插入任务区和进度日志建立闭环。
- 保留当前 `T7.3.5=In Progress` 和 Phase 6D/T7 历史结果；本次规划没有用新愿景覆盖旧 NO-GO，也没有伪装成已经执行的性能结果。

## 产物路径

- `docs/new_task_board.md`
- `docs/experiment_plan.md`
- `docs/new_risks.md`
- `tests/test_new_task_board_governance.py`
- `docs/new_tasks/T-RISK-20260722-01_performance_first_single_mode_phase9.md`
- `docs/tasks/T-RISK-20260722-01_performance_first_single_mode_phase9.md`

## 验证方式和结果

- 检查 T9.1.1—T9.8.3 共 34 个 task ID 唯一，8 个 Milestone 齐全；
- 检查 `T9.1.2/T9.7.3/T9.7.4=Blocked`，其余 Phase 9 task 为 Todo，证明阻塞已局部化；
- 检查 T9.1.3 不依赖 T9.1.2，且 official-exact 与 paper-constrained 的允许措辞不同；
- 检查双后端“不共享 kernel/likelihood/truth/RNG”、controller 不自评 learned surrogate、hidden teacher 不进入 deployable formal；
- 检查模型 tournament 的同输入/样本/action/compute 和 full restart ledger；
- 检查六态、无 postselection、完整 denominator、`10^4` cycles、LER/lifetime simultaneous CI 和第二实现重算；
- 检查 HIL 四层 latency boundary 与 CXXRTL/P&R/host 不得升级 measured 字段；
- 运行 `tests/test_new_task_board_governance.py`、task ID/status 机械审计和 `git diff --check`。

实际结果：`tests/test_new_task_board_governance.py` 为 `10 passed`；Phase 9 机械统计为 34 rows / 34 unique IDs / 0 duplicates，状态为 31 Todo、3 Blocked，8 个 Milestone 齐全；全任务板 authoritative 区 0 duplicate task ID，两份完成记录逐 byte 一致。`git diff --check` 无 whitespace error（仅工作区既有 LF/CRLF 提示）。首次回归暴露的是治理测试两个过度字面化断言（任务 ID 反引号和区间写法），只修正测试定位，没有放宽任何科学/证据门。

## 非简化实现复核

本次没有把“换成 GRU/Transformer”写成单一 demo task，而是在其前设置双 physics backend 和 codebook headroom，在其后设置 observed-only permission、matched compute、完整 selection ledger、slow/fast integration、六态长时 formal、independent recomputation 与真实 HIL。每个关键环节均有明确失败分支：physics 对拍失败则不跑 lifetime；codebook 无 headroom 则回 static；简单模型胜出则复杂模型 Dropped；LER/lifetime 不过门则降级；无板只保留 pre-board；缺 official artifact 时不写 Puviani surpass。

## 风险复核

- R-N162：Open/Critical/Immediate；official 与自建 artifact 混写风险，由 T9.1.2—T9.1.3 闭合；
- R-N163：Open/Critical/Soon；双 backend 相关错误，由 T9.2.2—T9.2.4 闭合；
- R-N164：Open/Critical/Immediate；模型权限/预算混因，由 T9.4 闭合；
- R-N165：Open/High/Soon；action space 与 fast-path/压缩冲突，由 T9.3—T9.5 闭合；
- R-N166：Open/Critical/Soon；surrogate/长时外推，由 T9.2.5/T9.6 闭合；
- R-N167：Open/High/Soon；伪 measured HIL/speed，由 T9.7 闭合；
- R-N168：Open/Critical/Immediate；三 SOTA 跨门与择优，由 T9.1.1/T9.6.1/T9.8 闭合。

## 是否需要继续插入 task

不需要。七项新风险已经由 Phase 9 正常顺序覆盖。只有出现新的 source/physics/task-signature blocker、且现有 T9 task 无法承接时才另插风险任务；不得为了获得正结果、替代 official artifact 或绕过实物板而新增旁路。

## 对任务板的同步

- 插入任务区新增本 task 并标记 Done；
- 新增 Phase 9 / Milestone 9.1—9.8 / 34 个 task；
- `T9.1.2/T9.7.3/T9.7.4` 局部 Blocked；
- 其余 Phase 9 task 初始化为 Todo；
- 当前 active `T7.3.5` 保持不变，Phase 9 首个可执行 task 为 `T9.1.1`。
