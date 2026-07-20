# T-RISK-20260717-02：插入 Route-A contract-centric 安全自适应双回路 Phase

- **日期：** 2026-07-17
- **状态：** Done
- **来源风险：** R-N084、R-N086、R-N098、R-N110
- **执行位置：** T6.2.2 后、Phase 7 前

## 输入材料

- 用户确认的路线 A：static/adaptive MAP 负责 LER，HMM/event/fallback 负责 tail safety，FPGA fast path 负责 deterministic latency，CNN/teacher/student 为可替换扩展；
- 用户给出的 unified integration bridge、场景、指标和通过门；
- `docs/new_task_board.md` 的 Phase 6 双轨依赖、T5 强分支证否和 Phase 7 旧叙事；
- `docs/new_risks.md` 中 `55/512 > 37/512`、compound/nominal fallback 反例、模块未统一与 hardware evidence boundary；
- Puviani 官方 MIT 仓库 `https://github.com/Matteo-Puviani/GQF` 及本地 T2.3.7 方向性复现边界。

## 执行前方案

1. 不重编号 Phase 7，使用 Phase 6A 和数值 Milestone 6.5—6.9，保持既有 task parser/历史记录稳定；
2. 让板卡无关工作只依赖 T6.2.2，只有 measured hardware task 依赖 T6.2.3/T6.4；
3. 先冻结 claim、execution contract、split 和统计门，再实现统一 adapter 与 policy，避免 evaluation 后改阈值；
4. 把 smooth LER、abrupt/OOD tail 和 `1e6`-cycle fixed-point/RTL 设为共同 promotion gate；
5. 把 static/drift decoder、GQF controller、FPGA hardware 分成不可混排 lane；
6. Phase 7 只消费 Route-A GO 后的证据，学习模块失败时自动降为消融或 supplement。

## 实际完成内容

- 将任务板标题与暂定核心贡献从 CNN-centric 改为 contract-centric、regime-aware safe adaptive dual-loop；
- 在 Phase 6 与 Phase 7 之间插入 Phase 6A，新增 5 个 Milestone、20 个 task：
  - 6.5：claim、unified contract、预注册；
  - 6.6：统一 comparator、regime policy、因果/校准审计；
  - 6.7：smooth、abrupt/OOD、integrated long-sequence RTL、promotion/falsification；
  - 6.8：static GKP、general drift、official GQF intake/exact reproduction/matched extension、FPGA normalization、innovation matrix；
  - 6.9：integrated P&R、actual-board measurement、高水平论文 GO/NO-GO；
- 把用户指定的共同 syndrome/MAP-LUT/Q-format/6-cycle/A-B bank/cadence/observed-only/budget contract、全部 comparator、policy transition、场景、指标和数值/统计门写入 task acceptance；
- 官方 GQF intake 要求固定 commit、MIT license、environment lock、upstream hash 和 patch series；paper-exact reproduction 未通过时禁止 “surpass NMF”；
- 调整 Phase 7 主图、Methods/Results 和 reviewer-risk task，使其消费 Route-A 而不是默认 teacher/student 主线；
- 在 `docs/experiment_plan.md` 增加 §18 低频修订，并新增 R-N110 跨域伪排行榜风险。

## 产物路径

- `docs/new_task_board.md`
- `docs/experiment_plan.md` §18
- `docs/new_risks.md`
- `docs/decoder_controller_terminology.json`
- `tests/test_new_task_board_governance.py`
- `docs/new_tasks/T-RISK-20260717-02_route_a_phase_insertion.md`
- `docs/tasks/T-RISK-20260717-02_route_a_phase_insertion.md`

## 验证方式和结果

- 检查任务 ID 唯一性、状态字段、Phase 6A Milestone 6.5—6.9 与 20 个 task 是否完整；
- 检查当前推荐任务仍为 T6.2.1，且 Phase 6A 明确从 T6.2.2 后开始、T6.9.2 才等待真板；
- 运行 `tests/test_new_task_board_governance.py`，锁定 unified contract、全部 comparator、场景、关键通过门、official GQF URL、外部 lane nonmixing 与 Phase 7 前置条件；
- 运行 Markdown 表格/重复 ID/`git diff --check` 审计；具体命令和最终数量记录在本 task 完成时的验证日志中。

实际结果：治理测试 `6 passed`，相邻 decoder/controller 术语绑定测试 `15 passed`；Phase 6A task `20/20` 且 ID 唯一；主任务定义区重复 ID 为 0；任务板与风险表 Markdown 表格列数错误为 0；两份完成记录 SHA-256 一致；`git diff --check` 无 whitespace error。检查同时发现并修正风险表 R-N037 中既有 raw pipe 导致的 Markdown 分栏问题，并把因 Phase 6 依赖行移动而过期的 `BIND-STUDENT-RTL-PLAN` 行锚更新到 T6.2.4 当前行；两者均不改变科学/实现状态。

## 非简化实现复核

本 task 是治理与实验设计任务，不把“列几个实验名称”视为完成。每个主要 claim 都拆成前置 contract、真实执行、统计/故障验收、失败分支和外部证据边界；CNN、GQF 和 FPGA 结论均设置不可绕过的否决门。未来代码 task 必须运行真实 checkpoint、official source、independent golden/CXXRTL、held-out seeds 与 actual board，不允许复用保存数值、demo FSM 或 P&R estimate 冒充结果。

## 风险复核

- R-N084 保持 Mitigated：旧 learned 强主张不恢复；只有 T6.7.4/T6.8.5 通过才有限重开；
- R-N086、R-N098 保持 Open：已获得明确正常 task 和关闭条件，但尚未执行新实验；
- 新增 R-N110（High/Soon）：禁止把 decoder/controller/hardware 不同域拼成 SOTA/fastest/surpass；
- R-N109 保持 Mitigated：Phase 6A 板卡无关部分不等待真板，证据升级边界不变。

## 是否需要继续插入 task

不需要。风险已由 T6.5—T6.9 正常顺序完整承接；当前执行指针保持 T6.2.1，不能跳过 T6.2.1/T6.2.2 直接进入 Phase 6A。

## 对任务板的同步

- 插入任务区新增 T-RISK-20260717-02 并标记 Done；
- 进度日志新增 Proposed -> In Progress -> Done；
- 当前推荐任务不变；
- Phase 7 增加 `T6.9.3=GO` 前置条件。
