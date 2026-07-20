# T-RISK-20260721-01：插入双证据 lane Phase 6D

- **Task ID：** T-RISK-20260721-01
- **标题：** 将论文重组为 multimode software LER 与 single-mode deterministic RTL 两条不可混淆的证据 lane
- **日期：** 2026-07-21
- **状态：** Done
- **来源风险：** R-N146—R-N153

## 输入材料

- 用户指定的新论文组织：multimode 软件仿真争取 LER SOTA；single-mode 六周期 RTL 负责 deterministic、atomic、fail-closed；CNN/student 只作 posterior/MLD 近似或蒸馏；
- T6.18.2 official structured-lattice CPD reproduction；
- T6.18.3 `d=3` balanced heteroscedastic multimode 9.6M-cycle development result；
- T6.19.1 single-mode 6-cycle/II=1 pre-board profile 与 T6.2.2/T6.7.3/T6.9.1 长序列/P&R 证据；
- Phase 6B `NO_GO_V5_EARLY_HEADROOM_STOP`、Phase 6C secondary integrity 与 T7.1—T7.2 restricted historical snapshot；
- `docs/multimode_strong_baseline_registry.md` 的一手文献和官方源码审计。

## 执行前方案

1. 不把 T6.18.3 的正结果直接升级；先确认 exact MLD、K-MWM、analog-MWPM 和 syndrome-only adaptive 强基线；
2. 把算法与硬件拆成两个独立 task signature、独立主指标、独立 GO/NO-GO，不做跨 lane 总分；
3. 新算法先过 causal headroom，再复现强 baseline、实现 posterior-predictive coset MLD、冻结全新 split，最后一次性 formal；
4. single-mode RTL 只重验 actual production top 的 property、long-sequence 和 multi-seed P&R，不声称执行 multimode MLD；
5. CNN/student 延后到 classical teacher 已冻结之后，只做 matched-budget approximation；
6. 保留现有 Phase 6C/T7 Done 产物为不可变历史快照，以新增 delta task 更新论文。

## 实际完成内容

- 在 `docs/new_task_board.md` 的 Phase 6C 与 Phase 7 之间插入 Phase 6D，共 7 个 Milestone、30 个 task：
  - M6.20：强 baseline、双 lane、全新 split 和 causal headroom；
  - M6.21：official exact MLD、analog-MWPM、K-MWM、static-mixture 与 noisy-auxiliary 边界；
  - M6.22：Window/EWMA/Kalman、SMC-EAP/GP、BOCPD/IMM 与 matched-budget 资格；
  - M6.23：observed-only posterior、posterior-predictive coset MLD、risk policy、收敛与 pilot；
  - M6.24：`d=3/5` multi-sigma untouched formal、tail/OOD 和 10% simultaneous-LCB SOTA 门；
  - M6.25：single-mode actual RTL 的 boundary/property/CXXRTL/P&R lane；
  - M6.26：CNN/student optional distillation、双 lane figure/claim matrix 和最终四态 verdict。
- 将 T6.20.1 标记 Done、T6.20.2 标记 In Progress；当前推荐改为 T6.20.2。
- 将已有 T7.3.1 partial 工作转为 Blocked/只读停放，等待 exact MLD/oracle 层级；没有删除或覆盖其工作文件。
- Phase 7 新增 T7.1.5、T7.2.6、T7.3.8、T7.3.9，并更新 T7.4 发布门；历史 T7.1.1—T7.2.5 保持 Done。
- 在 `docs/experiment_plan.md` 新增 §19；在 README 增加 baseline registry 入口；在 `docs/new_risks.md` 新增 R-N146—R-N153。

## 产物路径

- `docs/new_task_board.md`
- `docs/experiment_plan.md`
- `docs/new_risks.md`
- `docs/multimode_strong_baseline_registry.md`
- `README.md`
- `docs/new_tasks/T-RISK-20260721-01_dual_evidence_lane_phase6d.md`
- `docs/tasks/T-RISK-20260721-01_dual_evidence_lane_phase6d.md`

## 验证方式和结果

- 机械检查 T6.20.1—T6.26.4 的 task ID 唯一性、状态顺序和总数；
- 检查 `T6.20.1=Done`、`T6.20.2=In Progress`、`T7.3.1=Blocked`、`T6.9.2=Blocked`；
- 检查 exact MLD/K-MWM/analog-MWPM/static-mixture 与 Window/SMC/BOCPD 等强基线均有 task；
- 检查新 formal 明确不复用 T6.18.3，SOTA 门是对每个 eligible baseline 的 simultaneous 95% LCB `>10%`；
- 检查 multimode-LER→RTL、single-mode-latency→multimode、CNN→主门三类迁移均被禁止；
- 检查 Phase 7 旧 Done 产物未改状态，新稿只通过 delta task 更新；
- 运行任务板治理测试、Markdown 表结构检查和 `git diff --check`。

实际结果：新增 Phase 6D 专项治理测试后，`tests/test_new_task_board_governance.py` 为 `9 passed`；与 Phase 6C preregistration/integrity 相邻联合回归为 `30 passed`。首次联合回归的唯一错误来自系统 pytest 临时目录 ACL；改用工作区内已核验临时目录后全绿，并安全删除该临时目录。机械统计为 30 rows / 30 unique IDs / 0 duplicates，状态为 1 Done、1 In Progress、28 Todo；`git diff --check` 无 whitespace error（仅仓库既有 LF/CRLF 提示）。

## 非简化实现复核

本次没有用“新增一个 multimode task”代替完整实验闭环。Phase 6D 在正式大跑前设置两次可证否 early-stop：causal headroom 与 pilot；强 baseline 要求 official/source-transcribed/project-native 分级、brute-force/exact 对拍、同 backend factorial 和统一计算预算。Formal 覆盖距离、噪声强度、未见空间模式、连续/突变/OOD，并要求对所有 eligible baseline 同时过门。RTL lane 要求 actual module 的 property/cover/mutation、百万周期和多 seed P&R，不能复用 PASS 字符串。CNN/student 失败时有明确 Dropped 分支。

## 风险复核

- R-N146—R-N150、R-N152—R-N153：Open，Immediate/Soon；由 T6.20—T6.24 直接证伪；
- R-N151：Open/Soon；由 T6.26 冻结学习模块边界；
- R-N132：继续覆盖 single-mode full-state proof/near-wrap 缺口；
- R-N145：继续覆盖 immutable snapshot 与环境漂移；
- T6.9.2：继续 Blocked，所有 measured fields 保持 null。

## 是否需要继续插入 task

不需要。当前八项新风险已由 T6.20—T6.26 的正常顺序覆盖；任何为取得正 SOTA、替换失败 reproduction 或伪造板测而新增的旁路都会破坏新协议。只有 source/task signature 出现此前未登记的 blocking discrepancy 时，才按风险规则另行插入。

## 对任务板的同步

- 插入任务区新增本 task 并标记 Done；
- Phase 6D 新增 30 个 task；
- 当前推荐切换为 T6.20.2；
- T7.3.1 暂停为 Blocked；
- T6.9.2 保持 Blocked；
- 进度日志记录 User request → In Progress → Done，以及 T6.20.1 → T6.20.2 的切换。
