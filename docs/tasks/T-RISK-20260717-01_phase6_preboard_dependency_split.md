# T-RISK-20260717-01：Phase 6 真板依赖拆分

- 日期：2026-07-17
- 状态：Done

## 输入材料

- `docs/new_task_board.md` 中原 T6.1.1—T6.4.3 排程；
- T5.5.1—T5.5.4 的 bit-accurate golden、`cnn_fpga/rtl/gkp_fast_path_core.sv`、CXXRTL 等价和固定目标器件 P&R 记录；
- `docs/experiment_plan.md §14—§16`；
- `docs/new_risks.md` 中 R-N103—R-N108 的 evidence-ladder 边界；
- 用户要求：把“整个 Phase 6 等待真板”缩小为“只有板卡相关部分等待真板”，短期先完成 T6.2.1、T6.2.2 及长序列/故障路径软件仿真。

## 实际完成内容

1. 将 Phase 6 拆为两条依赖轨：板卡无关资格验证轨 `T6.2.1 -> T6.2.2`，以及实物板卡轨 `T6.1.1 -> T6.1.2 -> T6.1.3`；两轨在 T6.2.3 汇合。
2. 将当前推荐任务从被外部依赖阻塞的 T6.1.1 切换到 T6.2.1；T6.1.1 保持局部 Blocked，不再向 T6.2.1/T6.2.2 传播。
3. 把 T6.2.1 从泛化的“实现 RTL”升级为对现有 T5.5 core 的 production requirement-to-RTL 审计与缺口补强，显式排除硬编码 trace、组合 demo、删除状态路径和 activity harness 冒充核心实现。
4. 把 T6.2.2 扩展为独立 golden/CXXRTL 全状态逐周期对拍、每类至少 `1e5`/聚合不少于 `1e6` cycles 的长序列，以及 CRC/version、bank、reset、deadline、commit race 和抽象 transport 故障路径验收。
5. 将 T6.4.1/T6.4.3 明确保留为 T6.2.3 后的实际板卡/HIL 重复实验，禁止拿 T6.2.2 的预板结果替代板测证据。
6. 新增 `experiment_plan.md §17`、R-N109 和治理回归，锁定依赖图及 evidence ladder。

本任务只修订执行合同与治理文档，没有宣称 T6.2.1/T6.2.2 已完成，也没有生成 bitstream、真实 transport 或板测结果。

## 产物路径

- `docs/new_task_board.md`
- `docs/experiment_plan.md`
- `docs/new_risks.md`
- `tests/test_new_task_board_governance.py`
- `docs/new_tasks/T-RISK-20260717-01_phase6_preboard_dependency_split.md`
- `docs/tasks/T-RISK-20260717-01_phase6_preboard_dependency_split.md`

## 验证方式和结果

- 运行 `tests/test_new_task_board_governance.py`，检查当前指针、状态、双轨依赖、T6.2.2 长序列/故障验收及 T6.4 板测不可替代边界。
- 运行相邻的 `tests/test_protocol_hierarchy.py`，确认 Phase 6 调整没有破坏既有协议层级约束。
- 运行 `git diff --check`，检查补丁格式。

具体通过数量以本轮进度日志和最终汇报中的实际命令结果为准，不在执行前预填。

## 风险复核

- 新增并缓解 R-N109：全局真板阻塞会延迟合法的软件/RTL 缺陷发现，并混淆预板资格验证与实际板测证据。
- R-N105 仍为 Open：没有实物前，bitstream、vendor signoff、board、transport、power 均保持 false。
- T6.2.2 的抽象 transport fault model 只验证数字逻辑响应，不验证真实 UART/USB-SPI/JTAG 电气、驱动、吞吐或时序。

## 是否需要插入新 task

不需要。下一顺序任务直接为 T6.2.1；T6.1.1 等待实物信息，T6.2.3/T6.4 等待两轨汇合。

## 对任务板的同步说明

- 已在插入任务区登记本任务为 Done；
- 已将当前推荐任务改为 T6.2.1；
- 已同步 Phase 6 双轨依赖、T6.2.1/T6.2.2 验收、T6.2.3 汇合条件和 T6.4 板上重复条件；
- 已在进度日志记录 Proposed -> In Progress -> Done。
