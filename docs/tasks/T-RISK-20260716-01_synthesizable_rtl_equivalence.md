# T-RISK-20260716-01 可综合 RTL 前置任务完成记录

- Task ID：T-RISK-20260716-01
- 标题：补齐 T5.5.2 的可综合 fast-path RTL 前提
- 日期：2026-07-16
- 状态：Done

## 输入材料

T5.5.1 packed-word Python golden、v0--v7 image registry、T4.2 MAP/FSM/fallback contract、T4.3
atomic bank contract、Tang Nano 20K `GW2AR-LV18QN88C8/I7` target boundary。

## 实际完成内容

实现 CRC-protected input/output/state、双 bank/双 phase mirrored 1R1W BSRAM tables、精确 ties-to-even
插值、5+1-cycle pipeline、safe commit、event/fallback/frame/counter state。首轮非法 2R1W BRAM 被真实
Yosys synthesis 拒绝后，改为 8 个同步 1R1W physical copies；未以 LUT demo 绕过。

## 产物路径

详见 `docs/synthesizable_rtl_equivalence.md`；正式 machine artifact 为
`docs/t_risk_20260716_01_rtl_equivalence.json`，Source Data 为同名前缀 CSV。

## 验证方式和结果

从当前 RTL 生成 CXXRTL 后，226-cycle fault/commit trace 与 4,102-cycle exhaustive v0/v1 trace 共
4,328 rows 全字段 0 mismatch；4,316 valid MAP decisions；8/8 gates、8/8 mutations、13 focused tests
通过。Yosys elaboration 保留 8 memories、45,232 bits、multiplier，0 structural problems。

## 风险复核

R-N104 降为 `Mitigated / Medium / Monitor`。R-N103 的 target synthesis/post-route/board 字段尚未由本
插入任务升级；立即恢复 T5.5.2。activity harness 与官方 QN88 package pins 不能冒充 transport 或板测。

## 是否需要插入新 task

否。真实 2R1W 失败已在本任务内修复；T5.5.2 正常承接器件报告。

## 对 `docs/new_task_board.md` 的同步说明

T-RISK-20260716-01 `In Progress -> Done`；T5.5.2 `Blocked -> In Progress`；当前推荐恢复为 T5.5.2。

