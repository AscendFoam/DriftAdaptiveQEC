# T-RISK-20260830-01 正式资源 wall 可行性修复

- Task ID：`T-RISK-20260830-01`
- 标题：不缩分母的 powered twin resource-wall feasibility repair
- 日期：2026-08-30
- 状态：`In Progress`
- 来源风险：`R-N193`、`R-N198`、`R-N199`

## 输入材料

- V5c 终态失败记录：
  `docs/new_tasks/T-RISK-20260728-04_resource_preflight_wall_fail_full_v5c_20260827_220849.md`
- V5c raw resource evidence：
  `runs/t04_resource_preflight_full_v5c_20260827_220849/`
- 冻结 T04 config/plan/source、T03 design repair 与 T06 count confirmation。

## 目标

在不访问 formal outcomes、不缩小 518 cells / 2,085,888 rows、不改变 seed、
horizon、paired-cluster count、RESET estimand、3,043 门、margin 或 14-day gate 的
前提下，定位约 `3.549x` wall 超额的计算热点，并建立 exact-output 等价的实现或
调度修复。若不存在合规修复，则把 T04 终态保持为 resource NO-GO，而不是伪造
PASS。

## 计划产物

1. 可重放的 per-component wall attribution 与 deterministic LPT 预测；
2. 候选优化的 outcome-blind contract，逐项说明不变的数学/随机/证据语义；
3. scalar、density、RESET、backend-A/B 的 bitwise 或预注册数值等价测试；
4. 独立 resource projection/verifier；
5. 仅当 projected wall `<=1,209,600 s` 且 RSS/artifact/disk/concurrency 全过时，
   才使用全新 run ID 执行 fresh full-resource preflight。

## 通过标准

- 518 cells、2,085,888 rows、8 类 full-denominator profiles、3,043 门和所有
  claim-null 边界逐字段不变；
- 优化前后 deterministic row/NPY/receipt semantics 逐项等价，不能只验证 demo
  或小维 happy path；
- 投影方法继续使用 deterministic LPT、transient/inventory/physicality/joint-maxT
  全部组件，禁止 `sum/workers` 下界代替 makespan；
- 独立 verifier 在 fresh raw 启动前确认 projected wall 不超过 14 days；
- 若任一门失败，停止并保留 resource NO-GO，不启动重复 full run。

## 当前完成

- 已修复 late resource-gate failure 丢失 projection/decision 的诊断持久化缺陷；
- early-worker 与 late-wall fail-closed 回归均覆盖，聚焦测试 `25/25` 通过；
- 尚未修改 scientific kernel 或启动新 full run。

## claim 边界

本任务是 outcome-free infrastructure/resource repair。所有 official/Puviani/SOTA、
LER、lifetime、physical、hardware、rank 与 twin qualification 字段保持 `null`。
