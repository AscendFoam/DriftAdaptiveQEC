# T-RISK-20260728-04 V5b full-resource 主机重启中断记录

- 复核日期：2026-08-27
- 事务：`full_v5b_20260807_003952`
- 状态：`INCOMPLETE_EXTERNAL_HOST_RESTART`
- 性质：无终态基础设施中断；不是 resource PASS、科学 `NO-GO` 或 twin qualification

## 诊断

事务最后 heartbeat 为 2026-08-07 01:18:32，最后 resource sample 为
01:18:34；此后文件不再更新。owner PID `7440` 与原四个 Python children
`2288/23756/27936/29480` 均已消失（当前同号 `27936` 已被 Chrome 重用）。
owner 记录的 boot-session 为
`5386d77653b9da8a31430c21d39eba0f7bb5bd000825176ed1f59924c7dd7b88`，
当前 boot-session 为
`b67279a36b647096bfe7594a981c13cbea749767980fdc205c90348bc7f8f84f`，
因此确认又一次主机重启发生在 Python fail-closed handler 之外。

事务只有 `START_RESOURCE_PREFLIGHT`；没有 PASS/FAIL terminal event、
`resource_preflight.json` 或 `resource_preflight_failed.json`。stdout/stderr 均为空。
最后仍处于 `formal_lpt_four_worker_peak`、4 live children、0 completed profile；
resource sample sequence 为 `67`、monotonic=`2322.015 s`、aggregate RSS=
`466,513,920 B`。目录保留 0 receipt、0 published object 和 30 个未发布 staging
文件（最后核验 `7,768,490,750 B`）。这些 staging 不得复用或进入任何 seal。

## 后续边界

保持 V5 config/plan/seed/statistics 不变，以新 run ID、owner token 和 artifact
namespace fresh 重跑。不得补写旧 terminal、缩小 profile、删除分母或把中断解释为
科学结果。resource PASS 前，所有 official/Puviani/SOTA/LER/lifetime/physical/
hardware/rank 字段继续为 `null`。

这是第二次可证明的外部主机重启，但两次中断发生在不同运行时长，尚不构成同一代码
阶段的重复性故障；在当前 T04 内继续，不插入科学 task。若 fresh 运行在主机稳定时仍
重复停滞，再另行审计 checkpoint/resume 或 bounded-profile 事务设计。
