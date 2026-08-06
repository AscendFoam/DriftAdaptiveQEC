# T-RISK-20260728-04 V5 full-resource 主机重启中断记录

- 日期：2026-08-07
- 事务：`full_v5_20260730_102415`
- 状态：`INCOMPLETE_EXTERNAL_HOST_RESTART`
- 性质：无终态的基础设施中断证据；**不是**科学 `NO-GO`、resource PASS、twin qualification 或性能结论
- 后续：旧事务及其 staging 永久只读；以新 run ID 和同一 V5 immutable 合同 fresh 重跑

## 诊断结论

该事务于 2026-07-30 10:24:15 启动，最后有效 heartbeat/采样时间为
2026-07-30 20:49:22。当前没有任何 Python 进程，owner PID `18132` 与
children `5852/15240/26240/39964` 全部不存在。旧 owner 的 boot-session 为
`7ea76ca44623d28a0beffc0d962fd81904f1f347693527f3ad57497e9a5fbc78`，
当前主机 boot-session 为
`8c20f05e9597a4f8d8274bb796a4a476c89b231262108fe07630cde17158ca65`，
证明进程消失跨越了一次主机重启，而不是仍有隐藏 owner 可以恢复。

事务没有写出 `resource_preflight.json`、`resource_preflight_failed.json` 或
terminal attempt event；`attempts.jsonl` 只有合法的
`START_RESOURCE_PREFLIGHT`。这符合操作系统/主机在 Python fail-closed handler
之外终止的证据形态。stdout/stderr 均为 0 B；没有 Python traceback、OOM、GPU
错误或 backend disagreement 证据，因此不得把中断解释为代码失败或科学结果。

## 保留证据

目录：

- `runs/t04_resource_preflight_full_v5_20260730_102415/`
- `runs/t04_resource_supervisor_full_v5_20260730_102415/`

关键文件（bytes / SHA-256）：

- `owner.lock`：`457 / 37530925ff9e09dbcc72a061cb99dcea2572a3d0c32b22ed48749e11bfc22cce`
- `heartbeat.json`：`473 / 0435f8423d2984902fffebb84b043b2fe3005d7a41d8707b1634681636406ce6`
- `attempts.jsonl`：`778 / d81690b83f660280b0ac4996d3ed42825fceac8be24212c1a0c1e19bb9d066c8`
- `resource_samples.jsonl`：`4,527,311 / 343f803a0a09621aa5b0af2057c17a4fd106cda25b3c94e98053c91efc117e98`
- supervisor `stdout.log` / `stderr.log`：均为 `0 B / e3b0c442...b855`

最后 heartbeat：

- sequence：`1249`
- stage：`formal_lpt_four_worker_peak`
- profiles completed：`0`
- child PIDs：`5852/15240/26240/39964`

连续 resource sampling：

- 最后 sequence：`7413`
- 最后 monotonic：`37,506.297 s`
- 最后 aggregate RSS：`4,921,667,584 B`
- 最后 live children：`4`

旧事务保留：

- receipts：`0`
- published objects：`0`
- staging：`30` files / `8,122,809,710 B`

这 30 个 staging 文件没有 receipt，不能进入 inventory、resource projection、
preformal seal 或正式统计，也不能迁移到 fresh run。

## Fresh 重跑边界

- V5 config、plan、seed registry、historical scan、source snapshot 与统计合同不变；
- 使用新 run ID、新 owner token、新 artifact namespace；
- 不删除、不补写、不复用旧 staging；
- 仍需完整 8 receipts / 227,328 resource rows、两组四-worker stage、全 raw/NPY、
  LPT/inventory/attempt/heartbeat consumer PASS；
- resource PASS 前不生成 V2 preformal seal，不启动 518-cell formal；
- 全部 twin/LER/lifetime/physical/hardware/official/Puviani/SOTA/rank 字段保持 `null`。

## 风险与任务判断

该中断是一次可明确归因的外部主机重启，不要求修改科学代码、门或样本量。
因此不插入新 task；继续在 T-RISK-20260728-04 内以 fresh run 重试。若同一
V5 在无主机重启条件下再次停在相同阶段，才需要把它作为重复性运行缺陷重新诊断。
