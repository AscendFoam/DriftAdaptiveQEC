# T-RISK-20260727-01 高 cutoff 六态分层设计与风险定位

- Task ID：T-RISK-20260727-01
- 日期：2026-07-27—2026-07-28
- 状态：Done（`EXPLORATORY_RISK_SIGNAL`，不是 twin qualification PASS）
- Pilot manifest analysis：`8e48faad8cc5f94204a52e9f96c8c42850f7f8e0f937dfb8d188d925f91c4904`
- Diagnostic analysis：`c4ddfba3b060775000521dfd4a9c233ec3958acd12df36462dbfa3aef8b9a0ac`
- Diagnostic completion analysis：`b2e3232434ddbb6f749e47aa6ecbcc340beac5c4beeb7a95fcd9c7272b85d1a1`

## 输入材料

- T9.2.4 与 T-RISK-20260726-01 两个不可回写的 twin qualification NO-GO；
- R-N184：高能尾与六态聚合抵消；
- R-N185：trace-norm UCB coverage 与功效；
- R-N186：artifact writer CLI 可能在 `--help` 时误写；
- backend A/B、fresh runner、paired-cluster UQ、独立 coverage calibration 与 hardened confirmation；
- 完全独立于旧 formal 的 trajectory/heldout seed namespace。

## 执行方案

1. 先修复四个 artifact writer 的 argparse/零写合同，并用独立 null/local/margin 模拟校准 paired-cluster UCB。
2. 冻结六态 × 四 fault scenario × A/B × cutoff 16/20/24/28 的 32-cell design pilot；每 cell 72 条 trajectory、12 rounds。
3. 保存每条 round ledger、raw/heldout IQ 和每 trajectory 的终态 joint density；每个 receipt、CSV、NPZ、manifest 与 launch meta 均做 byte/analysis SHA-256 绑定。
4. 诊断只允许 `EXPLORATORY_RISK_SIGNAL`、`NO_LARGE_SIGNAL_INCONCLUSIVE` 或 `INCOMPLETE`，不允许产生资格化结论或释放下游。
5. cutoff32 仅由预注册规则触发：任一 24→28 state×scenario×backend density point `>0.075`，或任一 cutoff28 绝对 tail point超过其 margin。

## 实际完成内容

### Coverage/UQ 与冻结入口

- 完成 paired-cluster UQ calibration、power extension、第三 hardened confirmation 与独立审计；
- formal-domain 选择保持 384 clusters/state；pilot 只使用 12 clusters/state 做定位；
- 冻结 fresh3 pending config、release receipt、released child、pilot/diagnostic/bootstrap source 和外部 launcher literal；
- cutoff32 capability preflight 覆盖 A/B 的 96×96 joint density，但 production pilot 本身只执行 cutoff 16/20/24/28。

### Production pilot

- run：`runs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3`
- 32/32 cells、64 个 CSV/NPZ chunks、27,648/27,648 unique rows；
- 2,304 个终态密度矩阵：cutoff16/20/24/28 各 8×72；
- 零 exception、零 conservation failure、零重复 row/cell；
- 最大 density trace error `3.4001e-8`，最大 Hermiticity Frobenius error `0`，最小 eigenvalue `-3.0171e-16`；
- 10 项 cutoff16—32 capability checks 全部通过；
- PID 正常退出、owner lock 原子清理、stderr 为 0、heartbeat=`COMPLETE`；
- manifest byte SHA-256=`37e7606d2d8ad1d06b46dfec9b21580a2f325eae7a0b552a4109dd85f5dfef91`。

### Fail-closed 诊断修复

首次冻结 diagnostic 正确返回 `INCOMPLETE/ValueError`，原因是非 terminal rows 的 density quantization certificate 按 schema 为空，而 V1 对所有行执行 `float("")`。原 pilot、V1 launch meta 和日志均保留。

新增 `phase9_high_cutoff_design_bootstrap_v2.py`，不改动 pilot/V1 bootstrap/hash：

- terminal row 必须有 finite、non-negative certificate；
- non-terminal row 必须为空；
- missing、NaN、negative、wrong-row certificate 均 fail closed；
- V2 与 V2b 的失败 launch/log 分开保留；
- 最终 V2b 从已推送 Git commit `c863e2d64a5d528f3275ffdcd84a5ef68cb1983a` 启动，并绑定 V2 bootstrap、V1 diagnostic、V1 pilot manifest 与独立 launch meta。

### Diagnostic 结果

- 3,900/3,900 unique diagnostic gates；
- verdict=`EXPLORATORY_RISK_SIGNAL`；
- 24→28 density comparison 48 个 state×scenario×backend 点，最大 point=`0.09134328001208851 > 0.075`；
- cutoff28 有 240 个 absolute-tail diagnostics，其中 6 个 point 超过预注册 margin；
- cutoff32 follow-up candidate=`true`；
- candidate/strong exploratory rows分别为 375/699；
- completion receipt、report 与 1,161,786-byte Source Data 已独立重算并逐 binding 验证。

## 产物

- `docs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3_manifest.json`
- `docs/t_risk_20260727_01_high_cutoff_design_diagnostic_fresh3.json`
- `docs/t_risk_20260727_01_high_cutoff_design_diagnostic_fresh3_source_data.csv`
- `docs/t_risk_20260727_01_high_cutoff_design_diagnostic_fresh3_completion.json`
- `cnn_fpga/benchmark/phase9_high_cutoff_design_bootstrap_v2.py`
- `tests/test_phase9_high_cutoff_design_bootstrap_v2.py`
- `runs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3/`

## 验证

- 全量独立 raw audit：32 receipts、64 chunks、27,648 rows、2,304 densities、全部 byte/self hash、row/density alignment、物理性、守恒与 claim firewall PASS；
- completion/source-data 独立重算：3,900 unique gates、48 个 24→28 density 点、240 个 cutoff28 tail 点、6 个 trigger 与报告完全一致；
- V2 parser tests：7 passed；
- V1 diagnostic tests：15 passed；
- pilot tests：32 passed；
- fresh writer CLI zero-write tests：12 passed；
- task-specific 隔离回归：66 passed；治理文档测试另有 10 passed；
- 组合运行最初暴露 diagnostic fixture 替换 canonical verified modules 后未恢复的跨文件污染；补齐 `sys.modules`、package attribute 与 diagnostic globals 的 finally 恢复后，V2 parser、V1 diagnostic、pilot、CLI 与 governance 同进程组合回归为 76 passed；
- 此前 fresh3 hardening focused regression：346 passed。

## 反简化审计

- 没有把 10 项 capability preflight 写成 cutoff32 production evidence；
- 没有把 logical-survival 稳定替代完整 density/tail 收敛；
- 没有跨六态平均后掩盖 `+`、`-i` 等 tail-heavy 分层；
- 没有在当前 immutable run 中追加 cutoff32；
- 没有用 pilot 数据选择新 margin、阈值或正式样本量；
- 首次与 V2 首次 fail-closed 均保留，未覆盖日志或 launch meta；
- 没有用“单文件测试通过”掩盖组合测试污染；已实际修复并用 76 项同进程组合回归确认；
- 两个既有 twin NO-GO 均未回写；
- LER、lifetime、physical break-even、hardware measured、official/Puviani、external SOTA/rank 字段全部保持 literal `null`。

## 风险复核与插入任务

- R-N185 降为 Mitigated/Monitor：UCB 已在独立 null/local/margin 模拟上校准，并由 hardened confirmation 复核，但后续 powered formal 仍必须消费同一冻结统计合同。
- R-N186 降为 Mitigated/Monitor：四个 writer 已使用 argparse，12 项 help/unknown-arg 零写与 live artifact hash/mtime 测试通过。
- R-N184 仍为 Open/Critical/Immediate：24→28 与 cutoff28 tail 仍未收敛。
- 新增 R-N187，并插入 `T-RISK-20260728-01`：以新的 immutable child 执行 cutoff32/36 收敛扩展、必要 physics repair 与 fresh powered qualification。

## 对论文 claim 的影响

本任务只证明高 cutoff 风险真实存在且可定位，不能证明 twin qualified，更不能证明 LER、寿命、物理 break-even、FPGA 实测或任何外部 SOTA。六个 downstream task继续 Blocked；只有新任务的独立 powered formal PASS 才能重新考虑释放。

## 任务板同步

`docs/new_task_board.md` 将本任务标为 Done（Risk Signal），新增并启动 `T-RISK-20260728-01`；README 与风险表同步本次证据边界。
