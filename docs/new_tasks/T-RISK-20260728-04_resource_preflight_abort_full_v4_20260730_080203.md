# T-RISK-20260728-04 V4 full-resource 事务受控终止记录

- 日期：2026-07-30
- 事务：`full_v4_20260730_080203`
- 状态：`INCOMPLETE_RESOURCE_FAIL_CLOSED`
- 性质：失败尝试证据；**不是** task 完成记录、科学 `NO-GO`、twin qualification 或性能结论
- 后续：旧事务及其 staging 只读保留；修复后只能以新 run ID、V5 immutable 合同和新命名空间 fresh 重跑

## 终止原因

该事务运行期间的独立 consumer/anti-simplification 审计发现：即使四个 profile worker 自然完成，旧 V4 资源报告也不能形成足够严格、可被正式阶段合法消费的 release witness。主要问题包括：

1. profile 选择没有覆盖正式 518-cell LPT 队列的真实四个最高 transient cells，因而不能证明正式首批四 worker 峰值；
2. wall-clock 使用 worker 自报值，未同时绑定外部连续采样的 stage window，协调缩短四个自报值后仍有绕过风险；
3. per-cell artifact/transient 投影、top-4 inflight、确定性 LPT makespan、inventory finalize 与 mapping-anchor 开销未形成可逐项重算的闭合算术；
4. receipt/raw-row/NPY 的 identity、dtype、shape、有限性、文件精确长度和 RESET held-out window 尚未被 consumer 全量扫描；
5. attempt `START -> PASS`、owner token/process-creation、heartbeat、resource report、focused validation 和 preformal seal 的 live-byte 绑定不完整；
6. preformal/qualification consumer 的 schema 仍允许若干额外字段或旧 seal 版本，不能抵抗协调重写与 downgrade。

这些是会使后续 formal 消费不可信的 P0/P1 证据链缺口。继续让旧进程生成更多 profile 数据没有资格化价值，反而会把旧 V4 与修复后语义混在同一事务。因此在确认唯一 owner/children 身份后执行了定向受控终止；没有清理、移动或复用旧产物。

`resource_preflight_failed.json` 中四个 worker 的 `exitcode=4294967295` 是受控终止的直接进程表现，不是根因。采样显示峰值 aggregate RSS 只有 `1,128,509,440 B`，stdout/stderr 均为空；没有证据支持 OOM、host timeout、GPU 故障或 backend scientific disagreement。该失败不得解释为科学 `NO-GO`。

## 保留的不可变证据

旧运行目录：

- `runs/t04_resource_preflight_full_v4_20260730_080203/`
- `runs/t04_resource_supervisor_full_v4_20260730_080203/`

关键文件（`bytes / SHA-256`）：

- `resource_preflight_failed.json`：`3,717 / 39c6059258dc18e3ea42212b9a296a4fef4d0759eaca07a6c24bc671b7a20220`
- `attempts.jsonl`：`1,356 / 6a8221536c3cb6ee2301fc1415be7e7be518ae21dd0cc86c92fe046a1a440a6f`
- `heartbeat.json`：`470 / 5b0d872b7a03b18f24edabe947d24964e40b04c5af45211b95025766e8c20d9b`
- `resource_samples.jsonl`：`231,869 / 3e2f7656983873a0ca3e225a52680f164eecb526006f45cdb084ba0f007da335`
- supervisor `stdout.log` / `stderr.log`：均为 `0 B / e3b0c442...b855`

失败报告自哈希为
`45541cdd72430a0071cb10b58f680e1f997562e995be80a6d67c0b3f729f339e`，冻结输入包括：

- config SHA-256：`7ec664ca40c489c25074e5bf8005928e975662ab35d6c2678baa415e8be78fee`
- plan SHA-256：`d2e6fd930c0fb9f0271c0e0ef66e8396fefd1b424b0ca1a2ba01007f6e28376a`
- source snapshot SHA-256：`3e6a99a7172c6a3a46f92828cda4eaf0b78b5c7942296a778d468d7e3fb85364`
- resource sample count：`382`
- active-child sample count：`380`
- maximum observed live children：`4`
- peak aggregate RSS：`1,128,509,440 B`
- formal seed address accessed：`false`
- formal artifact namespace accessed：`false`

采样 hash chain 从 sequence `0 / starting` 连续到
sequence `381 / four_worker_concurrent_peak`。终止后已确认 owner PID
`45736` 与 children `9208/12932/13568/26652` 均不存在。最后 heartbeat
仍保留终止前的四 children snapshot；这正是失败事务现场证据，不能改写为完成态。

旧 staging 保留 `69` 个文件、`8,088,844,838 B`；没有 receipt 或已发布
content-addressed object。`START_RESOURCE_PREFLIGHT -> FAIL_RESOURCE_PREFLIGHT`
两事件闭合，失败报告与事件链均未声称 PASS。

## V5 修复边界

V5 不改变 powered formal 的科学问题、518-cell 分母、T06 已冻结的
`round=1536 / fault=768 per state`、3,043 门或全部 claim-null 边界；它只加固资源证据与正式消费链：

- 用正式 LPT 峰值四 cell 加四个跨 backend/action/profile 代表 cell，共 8 个 full-denominator receipts；
- 对 227,328 行 resource raw 和全部 NPY 逐行/逐块验证 identity、seed、shape、dtype、finite/domain、精确文件长度和 held-out hash；
- 逐 cell 投影 artifact/transient，重算全 518-cell top-4 inflight、确定性 LPT worker loads、mapping anchors 和 inventory-finalize wall；
- 由外部连续采样绑定两个四-worker stage window、RSS 和 PID 双射；
- 将 runtime fingerprint、owner/heartbeat、attempt witnesses、focused validation、resource report 和 live bytes 纳入 exact consumer；
- preformal seal 升为 V2，并在 qualification/verifier 中拒绝 V1 downgrade、未知字段和协调重写。

只有最终源码和全部 mutation/negative tests 冻结后，才允许物化一次新的
V5 contract/plan/seed ledger，并以全新 resource run ID 启动。V4 staging
不得参与 V5 的 receipt、投影、统计或 formal qualification。

## Claim 边界

本次终止及 V5 证据链修复均只涉及 resource/preformal qualification。
以下字段必须保持 `null`：

- `twin_qualification`
- `round_ler`
- `six_state_lifetime`
- `physical_break_even`
- `hardware_measured`
- `official_puviani_exact`
- `puviani_nmf_surpass`
- `external_sota`
- `rank`
