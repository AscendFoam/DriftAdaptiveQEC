# T-RISK-20260728-04 Resource / preformal V5 前瞻加固记录

- Task ID：`T-RISK-20260728-04`
- 标题：fresh powered dual-backend qualification
- 日期：2026-07-30
- 状态：`Blocked — V5c verified resource wall NO-GO; feasibility repair required`

## 输入材料

- T03 design repair independent PASS：
  `docs/new_tasks/T-RISK-20260728-03_cutoff36_44_bounded_repair.md`
- T05 high-dimensional joint-maxT statistical NO-GO：
  `docs/new_tasks/T-RISK-20260728-05_highdim_joint_maxt_no_go.md`
- T06 count selection / untouched confirmation independent PASS：
  `docs/new_tasks/T-RISK-20260728-06_count_selection_confirmation.md`
- T01 effect-dispatch / verifier-input repair：
  `docs/new_tasks/T-RISK-20260730-01_effect_dispatch_repair.md`
- 旧失败 resource 事务：
  `runs/t04_resource_preflight_full_20260730_0655/`
- 旧 V4 受控终止事务：
  `runs/t04_resource_preflight_full_v4_20260730_080203/`
- 冻结 config：
  `configs/phase9/t_risk_20260728_04_powered_twin_qualification.json`

## 执行方案与实际完成内容

本阶段没有缩小 518-cell 正式矩阵、2,085,888 行分母、3,043 门、
T06 `round=1536 / fault=768 per state` 或统计 margin。修复对象仅是
resource evidence 与 formal consumer 的可审计性。

### 1. Immutable receipt 与 runtime lineage

- receipt 顶层及嵌套 JSON 使用 exact schema，拒绝 extra/missing、
  duplicate key、bool/int type alias 与零 digest；
- config/plan/source/runner/namespace/runtime fingerprint 形成同一 store
  lineage；
- runtime fingerprint 精确记录 Python、NumPy、SciPy、psutil、platform、
  seed namespace 与四个值为 `"1"` 的 BLAS/OMP 线程变量；
- receipt commit 前和消费时均验证 lineage，不允许协调重写后重哈希。

### 2. Resource profile 与正式投影

V4 的 5-profile/61,440-row 抽样被替换为 8 个 full-denominator receipts：

- formal LPT 峰值四 cell：plan index `478/480/482/484`，均为
  c44 backend-A fault，`4608×12`；
- 代表四 cell：plan index `388/389/403/507`，覆盖 shared A/B、
  logical-B 与 probe-B；
- 总计 `227,328` raw rows、`15,360` RESET rows。

resource consumer 逐 cell 重算：

- object bytes 与 transient bytes；
- 全 518-cell top-4 inflight；
- 固定 worker 顺序下的 deterministic LPT loads/makespan；
- cutoff36/40/44 mapping anchors；
- joint-maxT 与 retained-density physicality；
- 8-receipt inventory finalize 的 measured wall，并保守投影到正式规模；
- formal artifact、inflight、analysis scratch 与保留磁盘的闭合算术。

### 3. Raw/NPY/对象全读审计

- 逐一读取全部 `227,328` CSV rows，验证 121-column header、row/trajectory/
  cell/backend identity、seed namespace、fault/density/raw/heldout index、
  action、label、cluster root、terminal 与全局 row-ID 唯一性；
- 全部数值字段检查 finite/domain；hash 字段只允许空或 64 位小写十六进制；
- held-out window hash 与 heldout NPY 对应行交叉验证；
- NPY 验证 role、shape、dtype、header offset、`offset+nbytes` 精确文件长度、
  分块 finite、probability/distance/hidden outcome/ack domain；
- 只允许预注册 shared/probe density alias，拒绝偶然同 hash 被当作 alias；
- live object tree 必须恰好等于 receipts 引用对象，staging 在 PASS 时必须为空。

### 4. 外部 wall、PID 与 attempt/heartbeat

- 连续 resource sampling 使用 sequence/previous-hash chain；
- formal-LPT 与 representative 两阶段各要求精确四 worker、PID 双射且两组不重叠；
- 每个 worker wall 不能超过外部 stage window；stage window 不能被四个协调缩短的
  worker 自报值绕过；
- heartbeat exact schema 绑定 owner token、PID、process creation time、
  monotonic/wall clock、period、stage 与完成数；
- attempt 必须为同 run 的 exact `START -> PASS`，且 START owner 与最终
  heartbeat 一致；report、start/pass witness 和 ledger 都按 live bytes 绑定。

### 5. Preformal V2 seal 与正式消费

- preformal seal 升为 `PHASE9-POWERED-TWIN-PREFORMAL-SEAL-V2`；
- seal 精确绑定 contract、resource report、attempt ledger、start/pass witness、
  focused validation 六项；
- resource consumer 会重放 source/config/plan、raw、NPY、inventory、
  sampling、heartbeat、statistics、physicality、projection、decision、no-copy
  与 attempt chain；
- focused validation 包含 source/resource/consumption/attempt/Python/platform/
  stdout/stderr hash，禁止跨 resource run 复用；
- qualification 与 independent verifier 只接受 V2，明确拒绝 V1 downgrade、
  extra schema、未知 binding 或 claim 非 null。

## 产物

源码与测试：

- `cnn_fpga/benchmark/phase9_immutable_object_store.py`
- `cnn_fpga/benchmark/phase9_powered_twin_contract.py`
- `cnn_fpga/benchmark/phase9_powered_twin_plan.py`
- `cnn_fpga/benchmark/phase9_powered_twin_preflight.py`
- `cnn_fpga/benchmark/phase9_powered_twin_preformal_audit.py`
- `cnn_fpga/benchmark/phase9_powered_twin_qualification.py`
- `cnn_fpga/benchmark/phase9_powered_twin_runtime.py`
- `cnn_fpga/benchmark/phase9_powered_twin_verifier.py`
- `tests/test_phase9_immutable_object_store.py`
- `tests/test_phase9_powered_twin_preflight.py`
- `tests/test_phase9_powered_twin_preformal_audit.py`
- `tests/test_phase9_powered_twin_qualification.py`

V5 immutable 产物：

- `docs/t_risk_20260728_04_powered_twin_contract_preflight_v5.json`
  - bytes：`5,927`
  - file SHA-256：
    `3ad6964bc5520e0b59263c811cbe79f74e8887dd2d41c57d3e0ec23a950aead4`
  - analysis SHA-256：
    `65d4230840f2f6d9d676ad39ff310f02880a44e926db661acdb528150ccf1196`
- `runs/t_risk_20260728_04_powered_twin_qualification_fresh1/plan_v5.json`
  - `219,492 B`
  - `8e009b67dbb5d4704a1466561e9a4aa115a88c140c70d4cb0c866ea28ed72b82`
- `runs/t_risk_20260728_04_powered_twin_qualification_fresh1/seed_registry_v5.json`
  - `1,691 B`
  - `542303f59c515223748238b5cdc8878936b1631416acf0637bd7a18f1267c5b6`
- `runs/t_risk_20260728_04_powered_twin_qualification_fresh1/historical_seed_scan_v5.json`
  - `28,622 B`
  - `4941067d3fb82ecc096b4060339ff860ccf24b151ccd92e2ccf79de685e11386`

V5 contract 得到：

- `PASS_OUTCOME_FREE_CONTRACT_PREFLIGHT`
- `518` cells
- `2,085,888` rows
- `482,304` primary densities
- `11/11` contract gates
- `7/7` parent semantic checks
- `formal_outcomes_accessed=false`
- `scientific_execution_released=false`
- `qualified_claim=null`
- 九个 claim boundary 字段全 `null`

## 验证方式和结果

### 冻结测试

在正式 bootstrap 同构环境中设置：

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

然后对 9 个冻结测试文件执行：

```text
198 tests / 0 failures / 0 errors / 1 skipped / 245.347 s
```

测试覆盖包括 coordinated receipt/report/seal mutation、extra/missing schema、
duplicate key、零 hash、type alias、NPY trailing bytes/NaN、row identity/seed、
PID/stage wall 协调缩短、LPT/profile shrink、owner/heartbeat/attempt 漂移、
focused validation 复用和 V1 seal downgrade。

另一次故意未设置四线程变量的相同运行得到 8 个 production receipt
fail-closed；严格 consumer 正确拒绝 runtime fingerprint 漂移。随后未放宽
consumer，而是用正式 process-entry 环境完成上述 198 项通过。

### 静态与幂等验证

- 8 个相关 Python 文件 `py_compile`：PASS
- `git diff --check`：PASS
- TODO/FIXME/demo/placeholder 空实现扫描：未发现简化实现；命中项仅为异常类、
  queue-empty polling 与零权重 RESET 缺失分支
- V5 materializer 第二次执行：
  4 个文件 bytes、SHA-256 和 mtime 全部不变

## 风险复核

- 新增 `R-N198`：resource PASS 被浅层 profile/hash/自报 wall 协调伪造的风险；
  当前保持 `Open / Critical / Immediate`，必须由 fresh V5 resource PASS 和
  live consumer replay 才能降级。
- `R-N193` 仍为 `Open / High / Immediate`：真实 wall/disk/RSS 尚未由新
  resource 事务终态证明。
- 旧 V4 是基础设施/证据链受控失败，不是科学 NO-GO；详情见
  `T-RISK-20260728-04_resource_preflight_abort_full_v4_20260730_080203.md`。
- 不插入新 task：V5 修复属于当前 T04 的 prerequisite consumer 加固，拆分会使
  同一 preregistration 的 source/report/seal lineage 断裂。

## 对论文 claim 的影响

该阶段只建立 outcome-free contract 和 resource/preformal 可消费性，不产生
twin、LER、lifetime、break-even、hardware measured 或外部排名证据。
`official_puviani_exact`、`puviani_nmf_surpass` 及全部 SOTA/rank 字段继续
保持 `null`。

## 任务板同步

- `docs/new_task_board.md`：T04 保持 `In Progress`，新增 V4
  `evidence-chain repair` 活动记录；
- `docs/new_risks.md`：新增 R-N198 与“不插入新 task”判断；
- `README.md`：增加 T04 V4 失败证据与 V5 resource/preformal 入口；
- 下一步：先提交并推送 V5 preregistration，再以全新 run ID 启动唯一
  full-resource preflight；PASS 后才允许创建 V2 preformal seal 和正式执行。

## 2026-08-30 V5c 终态补记

V5c `full_v5c_20260827_220849` 首次完成全部 8 个 resource profiles：

- 8/8 receipts、227,328 observed/expected rows、15,360 RESET/sidecar；
- 78 个唯一对象、10,860,153,597 B；staging 为空；
- 37,920 点连续采样、最大 4 children、peak RSS 5,386,903,552 B；
- no-copy inventory、逐row/NPY、seed firewall 与 claim-null 均保持；
- attempt ledger 终态为
  `START_RESOURCE_PREFLIGHT -> FAIL_RESOURCE_PREFLIGHT`，失败门仅为 `wall`。

独立使用冻结 plan、8 个 receipt commit、stage 采样和未修改
`stratified_projection` 重建的正式总 wall 约为 `4,293,112.1 s`，超过
`1,209,600 s` 上限约 `3.549x`。因此本 task 改为
`Blocked (resource wall)`；同合同换 run ID 重跑会确定性再失败，不得启动。

旧 failure report 没有持久化 gate 前已经计算的 projection/decision。本轮只为
未来失败新增 `completed_stage_evidence`，early-worker/late-wall 聚焦回归
`25/25` 通过；没有回写 V5c 或改变科学/资源合同。按 R-N199 插入
`T-RISK-20260830-01`，只允许不缩分母、不改 seed/estimand/margin、
不放宽 14-day 门的 exact-output 可行性修复。详见：

`docs/new_tasks/T-RISK-20260728-04_resource_preflight_wall_fail_full_v5c_20260827_220849.md`。

全部 official/Puviani/SOTA、LER、lifetime、physical、hardware、rank 与 twin
qualification 字段仍为 `null`。
