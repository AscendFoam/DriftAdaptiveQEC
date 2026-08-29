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
- early-worker 与 late-wall fail-closed 回归均覆盖；
- 新增只读取证工具
  `cnn_fpga/benchmark/phase9_powered_twin_resource_forensics.py`。它逐条验证
  37,920 点 sample hash chain、terminal heartbeat、failure/receipt content hash、
  all-null claim boundary 和 frozen config，再用 process-creation epoch、stage
  monotonic time 与 receipt mtime 重建 worker wall；不会读取 formal seed/outcome，
  也不会写入 V5c；
- 完整 518-cell 重放得到 projection SHA-256
  `61e17fba58d65c34e9c21be44eb709e52f70f610ac3aec385b5c7ea6a90cefae`：
  raw four-worker LPT=`4,273,756.735 s`，statistics=`258.953 s`，
  inventory projection=`19,095.164 s`，保守 positive physicality floor=`1 s`，
  total=`4,293,111.851 s`、14-day ratio=`3.549200x`；historical worker
  sub-timing 因旧 failure schema 未持久化而不能声称 bit-exact 恢复，重建把
  spawn latency 纳入 wall，方向只会更保守；
- 逐 role artifact 投影为 `129,165,799,929 B`。这纠正了旧手工汇总
  `135,082,918,009 B` 未扣除 shared/probe 显式 alias 的误差，但不改变
  artifact PASS / wall FAIL 结论；
- per-layer worker-wall attribution：fault=`2,315,946.01 s`、
  logical=`7,829,043.57 s`、probe=`626,621.59 s`、
  shared=`6,319,312.14 s`；最高的 c44 fault 8 cells 均约
  `123,281.24 s/cell`，其中 compound 每 cell 投影约 `5.926 GB`；
- ideal、零 contention 的 deterministic LPT 曲线证明：4/8/12/14 workers
  分别约 49.69/24.98/16.74/14.38 days，至少 15 workers 才理论上达到
  13.46 days。该数字只是并发下界，不授权改 scheduler；主机仅 16 physical
  cores，且实测单 compound child peak=`5,364,723,712 B`，所以 15-worker
  naive 并发尚无 RSS/CPU 证据；
- 新增 6 个 forensic/LPT 单元测试，和原 25 个 preflight 测试合计
  `31/31` 通过；
- 尚未修改 scientific kernel 或启动新 full run。

## 当前判断与下一步

- 冻结 four-worker 合同仍是确定性的 resource NO-GO；同合同新 run ID 不会修复。
- 仅加并发无法作为合规修复：15-worker 理想下界虽过 wall，但当前 RSS 与
  CPU contention 未证明，且改变 `max_workers=4` 必须另立 outcome-blind
  preregistered child contract。
- 下一步优先审计 backend B 的 exact-output 热点：constant-Hamiltonian split
  内重复 `expm`、固定 duration 的 Kraus/dephasing operator 重建，以及 validator
  中不参与门判定的 purity matrix multiply；任何缓存/移除都必须以同一输入的
  row/NPY byte equality、异常路径等价和 full projection speedup 证明后才可释放。

## claim 边界

本任务是 outcome-free infrastructure/resource repair。所有 official/Puviani/SOTA、
LER、lifetime、physical、hardware、rank 与 twin qualification 字段保持 `null`。
