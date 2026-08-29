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
- 已完成第一轮 backend-B exact-semantics 热点修复：pure-loss Kraus 利用解析
  superdiagonal 结构在 `(Fock, ancilla, Fock, ancilla)` tensor 上执行，qutrit
  amplitude channel 使用等价 diagonal/jump block；固定 duration 的 loss/dephasing/
  reset operator、constant Hamiltonian half-step、base Hamiltonian 与 joint ladder 只构造
  一次。没有删 split step、Kraus branch、noise channel、cutoff、density PSD 检查或
  formal row；
- 新增 `tests/test_phase9_backend_b_exact_output_optimization.py`：constant-H split
  对旧循环 byte-identical 且 `expm` 从8次降到1次，varying-H 仍逐 midpoint 执行；
  structured pure-loss/qutrit channel 对 dense Kraus reference 的 max-abs 差不超过
  `2e-16`，异常分支、只读缓存、Hamiltonian/reset 闭合均覆盖。cutoff44、8-step
  sense 同进程3次盲测中，旧dense median=`0.279241 s`、新structured
  median=`0.031850 s`，speedup=`8.767x`，该样本 output 的 max-abs 与 trace
  distance 都为0；这是 component microbenchmark，不外推为正式518-cell wall PASS；
- backend-B hash-bound qualification 已原子重建，analysis SHA-256 更新为
  `a302196cb98fad93d3d73c8abcbe3ac95430b79dfda87dbfb8293a1ac082c5aa`，
  13/13 live artifact checks 和 `205/205` focused tests通过；数值门、mutation、
  claim-null边界保持通过。旧V5c仍永久只读，尚未启动新 full run。
- fresh V6 `full_v6_20260830_043859` 在任何scientific row前 fail-closed：T03
  frozen loader 正确拒绝 optimized backend-B 与旧 binding `0b7f7e3f...` 不一致；
  0 receipt/staging/object、formal seed/outcome未访问、claim全null。直接默认替换
  旧 binding 的方案被安全审查拒绝且已完全撤回；V6永久只读，详见对应终态记录。

## 当前判断与下一步

- 冻结 four-worker 合同仍是确定性的 resource NO-GO；同合同新 run ID 不会修复。
- 仅加并发无法作为合规修复：15-worker 理想下界虽过 wall，但当前 RSS 与
  CPU contention 未证明，且改变 `max_workers=4` 必须另立 outcome-blind
  preregistered child contract。
- 下一步使用全新、outcome-blind 的 focused resource projection 对完整8类 profile
  重测该实现；只有 deterministic 518-cell LPT、RSS、artifact/disk、joint-maxT、
  physicality 和 inventory 全部门同时 PASS，才允许另一个全新 run ID。当前
  `8.767x` 只是 backend-B sense microbenchmark，不能替代完整投影或释放T04。
- 在此之前需要用户明确授权建立新的显式 byte-bound exact-equivalence child
  contract；旧T03 config/manifest/raw保持不变。未获授权不得默认注入replacement、
  放宽loader或启动V7。

## claim 边界

本任务是 outcome-free infrastructure/resource repair。所有 official/Puviani/SOTA、
LER、lifetime、physical、hardware、rank 与 twin qualification 字段保持 `null`。
