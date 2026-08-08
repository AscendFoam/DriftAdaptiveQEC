# Route-A integrated fixed-point / RTL 长序列资格验证

## 结论

T6.7.3 已在 **board-independent** 边界内通过。最终版本把冻结的 T6.7.1/T6.7.2 observed formal trajectory、V4 HMM/event posterior、prequential Window/EWMA preference、Route-A integer policy overlay 与 T6.2.2 production core 接入同一 trace：

- 4 个 smooth family + 6 个 fault family；
- 每 family `100,000 cycles`，聚合 `1,000,000 cycles`；
- Python integer golden 与 CXXRTL 对原 core 和 Route-A 可见字逐 cycle 比较；
- `0` bit mismatch、`0` undefined action、`0` route/core CRC word error、`0` silent overflow；
- 19/19 evidence gates、130/130 CXXRTL comparator mutations、12/12 semantic mutations 通过。

本结论不是板测、bitstream、物理 transport、真实 deadline、功耗或 measured latency 结论。

## 集成边界

### Slow loop

HMM/event model 仍在软件中执行，每 32 decisions 输出一次四态 posterior，顺序固定为：

1. normal；
2. smooth；
3. calibration shift；
4. burst。

posterior 用 largest-remainder 量化为四个 `uint8`，每行严格满足和为 255。FPGA 边界只接收 observed-only posterior、OOD、prequential expert preference 与 integrity/event 输入；logical truth/label 不进入 deployable trace。

### Fast path

新增 `route_a_policy_overlay.sv`，与 T6.2.2 `gkp_fast_path_core.sv` 组成 qualification top。整数合同为：

- tail enter：`p_calibration + p_burst >= 230`；
- tail exit：`p_calibration + p_burst <= 51`；
- adaptive ready：`p_normal + p_smooth >= 230` 且 `255-max(p) < 64`；
- OOD event：`ood_code > 192`；
- enter/recovery hysteresis：`2 / 8` posterior updates；
- bank0=EWMA，bank1=Window；
- integrity > leakage/reset > tail > uncertain > OPEN；
- auto commit 使用 `new_version=active_version+1`；
- host commit 只有在 OPEN、目标等于 selected bank 且无 policy pending 时才允许，否则显式 blocked；
- action metadata 与 production `out_word` 在同一 6-cycle source-to-action 边沿对齐。

输出并逐 bit 比较：

- 原 core `out_word[117:0]`、`state_word[231:0]`、MAP/active bank/version debug；
- Route-A `action_word[79:0]`、`state_word[95:0]`、`version_word[63:0]`；
- action/reason/selected bank/commit pending/host blocked debug。

## Unified replay provenance

- 每个 family 使用两条 frozen formal observed trajectory，共 20 条；
- 每条原始 trajectory 为 53,248 decision cycles；第二段在 cycle 53,248 显式 reset core/policy；
- 20/20 formal cache hits、0 misses；
- 每段记录 cell、seed、observed-trace hash、float posterior hash 与 uint8 posterior hash；
- aggregate `995,802 / 1,000,000 = 99.5802%` cycles 保持真实 unified replay；
- `4,198 / 1,000,000 = 0.4198%` cycles 是独立标记的 threshold/race/invalid-simplex directed safety vectors。

directed vector 不用于估计物理场景频率或 LER，只用于分支资格验证。

## 最终覆盖结果

### Policy / bank

- posterior updates：31,250；
- real prequential router boundaries：645；
- action cycles：OPEN 631,249；tail 267,780；uncertain 16,096；leakage/reset 850；integrity rollback 84,025；
- selected expert：EWMA 927,008 cycles；Window 72,992 cycles；
- tail latch entries/recoveries：47 / 42；未恢复项对应持续 tail 或 segment 末尾，不被伪造为已恢复；
- leakage entries：28；integrity entries：59。

### Commit / rollback race

- auto commit requests：99；
- ack / deliberate untrusted rejection：85 / 14；
- host attempts：75，全部被 policy admission gate 阻止；
- host/auto same-cycle collisions：14；
- config/commit race vectors：25；
- rollback attempts：25。

### Core fault bits

实际覆盖 invalid observation、OOD、CRC、untrusted bank、age、deadline、unexpected reset ack 与 leakage。production-contract 中结构不可达的 version-overtake/MAP-missing bits继续保持 0，不用伪造故障状态补覆盖。

### Abstract FIFO/receiver

- source/delivered packets：90,000 / 88,477；
- pause/backpressure：1,458 / 1,458 cycles；
- overflow：1,451，全部 accounted；
- drop/duplicate/reorder：18 / 18 / 18；
- sequence/deadline faults：72 / 126；
- explicit fault markers：1,541；
- max FIFO depth：8；
- final pending FIFO/markers：0 / 0；
- silent overflow：0。

## 反简化审计与失败历史

本 task 保留三轮完整结果演化：

1. 第一版 directed posterior 通过 bit-exact，但深审认为不能冒充 unified runner replay；
2. 第二版改为 99% 以上真实 formal posterior，仍零 mismatch，但 natural readout/reset posterior 从未选 Window，导致 commit/rollback race gate 失败；
3. 第三版只增加与 posterior 分布无关的最小安全仲裁向量，短轨先证明 race 可达，再完整重跑百万周期并通过。

通过标准没有因失败而放宽；旧 FAIL 原因在任务记录和最终机器报告中保留。

## 关键限制

1. HMM 本体未综合到 FPGA；当前是 software slow-loop posterior 到 synthesizable fast policy 的接口资格验证。
2. Yosys `proc/check/stat` 为结构检查，不是目标器件 P&R、LUT/FF/BRAM/DSP 或功耗报告；这些属于 T6.9.1。
3. CXXRTL host runtime 约 1,616 s，不是 FPGA latency。
4. `smooth_bank_posterior_min=77/255` 在当前 OPEN 条件下事实上是非绑定条件：OPEN 已要求最大 posterior `>191/255`，因此 Window 只会在 smooth-dominant posterior 下进入。它不能单独作为“0.30 smooth threshold 提供额外安全性”的证据。
5. 本 task 只证明 correctness/fail-closed/branch coverage；不改变 T6.7.1 中 static/Window LER 更低、T6.7.2 中 tail 主要等价 EWMA 的负面性能结论。

## 产物

- `cnn_fpga/runtime/route_a_fixed_policy_reference.py`
- `cnn_fpga/rtl/route_a_policy_overlay.sv`
- `cnn_fpga/rtl/route_a_integrated_qualification_top.sv`
- `cnn_fpga/rtl/route_a_integrated_cxxrtl_driver.cc`
- `cnn_fpga/benchmark/route_a_integrated_rtl_qualification.py`
- `tests/test_route_a_integrated_rtl_qualification.py`
- `docs/t6_7_3_route_a_integrated_rtl_qualification.json`
- `docs/t6_7_3_route_a_integrated_rtl_source_data.csv`
- `build/t6_7_3_route_a_integrated_rtl/route_a_integrated_trace.bin`（131 MB raw per-cycle Source Data）

## 验证

- Yosys：`Found and reported 0 problems`；
- CXXRTL：10×100,000 rows，全部 actual/expected digest 相等；
- focused + T6.7 adjacent：23 passed；
- Source Data：10 rows，每 family 指向完整 raw trace/digest；
- 九个 runner/RTL/model source hash 现场重算一致。
