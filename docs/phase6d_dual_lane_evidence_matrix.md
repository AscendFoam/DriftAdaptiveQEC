# T6.26.3 双证据 lane claim-evidence / 主图合同

## 结论

**`PASS_NONTRANSFERABLE_DUAL_LANE_EVIDENCE_AND_FIGURE_CONTRACT`**。论文必须保留两个并列但不可互相补门的 primary lane：

- multimode software：T6.18.3 的 opened task-local positive 结果为 `p_L=0.172261` vs static-Euclidean `0.236929`；但 Phase-6D v1 对 strongest retained static-mixture exact MLD 的 point/LCB 均为 `0.0%/0.0%`，因此 frozen-benchmark SOTA 未建立，pilot/formal 未访问。
- single-mode RTL：六周期、II=1、百万周期全公开向量零 mismatch、atomic/fail-closed property 与三 seed 27 MHz P&R 已在同一 exact top 上闭合；Fmax min/median/max=`36.794/37.736/37.869` MHz。
- CNN/student：T6.26.1--T6.26.2 Dropped，主证据中 absent；只能作为未来新路线的可替换近似，不是第三 primary lane。

## 原子 claim matrix

| Claim | Lane | State | Boundary |
| --- | --- | --- | --- |
| `MM_OPENED_TASK_LOCAL_GAIN` | `MULTIMODE_SOFTWARE_ALGORITHM` | `RESULTS_ONLY_NONRANKING` | This is task-local opened development evidence; it is not the Phase-6D strongest-baseline or frozen-benchmark SOTA result. |
| `MM_V1_CAUSAL_HEADROOM_NO_GO` | `MULTIMODE_SOFTWARE_ALGORITHM` | `MANDATORY_NEGATIVE` | All 13 development families and both retained baseline candidates remain visible; pilot and formal splits were not accessed. |
| `MM_FROZEN_BENCHMARK_SOTA_BLOCKED` | `MULTIMODE_SOFTWARE_ALGORITHM` | `BLOCKED_NOT_RUN` | T6.18.3 remains a task-local positive comparator result and cannot replace the unrun strongest-baseline promotion gate. |
| `RTL_DETERMINISTIC_SIX_CYCLE_II1` | `SINGLE_MODE_DETERMINISTIC_RTL` | `CURRENT_RESTRICTED` | Cycles are RTL/CXXRTL evidence; nanoseconds are a post-route clock model without transport, CDC, pins or physical jitter. |
| `RTL_ATOMIC_FAIL_CLOSED` | `SINGLE_MODE_DETERMINISTIC_RTL` | `CURRENT_RESTRICTED` | The proof scope is the stated two-state RTL model and does not establish physical CDC, metastability or unbounded liveness. |
| `RTL_POST_ROUTE_ESTIMATE` | `SINGLE_MODE_DETERMINISTIC_RTL` | `CURRENT_RESTRICTED` | All three critical paths terminate in the observability fold, so Fmax is a conservative qualification-harness estimate rather than bare-core or board source-to-action speed. |
| `RTL_BOARD_MEASUREMENT_BLOCKED` | `SINGLE_MODE_DETERMINISTIC_RTL` | `BLOCKED_NOT_RUN` | No pre-board clock model, post-route estimate or analytic power sensitivity may fill a measured field. |
| `RTL_SPEED_ADVANTAGE_PROHIBITED` | `SINGLE_MODE_DETERMINISTIC_RTL` | `PROHIBITED_POSITIVE` | The hardware contribution is deterministic, atomic and fail-closed pre-board architecture, not a cross-paper speed rank. |
| `LEARNING_APPROXIMATION_DROPPED` | `LEARNED_APPROXIMATION_EXTENSION` | `DROPPED_ABSENT` | Legacy CNN evidence remains an ablation only and cannot rescue either primary lane. |
| `DUAL_LANE_NONTRANSFERABILITY` | `META_CONTRACT` | `META_BOUNDARY` | The learning extension is dependent and absent, not a third primary evidence lane. |

## 主图合同

Panel A 只显示 multimode LER/tail/compute/evidence state，并同时显示 task-local positive 与 strongest-baseline NO-GO。Panel B 只显示 single-mode cycles/property/CXXRTL/post-route/resource/board-null。Learning 只作 Dropped/absent inset。禁止 global weighted score、跨 lane 箭头、用一条 lane 满足另一条 lane 的 gate。

三条 post-route critical path 都终止于 observability fold，因此 36.794 MHz 是 whole-harness conservative estimate，不是 bare-core 或真板速度。所有 board-measured 字段保持 null。
