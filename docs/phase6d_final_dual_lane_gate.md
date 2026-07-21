# T6.26.4 Phase 6D 最终双 lane gate

## 终态

**`GO_RTL_ONLY`**。

- Multimode software lane：`NO_GO`。T6.24.5=`Dropped`，strongest baseline=`static_mixture_exact_mld`，baseline/proposed `p_L=0.111979/0.111979`，relative point/LCB=`0.0%/0.0%`，pilot/formal 未访问。T6.18.3 只保留 opened task-local context。
- Single-mode RTL lane：`GO`。exact top 的 property、million-cycle CXXRTL 与三 seed P&R 均通过；6-cycle、II=1，最低 whole-harness Fmax=`36.794` MHz。该 Fmax 受 observability fold 影响，不是 bare-core/board speed。
- Learning extension：`DROPPED_ABLATION_ONLY`，不参与真值表也不改变 overall verdict。

没有加权总分或跨 lane 补门。T6.9.2 继续 Blocked，board-measured、fastest、multimode-in-RTL 与 frozen-benchmark multimode SOTA 均关闭。

## Claim 移交

| Claim | Final disposition | Placement | Final wording |
| --- | --- | --- | --- |
| `MM_OPENED_TASK_LOCAL_GAIN` | `RETAIN_CONTEXT_ONLY` | Results, Discussion, Supplement | Retain only as opened task-local multimode context alongside the mandatory strongest-baseline NO-GO. |
| `MM_V1_CAUSAL_HEADROOM_NO_GO` | `MANDATORY_NEGATIVE` | Results, Discussion, Supplement | Mandatory negative: Phase-6D v1 did not enter implementation or formal because usable causal headroom over the strongest retained static baseline was zero. |
| `MM_FROZEN_BENCHMARK_SOTA_BLOCKED` | `BLOCKED` | AbstractBoundary, Results, Limitations, Supplement | Frozen-benchmark multimode SOTA is not established. |
| `RTL_DETERMINISTIC_SIX_CYCLE_II1` | `PROMOTED_RESTRICTED` | Abstract, Methods, Results, Conclusion, Supplement | The exact single-mode converged RTL supports a six-cycle, II=1 pre-board architecture and passed one-million-cycle full-vector CXXRTL qualification. |
| `RTL_ATOMIC_FAIL_CLOSED` | `PROMOTED_RESTRICTED` | Abstract, Methods, Results, Conclusion, Supplement | The exact converged production top passes the stated pre-board atomic versioned-bank and fail-closed property contract. |
| `RTL_POST_ROUTE_ESTIMATE` | `PROMOTED_RESTRICTED` | Methods, Results, Supplement | The exact qualified top passes three-seed 27 MHz open-source P&R; reported Fmax/resources are whole-harness pre-board estimates. |
| `RTL_BOARD_MEASUREMENT_BLOCKED` | `BLOCKED` | Methods, Results, Limitations, Supplement | All physical-board fields remain null and unavailable. |
| `RTL_SPEED_ADVANTAGE_PROHIBITED` | `PROHIBITED_POSITIVE` | RelatedWork, Limitations, Supplement | No FPGA speed advantage is claimed. |
| `LEARNING_APPROXIMATION_DROPPED` | `DROPPED_ABLATION_ONLY` | Methods, Results, Supplement | CNN/student is absent from the primary Phase-6D result and retained only as a dropped/ablation status. |
| `DUAL_LANE_NONTRANSFERABILITY` | `MANDATORY_META_BOUNDARY` | TitleBoundary, Abstract, Methods, Discussion, Conclusion | The two evidence lanes are parallel and connected by a contract pattern, not by a common performance denominator or current decoder deployment. |

## Phase 7

下一顺序任务是 T7.1.5：保留 T7.1.1--T7.1.4 historical snapshot，新增 Phase6D delta。T7.2.6 等待该 delta；T7.3.8 写 strongest-baseline negative answer，T7.3.9 写 contract bridge 而不是共同性能分母。
