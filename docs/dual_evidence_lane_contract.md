# Phase 6D 双证据 lane 合同

> 本文档由 `T6.20.2` 的机器合同生成。两个 primary lane 并列但不互相补门；CNN/student 只是依赖性扩展。

## Lane 冻结

| Lane | 角色 | 对象 / action | 主指标 | 时间边界 | 部署状态 |
| --- | --- | --- | --- | --- | --- |
| `MULTIMODE_SOFTWARE_ALGORITHM` | PRIMARY_EVIDENCE_LANE | surface_square_gkp_multimode / logical-coset decision per correction round | per_round_p_L | software_source_to_decision, software_update_compute | SOFTWARE_ONLY_NOT_CURRENT_RTL |
| `SINGLE_MODE_DETERMINISTIC_RTL` | PRIMARY_EVIDENCE_LANE | single_mode_square_gkp_production_fast_path / bounded frame/event action with version and reason code | latency_cycles, initiation_interval_cycles, atomic_old_or_new, fail_closed_property | rtl_input_accept_to_action_valid, rtl_initiation_interval | ACTUAL_SINGLE_MODE_RTL_PREBOARD |
| `LEARNED_APPROXIMATION_EXTENSION` | OPTIONAL_DEPENDENT_EXTENSION | replaceable_cnn_or_student_approximation / posterior/LLR/coset-probability/action approximation | none | software_inference_only_unless_separately_integrated | OPTIONAL_NOT_AN_INDEPENDENT_PRIMARY_LANE |

## Claim 状态

| Claim | Lane | 状态 | 升级门 | 安全措辞 |
| --- | --- | --- | --- | --- |
| `C-MM-FROZEN-BENCHMARK-LER-SOTA` | `MULTIMODE_SOFTWARE_ALGORITHM` | CONDITIONAL_FUTURE | `T6.24.5` | On the frozen Phase-6D multimode benchmark, the proposed observed-only decoder passes the preregistered relative-LER gate against every eligible strongest deployable baseline. |
| `C-RTL-HISTORICAL-PREBOARD-IMPLEMENTATION` | `SINGLE_MODE_DETERMINISTIC_RTL` | CURRENT_RESTRICTED | `T6.2.2+T6.19.1` | The existing single-mode production RTL has a six-cycle, II=1 pre-board implementation supported by bit-accurate/CXXRTL replay and post-route estimates; this historical evidence does not yet constitute the Phase-6D property qualification. |
| `C-RTL-DETERMINISTIC-ATOMIC-FAIL-CLOSED` | `SINGLE_MODE_DETERMINISTIC_RTL` | CONDITIONAL_FUTURE | `T6.25.4` | After T6.25.4 passes, the single-mode production RTL may be claimed to have a six-cycle, II=1 pre-board fast path with atomic versioned-bank and fail-closed properties under the stated CXXRTL/property/P&R boundary. |
| `C-ML-OPTIONAL-APPROXIMATION` | `LEARNED_APPROXIMATION_EXTENSION` | OPTIONAL_FUTURE | `T6.26.2` | A matched-budget learned student may be retained as a replaceable approximation when it preserves the frozen classical decoder result and provides a separately demonstrated cost benefit. |
| `C-BOARD-MEASURED-PERFORMANCE` | `SINGLE_MODE_DETERMINISTIC_RTL` | BLOCKED_NULL | `T6.9.2` | Board-measured latency, deadline, resource-power and transport results remain unavailable until the physical-board protocol is executed. |

## Integration bridge

这些接口只复用 schema/事务合同，不把 multimode 软件方法自动提升为当前 RTL 实现。

- `IF-MM-POSTERIOR-TO-COSET-ACTION`: multimode observed-only posterior provider → multimode exact/approximate logical-coset decoder；SOFTWARE_ONLY。
- `IF-SLOW-PROPOSAL-TO-CANDIDATE-IMAGE`: host estimator or optional learned provider → schema-specific inactive-bank image adapter；CONTRACT_REUSE_ONLY_REQUIRES_SCHEMA_EQUIVALENCE。
- `IF-CANDIDATE-IMAGE-TO-INACTIVE-BANK`: single-mode production image validator → inactive A/B bank；SINGLE_MODE_RTL_ONLY。
- `IF-ATOMIC-COMMIT-TO-FAST-PATH`: versioned active-bank commit → six-cycle II=1 event/action datapath；SINGLE_MODE_RTL_ONLY。

## 禁止跨 lane 迁移

- `FT-MM-LER-TO-CURRENT-RTL` / `CROSS_LANE_IMPLEMENTATION_PROMOTION`：promote multimode LER to current RTL implementation。原因：different code family, action and precision。
- `FT-RTL-LATENCY-TO-MULTIMODE` / `CROSS_LANE_TIMING_PROMOTION`：attach six-cycle latency to multimode decoder。原因：multimode compute graph is not the RTL datapath。
- `FT-CNN-TO-ALGORITHM-SOTA` / `SURROGATE_TO_PRIMARY_PROMOTION`：use student agreement as the algorithm SOTA gate。原因：agreement cannot replace untouched LER comparison。
- `FT-CNN-TO-RTL-SAFETY` / `SURROGATE_TO_RTL_PROMOTION`：use model accuracy as RTL safety evidence。原因：software accuracy does not prove atomicity or fail-closed behavior。
- `FT-PREBOARD-TO-BOARD-MEASURED` / `EVIDENCE_LAYER_PROMOTION`：rename CXXRTL/P&R estimate as board measurement。原因：physical bitstream, transport and instrument evidence are absent。
- `FT-OPENED-DEVELOPMENT-TO-FORMAL` / `SPLIT_CONTAMINATION`：reuse T6.18.3 opened outcomes as Phase-6D formal。原因：development and untouched formal evidence must be disjoint。
- `FT-SOURCE-PRESENCE-TO-REPRODUCTION` / `PROVENANCE_PROMOTION`：treat pinned exact-MLD source as a reproduced baseline。原因：source audit is not execution or anchor agreement。
- `FT-TRUE-METRIC-CPD-TO-EXACT-ORACLE` / `ORACLE_HIERARCHY_COLLAPSE`：label true-metric CPD as exact decoding oracle。原因：metric knowledge does not perform logical-coset probability summation。
- `FT-CROSS-LANE-WEIGHTED-SCORE` / `GLOBAL_SCORE_PROHIBITED`：combine LER and latency/safety into one weighted rank。原因：primary metrics and task signatures are incommensurate。

## 当前证据边界

multimode LER SOTA 与 Phase-6D RTL property 主张仍是条件性未来 claim；现有 single-mode RTL 只能报告历史 bit-accurate/CXXRTL/P&R pre-board 证据；board-measured 字段保持 null。
