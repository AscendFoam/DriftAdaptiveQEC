# T7.2.2 Methods evidence-state contract

- verdict：`PASS_EVIDENCE_STATE_BOUNDED_METHODS`
- gates：`18/18`
- semantic mutations：`18/18`
- method rows：`18`
- V5：`20` dropped tasks，`0` downstream outputs
- board：`42` measured fields，nonnull=`0`

| component | state | online | offline truth | boundary |
| --- | --- | --- | --- | --- |
| `protocol_aligned_simulator` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `OFFLINE_TRUTH_ONLY_SCORING` | Not a calibrated cavity--transmon digital twin |
| `observation_truth_split` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `OFFLINE_TRUTH_ONLY_SCORING` | Truth cannot enter an adapter, posterior, image, or action |
| `unified_execution_contract` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `NONE` | A common cap is not equal measured cost |
| `v4_hmm` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `OFFLINE_TRUTH_ONLY_SCORING` | Must not be renamed IMM, BOCPD, or activation prediction |
| `v4_window_ewma` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `OFFLINE_TRUTH_ONLY_SCORING` | Tail behavior establishes EWMA-relative non-inferiority, not improvement |
| `v4_typed_bank` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `NONE` | V4 transactions are not V5 typed-policy residency evidence |
| `matched_baselines` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `OFFLINE_TRUTH_ONLY_SCORING` | Static joint MAP remains software-only for current RTL |
| `hidden_oracle` | `IMPLEMENTED_EVALUATED` | `NONE` | `OFFLINE_TRUTH_ONLY_SCORING` | Nondeployable upper bound; excluded from matched cost/rank |
| `v4_statistics` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `OFFLINE_TRUTH_ONLY_SCORING` | Prior-informed V1--V4 history must remain disclosed |
| `v4_integer_cxxrtl` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `NONE` | Sampled equivalence and mutation coverage are not exhaustive formal proof |
| `v4_post_route` | `IMPLEMENTED_EVALUATED` | `ONLINE_OBSERVED_ONLY` | `NONE` | Estimate only; no vendor signoff, transport, or board timing |
| `v5_headroom` | `DIAGNOSTIC_ONLY_EXECUTED` | `ONLINE_OBSERVED_ONLY` | `OFFLINE_TRUTH_ONLY_SCORING` | Negative entry audit is not a V5 formal performance result |
| `v5_four_split` | `CONDITIONALLY_REGISTERED_STOPPED` | `NONE_NOT_RUN` | `NONE_NOT_RUN` | No split manifest, power plan, or untouched V5 formal data exists |
| `v5_posterior` | `CONDITIONALLY_REGISTERED_STOPPED` | `NONE_NOT_RUN` | `NONE_NOT_RUN` | Dropped before implementation or calibration |
| `v5_map_risk` | `CONDITIONALLY_REGISTERED_STOPPED` | `NONE_NOT_RUN` | `NONE_NOT_RUN` | No compiler image, risk threshold, or quantized action exists |
| `v5_typed_policy` | `CONDITIONALLY_REGISTERED_STOPPED` | `NONE_NOT_RUN` | `NONE_NOT_RUN` | No V5 resident-bank transaction or event action was executed |
| `v5_qualification` | `CONDITIONALLY_REGISTERED_STOPPED` | `NONE_NOT_RUN` | `NONE_NOT_RUN` | No V5 golden, proof, trace, netlist, resource, or timing result |
| `physical_board` | `FUTURE_PHYSICAL_WORK_BLOCKED` | `PHYSICAL_INPUT_PENDING` | `PHYSICAL_SCORING_PENDING` | All 42 measured fields remain null |

V4 的 simulator、HMM、Window/EWMA、bank、统计、integer/CXXRTL 与 P&R estimate 均按各自证据层描述。
V5 只有 causal/action-space entry diagnostic 被执行；四分割、IMM/BOCPD、activation prediction、
posterior-predictive compiler、LER/CVaR gate、typed V5 bank、formal/CXXRTL/P&R 均在入口 NO-GO 后停止。
真实板卡流程是 future work，不能由 UART candidate、clock model 或 P&R 数值替代。
