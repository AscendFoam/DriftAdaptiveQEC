# T7.2.3 Results evidence contract

- verdict：`PASS_RESULTS_COMPLETE_NEGATIVE_AND_SECONDARY_BOUNDARIES`
- gates：`20/20`
- semantic mutations：`20/20`
- result rows：`27`
- V5：`20` dropped tasks，`0` downstream outputs
- Phase 6C：eligible secondary results=`5`，literature in Results=`0`
- board：`42` measured fields，nonnull=`0`

| result | state | grade | polarity | boundary |
| --- | --- | --- | --- | --- |
| `v4_locked_ewma` | `PRIMARY_RESTRICTED` | `PROJECT_NATIVE_MATCHED` | `POSITIVE` | Locked EWMA aggregate only; periodic is the sole Holm-confirmed family |
| `v4_static_ordering` | `MANDATORY_NEGATIVE` | `NEGATIVE` | `NEGATIVE` | Static joint MAP has lower average and calibration worst-window LER |
| `v4_window_ordering` | `MANDATORY_NEGATIVE` | `NEGATIVE` | `NEGATIVE` | Window MAP is the strongest deployable smooth comparator |
| `v4_oracle_gap` | `MANDATORY_NEGATIVE` | `NEGATIVE` | `NEGATIVE` | Static-to-oracle gap closure is negative |
| `v4_tail_noninferiority` | `MANDATORY_NEGATIVE` | `PROJECT_NATIVE_MATCHED` | `NEUTRAL` | Five families are exactly equal to EWMA; no broad tail improvement |
| `v4_fallback_cost` | `MANDATORY_NEGATIVE` | `PROJECT_NATIVE_MATCHED` | `NEGATIVE` | High fallback and unnecessary-fallback rates remain visible |
| `v4_false_updates` | `MANDATORY_NEGATIVE` | `PROJECT_NATIVE_MATCHED` | `NEGATIVE` | All family-specific false-update counts remain visible |
| `v4_failed_policy_families` | `MANDATORY_NEGATIVE` | `NEGATIVE` | `NEGATIVE` | Static-switch and freeze-all each failed all 38 safe tuples |
| `external_bocd_budget` | `MANDATORY_NEGATIVE` | `PROJECT_NATIVE_MATCHED` | `MIXED` | Paired LER outcome is inseparable from the 13,004.1-us budget failure |
| `v4_cxxrtl` | `PRIMARY_RESTRICTED` | `CXXRTL_PREBOARD` | `POSITIVE` | One million cycles and failure branches; no board inference |
| `v4_commit_rollback_attempts` | `PRIMARY_RESTRICTED` | `CXXRTL_PREBOARD` | `NEUTRAL` | All 75 commit and 25 rollback attempts stay in the denominator |
| `v4_post_route` | `PRIMARY_RESTRICTED` | `POST_ROUTE_ESTIMATE` | `POSITIVE` | Six-cycle II=1 and three-seed P&R are estimates |
| `physical_board_null` | `NULL_OR_CONTEXTUAL` | `BLOCKED` | `NULL` | All 42 measured fields remain null |
| `v5_causal_selector` | `DIAGNOSTIC_STOP` | `DIAGNOSTIC_ONLY` | `NEGATIVE` | Strict causal selector headroom is negative |
| `v5_action_headroom` | `DIAGNOSTIC_STOP` | `DIAGNOSTIC_ONLY` | `NEGATIVE` | Expanded action-space headroom is 0.02549%, below 12% |
| `v5_downstream_absence` | `NULL_OR_CONTEXTUAL` | `BLOCKED` | `NULL` | No untouched LER/tail, quantized, formal, CXXRTL, or P&R output exists |
| `p6c_single_cpd` | `SECONDARY_ELIGIBLE` | `PROJECT_NATIVE_MATCHED` | `EQUIVALENCE` | CI equals Euclidean CPD only in the frozen square/isotropic domain |
| `p6c_noh_cnot` | `SECONDARY_ELIGIBLE` | `PROJECT_NATIVE_MATCHED` | `POSITIVE` | ML lowers failure relative to CI on the matched two-GKP CNOT task |
| `p6c_official_cpd` | `SECONDARY_ELIGIBLE` | `OFFICIAL_CODE_REPRODUCTION` | `POSITIVE` | Official data plus partial small-distance CPD reproduction |
| `p6c_multimode_adaptive` | `SECONDARY_ELIGIBLE` | `PROJECT_NATIVE_MATCHED` | `POSITIVE` | Observed-only adaptive weighting improves aggregate and tail metrics |
| `p6c_aqec_project` | `SECONDARY_ELIGIBLE` | `PROJECT_NATIVE_MATCHED` | `NEGATIVE` | Project-native active/autonomous control fails to beat idle; official protocol blocked |
| `p6c_learned_zero` | `MANDATORY_NEGATIVE` | `INELIGIBLE` | `NULL` | Zero of 16 candidate families is same-task eligible |
| `p6c_gqf_blocked` | `NULL_OR_CONTEXTUAL` | `BLOCKED` | `NULL` | Zero of 15 exact checks and all 13 matched metrics null |
| `p6c_external_fpga` | `NULL_OR_CONTEXTUAL` | `LITERATURE_ONLY` | `NULL` | 18 normalized rows but zero exact same-task comparator |
| `legacy_cnn` | `HISTORICAL_EXTENSION` | `PROJECT_NATIVE_HISTORICAL` | `POSITIVE` | Four-scenario frozen software-HIL result is outside the Route-A ranking |
| `teacher_student_retention` | `HISTORICAL_EXTENSION` | `PROJECT_NATIVE_SIMULATION` | `POSITIVE` | Finite-model gain retention and compression only |
| `teacher_student_selection` | `MANDATORY_NEGATIVE` | `SELECTION_AUDIT` | `NEGATIVE` | All restarts, cap hits, and test-hindsight reversals remain disclosed |

V4 的 locked-EWMA 正对比与 static/Window/oracle-gap/tail/fallback 反证并列；
V5 只报告已执行的 entry diagnostic 和 downstream absence；
Phase 6C 只有 official-code reproduction 或 project-native matched 行可进入明确标注的 secondary Results；
文献值、blocked/null、P&R estimate 和 42 个板测 null 字段均不能生成主排名或实测结论。
