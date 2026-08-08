# T7.2.5 Supplementary 证据合同

- verdict：`PASS_SUPPLEMENT_COMPLETE_REPRODUCIBLE_AND_NONMIXING`
- gates：`24/24`
- semantic mutations：`24/24`
- evidence rows：`46`
- long RTL：`1,000,000` + `1,000,000` cycles
- Phase 6C atlas：`206` cells，global winner=`None`
- board measured fields：`0/42` non-null

| ID | 主题 | 状态 | 边界 |
| --- | --- | --- | --- |
| SUP-001 | folded_syndrome | `DEFINITION` | Half-open square-GKP residue with production tie rule |
| SUP-002 | logical_coset_map | `DEFINITION` | Periodic likelihood sum; MAP is not renamed CPD |
| SUP-003 | joint_correlated_map | `DEFINITION` | Two-dimensional software comparator differs from phase LUT |
| SUP-004 | pauli_ler | `DEFINITION` | p_L equals p_X+p_Y+p_Z; raw denominators retained |
| SUP-005 | window_tail | `DEFINITION` | 512-decision p95, worst and CVaR remain finite-horizon metrics |
| SUP-006 | paired_orientation | `DEFINITION` | Positive baseline-minus-candidate difference favours candidate |
| SUP-007 | gap_closure | `DEFINITION` | Negative static-to-oracle gap closure is not clipped |
| SUP-008 | observed_truth_split | `FROZEN_PARAMETER` | Truth is evaluator/oracle only |
| SUP-009 | quantization | `FROZEN_PARAMETER` | 10-bit ADC, 8-bit address, 2-bit fraction |
| SUP-010 | map_word | `FROZEN_PARAMETER` | Signed Q9.12, ties-to-even, saturation |
| SUP-011 | fast_path | `FROZEN_PARAMETER` | Six cycles and II=1 before physical transport |
| SUP-012 | cadence | `FROZEN_PARAMETER` | 32-decision posterior and 4,000-cycle image opportunity |
| SUP-013 | compute_budget | `FROZEN_PARAMETER` | 8,192 MAC/B/B and 5,000-us update ceilings |
| SUP-014 | hmm_contract | `FROZEN_PARAMETER` | Four causal states with 0.1/4.0/2.0 calibration parameters |
| SUP-015 | policy_tuple | `FROZEN_PARAMETER` | One tuple selected from 1,728 pilot candidates |
| SUP-016 | bank_transaction | `FROZEN_PARAMETER` | CRC/SHA/CAS A/B bank, six-cycle drain and LKG |
| SUP-017 | formal_split | `FROZEN_PARAMETER` | 12 calibration, 12 pilot and 24 formal clusters |
| SUP-018 | standard_binning | `COMPARATOR` | Weak common-grid reference |
| SUP-019 | static_joint_map | `COMPARATOR` | Strong software comparator, not current full joint-MAP RTL |
| SUP-020 | window_map | `COMPARATOR` | Strongest smooth deployable comparator |
| SUP-021 | ewma_map | `COMPARATOR` | Pilot-locked primary comparator |
| SUP-022 | kalman_map | `COMPARATOR` | Secondary matched deployable comparator |
| SUP-023 | route_a_v4 | `COMPARATOR` | Safety router over Window/EWMA shadows |
| SUP-024 | oracle | `COMPARATOR` | Truth-privileged non-ranking upper bound |
| SUP-025 | paired_bootstrap | `STATISTICAL_RULE` | 20,000 whole-cluster resamples at 95% confidence |
| SUP-026 | multiplicity | `STATISTICAL_RULE` | Holm only within a preregistered endpoint family |
| SUP-027 | failed_policy_families | `NEGATIVE_OR_FAILURE` | Static-switch and freeze-all each fail all 38 tuples |
| SUP-028 | static_window_counterevidence | `NEGATIVE_OR_FAILURE` | V4 is not the best deployable decoder |
| SUP-029 | calibration_tail | `NEGATIVE_OR_FAILURE` | 181/512 versus static 32/512 worst window |
| SUP-030 | bocd_budget | `NEGATIVE_OR_FAILURE` | 13,004.1-us worst update exceeds 5,000-us cap |
| SUP-031 | v5_entry_stop | `NEGATIVE_OR_FAILURE` | Causal/action headroom fails before implementation |
| SUP-032 | physical_board | `NEGATIVE_OR_FAILURE` | All 42 measured fields remain null |
| SUP-033 | generic_long_rtl | `RTL_REPRODUCTION` | One million cycles, ten fault families and all 61 attempts |
| SUP-034 | integrated_long_rtl | `RTL_REPRODUCTION` | 995,802 replay plus 4,198 directed cycles; all 75/25 attempts |
| SUP-035 | toolchain | `RTL_REPRODUCTION` | Trace/model/executable/log hashes and exact tool versions |
| SUP-036 | static_preboard_profile | `RTL_REPRODUCTION` | Only static MAP-LUT has eligible current RTL/P&R estimate |
| SUP-037 | single_ci_cpd | `PHASE6C_LOCATOR` | Project-native square/isotropic equivalence only |
| SUP-038 | two_gkp_cnot | `PHASE6C_LOCATOR` | Project-native matched gate failure, not memory LER |
| SUP-039 | structured_cpd | `PHASE6C_LOCATOR` | Official-code reproduction with small-distance caveat |
| SUP-040 | multimode_cpd | `PHASE6C_LOCATOR` | Independent project-native d=3 drift result |
| SUP-041 | learned_eligibility | `PHASE6C_LOCATOR` | Zero of 16 same-task eligible |
| SUP-042 | gqf_nmf | `PHASE6C_LOCATOR` | Blocked exact reproduction; 13 metrics null |
| SUP-043 | aqec | `PHASE6C_LOCATOR` | Negative project replay; official protocol blocked |
| SUP-044 | external_fpga | `PHASE6C_LOCATOR` | 18 literature rows and zero exact same-task comparator |
| SUP-045 | value_state_semantics | `NON_MIXING_RULE` | N/A, null, failed and negative remain distinct |
| SUP-046 | atlas_nonranking | `NON_MIXING_RULE` | 206 cells, no global score/winner or V5 rescue |

本合同把公式、冻结参数、完整 baseline/CI、负结果、RTL 长序列和 Phase 6C 来源定位绑定到同一附录；N/A、null、failed、negative、blocked 与 ineligible 不可互换，也不能跨 task signature 排名。
