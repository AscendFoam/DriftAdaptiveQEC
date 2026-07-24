# T9.1.4 Phase-9 baseline、检索与统计功效注册表

## 1. 结论

机器合同 verdict=`PASS_T9_1_4_BASELINE_SEARCH_POWER_REGISTRY_FROZEN`，通过 36/36 gates，并杀死 36/36 个针对性语义 mutation。这只是 `SEALED_PRE_OUTCOME` 协议，不是性能实验结果：所有 LER、lifetime、rank、registered-best、external-SOTA、official Puviani、physical lifetime 字段仍为 `null`。

T9.1.3 的 one-way semantic handoff 已按“先 live validate、再重建 canonical payload、最后比较 seal”的顺序消费；当前 branch 为 `QUALIFIED`，但其 `matched_phase9_ranking_eligible=false`。NO_GO typed-null branch 同样释放本任务，不会被删行。

## 2. Matched-deployable 目标榜

| ID | family / role | historical state | current Phase-9 state | boundary |
| --- | --- | --- | --- | --- |
| `standard_measurement_feedback_sbs` | fixed_controller / MANDATORY_BASELINE | `HISTORICAL_PRODUCTION_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | fixed protocol parameters; distinct from autonomous and zero-action anchors |
| `static_recovery_map` | static_decoder_controller / MANDATORY_BASELINE | `HISTORICAL_PRODUCTION_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | frozen calibration-only prior and bank; no formal adaptation |
| `markovian_feedback_fnn` | markovian_learned_controller / MANDATORY_BASELINE | `HISTORICAL_PRODUCTION_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | uses only the latest observed token; not NMF |
| `paper_constrained_nmf` | recurrent_controller / MANDATORY_BASELINE | `QUALIFIED_PAPER_CONSTRAINED_PROVENANCE_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | T9.1.3 is immutable provenance only; qualified and NO_GO branches both require Phase-9 requalification |
| `sliding_window_map` | classical_adaptive_decoder / MANDATORY_BASELINE | `HISTORICAL_PRODUCTION_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | single frozen W/stride; no per-scenario tuning or future samples |
| `ewma_adaptive_map` | classical_adaptive_decoder / MANDATORY_BASELINE | `HISTORICAL_PRODUCTION_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | alpha frozen on calibration only |
| `kalman_adaptive_map` | state_space_filter / MANDATORY_BASELINE | `HISTORICAL_PRODUCTION_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | canonical linear-Gaussian filter; distinct from the separately registered UKF |
| `ukf_adaptive_map` | nonlinear_state_space_filter / MANDATORY_BASELINE | `HISTORICAL_COMPONENT_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | nonlinear sigma-point filter with calibration-only hyperparameters; no formal tuning |
| `regime_hmm` | discrete_state_filter / MANDATORY_BASELINE | `HISTORICAL_COMPONENT_ADAPTER_PENDING` | `PENDING_PHASE9_ACTION_ADAPTER_NULL`; metrics/rank=`null` | causal forward recursion only; no Viterbi or smoothing |
| `interacting_multiple_model_filter` | hybrid_state_filter / MANDATORY_BASELINE | `MISSING_IMPLEMENTATION_TYPED_NULL` | `MISSING_IMPLEMENTATION_TYPED_NULL`; metrics/rank=`null` | must implement model interaction/mixing; HMM cannot be renamed IMM |
| `bocpd` | change_point_filter / MANDATORY_BASELINE | `HISTORICAL_PRODUCTION_BUDGET_FAILURE_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | the historical y-bar/bocd BOCD implementation is registered in the BOCPD family; hazard/run cap/threshold are frozen and historical timeout remains visible |
| `particle_smc` | sequential_monte_carlo / MANDATORY_BASELINE | `PARTIAL_SMOKE_IMPLEMENTATION_TYPED_NULL` | `PARTIAL_PRODUCTION_FORMAL_MISSING_NULL`; metrics/rank=`null` | plugin EAP and posterior marginalization must be reported separately |
| `bayesian_predictor` | bayesian_decoder / MANDATORY_BASELINE | `HISTORICAL_DIFFERENT_TASK_ADAPTER_PENDING` | `PENDING_PHASE9_PER_ROUND_ADAPTER_NULL`; metrics/rank=`null` | must be causal per-round under identical update/action cost |
| `cnn_posterior` | learned_sequence_model / MANDATORY_BASELINE | `HISTORICAL_PARAMETER_ESTIMATOR_DIAGNOSTIC_ONLY_RETRAIN_REQUIRED` | `PENDING_PHASE9_CAUSAL_POSTERIOR_RETRAIN_NULL`; metrics/rank=`null` | legacy checkpoint is diagnostic provenance only; Phase-9 CNN must use the same causal token stream, split and selection ledger |
| `gru_posterior` | learned_sequence_model / MANDATORY_BASELINE | `HISTORICAL_PRODUCTION_RETRAIN_REQUIRED` | `PENDING_PHASE9_RETRAIN_NULL`; metrics/rank=`null` | new tournament training only; old synthetic ranking is not inherited |
| `tcn_posterior` | learned_sequence_model / MANDATORY_BASELINE | `HISTORICAL_PRODUCTION_RETRAIN_REQUIRED` | `PENDING_PHASE9_RETRAIN_NULL`; metrics/rank=`null` | strict causal convolutions; no future padding |
| `causal_ssm_posterior` | learned_sequence_model / MANDATORY_BASELINE | `MISSING_IMPLEMENTATION_TYPED_NULL` | `MISSING_IMPLEMENTATION_TYPED_NULL`; metrics/rank=`null` | state dimension, discretization, stability and streaming update must be explicit |
| `causal_transformer_posterior` | learned_sequence_model / MANDATORY_BASELINE | `MISSING_IMPLEMENTATION_TYPED_NULL` | `MISSING_IMPLEMENTATION_TYPED_NULL`; metrics/rank=`null` | fixed causal context and KV cache; large variants only enter ceiling |
| `proposed_observed_only_posterior_predictive_risk_aware` | proposed_controller / PROPOSED_CANDIDATE | `FUTURE_IMPLEMENTATION_TYPED_NULL` | `NOT_IMPLEMENTED_NOT_SELECTED_NULL`; metrics/rank=`null` | architecture is selected by frozen tournament evidence, not by narrative preference |

其中 mandatory baseline 恰为 18 个。旧 production/PASS 只证明历史 task 的实现；raw/recorded IQ、同 action/codebook、同 precision、同 256-cycle cadence、同 1 ms 全链 wall-clock 和同 selection budget 未重资格前，任何一行都不能排名。

primary promotion 输入固定为 T9.2.7 同一 bit-accurate frontend 的 `FIXED_POINT_DERIVED_LLR`；raw IQ 与 binary syndrome 只作分别闭合的输入消融。paired common randomness 只绑定六态初态、device/scenario draw 与外生 innovation manifest；每个 policy 执行 action 后必须产生并保存自己的 transition/trace hash，禁止把同一完整物理轨迹重放给所有 controller。

## 3. Unbudgeted capacity ceiling

| ID | family / role | historical state | current Phase-9 state | boundary |
| --- | --- | --- | --- | --- |
| `high_particle_smc_ceiling` | sequential_monte_carlo / CAPACITY_CEILING | `FUTURE_PROFILE_TYPED_NULL` | `NOT_EXECUTED_NULL`; metrics/rank=`null` | same traces/actions but no deployable rank or SOTA vote |
| `full_grid_bayesian_ceiling` | bayesian_decoder / CAPACITY_CEILING | `FUTURE_PROFILE_TYPED_NULL` | `NOT_EXECUTED_NULL`; metrics/rank=`null` | accuracy ceiling only |
| `large_gru_ensemble_ceiling` | learned_sequence_model / CAPACITY_CEILING | `FUTURE_PROFILE_TYPED_NULL` | `NOT_EXECUTED_NULL`; metrics/rank=`null` | ensemble and enlarged state cannot vote in deployable ranking |
| `large_ssm_ceiling` | learned_sequence_model / CAPACITY_CEILING | `MISSING_IMPLEMENTATION_TYPED_NULL` | `NOT_EXECUTED_NULL`; metrics/rank=`null` | capacity ceiling only |
| `large_transformer_ceiling` | learned_sequence_model / CAPACITY_CEILING | `MISSING_IMPLEMENTATION_TYPED_NULL` | `NOT_EXECUTED_NULL`; metrics/rank=`null` | capacity ceiling only; no deployable or external-SOTA vote |

这些方法必须使用相同 observation、action 和 formal trace，但允许更大模型、更多 state/context、GPU 或更多 selection compute；因此 `rank=null`、`registered_best_vote=false`、`external_sota_vote=false`。

## 4. Privileged upper bound

| ID | family / role | historical state | current Phase-9 state | boundary |
| --- | --- | --- | --- | --- |
| `hidden_state_decoder_oracle` | oracle / PRIVILEGED_UPPER_BOUND | `HISTORICAL_PRODUCTION_ADAPTER_PENDING` | `PENDING_PHASE9_BACKEND_ADAPTER_NULL`; metrics/rank=`null` | reads hidden physical state and never ranks |
| `hidden_state_teacher` | teacher / TRAINING_TARGET_ONLY | `FUTURE_PHASE9_TEACHER_TYPED_NULL` | `NOT_EXECUTED_NULL`; metrics/rank=`null` | teacher deletion must leave every formal deployable action bit-identical |
| `future_suffix_smoother` | noncausal_oracle / DIAGNOSTIC_UPPER_BOUND | `FUTURE_DIAGNOSTIC_TYPED_NULL` | `NOT_EXECUTED_NULL`; metrics/rank=`null` | future information; never deployable, ranked or distilled without causal student verification |
| `finite_horizon_control_oracle` | control_oracle / SHORT_HORIZON_ASSUMED_MODEL_BOUND | `HISTORICAL_PRODUCTION_DIFFERENT_HORIZON` | `PENDING_PHASE9_SHORT_HORIZON_ADAPTER_NULL`; metrics/rank=`null` | not a global channel or long-lifetime oracle |

hidden-state teacher/oracle、future-suffix smoother 和 assumed-model short-horizon control tree 永不进入 deployable 排名。删除 teacher 后，formal deployable action 必须逐 bit 不变。

## 5. Protocol anchors

| ID | family / role | historical state | current Phase-9 state | boundary |
| --- | --- | --- | --- | --- |
| `standard_binning` | decoder_anchor / WEAK_SANITY_ANCHOR | `HISTORICAL_PRODUCTION_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_HARNESS_NULL`; metrics/rank=`null` | weak sanity anchor; never called strongest baseline |
| `autonomous_sbs_no_measurement` | protocol_anchor / NO_MEASUREMENT_PROTOCOL_ANCHOR | `HISTORICAL_PRODUCTION_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_PHYSICAL_TIME_NULL`; metrics/rank=`null` | not no-correction and not standard measurement-feedback sBs |
| `no_correction_idle_memory` | protocol_anchor / ZERO_ACTION_BREAK_EVEN_ANCHOR | `HISTORICAL_PRODUCTION_REQUALIFICATION_PENDING` | `PENDING_PHASE9_COMMON_PHYSICAL_TIME_NULL`; metrics/rank=`null` | zero active action; not autonomous sBs and not a decoder |

`standard_measurement_feedback_sbs`、`autonomous_sbs_no_measurement` 和 `no_correction_idle_memory` 是三种不同的 measurement/reset/action/wall-clock 协议，不能用一个含糊的 “sBs/no-feedback” 行合并。

## 6. Compute、selection 与 failure 合同

- online：batch=1、单 pinned physical core、1 thread、GPU disabled；每 256 cycles 更新一次，1,000,000 ns deadline 包含 preprocessing、feature、inference、state update、serialization 和 transfer；timeout 只导致不更新，六周期 LKG fast path 不被阻塞。
- envelope：131,072 parameters、1 MiB persistent state、16 MiB workspace、256 MiB peak RSS delta、262,144 online MAC/update、1,048,576 FLOP/update。frontend precision 和 codebook hash 仍为 typed null，由 T9.2.7/T9.3.4 在 pilot 前填充；未填时 matched eligibility 必为 false。
- 每 family 最多 64 个 selection evaluations；learning 最多 16 configs × 4 restarts。所有 checkpoint、失败、timeout/OOM 与未选中 run 均保留，`best-of-N` 无完整 ledger 直接不合格。
- failure status 包含 COMPLETE、DEADLINE_FALLBACK、TIMEOUT、OOM、NUMERIC_FAILURE、CORRUPT_SHARD、MISSING_TOOL、EXTERNAL_UNAVAILABLE、SIGNATURE_INELIGIBLE、TYPED_NULL_NO_GO；禁止 complete-case deletion 和 zero imputation。

## 7. Pilot 与 formal 功效

- pilot：128 个独立 clusters，8 个 macro scenario × 16；只打开一次，最多晋升一个 proposed candidate，不选择或删除 baseline，也不支持论文 claim。
- formal：每个 physics backend 独立 808 clusters，8 × 101；每个 cluster 联合保留六态、全部方法、round/window 和 reset/leakage/control/fallback/timeout。
- 设计目标不是“优于 0”：假定 baseline p_L floor=0.08、真实相对改善 15%，必须把 simultaneous relative LCB 推过 10%。
- 固定 18 comparator divisor 下 required N=806，planned N=808，approximate design power=0.901099。最终推断仍使用 100,000 次 paired outer-cluster maxT bootstrap。
- 上述 90% 是 round-LER comparator screen 的设计功效，不是 lifetime 或整个 multi-endpoint family 的 joint-power 声明；六态 survival/lifetime 功效在 T9.6.1 取得 event-rate proxy 前保持 typed `null`，最终 maxT 仍控制三组闭合 family 的错误率。
- pilot 后不按方差、方向或显著性扩样；不足时输出 `UNDERPOWERED_FIXED_N`，不能补 seed。

## 8. 文献检索、去重与 same-task eligibility

检索截止日为 `2026-07-25T01:37:24+08:00`。保存 6 条原始检索式；arXiv、出版商/DOI 和官方仓库完成 targeted primary verification。当前环境没有 structured CrossRef/Semantic Scholar connector，因此返回数保持 `null` 并记录 `SOURCE_UNAVAILABLE`，没有伪造 hit count。
bibliographic raw hits=23，canonical works=12，merged bibliographic versions=11；另有 4 条 repository/local/dataset 证据，只附着到 work family，不计作检索 hit 或额外 comparator。

| record | work | eligibility | exclusion / typed-null |
| --- | --- | --- | --- |
| `LIT-ROYER-2020` | Stabilization of Finite-Energy Gottesman-Kitaev-Preskill States (2020) | `CONDITIONAL_SYSTEM_BASELINE` | `INCOMPATIBLE_NO_MEASUREMENT_AUTONOMOUS_PROTOCOL`; NO_OFFICIAL_EXECUTABLE_ARTIFACT, INCOMPATIBLE_ACTION_SPACE, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-CAMPAGNE-2020` | Quantum error correction of a qubit encoded in grid states of an oscillator (2020) | `EXTERNAL_PHYSICAL_CONTEXT` | `INCOMPATIBLE_EXPERIMENTAL_DEVICE_AND_METRIC`; CODE_ON_REQUEST, DATA_ON_REQUEST, INCOMPATIBLE_PHYSICS_BACKEND, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-WAN-2020` | Memory-assisted decoder for approximate Gottesman-Kitaev-Preskill codes (2020) | `ADJACENT_BAYESIAN_CONTEXT` | `DIFFERENT_EXTRACTION_AND_FINAL_ONLY_ACTION_PROTOCOL`; NO_OFFICIAL_EXECUTABLE_ARTIFACT, INCOMPATIBLE_ACTION_SPACE, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-DENEEVE-2022` | Error correction of a logical grid state qubit by dissipative pumping (2022) | `EXTERNAL_PHYSICAL_CONTEXT` | `DIFFERENT_TRAPPED_ION_DISSIPATIVE_PROTOCOL`; CODE_ON_REQUEST, INCOMPATIBLE_ACTION_SPACE, INCOMPATIBLE_PHYSICS_BACKEND, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-NOH-2022` | Low-overhead fault-tolerant quantum error correction with the surface-GKP code (2022) | `EXCLUDED_DIFFERENT_CODE` | `DIFFERENT_MULTIMODE_SURFACE_GKP_CODE_AND_TASK`; DIFFERENT_CODE_OR_LOGICAL_DIMENSION, INCOMPATIBLE_ACTION_SPACE, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-WANG-2022` | Multidimensional Bose quantum error correction based on neural network decoder (2022) | `EXCLUDED_OUTER_CODE_AND_MODEL` | `DIFFERENT_MULTIDIMENSIONAL_SURFACE_GKP_TASK`; DIFFERENT_CODE_OR_LOGICAL_DIMENSION, NO_OFFICIAL_EXECUTABLE_ARTIFACT, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-SIVAK-2023` | Real-time quantum error correction beyond break-even (2023) | `EXTERNAL_PHYSICAL_TARGET` | `REAL_DEVICE_GAIN_NOT_COMMON_SIMULATOR_RANK`; CODE_ON_REQUEST, DATA_ON_REQUEST, INCOMPATIBLE_PHYSICS_BACKEND, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-LACHANCE-2024` | Autonomous Quantum Error Correction of Gottesman-Kitaev-Preskill States (2024) | `EXTERNAL_AUTONOMOUS_PHYSICAL_CONTEXT` | `AUTONOMOUS_RESERVOIR_ENGINEERING_NOT_MEASUREMENT_FEEDBACK`; NO_OFFICIAL_EXECUTABLE_ARTIFACT, INCOMPATIBLE_ACTION_SPACE, INCOMPATIBLE_PHYSICS_BACKEND |
| `LIT-PUVIANI-2025` | Non-Markovian feedback for optimized quantum error correction (2025) | `ALGORITHM_PORT_CANDIDATE_OFFICIAL_SURPASS_BLOCKED` | `OFFICIAL_EXACT_AND_COMMON_HARNESS_INCOMPLETE`; MISSING_OFFICIAL_CHECKPOINT, MISSING_AGENT_SEED_LEDGER, MISSING_SELECTION_LEDGER, MISSING_SIX_STATE_EVALUATOR, INCOMPATIBLE_ACTION_SPACE, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-BROCK-2025` | Quantum error correction of qudits beyond break-even (2025) | `EXTERNAL_DIFFERENT_LOGICAL_DIMENSION` | `GKP_QUTRIT_QUQUART_NOT_QUBIT_SIX_STATE`; DIFFERENT_CODE_OR_LOGICAL_DIMENSION, INCOMPATIBLE_PHYSICS_BACKEND, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-VAIDHYANATHAN-2026` | Quantum feedback control with a transformer neural network architecture (2026) | `ADJACENT_ARCHITECTURE_CONTEXT` | `NON_GKP_QUANTUM_CONTROL_TASK`; DIFFERENT_CODE_OR_LOGICAL_DIMENSION, INCOMPATIBLE_ACTION_SPACE, METRIC_OR_DENOMINATOR_MISMATCH |
| `LIT-SIVAK-2026` | Reinforcement learning control of quantum error correction (2026) | `ADJACENT_DRIFT_ADAPTATION_CONTEXT` | `SURFACE_COLOR_CODE_DEVICE_CONTROL_NOT_SINGLE_MODE_GKP`; PROPRIETARY_CODE, DIFFERENT_CODE_OR_LOGICAL_DIMENSION, INCOMPATIBLE_ACTION_SPACE, INCOMPATIBLE_PHYSICS_BACKEND |

DOI 先规范化后精确去重；无 DOI 时依次用 arXiv、规范化 title + first author，title token Jaccard 门为 0.90。Puviani 的 PRL、arXiv 和 GQF 只构成一个 work family。逐条筛选 ledger 位于 `docs/t9_1_4_literature_search_ledger.csv`。

截至 cutoff，已核验外部工作没有一项同时满足本项目的 input、history、trusted action、physics backend、online timing、no-postselection denominator、metric 和 compute signature；所以 external-SOTA 不是 false/negative，而是 `null`。允许的未来措辞至多是 “best among preregistered matched-deployable baselines under the frozen T9 task signature”。

## 9. 反简化审计

- 删除 mandatory baseline、把 IMM 改名为 HMM、或隐藏 missing implementation 均失败；
- oracle 入榜、future suffix、hidden truth、额外 IQ/history/action 权限均失败；
- 省略 preprocessing/transfer、放宽 GPU/thread/deadline、隐藏 best-of-N 或 timeout/OOM 均失败；
- 用 60 clusters、round-level pseudoreplication、pointwise CI、缩小 divisor、zero-denominator epsilon 或 pilot 后扩样均失败；
- 把 registry-best 自动升级 external SOTA、把 T9.1.3 自动升级 matched、填入 official/Puviani/physical claim 或把同一论文多算 comparator 均失败；
- 完整 audit 为 `36/36`。

## 10. 复现

```powershell
python -m cnn_fpga.benchmark.phase9_baseline_search_power_registry
python -m cnn_fpga.benchmark.phase9_baseline_search_power_registry --verify
python -m pytest -q tests/test_phase9_baseline_search_power_registry.py
```
