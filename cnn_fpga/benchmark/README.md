# benchmark/ — 实验基准与 HIL 验证套件

本目录包含实验基准脚本的完整集合，覆盖从纯数值仿真漂移基准到 FPGA 硬件在环 (HIL) 验证的全流程。脚本按实验阶段 (P1–P4) 组织；多数脚本是独立 CLI，窄范围科学验收 harness 也提供可测试的 Python API。

## 目录结构

| 文件 | 阶段 | 职责 |
|------|------|------|
| [adaptive_drift_alignment.py](adaptive_drift_alignment.py) | M1.3 | 同一逐样本 trace 上对齐 static、现有 Window/EKF 与 full-state model oracle；强制一窗因果延迟并报告 paired gap CI |
| [run_length_fsm_baseline.py](run_length_fsm_baseline.py) | T3.2.5 | 以真实 3-bit event FSM/ParamBank 重放 training-only 阈值网格，并在同 trace 比较 static、memoryless、run-length 与 truth event-cost lower bound |
| [regime_hmm_baseline.py](regime_hmm_baseline.py) | T3.2.6 | 训练/validation/evaluation 隔离的 four-state causal HMM、same-emission memoryless temporal ablation、posterior calibration 与 shared future-CNN budget |
| [latest_outcome_markovian_baseline.py](latest_outcome_markovian_baseline.py) | T3.2.7 | 训练精确匹配 history GRU 参数/MAC/动作/协议的 5-agent latest-outcome FNN，并在同 trace 报告 signed memory contrast 与 cutoff reversal |
| [autonomous_sbs_wallclock_baseline.py](autonomous_sbs_wallclock_baseline.py) | T3.2.8 | 以 7/10 us 协议原生 cycle 推进到共同 700 us，保留 per-cycle/per-us lifetime、raw measurement/reset/gate 账本与 cutoff sensitivity |
| [trajectory_lookup_control_oracle.py](trajectory_lookup_control_oracle.py) | T3.2.9 | exact 枚举两-cycle 16 branches，优化 open-loop 与 15-node causal lookup 各 3 restarts×两阶段，并保存 cutoff transfer、checkpoint 和指数资源审计 |
| [exponential_recurrence_baseline.py](exponential_recurrence_baseline.py) | T3.2.10 | 优化 75 参数 PRL-inspired 指数递推，保存 cutoff12/16 exact fidelity、Q 定点镜像，并在独立 event-cost lane 与 run-length FSM 同轨比较 |
| [memory_specific_ablation.py](memory_specific_ablation.py) | T3.2.11 | 对冻结 NMF 做 prefix-consistent shuffle/truncation/reset/latest-only，并复用同预算重训 MF，在 cutoff12/16 保留 signed mechanism counterevidence |
| [slow_loop_model_selection.py](slow_loop_model_selection.py) | T4.1.1 | 在共同 8-window four-regime task 与 4096 MAC/4096 B envelope 下，validation-only 比较 TCN、GRU、HMM、Kalman、指数递推和 FSM，并保存 rolling-cache/checkpoint/Source Data |
| [experimental_history_validation.py](experimental_history_validation.py) | T4.1.2 | 连接真实 syndrome/FSM/LLR/scheduler producer，验证 256×53 observed-only history、truth-leak denylist、fault/status/saturation 全路径和 16,384-row Source Data |
| [hybrid_state_output_validation.py](hybrid_state_output_validation.py) | T4.1.3 | 恢复注册 HMM，在 nominal/stress lane 验证 continuous/regime/risk/recovery-burden/uncertainty output、stage/hold 与 version/CRC/atomic bank 语义 |
| [hybrid_multiobjective_calibration.py](hybrid_multiobjective_calibration.py) | T4.1.4 | 将 T4.1.3 output 对齐未来 32 cycles，以 3/2/3 seed strict split 校准六项 loss、proper scores、fallback recall 与 frozen-output ablation |
| [offline_teacher_student_distillation.py](offline_teacher_student_distillation.py) | T4.1.5 | hash 恢复 T2.3.7 五个 frozen NMF teacher，以 3-split/3-restart 蒸馏 75 参数递推 student，验证 online causality、安全 fallback、artifact 和资源边界 |
| [parametric_map_lut_validation.py](parametric_map_lut_validation.py) | T4.2.1 | 从真实 ParamBank active K/b 编译 X/Z integer MAP ROM，穷举 8 banks×2 phases×1024 codes，验证地址收敛、5-cycle pipeline、II=1 与 image/version fail-closed |
| [experimental_event_fsm_validation.py](experimental_event_fsm_validation.py) | T4.2.2 | 8×128-cycle replay 验证六态 event FSM、六个饱和 counter、reset/fallback、双轴 frame、MAP/version 对齐和 6-cycle/II=1 software action contract |
| [conservative_fallback_validation.py](conservative_fallback_validation.py) | T4.2.3 | 16×256-cycle replay 覆盖 14-bit OOD/leakage/stale/CRC/SHA/version/deadline/MAP/ack taxonomy、trusted version、frame hold、恢复迟滞和 reason trace |
| [fast_path_fixed_point_validation.py](fast_path_fixed_point_validation.py) | T4.2.4 | 四档 precision×8 banks 的 87,040-code exhaustive audit，加 8-bank×4-seed paired exact-float/bit-accurate LER 与全 word/rounding/resource-proxy contract |
| [three_timescale_cadence_validation.py](three_timescale_cadence_validation.py) | T4.3.1 | 真实 scheduler/ParamBank/T4.2 fast-path trace、两种 evidence policy×4000 onset phases、minute/end-run due 与 age/cadence 集成门 |
| [atomic_parameter_bank_validation.py](atomic_parameter_bank_validation.py) | T4.3.2 | 3745-prefix/3745-byte corruption 穷举、chunk/order、manifest/CAS/stale/hysteresis negatives、A/B switch、pipeline latch、race 与 ack/readback 验证 |
| [closed_loop_fault_recovery_validation.py](closed_loop_fault_recovery_validation.py) | T4.3.3 | 8 场景×4 seeds×23996 cycles 验证 drift/burst/leakage/host timeout/通信中断/坏包/race/guard，逐周期 action safety、ack uncertainty、freshness refresh 与 monotonic LKG republish |
| [bounded_residual_rnn_teacher.py](bounded_residual_rnn_teacher.py) | T4.4.1 | 训练 3 个 fresh 72,853 参数 GRU restart，以 validation-only 选择 bounded 15-residual teacher，并保存 cutoff12/16 held-out、checkpoint/source hash、失败与 cap-hit 证据 |
| [bounded_residual_teacher_analysis.py](bounded_residual_teacher_analysis.py) | T4.4.2 | 冻结 selected teacher，提取 g/e hidden/control、forced-path p(g)、PCA、指数饱和、impulse/Jacobian memory，并把 leakage 限定为 reset+nominal OOD proxy |
| [low_dimensional_student_distillation.py](low_dimensional_student_distillation.py) | T4.4.3 | 训练 1/2/4-state×3-restart outcome-specific exponential students，validation-only 选维，报告 held-out imitation error，并导出 hash-bound pure-NumPy fail-closed artifact |
| [teacher_student_gain_retention.py](teacher_student_gain_retention.py) | T4.4.4 | 以全新 paired seeds 做 10-cycle standard/5×MF/teacher/handcrafted/student retention，并用独立 exact 2-cycle lane 加入 horizon-bound control oracle，显式报告 p(g)/e/leakage burden 与成本 |
| [teacher_student_branch_freeze.py](teacher_student_branch_freeze.py) | T4.4.5 | 只读验证 T4.4.1--T4.4.4 gates、源码/文件 hash 与预注册 retention，机器选择 qualified student-retention 或 drift/regime-aware MAP-LUT fallback，并保留 MF 反证/禁止 claim |
| [comparison_set_registry.py](comparison_set_registry.py) | T5.1.1 | 冻结 19 comparator/8 lanes，执行 no-correction 与 finite-energy probes，校验 16 artifacts/19 code bindings，并禁止异构旧结果拼成全局排名 |
| [mixed_scenario_matrix.py](mixed_scenario_matrix.py) | T5.1.2 | 执行 10 类 mixed noise/regime：6 个 paired decoder scenarios 加 loss、readout/ancilla、large-error 与 leakage 原生 lanes；禁止跨 lane 排名 |
| [oracle_gap_tail_report.py](oracle_gap_tail_report.py) | T5.1.3 | 重放 1,152 windows，报告 average/p95/worst、paired decoder-oracle gap、20k seed bootstrap、24-test Holm family 与独立 exact two-cycle control-reference gap |
| [algorithm_success_falsification.py](algorithm_success_falsification.py) | T5.1.4 | 只读核验 8 个 parents、8 项 strong predicates、Holm/tail 反证与 claim/reopen contract，机器选择 learned-performance 或 adaptive MAP/FPGA co-design fallback |
| [time_cost_fairness.py](time_cost_fairness.py) | T5.1.5 | 分列 common-700 μs protocol、common-100 μs controller 与 host estimator lanes，同时报告 cycle/μs、measurement/reset/gates、analytic cost、latency null 和排序反转 |
| [experimental_feasibility.py](experimental_feasibility.py) | T5.1.6 | 汇总 controller occupancy/reset/slew/cost、软件 fault fallback/unsafe rates、student fail-closed 与 7 个缺失 feasibility fields，禁止峰值 lifetime 升级 deployment claim |
| [displacement_large_error_causal.py](displacement_large_error_causal.py) | T5.2.1 | 17 幅度×8 seed-cluster 因果注入；分列 recovery/e-run、nearest-operation logical failure 与 identity-reference flip，禁止把 component recovery 冒充 physical-memory LER |
| [ancilla_readout_causal.py](ancilla_readout_causal.py) | T5.2.2 | bit/phase/readout 三条互斥 6-rate×8-seed causal lanes；分列主效应与全交叉负控，禁止 mixed intervention、global score 和 device/65× claim |
| [leakage_reset_causal.py](leakage_reset_causal.py) | T5.2.3 | leakage-injection/reset-failure 两条互斥 6-rate×8-seed lanes；分列检测、tail、availability 与 raw reset cost，保留 leakage-free null，禁止 post-selection、device-rate 和 LER claim |
| [logical_channel_reconstruction.py](logical_channel_reconstruction.py) | T5.3.1 | 4-cutoff×3-noise×QEC on/off 六态 CPTNI tomography；分列 full PTM、Choi/TNI、non-Pauli/leakage 与 raw-area/censored lifetime，禁止 postselection 与 break-even/device claim |
| [logical_channel_fidelity.py](logical_channel_fidelity.py) | T5.3.2 | 从 T5.3.1 raw outputs 重算 CPTNI `F_e/F_avg`、TP 公式高估与 1/3/4-point 短时率；主动瞬态不授予寿命，禁止 fake CI、指数拟合和 break-even claim |
| [logical_operational_boundary.py](logical_operational_boundary.py) | T5.3.3 | 用完整 31 点 active/passive `F_avg` 曲线冻结持续非劣与累计偿还边界；保留低 cutoff 失败，禁止终点比值、伪 full cost 与 paper coherence-gain claim |
| [qec_postselection_cost.py](qec_postselection_cost.py) | T5.3.4 | 隔离核算 300 μs online event/resource、offline rejection penalty、software safety 与 missing fields；禁止 conditional 增益、null 填零和跨 lane full-cost/break-even |
| [qec_channel_recovery_bound.py](qec_channel_recovery_bound.py) | T5.3.5 | 计算 finite-cutoff QEC-matrix/Petz 双边界，以 repaired primal/dual SDP 校验 small cutoffs，并分离 actual sBs 时序诊断与不可比 teacher/student 指标 |
| [held_out_ood_validation.py](held_out_ood_validation.py) | T5.4.1 | 以全新 disjoint seeds 重放 frozen decoder、sBs measurement/leakage kernels 与 scheduler communication faults，分 lane 报告 unseen family/range 和负结果，不生成 universal robustness score |
| [uncertainty_gated_fallback.py](uncertainty_gated_fallback.py) | T5.4.2 | development-only 选择 observed ensemble threshold，在 fresh matched OOD/nominal traces 对比 EWMA no-fallback 与 static last-known-good gate，并逐项报告 avoided/induced/unnecessary cost 和场景反转 |
| [causal_ablation_negative_results.py](causal_ablation_negative_results.py) | T5.4.3 | 在五条原生证据 lane 分别关闭六项机制，重算 same-trace 对照并保存负结果/claim 降级；禁止跨 metric 总分或端到端联合归因 |
| [multi_agent_seed_selection_audit.py](multi_agent_seed_selection_audit.py) | T5.4.4 | 重构 NMF/slow-loop/teacher/student/retention 的 validation-only selection，报告全体 agent/restart/seed 的 median/IQR/worst quartile 和不参与重选的 test-best optimism |
| [horizon_extrapolation_validation.py](horizon_extrapolation_validation.py) | T5.4.5 | 重拟合 2/5/10-cycle students，连同 frozen 32-cycle student 在 8 条 `1e6`-cycle streams 上审计全步 state bound、sampled action imitation、float32 与 reset recovery |
| [randomized_model_mismatch.py](randomized_model_mismatch.py) | T5.4.6 | 在四条不可拼榜的原生 lane 执行 64 个随机 mismatch cells，并从 raw physical strategy scores 重算 student-retention/fallback branch |
| [bit_accurate_hardware_reference.py](bit_accurate_hardware_reference.py) | T5.5.1 | 验证 packed input/output/state、binary parameter image、真实 5+1-cycle pipeline、atomic in-flight bank switch 与 deterministic golden trace |
| [gru_student_hardware_feasibility.py](gru_student_hardware_feasibility.py) | T5.5.4 | 对 full float GRU、完整参数 int8/Q3.14 lower-bound 与 4-state student 做 fail-closed 存储/MAC/CXXRTL/三 seed P&R/deadline/gain 比较 |
| [long_rtl_qualification.py](long_rtl_qualification.py) | T6.2.2 | 生成 10×100,000-cycle independent integer trace，以并行 CXXRTL 全字段对拍 production core，并审计抽象 FIFO/通信故障、恢复、饱和和 mutation gates |
| [route_a_claim_contract.py](route_a_claim_contract.py) | T6.5.1 | 冻结/验证 Route-A 11 个 canonical roles、三条 metric-disjoint comparison lanes、11 条 claim 的 evidence/activation/revocation/forbidden-wording 合同 |
| [unified_execution_contract_validation.py](unified_execution_contract_validation.py) | T6.5.2 | 验证七个 observed-only 方法和隔离 oracle 的统一 syndrome/LUT/bank/cadence/budget/deadline contract；生成逐方法 70-case fail-fast matrix |
| [route_a_preregistration.py](route_a_preregistration.py) | T6.5.3 | 在 T6.7 formal 结果前冻结 143 场景 cells、24 independent clusters、common validation-only threshold selector、paired cluster statistics、tail/catastrophic/nominal GO-NO-GO |
| [unified_comparator_runner.py](unified_comparator_runner.py) | T6.6.1 | 从同一 T6.5.2 phase packet trace 真实执行 standard/static/Window/EWMA/Kalman/Route-A，legacy CNN 真实 checkpoint 失败自动降级，oracle 独立分栏；验证 q/p bridge、matched cost、grid equivalence、prefix causality 与 mutations |
| [regime_aware_safe_policy_validation.py](regime_aware_safe_policy_validation.py) | T6.6.2 | production-cadence 结构长轨；验证 normal/tail/leakage/integrity/rollback/hysteresis、真实 A/B commit/readback、5+1-cycle ledger 与 mutation fail-closed，不作为 HMM 校准或 LER 证据 |
| [route_a_board_ready_preflight.py](route_a_board_ready_preflight.py) | T6.9.2 preboard | 绑定 UART 板级 top、constraints、routed netlist、`.fs`、CXXRTL/PHY 回归与 source hash；只签发 `PASS_PREBOARD_CANDIDATE_NOT_PHYSICAL_QUALIFICATION`，六项真板前置缺失时保持 measured claim 关闭 |
| [learned_model_eligibility_replay.py](learned_model_eligibility_replay.py) | T6.17.3 | 只读审计 16 个 learned/controller family 的 exact task signature、部署预算和 all-restart multiplicity；仅重放父任务已有逐样本锚点的 legacy TinyCNN，禁止训练、重选、跨 lane LER/latency 排名或回写 Phase 6B verdict |
| [aqec_secondary_wallclock_replay.py](aqec_secondary_wallclock_replay.py) | T6.18.1 | 在现有 exact finite-cutoff simulator 上以独立准静态 lifetime clusters 同轨运行 idle、measurement/reset、autonomous/reset 700 us curves；分列 area lifetime、低R² fit、event burden和cycle/us反转，不构造 Lachance reservoir substitute或零延迟/20%增益 claim |
| [official_structured_cpd_reproduction.py](official_structured_cpd_reproduction.py) | T6.18.2 | 绑定 official `LatticeAlgorithms.jl` commit/license/Julia manifest 与原始 JSON，分开验证 exact-CVP correctness、作者 JLD2 aggregate reanalysis、独立小距离 finite-size crossing 和 runtime/memory；analog adapter、upstream caveat 与小距离次序反转均 fail-closed |
| [multimode_posterior_weighted_cpd.py](multimode_posterior_weighted_cpd.py) | T6.18.3 | 只读消费四个 Julia formal shards，按32个 seed clusters重算 static/adaptive/oracle 的 p_L、worst/CVaR95、lag、runtime/memory和双侧Holm；绑定Source Data与stdout/stderr，保持oracle非排名、Phase6B hash不变和model-matched d=3 scope |
| [multimode_causal_headroom.py](multimode_causal_headroom.py) | T6.20.4 | 校验 12-seed×13-family×512-round train-only Julia raw ledger，独立重算七方法、五段 regret、50k paired bootstrap、BSV/T-join/alias/causality与15类mutation；未过15%/12%即输出 direct NO-GO，拒绝访问pilot/formal或删baseline救援 |
| [phase6c_preboard_profiles.py](phase6c_preboard_profiles.py) | T6.19.1 | 资格优先的项目预板 profile：仅已有 static MAP-LUT fast-path RTL 进入 CXXRTL/三种子 P&R 表；CI/V5/Direct-NN 无合格 RTL 则 N/A。Window/EWMA/Kalman 的真实 update/compiler/software-transfer/commit 另表报告，并从 3000 条原始行重算分位数；所有板测字段保持 null |
| [run_drift_suite.py](run_drift_suite.py) | P1 | 纯数值仿真漂移基准，对比 full QEC vs simplified 模型 |
| [run_hardware_emulation.py](run_hardware_emulation.py) | P2 | 硬件行为仿真，验证双环路运行时（快环/慢环）无真实 FPGA |
| [run_p2_mode_benchmark.py](run_p2_mode_benchmark.py) | P2 | 多 slow-loop 模式基准对比（fixed_baseline / oracle / model_artifact） |
| [run_hil_suite.py](run_hil_suite.py) | P3 | HIL 核心执行引擎，通过 FPGADriver 驱动 mock/real 后端 |
| [run_hil_mode_benchmark.py](run_hil_mode_benchmark.py) | P3 | HIL 多模式基准（mock / float / int8 / real-board） |
| [run_p3_param_sweep.py](run_p3_param_sweep.py) | P3 | 参数扫描调优（gain_clip / beta_smoothing / alpha_bias / gain_scale） |
| [run_p3_histogram_tuning.py](run_p3_histogram_tuning.py) | P3 | 直方图输入饱和调优（syndrome_limit / histogram_range_limit） |
| [run_p4_multiscenario_benchmark.py](run_p4_multiscenario_benchmark.py) | P4 | 冻结多场景正式基准，输出对比 CSV 和报告 |
| [run_p4_hybrid_vs_ukf_ablation.py](run_p4_hybrid_vs_ukf_ablation.py) | P4 | Hybrid vs UKF 消融实验（teacher / features / context 三组） |
| [run_p4_gap_diagnostic.py](run_p4_gap_diagnostic.py) | P4 | 差距诊断：同一窗口序列下对比多种预测模式 |
| [run_p4_no_teacher_params_stability.py](run_p4_no_teacher_params_stability.py) | P4 | 种子扫描稳定性检查（Hybrid Full vs No TeacherParams） |
| [run_p4_teacher_params_reencoding_controlled.py](run_p4_teacher_params_reencoding_controlled.py) | P4 | 受控三变体对比（Full / No TeacherParams / Reencoded） |
| [run_p4_teacher_representation_paired.py](run_p4_teacher_representation_paired.py) | P4 | 配对 teacher-representation 基准（gated v2–v9, selective, minimal） |
| [analyze_seed20260429_failure.py](analyze_seed20260429_failure.py) | 离线 | 分析特定种子基准失败原因 |
| [analyze_seed20260429_trace.py](analyze_seed20260429_trace.py) | 离线 | 逐窗口轨迹导出与聚合 |
| [summarize_p4_features_ablation.py](summarize_p4_features_ablation.py) | 离线 | Features 消融结果汇总（Markdown / CSV / LaTeX） |

## 脚本分层依赖关系

```
── 核心执行层 ──────────────────────────────────────────────────
│  run_hil_suite.py          → 提供 run_hil_session(), HILSlowJob
│  run_hardware_emulation.py → 提供 _run_repeat(), _aggregate_summaries()
└──────────────────────────────────────────────────────────────
       ↓ (直接调用)
── 基准编排层 ──────────────────────────────────────────────────
│  run_p2_mode_benchmark.py        → 调用 _run_repeat
│  run_hil_mode_benchmark.py       → 调用 run_hil_session
│  run_p3_param_sweep.py           → 调用 run_hil_session
│  run_p3_histogram_tuning.py      → 调用 run_hil_session
│  run_p4_multiscenario_benchmark.py → 调用 run_hil_session
│  run_p4_gap_diagnostic.py        → 调用 HILSlowJob, _build_mock_noise_provider
└──────────────────────────────────────────────────────────────
       ↓ (子进程调用)
── P4 消融/训练编排层 ──────────────────────────────────────────
│  run_p4_hybrid_vs_ukf_ablation.py          → 子进程调用 P4 benchmark
│  run_p4_no_teacher_params_stability.py      → 子进程调用 P4 benchmark
│  run_p4_teacher_params_reencoding_controlled.py → 子进程调用 P4 benchmark
│  run_p4_teacher_representation_paired.py    → 子进程调用 P4 benchmark
└──────────────────────────────────────────────────────────────
       ↓ (读取输出)
── 离线分析层 ──────────────────────────────────────────────────
   analyze_seed20260429_failure.py     → 读取 CSV/JSON 输出
   analyze_seed20260429_trace.py       → 读取 hil_events.json
   summarize_p4_features_ablation.py   → 读取基准输出 + 消融配置
```

## 核心函数说明

### `run_hil_suite.py`

`run_hil_session(config, run_dir)` 是 HIL 验证的中心函数：

1. 创建 `FPGADriver`（mock 或 real backend）
2. 创建 `SlowLoopRuntime`（慢环推理）
3. 驱动快环循环，轮询直方图窗口
4. 分发慢环推理任务，管理参数银行 stage/commit
5. 记录事件，计算时序/违规统计
6. 输出 `hil_summary.json`

### `run_hardware_emulation.py`

`_run_repeat(config, scenario, repeat_idx, run_dir, seed)` 执行一次完整的 P2 仿真重复：

1. 构建 `ParamBank`, `SlowLoopRuntime`, `DualLoopScheduler`, `FastLoopEmulator`
2. 运行全部快环周期
3. 返回包含 LER、溢出率、违规率的汇总字典

### `run_p4_multiscenario_benchmark.py`

正式基准执行器，支持：

- 场景 × 模式 × 重复的完整组合
- 分块/可恢复执行（`--repeat-start`, `--repeat-stop`, `--resume-only`）
- 配对种子（`--paired-seeds`）
- 输出：`comparison.csv`, `delta.csv`, `report.md`, `summary.json`

## 使用示例

### T5.0.1 文献趋势 registry

```bash
python -m cnn_fpga.benchmark.literature_trend_reproduction
```

该命令只读核验既有复现 artifacts 与本地来源锚，生成 14-target JSON 和 52-row Source Data；`PASS`
表示 registry 完整，不会把 pending secondary 或外部 reference 计作复现通过。

### T5.0.2 独立 cross-fidelity holdout

```bash
python -m cnn_fpga.benchmark.independent_cross_fidelity_holdout
```

该命令在排除 calibration 和 exploratory pilot 的正式点上重跑 cross-fidelity，并在独立参数网格上核验
P-Steane 小噪声解析公式。总状态允许“至少一个独立 family 通过”，但 JSON 会逐族保留失败；当前 main
cross-fidelity 为 `FAIL`、secondary P-Steane 为 `PASS`，不得把总 `PASS` 解读为主线通过。

### T5.1.1 完整 comparison set

```bash
python -m cnn_fpga.benchmark.comparison_set_registry
```

生成 19-comparator、8-lane registry 和 100-row Source Data。`PASS` 只表示 catalog、可执行绑定和
nonmixing gates 完整；统一 T5.1.2 matrix 仍是 preregistered、未执行状态。

### T5.1.2 mixed scenario matrix

```bash
python -m cnn_fpga.benchmark.mixed_scenario_matrix
```

生成 10 类场景、36 个 decoder seed-cluster、589,824 个 paired decoder decisions 和 116-row Source Data。
`PASS` 只表示 exact coverage、shared-trace causality、native component gates 与 provenance 通过；loss、
protocol fault、large-error/leakage component 和 syndrome-decoder 指标不组成全局 leaderboard。

### T5.1.3 average/tail 与双 oracle-gap

```bash
python -m cnn_fpga.benchmark.oracle_gap_tail_report
```

按 T5.1.2 trace 精确重放 1,152 windows，生成 7,139-row Source Data。6 个 seeds 是 bootstrap/sign-flip
cluster；24 项比较统一 Holm 校正。control reference 保持 cutoff12/16 exact two-cycle，不对 exact branch
expectation 伪造 sampling CI，也不与 syndrome `P_L` 跨 lane 排名。

### T5.1.4 算法成功/证否分支

```bash
python -m cnn_fpga.benchmark.algorithm_success_falsification
```

只读绑定 T5.1.1--T5.1.3 和 deployable MAP-LUT safety artifacts，生成 278-row claim/evidence/reopen
ledger。当前 strong learned-decoder branch 因无 matched candidate、0 Holm discoveries 和 calibration-shift
transient 反证失败；`PASS` 表示 fail-closed fallback 判定完整，不表示 CNN 或 learned algorithm 性能通过。

### T5.1.5 物理时间与控制成本公平化

```bash
python -m cnn_fpga.benchmark.time_cost_fairness
```

生成 12 个 protocol、10 个 controller、6 个 host-profile rows 和 537-row Source Data。protocol/controller
同时报 cycles 与 μs；`e` events 与 reset 分列。controller 与 target-board latency 未测保持 null，T4.1.1
host estimator profile 不转移到 controller/FPGA，也不生成跨 lane 总分。

### T5.1.6 实验可行性约束

```bash
python -m cnn_fpga.benchmark.experimental_feasibility
```

只读绑定 controller、fault-recovery、component fallback、student runtime 与 active branch，生成 10 个
controller rows、8 个 fault scenarios 和 408-row Source Data。`PASS` 表示 burden/null/fail-closed 报告完整；
multilevel leakage、saturation、matched/board/frontend latency 仍缺失，总体 deployment readiness 不成立。

### T5.2.1 displacement / large-distance 因果注入

```bash
python -m cnn_fpga.benchmark.displacement_large_error_causal
```

用 17 个预注册幅度、8 个独立 seed clusters 生成 136 recovery 与 272 logical seed rows。recovery depth/e-run
和最近操作相对 logical failure 在 `l_S/4` 达峰；identity-reference flip 同报，暴露 `l_S/2` 端点本身是
logical operation。所有 truth 仅供 evaluator，结果不是 physical-memory LER 或 device fault injection。

### T5.2.2 ancilla bit/phase 与 readout 独立因果注入

```bash
python -m cnn_fpga.benchmark.ancilla_readout_causal
```

运行 bit-only、phase-only、readout-only 三个互斥实验族，各含 6 个 rate、8 个独立 seed clusters 和
4,096 cycles/cell。主效应、解析 rate、whole-seed CI 与所有 cross-channel negative controls 同报；phase
不切换 Z-basis outcome，truth 不进入 deployable record。结果是 effective-model sensitivity，不是实验 65×、
physical-memory LER 或 device fault injection。

### T5.2.3 leakage 与 reset-failure 独立因果注入

```bash
python -m cnn_fpga.benchmark.leakage_reset_causal
```

分别扫描 leakage injection 与 reset failure，各含 6 个 rate、8 个独立 seed clusters、256 条 trajectory 和
512 个 evaluation cycles/trajectory；128-cycle burn-in 不计入 estimand。检测延迟在无真实 episode 时保持
`null`，false alarm、declared/safe availability、raw reset request/attempt/success/failure 以及多 lag correlation/
covariance 同报。formal sample 的 observed detection fraction 为 1 只表示本次有限样本结果，不是总体检测保证；
结果也不是 post-selection、device availability、physical-memory LER 或 hardware fault injection。

### T5.3.1 六态 matched logical channel 重构

```bash
python -m cnn_fpga.benchmark.logical_channel_reconstruction
```

### T5.3.2 CPTNI fidelity 与短时率

```powershell
python -m cnn_fpga.benchmark.logical_channel_fidelity
```

输出 `docs/t5_3_2_logical_channel_fidelity.json` 与对应 Source Data。脚本校验 T5.3.1 parent hash，逐
cycle 从六态未归一化 outputs 重算指标，并保留 qec-on 初始瞬态导致的 `unreliable`/null lifetime，不能把
deterministic cutoff/time-grid spread 写成统计 CI。

### T5.3.3 full-curve operational boundary

```powershell
python -m cnn_fpga.benchmark.logical_operational_boundary
```

输出 `docs/t5_3_3_logical_operational_boundary.json` 与 416-row Source Data。正式 verdict 只允许
finite-cutoff、300 μs、wall-clock matched operational boundary；active cost、best-passive physical reference、
coherence gain 和 experimental break-even 均保持未建立。

### T5.3.4 QEC 与 post-selection 成本

```powershell
python -m cnn_fpga.benchmark.qec_postselection_cost
```

输出 `docs/t5_3_4_qec_postselection_cost.json` 与 94-row Source Data。online channel、offline
post-selection 和定向 software safety campaign 不混排；12 个未测量 cost/latency/LER/reference fields 保持
null，task PASS 不升级 full-cost 或 experimental claim。

### T5.3.5 QEC-matrix/Petz channel-recovery bound

```powershell
python -m cnn_fpga.benchmark.qec_channel_recovery_bound
```

输出 `docs/t5_3_5_qec_channel_recovery_bound.json` 与 119-row Source Data。15 个 cutoff
4/6/8/10/12×三噪声 lane 同时保存 Petz theorem interval、修复后的 CPTP primal 下界与 shifted
dual-feasible 上界；另把 fixed-`Delta` 扩到 cutoff 48，并在 `Delta=0.44/0.34/0.28` 做能量敏感度。
actual nominal sBs 含交错 gate/reset/ancilla noise，和“10 μs pure loss 后任意 terminal recovery”的
调度不同，因此只报 diagnostic gap；T4.4.4 teacher/student 的 10-cycle lifetime 与 one-cycle `F_e`
不可相减，gap 保持 `null/INCOMPARABLE`。

### T5.4.1 held-out/OOD validation

```powershell
python -m cnn_fpga.benchmark.held_out_ood_validation
```

输出 `docs/t5_4_1_held_out_ood_validation.json` 与 280-row Source Data。formal campaign 含
24 个 drift、24 个 measurement-confusion、24 个 leakage-rate 和 32 个 communication seed cells；
冻结 T5.1.2 decoder，不用 OOD 数据选模型/阈值。telegraph adaptive reversal、periodic short-pause null 与
compound communication degradation 全部保留。四条 lane 的 metric 不拼接，task PASS 不代表 system/device
robustness，fallback 因果收益留给 T5.4.2。

### T5.4.2 uncertainty-gated fallback

```powershell
python -m cnn_fpga.benchmark.uncertainty_gated_fallback
```

输出 `docs/t5_4_2_uncertainty_gated_fallback.json` 与 517-row Source Data。阈值只用 T5.4.1
development seeds 选择，确认使用 12 个 fresh parent-disjoint clusters。总体 OOD absolute reduction 为
`0.00107490 [0.00001950,0.00227615]`，但 telegraph 显著获益、compound 显著受损，nominal 点估计也
轻微变差；因此只建立 mixture-qualified syndrome-decision 证据，不建立 universal/device safety。

### T5.4.3 causal ablation and negative-result ledger

```powershell
python -m cnn_fpga.benchmark.causal_ablation_negative_results
```

输出 `docs/t5_4_3_causal_ablation_negative_results.json` 与 338-row Source Data。history 在 cutoff12/16
方向反转；legacy CNN residual 只改善单 test split 的参数 MSE；regime NLL 改善但检测延迟增加；run-length
在 32/32 cells 弱于 memoryless；parameter update 只建立 event-cost 组件结果；fallback aggregate 为正但
compound/nominal 为负。五类原生 metric 不生成总分，结果不建立 integrated system、physical-memory、device
或 hardware 机制贡献。

### T5.4.4 multi-agent / seed selection-bias audit

```powershell
python -m cnn_fpga.benchmark.multi_agent_seed_selection_audit
```

输出 `docs/t5_4_4_multi_agent_seed_selection_audit.json` 与 420-row Source Data。审计六个 selection
episodes、255 个 evaluation units 和 39 组 median/IQR/worst quartile；active candidates 均由 validation
选择，test 不重选。teacher restart 0 在两条 test lane 均不是 hindsight best，旧 T4.1.5 又缺非选中
restart test metrics，因此 verdict 为 `PASS_WITH_WARNINGS`，不支持 best-of-N、optimizer-optimal 或硬件 claim。

### T5.4.5 training-horizon / long-recurrence extrapolation

```powershell
python -m cnn_fpga.benchmark.horizon_extrapolation_validation
```

输出 `docs/t5_4_5_horizon_extrapolation_validation.json`、9-model checkpoint 与 521-row Source Data。
2/5/10-cycle candidates 各三 restart 且 validation-only 选择；32-cycle 使用 frozen production student。8 条
streams 在 float64/float32 下均真实执行到 `10^6` cycles，保留 2-cycle/all-e 外推失败；10/32-cycle
mean+worst action MSE、解析 state bound 和 120 个 reset recovery 通过。它不是长时 Fock logical channel、
physical gain、leakage、device 或 hardware 证据。

### T5.4.6 randomized multi-factor model mismatch

```powershell
python -m cnn_fpga.benchmark.randomized_model_mismatch --device cuda
```

输出 `docs/t5_4_6_randomized_model_mismatch.json` 与 273-row Source Data。物理 lane 在 cutoff12、
10 cycles、batch16、float64 下对 32 个 gate/dephasing/timing-dynamics/compound cells 执行
standard/teacher/student nominal--mismatch 配对；其余 lane 分别执行 8 个完整 4×3 readout matrices、
16 个 persistent leakage/reset cells 与 8 个 frozen-decoder random drift cells。19/19 gates 通过，
student retention median/Q1/min 为 `0.998101/0.990413/0.897630`；同时保留 gate-bias/compound
teacher worst degradation `0.424155/0.395654`。结论不是 absolute、长时、device 或 hardware robustness。

### T5.5.1 packed-word bit-accurate Python RTL golden

```powershell
python -m cnn_fpga.benchmark.bit_accurate_hardware_reference
```

输出正式 JSON、16,503-row Source Data、4,116-row hash-chained trace 与 13,724-byte binary bank。
58-bit input 经真实 5-stage MAP 与 1-stage output register，在严格 6 cycles 后产生 118-bit output；
232-bit state 保存 FSM/health counters。16,384 个 code 与独立 integer reconstruction 0 mismatch，cycle
4000 defer/4001 commit 且 in-flight v0/v1 顺序正确。16/16 gates 只冻结 Python RTL golden；未生成 RTL，
Fmax/LUT/FF/BRAM/DSP 和 board 字段保持 null/false。

### T-RISK-20260716-01 / T5.5.2 synthesizable RTL 与目标器件 estimate

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.benchmark.rtl_fast_path_equivalence
python -m cnn_fpga.benchmark.target_device_synthesis --run-tools
```

前一命令生成并编译 CXXRTL，对 fault/commit trace 和 v0/v1×两 phase×全部 1024 codes 做 full-word
逐周期对拍：4,316 valid MAP rows、0 mismatch；RTL 以八个 mirrored 1R1W memories 合法实现四个
逻辑 2R1W tables。后一命令对 `GW2AR-LV18QN88C8/I7` 运行 Yosys `synth_gowin` 和 seed
1/7/19 的 nextpnr P&R。三 seed Fmax minimum/median/maximum 为
修复 harness 配置地址越界并重跑后的 min/median/max 为 `39.7456/39.8661/40.4318 MHz`，
均通过 27 MHz；最大 LUT4/DFF/BSRAM 为 `3362/865/8`，另用 1×MULT18X18、1×MULT9X9。
6-cycle core 在 27 MHz 的 estimate 为
`222.222 ns`。这些是开源工具 target-device estimate，不是 vendor signoff、bitstream、transport 或板测。

### T5.5.3 precision-resource-performance Pareto

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.rtl.generate_student_fixed_memories
python -m cnn_fpga.benchmark.student_rtl_equivalence
python -m cnn_fpga.benchmark.precision_resource_pareto
```

联合矩阵为 4 precision×3 K×3 student dimensions×3 multiplier parallelism=108 rows。只有
`selected_p10_a8_q9_12__k4__d4__p1` 最终 eligible：K=4 是六场景最小收敛软件 reference，不在 FPGA；
4-state signed-Q3.14 student 以一个 time-multiplexed DSP 执行 64 cycles。CXXRTL 512 updates、
7,680 outputs 与 72-bit state 0 mismatch；integrated 三 seed Fmax minimum/median/maximum 为
修复共享 harness 地址并重跑后的 min/median/max 为 `39.5726/40.3226/40.5351 MHz`，最大资源为
3,802 LUT4、1,022 DFF、8 BSRAM、2×MULT18X18、1×MULT9X9。其余 107 rows 只标
calibrated estimate，不能写成综合结果。

### T5.5.4 full/quantized GRU versus student feasibility

```powershell
$env:PYTHONPATH='.'
python -m cnn_fpga.rtl.generate_quantized_gru_memories
python -m cnn_fpga.benchmark.gru_student_hardware_feasibility
```

完整 float32/float64 GRU parameters alone 连 core 后至少需 135/261 个 BSRAM，超过 target 46；量化
完整参数 lower-bound 实际占 41 BSRAM，修复 bias 顺序/越界与共享 harness 地址后，三 seed 最差 Fmax
`39.1527 MHz`，但 CXXRTL 与独立 bit-vector
reference 以 signature `730990968` 确认顺序消费全部
72,266 weights/587 biases 已需 72,854 cycles=`2698.30 us`@27MHz。该 RTL 明确不是 functional GRU，
fake-quantized shadow 也没有 physical gain evidence。因此 optional quantized-GRU route Dropped，唯一通过
functional/capacity/deadline/gain 四门的是 64-cycle 4-state student。

在同一 orthonormal finite-cutoff GKP code、噪声和 `10 us` cycle 上运行 nominal sBs QEC-on 与 idle
QEC-off。正式矩阵为 cutoff 12/24/36/40×high/medium/low×on/off；逐 cycle 保存六个 unnormalized
code outputs，并重构 CPTNI full PTM、Choi/TNI、non-Pauli/leakage 和三轴 raw-area/censored lifetime。
36→40 terminal repeat 通过数值稳定门，但低 cutoff 性能方向反转完整保留；结果不是 infinite-cutoff theorem、
experimental tomography、physical-memory LER、break-even 或硬件时序。

### P1 漂移仿真基准

```bash
python -m cnn_fpga.benchmark.run_drift_suite --config cnn_fpga/config/experiment_drift.yaml
```

### P2 硬件行为仿真

```bash
python -m cnn_fpga.benchmark.run_hardware_emulation --config cnn_fpga/config/hardware_emulation.yaml
```

### P3 HIL 基准

```bash
python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil.yaml
```

### P4 多场景正式基准

```bash
python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark \
    --config cnn_fpga/config/p4_multiscenario.yaml \
    --repeats 10
```

### P4 消融实验

```bash
# 运行 teacher 组消融（数据集 + 训练 + 基准）
python -m cnn_fpga.benchmark.run_p4_hybrid_vs_ukf_ablation \
    --group teacher --stage all

# 仅运行基准（跳过已完成的数据集/训练）
python -m cnn_fpga.benchmark.run_p4_hybrid_vs_ukf_ablation \
    --group features --stage benchmark --skip-existing
```

## 输出文件说明

| 文件 | 生成者 | 内容 |
|------|--------|------|
| `hil_summary.json` | HIL 类脚本 | 单次 HIL 运行汇总（LER, 溢出, 时序, 事件） |
| `comparison.csv` | 基准脚本 | 多模式 LER/溢出率对比表 |
| `delta.csv` | P4 基准 | 各模式 vs static_linear / cnn_fpga 差值 |
| `report.md` | 基准/消融 | Markdown 格式报告 |
| `summary.json` | 基准/消融 | 聚合 JSON 结果 |
| `teacher_scalar_diagnostics.csv` | P4 基准 | Teacher 标量特征诊断数据 |
| `trace_rows.csv` | 轨迹分析 | 逐窗口预测轨迹 |
| `docs/t6_19_3_secondary_evidence_integrity.json` | `secondary_evidence_integrity_gate.py` | 六 lane、206-cell、24-gate只读完整性终态 |
| `docs/t6_19_3_secondary_evidence_integrity_source_data.csv` | `secondary_evidence_integrity_gate.py` | 一行一cell的source/raw/config/hash/evidence/value-state ledger |
| `docs/figures/t6_19_3_secondary_comparison_atlas.*` | `secondary_evidence_integrity_gate.py` | Python-only SVG/PDF/600-dpi TIFF/PNG非主排名图谱 |
| `docs/t6_25_1_single_mode_rtl_boundary_audit.json` | `single_mode_rtl_boundary_audit.py` | single-mode 5-top live-hash/capability/实例化边界与 converged-production-top 缺口，不生成板测或 fastest claim |

## 关键依赖

- **runtime**: `SlowLoopRuntime`, `DualLoopScheduler`, `FastLoopEmulator`, `ParamBank`, `LatencyInjector`
- **hwio**: `FPGADriver`, `DMAReadout`
- **decoder**: 各类 baseline（EKF, UKF, ParticleFilter, WindowVariance）
- **physics**: `LinearDecoder`, 综合征测量, 逻辑错误追踪
- **utils**: YAML 配置加载, 路径管理, JSON 序列化
