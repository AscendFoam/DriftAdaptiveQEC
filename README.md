# DriftAdaptiveQEC

`DriftAdaptiveQEC` 是一个围绕 **contract-centric、regime-aware 安全自适应双回路 GKP 解码** 的研究型工程仓库：static/adaptive MAP 承担 LER，HMM/event/fallback 承担 tail safety，FPGA fast path 承担确定性执行，CNN/teacher/student 是需要通过 matched promotion gate 的可替换学习模块。当前代码覆盖物理仿真、统一 comparator、Tiny-CNN、量化/RTL、软件 HIL 与多场景 benchmark；真实板卡证据仍与预板软件/RTL 资格分轨管理。

## 当前状态

- 当前阶段结论与后续计划主要见：
  - `docs/02_experiment_plan.md`（Part I 为当前阶段结论入口，Part II 为后续开发计划唯一入口）
  - `docs/paper_notes/README.md`
- 历史研究背景与早期阶段材料可参考：
  - `docs/legacy_context/reference_retired_2026-06-11/CNN_FPGA_GKP_工程化实验方案.md`
  - `docs/progress_summary/CNN_FPGA_GKP_阶段结论.md`（已退役索引，历史全文在 `docs/legacy_context/progress_summary_retired_2026-06-11/`）
- 自 `2026-05-05` 起，项目治理以以下文件为准：
  - `docs/00_project_snapshot.md`
  - `docs/01_legacy_audit.md`
  - `docs/02_experiment_plan.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
- 自 `2026-07-14` 起，用户指定的“按新任务板顺序完整推进”工作流另以以下文件为当前执行状态源：
  - `docs/new_task_board.md`：新任务顺序、状态和当前推荐任务；
  - `docs/rough_plan.md`：新任务序列的冻结原始规划，仅作历史/设计来源，不随日常任务改写；
  - `docs/experiment_plan.md`：新任务序列的实验计划来源；第 14—16 节记录 v2—v2.2 的低频修订；
  - `docs/new_tasks/`：每个新 task 的输入、产物、验证、风险和同步记录；
  - `docs/new_risks.md`：新任务序列风险与插入任务判断。
  - 既有 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md` 和历史证据包继续作为可复用事实来源，但不能用旧日志替代新任务的当前产物与重新验证。
- 当前阶段：`Phase 2: Controlled Development`
- 当前决策状态：`Go`
  - 原因：截至 `2026-05-08`，治理文件、recovery-scoped manifest、最小 P0/P3/P4 路径与 bounded software HIL 的确定性表述都已收口到可稳定接力状态。
  - 边界：这个 `Go` 只代表“允许继续开发”，不代表 `real_board`、真实 `.tflite` runtime 或正式多场景 P4 benchmark 已恢复。

## 仓库结构

- `physics/`: GKP 物理层与逻辑错误追踪
  - `physics/ideal_gkp_decoder.py`: ideal 1D/2D standard、periodic MAP 与 correlated joint MAP reference
  - `physics/finite_energy_gkp.py`: normalized Gaussian-envelope / damped-projector 四逻辑态、syndrome 与 sampled-Wigner reference
  - `physics/quadrature_conventions.py`: canonical、decoder-standardized、symplectic bridge 与 displacement-amplitude 四坐标 chart 的辛性和方差/波函数变换合同
  - `physics/logical_channel.py`: parity-output decoder 的 Pauli-twirled logical channel、PTM 与 fidelity metrics
  - `physics/fock_logical_channel.py`: 六 Pauli eigenstate 的 finite-cutoff CPTNI code-subchannel tomography、non-Pauli/leakage 与 matched QEC-on/off lifetime diagnostics
  - `physics/logical_channel_fidelity.py`: CPTNI leakage-inclusive `F_e/F_avg`、六态 direct replay、无指数短时率与 cutoff/time-grid 数值敏感度
  - `physics/operational_boundary.py`: matched active/passive 完整 `F_avg` 曲线的持续非劣与累计偿还边界，不生成终点比值或 coherence-gain 冒名指标
  - `physics/qec_cost_accounting.py`: QEC 事件计数、`Delta`-to-dB 与 post-selection acceptance/rejection penalty 的单位安全成本恒等式
  - `physics/channel_recovery_bound.py`: encoding--noise QEC matrix、transpose/Petz recovery fidelity、解析双边界，以及 small-cutoff primal/dual SDP 可行证书
  - `physics/finite_energy_trends.py`: train/eval 分离的 finite-energy shrinkage trend reproduction harness
  - `physics/drift_processes.py`: 七类可复现 synthetic drift、完整 `DriftState`、mixture sampler 与旧回调适配层
  - `physics/oracle_map.py`: 逐时读取完整 `DriftState` 的不可部署 periodic Gaussian-mixture Bayes oracle
  - `physics/oracle_gap.py`: static/dual/oracle paired gap、未截断 closure、McNemar 与 ratio bootstrap 指标
  - `physics/finite_squeezing_noise.py`: channel/data-GKP/ancilla-measurement/envelope 分解的 finite-squeezing effective sampler、covariance budget 与 exact ideal endpoint
  - `physics/protocol_ancilla_errors.py`: sBs stage-resolved ancilla fault overlay 与独立 sharpen--trim 四轮 `+y/-y` readout/reset/leakage state machine
  - `physics/control_imperfections.py`: AWG/DAC code、pulse miscalibration、latency drift/diffusion、virtual rotation 与 active-displacement physical residual effective model
  - `physics/fock_density_model.py`: finite-cutoff approximate-GKP density matrix、显式 bosonic channels、modular measurement backaction 与 cutoff/CPTP diagnostics
  - `physics/noise_transfer_surrogate.py`: signal/fluctuation/logical-jump 分离的 Heisenberg-inspired 中保真代理，含 loss、measurement、gain、alias 与 clipping 有效性诊断
  - `physics/cross_fidelity_validation.py`: Fock/effective/noise-transfer/direct-syndrome 四 lane 的共同 q-axis 指标、原生 occupancy、cutoff 与失效归因验证
  - `physics/fock_sbs_cycle.py`: 论文 analytic SBS X/Z Kraus 的 finite-cutoff completed one-round reference、hidden/observed branches、classical residual action、Pauli frame 与 logical projection
  - `physics/differentiable_sbs_trajectory.py`: PyTorch joint cavity--ancilla、15 参数 `R/ECD/D/VR`、逐段 CPTP idle、随机/回放 `g/e` 与 causal history-policy 的可微短时域 simulator
  - `physics/feedback_grape_gradient.py`: trajectory 穷举、reward/score 两项梯度、分项 finite difference、step sweep 与 repeated Monte Carlo estimator 验证
  - `physics/differentiable_sbs_feasibility.py`: GRU10--256--256--15 causal policy 的真实 trajectory/reward-score/Adam step，多轴 cutoff/batch/horizon CPU/CUDA 资源与数值 frontier 扫描
  - `physics/plot_differentiable_sbs_feasibility.py`: 从机器 JSON/CSV 生成 T2.3.6 Python-only SVG/PDF/TIFF/PNG 可编辑多面板图
  - `physics/nmf_directional_ranking.py`: strict train/validation/test/confirmation split 的 paper-scale MF/NMF Feedback-GRAPE 训练、schema-v3 checkpoint/resume、10-cycle directional lifetime ranking 与 cutoff-16 confirmation
  - `physics/latest_outcome_markovian.py`: g/e/leakage latest-token-only、与 history GRU 精确同为 72,853 参数/72,266 dense MAC 的 15-output Markovian feedback policy
  - `physics/autonomous_sbs.py`: 复用显式 joint cavity--ancilla gates 的 protocol-native nonselective measurement-feedback/autonomous sBs 物理时间演化与原始事件账本
  - `physics/trajectory_lookup_control_oracle.py`: 两-cycle 15-node causal history lookup、exact 16-branch probability-weighted fidelity 优化与 `2^(2C)` 资源增长 contract
  - `physics/exponential_recurrence_control.py`: PRL-inspired `g/e/leakage` 指数饱和 15-control 因果递推、exact branch 优化与 Q 定点镜像
  - `physics/memory_specific_ablation.py`: 对冻结 NMF 权重执行 prefix-consistent shuffle、sliding truncation、periodic hidden reset 与 latest-only 因果干预
  - `physics/plot_nmf_directional_ranking.py`: 对 T2.3.7 raw evaluation/summary/source hash/checkpoint hash fail-closed 审计，并生成 183 mm 论文图与长表 Source Data
- `cnn_fpga/`: 数据、模型、解码器、运行时、HIL、benchmark 主模块
  - `cnn_fpga/benchmark/adaptive_drift_alignment.py`: 现有 Window/EKF 的同 trace、一窗因果延迟、paired oracle-gap 验收 harness
  - `cnn_fpga/benchmark/standard_binning_baseline.py`: observed-only fixed half-cell decision、hidden-truth paired evaluator 与主要 decoder comparison schema guard
  - `cnn_fpga/benchmark/static_map_baseline.py`: evaluation-independent training-state moment match、冻结 periodic Gaussian MAP 与 8-seed paired validation
  - `cnn_fpga/benchmark/oracle_baseline.py`: exact-state/regime model-oracle schema、truth-only leakage flag 与显式 recovery-cost envelope
  - `cnn_fpga/benchmark/static_protocol_decoder.py`: exact stationary hidden-carry marginalization、sBs `g/e/leakage` posterior LUT 与显式 leakage fallback-cost policy
  - `cnn_fpga/benchmark/topk_lattice_coset_map.py`: single-mode 四逻辑陪集 joint alias top-K 累计、full periodic MAP 对照与确定性 operation/storage proxy
  - `cnn_fpga/benchmark/memory_assisted_bayesian_decoder.py`: 20-cycle observed-only joint periodic Bayesian posterior、same-prior final-outcome static comparator 与 logical-coset 末端决策
  - `cnn_fpga/benchmark/run_length_fsm_baseline.py`: training-only 非退化 run-length 阈值选择、same-trace static/memoryless/truth-oracle 事件代价比较与真实 `ParamBank` 冲突探针
  - `cnn_fpga/benchmark/regime_hmm_baseline.py`: 四态 synthetic regime trajectories、validation-only HMM calibration、same-emission memoryless ablation 与 future-CNN shared-budget audit
  - `cnn_fpga/benchmark/latest_outcome_markovian_baseline.py`: 复用 T2.3.7 frozen NMF，在同训练/动作/参数/MAC/trace 下训练五个 exact-budget MF 并保留 cutoff 排序反转
  - `cnn_fpga/benchmark/autonomous_sbs_wallclock_baseline.py`: cutoff12/16×三噪声下推进 7/10 us 原生 cycle 到共同 700 us，比较 per-cycle/per-us lifetime 与 measurement/reset/gate 计数
  - `cnn_fpga/benchmark/trajectory_lookup_control_oracle.py`: time-indexed open-loop/causal lookup 各 3-restart 两阶段优化、cutoff16 frozen transfer、checkpoint/Source Data/gate 编排
  - `cnn_fpga/benchmark/exponential_recurrence_baseline.py`: 指数递推 3-restart exact 物理优化、lookup/standard 对照，以及与 run-length FSM 同轨但分 metric-domain 的事件代价验证
  - `cnn_fpga/benchmark/memory_specific_ablation.py`: 复用 5 个 NMF 与 5 个同预算重训 MF checkpoint，在 cutoff12/16 闭环执行四类 memory-specific 消融与 paired bootstrap
  - `cnn_fpga/benchmark/slow_loop_model_selection.py`: 在共同 8-window four-regime task 与 4096 MAC/4096 B envelope 下，validation-only 比较 TCN/GRU/HMM/Kalman/指数递推/FSM
  - `cnn_fpga/benchmark/experimental_history_validation.py`: 以真实 syndrome/FSM/LLR/scheduler producer 做 8×2,048-cycle history schema、泄漏和故障覆盖验证
  - `cnn_fpga/benchmark/hybrid_state_output_validation.py`: 恢复 T4.1.1 HMM 并在 nominal/stress lane 验证 future hybrid output、block-bootstrap uncertainty 与 atomic bank recommendation
  - `cnn_fpga/benchmark/bounded_residual_rnn_teacher.py`: 3 个全新 72,853 参数 GRU restart、nominal-plus-hard-bounded 15 residual、validation-only 选模和 cutoff12/16 held-out teacher 证据
  - `cnn_fpga/benchmark/bounded_residual_teacher_analysis.py`: frozen teacher 的 g/e hidden/control、conditional p(g)、PCA/指数饱和、impulse/Jacobian memory 与 leakage OOD proxy 分析
  - `cnn_fpga/benchmark/low_dimensional_student_distillation.py` / `cnn_fpga/control/low_dimensional_recurrence.py`: 1/2/4-state×3-restart strict-split 指数递推蒸馏、validation-only 选维和 pure-NumPy fail-closed online artifact
  - `cnn_fpga/benchmark/teacher_student_gain_retention.py`: 全新 paired seeds 的 10-cycle teacher/student physical retention、全部五个 MF agents、显式 burden/cost 与独立 2-cycle exact control-oracle lane
  - `cnn_fpga/benchmark/teacher_student_branch_freeze.py`: hash-bound 只读消费 T4.4.1--T4.4.4，机器冻结 qualified student-retention 或 MAP-LUT fallback，并登记禁止 claim/撤销 gate
  - `cnn_fpga/benchmark/causal_ablation_negative_results.py`: history/CNN residual/regime/run-length/update/fallback 六项 native-lane mechanism-off 对照、负结果表与禁止跨 metric 拼榜合同
  - `cnn_fpga/benchmark/multi_agent_seed_selection_audit.py`: 六个 learned selection episodes 的 validation-only 重构、全 agent/restart/seed 分布与 test-best hindsight 偏差诊断
  - `cnn_fpga/benchmark/horizon_extrapolation_validation.py`: 2/5/10/32-cycle training-horizon sweep、`1e3/1e5/1e6` cycle 全步 GRU/student 递归、float32 shadow 与 reset sensitivity 审计
  - `cnn_fpga/benchmark/randomized_model_mismatch.py`: 64-cell 四原生 lane 随机失配验证；覆盖 full gate-bias vector、dephasing/timing/dynamics、完整 4×3 readout、leakage/reset 与 frozen-decoder drift
  - `cnn_fpga/benchmark/bit_accurate_hardware_reference.py` / `cnn_fpga/runtime/bit_accurate_hardware_reference.py`: 58/118/232-bit CRC words、真实 5+1-cycle pipeline、binary parameter image、atomic A/B switch 与 hash-chained RTL golden trace
  - `cnn_fpga/rtl/` / `cnn_fpga/benchmark/rtl_fast_path_equivalence.py` / `target_device_synthesis.py`: 可综合 fast-path RTL、CXXRTL full-word 对拍、Tang Nano 20K 三 seed synthesis/P&R 与 fail-closed 报告生成
  - `cnn_fpga/benchmark/precision_resource_pareto.py` / `student_rtl_equivalence.py`: 108 点 precision/K/state/parallelism 审计、4-state Q3.14 student CXXRTL 与 integrated 三 seed P&R；估计值和实际工具报告严格分列
  - `cnn_fpga/benchmark/gru_student_hardware_feasibility.py`: full-float storage fail-fast、完整参数 int8/Q3.14 functional shadow 与 lower-bound CXXRTL/P&R；量化 GRU fail-closed，4-state student 为唯一硬件路线
  - `cnn_fpga/benchmark/long_rtl_qualification.py` / `cnn_fpga/runtime/fast_production_core_reference.py`: T6.2.2 十类各 100,000-cycle 独立整数 golden/CXXRTL 全字段资格验证、抽象 bounded FIFO/receiver faults、恢复与 mutation audit
  - `cnn_fpga/benchmark/unified_comparator_runner.py`: T6.6.1 scalar phase packet→causal q/p bridge、六方法 common-trace 真实执行、legacy CNN schema/budget 自动降级、oracle 分栏、维度推导成本与 prefix/mutation qualification
  - `cnn_fpga/runtime/regime_aware_safe_policy.py`: T6.6.2/V4 强类型安全编排；Window/EWMA observed-only 双影子 bank、prequential promotion proof、tail/uncertain trusted EWMA、leakage/reset 与 integrity-only monotonic LKG rollback 共用原子 bank，并逐 action 输出 posterior/reason/version/deadline provenance
  - `cnn_fpga/benchmark/regime_aware_safe_policy_validation.py`: T6.6.2 production-cadence 结构长轨；验证 5+1-cycle ledger、完整双影子预算、tail EWMA promotion、hysteresis/residency、pending Window rejection、LKG commit、reset 与故障 fail-closed，不把 fixture 冒充已校准 HMM 或 LER 证据
  - `cnn_fpga/decoder/route_a_regime_posterior.py`: T6.6.3 Route-A 专用 `normal/smooth/calibration_shift/burst` causal Gaussian HMM 与 observed heavy-tail event head；class order、模型 payload 和推理 hash 可复核
  - `cnn_fpga/benchmark/route_a_posterior_calibration.py`: T6.6.3 calibration/pilot-only posterior、temperature、1,728 common threshold、full pilot LER/safety selector 与 causal Window/EWMA router 锁；保存 V2 static-switch/V3 freeze-all NO-GO，formal artifact 存在时拒绝重跑
  - `cnn_fpga/benchmark/route_a_causal_headroom.py`: T6.10.1 对 1,464 条 V4 formal 做 diagnostic-only 五专家逐 decision exact replay，并在 186 条全新 development trajectories 上执行 nested strict-causal selector、posterior-mixture/action-space、regret、预算和 mutation audit；当前诚实结论为 V5 early NO-GO
  - `cnn_fpga/benchmark/route_a_v5_final_evidence_gate.py`: T6.15.5 Phase 6B early-stop 终态门；重算 router/action 增量、验证 20 个 Dropped tasks 与零 V5 downstream outputs，撤销 V5 performance/formal/RTL/P&R claims，并只读开放 Phase 6C
  - `cnn_fpga/benchmark/official_structured_cpd_reproduction.py`: T6.18.2 official structured-lattice CPD 复现后处理与完整性门；绑定官方 Julia commit/manifest、exact-CVP checks、作者聚合重算、独立小距离 crossing、runtime/memory 和 17 类 semantic mutation
  - `cnn_fpga/benchmark/multimode_posterior_weighted_cpd.py`: T6.18.3 只读汇总器；对 9.6M-cycle official-validated d=3 structured family 重算 observed-only posterior-weighted CPD 的 paired LER、512-cycle tail、lag、runtime/memory、双侧 Holm 与 21 类 fail-closed mutation，oracle 永不进入 deployable ranking
  - `cnn_fpga/benchmark/phase6c_preboard_profiles.py`: T6.19.1 项目原生预板与软件慢路径 profile；重新绑定当前 RTL 的 4,316-row CXXRTL 与 GW2AR 三种子 P&R，只让 static MAP-LUT fast-path 进入硬件表，CI/V5/Direct-NN 保持 N/A；Window/EWMA/Kalman 的 1000-repeat host stages 分表且禁止冒充 FPGA/板测
  - `cnn_fpga/decoder/regime_hmm.py`: 32×8 observed-window featurizer、full-covariance Gaussian emissions 与严格 causal four-state forward posterior
  - `cnn_fpga/decoder/slow_loop_model_selection.py`: 六族统一 bounded-history adapter、small TCN/GRU、classical heads、rolling-HMM cache 与可审计资源画像
  - `cnn_fpga/decoder/hybrid_state_output.py`: continuous/regime/leakage/recovery-burden/uncertainty 的 future-only output 与 version/validity/CRC-bound bank proposal
  - `cnn_fpga/data/experimental_history.py`: 256×53 observed-only causal history、mask/cycle alignment、LLR/action provenance、scheduler status 与 truth-leak fail-closed audit
  - `cnn_fpga/runtime/run_length_fsm.py`: 3-bit 饱和 event FSM、phase tie-break、五态 local-safe action 与双 bank 原子同步
  - `cnn_fpga/runtime/exponential_recurrence.py`: 三状态指数证据核、恢复/leakage 滞回、真实 ParamBank 切换与 Q4.16/Q2.18 整数执行镜像
  - `cnn_fpga/runtime/fixed_point_chain.py`: ADC/replay、LLR LUT、threshold、wrapped state、update granularity 和 double-bank bit fault 的 paired precision--resource-proxy--LER 扫描
- `benchmark/`: P0 基础对比脚本
- `docs/`: 方案、阶段结论、恢复治理文件
  - `docs/codebase_overview/`: `physics/` 与 `cnn_fpga/` 代码阅读辅助文档
  - `docs/recovery_bootstrap/`: P0/P3/P4 recovery smoke 复用入口
  - `docs/protocols/`: benchmark / execution protocol 文档
  - `docs/evidence_packs/`: 已完成任务的证据包、gate 输出和边界说明
  - `docs/paper_materials/`: 论文材料、claim/evidence ledger、草稿骨架和风险审计
  - `docs/literature_matrix.md`: 当前新任务序列的四线文献、Zotero 覆盖和证据等级矩阵
  - `docs/gap_statement.md`: 基于四线证据的研究缺口、baseline/oracle contract 与英文 Introduction 草稿
  - `docs/claim_ladder.md` / `docs/claim_ladder.json`: 第一篇论文五层 claim-evidence 升级契约、逐 claim 措辞与机器可读 gate
  - `docs/low_cost_fpga_boundary.md` / `docs/low_cost_fpga_boundary.json`: Tang Nano 20K 低成本参考目标、资源/I/O、串行适配缺口和数字控制平面测量边界
  - `docs/two_domains_three_timescales.md` / `docs/two_domains_three_timescales.json`: FPGA/host 两计算域、fast/event/slow 三时间尺度、原子更新和失败分支契约
  - `docs/paper_parameter_registry.md` / `docs/paper_parameter_registry.json`: 核心 GKP 实验/理论参数的一手事实、secondary reference、项目假设和待校准 gate registry
  - `docs/decoder_controller_terminology.md` / `docs/decoder_controller_terminology.json`: decoder/controller、两类 oracle、recovery bound、teacher/student、host estimator 与 FPGA fast path 的术语和实现状态合同
  - `docs/protocol_hierarchy.md` / `docs/protocol_hierarchy.json`: sBs 主数字孪生、sharpen--trim 交叉验证、secondary protocol 的周期/观测/动作/不可模拟项与禁止混用合同
  - `docs/literature_trend_reproduction_table.md` / `docs/t5_0_1_literature_trend_reproduction.json`: 14 行文献趋势复现注册表，区分主线/secondary、数值/方向容差、calibration/holdout 和 pending/reference 边界
  - `docs/independent_cross_fidelity_holdout.md` / `docs/t5_0_2_independent_cross_fidelity_holdout.json`: calibration/pilot 完全隔离的 T5.0.2 独立 holdout；显式保留 main cross-fidelity FAIL 与 secondary P-Steane PASS
  - `docs/comparison_set_registry.md` / `docs/t5_1_1_comparison_set_registry.json`: 19-comparator、8-lane 的完整 comparison set，冻结信息/协议/时间/算力与 oracle/secondary 禁止混排规则
  - `docs/secondary_method_source_audit.md` / `docs/t6_16_1_secondary_method_source_audit.json`: 两张异构方法图的 11-source/12-method 一手审计；逐项冻结 decision object、观测/动作/权限、metric denominator、latency boundary、资源和 null/negative 边界，不生成跨 lane 排名
  - `docs/comparison_metric_ontology.md` / `docs/t6_16_2_comparison_ontology.json`: Phase 6C 六 lane、46 metric、六 timing boundary、五资源维度和八种 value-state 的 fail-closed ontology；只有 13-field task signature 完全一致才允许比较
  - `docs/secondary_experiment_preregistration.md` / `docs/t6_16_3_secondary_preregistration.json`: Phase 6C 九项 secondary experiment 的 21-field 预注册、独立 seed/CRN/统计/停止与失败合同；以 live semantic/exact locks 保证外部补充比较不能改写 Phase 6B `10%/12%` 门、tail 未运行、Dropped/absence 或板测 null
  - `docs/project_preboard_profiles.md` / `docs/t6_19_1_project_preboard_profiles.json` / Source Data: static MAP-LUT 完整 fast-path core 的 6-cycle/II=1、三种子目标器件 estimate，以及 Window/EWMA/Kalman 当前主机 update/compiler/软件事务阶段；资源不是 MAP ROM 单体面积，power/jitter/deadline/measured/physical transfer 均为 null
  - `docs/single_mode_cpd_equivalence.md` / `docs/t6_17_1_single_mode_cpd_equivalence.json`: single-mode square/isotropic Euclidean CPD=CI 的解析与穷举边界；完整 q10×q10 加 100 万 unwrapped boundary points 均 0 mismatch，并以 biased/correlated/finite-energy likelihood 反例证明 coset MAP 不是同一 comparator
  - `docs/noh_cnot_ci_ml_reproduction.md` / `docs/t6_17_2_noh_cnot_ci_ml_reproduction.json`: Noh 2022 Table-I 双 square-GKP CNOT 的 project-native matched 复现；8-shift 一手模型、32-seed CRN、9/12/13 dB raw counts/paired statistics、六锚点 discrepancy 与 100k true-facet likelihood oracle，outer-code 9.9 dB/latency/resources 继续为 null
  - `docs/learned_model_eligibility_replay.md` / `docs/t6_17_3_learned_model_eligibility_replay.json`: 16 个 Direct NN、causal estimator、learned/controller student、RL/NMF family 的 13-field signature 与 7-field budget 只读资格审计；legacy TinyCNN 在 206 样本上五次 bit-exact 重推，但 same-task decoder eligible=0，LER/latency 排名保持 null
  - `docs/aqec_common_wallclock_replay.md` / `docs/t6_18_1_aqec_common_wallclock_replay.json`: idle/measurement-reset/autonomous 三 anchor 的 6-cell×24-cluster exact finite-cutoff common-700-us replay；144,152-row raw ledger、20k paired bootstrap与144/144 cycle-vs-us ordering reversal，保留 active-QEC 弱于 idle 的负结果，Lachance reservoir结果仍为 literature-only/official-blocked
  - `docs/official_structured_cpd_reproduction.md` / `docs/t6_18_2_official_structured_cpd_reproduction.json` / Source Data: 固定 `third_party/LatticeAlgorithms.jl@01f9bf1f...`、Apache-2.0 与 `configs/literature/t6_18_2_julia_env/`；312+64+384 个 exact checks 0 mismatch，官方聚合阈值逐位重算，1,728,000-trial d=3/5/7 coarse crossing 与小距离次序反转分列披露
  - `docs/multimode_posterior_weighted_cpd.md` / `docs/t6_18_3_multimode_posterior_weighted_cpd.json` / Source Data: d=3 balanced heteroscedastic smooth/calibration/telegraph 的32-seed、9.6M-cycle project-native GO；adaptive/static-Euclidean `p_L=0.172261/0.236929`，absolute gain=`0.064668 [0.064413,0.064926]`，但严格禁止外推 unseen device、general multimode、SOTA、FPGA 或 Phase6B claim
  - `docs/mixed_scenario_matrix.md` / `docs/t5_1_2_mixed_scenario_matrix.json`: 10 类 mixed noise/regime 的 lane-local production matrix；36 个 decoder seed-cluster 与 loss/fault/component 原生 gates，禁止跨 lane 拼榜
  - `docs/oracle_gap_tail_report.md` / `docs/t5_1_3_oracle_gap_tail_report.json`: 1,152-window average/p95/worst、paired decoder-oracle gap、20k seed-cluster bootstrap、24-test Holm family 与独立 exact two-cycle control-reference gap
  - `docs/algorithm_success_falsification.md` / `docs/t5_1_4_algorithm_branch_verdict.json`: fail-closed 算法成功/证否门；当前强 learned-decoder 分支失败并自动转入 event-aware adaptive MAP/FPGA co-design，不保留 CNN 性能主张
  - `docs/time_cost_fairness.md` / `docs/t5_1_5_time_cost_fairness.json`: protocol/controller/host-estimator 三条公平 lanes；同时报告 cycle/μs、measurement/reset/gates、analytic cost 与 latency evidence/null，不生成跨 lane 总排名
  - `docs/experimental_feasibility.md` / `docs/t5_1_6_experimental_feasibility.json`: controller p(g)/p(e)、reset/slew/cost 与软件 fault fallback/unsafe burden 的 fail-closed 汇总；七项缺失 evidence 保持 null/MISSING，deployment readiness 未建立
  - `docs/displacement_large_error_causal.md` / `docs/t5_2_1_displacement_large_error_causal.json`: 17 幅度×8 seed-cluster 的独立 causal displacement campaign；分列 recovery/e-run、nearest-operation logical failure 和 identity-reference flip，不冒充 physical-memory LER
  - `docs/ancilla_readout_causal.md` / `docs/t5_2_2_ancilla_readout_causal.json`: bit-only、phase-only、readout-only 三条互斥 sBs effective causal lanes；6 rate×8 seed clusters、全交叉负控和 whole-seed CI，不复现实验 65× 或 device LER
  - `docs/leakage_reset_causal.md` / `docs/t5_2_3_leakage_reset_causal.json`: leakage injection 与 reset-failure 两条独立 causal lanes；96 个 seed cells、2,508-row Source Data，同报 detection/false alarm、occupancy/correlation tail、availability 与 raw reset cost，并保留 leakage-free 的 null 语义
  - `docs/logical_channel_reconstruction.md` / `docs/t5_3_1_logical_channel_reconstruction.json`: cutoff 12/24/36/40×三噪声×QEC on/off 的六态 CPTNI logical-channel reconstruction；17,266-row Source Data，同报 full PTM、Choi/TNI、non-Pauli/leakage 与 cycle/wall-clock lifetime
  - `docs/logical_channel_fidelity.md` / `docs/t5_3_2_logical_channel_fidelity.json`: 从六态 raw outputs 重算 leakage-inclusive `F_e/F_avg` 与无指数短时率；5,294-row Source Data，主动 lane 的 cycle-scale transient 明确保留为不合格寿命
  - `docs/logical_operational_boundary.md` / `docs/t5_3_3_logical_operational_boundary.json`: 12 个 full-curve matched comparisons 的 wall-clock operational boundary；416-row Source Data，coherence gain/full-cost/experimental break-even 保持未建立
  - `docs/qec_postselection_cost.md` / `docs/t5_3_4_qec_postselection_cost.json`: online QEC、offline post-selection、software safety 与 12 个 missing fields 的隔离成本账本；94-row Source Data，不生成跨 lane 总分或 full-cost break-even
  - `docs/qec_channel_recovery_bound.md` / `docs/t5_3_5_qec_channel_recovery_bound.json`: 15 条 small-cutoff Petz/SDP 双证书、cutoff 48 与三档能量扩展、actual sBs 时序失配诊断及 teacher/student 不可比审计；119-row Source Data，不把 arbitrary recovery 写成 decoder/controller
  - `docs/held_out_ood_validation.md` / `docs/t5_4_1_held_out_ood_validation.json`: frozen decoder、sBs confusion/leakage 与 software scheduler 的四条预注册 OOD lanes；104 seed cells、280-row Source Data，保留 telegraph reversal 与 short-pause null，不升级系统/装置稳健性
  - `docs/uncertainty_gated_fallback.md` / `docs/t5_4_2_uncertainty_gated_fallback.json`: observed-only EWMA→static matched gate；12 个 fresh confirmation clusters、517-row Source Data，聚合微弱获益但保留 compound 显著退化与 nominal 代价
  - `docs/causal_ablation_negative_results.md` / `docs/t5_4_3_causal_ablation_negative_results.json`: 六项 native-lane 因果消融与 claim 降级表；338-row Source Data，保留 history/run-length 负结果、regime delay 与 fallback 场景反转
  - `docs/multi_agent_seed_selection_audit.md` / `docs/t5_4_4_multi_agent_seed_selection_audit.json`: 6 个 selection episodes、255 evaluation units、39 组 median/IQR/worst-quartile 与 420-row Source Data；保留 teacher test-ranking reversal 和 legacy coverage warning
  - `docs/quadrature_normalization_contract.md` / `docs/t_risk_20260714_01_quadrature_validation.json`: GKP 四坐标 chart、Fourier reciprocal-lattice 审计、机器验证门与 legacy 失效对照
  - `docs/sbs_error_space_model.md`: `K_gg/K_ge/K_eg/K_ee` grouped CPTP instrument、`C_i` trickle-down 与 Pauli-frame 的实现/验证边界
  - `docs/sbs_observation_reset_model.md`: ideal Kraus、hidden `g/e/f/higher`、observed `g/e/leakage`、conditional reset 与 e/leakage runs 的分层模型
  - `docs/sbs_cycle_state_machine.md`: Table S3 的 18-phase constituent FSM、X→Z full-cycle 时间轴、branch/VR 接线与非实板时序边界
  - `docs/sbs_displacement_fault_trend.md` / `docs/t2_0_5_displacement_fault_trend.json` / `.csv`: 位移到最近逻辑操作距离的非单调 recovery-depth/同象限 e-run 趋势、预注册 seed/容差与失败诊断
  - `docs/sbs_occupancy_correlation_validation.md` / `docs/t2_0_6_occupancy_correlation.json` / `docs/t2_0_6_correlation_tail.csv`: hidden 与 syndrome-only occupancy 双估计、连续 leakage post-selection 和 long-lag correlation 收缩门
  - `docs/syndrome_stream_model.md`: 完整 `DriftState` 驱动的 causal mixed-state syndrome stream、observed/truth schema 隔离、recovery/leakage/logical 语义与统计验证边界
  - `docs/multiround_control_memory.md`: observed-only nearest-lift 多轮 memory，统一 residual/correction/confidence/frame/run/parameter-bank/deadline state 与未实现 fallback 边界
  - `docs/fast_monte_carlo_validation.md` / `docs/t2_1_3_fast_monte_carlo.json`: trajectory 向量化百万周期模拟、cluster CI 与 target-weighted burst/leakage rare strata 的验证和生产结果
  - `docs/finite_squeezing_effective_model.md` / `docs/t2_2_1_finite_squeezing_validation.json`: 分解式 finite-squeezing effective noise、非高斯 envelope contribution 与 6 点 high-squeezing limit 验证
  - `docs/protocol_ancilla_errors.md` / `docs/t2_2_2_protocol_ancilla_validation.json`: sBs/sharpen--trim 协议原生 ancilla bit/phase、readout、reset、leakage 信息流和 secondary non-execution 验证
  - `docs/control_imperfections.md` / `docs/t2_2_3_control_imperfection_validation.json`: request→AWG/DAC→pulse/latency/virtual-rotation→physical residual 的因果控制误差与解析矩验证
  - `docs/fock_density_model.md` / `docs/t2_3_1_fock_density_validation.json`: 独立 finite-Fock 态投影、loss/thermal/phase/Kerr/measurement/leakage-proxy 通道与截断收敛验证
  - `docs/fock_sbs_cycle.md` / `docs/t2_3_2_fock_sbs_cycle_validation.json`: canonical-coordinate analytic SBS 一轮、raw cutoff defect/CPTP completion、exact branch、photon-error 回泵与五点 cutoff 验证
  - `docs/noise_transfer_surrogate.md` / `docs/t2_3_8_noise_transfer_validation.json`: noise-transfer 代理的公式、3--12 dB 有效区/失效区、独立 MC 与 state/Fock q-domain 对齐
  - `docs/cross_fidelity_validation.md` / `docs/t2_3_3_cross_fidelity_validation.json`: 3--12 dB 四 lane LER/occupancy/`F_avg` 趋势、Fock cutoff 和 q/p 坐标失配归因
  - `docs/differentiable_sbs_trajectory.md` / `docs/t2_3_4_differentiable_trajectory_validation*.json`: 15 参数 joint cavity--ancilla 可微轨迹、Table S1 timing、history-policy、CPU/CUDA 物理性与资源画像
  - `docs/feedback_grape_gradient_validation.md` / `docs/t2_3_5_feedback_grape_gradient_validation.json`: Feedback-GRAPE reward/score 梯度分解、分项差分、baseline 与随机估计器证据
  - `docs/differentiable_sbs_feasibility.md` / `docs/t2_3_6_differentiable_sbs_feasibility.json` / `.csv`: 65 点 recurrent Adam training-kernel 的 cutoff/batch/2--10-cycle CPU/CUDA envelope、memory/runtime frontier 与 claim 边界
  - `docs/figures/t2_3_6_differentiable_sbs_feasibility.{svg,pdf,tiff,png}`: T2.3.6 source-traceable publication figure bundle
  - `docs/nmf_directional_ranking.md` / `docs/t2_3_7_nmf_directional_ranking.json` / `.pt` / `.csv`: 5+5 全 agent 的 strict-split MF/NMF 方向性 ranking、history-reset 反证、cutoff-16 confirmation、schema-v3 checkpoint 与 8,450-row Source Data
  - `docs/figures/t2_3_7_nmf_directional_ranking.{svg,pdf,tiff,png}`: T2.3.7 editable-text publication figure bundle；不等同于论文 1000-cycle 六态 lifetime 或硬件证据
  - `docs/dual_latency_budget.md` / `docs/dual_latency_budget.json` / `docs/t2_4_1_dual_latency_budget_validation.json`: 外部文献 measurement/ADC/control/DAC 时间轴与项目 UART/replay/FPGA/action 配置、容量下界、未测 `null` 字段的不可混用双预算和 23-gate hash-bound 审计
  - `docs/timing_fault_model.md` / `docs/t2_4_2_timing_fault_validation.json` / `.csv`: 基于真实 scheduler/parameter-bank 的 7 场景、8 paired-seed backlog/jitter/deadline/burst/pause/conflict/FIFO stress、LER 与三层 availability 证据；明确不是目标板实测
  - `docs/fixed_point_chain.md` / `docs/t2_4_3_fixed_point_validation.json` / `docs/t2_4_3_precision_resource_ler.csv`: 6-axis 位级 OAT、5 个 joint profiles、4 类 bank faults 的 368-run precision--representation--LER 证据；LUT/BRAM/DSP/Fmax 保持未综合
  - `docs/figures/t2_4_3_precision_resource_ler.{svg,pdf,tiff,png}`: Python-only、183 mm、paired-CI 与 Source Data/hash 绑定的 T2.4.3 科研图
  - `docs/standard_binning_baseline.md` / `docs/t3_1_1_standard_binning_validation.json`: no-tuning standard-binning 行、当前/未来主要 decoder comparison 注册表、72k paired counterevidence 与源码绑定审计
  - `docs/static_map_baseline.md` / `docs/t3_1_2_static_map_validation.json` / `.csv`: 训练/评测隔离的 static MAP、total-covariance 参数、8-seed 576k paired Source Data 与旧 adaptive gate 的证据降级
  - `docs/oracle_baseline.md` / `docs/t3_1_3_oracle_validation.json` / `.csv`: 4-regime 320k nondeployable model oracle、8k-cycle hidden leakage flag/cost envelope 与 oracle alias 边界
  - `docs/static_protocol_decoder.md` / `docs/t3_1_4_static_protocol_decoder_validation.json` / `.csv`: 640k-cycle observed-only sBs branch Bayes baseline、exact likelihood/Markov calibration、fallback cost 与 branch/logical target 分离
  - `docs/topk_lattice_coset_map.md` / `docs/t3_1_5_topk_map_validation.json` / `.csv`: 288k-sample K=1--128 periodic-MAP approximation、LLR/LER convergence、Source Data 与未综合成本边界
  - `docs/memory_assisted_bayesian_decoder.md` / `docs/t3_2_1_memory_bayesian_validation.json` / `.csv`: 4,096-episode causal-history Bayesian baseline、Student-t seed-cluster CI、128/256 grid convergence 与未综合成本边界
  - `docs/paper_readers/wan_memory_assisted_2020/`: Wan et al. 2020 主来源的 T3.2.1 task-scoped reader、图卡、source map 与显式未翻译范围
  - `docs/continuous_adaptive_map.md` / `docs/t3_2_2_continuous_adaptive_map_validation.json` / `.csv`: 4 类连续漂移下 full-covariance latest-window/EWMA/Kalman adaptive periodic MAP、157 万 paired samples、training-only 调参与未综合成本边界
  - `docs/sliding_window_syndrome_estimator.md` / `docs/t3_2_3_sliding_window_validation.json` / `.csv`: 同 384/1 observation-update budget 下 384--1536 uniform sliding-window 扫描、增量圆特征、training-only 384 边界选择与长窗负结果
  - `docs/postselection_diagnostic.md` / `docs/t3_2_4_postselection_validation.json` / `.csv`: training-only posterior-risk survival curve、observed/random/truth-upper 三 lane、rejection-penalty cost 与不可进入在线主增益边界
  - `docs/run_length_fsm_baseline.md` / `docs/t3_2_5_run_length_fsm_validation.json` / `.csv`: observed-only 五态 run-length FSM、真实参数银行、24-grid/384k paired event-cost 验证及弱于 memoryless 的负结果边界
  - `docs/regime_hmm_baseline.md` / `docs/t3_2_6_regime_hmm_validation.json` / `.csv`: normal/burst/leakage/calibration-shift causal posterior、same-emission temporal ablation、4096-window Source Data 与未来 CNN 共享预算边界
  - `docs/latest_outcome_markovian_baseline.md` / `docs/t3_2_7_latest_outcome_markovian_validation.json` / `.pt` / `.csv`: 5-agent exact-budget latest-outcome MF、同 trace frozen NMF 对照、18,023-row Source Data 与 cutoff-dependent memory 负结果
  - `docs/autonomous_sbs_wallclock_baseline.md` / `docs/t3_2_8_autonomous_sbs_wallclock_validation.json` / `.csv`: protocol-native autonomous/measurement-feedback 共同 wall-clock 比较、4,362-row Source Data、原始事件成本与单位排序反转
  - `docs/trajectory_lookup_control_oracle.md` / `docs/t3_2_9_trajectory_lookup_control_oracle.json` / `.pt` / `.csv`: 15-node/225-scalar finite-horizon control reference、3,418-row Source Data、branch-skew 与指数部署禁区
  - `docs/exponential_recurrence_baseline.md` / `docs/t3_2_10_exponential_recurrence_validation.json` / `.pt` / `.csv`: 75-trainable/105-stored 指数递推、1,888-row 双 metric-domain Source Data、cutoff transfer 与定点/事件比较
  - `docs/memory_specific_ablation.md` / `docs/t3_2_11_memory_specific_ablation_validation.json` / `.csv`: 5-agent×双 cutoff 四类 history intervention、28,230-row 曲线数据与跨 cutoff memory-mechanism 证否
  - `docs/slow_loop_model_selection.md` / `docs/t4_1_1_slow_loop_model_selection_validation.json` / `.pt` / `.csv`: 六族匹配预算选型、5-restart neural checkpoints、24,240-row Source Data 与 rolling-HMM 资源审计
  - `docs/experimental_history_input.md` / `docs/t4_1_2_experimental_history_validation.json` / `.csv`: 256×53 observed-only schema、真实 producer 对齐、17-gate leakage/fault/saturation audit 与 16,384-row Source Data
  - `docs/hybrid_state_output.md` / `docs/t4_1_3_hybrid_state_output_validation.json` / `.csv`: future hybrid state、T4.1.1 HMM bridge、block-bootstrap uncertainty、58 次 atomic commit 与 456-row Source Data
  - `docs/hybrid_multiobjective_calibration.md` / `docs/t4_1_4_hybrid_multiobjective_validation.json` / `.csv`: 六项 loss、3/2/3 strict split、proper-score calibration、448-row future targets 与 fallback 无选择性的负结果
  - `docs/offline_teacher_student_distillation.md` / `docs/t4_1_5_teacher_student_validation.json` / `.csv` / `docs/t4_1_5_distilled_student_checkpoint.json`: 5-agent frozen offline teacher、75-trainable/105-scalar recurrence student、15,360-row strict-split Source Data 与 physical-gain 未验证边界
  - `docs/parametric_map_lut.md` / `docs/t4_2_1_parametric_map_lut_validation.json` / `.csv` / `docs/t4_2_1_parametric_map_lut_bank_images.json`: active K/b 反解、X/Z integer LLR ROM、16,384-row exhaustive grid、5-cycle/II=1 software pipeline 与 non-hardware 边界
  - `docs/experimental_event_fsm.md` / `docs/t4_2_2_experimental_event_fsm_validation.json` / `.csv`: 六态 observed-event FSM、饱和 run counters、Pauli/phase-frame、1,024-row fault/transition replay 与 6-cycle/II=1 software contract
  - `docs/conservative_fallback_health.md` / `docs/t4_2_3_conservative_fallback_validation.json` / `.csv`: 14-bit health/integrity registry、trusted image/version、frame-hold/reset fallback、4,096-row fault replay 与 non-hardware 边界
  - `docs/fast_path_fixed_point.md` / `docs/t4_2_4_fast_path_fixed_point_validation.json` / `docs/t4_2_4_fast_path_fixed_point_ler.csv` / `docs/t4_2_4_fast_path_fixed_point_exhaustive_codes.csv`: MAP→health→event→frame 全 word contract、87,040-code audit 与 128-cluster paired LER impact
  - `docs/three_timescale_cadence.md` / `docs/t4_3_1_three_timescale_cadence_validation.json` / `docs/t4_3_1_adaptation_lag_phase_sweep.csv` / `docs/t4_3_1_cadence_execution_trace.csv`: fast/event/window/slow/commit/recalibration cadence、4000-phase 双口径 adaptation lag 与真实 scheduler/T4.2 trace
  - `docs/atomic_parameter_bank.md` / `docs/t4_3_2_atomic_parameter_bank_validation.json` / `docs/t4_3_2_atomic_parameter_bank_source_data.csv`: 完整 MAP-LUT image 的 version/CRC/SHA/timestamp/CAS 双 bank 事务、atomic switch、hysteresis、ack/readback 与 7518-row negative evidence
  - `docs/closed_loop_fault_recovery.md` / `docs/t4_3_3_closed_loop_fault_recovery_validation.json` / `docs/t4_3_3_closed_loop_fault_recovery_source_data.csv`: drift/burst/leakage/timeout/通信中断/race 的 32-run、767872-cycle closed-loop safety、ack uncertainty、guard 与 monotonic LKG republish
  - `docs/bounded_residual_rnn_teacher.md` / `docs/t4_4_1_bounded_residual_rnn_teacher_validation.json` / `.pt` / `.csv`: 3-restart fresh GRU teacher、15-output hard action bounds、1,074-row strict-split Source Data、checkpoint non-reuse 与 cap-hit/非全局收敛边界
  - `docs/teacher_hidden_control_analysis.md` / `docs/t4_4_2_teacher_hidden_control_analysis.json` / `.csv`: 128-half-cycle hidden/control、strict-split p(g) belief probe、PCA/指数/有效记忆、2,089-row Source Data 与 leakage 非原生边界
  - `docs/low_dimensional_student_distillation.md` / `docs/t4_4_3_low_dimensional_student_validation.json` / candidate `.pt` / student `.json` / `.csv`: 4-state/95-scalar selected recurrence、held-out imitation error、58,356-row Source Data 与 gain-retention 未验证边界
  - `docs/horizon_extrapolation_validation.md` / `docs/t5_4_5_horizon_extrapolation_validation.json` / candidate `.pt` / `.csv`: 四 training horizons、8 条真实百万-cycle streams、hidden/state bound、最坏流 imitation、float32 与 reset 证据；长时 physical gain 保持未建立
  - `docs/randomized_model_mismatch.md` / `docs/t5_4_6_randomized_model_mismatch.json` / `.csv`: 64 个 parent-disjoint random cells、32-cell paired finite-cutoff control、完整 readout/leakage/reset/drift lanes 与 fail-closed student-retention branch；装置/长时/硬件 claim 保持关闭
  - `docs/bit_accurate_hardware_reference.md` / `docs/t5_5_1_bit_accurate_hardware_reference.json` / Source Data / golden trace / binary bank: packed-word Python RTL golden、16,384-code parity 与 atomic in-flight bank switch；RTL/synthesis/Fmax/resource/board 字段保持未测
  - `docs/synthesizable_rtl_equivalence.md` / `docs/target_device_synthesis.md` / T-RISK/T5.5.2 machine artifacts: 4,316 valid-row RTL 对拍、8 BSRAM 映射、三 seed Fmax/resource/critical-path estimate；vendor signoff/bitstream/真板仍未建立
  - `docs/production_rtl_audit.md` / `docs/t6_2_1_production_rtl_audit.json` / Source Data: production synchronous management top、514-word CRC32 配置、inactive A/B bank、CAS/safe-boundary commit、drain guard 与 1,681-cycle CXXRTL；transport/CDC/P&R/真板仍未建立
  - `docs/long_rtl_qualification.md` / `docs/t6_2_2_long_rtl_qualification.json` / Source Data: 10×100,000-cycle board-independent long trace、全 visible RTL word 0 mismatch、fault/recovery/saturation 与抽象 transport fail-closed；不替代真实 transport/bitstream/板测
  - `docs/route_a_claim_contract.md` / `docs/t6_5_1_route_a_claim_contract.json` / Source Data: safe adaptive dual-loop 的 11-role canonical ledger、decoder/GQF/hardware 三条不可混排 lane 与逐 claim activation/revocation/wording gate
  - `docs/unified_execution_contract.md` / `docs/t6_5_2_unified_execution_contract.json` / Source Data: 七个 observed-only comparator 的同输入、phase-LUT/Q9.12、A/B bank、cadence、matched-budget 与 deadline 合同，isolated oracle truth schema 和 70-case per-method fail-fast matrix；full 2D joint MAP 的 current-RTL 边界显式关闭
  - `docs/unified_comparator_runner.md` / `docs/t6_6_1_unified_comparator_runner.json` / Source Data: 六个 common-trace adapter、真实 legacy CNN failure branch、隔离 oracle、standard/periodic grid 穷举、逐事件/累计计算成本与 integration-only Route-A 负结果
  - `docs/regime_aware_safe_policy.md` / `docs/t6_6_2_regime_aware_safe_policy.json` / Source Data: T6.6.2 20,061-cycle V4 结构长轨、Window/EWMA 双影子总预算、tail EWMA 原子提交、integrity LKG rollback 与 6-cycle action provenance
  - `docs/route_a_posterior_calibration.md` / `docs/t6_6_3_route_a_posterior_threshold_lock.json` / Source Data: T6.6.3 全量 calibration/pilot、V4 common tuple/router/EWMA baseline lock、完整 pilot selector；V2/V3 NO-GO、fallback/lag 与 formal 未访问边界完整披露
  - `docs/route_a_smooth_formal.md` / `docs/t6_7_1_smooth_formal_matrix.json` / Source Data: T6.7.1 576 条 untouched smooth formal trajectories、七方法逐窗口 Pauli/paired/action 计数、独立 varying-state oracle 与 seed-cluster CI；锁定 EWMA contrast 通过，但 static/Window 更低、oracle-gap 为负且优势集中 periodic 的边界完整披露
  - `docs/route_a_tail_formal.md` / `docs/t6_7_2_abrupt_ood_tail_formal_matrix.json` / Source Data: T6.7.2 888 条 abrupt/OOD+nominal formal trajectories、逐窗口 Pauli/paired/action 与逐事件 lag；catastrophic/calibration/nominal 门通过，但主要为 locked-EWMA 等价、static calibration 更强以及高 fallback/false-update 的边界完整披露
  - `docs/route_a_integrated_rtl_qualification.md` / `docs/t6_7_3_route_a_integrated_rtl_qualification.json` / Source Data: T6.7.3 20条frozen formal trajectory、99.5802% unified posterior replay、10×100k production core+Route-A逐word CXXRTL 0 mismatch与完整commit/rollback/FIFO故障覆盖；明确HMM仍在软件且非板测/P&R/LER优势
  - `docs/route_a_promotion_falsification_gate.md` / `docs/t6_7_4_route_a_promotion_gate.json` / Source Data: T6.7.4 从1,464条raw trajectory、两份大CSV与131MB trace独立重算；合同系统受限GO与Window/static/tail/CNN/HMM/板测负边界同时机器冻结
  - `docs/static_gkp_same_model_lane.md` / `docs/t6_8_1_static_gkp_same_model_lane.json` / Source Data: T6.8.1 same-trace static GKP lane；Route-A相对static average优势被paired CI证否，K=4/full在完整1024²输入域hard-action等价并报告非板测成本代理
  - `docs/external_drift_adaptive_lane.md` / `docs/t6_8_2_external_drift_adaptive_lane.json` / Source Data: T6.8.2 pinned external BOCD common-trace lane；Route-A paired LER较低但外部strict worst wall-clock超限，故只保留描述性结果并禁止matched-budget/general-SOTA升级
  - `docs/gqf_official_intake.md` / `docs/t6_8_3_gqf_official_intake.json` / `configs/gqf_official/`: T6.8.3 Puviani official GQF intake；固定 pristine upstream、隔离四补丁和 Python/CUDA locks，CPU 真实环境一步通过但 GPU cuSolver 路径未合格，paper-exact/超过 NMF 仍禁止
  - `docs/gqf_paper_exact_reproduction.md` / `docs/t6_8_4_gqf_paper_exact_reproduction.json` / Source Data: T6.8.4 paper/source/code exact审计；18项阻断、20-agent显式null ledger与六态三seed reduced official standard probe，exact资格0/15，MF/NMF ordering及surpass禁止
  - `docs/gqf_route_a_matched_comparison_gate.md` / `docs/t6_8_5_gqf_route_a_matched_comparison_gate.json`: T6.8.5 前置失败负分支；8项eligibility全失败，未生成不公平comparison，全部性能/成本字段null并冻结恢复条件
  - `docs/fpga_qec_decoder_normalization.md` / `docs/t6_8_6_fpga_decoder_normalization.json` / Source Data: T6.8.6 一手 FPGA QEC decoder 规范化；8个外部具体实现与2个项目证据行分离core/per-round/iteration/source-to-action/closed-loop，same-task comparator为0，fastest/SOTA/速度优势禁止
  - `docs/external_fpga_decoder_refresh.md` / `docs/t6_19_2_external_fpga_normalization.json` / Source Data: T6.19.2 截止2026-07-20的外部FPGA QEC刷新；实时继承8个旧实现并新增10个一手实现，逐行冻结task signature、latency statistic、resource与evidence state；exact same-task comparator仍为0，禁止跨code family的raw-ns排名和faster/SOTA claim
  - `docs/route_a_innovation_advantage_claim_matrix.md` / `docs/t6_8_7_route_a_claim_matrix.json` / Source Data: T6.8.7 十条原子 innovation/advantage 主张；四类对手分别绑定 required/current/gap/revocation 与 report/source/config/seed/hash，静态、general-SOTA、NMF-surpass、FPGA-speed 负边界 fail closed，T6.9 证据显式 pending/null
  - `docs/route_a_hardware_pareto.md` / `docs/t6_9_1_route_a_hardware_pareto.json` / Source Data / durable netlists and tool logs: T6.9.1 integrated Route-A no-student/student-sidecar 两个真实 profile 各三 seed P&R；报告 Fmax/resource/六周期 clock model 和解析 power sensitivity，并保持 vendor/bitstream/transport/board/measured/speed claim 关闭
  - `docs/route_a_board_measurement_blocker.md` / `docs/t6_9_2_route_a_board_measurement_blocker.json`: T6.9.2 实物板测 fail-closed prerequisite contract；6项physical prerequisite缺失、42个measured字段全null，禁止把P&R换算复制为板测，恢复需实板/transport/timestamp/bitstream/百万周期完整链
  - `docs/route_a_board_preboard_candidate.md` / `docs/t6_9_2_preboard_bitstream_candidate.json`: T6.9.2 板到前候选闭环；40/96-byte framed UART、单脉冲事件门控、实际波特率比例 PHY 与完整栈 CXXRTL、GW2AR P&R/`.fs` 打包，Fmax `83.9701 MHz`、LUT4/DFF/BSRAM=`6532/2969/8`；manifest 强制标记未烧录/未测量，逐帧链路不替代满速百万周期 HIL
  - `docs/route_a_final_evidence_gate.md` / `docs/t6_9_3_route_a_final_evidence_gate.json` / Source Data: T6.9.3 十一条最终原子主张与高水平论文GO/NO-GO；17/17 gates和mutations把完整论文判为NO-GO，仅允许受限pre-board system draft，并保持static/tail/external-budget/GQF/board/FPGA-speed负边界可见
  - `docs/route_a_causal_headroom.md` / `docs/t6_10_1_causal_headroom.json` / Source Data: T6.10.1 exact formal diagnostic replay 与全新 development nested headroom；strict-causal router `-0.2322%`、fixed mixture `+0.4587%`、纯 action-space 增量上界 `0.02549%`，因此触发 Phase 6B early NO-GO 而不启动新 formal
  - `docs/route_a_v5_final_evidence_gate.md` / `docs/t6_15_5_route_a_v5_final_evidence_gate.json` / Source Data: T6.15.5 early-stop claim registry 与 absence proof；12/12 gates、6/6 mutations，终态 `NO_GO_V5_EARLY_HEADROOM_STOP`，Phase 6C 只能 read-only auxiliary
  - `docs/route_a_preregistration.md` / `docs/t6_5_3_route_a_preregistration.json` / Source Data: 新 formal 数据 result-blind 的 143-cell/24-cluster 场景设计、shared threshold selector、equal-family paired bootstrap、512-window tail、catastrophic/nominal non-inferiority 与失败降级门
  - `docs/precision_resource_performance_pareto.md` / `docs/t5_5_3_precision_resource_pareto.json` / Source Data: 唯一 p10/K4-reference/state4/P1 点、7,680-code student RTL 对拍与 integrated resource/Fmax；在线 top-K/P2/P4/真板仍关闭
  - `docs/gru_student_hardware_feasibility.md` / `docs/t5_5_4_gru_student_hardware_feasibility.json` / Source Data: full/quantized/student 同口径存储、MAC、BRAM/DSP、Fmax、latency 与 gain gate；quantized lower-bound 不冒充 functional RTL
  - `docs/teacher_student_gain_retention.md` / `docs/t4_4_4_teacher_student_gain_retention.json` / `.csv`: cutoff12/16 paired retention point/CI、MF 排名反转、g/e/leakage burden、解析成本与 short-horizon oracle 边界
  - `docs/teacher_student_branch_freeze.md` / `docs/t4_4_5_teacher_student_branch_freeze.json` / `.csv`: 8-predicate fail-closed branch freeze、112-row parent/claim/revocation ledger 与 armed MAP-LUT fallback
  - `docs/sidecar/`: sidecar 扩展实验治理与 worktree 规划
  - `docs/new_tasks/`: `docs/new_task_board.md` 中每个新 task 的完成记录
  - `docs/new_risks.md`: 新任务序列的风险登记与插入任务判断
  - `docs/legacy_context/`: 已退役或只作历史参考的旧计划、旧分析和归档材料
  - `docs/progress_summary/`: 已退役阶段结论索引；当前阶段结论统一维护在 `docs/02_experiment_plan.md`
- `runs/`, `artifacts/`: 运行产物与历史证据

## 当前已确认的入口

- P0 基线脚本：
  - `python benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test`
- 训练入口：
  - `python -m cnn_fpga.model.train --config cnn_fpga/config/experiment_static_theta_v2.yaml`
- HIL 入口：
  - `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil.yaml`
- P4 入口：
  - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_hybrid_b_long.yaml`

这些入口代表“代码中存在”；其中当前已经重新验收通过的 bounded recovery 路径，请以 `docs/recovery_bootstrap/P0_smoke_bootstrap.md`、`docs/recovery_bootstrap/P3_software_hil_bootstrap.md` 与 `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md` 为准。旧治理状态仍见 `docs/04_task_board.md` 与 `docs/07_handoff.md`；自 2026-07-14 起，本轮顺序执行的唯一当前任务状态以 `docs/new_task_board.md` 为准。

## 环境说明

当前仓库根目录现已补一份 recovery 期最小依赖说明文件：

- `requirements-recovery.txt`

它只覆盖当前已复验的 `P0/P3/P4 recovery smoke` 路径，不等于完整训练链、`.tflite` runtime 或 `real_board` HIL 全环境。

如果只是先把当前 recovery 路径装到一个新解释器里，可执行：

```powershell
python -m pip install -r requirements-recovery.txt
```

截至 `2026-05-06`，本机已确认的解释器分工如下：

- `C:\Python313\python.exe`
  - 有 `yaml`
  - 无 `numpy / torch / tensorflow`
  - 不适合作为项目运行解释器
- `C:\ProgramData\anaconda3\python.exe`
  - 有 `numpy + yaml`
  - 无 `torch / tensorflow`
  - 当前恢复期推荐的最小 smoke 解释器
- `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
  - 有 `numpy + yaml + torch`
  - `torch.cuda.is_available() = True`
  - 适合作为后续训练环境候选
  - 这是 legacy 开发常用环境
  - 但当前不作为恢复期最小 smoke 解释器
- `C:\ProgramData\anaconda3\envs\QuantumEnv\python.exe`
  - 有 `numpy + yaml + torch`
  - 可作为训练/实验候选环境
- `C:\ProgramData\anaconda3\envs\TF1_14\python.exe`
  - 有 `tensorflow`
  - 缺 `yaml`
  - 当前不适合作为完整仓库入口环境

另一个关键事实是：仓库和工作区内未找到文档中多次提到的 `.venvs/tf311`，所以它目前不能被当成现成可用前提。

当前还没有“一次覆盖整个仓库所有路径”的统一根级环境文件；训练链、`.tflite` runtime 与真板 HIL 仍需继续按恢复期边界单独说明。

因此，在继续任何新功能或新 benchmark 之前，请先确认：

1. 依赖矩阵确认
2. 当前任务是否有清晰 task package
3. 验证结果是否会回写到治理文档

## 当前推荐最小入口

恢复期当前推荐的最小 smoke 命令为：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda
```

截至 `2026-05-06`，该命令已在当前机器上跑通，输出位于：

- `runs/smoke_test_anaconda/n10_r2_s0.250_ler_curve_compare.csv`
- `runs/smoke_test_anaconda/n10_r2_s0.250_summary.json`

最小 smoke 的复用说明已整理到：

- `docs/recovery_bootstrap/P0_smoke_bootstrap.md`

如果目标是 `P3/P4 recovery smoke`，请继续参照：

- `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
- `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`

## 复用建议

- 如果目标只是恢复期最小 smoke，优先用 `C:\ProgramData\anaconda3\python.exe`
- 如果目标是后续 torch 训练或更重的模型实验，优先切到 `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
- `DLEnv` 是 legacy 开发常用环境，但不应反向取代恢复期最小 smoke 口径
- `requirements-recovery.txt` 只承诺 `numpy + PyYAML` 这一层的 recovery smoke 依赖，不承诺训练、`.tflite` 或真板环境已经恢复

## 工作方式

- 项目状态以仓库文件为准，不以聊天上下文为准
- 当前已退出恢复期，但继续开发仍必须是有界任务，且要保持验证与文档一致性
- 不把 `mock`、`placeholder`、未来计划或未复验结果写成“已完成事实”

具体协作规范见 `AGENTS.md` 与 `CLAUDE.md`。
