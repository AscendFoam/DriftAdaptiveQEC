# DriftAdaptiveQEC

`DriftAdaptiveQEC` 是一个围绕 “CNN + FPGA 快慢回路协同近似 GKP 解码” 的研究型工程仓库。当前代码已经覆盖物理仿真、数据集生成、Tiny-CNN 训练、量化/导出、软件侧 HIL 与 P4 多场景 benchmark。仓库现已完成第一轮恢复期治理收尾，进入“受控继续开发”阶段。

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
