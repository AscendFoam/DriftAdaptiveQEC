# Decoder / controller / oracle / teacher / student / bound 术语冻结

**Task：** T1.4.5  
**状态：** Frozen contract  
**机器真源：** `docs/decoder_controller_terminology.json`

## 1. 为什么必须冻结

本项目同时存在 logical-coset 解码、sBs 控制、慢变量估计、teacher-to-student 蒸馏、
FPGA 执行和 channel-level recovery 分析。它们的输入信息、输出、优化目标和可部署性不同。
如果都简称“decoder”“controller”或“oracle”，就会出现三类严重错误：

1. 把读取 hidden truth 的模型上界包装成实际算法；
2. 把模型训练出的 RNN control teacher 包装成 FPGA decoder；
3. 把允许任意 recovery channel 的 Petz/QEC-matrix bound 包装成 per-shot MAP 性能。

因此每个角色都按七个维度冻结：decision question、information set、output contract、objective、
causality、horizon、deployability。只要其中任一项改变，就必须改 canonical term 或明确加 qualifier。

## 2. 十一个 canonical roles

| ID | Canonical name | 核心问题 | 输出 | 当前状态 |
| --- | --- | --- | --- | --- |
| <!-- term-id: TERM-DECODER --> `TERM-DECODER` | `deployable_syndrome_decoder` | 当前 syndrome 属于哪个 logical coset？ | parity/class、frame、bounded correction | software implemented；hardware unverified |
| <!-- term-id: TERM-DECODER-ORACLE --> `TERM-DECODER-ORACLE` | `decoder_oracle` | 若知道当步真实 DriftState，模型内 Bayes decision 是什么？ | coset posterior/MAP action | nondeployable software reference |
| <!-- term-id: TERM-CONTROL-POLICY --> `TERM-CONTROL-POLICY` | `causal_control_policy` | 当前 history 下下一轮 sBs/control 如何设置？ | gate residual / bank action | software run-length、latest-outcome MF 与 exact short-horizon exponential recurrence；hardware unverified |
| <!-- term-id: TERM-CONTROL-ORACLE --> `TERM-CONTROL-ORACLE` | `finite_horizon_control_oracle` | 固定 ansatz/horizon 内可找到的最佳因果 control tree 是什么？ | history-indexed controls/value | two-cycle empirical multi-start software reference；exponential, nondeployable |
| <!-- term-id: TERM-RECOVERY-BOUND --> `TERM-RECOVERY-BOUND` | `channel_recovery_bound` | 允许任意 recovery channel 时可恢复到什么 fidelity？ | channel fidelity bound/gap | planned, nondeployable bound |
| <!-- term-id: TERM-TEACHER --> `TERM-TEACHER` | `feedback_grape_teacher` | 模型能否发现有用的 history-dependent sBs policy？ | bounded control residual/state trace | frozen five-agent offline software teacher |
| <!-- term-id: TERM-STUDENT --> `TERM-STUDENT` | `distilled_controller_student` | 低维确定性策略能保留多少 teacher gain？ | recurrence/FSM control residual | software imitation candidate；physical/hardware gates open |
| <!-- term-id: TERM-HOST-ESTIMATOR --> `TERM-HOST-ESTIMATOR` | `host_drift_estimator` | observed windows 能估出哪些慢变量/regime？ | state/uncertainty/bank proposal | partial software；含 synthetic causal HMM regime baseline |
| <!-- term-id: TERM-FAST-PATH --> `TERM-FAST-PATH` | `fpga_fast_path_executor` | 如何在 deadline 内安全执行 latched action？ | action/frame/health/version trace | end-to-end bit-accurate software contract；not real FPGA |
| <!-- term-id: TERM-DELAYED-TRUTH --> `TERM-DELAYED-TRUTH` | `delayed_hidden_truth_parameter_reference` | 延迟读取 mock target 时参数链如何表现？ | delayed NoisePrediction | implemented test-only |
| <!-- term-id: TERM-LEGACY-TEACHER-SOURCE --> `TERM-LEGACY-TEACHER-SOURCE` | `legacy_teacher_feature_source` | residual pipeline 用哪个 classical reference estimator？ | reference prediction/K,b features | implemented legacy name |

## 3. Decoder 与 controller 的第一性区别

| 维度 | Syndrome decoder | Control policy |
| --- | --- | --- |
| 被回答的问题 | “误差属于哪个 logical coset，应施加/追踪什么 recovery？” | “下一轮协议门参数、residual 或 bank action 应如何设置？” |
| 典型输入 | 当前 syndrome/LLR + frozen/estimated model | g/e/leakage history + internal state + recent actions |
| 典型输出 | parity/class、Pauli frame、analog correction | sBs gate residual、parameter-bank selection、safe action |
| 目标 | 当前/解码模型内 logical decision risk | 多步 lifetime/fidelity + control/reset/leakage/cost |
| 当前代码 | `LinearDecoder`、periodic MAP | 已有 run-length event-controller baseline；尚无完整物理闭环/最优 sBs policy |

一个网络的结构不决定名称，输出 contract 才决定名称。RNN 若输出 sBs 门参数，它是
`control teacher/policy`；只有输出 logical class/recovery decision 时才可叫 decoder。本项目
Feedback-GRAPE RNN 属于前者。

## 4. 三种上界必须分开

| 上界 | 特权信息/自由度 | 受限于 | 不回答 |
| --- | --- | --- | --- |
| `decoder_oracle` | 当步 hidden true DriftState | 已声明 periodic mixture decoder model | 不给 sBs control optimum，不是 channel optimum |
| `finite_horizon_control_oracle` | offline 完整 measurement tree；节点决策只看 causal prefix | 固定 sBs ansatz、model、short horizon | 不给 logical-coset MAP，不是任意 recovery |
| `channel_recovery_bound` | encoding + noise channel，可允许任意 recovery | cutoff/channel representation 与 bound tightness | 不生成在线 syndrome action，不证明 controller 可实现 |

控制 oracle 还必须满足一个反作弊条件：decision tree 的节点只能依赖已经发生的 prefix。
若动作使用了未来 measurement outcome，只能命名为 `hindsight trajectory bound`，不能进入
causal control-oracle comparison。

三种 gap 必须分列：

- decoder-oracle gap：static/adaptive decoder 到 hidden-state MAP reference；
- control-oracle gap：teacher/student policy 到 finite-horizon ansatz reference；
- channel-recovery gap：实际 protocol/recovery 到编码—噪声 recoverability reference。

禁止使用一个无修饰的 “oracle gap” 同时承载三者。

## 5. Teacher、student、host estimator 与 fast path

### 5.1 Feedback-GRAPE teacher

`feedback_grape_teacher` 是 offline model-aware recurrent **control** teacher。其在线 policy
必须因果，但训练通过可微动力学、reward derivative 和 trajectory log-probability path 完成。
它既没有 optimality 保证，也不能进入在线 runtime。T2.3.4--T2.3.7 已完成 simulator、gradient、
resource 和 directional-ranking gates；T4.1.5 只恢复冻结五-agent ensemble 生成离线 target。不能称
`RNN oracle`、`RNN decoder` 或 deployed controller。

### 5.2 Distilled controller student

`distilled_controller_student` 是从 teacher trajectory 拟合的指数递推或有限状态策略。T4.1.5 已实现
75-trainable/105-stored software recurrence、strict split imitation 和 health/leakage zero-residual fallback；
尚未证明 physical/lifetime gain retention。只有 boundedness、fixed-point、RTL、post-route、board
deadline 和 fault tests 都通过后，才可称 deployable。student 只是 fast-path 候选，不等于 FPGA 系统。

### 5.3 Host drift estimator

host estimator 只消费 observed validated windows，输出 slow state、uncertainty、regime 或完整
inactive-bank proposal；不输出 cycle-critical action，也不读取 `target_params`。CNN/TCN/GRU
若承担这个角色，名称仍是 estimator，不因为使用神经网络就成为 teacher 或 decoder。

### 5.4 FPGA fast-path executor

fast path 读取 latched bank/version，在本地执行 MAP/LUT/student、frame、event FSM、fallback 和
trace。当前已有 `FastLoopEmulator`、T4.2.1 version-bound integer MAP-LUT、T4.2.2 六态 event/frame 和
T4.2.3 traceable conservative fallback software pipeline。T4.2.3 是 frame-hold/reset policy，不含 OOD
生成/物理校准、自动 bank rollback 或 transport watchdog。T4.2.4 已把 MAP→health→event→frame 组合成
end-to-end bit-accurate Python reference并报告 paired LER；仍没有
RTL/real-board closure。它不得等待 host、在线训练或
执行无界优化。

## 6. 当前源码的两个 legacy 陷阱

### 6.1 `oracle_delayed`

`SlowLoopRuntime._predict_from_delayed_oracle()` 从 mock window 的 target prediction 取 hidden
truth，延迟若干窗口并加 synthetic prediction noise。它输出的是参数 estimate，不计算 syndrome
posterior 或 control-policy optimum。因此新 prose 必须写：

> mock delayed hidden-truth parameter reference (`oracle_delayed` legacy mode)

禁止写 `oracle decoder`、`control oracle` 或 deployable estimator。

### 6.2 `teacher_mode` / `teacher_prediction`

当前 `teacher_mode` 只在 `window_variance/EKF/UKF/particle_filter` 中选择 reference estimator，
供 residual/statcalib pipeline 构造辅助 prediction、K/b 或 features。它没有可微 sBs trajectory、
Feedback-GRAPE、RNN hidden state 或 control residual。新 prose 必须写 `legacy reference-estimator
feature source`；未来兼容迁移可改名 `reference_estimator_mode`，但本 task 不破坏历史 config。

## 7. 现有产物映射与完成度

| Artifact | Canonical role | 证据身份 |
| --- | --- | --- |
| `physics/error_correction.py::LinearDecoder` | deployable syndrome decoder building block | software only |
| `physics/ideal_gkp_decoder.py::map_decode_2d` | periodic MAP decoder building block | software only |
| `physics/oracle_map.py` | decoder oracle | implemented nondeployable assumed-model reference |
| `cnn_fpga/runtime/slow_loop_runtime.py` deployable modes | host drift estimator | partial software；stale/deadline gates incomplete |
| `cnn_fpga/decoder/slow_loop_model_selection.py::RollingGaussianHMMAdapter` | host drift estimator pilot backbone | matched-budget synthetic four-regime validation winner；非 richer-input、device、fixed-point 或硬件证据 |
| `cnn_fpga/data/experimental_history.py::ExperimentalHistoryBuilder` | host drift estimator input contract | 256×53 observed-only causal software history；真实 syndrome/FSM/LLR/scheduler producer adapter，非 IQ/ADC/device/RTL/board 证据 |
| `cnn_fpga/decoder/hybrid_state_output.py::HybridStateOutput` | host drift estimator output contract | future-only continuous/regime/risk/recovery-burden/uncertainty 与 inactive-bank proposal；非逐周期 controller action、physical calibration 或 hardware evidence |
| `slow_loop_runtime.py::oracle_delayed` | delayed hidden-truth parameter reference | test-only |
| `slow_loop_runtime.py::teacher_mode` | legacy teacher feature source | historical naming only |
| `cnn_fpga/runtime/fast_loop_emulator.py` | FPGA fast-path executor | software emulation only |
| `cnn_fpga/runtime/parametric_map_lut.py::ParametricMAPLUTRuntime` | FPGA fast-path executor | T4.2.1 X/Z marginal integer ROM + 5-cycle/II=1 software contract；非 RTL/综合/板测 |
| `cnn_fpga/runtime/experimental_event_fsm.py::ExperimentalEventFSM` | FPGA fast-path executor | T4.2.2 六态 observed-event/frame + 1-cycle register software contract；非完整 fallback、物理 recovery、RTL/综合/板测 |
| `cnn_fpga/runtime/conservative_fallback.py::ConservativeFallbackController` | FPGA fast-path executor | T4.2.3 14-bit health/integrity + trusted-version frame-hold/reset software contract；非 OOD 校准、自动 rollback、transport/RTL/综合/板测 |
| `cnn_fpga/runtime/fast_path_fixed_point.py::BitAccurateFastPath` | FPGA fast-path executor | T4.2.4 MAP→health→event→frame 全 word Python reference + model-matched paired LER；非 correlated/OOD/device/RTL/综合/板测 |
| `cnn_fpga/runtime/run_length_fsm.py::RunLengthParameterBankFSM` | causal control policy | observed-only software event baseline；physical feedback/RTL/board unverified |
| `physics/latest_outcome_markovian.py::BudgetMatchedMarkovianPolicy` | causal control policy | 72,853 参数 latest-outcome software MF；two-level production lane 无 leakage，非 device/RTL/board evidence |
| `physics/exponential_recurrence_control.py::ExponentialSaturationControlPolicy` | causal control policy | 75-trainable/105-stored two-cycle exact recurrence；separate leakage-aware event/Q mirror，非 teacher/RTL/board evidence |
| `physics/trajectory_lookup_control_oracle.py::CausalHistoryLookupPolicy` | finite-horizon control oracle | two-cycle 15-node exact-branch empirical reference；nondeployable，非全局最优证明 |
| `physics/nmf_directional_ranking.py::PaperScaleNMFPolicy` | Feedback-GRAPE teacher | T2.3.7 frozen five-agent offline software teacher；非 oracle/online/hardware |
| `cnn_fpga/control/teacher_student.py::DistilledRecurrenceStudent` | distilled controller student | T4.1.5 software imitation candidate；105 scalars、health/leakage safe fallback，physical/fixed-point/RTL/board gates open |
| T5.3.5 | channel-recovery bound | planned, not implemented |

仍为 `planned` 的 channel bound 与 student RTL/board 层不得出现在 completed implementation list；
software teacher/student 也不能用 imitation MSE 代替 physical gain、fixed-point 或硬件验证。

## 8. 强制命名例句

允许：

- “The adaptive decoder closes part of the gap to a nondeployable full-state decoder oracle.”
- “The recurrent control teacher is compared with a finite-horizon control oracle.”
- “The actual sBs recovery remains below the QEC-matrix/Petz channel-recovery bound.”
- “A host drift estimator proposes a versioned bank consumed by the deterministic FPGA fast path.”
- “The distilled controller student is a deployment candidate pending fixed-point/RTL/board gates.”

禁止：

- “Our decoder oracle is deployed on FPGA.”
- “The RNN oracle/decoder outputs 15 sBs gate parameters.”
- “Petz decoder beats MAP.”
- “`teacher_mode=window_variance` proves the Feedback-GRAPE teacher exists.”
- “`oracle_delayed` is the oracle MAP baseline.”
- “The student is the complete FPGA controller.”

## 9. 报告与图表规则

1. 图表/caption 中写全 canonical role，禁止单写 `oracle`。
2. decoder/control/channel 三种 gap 使用不同列、分母和标题。
3. teacher 与 student 同表时同时报告 teacher selection、distillation error、gain retention 和成本。
4. host estimator 的 state-estimation score 与 decoder/controller logical score 分开。
5. software emulation、RTL、post-route、board core 和 HIL 继续遵守 claim ladder，不因叫 fast path
   自动升级 evidence。
6. 历史 artifact 字段可保留原名，但任何新 prose、task record 和主表必须加本 registry qualifier。

## 10. 非 demo 审计结论

本 task 没有只写词汇表。机器 registry 为每个角色冻结七个决策维度、当前实现状态和精确 artifact
binding；十条 pairwise conflation rule 直接覆盖三种上界、teacher/student、host/fast path 和两种
legacy 陷阱。测试还会读取真实代码行，证明 `oracle_map`、`oracle_delayed`、`teacher_mode` 和
`FastLoopEmulator` 的当前行为与术语映射一致，并锁住 planned 角色不得被伪装为 implemented。
