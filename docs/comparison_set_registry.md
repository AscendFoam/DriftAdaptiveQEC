# T5.1.1 完整 comparison set registry

## 1. 结论

本任务冻结 19 个 comparator、8 个允许的 comparison lanes、16 个既有 PASS artifacts、19 个当前实现
bindings 和 100 行 Source Data。registry 的 14/14 gates 为 `PASS`，表示要求项、代码/证据绑定、信息预算、
协议/时间/算力口径、deployability 和禁止混排规则完整；它**不是一张全局排行榜**。

当前统一 scenario matrix 状态为 `PREREGISTERED_NOT_EXECUTED_T5_1_2`。旧 T3/T4 task-local 数字不能在
本任务中被直接拼接排序；T5.1.2 必须在 shared inputs、paired seeds、相同 protocol/metric contract 下重跑
相应 lane。

机器 registry：`docs/t5_1_1_comparison_set_registry.json`；逐行账本：
`docs/t5_1_1_comparison_set_registry_source_data.csv`。

## 2. 完整 comparator 清单

| ID | 类别 | 当前定位 | 允许 lane / 关键边界 |
| --- | --- | --- | --- |
| `no_correction_idle_memory` | protocol anchor | executable zero-action anchor | wall-clock；无 gate/measurement/reset/frame/update |
| `standard_binning` | decoder | main anchor | current/continuous/episode/finite-energy decoder lanes |
| `standard_measurement_feedback_sbs` | control protocol | main anchor | wall-clock、matched control、short oracle anchor |
| `autonomous_sbs` | control protocol | separate protocol lane | wall-clock；7 μs literature cycle，不是训练 optimum |
| `static_periodic_map` | decoder | main candidate | current/continuous/episode；training-frozen |
| `topk_periodic_map` | decoder approximation | implementation sensitivity | current syndrome；不是 surface K-MWM |
| `decoder_oracle_map` | decoder bound | nondeployable reference | hidden true DriftState；不是 control/channel oracle |
| `finite_energy_static_shrinkage` | decoder | separate fidelity lane | syndrome-level effective model，不是 full Fock recovery |
| `memory_bayesian` | decoder | episode-only candidate | bounded episode；不可与 continuous table 混排 |
| `ewma_adaptive_map` | decoder/estimator | main candidate | continuous drift；one-window delay |
| `kalman_adaptive_map` | decoder/estimator | main candidate | continuous drift；constant-velocity assumed model |
| `sliding_window_map` | decoder/estimator | main candidate | training-selected window；evaluation 不选长度 |
| `run_length_event_controller` | event controller | component only | detection/action diagnostics；不是直接 LER row |
| `regime_hmm_estimator` | host estimator | component only | regime diagnostics；未接 adapter 前不进主榜 |
| `latest_outcome_mf_fnn` | control policy | exact-budget baseline | matched control；保留 cutoff reversal |
| `exponential_recurrence_controller` | control policy | small-state candidate | matched control；与 distilled student 分列 |
| `bounded_residual_rnn_teacher` | controller teacher | offline reference | 非全局 optimum、非部署结论 |
| `distilled_low_dimensional_student` | controller student | qualified candidate | retention-only；无 OOD/RTL/board promotion |
| `finite_horizon_control_oracle` | control bound | nondeployable 2-cycle reference | 禁止外推为 10-cycle/global bound |

只有 `decoder_oracle_map` 允许在线 evaluator 读取 hidden truth，并被强制标为 nondeployable。control oracle 的
offline full-tree optimization与在线 realized-prefix lookup 分开，节点不使用未来 outcome；它也保持
nondeployable 和 two-cycle-only。

## 3. 八个公平 comparison lanes

| Lane | 成员数 | 共同目标 | 禁止混排规则 |
| --- | ---: | --- | --- |
| `decoder_current_syndrome` | 4 | 当前 syndrome → logical coset | top-K 是 sensitivity；decoder oracle 只作 bound |
| `decoder_continuous_drift` | 6 | 因果 drifting-syndrome decoding | T5.1.2 重建 shared adapter，不拼旧数字 |
| `decoder_episode_memory` | 4 | bounded episode decision | 不与 continuous per-round 结果混排 |
| `finite_energy_effective` | 2 | finite-energy noisy-syndrome correction | 不升级为 full Fock/channel optimum |
| `protocol_wallclock` | 3 | common physical-time memory preservation | 同报 per-cycle/wall-clock/event burden，不预设方向 |
| `control_matched_model` | 5 | bounded-residual causal sBs control | 同 cutoff/horizon/seed/budget；禁止 universal NMF |
| `event_and_regime_components` | 2 | event/regime component diagnostics | 无明确 adapter 前不得进入 LER leaderboard |
| `control_oracle_short_horizon` | 2 | exact short causal control tree | 2-cycle 不能外推 10-cycle |

一个 comparator 可出现在多个兼容 lane，但不存在 `global_leaderboard`。决策目标不同的 decoder、controller、
host estimator、protocol timing 和 oracle 不能只因都有一个“error/lifetime”标量就合并。

## 4. No correction 新增物理锚点

此前仓库没有可执行的 no-correction protocol baseline。本任务在
`physics/autonomous_sbs.py` 新增 `IdleMemorySimulator`：与 standard measurement-feedback sBs 使用相同
finite-cutoff 初态和 10 μs 报告网格，只传播 cavity idle-loss channel，不执行 sBs gate、measurement、reset、
frame update 或 parameter update。

cutoff 6、3×10 μs probe 得到 fidelity `1.0 → 0.874737`、logical-Z
`0.999500 → 0.865604`；全部五类操作计数为 0，trace/hermiticity error 为 0，最小 eigenvalue
`-4.87e-18`。其 final density 与 standard sBs 最大差 `0.342927`，证明不是把 standard 曲线改名；
`10 μs×1` 与 `5 μs×2` final-density 最大差 `1.11e-16`，验证 idle channel 半群一致性。这里不要求
no correction 一定更差，避免在有限模型中硬编码期望排序。

## 5. Finite-energy static 不是空标签

`finite_energy_static_shrinkage` 直接执行 T1.2.3 的 5-point、120k train/300k held-out eval harness。
五个 fitted gains 随 `Delta` 下降向 1 单调移动，logical/MSE advantage 收缩，全部 shrinkage MSE 不差于
standard；五点 paired logical gain CI lower 均大于 0。它只属于 syndrome-level effective finite-energy
lane，不能冒充完整 approximate-GKP/Fock recovery 或 channel optimum。

## 6. Secondary 与 counterevidence

Knill/P-Steane 不进入 sBs 主排名：Knill/qunaught 保持 reference-only；P-Steane 虽已通过 T5.0.2 的
252-point small-noise analytic holdout，仍不是 sBs controller 或 FPGA physical squeezing。

MF/FNN 必须保留 cutoff-dependent reversal；teacher 只作 offline matched-model reference；student 只保留
qualified gain-retention。禁止写成 universal NMF superiority，也禁止把当前 float/software evidence升级为
OOD、multilevel leakage、RTL、FPGA 或 board 结论。

## 7. 非 demo 审计

- 19 个 required IDs 顺序、唯一性和双向 lane membership 由代码强制；少一项或错一 lane 直接报错；
- 19 个 source fragments 与文件 SHA-256、16 个 parent artifact SHA-256/PASS 状态逐次重查；
- no-correction 与 finite-energy 两项不只登记名字，而是在 runner 中实际执行数值 probe；
- 28 个 lane memberships 逐行写入 Source Data，component/oracle/secondary exclusion 另有机器 rows；
- mutation tests 覆盖缺项、deployable hidden-truth 注入和单向 lane 漂移；
- registry PASS 明确不代表 T5.1.2 matrix 已执行，不允许把旧异构任务结果拼成新结论。

## 8. 复现

```powershell
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m cnn_fpga.benchmark.comparison_set_registry
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:\ProgramData\anaconda3\envs\DLEnv\python.exe' -m pytest -q tests/test_comparison_set_registry.py tests/test_autonomous_sbs.py
```
