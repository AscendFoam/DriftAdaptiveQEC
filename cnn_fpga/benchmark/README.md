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

## 关键依赖

- **runtime**: `SlowLoopRuntime`, `DualLoopScheduler`, `FastLoopEmulator`, `ParamBank`, `LatencyInjector`
- **hwio**: `FPGADriver`, `DMAReadout`
- **decoder**: 各类 baseline（EKF, UKF, ParticleFilter, WindowVariance）
- **physics**: `LinearDecoder`, 综合征测量, 逻辑错误追踪
- **utils**: YAML 配置加载, 路径管理, JSON 序列化
