# benchmark/ — 实验基准与 HIL 验证套件

本目录包含实验基准脚本的完整集合，覆盖从纯数值仿真漂移基准到 FPGA 硬件在环 (HIL) 验证的全流程。脚本按实验阶段 (P1–P4) 组织，每个脚本均为独立的 CLI 入口点。

## 目录结构

| 文件 | 阶段 | 职责 |
|------|------|------|
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
