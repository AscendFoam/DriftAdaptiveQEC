# P4 Benchmark Recovery Bootstrap

## 1. 目的

本文件只服务恢复期的 P4 benchmark 最小复用入口。

目标不是恢复正式长跑口径，而是让后续会话不用重新猜：

1. 当前最小 P4 benchmark 该怎么跑
2. 它复用的是哪条 HIL 入口
3. 它到底跑的是哪种 backend 与哪类 inference artifact

## 2. 先装哪份依赖

恢复期当前推荐先安装根目录的：

- `requirements-recovery.txt`

安装命令：

```powershell
python -m pip install -r requirements-recovery.txt
```

当前这份 manifest 只承诺覆盖：

1. `mock-backed P4 recovery smoke`
2. `mock-backed software HIL` 的 `artifact_npz + inproc` 复用路径
3. `P0/P3/P4 recovery smoke` 所需的 `numpy + PyYAML`

它当前不承诺覆盖：

1. `real_board` HIL backend
2. `.tflite` runtime / export
3. 正式多场景长跑 benchmark
4. `DLEnv` 训练链

## 3. 当前推荐路径

恢复期当前推荐保留两条有界 `P4` 复用路径：

- 最快最小路径：
  - interpreter: `C:\ProgramData\anaconda3\python.exe`
  - entry: `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark`
  - config: `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
  - scenario filter: `static_bias_theta`
  - mode filters:
    - `static_linear`
    - `cnn_fpga`
  - paired seeds: `true`
- 当前 frozen baseline 验收路径：
  - interpreter: `C:\ProgramData\anaconda3\python.exe`
  - entry: `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark`
  - config: `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
  - scenario filter: `static_bias_theta`
  - mode filters:
    - `static_linear`
    - `window_variance`
    - `ekf`
    - `cnn_fpga`
  - paired seeds: `true`

两条路径共同固定：

- HIL backend: `mock`
- inference service mode: `inproc`
- `cnn_fpga` inference backend: `artifact_npz`
- fixed artifact path: `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`

这两条路径的共同含义是：

- 它们都是 `P4 benchmark wrapper over software HIL`
- 它们复用的是 `T6` 已复验的 `mock-backed software HIL` 主链
- 它们不是 `real_board`
- 它们不是 `.tflite` runtime
- 它们也不是正式多场景 P4 长跑配置

## 4. 为什么选择这条路径

选择这些有界路径，是因为它们同时满足三件事：

1. 仍然走 `run_p4_multiscenario_benchmark.py -> run_hil_session(...)` 的真实 P4 入口
2. 保留最小 benchmark 对照，而不是退化成单次 HIL smoke
3. 继续避开 `.venvs/tf311`、`.tflite` 和 placeholder 真板 backend

相对地，以下路径当前不作为恢复期最小入口：

- `cnn_fpga/config/p4_multiscenario_smoke.yaml`
  - 因为它继承正式 `hardware_hil.yaml`，默认 slow-loop inference service 仍是 `subprocess`
- `backend=board`
  - 因为 `board_backend.py` 仍是 placeholder
- `backend=tflite`
  - 因为恢复期最小入口不应重新引入真实 `.tflite` 与 `tflite_stub_v1` 混淆
- 正式四场景四模式长跑
  - 因为当前任务目标是“最小复验”，不是恢复正式 benchmark 时长

## 5. 当前 recovery smoke 配置

- [cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml](/d:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml)

该配置的关键约束：

1. `base_config: p4_multiscenario_smoke.yaml`
2. 显式覆盖 `paths.model_dir: artifacts/models/static_theta_v2`
3. 显式固定 `slow_loop.model_artifact.path`
4. 显式固定 `slow_loop.inference_service.model_path`
5. 显式覆盖 `slow_loop.inference_service.mode: inproc`
6. 显式覆盖 `slow_loop.inference_service.backend: artifact_npz`
7. 显式覆盖 `hil.backend: mock`
8. 将 `n_slow_updates` 缩到 `8`
9. 将 `error_model` 的三类失败概率临时设为 `0.0`
10. 保留 `frozen_baseline_set` 口径，但 recovery 复验只筛一个场景，并在有界范围内运行两种模式或四种模式

## 6. 运行命令

最快最小路径：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds
```

当前 frozen baseline 单场景全模式 smoke：

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds
```

## 7. 预期输出

运行成功后，会在 `runs/p4_benchmark/` 下生成一条 `p4multi_*` 目录，至少包含：

- `summary.json`
- `comparison.csv`
- `delta.csv`
- `report.md`
- `progress.jsonl`

后续复核时，应优先确认：

1. `summary.json` 中 `filters.scenario == ["static_bias_theta"]`
2. `summary.json` 中 `filters.mode` 与实际执行的 mode 集一致
3. `summary.json` 中 `protocol.seed_pairing == "paired"`
4. `comparison.csv` 中 `cnn_fpga` 行的 `artifact_path` 指向固定 `.npz`
5. 相关 repeat 目录下的 `hil_summary.json` 继续体现：
   - `backend == "mock"`
   - `inference_service_mode == "inproc"`

截至 `2026-05-08`，最快最小路径已在当前机器上完成一次新的两模式复验：

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds`
- 新运行目录：
  - `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308`
- launch / filter 结果：
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `repeats = 1`
  - `seed_pairing = paired`
  - `scenario = static_bias_theta`
  - `modes = static_linear, cnn_fpga`
- comparison 结果：
  - `Static Linear final_ler = 1.00890625`
  - `Static Linear overflow_rate = 0.0020625`
  - `CNN-FPGA final_ler = 0.72109375`
  - `CNN-FPGA overflow_rate = 0.002375`
  - winner: `cnn_fpga`
  - `runner_up_gap = 0.2878125`
- `cnn_fpga` repeat HIL summary 结果：
  - `backend = mock`
  - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
  - `inference_service_mode = inproc`
  - `n_slow_updates_finished = 8`
  - `n_commits_applied = 8`

截至 `2026-05-08`，frozen baseline 单场景全模式 smoke 也已在当前机器上完成一次新的四模式复验：

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds`
- 新运行目录：
  - `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
- launch / filter 结果：
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `repeats = 1`
  - `seed_pairing = paired`
  - `scenario = static_bias_theta`
  - `modes = static_linear, window_variance, ekf, cnn_fpga`
- comparison 结果：
  - `Static Linear final_ler = 0.99575`
  - `Static Linear overflow_rate = 0.00246875`
  - `Window Variance final_ler = 0.57440625`
  - `Window Variance overflow_rate = 0.00221875`
  - `EKF final_ler = 0.6795`
  - `EKF overflow_rate = 0.0019375`
  - `CNN-FPGA final_ler = 0.7248125`
  - `CNN-FPGA overflow_rate = 0.00290625`
  - winner: `window_variance`
  - `runner_up_gap = 0.10509375`
- repeat HIL summary 结果：
  - 四个 mode 都是 `backend = mock`
  - 四个 mode 都是 `inference_service_mode = inproc`
  - `static_linear / window_variance / ekf` 的 `artifact_path = null`
  - `cnn_fpga` 的 `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
  - 四个 mode 都有：
    - `n_slow_updates_finished = 8`
    - `n_commits_applied = 8`

## 8. 这条路径证明了什么

它证明的是：

1. 当前仓库还有一条可直接执行的最小 P4 benchmark 入口
2. `run_p4_multiscenario_benchmark.py` 可以在恢复期继续复用 `mock + artifact_npz + inproc` 的 HIL 主链
3. 恢复期可以先在不碰 `.tflite`、不碰 `real_board` 的前提下恢复 P4 最小 benchmark 复验

## 9. 这条路径不证明什么

它不证明：

1. `real_board` 已完成
2. `.tflite` runtime 已完成
3. 正式四场景四模式 P4 长跑已恢复
4. P4 已达到逐字确定性复现

这些内容应继续留给后续任务：

- `T10`：基于 `T8 + T9` 重新做一次 `Go / Repair` gate review
