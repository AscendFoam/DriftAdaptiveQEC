# P3 Software HIL Bootstrap

## 1. 目的

本文件只服务恢复期的软件 HIL 最小复用入口。

目标不是给出正式 P3/P4 benchmark 口径，而是让后续会话不用重新猜：

1. 当前最小 software HIL 该怎么跑
2. 它到底跑的是哪种 backend
3. 它用的是哪类 inference artifact

## 2. 当前推荐最小路径

恢复期当前推荐的软件 HIL 最小路径是：

- interpreter: `C:\ProgramData\anaconda3\python.exe`
- entry: `python -m cnn_fpga.benchmark.run_hil_suite`
- config: `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- HIL backend: `mock`
- slow-loop mode: `model_artifact`
- inference service mode: `inproc`
- inference backend: `artifact_npz`
- artifact selector: `latest_float`

这条路径的含义是：

- 它是 `P3 software HIL`
- 它不是 `real_board`
- 它不是 `.tflite` runtime
- 它也不依赖 `.venvs/tf311`

## 3. 为什么选择这条路径

选择这条最小路径，是因为它同时满足三件事：

1. 保留了 `mock backend + artifact-based slow loop` 的主链语义
2. 避开了当前工作区里不可直接依赖的 `.venvs/tf311`
3. 依赖只需要当前已验证可用的 `numpy + yaml` 环境

相对地，以下路径当前不作为恢复期最小入口：

- `hil.backend=board`
  - 因为 `board_backend.py` 仍是 placeholder
- `inference_backend=tflite`
  - 因为恢复期最小入口不应再引入 `.tflite` / `tflite_stub_v1` 混淆
- `inference_service.mode=subprocess`
  - 因为主配置默认会碰到 `.venvs/tf311/bin/python`

## 4. 当前最小 smoke 配置

- [cnn_fpga/config/hardware_hil_recovery_smoke.yaml](/d:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/config/hardware_hil_recovery_smoke.yaml)

该配置的关键约束：

1. `base_config: hardware_hil.yaml`
2. 显式覆盖 `hil.backend: mock`
3. 显式覆盖 `inference_service.mode: inproc`
4. 显式覆盖 `inference_service.backend: artifact_npz`
5. 将 `n_slow_updates` 缩到 `2`
6. 将 `error_model` 故障概率临时设为 `0.0`
7. 将 `mock_signal.type` 设为 `static`

## 5. 运行命令

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml
```

## 6. 预期输出

运行成功后，会在 `runs/hil_suite/` 下生成一条 `hardware_hil_recovery_smoke_*` 目录，至少包含：

- `hil_summary.json`
- `hil_events.json`

后续复核时，应优先确认：

1. `backend == "mock"`
2. `artifact_path` 指向 `.npz`
3. `inference_service_mode == "inproc"`
4. `n_slow_updates_finished >= 1`
5. `n_commits_applied >= 1`

截至 `2026-05-07`，该路径已在当前机器上再次复验通过：

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 新运行目录：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
- 复核结果：
  - `backend = mock`
  - `n_windows_ready = 2`
  - `n_slow_updates_started = 2`
  - `n_slow_updates_finished = 2`
  - `n_commits_requested = 2`
  - `n_commit_acks_observed = 2`
  - `n_commits_applied = 2`
  - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
  - `inference_service_mode = inproc`
- 事件计数：
  - `window_ready = 2`
  - `slow_update_started = 2`
  - `slow_update_finished = 2`
  - `commit_applied = 2`
  - `commit_ack_asserted = 2`
  - `fast_budget_violation = 1`

同时需要保留一个恢复期观察：

- 两次复验的 control-plane 字段一致，但 `final_ler` 与 `overflow_rate` 存在小幅差异。
- 当前更合理的解释是“该路径已可复验，但还不是逐字确定性复现”。
- 这不影响其作为 `T6` 最小 software HIL 路径成立，但后续若要提升到更严格 benchmark 口径，应继续追踪随机源控制。

## 7. 这条路径证明了什么

它证明的是：

1. 当前仓库还有一条可以直接执行的软件 HIL 最小入口
2. `run_hil_suite.py -> FPGADriver(mock) -> SlowLoopRuntime(model_artifact)` 这条主链在当前环境上仍能启动并产出摘要
3. 恢复期可以先在不碰 `.tflite`、不碰 `real_board` 的前提下继续推进 P3/P4 入口修复

## 8. 这条路径不证明什么

它不证明：

1. `real_board` 已完成
2. `.tflite` runtime 已完成
3. `tflite_stub_v1` 与真实 `.tflite` 已被重新验收
4. P4 正式 benchmark 已恢复

这些内容应继续留给后续任务：

- `T5`：缓存/生成物噪声治理
- `T7`：重新验收一个 P4 benchmark 最小路径
