# P3 Software HIL Bootstrap

## 1. 目的

本文件只服务恢复期的软件 HIL 最小复用入口。

目标不是给出正式 P3/P4 benchmark 口径，而是让后续会话不用重新猜：

1. 当前最小 software HIL 该怎么跑
2. 它到底跑的是哪种 backend
3. 它用的是哪类 inference artifact

## 2. 先装哪份依赖

恢复期当前推荐先安装根目录的：

- `requirements-recovery.txt`

安装命令：

```powershell
python -m pip install -r requirements-recovery.txt
```

当前这份 manifest 只承诺覆盖：

1. `mock + model_artifact + artifact_npz + inproc` 这条 software HIL 最小路径
2. `P0/P3/P4 recovery smoke` 复用所需的 `numpy + PyYAML`

它当前不承诺覆盖：

1. `real_board` HIL backend
2. `.tflite` runtime / export
3. `subprocess + .venvs/tf311`
4. `DLEnv` 训练链

## 3. 当前推荐最小路径

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

## 4. 为什么选择这条路径

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

## 5. 当前最小 smoke 配置

- [cnn_fpga/config/hardware_hil_recovery_smoke.yaml](/d:/Codes/Quantum/DriftAdaptiveQEC/cnn_fpga/config/hardware_hil_recovery_smoke.yaml)

该配置的关键约束：

1. `base_config: hardware_hil.yaml`
2. 显式覆盖 `hil.backend: mock`
3. 显式覆盖 `inference_service.mode: inproc`
4. 显式覆盖 `inference_service.backend: artifact_npz`
5. 将 `n_slow_updates` 缩到 `2`
6. 将 `error_model` 故障概率临时设为 `0.0`
7. 将 `mock_signal.type` 设为 `static`

## 6. 运行命令

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml
```

## 7. 预期输出

运行成功后，会在 `runs/hil_suite/` 下生成一条 `hardware_hil_recovery_smoke_*` 目录，至少包含：

- `hil_summary.json`
- `hil_events.json`

后续复核时，应优先确认：

1. `backend == "mock"`
2. `artifact_path` 指向 `.npz`
3. `inference_service_mode == "inproc"`
4. `n_slow_updates_finished >= 1`
5. `n_commits_applied >= 1`

截至 `2026-05-08`，该路径已在当前机器上完成新的确定性复验：

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 新运行目录：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104`
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104`
- 两次 run 的共同复核结果：
  - `backend = mock`
  - `n_windows_ready = 2`
  - `n_slow_updates_started = 2`
  - `n_slow_updates_finished = 2`
  - `n_commits_requested = 2`
  - `n_commit_acks_observed = 2`
  - `n_commits_applied = 2`
  - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
  - `inference_service_mode = inproc`
  - `final_ler = 0.454375`
  - `overflow_rate = 0.002`
- 逐字比对结果：
  - `hil_summary.json` 的 SHA256 一致
  - `hil_events.json` 的 SHA256 一致

`T12` 对随机源链路的收口说明：

- `run_hil_suite.py`
  - mock noise provider: `seed + 17`
  - physical noise bridge: `seed + 19`
  - driver / mock backend: `seed + 7`
  - slow loop runtime: `seed + 31`
  - host latency injector: `seed + 43`
- `mock_fpga.py`
  - `FastLoopEmulator` 接收派生 seed
- `fast_loop_emulator.py`
  - 快回路误差采样使用 `self._rng = default_rng(seed)`
  - 综合征测量噪声使用独立 `self._measurement_rng = default_rng(seed + 1)`
- `physics/syndrome_measurement.py`
  - `RealisticSyndromeMeasurement` 现可接收显式 `rng`
  - recovery 路径已不再依赖全局 `np.random` 进行测量噪声采样

当前更准确的恢复期表述是：

- 这条 `mock + model_artifact + artifact_npz + inproc` 的 bounded software HIL recovery smoke
- 已在当前机器上完成逐字一致复验
- 但它仍然不是 `real_board` HIL，也不是 `.tflite` runtime 验收

## 8. 这条路径证明了什么

它证明的是：

1. 当前仓库还有一条可以直接执行的软件 HIL 最小入口
2. `run_hil_suite.py -> FPGADriver(mock) -> SlowLoopRuntime(model_artifact)` 这条主链在当前环境上仍能启动并产出摘要
3. 恢复期可以先在不碰 `.tflite`、不碰 `real_board` 的前提下继续推进 P3/P4 入口修复

## 9. 这条路径不证明什么

它不证明：

1. `real_board` 已完成
2. `.tflite` runtime 已完成
3. `tflite_stub_v1` 与真实 `.tflite` 已被重新验收
4. P4 正式 benchmark 已恢复

这些内容应继续留给后续任务：

- `T5`：缓存/生成物噪声治理
- `T7`：重新验收一个 P4 benchmark 最小路径
