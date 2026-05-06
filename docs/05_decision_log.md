# Decision Log

## D-2026-05-05-01

- 日期：`2026-05-05`
- 决策：项目进入 `Repair`，而不是 `Go`、`Salvage` 或 `Stop`

### 背景

项目已经积累了较多代码、配置和实验结果，但仓库治理文件缺失，默认运行环境也未恢复到可直接复现状态。

### 依据

1. `cnn_fpga/` 与 `physics/` 主代码真实存在
2. P0/P2/P3/P4 入口脚本存在
3. `board_backend.py` 仍是 placeholder 风格，说明项目不能被表述为“已真板完成”
4. 默认 `python 3.13.7` 下最小 benchmark 因缺少 `numpy` 失败，说明当前仍需做入口恢复

### 结论

当前最合理路线是：

- 保留现仓库
- 先恢复治理、依赖、入口和最小验证
- 暂不继续扩功能或扩 teacher-representation 分支

### 直接影响

1. 当前唯一任务切到 `T1: 确认依赖矩阵与最小入口`
2. 恢复期默认不新增实验主线
3. 后续所有结论更新都要以复验结果为前提

## D-2026-05-06-01

- 日期：`2026-05-06`
- 决策：将恢复期推荐最小 smoke 解释器固定为 `C:\ProgramData\anaconda3\python.exe`

### 背景

为完成 `T1`，本轮对当前机器上的多套解释器进行了最小依赖探测与 smoke 验证。

### 依据

1. `C:\Python313\python.exe`
   - 无 `numpy`
   - 不能运行最小 P0 benchmark
2. `C:\ProgramData\anaconda3\python.exe`
   - 有 `numpy + yaml`
   - 已成功跑通：
     - `benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda`
3. `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
   - 有 `torch`
   - 更适合作为后续训练环境候选，而不是恢复期最小 smoke 环境
4. 仓库内未找到 `.venvs/tf311`

### 结论

恢复期当前最稳妥的环境分工为：

1. 最小 smoke / P0：`C:\ProgramData\anaconda3\python.exe`
2. 后续 torch 训练候选：`C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
3. 默认 `python 3.13.7` 不再作为项目运行解释器

### 直接影响

1. `T1` 可标记完成
2. 当前唯一任务切换到 `T2`
3. 后续若补 bootstrap 文档，应以该分工为起点

## D-2026-05-06-02

- 日期：`2026-05-06`
- 决策：将恢复期 P0 smoke 的最终交接口径固定为“轻量解释器 smoke + DLEnv 训练候选”

### 背景

已确认：

1. `C:\ProgramData\anaconda3\python.exe`
   - 有 `numpy + yaml`
   - 已成功跑通最小 P0 smoke
2. `C:\ProgramData\anaconda3\envs\DLEnv\python.exe`
   - 有 `numpy + yaml + torch`
   - 更适合作为后续训练环境候选

### 结论

1. 恢复期最小 smoke 继续用轻量解释器，不切换到 DLEnv
2. `DLEnv` 保留为 legacy 开发常用训练环境候选
3. `docs/P0_smoke_bootstrap.md` 作为当前最小复用说明

### 直接影响

1. `T2` 可标记完成
2. 当前唯一任务切换到 `T3`

## D-2026-05-06-03

- 日期：`2026-05-06`
- 决策：固定 HIL / P4 的恢复期表述口径为“软件 HIL 已有主链，但必须显式标注 backend 与 artifact type；real-board 仍是 placeholder；`tflite_stub_v1` 不等于真实 TFLite 部署”

### 背景

完成 `T3` 边界审计后，已确认当前项目最容易被误写的，不是 P0 环境，而是 P3/P4 链路里的真实实现边界。

### 依据

1. `cnn_fpga/benchmark/run_hil_suite.py`
   - 通过 `hil.backend` 选择 backend
   - `backend == "mock"` 时构造 mock noise provider
   - 真实写出 `hil_events.json` 与 `hil_summary.json`
2. `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`
   - 直接调用 `run_hil_session(...)`
   - P4 benchmark 没有绕开 HIL backend realism 限制
3. `cnn_fpga/hwio/board_backend.py`
   - 文件顶层直接声明为 `Placeholder real-board backend`
   - `schedule_commit(...)` 返回含多个 `None` 字段的占位元信息
   - `step(...)` 仅刷新状态并返回空事件
4. `cnn_fpga/model/export.py` 与 `cnn_fpga/runtime/inference_service.py`
   - 导出路径存在真实 `.tflite` 与 `tflite_stub_v1` 两条分支
   - 推理路径明确区分 `tflite_service` 与 `tflite_stub_service`

### 结论

恢复期后续所有涉及 P3/P4 的描述、报告和复验，都必须显式回答两个问题：

1. backend 是 `mock` 还是 `board`
2. inference artifact 是 `artifact_npz`、真实 `.tflite`，还是 `.tflite.json` stub

### 直接影响

1. `T3` 可标记完成
2. 当前唯一任务切换到 `T4`
3. 后续 `T4/T6/T7` 默认不能把 `real_board` 或 `tflite_stub_v1` 写成已完成真实部署

## D-2026-05-06-04

- 日期：`2026-05-06`
- 决策：将恢复期最小 software HIL bootstrap 固定为“`mock + model_artifact + artifact_npz + inproc`，并使用 `C:\ProgramData\anaconda3\python.exe` 运行 `hardware_hil_recovery_smoke.yaml`”

### 背景

为完成 `T4`，需要把“软件 HIL 可复验”从文档口径落成一条当前机器上实际可跑的最小路径，同时避免把 `.venvs/tf311`、`.tflite` 或 `real_board` 引入恢复期最小入口。

### 依据

1. `cnn_fpga/config/hardware_hil.yaml`
   - 默认 `inference_service.python_executable` 指向 `.venvs/tf311/bin/python`
   - 对当前 Windows 恢复期最小路径不稳妥
2. `artifacts/models/static_theta_v2/`
   - 已存在可直接读取的 `tiny_cnn_... .npz` artifact
3. `cnn_fpga/model/tiny_cnn.py`
   - `tiny_cnn_float` artifact 的推理路径可由 NumPy 实现
4. `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - 已显式固定：
     - `hil.backend=mock`
     - `slow_loop.mode=model_artifact`
     - `inference_service.mode=inproc`
     - `inference_service.backend=artifact_npz`
5. 实际运行结果：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - 运行目录：`runs/hil_suite/hardware_hil_recovery_smoke_20260506_021326_3ae9f9176104`
   - 关键结果：
     - `backend = mock`
     - `n_slow_updates_finished = 2`
     - `n_commits_applied = 2`
     - `artifact_path = ...static_theta_v2...npz`
     - `inference_service_mode = inproc`

### 结论

恢复期最小 software HIL 入口已经恢复，但其结论边界必须保持为：

1. 已恢复的是 `mock-backed software HIL smoke`
2. 不是 `real_board HIL`
3. 不是 `.tflite` runtime 验收
4. 不是 P4 正式 benchmark 验收

### 直接影响

1. `T4` 可标记完成
2. 当前唯一任务切换到 `T5`
3. 后续 `T6` 若要做正式最小复验，应继承同一条 bootstrap 路径或在此基础上明确升级
