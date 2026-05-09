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

## D-2026-05-07-01

- 日期：`2026-05-07`
- 决策：将仓库噪声治理口径固定为“先分类治理、暂不破坏性清理；缓存/字节码后续移出版本库；`runs/` 与 `artifacts/` 在恢复期暂作历史证据保留”

### 背景

`T4` 已恢复最小 software HIL bootstrap，但仓库内仍混有大量历史缓存文件与实验输出。如果不先固定这些内容在恢复期的语义边界，`T6/T7` 的复验结果会继续与历史产物混淆。

### 依据

1. `.gitignore` 已忽略：
   - `__pycache__/`
   - `runs/`
   - `artifacts/`
2. 但 Git 中仍已有大量历史已跟踪文件：
   - 已跟踪缓存/字节码文件：`116`
   - 已跟踪 `runs/` 文件：`1841`
   - 已跟踪 `artifacts/` 文件：`110`
3. 当前工作区仍可见 `9` 个 `__pycache__` 目录，`.pyc` 总数为 `133`
4. 当前最小 software HIL 路径仍依赖 `artifacts/models/static_theta_v2/...npz`，说明不能在恢复期直接粗暴清空历史产物

### 结论

恢复期对仓库噪声的统一口径固定为：

1. `__pycache__/` / `.pyc`
   - 属于应最终移出版本库的历史噪声
   - 但 `T5` 不直接做批量删除或 untrack
2. `runs/`
   - 视为历史运行证据
   - 后续文档必须引用具体 run dir，不能把整个目录写成新的事实来源
3. `artifacts/`
   - 先保留当前 bootstrap 所需 artifact
   - 后续再拆分“bootstrap 必需”与“历史归档”
4. 本地临时文档文件
   - 立即通过 `.gitignore` 继续约束
   - 例如 `*.drawio.dtmp`

### 直接影响

1. `T5` 可标记完成
2. 新增 `docs/06_repo_noise_governance.md`
3. 当前唯一任务切换到 `T6`
4. 在专门 cleanup 任务出现前，后续任务默认不做 `runs/`、`artifacts/`、`__pycache__/` 的破坏性清理

## D-2026-05-07-02

- 日期：`2026-05-07`
- 决策：将恢复期最小 software HIL 路径提升为“已二次复验通过，但暂按可复验而非逐字确定性复现表述”，并将当前唯一任务切换到 `T7`

### 背景

`T4` 已经恢复出一条最小 software HIL bootstrap 路径，但要进入 `Phase 1: Recovery`，还需要确认它不是一次性成功，也没有被历史结果语义掩盖。

### 依据

1. 使用同一命令再次运行：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
2. 新运行目录：
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
3. 新 summary 结果再次确认：
   - `backend = mock`
   - `n_windows_ready = 2`
   - `n_slow_updates_started = 2`
   - `n_slow_updates_finished = 2`
   - `n_commits_applied = 2`
   - `artifact_path = ...static_theta_v2...npz`
   - `inference_service_mode = inproc`
4. 新 events 结果再次确认：
   - `window_ready = 2`
   - `slow_update_started = 2`
   - `slow_update_finished = 2`
   - `commit_applied = 2`
   - `commit_ack_asserted = 2`
   - `fast_budget_violation = 1`
5. 两次复验的 control-plane 字段一致，但 `final_ler` 与 `overflow_rate` 存在小幅差异
6. 代码侧仍可见一部分全局 `np.random` 路径，例如：
   - `physics/syndrome_measurement.py`

### 结论

恢复期当前可以把这条最小 software HIL 路径表述为：

1. `mock-backed software HIL` 已完成二次复验
2. `backend`、`artifact_path`、`inference_service_mode` 已再次固定
3. 当前更适合表述为“可复验”
4. 暂不把它表述为“逐字确定性复现”

### 直接影响

1. `T6` 可标记完成
2. `docs/P3_software_hil_bootstrap.md` 应更新为包含最新 run 证据
3. 当前唯一任务切换到 `T7`
4. `T7` 必须继承同一条 `mock + model_artifact + artifact_npz + inproc` 口径

## D-2026-05-08-01

- 日期：`2026-05-08`
- 决策：将恢复期最小 P4 benchmark 路径固定为“`mock-backed P4 recovery smoke`，显式复用 `T6` 的 software HIL 主链，并将当前唯一任务切换到 `T8`”

### 背景

`T6` 已把最小 software HIL 路径重新验收到“可复验”状态，但 `run_p4_multiscenario_benchmark.py` 还没有在同一套 `mock + model_artifact + artifact_npz + inproc` 口径下完成新的最小复验。

### 依据

1. 新增恢复期专用 P4 配置：
   - `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
   - 显式固定：
     - `hil.backend=mock`
     - `slow_loop.inference_service.mode=inproc`
     - `slow_loop.inference_service.backend=artifact_npz`
     - `slow_loop.model_artifact.path=...static_theta_v2...npz`
2. 新增恢复期专用 bootstrap 文档：
   - `docs/P4_benchmark_recovery_bootstrap.md`
3. 实际运行命令：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds`
4. 新运行目录：
   - `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308`
5. 新 summary 结果确认：
   - `protocol_id = p4_hil_recovery_smoke_v1`
   - `scenario = static_bias_theta`
   - `modes = static_linear, cnn_fpga`
   - `seed_pairing = paired`
6. 新 comparison 结果确认：
   - `Static Linear final_ler = 1.00890625`
   - `CNN-FPGA final_ler = 0.72109375`
   - `CNN-FPGA artifact_path = ...static_theta_v2...npz`
7. 新 repeat HIL summary 再次确认：
   - `backend = mock`
   - `n_slow_updates_finished = 8`
   - `n_commits_applied = 8`
   - `cnn_fpga` repeat 中 `inference_service_mode = inproc`

### 结论

恢复期当前可以把这条 P4 路径表述为：

1. `mock-backed P4 benchmark minimal recovery path` 已完成一次新的复验
2. 它显式复用了 `T6` 已固定的软件 HIL 主链
3. 当前更适合表述为“P4 recovery smoke 已可复验”
4. 暂不把它表述为“正式四场景四模式 frozen benchmark 已恢复”

### 直接影响

1. `T7` 可标记完成
2. `docs/P4_benchmark_recovery_bootstrap.md` 应更新为包含最新 run 证据
3. 当前唯一任务切换到 `T8`
4. `T8` 需要基于 `T6 + T7` 的证据，决定项目继续 `Repair` 还是进入 `Go`

## D-2026-05-08-02

- 日期：`2026-05-08`
- 决策：基于 `T6 + T7` 的现有证据，项目继续保持 `Repair`，不立即进入 `Go`

### 背景

`T8` 的目标不是再跑一个新 benchmark，而是判断当前仓库是否已经从“恢复可信度”跨过了进入正常开发的门槛。

### 依据

1. `T6` 已确认最小 software HIL 路径可复验：
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
2. `T7` 已确认最小 P4 benchmark 路径可复验：
   - `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308`
3. 但 `T7` 当前仍只覆盖：
   - `single-scenario`
   - `two-mode`
   - `repeats = 1`
4. 根目录仍缺少最小依赖 manifest：
   - 无 `requirements.txt`
   - 无 `pyproject.toml`
   - 无 `environment.yml`
5. `T6` 仍保留“可复验而非逐字确定性复现”的随机性观察
6. `docs/review/T8_gate_review.md` 已明确给出 verdict：
   - `Continue Repair`

### 结论

当前最合理的 gate 判断是：

1. 项目已经进入 `Phase 1: Recovery`
2. 最小 P3/P4 recovery path 已经恢复
3. 但现阶段证据还不足以把仓库决策状态切到 `Go`

### 直接影响

1. `T8` 可标记完成
2. 当前决策状态继续保持 `Repair`
3. 当前唯一任务切换到 `T9`
4. `T9` 应优先扩大 P4 frozen baseline 的 recovery 证据，而不是重新发散到新功能或长跑

## D-2026-05-08-03

- 日期：`2026-05-08`
- 决策：将恢复期 `P4 frozen baseline` 证据提升为“`single-scenario + four-mode + repeats=1` smoke 已复验”，并将当前唯一任务切换到 `T10`

### 背景

`T8` 的 gate review 已明确指出，`T7` 的 `single-scenario + two-mode + repeats=1` 证据还不足以支撑进入 `Go`。因此先执行 `T9`，在不扩到正式多场景长跑的前提下，把单场景 mode 集补齐到冻结的四类 baseline。

### 依据

1. `T9` task package 已固定：
   - `docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
2. 实际运行命令：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds`
3. 新运行目录：
   - `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
4. `summary.json` 已确认：
   - `protocol_id = p4_hil_recovery_smoke_v1`
   - `scenario = static_bias_theta`
   - `modes = static_linear, window_variance, ekf, cnn_fpga`
   - `seed_pairing = paired`
5. `comparison.csv` 已确认：
   - `Static Linear final_ler = 0.99575`
   - `Window Variance final_ler = 0.57440625`
   - `EKF final_ler = 0.6795`
   - `CNN-FPGA final_ler = 0.7248125`
   - scenario winner: `window_variance`
6. 各 mode 的 repeat HIL summary 已继续确认：
   - 四个 mode 都是 `backend = mock`
   - 四个 mode 都是 `inference_service_mode = inproc`
   - `static_linear / window_variance / ekf` 的 `artifact_path = null`
   - `cnn_fpga` 的 `artifact_path = ...static_theta_v2...npz`
7. 当前边界仍然成立：
   - 这仍是 `mock-backed P4 recovery smoke`
   - 不是 `real_board`
   - 不是 `.tflite` runtime 验收
   - 不是正式多场景 frozen benchmark

### 结论

当前最合理的仓库状态表述是：

1. `T9` 已完成
2. 恢复期 `P4` 证据已从“两模式最小路径”扩到“冻结 baseline 四模式单场景 smoke”
3. 但 `T9` 本身还不直接改变 `Repair / Go` 决策状态
4. 下一唯一任务应切换到 `T10`，对 `T8 + T9` 的组合证据重新做一次 gate review

### 直接影响

1. `T9` 可标记完成
2. `docs/P4_benchmark_recovery_bootstrap.md` 应更新为包含最新四模式 run 证据
3. 当前唯一任务切换到 `T10`
4. `T10` 应优先回答“`T9` 是否足以支撑进入 `Go`”，而不是继续扩 benchmark 或顺手扩功能

## D-2026-05-08-04

- 日期：`2026-05-08`
- 决策：基于 `T8 + T9` 的组合证据，项目继续保持 `Repair`，不进入 `Go`，并将当前唯一任务切换到 `T11`

### 背景

`T9` 已经把 `P4 frozen baseline` 的 recovery 证据从“两模式最小路径”扩到“单场景四模式 smoke”。`T10` 的目标就是正式判断：这组增强后的证据，是否已经足以把仓库从 `Repair` 切到 `Go`。

### 依据

1. `T10` gate review 文档已完成：
   - `docs/review/T10_gate_review.md`
2. `T9` 已确认：
   - `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
   - `single-scenario + four-mode + repeats=1`
   - `backend = mock`
   - `cnn_fpga artifact_path = ...static_theta_v2...npz`
3. `T6` 已确认最小 software HIL 路径可复验，但仍不是逐字确定性复现：
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
4. 根目录当前仍无最小依赖 manifest：
   - `requirements.txt`
   - `pyproject.toml`
   - `environment.yml`
5. 因此当前仍存在三个关键缺口：
   - 环境可移植性未收口
   - software HIL 确定性复现未收口
   - P4 仍未从单场景 smoke 提升到正式多场景 frozen benchmark

### 结论

当前最合理的 gate 判断是：

1. 项目继续保持在 `Phase 1: Recovery`
2. 决策状态继续维持 `Repair`
3. `T9` 虽然显著增强了 P4 recovery 证据，但还不足以单独支撑进入 `Go`
4. 下一唯一任务应先切到 `T11`，补 recovery 期最小依赖 manifest

### 直接影响

1. `T10` 可标记完成
2. 当前决策状态继续保持 `Repair`
3. 当前唯一任务切换到 `T11`
4. `T11` 应优先收口 recovery smoke 的依赖 manifest，而不是继续扩 benchmark 长跑或新功能

## D-2026-05-08-05

- 日期：`2026-05-08`
- 决策：在根目录新增 `requirements-recovery.txt` 作为 recovery-scoped 最小依赖 manifest，并显式保持它不是完整仓库环境文件

### 背景

`T10` 的 gate review 已明确指出：当前最容易收口、也最影响接力效率的缺口，就是根目录仍没有一份作用域诚实的最小依赖说明。`T11` 的目标就是把已经复验过的 `P0/P3/P4 recovery smoke` 路径，收口成一个可直接引用的根级 manifest。

### 依据

1. 当前已复验的 recovery 入口只有三类：
   - `benchmark/compare_full_vs_simplified_ler.py --no-plot`
   - `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml ...`
2. 这些 recovery 入口的第三方依赖边界当前可收口到：
   - `numpy`
   - `PyYAML`
3. `benchmark/compare_full_vs_simplified_ler.py` 中的 `matplotlib` 只在去掉 `--no-plot` 时才会触发，因此不应被写成 recovery smoke 的必需依赖
4. `torch`、`tensorflow`、`tflite-runtime` 与 `real_board` backend 只属于训练链、`.tflite` 或真板路径，不属于当前已复验的 recovery smoke 默认范围
5. 当前推荐解释器分工仍成立：
   - recovery smoke：`C:\ProgramData\anaconda3\python.exe`
   - 训练候选：`C:\ProgramData\anaconda3\envs\DLEnv\python.exe`

### 结论

恢复期当前采用如下依赖口径：

1. 根目录新增 `requirements-recovery.txt`
2. 它只覆盖 `P0/P3/P4 recovery smoke`
3. 它故意不命名为 `requirements.txt`，避免被误读为“完整仓库环境已恢复”
4. 它故意不引入 `torch`、`tensorflow`、`tflite-runtime`、`matplotlib`
5. 训练链、`.tflite` 路径与真板路径仍需继续按恢复期边界单独说明

### 直接影响

1. `T11` 可标记完成
2. `README.md` 与 `docs/P0_smoke_bootstrap.md`、`docs/P3_software_hil_bootstrap.md`、`docs/P4_benchmark_recovery_bootstrap.md` 应改为显式引用 `requirements-recovery.txt`
3. 根目录“完全没有 manifest”的风险已收口为“只有 recovery-scoped manifest，完整环境仍未统一”
4. 当前唯一任务切换到 `T12`
5. `T12` 应优先收口 software HIL recovery smoke 的随机源与确定性表述，而不是继续扩 benchmark 长跑或新功能

## D-2026-05-08-06

- 日期：`2026-05-08`
- 决策：将恢复期最小 software HIL recovery smoke 的表述从“可复验”提升为“在固定 seed 链路下已完成逐字一致复验”，但严格限定在 `mock + model_artifact + artifact_npz + inproc` 这条 bounded 路径内

### 背景

`T11` 已经补齐 recovery-scoped manifest，但 `T6/T10` 仍保留一个关键缺口：`hardware_hil_recovery_smoke` 的 control-plane 字段虽然稳定，`final_ler` 与 `overflow_rate` 仍存在小幅 run-to-run 差异。`T12` 的目标就是在不改 benchmark 主线语义的前提下，把这个 bounded recovery path 的随机源收口到更强的可复现状态。

### 依据

1. `physics/syndrome_measurement.py`
   - `RealisticSyndromeMeasurement` 现支持注入显式 `rng`
   - recovery 路径的测量噪声、shot noise 与 ancilla 扰动不再依赖全局 `np.random`
2. `cnn_fpga/runtime/fast_loop_emulator.py`
   - 快回路误差采样继续使用 `default_rng(seed)`
   - 综合征测量额外拆出 `default_rng(seed + 1)`
   - 两类随机源已显式分离且都挂在同一条 seed 链上
3. `run_hil_suite.py` 当前 recovery seed 链路已明确：
   - mock noise provider: `seed + 17`
   - physical noise bridge: `seed + 19`
   - driver / mock backend: `seed + 7`
   - slow loop runtime: `seed + 31`
   - host latency injector: `seed + 43`
4. 使用同一命令连续复验两次：
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
5. 两次新运行目录分别为：
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104`
   - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104`
6. 文件级对比结果：
   - `hil_summary.json` SHA256 完全一致
   - `hil_events.json` SHA256 完全一致
7. 两次 run 的共同关键结果：
   - `backend = mock`
   - `artifact_path = ...static_theta_v2...npz`
   - `inference_service_mode = inproc`
   - `n_slow_updates_finished = 2`
   - `n_commits_applied = 2`
   - `final_ler = 0.454375`
   - `overflow_rate = 0.002`

### 结论

恢复期当前可以把这条最小 software HIL 路径表述为：

1. `mock-backed software HIL recovery smoke` 已完成逐字一致复验
2. 它的确定性结论仅限于当前 bounded 路径与当前机器上的 seed 链
3. 这不等于 `real_board` 已恢复
4. 这也不等于 `.tflite` runtime 或正式多场景 P4 benchmark 已恢复

### 直接影响

1. `T12` 可标记完成
2. `docs/P3_software_hil_bootstrap.md` 应更新为包含新的确定性复验证据与随机源链路说明
3. `docs/01_legacy_audit.md`、`docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 应同步移除“该路径仍有 run-to-run 数值漂移”的旧表述
4. 当前唯一任务清空，等待下一张任务包

## D-2026-05-08-07

- 日期：`2026-05-08`
- 决策：第一轮 `Phase 1: Recovery` 收尾完成，项目退出恢复期，进入 `Phase 2: Controlled Development`，决策状态切换为 `Go`

### 背景

`T10` 时点仍不能进入 `Go`，因为当时还缺少 recovery-scoped manifest 与 bounded software HIL 的确定性收口。`T11` 和 `T12` 完成后，需要再做一次正式的 recovery exit review，判断当前仓库是否已经满足工作流里“Recovery-Ready”的退出条件。

### 依据

1. `docs/reference/AI_coding_workflow.md` 第 4 节迁移/恢复工作流给出的退出标准，当前已满足：
   - 最小路径可按文档运行
   - MVP 范围明确
   - task board / handoff / risks 稳定
   - smoke / 等价验证存在
   - mock / stub / placeholder 已标记
   - 不再依赖旧长 session 才能理解仓库状态
2. `T11` 已完成：
   - 根目录新增 `requirements-recovery.txt`
   - recovery 期最小依赖边界已固定
3. `T12` 已完成：
   - bounded software HIL recovery smoke 已逐字一致复验
4. `P0/P3/P4` 的 bounded recovery 入口都已有新的文档化证据：
   - `docs/P0_smoke_bootstrap.md`
   - `docs/P3_software_hil_bootstrap.md`
   - `docs/P4_benchmark_recovery_bootstrap.md`
5. `docs/review/T13_recovery_exit_review.md`
   - 已给出 verdict：`Allow`

### 结论

当前最合理的仓库状态表述是：

1. 第一轮恢复期目标已完成
2. 项目可以从 `Repair` 切换到 `Go`
3. 这里的 `Go` 只代表“允许继续做 bounded 开发任务”
4. 它不代表：
   - `real_board` 已恢复
   - 真实 `.tflite` runtime 已恢复
   - 正式多场景 frozen benchmark 已恢复

### 直接影响

1. `AGENTS.md` 与 `README.md` 应切换到新的阶段与决策状态
2. `docs/04_task_board.md`、`docs/07_handoff.md` 应标记 recovery 已收尾，当前唯一任务待重新定义
3. 后续任务默认从“恢复期修复”切换为“受控继续开发”，但仍保留真实边界与验证硬规则

## D-2026-05-08-08

- 日期：`2026-05-08`
- 决策：按 AI Coding 工作流重新初始化 Phase 2 任务板，并将当前唯一任务设为 `T14: P4 frozen benchmark protocol audit and bounded run plan`

### 背景

`T13` 已完成 recovery exit，项目进入 `Phase 2: Controlled Development`。但 `docs/04_task_board.md` 仍处于“下一任务待定义”状态，无法直接指导后续 Worker 会话。

### 依据

1. `docs/reference/AI_coding_workflow.md` 要求 Captain 明确：
   - 当前唯一任务
   - Worker 任务包
   - Allowed files
   - Forbidden scope
   - Verification
   - Docs to update
2. `docs/02_experiment_plan.md` 的后续优先级和禁止项同时表明：
   - P4 多场景证据重要
   - 但不应无准备启动长时间正式 benchmark
3. 当前已有证据边界：
   - `T9` 只覆盖 `single-scenario + four-mode + repeats=1`
   - `T12` 只覆盖 bounded software HIL deterministic recovery smoke
   - `real_board` 与 `.tflite` 路径仍不能被外推

### 结论

Phase 2 先按以下顺序推进：

1. `T14`: P4 frozen benchmark protocol audit
2. `T15`: P4 multi-scenario frozen baseline bounded smoke
3. `T16`: P4 evidence gate review
4. `T17` / `T18`: training 与 `.tflite` 独立 manifest
5. `T19` / `T20`: repo cleanup manifest 与 real-board readiness

当前唯一任务为 `T14`，本任务只做 protocol audit 和 bounded run plan，不启动 benchmark 长跑。

### 直接影响

1. `docs/04_task_board.md` 成为 Phase 2 Worker 会话的任务主状态。
2. `docs/tasks/Phase2/` 新增 `T14` 至 `T20` 的任务包。
3. `docs/07_handoff.md` 与 `docs/08_risks_and_open_questions.md` 同步更新为 `T14` 当前任务口径。
4. 后续 Worker 不应绕过 `T14` 直接执行 `T15` 或长跑 benchmark。

## D-2026-05-09-01

- 日期：`2026-05-09`
- 决策：接受 `T15` 的 `PASS_WITH_WARNINGS` review，标记 `T15` 完成，并将当前唯一任务切换为 `T16: P4 benchmark evidence review and next-gate decision`

### 背景

`T15` 已按 `T14` 固定的 bounded matrix 完成双场景、五模式、`repeats=2` 的 P4 development run。`docs/review/T15_frozen_smoke_review.md` 给出 verdict：`PASS_WITH_WARNINGS`，且没有 blocking issue。

### Warning 分类

1. N1: handoff 多个状态节未同步更新
   - 分类：`accepted`
   - 处理：Captain 在本次整合中更新 `docs/04_task_board.md` 与 `docs/07_handoff.md`
2. N2: `hybrid_residual_b` teacher diagnostics 全零
   - 分类：`deferred`
   - 处理：写入 `docs/08_risks_and_open_questions.md` 的 R10，并要求 `T16` gate review 判断
3. N3: `delta_rows` 全部为 null
   - 分类：`accepted`
   - 处理：记录为 strong-baseline config 不包含 `static_linear` / `cnn_fpga` 的预期后果，提醒 `T16` 不要误判

### 依据

1. `T15` matrix 与 `T14` protocol 完全匹配：
   - scenarios: `static_bias_theta`, `linear_ramp`
   - modes: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
   - repeats: `2`
   - seed policy: `paired`
2. 新 run dir：
   - `runs/p4_benchmark/p4multis_20260508_221718_b82874_48280`
3. Review 已确认：
   - `missing_runs = []`
   - 10 个 comparison rows 均 `coverage = 1.0`
   - no forbidden scope violation
   - no code / config / ParamMapper changes

### 结论

`T15` 可以标记完成，但其结果只升级为 `development bounded run` evidence，不升级为正式四场景 frozen benchmark 结论。

下一唯一任务为 `T16`。`T16` 只做 gate review，不运行新 benchmark。

### 直接影响

1. `docs/04_task_board.md` 标记 `T15` 完成，并切换 `Current Unique Task` 到 `T16`。
2. `docs/07_handoff.md` 补齐 `T14/T15` 完成记录、T15 evidence 与当前任务摘要。
3. `docs/08_risks_and_open_questions.md` 增加 R10，并记录 warning 分类。
4. 在 `T16` 完成前，不继续扩大 P4 benchmark。

## D-2026-05-09-02

- ???`2026-05-09`
- ????? `T16` milestone review ? `PASS_WITH_WARNINGS`????? `T16` gate verdict?????? `T17` ????????

### ??

`T16` gate review ????verdict ? `Conditional`??? milestone review ??? `PASS_WITH_WARNINGS`???? blocking issue?

### Warning ??

1. N1: `T16` review ????????? handoff / task board ??
   - ???`accepted`
   - ?????? `docs/04_task_board.md` ? `docs/07_handoff.md` ??
2. N2: `T16` review ??????R5/R9 ?????????
   - ???`deferred`
   - ???????? `docs/08_risks_and_open_questions.md`????????

### ??

1. `T16` gate verdict ??? `Conditional`?????????????????? P4 benchmark?
2. `T16` milestone review ???? blocking issue?
3. ??????????????? P4?????????? manifest/bootstrap `T17`?

### ??

`T16` milestone review ????? gate ?????? Captain ??????????????? `T17`?

### ????

1. `docs/04_task_board.md` ??????? `T17`?
2. `docs/07_handoff.md` ?????????? `T17`?
3. `docs/08_risks_and_open_questions.md` ?? R10??? R11??? `T16` ? warning ??????????
