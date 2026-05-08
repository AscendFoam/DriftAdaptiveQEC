# Handoff

## 1. 当前状态

- 日期：`2026-05-08`
- 阶段：`Phase 1: Recovery`
- 决策：`Repair`
- 当前唯一任务：`T11: 补一份恢复期最小依赖 manifest（优先覆盖 P0/P3/P4 recovery smoke）`

## 2. 本轮已完成

1. 完成了 `T1`，固定恢复期解释器分工，并跑通最小 P0 smoke
2. 完成了 `T2`，补充了 `docs/P0_smoke_bootstrap.md`
3. 完成了 `T3`，补充了：
   - `docs/tasks/P0/T3_hil_p4_boundary_audit.md`
   - `docs/03_hil_p4_boundary_audit.md`
4. 完成了 `T4`，补充了：
   - `docs/tasks/P0/T4_software_hil_bootstrap_and_smoke.md`
   - `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - `docs/P3_software_hil_bootstrap.md`
5. 完成了 `T5`，补充了：
   - `docs/tasks/P0/T5_repo_noise_governance.md`
   - `docs/06_repo_noise_governance.md`
6. 完成了 `T6`，补充了：
   - `docs/tasks/P0/T6_software_hil_reverification.md`
   - `docs/P3_software_hil_bootstrap.md` 的二次复验证据
7. 完成了 `T7`，补充了：
   - `docs/tasks/P0/T7_p4_benchmark_reverification.md`
   - `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
   - `docs/P4_benchmark_recovery_bootstrap.md`
8. 完成了 `T8`，补充了：
   - `docs/tasks/P0/T8_gate_review_and_phase_decision.md`
   - `docs/review/T8_gate_review.md`
9. 完成了 `T9`，补充了：
   - `docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
   - `docs/P4_benchmark_recovery_bootstrap.md` 的四模式复验证据
10. 完成了 `T10`，补充了：
   - `docs/tasks/P0/T10_gate_review_after_t9.md`
   - `docs/review/T10_gate_review.md`
11. 同步更新了治理文档中的 task board、decision log、legacy audit 与风险口径

## 3. 已验证事实

### 3.1 环境与 P0 smoke

- 默认 `python 3.13.7` 跑最小 benchmark 仍会因缺少 `numpy` 失败
- `C:\ProgramData\anaconda3\python.exe` 已成功跑通：
  - `benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda`
- 当前恢复期解释器分工：
  - 最小 smoke：`C:\ProgramData\anaconda3\python.exe`
  - 训练候选：`C:\ProgramData\anaconda3\envs\DLEnv\python.exe`

### 3.2 HIL / P4 边界

- `run_hil_suite.py`
  - 是 software HIL orchestration 入口
  - 通过 `hil.backend` 选 backend
  - `mock` 路径会构造 mock noise provider，并写出 `hil_events.json` / `hil_summary.json`
- `run_p4_multiscenario_benchmark.py`
  - 直接调用 `run_hil_session(...)`
  - P4 benchmark 的真实性继承自同一条 HIL backend / artifact 链路
- `board_backend.py`
  - 文件顶层直接写明 `Placeholder real-board backend`
  - `schedule_commit(...)` 返回占位元信息
  - `step(...)` 返回空事件列表
- `export.py` + `inference_service.py`
  - 真实 `.tflite` 与 `.tflite.json` stub manifest 两条路径并存
  - runtime 输出会区分 `tflite_service` 与 `tflite_stub_service`

### 3.3 已恢复的最小 software HIL 路径

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 最新复验运行目录：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
- 固定口径：
  - backend: `mock`
  - slow-loop mode: `model_artifact`
  - inference service mode: `inproc`
  - inference backend: `artifact_npz`
  - artifact path: `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- 最新关键结果：
  - `n_windows_ready = 2`
  - `n_slow_updates_finished = 2`
  - `n_commits_applied = 2`
  - `fast_budget_violation = 1`
- 当前表述边界：
  - 该路径已“可复验”
  - 还不应写成“逐字确定性复现”

### 3.4 仓库噪声治理现状

- `.gitignore` 已忽略：
  - `__pycache__/`
  - `runs/`
  - `artifacts/`
- 但当前 Git 历史中仍存在大量已跟踪噪声：
  - 已跟踪缓存/字节码文件：`116`
  - `__pycache__` 目录数：`9`
  - 当前工作区 `.pyc` 总数：`133`
  - 已跟踪 `runs/` 文件：`1841`
  - 已跟踪 `artifacts/` 文件：`110`
- `T5` 已固定恢复期口径：
  - 先治理，后清理
  - `runs/` / `artifacts/` 在恢复期只视作历史证据
  - `__pycache__/` / `.pyc` 需要后续有界 cleanup，但不在当前轮次执行

### 3.5 T7 最小 P4 benchmark 复验结果

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds`
- 新运行目录：
  - `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308`
- 新 protocol / filter 关键结果：
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `repeats = 1`
  - `seed_pairing = paired`
  - `scenario = static_bias_theta`
  - `modes = static_linear, cnn_fpga`
- 新 comparison 关键结果：
  - `Static Linear final_ler = 1.00890625`
  - `Static Linear overflow_rate = 0.0020625`
  - `CNN-FPGA final_ler = 0.72109375`
  - `CNN-FPGA overflow_rate = 0.002375`
  - scenario winner: `cnn_fpga`
  - `runner_up_gap = 0.2878125`
- 新 repeat HIL summary 关键结果：
  - 两个 mode 的 repeat summary 都是 `backend = mock`
  - `cnn_fpga` repeat 中：
    - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
    - `inference_service_mode = inproc`
    - `n_slow_updates_finished = 8`
    - `n_commits_applied = 8`
- 当前表述边界：
  - 该路径是 `mock-backed P4 recovery smoke`
  - 不是 `real_board`
  - 不是 `.tflite` runtime 验收
  - 不是正式四场景四模式 frozen benchmark 已恢复

### 3.6 T8 gate review 结论

- gate review 文档：
  - `docs/review/T8_gate_review.md`
- 结论：
  - `Continue Repair`
- 当前不进入 `Go` 的主要原因：
  - `T7` 仍只覆盖 `single-scenario + two-mode + repeats=1`
  - 根目录仍缺少最小依赖 manifest
  - software HIL 仍是“可复验”而非“逐字确定性复现”
- 当前可以确认的积极结论：
  - 最小 P3/P4 recovery path 都已经重新变成可接力的事实

### 3.7 T9 frozen baseline 单场景全模式 smoke 结果

- 命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds`
- 新运行目录：
  - `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
- 新 protocol / filter 关键结果：
  - `protocol_id = p4_hil_recovery_smoke_v1`
  - `repeats = 1`
  - `seed_pairing = paired`
  - `scenario = static_bias_theta`
  - `modes = static_linear, window_variance, ekf, cnn_fpga`
- 新 comparison 关键结果：
  - `Static Linear final_ler = 0.99575`
  - `Static Linear overflow_rate = 0.00246875`
  - `Window Variance final_ler = 0.57440625`
  - `Window Variance overflow_rate = 0.00221875`
  - `EKF final_ler = 0.6795`
  - `EKF overflow_rate = 0.0019375`
  - `CNN-FPGA final_ler = 0.7248125`
  - `CNN-FPGA overflow_rate = 0.00290625`
  - scenario winner: `window_variance`
  - `runner_up_gap = 0.10509375`
- 新 repeat HIL summary 关键结果：
  - 四个 mode 的 repeat summary 都是 `backend = mock`
  - 四个 mode 的 repeat summary 都是 `inference_service_mode = inproc`
  - 四个 mode 的 repeat summary 都有：
    - `n_slow_updates_finished = 8`
    - `n_commits_applied = 8`
  - `static_linear / window_variance / ekf` repeat 中：
    - `artifact_path = null`
  - `cnn_fpga` repeat 中：
    - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- 当前表述边界：
  - 该路径是 `mock-backed P4 recovery smoke`
  - 不是 `real_board`
  - 不是 `.tflite` runtime 验收
  - 不是正式多场景 frozen benchmark 已恢复
  - 当前仍只是 `single-scenario + four-mode + repeats=1`

### 3.8 T10 gate review 结论

- gate review 文档：
  - `docs/review/T10_gate_review.md`
- 结论：
  - `Continue Repair`
- 当前不进入 `Go` 的主要原因：
  - 根目录仍缺少最小依赖 manifest
  - software HIL 仍是“可复验”而非“逐字确定性复现”
  - `T9` 仍只覆盖 `single-scenario + four-mode + repeats=1`
- 当前可以确认的积极结论：
  - `T9` 已经把 P4 recovery 证据增强到“冻结 baseline 四模式单场景 smoke”
  - 当前仓库更适合先补环境可移植性，而不是继续扩 benchmark 长跑

## 4. 当前判断

项目当前的主要阻塞已经从“P4 最小路径是否还能跑”进一步转为“如何把当前 recovery 级证据收口成更可移植、更可复现的仓库状态”：

1. `T6` 已确认最小 software HIL 路径可复验
2. `T7` 已确认最小 P4 benchmark 路径可复验
3. `T8` 已明确在 `T7` 证据下仍应继续 `Repair`
4. `T9` 已把 P4 recovery 证据扩到 `single-scenario + four-mode + repeats=1`
5. `T10` 已明确在 `T8 + T9` 证据下仍应继续 `Repair`
6. 真板 backend 仍是 placeholder，不能被写成已验收能力
7. `.tflite` 路径仍必须区分真实 runtime 与 stub 回退
8. 当前最适合先收口的是 recovery 期最小依赖 manifest，而不是继续扩长跑

## 5. 已完成任务包

- `T1`：`docs/tasks/P0/T1_environment_and_min_entry.md`
- `T2`：`docs/tasks/P0/T2_smoke_reuse_and_bootstrap.md`
- `T3`：`docs/tasks/P0/T3_hil_p4_boundary_audit.md`
- `T4`：`docs/tasks/P0/T4_software_hil_bootstrap_and_smoke.md`
- `T5`：`docs/tasks/P0/T5_repo_noise_governance.md`
- `T6`：`docs/tasks/P0/T6_software_hil_reverification.md`
- `T7`：`docs/tasks/P0/T7_p4_benchmark_reverification.md`
- `T8`：`docs/tasks/P0/T8_gate_review_and_phase_decision.md`
- `T9`：`docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
- `T10`：`docs/tasks/P0/T10_gate_review_after_t9.md`

关键产出：

- `docs/P0_smoke_bootstrap.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/P3_software_hil_bootstrap.md`
- `docs/06_repo_noise_governance.md`
- `docs/P4_benchmark_recovery_bootstrap.md`
- `docs/review/T8_gate_review.md`
- `docs/review/T10_gate_review.md`
- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`

## 6. 下一步建议

下一唯一任务建议为 `T11: 补一份恢复期最小依赖 manifest（优先覆盖 P0/P3/P4 recovery smoke）`。

执行前约束：

1. 应优先覆盖 recovery 期实际已复验的 `P0/P3/P4` smoke 路径，不追求一次性覆盖完整训练链
2. 应显式写清该 manifest 不覆盖的环境范围，例如 `DLEnv` 训练链、`.tflite`、`real_board`
3. 应同步更新治理文档中的环境与复用口径
4. 不要在 `T11` 顺手做 `runs/`、`artifacts/`、`__pycache__/` 的大规模清理或代码扩展

## 7. 暂不继续的事项

在 `T11` 收口最小依赖 manifest 前，暂不继续：

1. 新的 teacher-representation benchmark 扩展
2. 长时间 P4 正式长跑
3. 真板 backend 能力扩写
4. 大规模 repo cleanup
