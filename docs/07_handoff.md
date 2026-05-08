# Handoff

## 1. 当前状态

- 日期：`2026-05-08`
- 阶段：`Phase 2: Controlled Development`
- 决策：`Go`
- 当前唯一任务：`T15: P4 multi-scenario frozen baseline bounded smoke`
- 任务包：`docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`

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
11. 完成了 `T11`，补充了：
   - `docs/tasks/P0/T11_recovery_dependency_manifest.md`
   - `requirements-recovery.txt`
   - `docs/P0_smoke_bootstrap.md`、`docs/P3_software_hil_bootstrap.md`、`docs/P4_benchmark_recovery_bootstrap.md` 中对 root manifest 的统一引用
12. 完成了 `T12`，补充了：
   - `docs/tasks/P0/T12_software_hil_determinism_recovery.md`
   - `physics/syndrome_measurement.py`
   - `cnn_fpga/runtime/fast_loop_emulator.py`
   - `docs/P3_software_hil_bootstrap.md` 的确定性复验证据
13. 完成了 `T13`，补充了：
   - `docs/tasks/P0/T13_recovery_exit_and_closeout.md`
   - `docs/review/T13_recovery_exit_review.md`
   - recovery exit 的阶段/状态切换
14. 同步更新了治理文档中的 task board、decision log、legacy audit 与风险口径
15. 作为 Phase 2 Captain 初始化，按 `docs/reference/AI_coding_workflow.md` 校正了 00~08 治理文档，并建立 Phase 2 任务包队列：
   - `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
   - `docs/tasks/Phase2/T15_p4_multiscenario_frozen_smoke.md`
   - `docs/tasks/Phase2/T16_p4_evidence_gate_review.md`
   - `docs/tasks/Phase2/T17_training_manifest_bootstrap.md`
   - `docs/tasks/Phase2/T18_tflite_manifest_and_smoke_plan.md`
   - `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`
   - `docs/tasks/Phase2/T20_real_board_readiness_checklist.md`

## 3. 已验证事实

### 3.1 环境与 P0 smoke

- 默认 `python 3.13.7` 跑最小 benchmark 仍会因缺少 `numpy` 失败
- `C:\ProgramData\anaconda3\python.exe` 已成功跑通：
  - `benchmark/compare_full_vs_simplified_ler.py --n-rounds 10 --repeats 2 --no-plot --output-dir runs/smoke_test_anaconda`
- 根目录现已新增：
  - `requirements-recovery.txt`
  - 当前覆盖包集：`numpy + PyYAML`
  - 当前覆盖范围：`P0/P3/P4 recovery smoke`
  - 当前不覆盖：`DLEnv` 训练链、`.tflite` runtime / export、`real_board` HIL backend
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
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104`
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104`
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
  - `final_ler = 0.454375`
  - `overflow_rate = 0.002`
- 当前表述边界：
  - 该路径已完成逐字一致复验
  - 仍不应写成 `real_board` 或正式多场景 benchmark 已恢复

### 3.4 T12 确定性复验结果

- 两次连续复验命令：
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
  - `& 'C:\ProgramData\anaconda3\python.exe' -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 对比 run dir：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172221_3ae9f9176104`
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260508_172232_3ae9f9176104`
- 对比结果：
  - `hil_summary.json` 的 SHA256 一致
  - `hil_events.json` 的 SHA256 一致
- 最小修复说明：
  - `RealisticSyndromeMeasurement` 现在支持注入显式 `rng`
  - `FastLoopEmulator` 将快回路噪声 RNG 与测量噪声 RNG 分开，并沿 seed 链显式传递
  - recovery 路径已不再依赖综合征测量中的全局 `np.random`

### 3.5 仓库噪声治理现状

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

### 3.6 T7 最小 P4 benchmark 复验结果

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

### 3.7 T8 gate review 结论

- gate review 文档：
  - `docs/review/T8_gate_review.md`
- 结论：
  - `Continue Repair`
- 当前不进入 `Go` 的主要原因：
  - `T7` 仍只覆盖 `single-scenario + two-mode + repeats=1`
  - 根目录仍缺少最小依赖 manifest
  - 当时 software HIL 仍是“可复验”而非“逐字确定性复现”
- 当前可以确认的积极结论：
  - 最小 P3/P4 recovery path 都已经重新变成可接力的事实

### 3.8 T9 frozen baseline 单场景全模式 smoke 结果

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

### 3.9 T10 gate review 结论

- gate review 文档：
  - `docs/review/T10_gate_review.md`
- 结论：
  - `Continue Repair`
- 当前不进入 `Go` 的主要原因：
  - 根目录仍缺少最小依赖 manifest
  - 当时 software HIL 仍是“可复验”而非“逐字确定性复现”
  - `T9` 仍只覆盖 `single-scenario + four-mode + repeats=1`
- 当前可以确认的积极结论：
  - `T9` 已经把 P4 recovery 证据增强到“冻结 baseline 四模式单场景 smoke”
  - 当前仓库更适合先补环境可移植性，而不是继续扩 benchmark 长跑

### 3.10 T11 recovery 期最小依赖 manifest 结果

- task package：
  - `docs/tasks/P0/T11_recovery_dependency_manifest.md`
- 根目录 manifest：
  - `requirements-recovery.txt`
- manifest 当前包含：
  - `numpy`
  - `PyYAML`
- manifest 当前覆盖：
  - `benchmark/compare_full_vs_simplified_ler.py --no-plot`
  - `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
  - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml ...`
- manifest 当前不覆盖：
  - `torch` 训练链 / `DLEnv`
  - `tensorflow` / `tflite-runtime`
  - `.tflite` export/runtime
  - `real_board` HIL backend
  - 去掉 `--no-plot` 后的 `matplotlib`
- 文档同步结果：
  - `README.md`
  - `docs/P0_smoke_bootstrap.md`
  - `docs/P3_software_hil_bootstrap.md`
  - `docs/P4_benchmark_recovery_bootstrap.md`
  都已改为显式引用 `requirements-recovery.txt`

## 4. 当前判断

项目当前判断已经从“是否还能退出 Recovery”切换为“在继续开发前，下一张 bounded 任务包应该优先补哪块正式证据或环境说明”：

1. `T6` 已确认最小 software HIL 路径可复验
2. `T7` 已确认最小 P4 benchmark 路径可复验
3. `T8` 已明确在 `T7` 证据下仍应继续 `Repair`
4. `T9` 已把 P4 recovery 证据扩到 `single-scenario + four-mode + repeats=1`
5. `T10` 已明确在 `T8 + T9` 证据下仍应继续 `Repair`
6. `T11` 已把 recovery 期最小依赖 manifest 收口到可接力状态
7. 真板 backend 仍是 placeholder，不能被写成已验收能力
8. `.tflite` 路径仍必须区分真实 runtime 与 stub 回退
9. `T12` 已把 bounded software HIL recovery smoke 收口到逐字一致复验
10. `T13` 已确认 recovery exit 条件满足，项目可进入受控继续开发
11. Phase 2 首个任务选为 `T14`，先审计 P4 frozen benchmark protocol 和 bounded run plan，不直接启动长跑
12. `T15` 是否运行多场景 bounded smoke，必须等待 `T14` 产出明确 run matrix 后再执行

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
- `T11`：`docs/tasks/P0/T11_recovery_dependency_manifest.md`
- `T12`：`docs/tasks/P0/T12_software_hil_determinism_recovery.md`
- `T13`：`docs/tasks/P0/T13_recovery_exit_and_closeout.md`

关键产出：

- `requirements-recovery.txt`
- `docs/P0_smoke_bootstrap.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/P3_software_hil_bootstrap.md`
- `docs/06_repo_noise_governance.md`
- `docs/P4_benchmark_recovery_bootstrap.md`
- `docs/review/T8_gate_review.md`
- `docs/review/T10_gate_review.md`
- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`

## 6. 当前唯一任务包摘要

Task ID: `T14`

Goal: 审计 P4 frozen benchmark 的正式口径与 recovery smoke 口径，产出可指导 `T15` 的 bounded run plan。

Allowed files:

- `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`
- `docs/P4_benchmark_development_protocol.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Forbidden scope:

- 不修改 benchmark runner、config、ParamMapper 或 baseline 集合
- 不启动正式长跑 benchmark
- 不把 `mock-backed` 结果写成 `real_board` 或 `.tflite` 验收

Verification:

- 只读审计 P4 config、runner 参数与既有 run evidence
- 不要求产生新 `runs/` 结果

Current `T14` output status:

- `docs/P4_benchmark_development_protocol.md` has now been added as the Phase 2 P4 development protocol.
- `docs/review/T14_protocol_audit_review.md` verdict = `PASS`
- Non-blocking review notes were accepted and addressed:
  - `T14` worker output summary now lists all modified docs.
  - `docs/P4_benchmark_development_protocol.md` now explains the strong-baseline config inheritance from `p4_multiscenario_hybrid_b_long.yaml`.
- The document fixes three layers of wording:
  - recovery smoke
  - development bounded run
  - formal frozen benchmark
- It also fixes the recommended `T15` bounded matrix:
  - scenarios: `static_bias_theta`, `linear_ramp`
  - modes: `ekf`, `ukf`, `constant_residual_mu`, `rls_residual_b`, `hybrid_residual_b`
  - repeats: `2`
  - seed policy: `--paired-seeds`
  - interpreter: `C:\ProgramData\anaconda3\python.exe`
  - config: `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml`
- `T14` does not reopen the full four-scenario formal run and does not change benchmark semantics.

## 7. 下一步建议

第一轮 recovery 收尾已完成，Phase 2 当前按任务板顺序推进。

建议优先级：

1. 现在可以进入 `T15`，但只能使用 `docs/P4_benchmark_development_protocol.md` 中定义的双场景、五模式、`repeats=2` bounded matrix
2. `T15` 完成后应进入 `T16` gate review，而不是继续扩大 benchmark
3. 之后再单开训练链 / `.tflite` / 真板路径的独立 manifest 与边界任务
4. 继续保持 `mock` / `.tflite` / `real_board` 边界表述诚实
5. 仍不要顺手做 `runs/`、`artifacts/`、`__pycache__/` 的大规模清理或无界扩功能

## 8. 暂不继续的事项

在新任务包明确之前，暂不继续：

1. 新的 teacher-representation benchmark 扩展
2. 长时间 P4 正式长跑
3. 真板 backend 能力扩写
4. 大规模 repo cleanup
