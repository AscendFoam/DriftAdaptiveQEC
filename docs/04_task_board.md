# Task Board

## Phase 0: Stabilization

- [x] T0: 冻结 legacy 状态并完成只读审计
- [x] T1: 确认依赖矩阵与最小入口
- [x] T2: 跑通最小 P0 smoke benchmark，或把阻塞固定为可执行修复项
- [x] T3: 审计 HIL / P4 链路中的 mock、stub、placeholder 边界
- [x] T4: 补软件 HIL 最小 bootstrap / smoke test
- [x] T5: 清点并处理仓库中的缓存/生成物噪声治理策略

## Phase 1: Recovery

- [x] T6: 重新验收一个软件 HIL 最小路径
- [x] T7: 重新验收一个 P4 benchmark 最小路径
- [x] T8: 决定是否进入 `Go` 或继续 `Repair`
- [x] T9: 重新验收一个 P4 frozen baseline 单场景全模式 smoke path
- [x] T10: 基于 `T8 + T9` 重新做一次 `Go / Repair` gate review
- [ ] T11: 补一份恢复期最小依赖 manifest（优先覆盖 P0/P3/P4 recovery smoke）

## Current Unique Task

`T11: 补一份恢复期最小依赖 manifest（优先覆盖 P0/P3/P4 recovery smoke）`

当前已知事实：

- `T9` 已把 recovery 级 `P4 frozen baseline` 证据扩到单场景全模式：
  - task package:
    - `docs/tasks/P0/T9_p4_frozen_baseline_single_scenario_all_modes.md`
  - command:
    - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode window_variance --mode ekf --mode cnn_fpga --paired-seeds`
  - run dir:
    - `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732`
  - protocol:
    - `protocol_id = p4_hil_recovery_smoke_v1`
    - `repeats = 1`
    - `seed_pairing = paired`
    - `frozen_baseline_set = static_linear, window_variance, ekf, cnn_fpga`
  - filters:
    - `scenario = static_bias_theta`
    - `modes = static_linear, window_variance, ekf, cnn_fpga`
- `T9` 的 P4 汇总结果已固定：
  - `Static Linear`: `final_ler = 0.99575`, `overflow_rate = 0.00246875`
  - `Window Variance`: `final_ler = 0.57440625`, `overflow_rate = 0.00221875`
  - `EKF`: `final_ler = 0.6795`, `overflow_rate = 0.0019375`
  - `CNN-FPGA`: `final_ler = 0.7248125`, `overflow_rate = 0.00290625`
  - scenario winner: `window_variance`
  - `runner_up_gap = 0.10509375`
- `T9` 底层 repeat HIL summary 已继续确认：
  - 四个 mode 的 repeat summary 都是 `backend = mock`
  - 四个 mode 的 repeat summary 都是 `inference_service_mode = inproc`
  - 四个 mode 的 repeat summary 都有：
    - `n_slow_updates_finished = 8`
    - `n_commits_applied = 8`
  - `static_linear / window_variance / ekf` repeat summary 中：
    - `artifact_path = null`
  - `cnn_fpga` repeat summary 中：
    - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- `T9` 的真实性边界也已固定：
  - 这是 `mock-backed P4 recovery smoke`
  - 不是 `real_board-backed P4`
  - 不是 `.tflite` runtime 验收
  - 不是正式四场景 frozen benchmark 已恢复
  - 当前仍只是 `single-scenario + four-mode + repeats=1`
- `T10` 已完成新的 gate review：
  - review doc:
    - `docs/review/T10_gate_review.md`
  - verdict:
    - `Continue Repair`
  - 当前仍不进入 `Go`` 的主要原因：
    - 根目录仍缺最小依赖 manifest
    - software HIL 仍是“可复验”而非“逐字确定性复现”
    - `T9` 仍只覆盖 `single-scenario + four-mode + repeats=1`
- `T5` 的仓库噪声治理口径仍然生效：
  - `runs/` / `artifacts/` 仅视作历史证据
  - `__pycache__/` / `.pyc` 未来要有界移出版本库，但不在当前任务执行

本任务完成标准：

1. 在仓库根目录新增一份 recovery 期最小依赖 manifest，至少覆盖：
   - `benchmark/compare_full_vs_simplified_ler.py`
   - `python -m cnn_fpga.benchmark.run_hil_suite --config cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
   - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml ...`
2. 显式写清该 manifest 覆盖的是 `P0/P3/P4 recovery smoke`，而不是完整训练链或正式长跑全环境
3. 同步更新治理文档中的环境与复用口径，避免继续完全依赖“本机解释器路径记忆”
4. 如果无法收口，应输出可复现的阻塞与不覆盖范围，而不是扩大修改范围

注意：

- `T11` 不得借补 manifest 顺手改 benchmark 口径、baseline 集合或 ParamMapper 主线语义
- `T11` 不得把 `DLEnv`、`.tflite`、`real_board` 或完整训练链写成“默认已恢复环境”
- `T11` 不得跳过对当前不覆盖范围的显式说明
- `T11` 不得顺手开展新训练、teacher-representation 或真板任务
- `T11` 不得顺手做 `runs/`、`artifacts/`、`__pycache__/` 的大规模清理
