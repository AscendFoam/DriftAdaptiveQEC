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
- [ ] T9: 重新验收一个 P4 frozen baseline 单场景全模式 smoke path

## Current Unique Task

`T9: 重新验收一个 P4 frozen baseline 单场景全模式 smoke path`

当前已知事实：

- `T6` 已把恢复期最小 software HIL 路径再次验收到“可复验”状态：
  - run dir: `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104`
  - fixed path: `hil.backend=mock` + `slow_loop.mode=model_artifact` + `inference_service.mode=inproc` + `inference_service.backend=artifact_npz`
  - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- `T7` 已新增恢复期专用 P4 配置与 bootstrap：
  - `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml`
  - `docs/P4_benchmark_recovery_bootstrap.md`
- `T7` 已用 `C:\ProgramData\anaconda3\python.exe` 跑通最小 P4 benchmark：
  - command:
    - `python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark --config cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml --scenario static_bias_theta --mode static_linear --mode cnn_fpga --paired-seeds`
  - run dir:
    - `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308`
  - protocol:
    - `protocol_id = p4_hil_recovery_smoke_v1`
    - `repeats = 1`
    - `seed_pairing = paired`
  - filters:
    - `scenario = static_bias_theta`
    - `modes = static_linear, cnn_fpga`
- `T7` 的 P4 汇总结果已固定：
  - `Static Linear`: `final_ler = 1.00890625`, `overflow_rate = 0.0020625`
  - `CNN-FPGA`: `final_ler = 0.72109375`, `overflow_rate = 0.002375`
  - scenario winner: `cnn_fpga`
  - `runner_up_gap = 0.2878125`
- `T7` 底层 repeat HIL summary 已继续确认：
  - 两个 mode 的 repeat summary 都是 `backend = mock`
  - `cnn_fpga` repeat summary 中：
    - `inference_service_mode = inproc`
    - `artifact_path = artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
    - `n_slow_updates_finished = 8`
    - `n_commits_applied = 8`
  - `static_linear` repeat summary 中：
    - `artifact_path = null`
    - `n_slow_updates_finished = 8`
    - `n_commits_applied = 8`
- `T7` 的真实性边界也已固定：
  - 这是 `mock-backed P4 recovery smoke`
  - 不是 `real_board-backed P4`
  - 不是 `.tflite` runtime 验收
  - 不是正式四场景四模式长跑
- `T8` 已完成一次 gate review：
  - review doc: `docs/review/T8_gate_review.md`
  - 当前结论：`Continue Repair`
  - 当前不进入 `Go` 的主要原因：
    - `T7` 仍只覆盖 `single-scenario + two-mode + repeats=1`
    - 根目录仍缺最小依赖 manifest
    - software HIL 仍是“可复验”而非“逐字确定性复现”
- `T5` 的仓库噪声治理口径仍然生效：
  - `runs/` / `artifacts/` 仅视作历史证据
  - `__pycache__/` / `.pyc` 未来要有界移出版本库，但不在当前任务执行

本任务完成标准：

1. 在 `T7` 已恢复的最小 P4 benchmark 路径基础上，把 frozen baseline 集从两种 mode 扩到四种正式 baseline
2. 继续显式写清 `backend`、`artifact type`、`run dir`、`scenario` 与 `mode` 过滤条件
3. 如果失败，输出可复现的阻塞证据，而不是扩大修改范围

注意：

- `T9` 不得把 recovery smoke 写成正式四场景四模式 frozen benchmark 已恢复
- `T9` 不得把 `mock-backed P4` 写成 `real_board-backed P4`
- `T9` 不得省略 `backend`、`artifact type`、`run dir` 与过滤条件
- `T9` 不得绕过 `docs/06_repo_noise_governance.md` 的治理口径
- `T9` 不得顺手做 `runs/`、`artifacts/`、`__pycache__/` 的大规模清理
