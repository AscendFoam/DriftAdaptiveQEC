# Task Board

## Phase 0: Stabilization

- [x] T0: 冻结 legacy 状态并完成只读审计
- [x] T1: 确认依赖矩阵与最小入口
- [x] T2: 跑通最小 P0 smoke benchmark，或把阻塞固定为可执行修复项
- [x] T3: 审计 HIL / P4 链路中的 mock、stub、placeholder 边界
- [x] T4: 补软件 HIL 最小 bootstrap / smoke test
- [ ] T5: 清点并处理仓库中的缓存/生成物噪声治理策略

## Phase 1: Recovery

- [ ] T6: 重新验收一个软件 HIL 最小路径
- [ ] T7: 重新验收一个 P4 benchmark 最小路径
- [ ] T8: 决定是否进入 `Go` 或继续 `Repair`

## Current Unique Task

`T5: 清点并处理仓库中的缓存/生成物噪声治理策略`

当前已知事实：

- `T1` 已确认恢复期解释器分工
- `T2` 已补 `docs/P0_smoke_bootstrap.md`
- `T3` 已补 `docs/03_hil_p4_boundary_audit.md`
- `T4` 已补 `docs/P3_software_hil_bootstrap.md`
- `T4` 已用 `C:\ProgramData\anaconda3\python.exe` 跑通 `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`
- 当前最小 software HIL 路径已固定为：
  - `hil.backend=mock`
  - `slow_loop.mode=model_artifact`
  - `inference_service.mode=inproc`
  - `inference_service.backend=artifact_npz`
- `cnn_fpga/hwio/board_backend.py` 仍是 placeholder real-board backend
- `.tflite` 路径仍需继续区分真实 runtime 与 stub manifest

本任务完成标准：

1. 列出当前已跟踪缓存、生成物与目录噪声的主要来源
2. 明确哪些应保留、哪些应忽略、哪些应后续清理出版本库
3. 先形成治理策略，再决定是否执行清理动作

注意：

- `T5` 应先以治理策略为主，不要直接做破坏性清理。
