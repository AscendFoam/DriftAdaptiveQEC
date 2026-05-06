# Handoff

## 1. 当前状态

- 日期：`2026-05-06`
- 阶段：`Phase 0: Stabilization`
- 决策：`Repair`
- 当前唯一任务：`T5: 清点并处理仓库中的缓存/生成物噪声治理策略`

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
5. 已实际跑通一条恢复期最小 software HIL smoke
6. 同步更新了治理文档中的 task board、decision log、legacy audit 与风险口径

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
- 运行目录：
  - `runs/hil_suite/hardware_hil_recovery_smoke_20260506_021326_3ae9f9176104`
- 固定口径：
  - backend: `mock`
  - slow-loop mode: `model_artifact`
  - inference service mode: `inproc`
  - inference backend: `artifact_npz`
  - artifact path: `artifacts/models/static_theta_v2/tiny_cnn_20260319_151717_b87c6c227b57.npz`
- 关键结果：
  - `n_slow_updates_finished = 2`
  - `n_commits_applied = 2`
  - `slow_update_violation_rate = 0.0`
  - `fast_cycle_violation_rate = 0.000125`
  - `dma_timeouts = 0`
  - `cnn_infer_failures = 0`

## 4. 当前判断

项目当前的主要阻塞已经从“P0 跑不起来”进一步转为“治理噪声与更正式复验之间的衔接”：

1. 根目录仍缺统一依赖说明文件
2. 仓库中仍有大量缓存、生成物与历史运行噪声需要先立治理规则
3. 真板 backend 仍是 placeholder，不能被写成已验收能力
4. `.tflite` 路径仍必须区分真实 runtime 与 stub 回退
5. `T4` 恢复的是 bootstrap-level software HIL smoke，不等于 `T6/T7` 的正式最小复验

## 5. 已完成任务包

- `T1`：`docs/tasks/P0/T1_environment_and_min_entry.md`
- `T2`：`docs/tasks/P0/T2_smoke_reuse_and_bootstrap.md`
- `T3`：`docs/tasks/P0/T3_hil_p4_boundary_audit.md`
- `T4`：`docs/tasks/P0/T4_software_hil_bootstrap_and_smoke.md`

关键产出：

- `docs/P0_smoke_bootstrap.md`
- `docs/03_hil_p4_boundary_audit.md`
- `docs/P3_software_hil_bootstrap.md`
- `cnn_fpga/config/hardware_hil_recovery_smoke.yaml`

## 6. 下一步建议

下一唯一任务建议为 `T5: 清点并处理仓库中的缓存/生成物噪声治理策略`。

执行前约束：

1. 先以治理策略为主，不要直接做破坏性清理
2. 保留 `T4` 的 software HIL bootstrap 入口，避免清理时误伤当前恢复期最小路径
3. 后续 `T6/T7` 仍要继续显式写清 backend 与 inference artifact type

## 7. 暂不继续的事项

在 `T5/T6/T7` 形成治理与最小复验闭环前，暂不继续：

1. 新的 teacher-representation benchmark 扩展
2. 长时间 P4 正式长跑
3. 真板 backend 能力扩写
