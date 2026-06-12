# 并行 Sidecar 扩展实验治理

## 1. 状态

- 日期：`2026-06-12`
- 当前版本：`PSE1`
- 范围：docs-only sidecar 治理刷新
- 作用：把 sidecar 从“长期分支/worktree 同步模式”调整为“main 控制台 + 新增-only执行面 + `runs/sidecar` 隔离结果”。

本文兼容旧 `PSE0` 路径引用，但新规则以本文件和同目录 `00~04` 精简治理文档为准。

## 2. 核心决策

1. 旧 `.wt/tcn`、`.wt/teach`、`.wt/bank`、`.wt/ctrl` 不再作为必须持续同步的工作分支。
2. 旧分支中的 S0 设计思路收编到 `docs/sidecar/lane_plans/`，作为 main 控制台下的候选路线。
3. sidecar 可以基于 main 当前代码新增实验 helper、独立模块、task-scoped config 或文档，但不得破坏原有主线逻辑。
4. 默认不修改既有主线行为；如必须碰已有 registry 或入口，只允许 default-off 的 additive change，并必须有回归验证证明旧路径不变。
5. sidecar 结果统一写入 `runs/sidecar/<lane_id>/<run_id>/`，不得写入或改写主线历史 run root。
6. sidecar 输出不自动进入论文、主线 task board 或当前事实口径；晋升必须走 `04_sidecar_promotion_gate.md`。

## 3. 主线 Anchor

sidecar 必须保留以下事实边界：

- `T24` 仍是历史 frozen-set software-HIL ranked table anchor。
- `T64-T70` 是 `statcalib` bounded extension-lane evidence，不是 mature comparator promotion。
- `T48` 是 current-host isolated true `.tflite` runtime truth，不是 HIL/board closure。
- `T49/T71/T72` 是 real-board gate/provenance truth，不是真板执行成功。
- 当前唯一主线任务仍以 `docs/04_task_board.md` 与 `docs/07_handoff.md` 为准。

## 4. Sidecar Lane 定义

一条 sidecar lane 至少需要：

1. lane ID，例如 `sidecar_slowloop_temporal_tcn`。
2. 一份位于 `docs/sidecar/lane_plans/` 的设计文档或 task package。
3. 明确的新增文件范围。
4. 明确的 forbidden scope。
5. 明确的 run dir：`runs/sidecar/<lane_id>/<run_id>/`。
6. 明确的 evidence level：`S0_design`、`S1_toy_or_replay`、`S2_bounded_sidecar_benchmark` 或 `S3_promotion_candidate`。

## 5. 新增-only 代码规则

后续 sidecar 如果需要代码，优先新增：

- `cnn_fpga/sidecar/<lane_id>/...`
- `cnn_fpga/benchmark/sidecar_<lane_id>_*.py`
- `cnn_fpga/config/sidecar_<lane_id>_*.yaml`
- `tests/test_sidecar_<lane_id>_*.py`

默认禁止：

- 修改 `ParamMapper` 主线语义。
- 修改 `SlowLoopRuntime` 默认行为。
- 修改 P4 runner 的历史 baseline/scenario 口径。
- 修改 `board_backend.py` placeholder 语义。
- 修改 canonical training / `.tflite` / real-board artifact。

若一个 lane 必须扩展已有入口，必须满足：

1. 变更是 additive、default-off、显式 opt-in。
2. 旧主线路径有 focused regression 证明不变。
3. 任务包明确标为 `contract_or_registry_touch`。
4. 结果仍不得直接写入主线事实口径。

## 6. Run Directory 规则

sidecar 运行结果只能写入：

```text
runs/sidecar/<lane_id>/<run_id>/
```

每个 run root 至少包含：

- `sidecar_manifest.json`
- `command.txt`
- `workspace_status.txt`
- `stdout.log`
- `stderr.log`
- `sidecar_summary.json` 或 `sidecar_summary.csv`

不得：

1. 写入 `runs/p4_benchmark/T24_*`、`T64_*`、`T66_*`、`T67_*`、`T68_*`、`T69_*` 等历史目录。
2. resume 或 regenerate 任何历史主线 run root。
3. 把整个 `runs/` 或 `artifacts/` 目录当作事实来源。
4. 让 sidecar run root 成为 `docs/04_task_board.md` 的当前主线事实。

## 7. Worktree / Clone 规则

PSE1 后，worktree 不再是 sidecar 的强制组织方式。

允许三种执行面：

| 执行面 | 适用场景 | 要求 |
| --- | --- | --- |
| main checkout +新增-only文件 | docs、toy、replay、轻量 helper | 不得破坏主线逻辑；结果写 `runs/sidecar` |
| 短生命周期 worktree | 需要并行会话或隔离测试 | 从当前 main 稳定点创建，结束后只回收 summary / manifest |
| clean short-path clone | 多日长跑或 Windows 路径风险 | 记录 launch provenance，不把 clone 当主线事实 |

旧 `.wt/*` 可保留为 read-only historical workspace，也可以后续清理；不再要求 rebase 到 main。

## 8. 红线

sidecar 不得声称：

- `real-board validated`
- `tflite deployed`
- `mature calibration comparator`
- `paper-grade expanded benchmark completed`
- `T24 rewritten`
- `statcalib promoted`
- `unique threshold proven`

sidecar 不得绕过当前主线唯一任务机制，也不得把并行实验结果直接写入论文 claim。

## 9. 当前建议

可以在 main 上继续维护 sidecar registry、lane plan 和新增-only实验代码。旧 Wave A 分支不建议继续逐个整理；如需执行某条路线，从当前 main 新开 `S1` task package，再按本目录规则推进。
