# 并行 Sidecar Worktree 计划

## 1. 状态

- 日期：`2026-06-08`
- 范围：worktree 与任务包规划
- 治理来源：`docs/parallel_sidecar_extension_governance.md`
- 当前主线任务仍为：`T69`
- 本计划允许创建 Wave A worktree，并要求每条 Wave A lane 先写 `S0_design` 任务包；不授权运行实验、benchmark、训练、`.tflite` smoke 或 real-board smoke。

本计划假设用户可以接受维护多个 worktree。因此计划把 worktree 数量与执行强度拆开：可以开较多 worktree，但重计算任务仍必须单独排期，确保 provenance、资源竞争和主线 `T69` 的独立性可审计。

## 2. Worktree Root 约定

当前本地推荐 root：

```text
.wt/
```

`.wt/` 必须被 `.gitignore` 忽略。`.worktrees/` 可保留给轻量或非 Windows 长路径敏感场景；Wave A 使用 `.wt/<short>`，以避开历史 `runs/` 深路径在 Windows checkout 时触发的 filename-too-long 问题。对于多日长跑或路径长度敏感任务，可在后续任务中使用短路径 root：

```text
C:/daqec_<lane_short_name>/
```

短路径 clone/worktree 属于执行层动作，需要单独记录 launch provenance。

## 3. 主线 Worktree

| 角色 | Branch | Worktree | 状态 | 规则 |
| --- | --- | --- | --- | --- |
| Mainline T69 | `main` 或 clean committed `main` clone | 当前 checkout 或 clean short-path clone | 独立保留 | 只执行 `T69`，不运行 sidecar 实验 |

## 4. 优先创建 Worktrees（Recommended Now）

这些 lane 是第一批候选，因为它们保留 fast/slow-loop 架构，不要求 real-board 或真实 `.tflite` runtime。

| Lane ID | Branch | Proposed worktree | First task level | First safe output | Verification shape |
| --- | --- | --- | --- | --- | --- |
| `sidecar_slowloop_temporal_tcn` | `codex/sidecar-temporal-tcn-residual` | `.wt/tcn` | `S0_design` then `S1_toy_or_replay` | temporal histogram stack design plus toy/cached replay manifest | sidecar manifest、无主线 runner 改动、residual metric table |
| `sidecar_features_moments_ewma` | `codex/sidecar-moments-ewma-feature-ablation` | `.wt/mom` | `S0_design` then `S1_toy_or_replay` | moments / EWMA / entropy feature ablation memo and cached-feature probe | sidecar summary plus feature completeness check |
| `sidecar_teacher_adaptive_confidence` | `codex/sidecar-adaptive-teacher-replay` | `.wt/teach` | `S0_design` then `S1_toy_or_replay` | sliding-window teacher replay and confidence fields | replay-only summary、无 benchmark 晋升 claim |
| `sidecar_teacher_confidence_fallback` | `codex/sidecar-confidence-fallback-policy` | `.wt/fallback` | `S0_design` then `S1_toy_or_replay` | fallback / freeze policy table using existing teacher outputs | policy replay summary and safety red-line checks |
| `sidecar_teacher_conditioned_head` | `codex/sidecar-film-teacher-conditioned-head` | `.wt/film` | `S0_design` then small retrain plan | teacher-conditioned residual-head design | train plan only until a separate execution task exists |
| `sidecar_fastloop_gain_bank` | `codex/sidecar-gain-scheduled-bank-sim` | `.wt/bank` | `S0_design` then `S1_toy_or_replay` | piecewise-affine bank-selection replay design | bank thrashing、range、rollback metrics |
| `sidecar_control_commit_rollback` | `codex/sidecar-atomic-commit-rollback` | `.wt/ctrl` | `S0_design` then contract tests | atomic commit / rollback contract tests | deterministic mock-HIL contract test output |
| `sidecar_fastloop_lut_affine` | `codex/sidecar-lut-assisted-affine` | `.wt/lut` | `S0_design` only at first | LUT-assisted bounded residual memo | fixed-point envelope checklist |

## 5. 试点级 Worktrees（Pilot-Only）

这些 lane 可以存在，但不应在 recommended-now lane 产生可解释证据之前抢占第一波执行资源。

| Lane ID | Branch | Proposed worktree | First task level | Constraint |
| --- | --- | --- | --- | --- |
| `sidecar_slowloop_gru_lstm` | `codex/sidecar-gru-lstm-low-priority` | `.wt/gru` | `S0_design` | 位于 TCN 之后，作为低优先级 ablation |
| `sidecar_slowloop_s4_mamba` | `codex/sidecar-s4-mamba-toy` | `.wt/s4` | `S0_design` then toy only | 不做 full slow-loop replacement |
| `sidecar_lowbit_micro_head` | `codex/sidecar-lowbit-fastloop-micro-head` | `.wt/lowbit` | `S0_design` | 标记为 `contract_change`，不进入即时 benchmark |

## 6. 研究备忘级 Worktrees（Research-Only）

这些 lane 对论文定位和未来架构有用，但不得晋升到近期 benchmark 队列。

| Lane ID | Branch | Proposed worktree | Output |
| --- | --- | --- | --- |
| `sidecar_positioning_recurrent_transformer` | `codex/sidecar-recurrent-transformer-positioning` | `.wt/rtrans` | literature positioning memo |
| `sidecar_positioning_surface_gkp` | `codex/sidecar-surface-gkp-positioning` | `.wt/surface` | task-interface memo，不做 full benchmark |
| `sidecar_future_qldpc_gkp` | `codex/sidecar-qldpc-gkp-future` | `.wt/qldpc` | future architecture sketch |
| `sidecar_positioning_gnn_detector` | `codex/sidecar-gnn-detector-literature` | `.wt/gnn` | compare-only literature note |

## 7. 创建波次

### Wave A：先创建并准备

1. `sidecar_slowloop_temporal_tcn`
2. `sidecar_teacher_adaptive_confidence`
3. `sidecar_fastloop_gain_bank`
4. `sidecar_control_commit_rollback`

理由：四条 lane 覆盖彼此区分度最高的轴线：slow-loop temporal modeling、adaptive teacher、fast-loop bank structure、control safety。

### Wave A 当前创建状态（2026-06-08）

| Lane ID | Branch | Worktree | S0 task package | Execution status |
| --- | --- | --- | --- | --- |
| `sidecar_slowloop_temporal_tcn` | `codex/sidecar-temporal-tcn-residual` | `.wt/tcn` | `.wt/tcn/docs/tasks/Phase2/sidecar_slowloop_temporal_tcn_s0_design.md` | `not_run` |
| `sidecar_teacher_adaptive_confidence` | `codex/sidecar-adaptive-teacher-replay` | `.wt/teach` | `.wt/teach/docs/tasks/Phase2/sidecar_teacher_adaptive_confidence_s0_design.md` | `not_run` |
| `sidecar_fastloop_gain_bank` | `codex/sidecar-gain-scheduled-bank-sim` | `.wt/bank` | `.wt/bank/docs/tasks/Phase2/sidecar_fastloop_gain_bank_s0_design.md` | `not_run` |
| `sidecar_control_commit_rollback` | `codex/sidecar-atomic-commit-rollback` | `.wt/ctrl` | `.wt/ctrl/docs/tasks/Phase2/sidecar_control_commit_rollback_s0_design.md` | `not_run` |

创建说明：第一次尝试使用 `.worktrees/<long-name>` 时触发 Windows `Filename too long`，未形成有效登记 worktree；最终使用短路径 `.wt/<short>` 加 `core.longpaths=true` 创建完成。`.wt/` 和 `.worktrees/` 均由 `.gitignore` 忽略。

### Wave B：等 Wave A 任务包存在后再创建

1. `sidecar_features_moments_ewma`
2. `sidecar_teacher_confidence_fallback`
3. `sidecar_teacher_conditioned_head`
4. `sidecar_fastloop_lut_affine`

理由：这些路线有价值，但更依赖第一波 feature 与 safety 决策。

### Wave C：research-only backlog

1. `sidecar_slowloop_gru_lstm`
2. `sidecar_slowloop_s4_mamba`
3. `sidecar_positioning_recurrent_transformer`
4. `sidecar_positioning_surface_gkp`
5. `sidecar_future_qldpc_gkp`
6. `sidecar_positioning_gnn_detector`

理由：扩大论文定位广度，但不强行变成高成本执行任务。

## 8. 单 Lane 任务包要求（Per-Lane Task Package）

任一 lane 改代码或跑实验前，必须先在该 lane 的 worktree 下创建任务包：

```text
docs/tasks/Phase2/<lane_id>_s0_design.md
```

每个包必须包含：

1. Status
2. Goal
3. Allowed files
4. Docs to update
5. Forbidden scope
6. Required inputs
7. Run directory policy
8. Verification
9. Promotion status
10. Worker output requirements

任务包必须明确 lane 分类：

- `recommended_now`
- `pilot_only`
- `research_only`
- `contract_change`

## 9. 执行限制

可以创建多个 worktree，但重执行必须受控：

1. `T69` 优先于 sidecar benchmark execution。
2. sidecar docs-only 和 toy/replay lane 可以在 `T69` 活跃时推进。
3. 不得同时运行多个多日 sidecar benchmark，除非 Captain 任务记录 compute plan。
4. sidecar 进程不得写入 active `T69` run root。
5. 如果 sidecar lane 需要长时间后台启动，应使用专用短路径 clone/worktree，并记录 `host_launch_meta.json`。

## 10. 立即行动

本计划之后的安全行动是：

1. 创建 Wave A worktrees 与 `codex/sidecar-*` 分支。
2. 在每个 Wave A worktree 先创建中文 `S0_design` 任务包。
3. 任务包未被后续审阅或授权前，不执行 lane。
4. main 分支的 `T69` 执行计划保持独立。

Wave A worktree 的创建不等于实验开始；它只建立隔离工作面和 `S0_design` 文档入口。
