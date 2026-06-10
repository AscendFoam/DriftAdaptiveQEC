# 并行 Sidecar 扩展实验治理

## 1. 状态

- 日期：`2026-06-08`
- 范围：docs-only 治理设置
- 当前主线任务仍为：`T69: FR8 statcalib clean-winner tie-break bounded benchmark`
- 仅凭本文档，不授权任何实验、分支、worktree、benchmark、训练、`.tflite` smoke 或 real-board smoke。
- 后续文档默认优先使用中文；保留 `sidecar`、`promotion gate`、`.tflite`、`real-board` 等技术标识以便检索和对齐既有治理文本。

本文定义在主线 benchmark lane 继续按 `T69` 推进时，如何准备并行 sidecar 扩展路线。sidecar lane 是隔离的研究或工程实验路线，可以产生候选证据，但在后续 Captain gate 之前不能成为主线事实来源。

## 2. Frozen Anchor Manifest

### 2.1 主线权威 anchor

当前权威的 frozen ranked benchmark anchor 仍是：

- 任务：`T24: P4 bounded formal software revalidation execution`
- run root：`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743`
- 证据等级：mock-backed software-HIL formal software revalidation
- 冻结场景：
  - `static_bias_theta`
  - `linear_ramp`
  - `step_sigma_theta`
  - `periodic_drift`
- 历史冻结模式：
  - `static_linear`
  - `window_variance`
  - `ekf`
  - `ukf`
  - `hybrid_residual_b`
- 主线结论边界：
  - `T24` 可以被引用为已完成 frozen-set formal software revalidation。
  - `T24` 不是 `.tflite` runtime validation。
  - `T24` 不是 real-board validation。
  - `T24` 不是 paper-grade expanded benchmark evidence。

### 2.2 Statcalib extension-lane anchors

statcalib 证据序列必须继续单独标注：

| Task | 作用 | 边界 |
| --- | --- | --- |
| `T64` | 首个 clean-provenance bounded statcalib extension-lane benchmark | 只是一条单独 extension lane |
| `T65` | T64 report/artifact consistency guard | 不产生新的 benchmark 证据 |
| `T66` | local statcalib sensitivity grid | 仅回答有界鲁棒性 |
| `T67` | teacher-anchor dependence check | 不关闭 `R24` |
| `T68` | generated-only robustness existence check | 存在 full generated-only winners，但最强答案仍是 tie set |
| `T69` | 当前计划中的 clean-winner tie-break benchmark | 本治理任务不执行 |

这些任务都不能改写 `T24`。它们也不能把 statcalib 升级成成熟 calibration comparator。后续 gate 可以提出一个有界主线候选，但仍必须把 comparator claim 限定在实际产生的证据内。

### 2.3 当前 `R24` 边界

`R24` 仍是 statcalib 过度声称的主动风险边界。`T68` 之后，问题已经收窄：full generated-only winners 确实存在，但最强 clean answer 仍是 `window_variance_t001 = t003 = t005` tie set，且部分预声明候选仍为 `mixed`。`T69` 的主线目标就是测试这个 tie set 是否收窄。

sidecar 实验不得声称 `R24` 已关闭。

## 3. Sidecar Lane 定义

一条 sidecar lane 只有同时满足以下条件才有效：

1. 使用 `codex/sidecar-` 前缀的专用分支。
2. 使用专用 worktree 或 clone path。
3. 在文档、run root 和 artifact manifest 中一致使用同一个 lane ID。
4. 不修改历史 run root。
5. 不写入 `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_*`。
6. 不改变 `T24`、`T64`、`T65`、`T66`、`T67`、`T68` 或 `T69` 的事实口径。
7. 记录 backend 与 inference artifact type。
8. 在后续 Captain promotion gate 前，所有结果都标注为 sidecar outputs。

推荐 lane ID 格式：

```text
sidecar_<axis>_<short_name>
```

示例：

- `sidecar_slowloop_temporal_tcn`
- `sidecar_teacher_adaptive_confidence`
- `sidecar_fastloop_gain_bank`

## 4. Sidecar Artifact Schema

每条 sidecar lane 必须写入名为：

```text
sidecar_manifest.json
```

manifest 必须包含以下字段：

```json
{
  "schema_version": "sidecar_manifest_v1",
  "lane_id": "sidecar_slowloop_temporal_tcn",
  "lane_title": "Temporal histogram stack plus tiny TCN residual b head",
  "branch": "codex/sidecar-temporal-tcn-residual",
  "worktree_path": ".wt/tcn",
  "base_commit": "",
  "created_from": "main",
  "current_mainline_task": "T69",
  "frozen_anchor_task": "T24",
  "evidence_boundary": "sidecar_mock_backed_software_hil_or_toy_only",
  "backend": "mock_or_not_run",
  "inference_artifact_type": "artifact_npz_or_not_applicable",
  "fast_loop_contract_change": false,
  "historical_run_roots_read": [],
  "new_run_roots": [],
  "source_files_modified": [],
  "docs_modified": [],
  "metrics_files": [],
  "summary_files": [],
  "promotion_status": "not_requested",
  "red_line_acknowledgement": {
    "does_not_rewrite_t24": true,
    "does_not_rewrite_t69": true,
    "does_not_claim_tflite_runtime": true,
    "does_not_claim_real_board": true,
    "does_not_claim_mature_calibration_comparator": true
  }
}
```

如 lane 产生表格输出，除非任务包另有说明，否则使用：

- `sidecar_summary.csv`
- `sidecar_candidates.csv`
- `sidecar_safety_checks.csv`
- `sidecar_provenance.json`

## 5. Promotion Gate

sidecar 输出分为四级，任何 sidecar 分支不得跳级：

| Level | 含义 | 允许声称 |
| --- | --- | --- |
| `S0_design` | docs-only 设计，不执行实验 | 路线已定义且边界清楚 |
| `S1_toy_or_replay` | toy simulation、cached replay 或 contract test | 路线有初步 sidecar signal |
| `S2_bounded_sidecar_benchmark` | 专用 sidecar run，并与 frozen anchor 对比 | 路线有有界 sidecar evidence |
| `S3_integration_candidate` | Captain review 后成为未来主线任务候选 | 可以被提议为主线任务包 |

从 `S2` 晋升到 `S3` 至少需要：

1. 与 `T24` 或 Captain 批准的 derivative anchor 做 frozen-anchor A/B comparison。
2. deterministic replay 或记录清楚的 repeat-stability check。
3. clean provenance：记录 branch、worktree path、command、interpreter 和 commit。
4. 未修改任何历史 run root。
5. 除非 lane 被明确分类为 `contract_change`，否则不改变 benchmark runner 或 runtime 语义。
6. 记录 backend 与 inference artifact 边界。
7. 根据 lane 类型，通过 residual bounds、parameter range、bank switch behavior 或 rollback behavior 等安全检查。
8. Reviewer 确认 lane 仍保留 drift-adaptive fast/slow-loop 叙事。

晋升到主线必须新开 Captain 任务包，不会自动发生。

## 6. Run Directory 规则

主线任务 run root 继续使用任务专属位置，例如：

```text
runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_*
```

sidecar lane 必须使用：

```text
runs/sidecar/<lane_id>/<timestamp_or_run_id>/
```

每个 sidecar run root 必须包含：

- `sidecar_manifest.json`
- `command.txt`
- 如后台或 detached 启动，包含 `host_launch_meta.json`
- `workspace_status.txt`
- `stdout.log`
- `stderr.log`
- 一个 summary 文件，使用 `sidecar_summary.json` 或 `sidecar_summary.csv`

sidecar run root 不得：

1. 写入 `runs/p4_benchmark/T69_statcalib_clean_winner_tiebreak_*`
2. 改写 `runs/p4_benchmark/T24_*`、`T64_*`、`T66_*`、`T67_*`、`T68_*` 或 `T69_*` 下的历史 run root
3. 在不列出精确路径的情况下依赖历史 `runs/` 目录内容
4. 在 `docs/04_task_board.md` 中被引用为当前主线事实

## 7. Worktree 隔离规则

当前推荐本地 worktree root：

```text
.wt/<lane-short-name>
```

`.wt/` 必须被 `.gitignore` 忽略，避免把 linked worktree 内容误纳入主 checkout 状态。`.worktrees/` 可保留给轻量或非 Windows 长路径敏感场景，但本仓库历史 `runs/` 路径较深，Wave A 优先使用 `.wt/<short>`。对于多日长跑、Windows 路径长度敏感或需要更强进程隔离的任务，仍可另行使用短路径 clone/worktree，例如 `C:/daqec_<lane_short_name>/`；这属于执行动作，需要单独记录并验证。

每个 worktree 必须具备：

1. 一个分支
2. 一个 lane ID
3. 一个 owner thread 或 worker
4. 代码修改前先有一个任务包
5. 不隐式依赖另一个 worktree 中的 uncommitted changes

如果两条 lane 需要共享 helper，先允许有限重复；只有后续 integration task 证明抽象值得中心化时，才可提炼共享实现。不得在 sidecar 探索中顺手重构主线 benchmark/runtime 代码。

## 8. 红线

sidecar 工作不得：

1. 改写历史 `T24` frozen ranked table
2. 修改或重标注 `T64`、`T65`、`T66`、`T67`、`T68` 或 `T69` artifact
3. 声称 `real-board validated`
4. 声称 `tflite deployed`
5. 声称 `mature calibration comparator`
6. 声称 `paper-grade expanded benchmark`
7. 静默改变 `ParamMapper` 主线语义
8. 静默改变主线任务使用的 `SlowLoopRuntime` 语义
9. 在没有 `contract_change` 任务包的情况下改变 fast-loop input/output contract
10. 把 `.tflite` runtime recovery 或 real-board validation 当作普通算法 sidecar lane
11. 把 research-only code-family 工作并入近期 benchmark 队列

## 9. 允许优先准备的 Sidecar 类型

以下类型可以立即规划：

| 类型 | 第一安全动作 |
| --- | --- |
| temporal histogram plus tiny TCN residual head | `S0_design`，之后再做小规模 `S1_toy_or_replay` |
| histogram moments / EWMA feature ablation | `S0_design`，之后再做 cached-feature probe |
| adaptive syndrome-only teacher | `S0_design`，之后再做 teacher replay |
| confidence-gated fallback teacher | `S0_design`，之后再做 policy replay |
| teacher-conditioned residual head | `S0_design`，之后再写小 retrain plan |
| gain-scheduled / piecewise-affine bank | `S0_design`，之后再做 mock bank-selection replay |
| atomic commit / rollback control | `S0_design`，之后再做 contract tests |

以下类型可以写 research-only memo，但不进入执行 lane：

- S4 / Mamba slow-loop estimator
- recurrent transformer positioning
- surface-GKP / outer-code soft-information positioning
- QLDPC-GKP future architecture
- GNN detector-history literature note

## 10. Captain Review Checklist

sidecar lane 开始执行前，Captain 必须确认：

1. lane 有任务包，且包含 Allowed files、Forbidden scope、Verification、Docs to update。
2. branch name 以 `codex/sidecar-` 开头。
3. worktree path 与 active mainline checkout 隔离。
4. run root policy 是 `runs/sidecar/<lane_id>/...`。
5. lane 声明自己是 `S0_design`、`S1_toy_or_replay`、`S2_bounded_sidecar_benchmark` 还是 `research_only`。
6. lane 声明是否改变 fast-loop contract。
7. lane 声明是否读取历史 run root。
8. lane 拒绝 `.tflite`、real-board 和 mature-comparator claim，除非这些 claim 被单独 future gate 管住。

## 11. 当前决策

可以现在规划并创建并行 sidecar worktree，但每条 lane 必须先有 `S0_design` 任务包。任何实验执行、长跑 benchmark、`.tflite` smoke 或 real-board smoke 仍需要后续单独授权。

`T69` 仍是当前唯一主线任务，main 分支的 `T69` 执行必须与所有 sidecar worktree 保持独立。
