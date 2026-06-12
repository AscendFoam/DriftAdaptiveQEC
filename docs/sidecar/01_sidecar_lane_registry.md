# 01 Sidecar Lane 注册表

## 1. 注册规则

每条 lane 至少登记：

- `lane_id`
- `status`
- `owner_surface`
- `first_safe_task`
- `allowed_output_root`
- `code_policy`
- `promotion_status`

状态枚举：

- `candidate`
- `s0_design`
- `s1_toy_or_replay`
- `s2_bounded_sidecar_benchmark`
- `s3_promotion_candidate`
- `retired`

## 2. 当前 Lane

| Lane ID | 状态 | 路线 | 第一安全任务 | 结果目录 |
| --- | --- | --- | --- | --- |
| `sidecar_slowloop_temporal_tcn` | `s0_design` | 最近若干 slow-loop 窗口 histogram / delta / summary 输入 tiny TCN，输出 residual-`b` 候选 | cached replay design | `runs/sidecar/sidecar_slowloop_temporal_tcn/` |
| `sidecar_teacher_adaptive_confidence` | `s0_design` | syndrome-only / teacher-output replay，定义 confidence、freeze、fallback 策略 | offline replay table | `runs/sidecar/sidecar_teacher_adaptive_confidence/` |
| `sidecar_fastloop_gain_bank` | `s0_design` | 有限 gain bank / piecewise-affine 参数选择，不改变 fast-loop contract | bank-selection toy replay | `runs/sidecar/sidecar_fastloop_gain_bank/` |
| `sidecar_control_commit_rollback` | `s0_design` | 参数提交、ack、rollback、freeze 的控制安全 contract | deterministic mock contract tests | `runs/sidecar/sidecar_control_commit_rollback/` |

## 3. 旧 Worktree 状态

旧 `.wt/*` 只保留为历史参考：

| 旧 worktree | 旧分支 | PSE1 处理 |
| --- | --- | --- |
| `.wt/tcn` | `codex/sidecar-temporal-tcn-residual` | retired/read-only |
| `.wt/teach` | `codex/sidecar-adaptive-teacher-replay` | retired/read-only |
| `.wt/bank` | `codex/sidecar-gain-scheduled-bank-sim` | retired/read-only |
| `.wt/ctrl` | `codex/sidecar-atomic-commit-rollback` | retired/read-only |

这些 worktree 不再要求 rebase 到 main，也不应作为新执行起点。后续执行从当前 main 的 lane registry 和 lane plan 开始。

## 4. 新 Lane 登记模板

```markdown
| `sidecar_<axis>_<short_name>` | `candidate` | <一句话路线> | `<S0/S1>` | `runs/sidecar/sidecar_<axis>_<short_name>/` |
```

新增 lane 前必须说明：

1. 是否新增代码。
2. 是否需要读取历史 run root。
3. 是否改变 fast-loop 或 slow-loop contract。
4. 是否可能触碰 `.tflite` 或 real-board 边界。

