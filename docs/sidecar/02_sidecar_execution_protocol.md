# 02 Sidecar 执行协议

## 1. 执行级别

| Level | 含义 | 允许输出 |
| --- | --- | --- |
| `S0_design` | 只写设计和边界 | `docs/sidecar/lane_plans/*.md` |
| `S1_toy_or_replay` | toy simulation、cached replay、contract tests | `runs/sidecar/<lane_id>/<run_id>/` |
| `S2_bounded_sidecar_benchmark` | 有界 sidecar benchmark | `runs/sidecar/<lane_id>/<run_id>/` + summary |
| `S3_promotion_candidate` | 候选晋升 | promotion review 文档 |

## 2. 新增-only 代码策略

默认允许新增：

- 新的 sidecar helper。
- 新的 sidecar config。
- 新的 sidecar tests。
- 新的 sidecar docs。

默认禁止修改：

- 主线 benchmark 口径。
- 主线 baseline/scenario 集合。
- `ParamMapper` 默认语义。
- `SlowLoopRuntime` 默认语义。
- `board_backend.py` placeholder 事实。
- canonical artifacts 或历史 run root。

如果必须修改已有文件，只能是 additive、default-off、explicit opt-in，并且要有 regression 证明旧路径不变。

## 3. Run Root

所有 sidecar 输出写入：

```text
runs/sidecar/<lane_id>/<run_id>/
```

推荐 `run_id`：

```text
YYYYMMDD_HHMMSS_<short_commit>
```

每个 run root 必须包含：

- `sidecar_manifest.json`
- `command.txt`
- `workspace_status.txt`
- `stdout.log`
- `stderr.log`
- summary 文件

## 4. Provenance

必须记录：

1. 当前 commit。
2. 是否 dirty。
3. 命令。
4. 解释器。
5. 输入 config。
6. 读取的历史 run root 精确路径。
7. 新生成的输出路径。
8. evidence level。

## 5. 并行执行

允许多个 sidecar 并行规划；重计算执行需要单独任务包。多日长跑建议使用短生命周期 worktree 或 clean short-path clone，而不是复活旧 `.wt/*` 长期分支。

## 6. 验证

最小验证包括：

1. 文档检查：路径、lane id、run root 一致。
2. schema 检查：`sidecar_manifest.json` 字段完整。
3. 边界检查：不出现禁止 claim。
4. 如有代码：focused tests + 旧路径回归。
5. 如有运行：summary 与 manifest 一致。

