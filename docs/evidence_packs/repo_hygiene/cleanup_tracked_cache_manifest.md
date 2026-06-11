# Tracked Cache Cleanup Manifest

## 1. Scope

本 manifest 只覆盖 Git 已跟踪的缓存/字节码文件：

- `**/__pycache__/**`
- `*.pyc`

本 manifest 不覆盖：

- `runs/`
- `artifacts/`
- 源码、配置、文档

本 manifest 是 `T19` 的只读产物，不等于已执行 cleanup。

## 2. Read-Only Inventory

清点基线：

- command: `git ls-files`
- date: `2026-05-10`
- task: `T19`

只读统计结果：

- tracked `.pyc` files: `116`
- tracked `__pycache__` files: `116`
- tracked standalone `.pyc` outside `__pycache__`: `0`
- tracked `__pycache__` directories: `9`

目录分布：

| Directory | Tracked files |
| --- | ---: |
| `cnn_fpga/__pycache__` | 1 |
| `cnn_fpga/benchmark/__pycache__` | 19 |
| `cnn_fpga/data/__pycache__` | 3 |
| `cnn_fpga/decoder/__pycache__` | 21 |
| `cnn_fpga/hwio/__pycache__` | 17 |
| `cnn_fpga/model/__pycache__` | 18 |
| `cnn_fpga/runtime/__pycache__` | 25 |
| `cnn_fpga/utils/__pycache__` | 6 |
| `physics/__pycache__` | 6 |

结论：

1. 当前已跟踪缓存文件全部落在 `__pycache__` 目录内。
2. 当前没有额外的已跟踪散落 `.pyc` 文件需要单独处理。
3. 因为 `.gitignore` 已包含 `__pycache__/`，cleanup 的核心是 `git rm --cached` 解除历史跟踪，而不是补忽略规则。

## 3. Target Paths

后续若 Captain 明确批准执行 cleanup，目标路径应严格限制为以下 9 个目录：

```text
cnn_fpga/__pycache__
cnn_fpga/benchmark/__pycache__
cnn_fpga/data/__pycache__
cnn_fpga/decoder/__pycache__
cnn_fpga/hwio/__pycache__
cnn_fpga/model/__pycache__
cnn_fpga/runtime/__pycache__
cnn_fpga/utils/__pycache__
physics/__pycache__
```

不应扩大到：

- `runs/`
- `artifacts/`
- `.pytest_cache/`
- `.mypy_cache/`
- 任何未出现在 `git ls-files` 中的路径

## 4. Cleanup Command Draft

以下命令只是后续执行任务的草案，本轮不执行。

### 4.1 Preflight

```powershell
git ls-files -- "cnn_fpga/__pycache__/*" "cnn_fpga/benchmark/__pycache__/*" "cnn_fpga/data/__pycache__/*" "cnn_fpga/decoder/__pycache__/*" "cnn_fpga/hwio/__pycache__/*" "cnn_fpga/model/__pycache__/*" "cnn_fpga/runtime/__pycache__/*" "cnn_fpga/utils/__pycache__/*" "physics/__pycache__/*"
```

预期：

- 仅返回缓存/字节码路径
- 总数应为 `116`

### 4.2 Bounded Untrack Draft

```powershell
git rm --cached -r -- cnn_fpga/__pycache__ cnn_fpga/benchmark/__pycache__ cnn_fpga/data/__pycache__ cnn_fpga/decoder/__pycache__ cnn_fpga/hwio/__pycache__ cnn_fpga/model/__pycache__ cnn_fpga/runtime/__pycache__ cnn_fpga/utils/__pycache__ physics/__pycache__
```

说明：

1. 该命令只解除 Git 跟踪，不删除工作区文件。
2. 该命令显式列出目标目录，避免误触 `runs/`、`artifacts/` 或源码目录。
3. 执行前仍需 Captain 明确批准，并建议在独立 cleanup 执行任务中落地。

## 5. Rollback Plan

若后续执行任务在提交前需要回滚，优先使用：

```powershell
git restore --staged -- cnn_fpga/__pycache__ cnn_fpga/benchmark/__pycache__ cnn_fpga/data/__pycache__ cnn_fpga/decoder/__pycache__ cnn_fpga/hwio/__pycache__ cnn_fpga/model/__pycache__ cnn_fpga/runtime/__pycache__ cnn_fpga/utils/__pycache__ physics/__pycache__
```

若已形成单独 cleanup commit，再通过单独 revert commit 回滚，不改写历史。

不建议：

- `git reset --hard`
- 混合回滚其他无关文件
- 顺手把 `runs/` 或 `artifacts/` 一并纳入回滚/清理

## 6. Acceptance Criteria

后续真正执行 cleanup 时，验收应满足：

1. `git ls-files` 不再返回上述 9 个 `__pycache__` 目录下的任何文件。
2. `git ls-files | rg "__pycache__|\\.pyc$"` 返回 `0` 条结果。
3. 工作区源码文件不发生内容修改。
4. `runs/` 与 `artifacts/` 不进入本次变更。
5. `.gitignore` 继续保留 `__pycache__/` 规则，避免缓存文件重新被纳入版本库。

## 7. Non-Touch Boundaries

`T19` 明确不触碰：

- `runs/` 的历史结果治理
- `artifacts/` 的 bootstrap 必需项拆分
- benchmark 口径
- `.tflite` runtime 边界
- `real_board` HIL 边界
- 任意源码修订

## 8. Recommended Next Step

如需真正执行物理 untrack，应由 Captain 单开后续 cleanup 执行任务，并至少附带：

1. 本 manifest 作为唯一路径清单。
2. 执行前 `git status` 截面。
3. 执行后 `git ls-files` 验证结果。
4. 明确声明本轮仍不处理 `runs/` / `artifacts/`。
