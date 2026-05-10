# T19 Review: Tracked Cache Cleanup Manifest

**Task ID**: `T19`
**Reviewer**: Claude Code Reviewer
**Date**: 2026-05-10

---

## Verdict: PASS

---

## 1. Task Completion Check

T19 目标：为已跟踪的 `__pycache__/` 与 `.pyc` 文件制定有界 cleanup manifest 与验收标准。

逐条验收：

| Done Criteria | 状态 | 说明 |
|---------------|------|------|
| 产出 tracked cache cleanup manifest | 完成 | `docs/cleanup_tracked_cache_manifest.md` 新增，8 节，覆盖 scope / inventory / targets / commands / rollback / acceptance / boundaries / next step |
| 明确目标文件类别、清点结果、命令草案、回滚方式与验收标准 | 完成 | 第 2 节清点、第 3 节目标路径、第 4 节命令草案、第 5 节回滚、第 6 节验收 |
| 不执行删除，不执行 `git rm` | 确认无违规 | Worker 报告"未执行 `git rm`"；独立验证 `git ls-files` 仍返回 116 个缓存文件 |
| 不触碰 `runs/` 和 `artifacts/` | 确认无违规 | diff 中无 `runs/` 或 `artifacts/` 相关变更 |
| 为后续 Captain 决定是否执行物理 cleanup 留出边界 | 完成 | 第 7/8 节明确列出不触碰范围和推荐后续步骤 |

## 2. Scope Compliance

### Allowed files — 全部合规

diff 涉及的文件：

- `docs/cleanup_tracked_cache_manifest.md`（新增） — allowed
- `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md` — allowed（仅追加 Worker Output Summary）
- `docs/06_repo_noise_governance.md` — allowed
- `docs/04_task_board.md` — allowed
- `docs/07_handoff.md` — allowed
- `docs/08_risks_and_open_questions.md` — allowed

无 `docs/` 以外的文件被修改。无代码文件变更。

### Forbidden scope — 无违规

- 未删除 `runs/` — 确认
- 未删除 `artifacts/` — 确认
- 未执行批量破坏性命令 — 确认
- 未改源码 — 确认

## 3. Inventory Accuracy (Independent Verification)

我用独立命令验证了 manifest 中的清点数字：

| Manifest Claim | Independent Verification | Match |
|----------------|-------------------------|-------|
| tracked `.pyc` files: `116` | `git ls-files \| grep -c "__pycache__"` → `116` | Yes |
| tracked standalone `.pyc` outside `__pycache__`: `0` | `git ls-files \| grep "\.pyc$" \| grep -v "__pycache__" \| wc -l` → `0` | Yes |
| tracked `__pycache__` directories: `9` | `git ls-files \| grep "__pycache__" \| sed 's\|/__pycache__/.*\|\|' \| sort -u` → 9 unique paths | Yes |

9 个目录路径与 manifest 第 2/3 节完全一致：
`cnn_fpga`、`cnn_fpga/benchmark`、`cnn_fpga/data`、`cnn_fpga/decoder`、`cnn_fpga/hwio`、`cnn_fpga/model`、`cnn_fpga/runtime`、`cnn_fpga/utils`、`physics`

## 4. Pseudo-implementation / Mock / Stub / Hardcode Check

**未发现伪实现。** 本任务是只读文档任务。

关键表述验证：
- 第 1 节："本 manifest 是 `T19` 的只读产物，不等于已执行 cleanup" — 边界诚实
- 第 4 节："以下命令只是后续执行任务的草案，本轮不执行" — 明确区分计划与已完成
- 第 5 节回滚方案完整，且标注"不建议 `git reset --hard`"
- 第 6 节验收标准面向未来的执行任务，未伪造成已通过

Worker 也没有在 task board 上勾选 T19 为 `[x]`，而是保持了 `[ ]` 状态。这符合"审核前不标记完成"的纪律。

## 5. Missing Tests / Validation

验证方式为只读清点（`git ls-files` / `rg`），这是任务包 `Verification` 中明确要求的。经独立复验，数字全部准确。

**无缺失验证**。

## 6. Over-engineering Check

**未发现过度工程。** 文档结构简洁直接：

- 8 节内容全部服务于"清点 + 方案 + 回滚 + 验收"的线性目标
- 没有引入不必要的抽象或工具
- 命令草案是标准的 `git rm --cached` 和 `git restore --staged`，没有自定义脚本
- 没有试图用自动化工具替代人工审核流程

## 7. Existing Functionality Preservation

**无破坏。** 本次任务无代码变更，无文件删除，所有改动均在 `docs/` 目录内。

## 8. Documentation Honesty Check

文档没有把计划写成事实：

- "本 manifest 是 `T19` 的只读产物，不等于已执行 cleanup" — 明确
- "以下命令只是后续执行任务的草案，本轮不执行" — 明确
- 第 7 节列出 6 项"明确不触碰"的范围
- Task board 中 T19 仍为 `[ ]`（未勾选），符合"审核前不标记完成"的要求

治理文件更新一致性：
- `06_repo_noise_governance.md` 第 3/7/8 节同步更新
- `08_risks_and_open_questions.md` R4/R7 更新为反映 T19 产出但未执行的状态
- `07_handoff.md` 追加 T19 Worker 只读结果，状态标注"等待 Reviewer / Captain 收口"

## 9. Blocking Issues

**无。**

## 10. Non-blocking Issues

### N1: manifest 第 4.1 节 preflight 命令中的 glob 模式

第 4.1 节的 preflight 命令使用了 `"cnn_fpga/__pycache__/*"` 这样的 glob 模式。在 Windows PowerShell 中，`git ls-files` 接收这些参数时行为取决于 shell 的 glob 展开。建议后续执行任务中直接使用不带 glob 的路径参数（如第 4.2 节已经做的那样），或改用 bash 执行。不影响当前只读任务。

### N2: 清点数字与当前工作区 `.pyc` 总数的差异说明

manifest 记录 tracked `.pyc` = `116`，而 `06_repo_noise_governance.md` 提到当前工作区 `.pyc` 文件总数 = `133`。差异 `133 - 116 = 17` 应来自于 `.gitignore` 忽略后新产生的未跟踪 `.pyc`。manifest 没有显式解释这个差异，但不影响结论（因为任务只处理已跟踪文件）。建议后续执行任务中如需复查，可加注一条说明。

## 11. Recommended Next Action

1. T19 审查通过，可由 Captain 标记完成并提交 git。
2. T19 标记完成后，下一任务建议按 task board 顺序进入 T20（Real-board HIL readiness checklist）。
3. 若要真正执行缓存 untrack，应单开一个有界执行任务，严格按 `docs/cleanup_tracked_cache_manifest.md` 的 9 个目录和回滚方案落地。不应借 T20 或其他任务顺手执行。

---

**Verdict: PASS**
**Blocking issues: 无**
**Non-blocking issues: N1（PowerShell glob 行为），N2（tracked vs workspace `.pyc` 差异未显式说明）**
