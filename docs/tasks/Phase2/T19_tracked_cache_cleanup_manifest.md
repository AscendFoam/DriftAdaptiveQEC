# T19: Tracked Cache Cleanup Manifest

Task ID: `T19`

Goal: 为已跟踪的 `__pycache__/` 与 `.pyc` 文件制定有界 cleanup manifest 与验收标准。

Why now: `docs/06_repo_noise_governance.md` 已固定“先治理后清理”，但物理 cleanup 仍未执行。该任务只处理缓存/字节码，不碰 `runs/` 和 `artifacts/`。

Allowed files:

- `docs/tasks/Phase2/T19_tracked_cache_cleanup_manifest.md`
- `docs/cleanup_tracked_cache_manifest.md`
- `docs/06_repo_noise_governance.md`
- `docs/04_task_board.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`

Forbidden scope:

- 不删除 `runs/`
- 不删除 `artifacts/`
- 不执行批量破坏性命令，除非 Captain 后续明确批准 cleanup 执行任务
- 不改源码

Inputs to read:

- `docs/06_repo_noise_governance.md`
- `.gitignore`
- `git ls-files`

Expected output:

- `docs/cleanup_tracked_cache_manifest.md`
- 明确列出：
  - 目标文件类别
  - cleanup 命令草案
  - 回滚方式
  - 验收标准
  - 不触碰范围

Verification:

- 只读清点，不执行删除。

Docs to update:

- `docs/06_repo_noise_governance.md`
- `docs/07_handoff.md`
- `docs/08_risks_and_open_questions.md`
- `docs/04_task_board.md`（Captain 整合阶段）

Reviewer type: `normal`

## Worker Output Summary

- Output type: `docs/cleanup_tracked_cache_manifest.md`
- Read-only inventory confirmed:
  - tracked `.pyc` files: `116`
  - tracked `__pycache__` directories: `9`
  - tracked standalone `.pyc` outside `__pycache__`: `0`
- Cleanup execution was not performed
- `runs/` and `artifacts/` were not touched
- Updated docs:
  - `docs/cleanup_tracked_cache_manifest.md`
  - `docs/06_repo_noise_governance.md`
  - `docs/04_task_board.md`
  - `docs/07_handoff.md`
  - `docs/08_risks_and_open_questions.md`
