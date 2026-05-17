# CLAUDE

## 1. Roles

Claude Code can act as either a **Worker** or a **Reviewer** depending on the task assignment.

### 1.1 Worker Role

When assigned as a Worker, Claude Code is responsible for one task package only.

Worker must:

1. Read:
   - `README.md`
   - `docs/02_experiment_plan.md`
   - `docs/04_task_board.md`
   - `docs/07_handoff.md`
   - the assigned task package
2. Modify only files listed in `Allowed files`.
3. Avoid everything listed in `Forbidden scope`.
4. Keep changes narrowly scoped.
5. Run the task's `Verification` commands, or clearly explain why a command could not run.
6. Update only the docs named in `Docs to update`.
7. Stop after the assigned task and report:
   - what changed
   - verification results
   - remaining risks

Worker must not:

1. Claim the next task.
2. Mark the task as complete in `docs/04_task_board.md`.
3. Modify unrelated files.
4. Revert user changes.
5. Add mock, stub, or hardcoded behavior unless the task explicitly asks for a prototype placeholder and documents it as such.

### 1.2 Reviewer Role

When assigned as a Reviewer, Claude Code acts as a read-only reviewer unless explicitly instructed otherwise.

The reviewer should inspect the diff for one assigned task package and decide whether the work satisfies the task.

Valid verdicts:

1. `PASS`
2. `PASS_WITH_WARNINGS`
3. `BLOCK`

Default posture: prioritize bugs, behavioral regressions, fake completion, missing tests, unsafe execution, and documentation that overstates implemented capability.

---

## 2. Files To Read Before Review

For every review, read:

1. `docs/02_experiment_plan.md`
2. `docs/04_task_board.md`
3. `docs/07_handoff.md`
4. `docs/08_risks_and_open_questions.md`
5. The assigned task package under `docs/tasks/`

For implementation tasks, also inspect the relevant changed files and the git diff.

---

## 3. Review Checklist

Check:

1. Did the worker complete the task goal?
2. Did the worker stay within `Allowed files`?
3. Did the worker avoid `Forbidden scope`?
4. Are there mocks, stubs, hardcoded outputs, or fake success paths?
5. Are tests or verification adequate for the risk level?
6. Did the change break existing chat/runtime/benchmark/release behavior?
7. Are model/data contracts clear and JSON-serializable where needed?
8. Are safety limits present for execution-related work?
9. Are docs updated without claiming planned work is already complete?
10. Are remaining risks documented in `docs/08_risks_and_open_questions.md` when relevant?

For high-risk tasks marked `adversarial`, be stricter about:

1. execution safety
2. data leakage
3. benchmark validity
4. release packaging regressions
5. over-engineering or hidden coupling

---

## 4. Required Output

Write the review to:

`docs/review/<TaskID>_review.md`

Use this structure:

```markdown
# Review: <TaskID>

Verdict: PASS / PASS_WITH_WARNINGS / BLOCK

## Blocking Issues

## Non-Blocking Issues

## Missing Tests

## Suspicious Implementation Details

## Recommended Next Action
```

Also write a human-facing explanation to:

`docs/for_human/<TaskID>_review_explanation.md`

It should explain:

1. What the task is trying to accomplish in plain language.
2. What the implementation changed.
3. Why the review verdict was given.
4. What the next step should be.

---

## 5. Verdict Guidance

Use `PASS` when:

1. The task goal is met.
2. Verification is adequate.
3. No blocking issue remains.
4. Documentation matches reality.

Use `PASS_WITH_WARNINGS` when:

1. The task is usable.
2. Remaining issues are real but can be deferred.
3. Risks are documented.

Use `BLOCK` when:

1. The task goal is not met.
2. The worker modified forbidden files or unrelated behavior.
3. A safety issue exists.
4. Tests/verification are missing for a risky change.
5. The implementation is fake, hardcoded, or mostly stubbed.
6. Documentation claims future work as completed.


## 审查重点

1. 当前任务是否真的完成
2. 是否越过 `Allowed files` 或 `Forbidden scope`
3. 是否把计划、placeholder、mock、stub 写成已完成事实
4. 是否修改了 benchmark 口径却没有明确记录
5. 是否存在 hardcode、假结果、跳过验证或只在作者机器成立的环境假设
6. 是否破坏了当前 `Repair` 阶段“先恢复可信度”的目标

## 本项目特别关注的问题

- `mock FPGA` 与 `real board` 的边界是否写清
- `.tflite` 真导出、artifact stub、subprocess 推理是否被混淆
- teacher-representation 分支结果是否被过度外推
- benchmark 中 seed、config、artifact 路径、git commit 是否可追溯
- 文档是否引用了真实存在的代码入口
