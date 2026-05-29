# T65 Review

- Verdict: `PASS_WITH_WARNINGS`

基于你刚才的澄清，我不再把当前 worktree 里混入的 `docs/follow-up_plan/**`、`docs/汇报用/**` 等额外改动视为 T65 的阻塞项。本 review 现在只对 T65 task-local 内容做结论：在这个前提下，T65 的主体目标已经完成，且验证成立。

我核对了任务包、T65 本地代码/文档、`T64/T24` 既有 artifact，并实跑了这次任务要求的轻量验证：

- `python -m unittest tests.test_fr8_extension_lane_consistency`
- `python -m py_compile cnn_fpga/benchmark/audit_fr8_extension_lane_consistency.py`
- `python -m cnn_fpga.benchmark.audit_fr8_extension_lane_consistency ...`

三项都通过；审计 helper 也给出 `8/8` checks passed。

## Blocking issues

- None.

在“额外 diff 已由你单独审核，不计入本次 Worker 边界违规”的前提下，我没有看到 T65 task-local 内容触发任务包里的 review no-go：

- 没有新 run root
- 没有修改 `runs/` 下历史 artifact
- 没有改 benchmark / runtime / comparator 语义
- 没有把 T64 升格成 `.tflite`、real-board 或 paper-grade expanded benchmark

## Non-blocking issues

1. 当前 verdict 依赖了你的补充说明，而不是仅凭 git diff 的纯净性就能独立推出。
   - 所以我给 `PASS_WITH_WARNINGS`，而不是纯 `PASS`。
   - 这条 warning 不是技术实现问题，而是审查边界需要依赖额外上下文说明。

2. 新增的 `audit_fr8_extension_lane_consistency.py` 是有意做成 T64-specific guard，不是通用 FR8 framework。
   - 这符合 T65 的 bounded 目标。
   - 但后续如果出现新的 extension-lane run，不应把这次 helper 的一次通过自动外推成“所有后续 FR8 lane 都已自动受保护”。

3. `docs/review/T65_review.md` 和 `docs/for_human/T65_explanation.md` 的原始草稿方向基本是对的，但原 review 过早给了纯 `PASS`，没有把“结论依赖额外 scope 澄清”写出来。

## Missing tests

1. 对 T65 本身来说，没有阻塞级别的缺测。当前测试已经覆盖了这次 hardening 最关心的回归面：
   - duplicate `running` guard
   - provenance wording drift
   - execution-shape wording drift
   - 当前保留 `T64/T24` artifact set 的 full audit pass

2. 仍可补的非阻塞测试：
   - 一个 synthetic failure case，专门断言 `frozen_subset_matches_t24` 失败时 helper 会报错
   - 一个 synthetic failure case，专门断言 boundary phrases 缺失时 helper 会报错

## Suspicious implementation details

1. `audit_fr8_extension_lane_consistency.py` 对期望 scenarios、frozen modes、以及报告中的部分 required phrases 做了显式硬编码。
   - 这对 T65 是合理的，因为它本来就是 `T64 closeout guard`。
   - 但也说明它是刻意做窄的审计器，不是可无缝泛化的未来框架。

2. `test_current_t64_artifacts_pass_full_audit()` 依赖仓库里保留的 `T64/T24` run 目录。
   - 这很好地保护了当前 frozen artifact set。
   - 但它更接近 repository-state integration guard，而不是完全可移植的纯单元测试。

## Recommended next action

1. Captain 可以把 T65 按 `PASS_WITH_WARNINGS` 接受为：
   - `T64` 报告措辞已收口
   - `R28` 类型的 report/artifact consistency gap 已被显著收紧
   - `T64` 结果包现在可以作为 self-audited bounded extension-lane artifact 被更安全地复用

2. 后续继续保持三条边界：
   - `T24` 仍然是 authoritative frozen ranked table
   - `statcalib` 仍然只是 separately labeled extension lane
   - 证据等级仍然只是 mock-backed software-HIL，不是 `.tflite`，不是 real-board，也不是成熟 calibration comparator 定论

3. 如果以后还会频繁出现“当前任务 diff 混有其他已审核工作”的情况，最好在提交前明确分组或说明，减少 reviewer 再次把 mixed diff 误判成任务越界。
