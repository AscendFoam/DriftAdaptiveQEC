# T8 Gate Review

Date:
`2026-05-08`

Verdict:
`Continue Repair`

Decision:
当前项目已经达到“最小 P3/P4 recovery path 可复验”的状态，但还不适合把仓库决策状态从 `Repair` 提升为 `Go`。

## Evidence Reviewed

1. `docs/03_hil_p4_boundary_audit.md`
2. `docs/P3_software_hil_bootstrap.md`
3. `docs/P4_benchmark_recovery_bootstrap.md`
4. `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`
5. `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/summary.json`
6. `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/comparison.csv`
7. `docs/06_repo_noise_governance.md`
8. `docs/08_risks_and_open_questions.md`

## What Is Now Credible

1. 最小 software HIL 路径已经二次复验，且 `backend`、`artifact_path`、`inference_service_mode` 已固定为 `mock + artifact_npz + inproc`。
2. 最小 P4 benchmark 路径已经复验，且明确复用了同一条 HIL 主链。
3. `mock` / `stub` / `placeholder` 的边界已经被写清，后续不再依赖旧长 session 才能解释当前仓库状态。
4. task board、handoff、risk 和 bootstrap 文档已经能支持新的接力开发。

## Why This Is Not Go Yet

1. `T7` 只覆盖了 `single-scenario + two-mode + repeats=1` 的 recovery smoke，还没有恢复到 frozen baseline 的更强 P4 证据。
2. 根目录仍然缺少可移植的最小依赖清单，当前运行环境仍高度依赖本机解释器路径。
3. 最小 software HIL 路径仍更接近“可复验”而非“逐字确定性复现”，随机源控制问题尚未收口。

## Practical Conclusion

当前最合理的判断是：

1. 项目已经从 `Phase 0: Stabilization` 进入 `Phase 1: Recovery`
2. 继续维持 `Repair`
3. 下一唯一任务应继续增强 P4 recovery 证据，而不是立刻恢复正式长跑或新功能扩展

## Next Unique Task

`T9: 重新验收一个 P4 frozen baseline 单场景全模式 smoke path`

Reason:

1. 它直接收敛 `T7` 目前“只有两种 mode”的缺口
2. 它仍然是有界任务，不会把恢复期重新扩成正式长跑
3. 它最直接服务于下一次 `Go / Repair` gate review
