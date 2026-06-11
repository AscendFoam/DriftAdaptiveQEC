# T10 Gate Review

Date:
`2026-05-08`

Verdict:
`Continue Repair`

Decision:
`T9` 已经把 `P4 frozen baseline` 的 recovery 证据从“两模式最小路径”扩到“单场景四模式 smoke”，但当前项目仍不适合把仓库决策状态从 `Repair` 提升为 `Go`。

## Evidence Reviewed

1. `docs/review/T8_gate_review.md`
2. `docs/recovery_bootstrap/P3_software_hil_bootstrap.md`
3. `docs/recovery_bootstrap/P4_benchmark_recovery_bootstrap.md`
4. `runs/hil_suite/hardware_hil_recovery_smoke_20260507_234638_3ae9f9176104/hil_summary.json`
5. `runs/p4_benchmark/p4multis_20260508_001316_0c12d7_39308/summary.json`
6. `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/summary.json`
7. `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732/comparison.csv`
8. `docs/06_repo_noise_governance.md`
9. `docs/08_risks_and_open_questions.md`

## What Is Now Credible

1. 最小 software HIL 路径已经二次复验，且 `backend`、`artifact_path`、`inference_service_mode` 已固定为 `mock + artifact_npz + inproc`。
2. `P4 frozen baseline` 的四个正式 baseline `static_linear / window_variance / ekf / cnn_fpga` 已在同一个 `single-scenario + repeats=1` recovery smoke 中复验通过。
3. `mock` / `stub` / `placeholder` 的边界已经被写清，后续不再依赖旧 session 才能解释当前仓库状态。
4. task board、handoff、risk、bootstrap 文档已经可以支持新的接力开发。

## Why This Is Not Go Yet

1. 根目录仍然缺少可移植的最小依赖 manifest，当前运行方式仍高度依赖本机解释器路径与手工环境约定。
2. 最小 software HIL 路径仍更接近“可复验”而非“逐字确定性复现”，随机源控制问题尚未收口。
3. `T9` 虽然补齐了四种 frozen baseline，但当前证据仍只覆盖 `single-scenario + four-mode + repeats=1`，不等于正式多场景 frozen benchmark 已恢复。

## Practical Conclusion

当前最合理的判断是：

1. 项目继续保持在 `Phase 1: Recovery`
2. 决策状态继续维持 `Repair`
3. 下一唯一任务应先补 recovery 期最小依赖 manifest，而不是继续扩 benchmark 长跑或新功能

## Next Unique Task

`T11: 补一份恢复期最小依赖 manifest（优先覆盖 P0/P3/P4 recovery smoke）`

Reason:

1. 它直接收敛当前最明确、最可移植性的缺口。
2. 它是比“继续扩 benchmark 场景”更小、更有界的恢复期任务。
3. 它不会误把 `DLEnv`、`.tflite`、`real_board` 或完整训练链写成已经恢复完成。
