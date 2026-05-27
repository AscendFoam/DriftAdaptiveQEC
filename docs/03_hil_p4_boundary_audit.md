# Architecture And HIL / P4 Boundary Audit

本文件对应 `docs/reference/AI_coding_workflow.md` 中 `03_architecture.md` 的工程边界部分。由于当前项目最容易出错的架构点是 HIL / P4 / `.tflite` 真实性边界，本文件保留原 boundary audit，并把它作为 Phase 2 Worker 的架构约束源。

## 1. 目的

本文件用于固定恢复期对 `P3 software HIL`、`P3 real-board HIL`、`P4 benchmark`、以及 `.tflite` 部署链路的统一表述口径。

它要回答的不是“这些代码能不能继续做”，而是：

1. 哪些链路已经是可描述的软件实现
2. 哪些只是 `mock` / `stub` / `placeholder`
3. 后续文档和复验报告应该怎么写，才不会夸大完成度

## 2. 架构总览

当前主线可以按三层理解：

1. Physics / data layer
   - `physics/` 负责 GKP 相关噪声、综合征测量与逻辑跟踪。
   - 历史 P0/P1/P2 结果来自这一层与 benchmark 层的组合。
2. Runtime / HIL layer
   - `cnn_fpga/runtime/` 负责 fast loop、slow loop、feature building、latency 与 param bank。
   - `cnn_fpga/hwio/mock_fpga.py` 提供 mock backend。
   - `cnn_fpga/hwio/board_backend.py` 仍是 real-board placeholder。
3. Benchmark / evidence layer
   - `benchmark/compare_full_vs_simplified_ler.py` 是 P0 smoke 入口。
   - `cnn_fpga/benchmark/run_hil_suite.py` 是 P3 software HIL 入口。
   - `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` 是 P4 benchmark wrapper，内部仍调用同一 HIL session stack。

## 3. 结论摘要

- `cnn_fpga/benchmark/run_hil_suite.py` 是真实存在的软件 HIL orchestration 入口，但它的“真实性”取决于 `hil.backend` 和 slow-loop inference artifact。
- `cnn_fpga/hwio/mock_fpga.py` 是当前可用的板侧行为仿真后端，属于 `mock backend`，不是 `real_board`。
- `cnn_fpga/hwio/board_backend.py` 仍是真板后端占位骨架；`cnn_fpga/hwio/fpga_driver.py` 也把 `board/real` backend 标为 future integration，不能写成“真板 HIL 已完成”。
- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` 不是一条独立于 HIL 的更真实执行链，它只是批量、多场景地调用同一个 `run_hil_session(...)`。
- `.tflite` 路径有两种：真实 `.tflite` 导出/推理路径，以及 `tflite_stub_v1` 回退路径。后者有工程价值，但不能被表述为真实 TFLite 部署完成。

## 4. Boundary Matrix

| Component | Intended role | Actual status now | Boundary tag | Key evidence | Recommended wording |
| --- | --- | --- | --- | --- | --- |
| `cnn_fpga/benchmark/run_hil_suite.py` | P3 HIL 会话入口 | 真实的软件 HIL orchestration。它通过 `hil.backend` 选择 backend，`backend == "mock"` 时构造 mock noise provider，并产出 `hil_events.json` / `hil_summary.json` | `software_hil_orchestrator` | `backend_name = ...hil.backend...`；`noise_provider = _build_mock_noise_provider(...) if backend_name == "mock"`；`save_json(... "hil_events.json")` | “软件 HIL 会话入口已存在，但结果真实性取决于 backend 与 inference artifact。” |
| `cnn_fpga/hwio/mock_fpga.py` | 板侧 FPGA 行为仿真 | 已实现 event-driven `mock backend`，能产生 `window_ready`、`commit_applied`、`commit_ack_asserted` 等事件，并维护 DMA/param-bank 语义 | `mock_backend` | 文件注释 `Mock FPGA backend for P3 HIL event-driven validation.`；`metadata={"backend": "mock_fpga"}` | “当前 P3 可复验路径优先基于 mock FPGA backend。” |
| `cnn_fpga/hwio/board_backend.py` + `cnn_fpga/hwio/fpga_driver.py` | 真板 MMIO/DMA backend | 仍是 `placeholder` 骨架，且 driver 层把 `board/real` backend 视为 future integration。`schedule_commit(...)` 返回大量 `None` 元信息，`step(...)` 仅刷新状态并返回空事件 | `placeholder_real_board_backend` | 文件注释 `Placeholder real-board backend...`；`return {"target_bank": None, ... "version": None, "ack_delay_us": None}`；`step(...): return []`；`fpga_driver.py` 中 `board/real` backend 被标为 reserved for future real-board integration | “真板 backend 还停在占位骨架，不能写成 real-board HIL 已完成。” |
| `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` | P4 多场景 benchmark 入口 | 真实存在的 benchmark 包装层，但核心执行仍直接调用 `run_hil_session(...)`，并未绕开 HIL backend 边界 | `p4_wrapper_over_hil` | `from cnn_fpga.benchmark.run_hil_suite import run_hil_session`；循环中直接 `summary = run_hil_session(...)` | “P4 benchmark 的真实性继承自同一条 HIL backend / artifact 链路。” |
| `cnn_fpga/model/export.py` | 导出部署产物 | 优先尝试真实 `.tflite` 导出；失败时自动回退为 `.tflite.json`，格式为 `tflite_stub_v1` | `true_tflite_or_stub_export` | `report = _export_true_tflite(...)`；`except Exception` -> `_export_tflite_stub(...)`；manifest `format: "tflite_stub_v1"` | “导出链支持真实 TFLite，也支持带明确标签的 stub 回退。” |
| `cnn_fpga/runtime/inference_service.py` | slow-loop 推理服务抽象 | `TFLiteHistogramPredictor` 同时支持真实 `.tflite` 与 `.tflite.json` stub manifest；两者输出 `source` 不同 | `true_tflite_or_stub_runtime` | stub 路径 `source="tflite_stub_service"`；真实路径 `source="tflite_service"` | “`backend=tflite` 还要继续区分真实 runtime 与 stub manifest runtime。” |

## 5. 统一口径规则

1. 可以写“P3 software HIL 主链存在”，但必须同时标注 `hil.backend` 和 inference artifact type。
2. 不能写“real-board HIL 已完成”或“真板 backend 已验收”，除非后续有独立真板证据覆盖 `board_backend.py` 当前占位状态。
3. 不能把 `P4 benchmark` 写成比 `run_hil_session(...)` 更真实的一条独立执行链；它只是同一 HIL 会话的批量包装与汇总。
4. 不能因为配置里写了 `backend=tflite`，就默认它一定是“真实 TFLite 部署”。必须继续区分：
   - 真实 `.tflite` -> `tflite_service`
   - `.tflite.json` stub manifest -> `tflite_stub_service`
5. 后续恢复期默认应优先选择“`mock backend` + 显式 artifact 标签”的最小复验路径，再考虑 `.tflite` 或 `real_board` 条件扩展。

## 6. 对后续任务的约束

- 后续所有 P3/P4 任务都必须显式写清：
  - backend 是 `mock` 还是 `board`
  - inference artifact 是 `artifact_npz`、真实 `.tflite`，还是 `.tflite.json` stub
- `T14/T15` 进入 P4 证据增强时，必须继承同一套边界标签；不能只写“P4 已跑通”。
- `T18` 进入 TFLite manifest / smoke plan 时，必须区分 `tflite_service` 和 `tflite_stub_service`。
- `T20` 进入 real-board readiness 时，不得修改 placeholder 语义或写成真板已完成。
- `T20` 的预期输出只能是 readiness checklist / 缺口清单 / 最小 smoke 验收标准；除非后续另开任务并具备真实设备证据，否则不得把 `board_backend.py` 的 placeholder 状态升级为已验收能力。
- `T20` 已补出 `docs/real_board_hil_readiness.md`，后续真板任务应优先引用其中的 placeholder 证据、前置条件、最小 smoke 验收标准与禁止表述。
- `T21` 若作为 milestone review，必须继续区分 `readiness checklist` 与 `hardware validation`；任何只读总结都不能把 `board_backend.py` 的占位状态改写成现实板级完成。
- `T22` 若制定 real-board smoke execution plan，也只能产出计划、审计清单和量化阈值草案；除非后续真实硬件任务产出设备、寄存器、DMA 与 commit/ack 证据，否则仍不得写成 `hardware_validated`。
- `T22` 已补出 `docs/real_board_smoke_execution_plan.md`，后续真板任务应优先引用其中的 host-platform decision points、AXI/DMA 审计清单、Layer A-D 量化阈值草案与 evidence pack 要求。
- 即使 `docs/real_board_smoke_execution_plan.md` 已存在，也只能写成 `execution plan exists, but it has not been executed`，不得因为 plan 文档就升级为真板已验证。
- `T23` 若锁定 P4 formal benchmark protocol，也只能产出协议、矩阵、预算、evidence pack 与 go/no-go 条件；不得把 protocol lock 写成 formal benchmark 已执行。
- `T24` 已执行并通过 Captain 收口为 `PASS_WITH_WARNINGS`，但其结果仍只能在 `mock-backed P4 wrapper over software HIL` 边界内使用；不得把结果写成 `.tflite` runtime 或 `real_board` validation。
- `T25` 已完成 result-boundary gate review；T24 可作为 completed frozen-set formal software revalidation，但仍不能升级为 runtime / board / paper-grade evidence。
- `T27` 已完成 teacher diagnostics 与机制证据路径审计；它只缩窄 R10/R20，不代表机制证据已修复。
- `T28` 若修复 teacher diagnostics missing-vs-zero 语义，只能在 `mock-backed` software HIL 边界内做最小代码/报告语义修复与 smoke，不得扩展 frozen set、baseline、`.tflite` runtime 或真板范围。
- `T28` 已完成 missing-vs-zero writer 语义修复和最小 smoke；`T29` 已修复 P4 markdown report 重复表头并通过 review。`T26` 已完成 calibration/statcalib docs-only feasibility gate，结论为 `CONDITIONAL_GO`；`T30` 已完成 statcalib separate comparator interface contract 和 interface-level tests，但未接入 slow loop、未进入 frozen benchmark、未形成 runtime/board evidence。
- `T36` 已完成 `seed=20260429` teacher-representation 收益收缩诊断，结论仍是 summary/final-snapshot-level hypothesis；不得把该诊断输出写成新的 benchmark evidence。
- `T38` 若做 `seed=20260429` trace-export probe，只能在 unchanged semantics 下补 per-window trace；不得启动新长跑、不得扩 baseline/scenario、不得改变 P3/P4/HIL 边界。
- `T38` 已完成并通过 Captain `PASS` 收口；其输出只能表述为 single-seed trace-level mechanism evidence，不能表述为 formal benchmark、runtime validation、real-board validation 或 mitigation success。
- `T31` 只允许补 training-chain portable dependency-lock plan；不得改变 P3/P4/HIL 边界，不得安装依赖、运行训练、运行 benchmark、创建 run/artifact，或把本机 `DLEnv` 事实写成跨机器保证。
- `T31` 已完成并通过 Captain `PASS` 收口；其输出是 dependency-lock planning evidence，不是 clean-environment rebuild proof。
- `T39` 只允许在 training-chain CPU-only clean-environment 范围内产出 draft lock 和 dry-run bootstrap；不得扩展到 GPU/CUDA portability、`.tflite` runtime、P3/P4 benchmark、real-board HIL、cleanup、baseline/scenario/seed policy 或正式训练结果。
- `T39` 已完成并通过 Captain `PASS` 收口；其输出是 clean-environment draft lock 与 dry-run/import-level bootstrap evidence，不是 real-training result、benchmark、`.tflite` runtime validation 或 real-board validation。
- `T40` 只允许在同一 clean CPU-only lane 内执行一次最小 real-training smoke，并把 `model_dir` / `report_dir` 重定向到 task-scoped isolated directories；不得改写 canonical historical `artifacts/models/*`、`artifacts/reports/*`，不得扩展到 benchmark、`.tflite` runtime、P3/P4/HIL、real-board、cleanup 或 GPU/CUDA portability。
- `T40` 已完成并通过 Captain `PASS` 收口；其输出是 clean CPU-only lane 的 one-run training smoke evidence，不是 benchmark、true `.tflite` runtime validation、GPU portability proof 或 real-board validation。
- `T33` 是 repo-hygiene execution task，不改变任何 HIL / P4 / `.tflite` / real-board evidence level；它只能把 manifest-listed tracked cache 从 Git index 中移除，不能被表述为新的实验或部署证据。
- `T34` 若推进，只允许整理 claim/evidence ledger 与 figure-table outline；它不能把 mock-backed software HIL、stub `.tflite`、clean-env smoke 或 readiness 文档升级成更高 evidence level。
- `T34` 已完成并通过 Captain `PASS` 收口；其输出是 paper-assembly ledger artifact，不是新的 benchmark、runtime、training 或 board evidence。
- `T35` 已完成并通过 Captain `PASS` 收口；其输出只是一份 paper draft skeleton 与 reviewer-risk audit，不是新的 runtime、board、benchmark 或 reproducibility evidence。
- `T41` 已完成并通过 Captain `PASS` 收口；其输出只是 Milestone 2K paper-assembly gate review，不是新的 runtime、board、benchmark 或 reproducibility evidence。
- `T42` 若推进，只允许做 docs-only 的 Background / Related Work scaffold 与 method-positioning calibration；它不能补写不存在的证据，也不能把历史 pre-recovery 结论静默升级为当前已复验事实。
- `T43` 已完成并由 Captain 以 `PASS` 收口；其输出只是 Background / Related Work prose draft，不得把 prose 文字升级为新的 HIL / benchmark / `.tflite` / hardware 证据。
- `T44` 进入 `Research Reality Recovery Mode` 后，只允许冻结和审计 claim/evidence/material truth；不得借 recovery baseline 任务改写任何 HIL、P4、`.tflite` 或真板边界。
- `T45` 已完成并由 Captain 以 `PASS` 收口；它只锁定 benchmark-expansion protocol，不改变任何 HIL / P4 / `.tflite` / real-board 边界。
- `T46` 已完成并由 Captain 以 `PASS` 收口；其输出只是 docs-only 的 multi-seed mechanism/intervention plan 与 trace pack，不是新的实验结果。
- `T54` 已完成并由 Captain 按 `PASS` 收口；它只把单 seed 诊断升级为 bounded multi-seed trace-only diagnostic generalization，不改变当前 mock-backed P4 wrapper over software HIL 的事实边界。
- `T55` 只允许在同一条 mock-backed P4 wrapper over software HIL lane 内做一个 config-only 的 I1 干预测试：复用既有 6-seed 模型资产，只测试 `residual_clip_b: 0.06` 的 Gated v5 intervention，不得借机扩 baseline、扩 scenario、重训模型、触碰真板 / `.tflite` / cleanup，或把 intervention result 写成 causal proof。
- `T55` 已完成并由 Captain 按 `PASS` 收口；它仍然只提供 mock-backed software-HIL lane 内的 bounded intervention evidence，不升级任何 runtime / board / paper-grade 边界事实。
- `T56` 已完成并通过 `PASS` 收口；其作用只是在 docs-only 层面重写 claim wording 和 next-lane recommendation，不能运行任何新 benchmark、trace export、`.tflite`、真板、cleanup 或 comparator 执行。
- `T47` 现在只能作为 hedge-conditioned docs-only paper-material lane 推进；它不改变 HIL / P4 / `.tflite` / real-board 边界。

## 7. 当前推荐表述

- 可以说：`P3 software HIL scaffold exists and is mock-backed unless explicitly proven otherwise.`
- 可以说：`P4 benchmark currently reuses the same HIL session stack and inherits its realism limits.`
- 可以说：`real-board readiness checklist exists, but hardware validation evidence does not.`
- 可以说：`real-board smoke execution plan exists, but it has not been executed.`
- 可以说：`P4 frozen-set formal software revalidation has been executed and reviewed as mock-backed software HIL evidence.`
- 可以说：`P4 formal software revalidation is not deployment, true TFLite runtime, or hardware validation.`
- 可以说：`statcalib now has an interface-only contract and tests, but it is not integrated into slow-loop or formal benchmark evidence.`
- 可以说：`T36 narrows seed=20260429 to a residual-amplitude / teacher-delta hypothesis, but trace-level causality is still open.`
- 可以说：`T38 adds bounded single-seed trace evidence for seed=20260429, but R10 remains open until mitigation and broader confirmation exist.`
- 可以说：`Teacher diagnostics missing-vs-zero output semantics have been repaired for the current writer path, but this is still not a full mechanism-evidence repair.`
- 可以说：`T31 produced a training-chain portable dependency-lock plan, but did not rebuild a clean environment.`
- 可以说：`T39 completed a CPU-only clean-environment draft lock and dry-run bootstrap, but it is not a real training run, benchmark, GPU portability proof, TFLite runtime validation, or real-board validation.`
- 可以说：`T40 completed one isolated CPU-only minimal training smoke, but it is still not a benchmark, TFLite runtime validation, GPU portability proof, or real-board validation.`
- 可以说：`T42 completed only a docs-only Background / Related Work scaffold plus method-positioning calibration; it does not upgrade any HIL, benchmark, TFLite, reproducibility, or hardware evidence.`
- 可以说：`T43 is limited to Background / Related Work prose drafting under the same evidence boundary; it is not full-manuscript expansion and cannot be used to claim stronger validation status.`
- 可以说：`Research Reality Recovery Mode is a governance freeze and audit mode, not an evidence upgrade.`
- 不可以说：`real-board HIL complete`
- 不可以说：`tflite deployed`，除非已明确是 `tflite_service` 而不是 `tflite_stub_service`
## 2026-05-24 Captain Update (T47/T57 boundary supersession)

- `T47` 已完成并由 Captain 以 `PASS` 收口；其输出只是在 docs-only 层面冻结 paper ablation/material ledger，不改变任何 HIL / P4 / `.tflite` / real-board 边界。
- 当前唯一任务 `T57` 仍处在同一条 mock-backed P4 wrapper over software HIL 边界内。
- `T57` 只能作为锁定 `T24` protocol 的 FR7 feature/teacher ablation bounded re-execution 推进，不得被改写成 benchmark expansion、`.tflite` runtime validation、real-board validation 或 mechanism closure。

## 2026-05-26 Captain Update (T58/T59 boundary supersession)

- `T58` 已由 Captain 接受为 `PASS_WITH_WARNINGS`；其 warning `N1-N4` 全部按 `accepted` 处理，且没有新的 `deferred` / `rejected` warning。
- `T58` 只关闭 `FR6` 的 docs-only figure-pack 缺口，不改变任何 HIL / P4 / `.tflite` / real-board 边界事实。
- 当前唯一任务切换为 `T59: Statcalib separate comparator lane integration and bounded smoke`。
- `T59` 允许新增一个单独标记的 `statcalib` slow-loop comparator lane，但不得把它静默插入 frozen `T24` ranked set，也不得改写 `ParamMapper` 现有主线语义。
- `T59` 的 bounded smoke 产出即使成功，也仍只属于 mock-backed software-HIL lane 的 bounded integration evidence；它不是 `.tflite` validation，不是 real-board validation，也不是 `FR8` 正式结果表本身。

## 2026-05-26 Captain Update (T57/T58 boundary supersession)

- `T57` 已由 Captain 接受为 `PASS`；其输出仍严格停留在同一条 mock-backed `P4` wrapper over software-HIL truth boundary 内。
- `T57` 关闭的只是 `FR7` 的 frozen-set result-table gap，不得被改写成 mechanism closure、causal proof、expanded benchmark evidence、`.tflite` validation 或 real-board validation。
- `T57` 进一步收紧了 paper wording 边界：由于 `hybrid_no_teacher_params` 在 4 个 frozen scenarios 中都成为最佳模式，teacher params necessary 的表述不再安全。
- 当前唯一任务切换为 `T58: FR6 multi-seed mechanism/intervention figure pack`。
- `T58` 是 docs-only 任务，只能复用既有 `T54/T55/T56` 证据，且不得运行任何新的 benchmark、trace export、intervention、retraining、`.tflite` 或 real-board 工作。

## 2026-05-26 Captain Update (T59/T60 boundary supersession)

- `T59` has been accepted as `PASS_WITH_WARNINGS`; its warning classification is `W1 deferred`, `W2 accepted`, `W3 deferred`.
- `T59` only closes separate-lane integration and one bounded smoke. It does not open `FR8`, and it does not upgrade the evidence to formal comparator ranking.
- The deferred items from `T59` confirm that pre-FR8 work is still required at the semantics and regression-coverage layer.
- The current unique task is now `T60: Statcalib lane isolation and regression hardening`.
- `T60` may harden `statcalib` semantics and tests only. It must not create a new run root, rerun the T59 smoke, widen benchmark scope, or touch theory-only branch materials.
- Even after T60, any future fairness or `FR8` task remains inside the same mock-backed software-HIL truth boundary unless a later task explicitly upgrades that evidence.

## 2026-05-27 Captain Update (T60/T61 boundary supersession)

- `T60` has been accepted as `PASS`; it introduces no new warning item and does not change any HIL / P4 / `.tflite` / real-board truth boundary.
- `T60` closes the T59 cross-mode semantics blocker and regression-gap blocker only. It does not convert the statcalib lane into formal comparator evidence.
- `R26` should now be treated as closed. `R27` remains open and is now limited to provenance-clean fairness/robustness sanity before any `FR8` task.
- The current unique task is now `T61: Statcalib clean-provenance fairness sanity rerun`.
- `T61` still stays fully inside the same mock-backed software-HIL truth boundary: same T59 smoke config family, same two scenarios, same three modes, `--paired-seeds`, and only `repeats=2` as a bounded CLI-strengthening step.
- Even if `T61` succeeds, that result still is not `.tflite` runtime validation, real-board validation, or `FR8` by itself.

## 2026-05-27 Captain Update (T61/T62 boundary supersession)

- `T61` has been judged `BLOCK`; the fairness signal persisted, but the clean-provenance goal failed because the final artifact was not anchored to one single clean commit identity.
- That blocked result does not change any HIL / P4 / `.tflite` / real-board truth boundary.
- `T61` remains only a blocked mock-backed software-HIL sanity rerun artifact; it is not `FR8`, not `.tflite` validation, and not real-board validation.
- The current unique task is now `T62: Statcalib provenance-isolated fairness rerun`.
- `T62` stays inside the exact same mock-backed software-HIL truth boundary: same T59/T61 config family, same two scenarios, same three modes, `--paired-seeds`, and `repeats=2`.
- `T62` is allowed to repair provenance isolation only. It must not widen comparator scope, change source/config semantics, or mix mainline experiment work with theory-only branch materials.
