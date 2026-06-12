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
- `T20` 已补出 `docs/evidence_packs/deployment_boundary/real_board_hil_readiness.md`，后续真板任务应优先引用其中的 placeholder 证据、前置条件、最小 smoke 验收标准与禁止表述。
- `T21` 若作为 milestone review，必须继续区分 `readiness checklist` 与 `hardware validation`；任何只读总结都不能把 `board_backend.py` 的占位状态改写成现实板级完成。
- `T22` 若制定 real-board smoke execution plan，也只能产出计划、审计清单和量化阈值草案；除非后续真实硬件任务产出设备、寄存器、DMA 与 commit/ack 证据，否则仍不得写成 `hardware_validated`。
- `T22` 已补出 `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md`，后续真板任务应优先引用其中的 host-platform decision points、AXI/DMA 审计清单、Layer A-D 量化阈值草案与 evidence pack 要求。
- 即使 `docs/evidence_packs/deployment_boundary/real_board_smoke_execution_plan.md` 已存在，也只能写成 `execution plan exists, but it has not been executed`，不得因为 plan 文档就升级为真板已验证。
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
## 2026-06-12 Captain Update (T76/T77 boundary supersession)

- `T76` 已由 Captain 以 `PASS_WITH_WARNINGS` 收口。
- 它只在 docs-only 主线中完成 rendered preview、人工 QA、contact sheet / PDF bundle 和 Results-section assembly；它没有执行 benchmark、没有重跑 `.tflite`、没有执行真板 smoke，也没有改变任何 HIL / `.tflite` / real-board / benchmark 证据等级。
- `T76` 的残余 warning 只落在 paper-facing traceability/schema 精细度上：`R34` 约束的是 preview-source / stable-ID 书写粒度，而不是任何运行时或板级边界。
- 因此当前唯一任务切换为 `T77: 论文 note-draft 结果层同步与 T76 traceability hardening`。
- `T77` 仍是 docs-only 主线同步任务；它只允许把已有 `T74/T75/T76` 结果层材料同步到 `docs/paper_notes/*.tex`，并补强 `T76` 的 traceability 书写，不得改写 HIL / `.tflite` / real-board / benchmark 边界事实。

## 2026-06-12 Captain Update (T75/T76 boundary supersession)

- `T75` 已被 Captain 接受为 `PASS`。
- `T75` 只是在 docs-only 主线中完成 Results authoring、caption/placement lock、appendix bridge 和最终成图资产；它没有执行 benchmark、没有重跑 `.tflite`、没有执行真板 smoke，也没有改变任何 HIL / `.tflite` / real-board / benchmark 证据等级。
- 因此当前唯一任务切换为 `T76: Rendered figure QA and results-section assembly pack`。
- `T76` 仍是 docs-only 主线质量控制与装配任务；它只允许在 `T75` 已锁定的 prose/asset 边界内做 rendered preview QA、版式可读性审查和 manuscript-facing Results-section assembly，不得改写 HIL / `.tflite` / real-board / benchmark 边界事实。

## 2026-06-12 Captain Update (T74/T75 boundary supersession)

- `T74` 已被 Captain 接受为 `PASS`。
- 它只把主线 simulation/material-first 路线整理成 paper-ready 表、图、caption、插入映射和 traceability 资产；它没有执行 benchmark、没有重跑 `.tflite`、没有执行真板 smoke，也没有改变任何 HIL / `.tflite` / real-board / benchmark 证据等级。
- 因此当前唯一任务切换为 `T75: Main-text results prose and final figure authoring pack`。
- `T75` 仍是 docs-only 主线 authoring 任务；它只能基于 `T74` stable IDs 与既有证据做主文 Results 段落和最终成图 authoring，不得改写 HIL / `.tflite` / real-board / benchmark 边界事实。

## 2026-06-12 Captain Update (T73/T74 boundary supersession)

- `T73` 已被 Captain 接受为 `PASS`。
- 它只把主线 paper-facing claim/evidence、result/figure、risk 和 README 入口刷新到 post-`T72` 状态；它没有执行 benchmark、没有重跑 `.tflite`、没有执行真板 smoke，也没有改变任何 HIL / `.tflite` / real-board / benchmark 证据等级。
- 因此当前唯一任务切换为 `T74: Paper-ready simulation result and figure pack`。
- `T74` 仍是 docs-only 主线 paper-material 打包任务；它只能整理已有仿真证据、图表、caption 与 traceability 资产，不得改写 HIL / `.tflite` / real-board / benchmark 边界事实。

## 2026-06-11 Captain Update (T72/T73 boundary supersession)

- `T72` 已被 Captain 接受为 `PASS_WITH_WARNINGS`。
- 它只把 current-host 真板前提 gate / transfer-pack 提升为更严谨的 code-backed、checked-in、role-aware、可 replay / regeneration 的 read-only 包；它没有执行真板 smoke，没有验证 `board_backend.py`，也没有把证据升级为 `hardware_validated`。
- `T72` 关闭了 `R31`，但没有关闭 `R13/R14`，也没有打开 `T37`。
- 新的边界风险是 `R32`：future-host 最小 config 场景下，path provenance 仍不能精确区分 YAML 显式字段与代码默认值回退。
- 因此当前唯一任务切换为 `T73: Mainline claim/evidence and result/figure/risk ledger refresh`。
- `T73` 是 docs-only 主线台账任务；它只能刷新 claim/result/risk 口径，不得改写 HIL / `.tflite` / real-board / benchmark 证据等级。

## 2026-06-11 Captain 优先级调整（paper-first / board-lowest）

- 当前暂无可用的 `Linux + FPGA` 硬件宿主，因此真板执行路线在计划层面也维持 `blocked + lowest-priority backlog`。
- 在硬件条件变化前，main 分支只保留 read-only real-board truth / gate / provenance 维护，不新增任何真板 execution 任务来抢占 `T73`、`T74` 这类 paper-material 主线任务。
- 因而当前关于 real-board 的最强可写事实仍然只是 `T49/T71/T72` 的 gate / regeneration / provenance 边界，而不是任何执行成功或硬件验证。

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

## 2026-05-27 Captain Update (T62/T63 boundary supersession)

- `T62` has now been accepted as `PASS`.
- `T62` closes the specific provenance blocker from `T61`, but it does not change any HIL / P4 / `.tflite` / real-board truth boundary.
- `T62` remains bounded mock-backed software-HIL evidence only. It is not `FR8`, not `.tflite` validation, and not real-board validation.
- The current unique task is now `T63: FR8 statcalib comparator gate review`.
- `T63` is docs-only and stays entirely inside the same boundary: it may review whether a bounded FR8 task should exist, but it must not itself generate new comparator evidence or upgrade the current truth boundary.

## 2026-05-27 Captain Update (T63/T64 boundary supersession)

- `T63` has now been accepted as `PASS`.
- `T63` does not change any HIL / P4 / `.tflite` / real-board truth boundary. It only concludes that one bounded FR8 extension-lane task may proceed next.
- The current unique task is now `T64: FR8 statcalib extension-lane bounded benchmark`.
- `T64` must stay inside the same truth boundary: mock-backed software-HIL only, not `.tflite`, not real-board, not paper-grade expansion.
- `T64` may compare `statcalib` against the frozen benchmark set only as a separately labeled extension lane. It must not silently rewrite the historical `T24` frozen ranked table or overstate the minimal heuristic lane as a completed formal comparator.

## 2026-05-29 Captain Update (T64/T65 boundary supersession)

- `T64` has now been accepted as `PASS_WITH_WARNINGS`.
- `T64` changes no HIL / P4 / `.tflite` / real-board truth boundary. It adds one clean-provenance bounded extension-lane benchmark only.
- The strongest boundary fact after T64 is still narrow:
  - `T24` remains the historical frozen ranked table
  - `statcalib` remains only a separately labeled extension lane
  - current evidence remains mock-backed software-HIL only
- `R24` remains open because T64 does not transform the minimal statcalib lane into a mature validated calibration comparator.
- `R28` is now open because the report wording around execution shape and finish-timestamp provenance still needs an explicit consistency guard.
- The current unique task is now `T65: FR8 extension-lane consistency guard and report closeout`.
- `T65` must harden report/audit consistency only. It must not change benchmark semantics, create a new run root, or widen deployment-boundary claims.

## 2026-05-29 Captain Update (T65/T66 boundary supersession)

- `T65` has now been accepted as `PASS_WITH_WARNINGS`.
- `T65` changes no HIL / P4 / `.tflite` / real-board truth boundary. It closes report/audit consistency only.
- `R28` should now be treated as closed: the T64 pack is self-audited against its run artifacts and frozen-subset anchor.
- `R24` remains open: even after T65, the statcalib lane is still only a separately labeled bounded extension lane, not a mature validated calibration comparator.
- The current unique task is now `T66: FR8 statcalib sensitivity bounded benchmark`.
- `T66` must remain inside the same truth boundary: mock-backed software-HIL only, not `.tflite`, not real-board, not paper-grade benchmark expansion.
- `T66` may probe only a predeclared bounded statcalib sensitivity grid under clean provenance; it must not rewrite the historical `T24` frozen ranked table or change statcalib/runtime semantics.

## 2026-06-01 Captain Update (T66/T67 boundary supersession)

- `T66` has now been accepted as `PASS_WITH_WARNINGS`.
- `T66` changes no HIL / P4 / `.tflite` / real-board truth boundary. It adds one bounded local sensitivity result pack only.
- The strongest boundary fact after T66 is still narrow:
  - `T24` remains the historical frozen ranked table
  - `statcalib` remains only a separately labeled extension lane
  - current evidence remains mock-backed software-HIL only
- `R24` remains open because T66 does not prove teacher-anchor independence or mature calibration-comparator validity.
- The current unique task is now `T67: FR8 statcalib teacher-anchor dependence bounded benchmark`.
- `T67` must stay inside the same truth boundary: mock-backed software-HIL only, not `.tflite`, not real-board, not paper-grade benchmark expansion.
## 2026-06-05 Captain Update (T67/T68 boundary supersession)

- `T67` has now been accepted as `PASS_WITH_WARNINGS`.
- `T67` changes no HIL / P4 / `.tflite` / real-board truth boundary. It adds one bounded teacher-anchor result pack only.
- The strongest boundary facts after T67 are still narrow:
  - `T24` remains the historical frozen ranked table
  - `statcalib` remains only a separately labeled extension lane
  - current evidence remains mock-backed software-HIL only
- `R24` remains open because T67 does not produce a clean generated-only result pack; two comparison rows remain `mixed`.
- The current unique task is now `T68: FR8 statcalib generated-only robustness bounded benchmark`.
- `T68` must stay inside the same truth boundary: mock-backed software-HIL only, not `.tflite`, not real-board, not paper-grade benchmark expansion.

## 2026-06-08 Captain Update (T68/T69 boundary supersession)

- `T68` has now been accepted as `PASS_WITH_WARNINGS`.
- `T68` changes no HIL / P4 / `.tflite` / real-board truth boundary. It adds one bounded generated-only robustness result pack only.
- The strongest boundary facts after T68 are still narrow:
  - `T24` remains the historical frozen ranked table
  - `statcalib` remains only a separately labeled extension lane
  - current evidence remains mock-backed software-HIL only
- `R24` remains open because T68 does not yield a unique final threshold and does not make the whole predeclared grid uniformly clean.
- The current unique task is now `T69: FR8 statcalib clean-winner tie-break bounded benchmark`.
- `T69` must stay inside the same truth boundary: mock-backed software-HIL only, not `.tflite`, not real-board, not paper-grade benchmark expansion.

## 2026-06-10 Captain Update (T69/T70 boundary supersession)

- `T69` has now been accepted as `PASS_WITH_WARNINGS`.
- `T69` changes no HIL / P4 / `.tflite` / real-board truth boundary. It adds one bounded clean-winner tie-break result pack only.
- The strongest boundary facts after T69 are still narrow:
  - `T24` remains the historical frozen ranked table
  - `statcalib` remains only a separately labeled extension lane
  - current evidence remains mock-backed software-HIL only
  - the strongest clean answer is a persistent three-way tie, not a unique threshold
- `R24` remains open because T69 still does not promote the lane into a mature validated calibration comparator and does not make the broader predeclared grid uniformly clean.
- The current unique task is now `T70: FR8 statcalib bounded closure pack and promotion gate`.
- `T70` must stay inside the same truth boundary: no new run root, no `.tflite`, no real-board, no paper-grade benchmark expansion, and no rewrite of `T24` or historical FR8 run roots.

## 2026-06-10 Captain Update (T70/T50 boundary supersession)

- `T70` has now been accepted as `PASS`.
- `T70` changes no HIL / P4 / `.tflite` / real-board truth boundary. It adds one closure artifact and two explicit gates only.
- The strongest boundary facts after T70 are still narrow:
  - `T24` remains the historical frozen ranked table
  - `statcalib` remains only a separately labeled extension lane
  - current FR8 evidence remains mock-backed software-HIL only
  - no promotion and no unique-threshold claim are currently supported
- `R24` remains open because closure-pack completion does not promote the lane into a mature validated calibration comparator.
- The current unique task is now `T50: Training reproducibility and material-regeneration pack`.
- `T50` must stay outside the HIL/P4/deployment boundary: it may strengthen training reproducibility and material-chain evidence only, and it must not widen into `.tflite`, real-board, benchmark reruns, or theory-branch execution.

## 2026-06-10 Captain Update (T50/T48 boundary supersession)

- `T50` has now been accepted as `PASS`.
- `T50` changes no HIL / P4 / `.tflite` / real-board truth boundary. It adds one training/material evidence pack and one clean CPU-only bounded train+eval rerun only.
- The strongest boundary facts after T50 are still narrow:
  - `T24` remains the historical frozen ranked table
  - current FR8 evidence remains mock-backed software-HIL only
  - current training evidence is still bounded clean CPU-only evidence only
  - no `.tflite` runtime or real-board recovery claim is yet supported
- `R11` remains open because T50 does not prove full reproducibility or portability.
- The current unique task is now `T48: True .tflite runtime smoke gate`.
- `T48` must stay inside the `.tflite` runtime truth boundary only: no benchmark/HIL widening, no real-board semantics, no training rerun expansion, and no paper-grade deployment retelling.

## 2026-06-10 Captain Update (T48/T49 boundary supersession)

- `T48` has now been accepted as `PASS`.
- `T48` changes no HIL / P4 / real-board truth boundary. It only upgrades one narrow software-side runtime fact: under one isolated `tensorflow==2.21.0` environment on the current machine, preserved `static_theta_v2` float / int8 `.tflite` artifacts can really load and execute.
- The strongest boundary facts after T48 are still narrow:
  - `T24` remains the historical frozen ranked table
  - current FR8 evidence remains mock-backed software-HIL only
  - real-board validation is still absent
  - default-environment `.tflite` compatibility is still absent
- `board_backend.py` remains placeholder-backed, so the real-board boundary is still not crossed.
- The current unique task is now `T49: Real-board smoke execution gate`.
- `T49` must stay inside current-host real-board precondition truth only: no benchmark/HIL widening, no write-side MMIO/DMA/register activity, and no retelling of host-probe facts as real-board validation.

## 2026-06-10 Captain Update (T49/T71 boundary supersession)

- `T49` has now been accepted as `PASS_WITH_WARNINGS`.
- `T49` does not cross any HIL / P4 / real-board truth boundary. It only upgrades one narrow real-board fact: on the current Windows host, the honest gate verdict is `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`.
- The strongest boundary facts after T49 are:
  - `T24` remains the historical frozen ranked table
  - current FR8 evidence remains mock-backed software-HIL only
  - current-host true `.tflite` runtime remains narrow and isolated
  - real-board validation is still absent
  - `board_backend.py` remains placeholder-backed
- `T49` warning handling is `W1/W2/W3 = deferred -> R30`, so the current-host `NO_GO` stands, but the reusable gate path is not yet future-host hard enough.
- The current unique task is now `T71: Real-board gate regeneration and host-transfer pack`.
- `T71` must stay inside read-only real-board gate truth only: role-aware device-path readiness, checked-in artifact regeneration, and replay/regression hardening. It is not permission to open `T37`, run board smoke, or retell any result as real-board validation.
