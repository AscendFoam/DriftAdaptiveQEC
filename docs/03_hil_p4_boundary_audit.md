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

## 7. 当前推荐表述

- 可以说：`P3 software HIL scaffold exists and is mock-backed unless explicitly proven otherwise.`
- 可以说：`P4 benchmark currently reuses the same HIL session stack and inherits its realism limits.`
- 可以说：`real-board readiness checklist exists, but hardware validation evidence does not.`
- 可以说：`real-board smoke execution plan exists, but it has not been executed.`
- 可以说：`P4 frozen-set formal software revalidation has been executed and reviewed as mock-backed software HIL evidence.`
- 可以说：`P4 formal software revalidation is not deployment, true TFLite runtime, or hardware validation.`
- 可以说：`Teacher diagnostics root cause has been narrowed, but the mechanism-evidence path is not repaired yet.`
- 不可以说：`real-board HIL complete`
- 不可以说：`tflite deployed`，除非已明确是 `tflite_service` 而不是 `tflite_stub_service`
