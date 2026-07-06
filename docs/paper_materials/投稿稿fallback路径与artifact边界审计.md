# 投稿稿 fallback 路径与 artifact 边界审计

本文档服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它回答一个审稿人会直接追问的问题：当前投稿稿的数值表、runtime 说明和硬件占位符，是否把 `mock`、`.npz` model artifact、`.tflite`、`.tflite.json` stub、real-board placeholder 或 historical artifacts 混写成了同一种证据。

本文档不新增实验，不运行 benchmark，不改写 `runs/` / `artifacts/`，不改变 `T24`、`T48`、`T49/T71/T72`、`T64-T70` 或 `T90` 的证据等级。它是一份文档级边界审计，不是自动化 fallback absence proof。

## 一句话结论

当前投稿稿可以把主结果写成 `mock-backed software-HIL` frozen ranking；可以把 hybrid residual branch 写成使用 preserved `.npz` model artifact 的 software-HIL 慢环路；可以把 `T48` 写成 isolated current-host true `.tflite` support；可以把 `T49/T71/T72` 写成 read-only real-board gate / provenance。当前不能写成 `.tflite` main ranking、real-board execution、deployment closure、paper-grade expanded benchmark 或 “无 fallback 风险已机械证明”。

## 1. 审计问题

| 问题 | 当前答案 | 证据锚点 |
| --- | --- | --- |
| T24 主结果是否是真板或 `.tflite` 结果？ | 否。T24 是 `mock-backed software-HIL` formal revalidation。 | `docs/review/T24_review.md`、`docs/review/T25_p4_formal_evidence_gate_review.md`、`runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/summary.json` |
| T24 的 hybrid 模式是否用了可追踪 artifact？ | 是，但它是 `.npz` model artifact，不是 `.tflite`。T24 `comparison.csv` 中 hybrid rows 记录 `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz`；baseline rows 的 `artifact_path` 为空。 | `comparison.csv`、`summary.json`、`cnn_fpga/runtime/inference_service.py` |
| P4 benchmark 是否绕过 HIL backend 边界？ | 否。`run_p4_multiscenario_benchmark.py` 直接调用 `run_hil_session(...)`，所以 P4 真实性继承同一 HIL backend / artifact 链路。 | `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py`、`docs/03_hil_p4_boundary_audit.md` |
| `.tflite` runtime 是否可和 `.tflite.json` stub 混写？ | 不能。runtime 代码显式区分 `source="tflite_service"` 和 `source="tflite_stub_service"`；T48 还显式拒绝 stub sidecars。 | `cnn_fpga/runtime/inference_service.py`、`artifacts/t48_true_tflite_runtime_gate/t48_true_tflite_runtime_gate.json` |
| real-board gate 是否证明真板运行成功？ | 否。当前 gate verdict 是 `NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE`；`board_backend.py` 仍是 placeholder real-board backend。 | `artifacts/t49_real_board_smoke_execution_gate/t49_real_board_smoke_execution_gate.json`、`artifacts/t71_real_board_gate_regeneration_pack/current_host_regenerated_gate.json`、`docs/review/T72_review.md`、`cnn_fpga/hwio/board_backend.py` |

## 2. 当前 evidence / artifact 分层

| 层级 | 当前 artifact / 入口 | 当前可写用途 | 禁止外推 |
| --- | --- | --- | --- |
| T24 主结果 | `runs/p4_benchmark/T24_formal_software_revalidation_20260510_200743/comparison.csv`、`summary.json` | 四场景、五模式、paired-seed、`repeats=2` frozen software-HIL ranking | `.tflite` ranking、real-board ranking、expanded benchmark、SOTA |
| T24 hybrid slow-loop artifact | `artifacts/models/runtime_b_residual_v1/tiny_cnn_20260401_083648_2fc740424c0d.npz` | preserved `.npz` model artifact 支持的 hybrid residual branch | true `.tflite` runtime、portable deployment、board execution |
| Baseline modes | T24 rows with empty `artifact_path` | EKF/UKF/constant/RLS 等 software-HIL baseline comparison | learned artifact parity 或 `.tflite` baseline |
| P4 wrapper | `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` | 批量、多场景调用同一 HIL session stack | 更真实的独立 board path |
| HIL orchestrator | `cnn_fpga/benchmark/run_hil_suite.py` | `hil.backend` 决定 mock/board；mock 时构造 mock noise provider | 不标注 backend 的 “HIL complete” |
| `.tflite` support | `T48` isolated runtime gate | selected preserved float/int8 `.tflite` artifacts 在当前 isolated env 中可真实执行 | default env、HIL closure、deployment closure |
| `.tflite` stub support | `.tflite.json` with `format="tflite_stub_v1"` | 可作为显式标记的工程 fallback / manifest path | 真实 TFLite runtime |
| Real-board gate | `T49/T71/T72` gate / regeneration / review | read-only readiness and provenance with current-host `NO_GO` | board validation、hardware execution、source-vs-board agreement |
| FR8 / statcalib | `docs/evidence_packs/statcalib_fr8/fr8_statcalib_bounded_closure_pack.md` | supplement-side extension lane and no-promotion gate | mature comparator、T24 replacement、unique threshold |

## 3. 稿件当前应如何写

可以写：

- `The main comparison is a locked mock-backed software-HIL benchmark.`
- `The hybrid residual branch uses a preserved .npz model artifact in the slow loop.`
- `The .tflite evidence is isolated current-host runtime support for selected preserved artifacts, not the main benchmark path.`
- `The real-board material is a read-only gate with a current-host NO_GO verdict.`
- `The fallback and artifact taxonomy is documented, and a first source-data audit helper now checks the current main tables and figure source CSVs; a full table-to-artifact audit remains future work.`

不要写：

- `The main results were obtained with true .tflite runtime.`
- `The system was validated on a real FPGA board.`
- `The board backend is implemented and accepted.`
- `The fallback-free runtime path has been mechanically proved for every result row.`
- `T48 closes deployment.`
- `T49/T71/T72 prove real-board execution success.`
- `FR8/statcalib is promoted into the frozen main table.`

## 4. 当前稿件动作

1. 在 `CNN_FPGA_GKP_submission_draft.tex` 附录加入 `Artifact-Type and Fallback Boundary Audit`。
2. 把 T24 主结果、hybrid `.npz` artifact、T48 `.tflite` gate、T49/T71/T72 board gate 与 FR8/statcalib extension lane 分列说明。
3. 在附录中显式保留 limitation：当前审计是文档级和源码/证据路径级，不是自动化逐行 artifact hash / fallback absence proof。
4. 不改摘要强度，不把当前审计写成新的实验结果。

## 5. 后续真正投稿前还需要的更强材料

1. 扩展机器可读 table-to-artifact audit helper：当前第一版已读取 TeX/source-data 表、T24 `comparison.csv`、T24 `summary.json` 和 figure source CSV，检查主结果、Fig. 2-4 source data 与图件 manifest；后续仍需覆盖全文表格。
2. 每个 result row 的 artifact hash / config hash / git commit / runner version / environment manifest；当前第一版只记录 file-level hash 与 shared hybrid model artifact hash。
3. `.tflite` path 的 negative guard：明确拒绝 `.tflite.json` stub 通过 true-runtime gate。
4. real-board path 的 future-host gate：设备路径、bitstream/RTL/DMA contract、AXI map、latency/resource、source-vs-board vectors。
5. 结果不确定性：至少 paired uncertainty / bootstrap / confidence interval，而不是仅 `n=2` descriptive SD。

## 6. 风险结论

当前投稿稿在 artifact/fallback 边界上可以继续推进，因为主结果、runtime support 和 board gate 已被分层，且已有第一版 source-data 机械审计；但它仍不能称为 submission-ready 的硬证据包。审稿人若要求 reproducible artifact package，当前最可能被指出的缺口不是“完全没有 provenance”，而是“provenance 仍只覆盖当前主表和图件 source data，还缺全文 table audit、per-row hash manifest 和更强统计/holdout 实验”。
