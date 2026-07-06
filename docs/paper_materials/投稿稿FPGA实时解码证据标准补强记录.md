# 投稿稿 FPGA 实时解码证据标准补强记录

日期：2026-07-06

## 变更对象

- `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 变更内容

- 在 Related Work 的 `Real-time and FPGA QEC decoders` 段落中补充比较单位：既有 FPGA / real-time decoder 文献报告的是完整 stabilizer-code decoder workload 的 device-level latency、memory / resource 与 integration evidence；本文当前只讨论 GKP physical-layer affine correction surface。
- 在 `FPGA-facing datapath contract` 小节补充当前可支撑的三层软件证据：
  - analytical operation-count：affine fast path 为 4 次乘法、4 次加法和 6 个 stored state scalars；
  - Q4.20 software fixed-point emulation：受控样本中未观察到 residual-boundary crossing-rate 改变；
  - software runtime counters：记录 commit、fast-cycle violation、overflow 和 saturation。
- 明确 stronger hardware claim 的缺口：measured latency、resource、power、bit-accurate fixed-point agreement、source-vs-board trace、synthesis 和 board data。

## 可写边界

- 可以写成：当前 affine fast path 是一个低复杂度、固定点可检查、可观测的 future FPGA experiment target。
- 可以写成：当前证据支持 datapath contract / software feasibility / analytical cost rationale。
- 不可以写成：已完成 FPGA synthesis、timing closure、resource / power measurement、real-board execution、source-vs-board agreement 或 board-level latency measurement。

## 验证计划

- 重新生成 `submission_draft_source_data_manifest.csv/json`。
- 运行 `audit_submission_draft_source_data.py`，确认 source-data 审计仍为 `PASS_WITH_LIMITATIONS` 且无 failed checks。
- 重新编译 `CNN_FPGA_GKP_submission_draft.tex`。
- 扫描内部项目语、硬件 overclaim 语和 LaTeX warning / error。
