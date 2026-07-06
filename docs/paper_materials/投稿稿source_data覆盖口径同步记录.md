# 投稿稿 source-data 覆盖口径同步记录

日期：2026-07-03

对应稿件：`docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`

## 本轮修改

- 将 `Runtime Artifacts and Reproducibility Limits` 中的 source-data coverage 句子更新为当前机械审计实际覆盖范围。
- 新表述列明当前 helper 覆盖 main result、paired-delta、ablation / mechanism、statistical-calibration、controlled oracle / wrapped-Gaussian、sequence-controlled baseline、holdout stress、fast-path cost、fixed-point parity、runtime-discipline、logical-channel surrogate 和 metric-readiness matrix。
- 同时保留限制：helper 仍未覆盖每一个 narrative scope / availability / requirements table。
- 将稿末 `Claim Scope and Source Data` 表拆成 primary claims 与 diagnostic/runtime/validation continuation 两个表，并补齐到当前 source-data 层级：controlled baseline sanity checks、holdout drift stress diagnostics、runtime counters 与 Fig. 5 validation-contract summary 分别列为独立 claim guardrail。

## 边界

- 本轮只同步稿件文字与 `投稿稿source_data机械审计报告.md` 的当前 coverage 事实。
- 本轮不新增 source-data 文件、不运行新 benchmark、不补 CI / p-value、不补 holdout formal benchmark、不补 hardware evidence。
- 新增 claim-source-data 行只登记已有表格、图件与 source CSV 的可写边界，不改变任何实验数值或证据等级。
- `PASS_WITH_LIMITATIONS` 仍然只表示机械一致性检查通过，不等于完整复现、强统计、deployment 或 real-board validation。

## 验证

- 需要重新运行 source-data 机械审计。
- 需要重新扫描内部治理 / 待办词。
- 需要重新编译投稿稿。
