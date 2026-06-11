# T14 Review: P4 Frozen Benchmark Protocol Audit

**Reviewer**: Claude Code (normal review)
**Date**: 2026-05-08
**Task package**: `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md`

---

## Verdict: **PASS**

---

## 1. Blocking Issues

None.

---

## 2. Non-blocking Issues

### N1: Worker Output Summary 未列出所有已更新文档

`T14` 任务包的 Worker Output Summary 只列出了 `docs/protocols/benchmark/P4_benchmark_development_protocol.md` 作为产出，但 diff 显示同时更新了 `docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 和 `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md` 自身。Summary 应完整列出所有修改文件，方便后续 review 追溯。影响低——实际 diff 清晰可查。

### N2: Protocol 文档引用 `p4_multiscenario_hybrid_b_long.yaml` 未解释继承链

Section 2.3 提到 `Base long config: cnn_fpga/config/p4_multiscenario_hybrid_b_long.yaml`，但未解释它在 `p4_multiscenario_strong_baselines.yaml` 中以 `base_config` 形式被继承的关系。对不熟悉配置继承链的 reader 可能造成困惑。不阻塞，但建议 T15 worker 阅读时注意。

---

## 3. Missing Validation

无。本任务是 documentation-only audit，只要求只读核查。以下关键声明已通过 review 验证为正确：

| 声明 | 验证结果 |
|------|----------|
| Runner CLI 支持 `--scenario/--mode/--repeats/--paired-seeds/--run-dir/--repeat-start/--repeat-stop/--resume-only` | 确认：`run_p4_multiscenario_benchmark.py` 第 28–71 行有全部 8 个参数 |
| `p4_multiscenario_strong_baselines.yaml` 的 `frozen_baseline_set` 为 `[ekf, ukf, constant_residual_mu, rls_residual_b, hybrid_residual_b]` | 确认：config 第 16 行 |
| `p4_multiscenario_recovery_smoke.yaml` 的 `frozen_baseline_set` 为 `[static_linear, window_variance, ekf, cnn_fpga]` | 确认：config 第 46 行 |
| Recovery smoke 的 backend/artifact/interpreter 固定口径 | 确认：与 `07_handoff.md` 3.1/3.3/3.6 节一致 |
| T9 run dir `runs/p4_benchmark/p4multis_20260508_121828_0c12d7_46732` 存在 | 引自历史文档，与 handoff 3.8 节一致 |

---

## 4. Suspicious Implementation Details

无。整个 diff 由纯文档组成，不涉及代码、配置或 benchmark 语义修改。具体检查：

- **无硬编码**：文档中的参数值（interpreter 路径、config 路径、mode/scenario 名称）均引自实际代码/配置，非臆造
- **无 mock/stub 伪装**：文档 Section 10 明确声明了 T14 不声称的五项内容
- **无计划写成事实**：Section 6 标题为 "Approved T15 Bounded Run Plan"，是推荐方案而非已执行结果
- **无边界混淆**：recovery smoke / development bounded run / formal frozen benchmark 三层口径定义清晰

---

## 5. Scope Compliance

### 5.1 Allowed files — 全部合规

| 文件 | 是否修改 | 合规 |
|------|----------|------|
| `docs/tasks/Phase2/T14_p4_frozen_benchmark_protocol_audit.md` | 是（追加 Worker Output Summary） | OK |
| `docs/protocols/benchmark/P4_benchmark_development_protocol.md` | 是（新增） | OK |
| `docs/07_handoff.md` | 是（追加 T14 output status + 更新下一步建议） | OK |
| `docs/08_risks_and_open_questions.md` | 是（更新 R5/R9 和 Q10） | OK |
| `docs/04_task_board.md` | 否（worker 有意保持，等 Captain 整合） | OK |

### 5.2 Forbidden scope — 全部合规

- `cnn_fpga/benchmark/run_p4_multiscenario_benchmark.py` — 未修改，已验证 diff
- `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` — 未修改，已验证 diff
- `cnn_fpga/config/p4_multiscenario_recovery_smoke.yaml` — 未修改，已验证 diff
- `cnn_fpga/decoder/param_mapper.py` — 未修改
- baseline 集合、场景定义、seed 口径、ParamMapper 语义 — 未改变
- 正式长跑 benchmark — 未启动
- mock-backed 结果未写成真板或 .tflite 验收 — 文档反复明确标注边界

---

## 6. T14 Done Criteria Checklist

| # | Done Criterion | Status |
|---|----------------|--------|
| 1 | 明确正式 P4 frozen benchmark 与 recovery smoke 的区别 | **PASS** — Section 2 定义三层口径，Section 3 固定已有证据 |
| 2 | 明确下一步可运行的 bounded smoke 参数 | **PASS** — Section 6 给出完整 matrix（scenario/mode/repeat/seed/interpreter/config） |
| 3 | 明确不允许本任务直接改变 benchmark 口径 | **PASS** — Section 4 列出 7 项 frozen 约束 |
| 4 | 输出能被 T15 直接复用的 Worker 运行计划 | **PASS** — Section 7 提供可直接执行的 PowerShell 命令草案 |
| 5 | 未修改代码、未产生新事实性 benchmark 结论 | **PASS** — diff 纯文档，无代码/config 变更 |

---

## 7. Recommended Next Action

1. **Captain 整合**：Captain 应审查本 review，确认 T14 完成后更新 `docs/04_task_board.md`，将 T14 标记为 `[x]`，切换 Current Unique Task 至 `T15`
2. **T15 启动**：T15 worker 应严格按照 `docs/protocols/benchmark/P4_benchmark_development_protocol.md` Section 6–7 的 bounded matrix 和命令草案执行，不超出定义范围
3. **注意 config 切换**：T15 从 `p4_multiscenario_recovery_smoke.yaml` 切换到 `p4_multiscenario_strong_baselines.yaml`，baseline 集合完全不同（`static_linear/window_variance/ekf/cnn_fpga` → `ekf/ukf/constant_residual_mu/rls_residual_b/hybrid_residual_b`）。T15 worker 需确认 `p4_multiscenario_strong_baselines.yaml` 在 `C:\ProgramData\anaconda3\python.exe` 下能正常运行（该 config 的 `hybrid_residual_b` mode 需要 `artifacts/models/runtime_b_residual_v1/` 下的 model artifact）
