# Retired Reference Documents

本目录归档从 `docs/reference/` 退役的早期建议、实验设计和环境方案文档。归档日期：2026-06-11。

这些文档保留为历史上下文和旧决策来源，不再承担当前计划、当前事实或当前证据状态。当前项目事实仍以 `docs/02_experiment_plan.md`、`docs/04_task_board.md`、`docs/07_handoff.md`、`docs/08_risks_and_open_questions.md` 和对应 task/review 文档为准。

## 归档文件

| 文件 | 退役原因 | 已保留到当前计划的内容 |
| --- | --- | --- |
| `科研纠偏意见.md` | 原始 recovery-first 纠偏意见，所指 `reality_recovery` 治理文件已退役；当前阶段已回到受控开发 | claim/evidence ledger、result/figure ledger、反作弊审计、human brief、论文 claim 必须先证据化 |
| `CNN_FPGA_GKP_工程化实验方案.md` | 早期工程实施方案混合了软件 HIL、`.tflite`、真板与旧 P3/P4 状态；部分状态口径已过时 | 双回路架构、接口/数据契约、ParamMapper 语义冻结、fixed-point、latency、commit/fallback、日志溯源、TFLite 准入 |
| `CNN_FPGA_GKP_实验设计.md` | 早期技术分析和实验设计包含预期性能估计、硬件实现假设和过宽 baseline 叙事 | 漂移场景、变量控制、指标体系、统计方法、可复现清单、baseline taxonomy |
| `CNN_FPGA_GKP_TFLite_独立环境方案.md` | 独立环境建议仍可参考，但当前 T48 已形成更窄的 current-host true runtime 边界；不能继续放在 reference 中承担当前 runtime 状态 | isolated env bootstrap、true `.tflite`/stub 区分、TensorFlow interpreter 本机验证、LiteRT/板端 runtime 后置 |

## 使用规则

1. 只把本目录文件作为历史原文和参考素材使用。
2. 不从本目录直接引用任何“已完成”“当前状态”“真板验证”“paper-grade benchmark”结论。
3. 如果要恢复其中的实验、runtime、HIL、真板或论文建议，先在当前任务体系中写成独立任务包，明确 Allowed files、Forbidden scope 和 Verification。
4. 任何来自本目录的技术建议进入正文前，应先同步到 `docs/02_experiment_plan.md` 的后续计划部分或新的任务文档，并重新核对当前边界。
