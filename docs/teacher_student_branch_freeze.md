# T4.4.5 teacher-student strong/falsified 分支冻结

## 结论

机器判定激活 `qualified_student_retention`，没有激活
`drift_regime_aware_map_lut` fallback。这里的“strong”只表示：T4.4.1--T4.4.4 的 hash-bound
证据同时通过，4-state student 在新的 paired seeds、cutoff 12/16 和三个预注册物理指标上保留了至少
90% 的 teacher-vs-standard gain。它不是“所有 NMF 都优于 MF”、机制唯一性、原生 leakage、长时 OOD、
量化 RTL 或硬件结论。

机器产物：

- `docs/t4_4_5_teacher_student_branch_freeze.json`；
- `docs/t4_4_5_teacher_student_branch_freeze_source_data.csv`；
- 判定器 `cnn_fpga/benchmark/teacher_student_branch_freeze.py`；
- 回归 `tests/test_teacher_student_branch_freeze.py`。

## 只读输入与判定规则

判定器不重新训练、不重新运行物理 evaluation，也不使用 T4.4.4 结果选择新阈值。它只读消费：

| Parent | 冻结证据 | 本任务复核 |
| --- | --- | --- |
| T4.4.1 | fresh 3-restart bounded GRU teacher | status、21 gates、implementation、checkpoint/source hash、fresh/validation-only 语义 |
| T4.4.2 | frozen post-hoc hidden/control analysis | status、17 gates、implementation/source hash、zero optimizer step、leakage OOD 语义 |
| T4.4.3 | strict-split 4-state student | status、16 gates、implementation/checkpoint/student/source hash、evaluation-blind 与 fail-closed 语义 |
| T4.4.4 | dual-cutoff physical retention | status、18 gates、implementation/source hash、预注册 0.90 point/CI gate 与六个 stochastic 指标 |

仅当 8 个 evidence predicates 全真时才激活 qualified branch；任一 predicate 为假就激活 fallback，并把
teacher/distillation 的 active claims 替换为单一 MAP-LUT fallback claim。缺少 parent 则直接拒绝输入，不能用
“缺证据”伪装成一次成功的 branch decision。

## 当前 qualified branch 允许的 claim

1. fresh bounded-residual teacher 在 matched two-level finite-cutoff ten-cycle simulator 和既定 strict split 内可复现；
2. frozen 4-state student 在 cutoff 12/16 的新 paired seeds 上，对 selection score、fidelity lifetime 和
   logical-Z lifetime 的 point retention 与 paired-bootstrap 95% CI lower 均不低于 0.90；
3. 只可报告 float 解析成本：student 为 95 scalars/87 MAC，teacher 为 72,853 scalars/72,266 MAC。

T4.4.4 中最低 point retention 为 `0.981457`，最低 CI lower 为 `0.944501`。以上是 finite-model、
metric-specific retention，不是一般算法排序定理。

## 必须保留的反证与禁止 claim

cutoff 12 的五-agent MF mean selection score `0.557115` 高于 teacher `0.552952`；cutoff 16 则 teacher
`0.593930` 高于 MF mean `0.579684`。这个 reversal 是 strong branch 的组成条件而不是要删除的异常：它强制
保留“student retention 成立，但 universal NMF-over-MF 被否证”的解释。

以下结论继续禁止：universal NMF-over-MF、唯一 belief/单指数机制、全局优化或 oracle 最优、10-cycle
control oracle、原生 multilevel leakage/SPAM、long-horizon/OOD retention，以及 quantized RTL/synthesis/
FPGA/board/device performance。

## Fallback 与撤销条件

fallback 是现有 observed-only drift/regime-aware MAP-LUT、conservative health/event FSM 和 atomic parameter
bank 路径；它不保留 teacher 或 distillation claim。T5.2 leakage/SPAM、T5.4 OOD/长时、T5.5 fixed-point/
资源/deadline、T6 RTL/transport/board 任一 gate 失败，或 T4.4.1--T4.4.4 任一 hash/gate 失效，均撤销
qualified branch 并激活 fallback。

## 非 demo 审计

- Source Data 共 112 rows：4 个 parent artifacts、72 个 parent gates、7 个 file bindings、6 个 retention
  gates、8 个 branch predicates、3 个 active claims、7 个 prohibited claims、5 个 revocation triggers；
- 11/11 machine contract gates 通过；
- 17 项直接测试通过，其中 9 类独立 evidence mutation 均证明自动 fallback 且 teacher claim 被删除；
- decision contract hash 为
  `c0d7e9ffd305fea0666140bbf47635a34493b4ab4632ca3e809092307193a91c`。

