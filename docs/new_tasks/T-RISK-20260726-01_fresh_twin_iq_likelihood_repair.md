# T-RISK-20260726-01 fresh twin IQ/likelihood 修复与资格复核

- Task ID：T-RISK-20260726-01
- 标题：在不回写 T9.2.4 NO-GO 的前提下修复双后端 IQ/likelihood 语义并执行有功效的 fresh qualification
- 日期：2026-07-26—2026-07-27
- 状态：Done（Scientific NO-GO）
- Formal verdict：`NO_GO_FRESH_TWIN_QUALIFICATION`
- Formal analysis：`e15397357374073a31b66edbc3417f799eb974db5d7789fed4e57850b94485c9`
- Post-outcome audit：`d065f2eb939cd0e84e55545249728e77699776d1860d7673064a384b3e570dd9`
- Raw execution：`b2595c43bcee3d8c4177d984cd1e5b91c7a5e1256a9e0ca8366d50b680c28596`

## 输入材料

- T9.2.4 不可变正式 `NO_GO_TWIN_QUALIFICATION`、release 与 post-outcome audit；
- T9.2.2/T9.2.3 双 backend 及 T9.2.6 raw-IQ/platform interface freeze；
- `R-N182`：旧正式实验的 IQ/likelihood 语义混合、伪重复和 1,042-way max-T 功效不足；
- 旧正式矩阵只用于历史失败定位，不进入 fresh threshold、margin、seed 或 formal outcome；
- 所有 official/Puviani、LER/lifetime、真实板、physical break-even 与 SOTA 字段保持 `null`。

## 执行方案

1. 以不依赖 A/B backend、NumPy 或 backend RNG 的第三解析锚冻结完整二维 Gaussian mixture 语义。
2. 将 raw IQ 与 likelihood evidence 降为诊断；正式主 estimand 使用 pre-prior Rao–Blackwell predictive moments/CDF、common-heldout proper score/LLR/posterior 和 RB reset success。
3. 将旧 1,042-way max-T 改为预注册 cellwise TOST + 全 cell intersection-union；same-backend null 与 A/B alternative 分开做 design power。
4. 冻结全新且互斥的 round/trajectory/heldout seed split，正式样本量为 768 round/backend 与 256 trajectory/backend。
5. 实现 592-cell、528,384-row、92-column 的 append-only/hashing/resumable runner；runner 不计算 verdict。
6. 实现不 import physics/runner 的独立 verifier，流式校验 CSV/ZIP/NPZ、量化证书和 1,589 个门，并只产生 PASS/NO-GO/INCOMPLETE 三分支。
7. 在正式运行前提交 exact implementation，执行 one-shot preformal audit/seal；正式后再做独立 post-outcome audit 与反简化审查。

## 实际完成内容

### 语义与功效修复

- 第三解析 reference 覆盖 mixture density/evidence/posterior、LLR、predictive moments/CDF、gain Jacobian、sample-count 与 sigma 语义。
- 独立 IQ diagnostic：20/20 gates，analysis=`7bd63905...`。
- empirical readout power：19/19 gates，27,648 windows，analysis=`f80ff63d...`。
- design power：22/22 gates，1,589-gate materialized blueprint，analysis=`c073b65d...`；冻结 768/256 样本量。
- 历史 NO-GO lineage receipt：analysis=`afe4ab09...`；旧 cell/source/archive 未被 fresh 模块读取。

### Runner、verifier 与 seal

- runner 固定 592 chunks、528,384 rows、92 fields，支持原子 CSV+NPZ chunk、append-only SHA-256 attempt chain、exact resume、heartbeat、final ZIP 和完整 denominator。
- raw archive 保存全部 IQ/heldout IQ；需要的 shared/probe/terminal density 以 complex64 保存，同时记录 actual Frobenius error、可独立重算的 IEEE-754 certificate 与 trace-distance UCB。
- fault 初态使用六态 balanced cycle `43/43/43/43/42/42`，避免 vacuum shortcut 令 logical survival 永久为空。
- verifier 采用 4-chunk LRU 惰性读取；独立重算 1,589 gates、IUT/TOST、RB reset、PTM、fault tail 与 cutoff 12→16/16→20。
- preformal seal 将 config、runner、verifier、reference、11 个直接/传递 runtime dependency、测试和 evidence bytes 绑定到 exact commit。

### 首次 fail-closed 与修复

第一次 seal 在 runner 读取 formal seed/cell 前失败：

`RuntimeError: preformal seal omits a required runner input`

根因是 `build_snapshot` 已统计 11 个 runtime dependency，但 `build_audit` 又使用 direct-only source assembler，持久化 seal 丢失这些 binding。修复不是放宽 runner，而是：

- 用单一 `_all_source_bindings()` 同时服务 snapshot 与 persisted audit；
- 增加“runtime dependency 不可从 persisted audit 丢失”的回归测试；
- 原 seal/audit/traceback 全部归档；
- 重新提交 exact implementation，重新执行 one-shot seal；
- 用真实 runner `verify_preformal_seal` 验证 11/11 binding 后才重新启动。

失败发生在 `_resume_state` 前：0 formal seed、0 cell、0 scientific artifact，verdict 保持 `null`。

### 正式事务

- run ID：`f8f2ef750245debb74e771c3576514a5ba460ac1808c44223493d7b8b39ecfd8`；
- 592/592 chunks、528,384/528,384 rows、0 exception；
- attempt chain：1 `RUN_STARTED` + 592 `CHUNK_COMMITTED` + 1 `FINALIZED`；
- CSV：707,688,383 B，SHA-256=`d875f122...`；
- ZIP：824,505,231 B，SHA-256=`b519bd4b...`；
- ZIP 594 个唯一成员、CRC PASS，无路径逃逸、重复或未登记成员；
- finalized 重入实测 35.6 s，attempt/manifest/CSV/ZIP/heartbeat 的 hash 与 mtime 均不变；
- runner 结束时 scientific verdict、qualified claim 和 15 个 claim 字段仍全部为 JSON `null`。

## 正式结果

独立 verifier 得到：

- 总门：1,589；
- 通过：1,562；
- 失败：27；
- verdict：`NO_GO_FRESH_TWIN_QUALIFICATION`；
- qualified claim：`null`；
- 六个 downstream release：全部 `false`；
- 历史 T9.2.4 NO-GO：保持。

失败分布：

| 失败族 | 数量 | 结论 |
| --- | ---: | --- |
| A/B physical density | 9 | 3 个 vacuum-f RESET + 6 个 fault terminal density；point 很小但当前高维 Frobenius→trace SE 极保守，未证明等价 |
| fault scalar | 10 | 8 个 mean-photon + 2 个 level-L1；总体 point 低于 margin，但六态分层显示差异可达 0.1—0.44，存在抵消与潜在真实 backend 差异 |
| cutoff density | 8 | 12→16 四场景 point=`0.1288—0.2000 > 0.1`；16→20 分态后 step/telegraph/burst 最大=`0.1226/0.1168/0.1070`，属于真实 state-conditioned 截断未收敛 |

IQ conditional、likelihood/score/posterior、reset/leakage、logical PTM/survival 等其余 1,562 门全部通过；这不能补偿任一失败门。

## 验证方式和结果

- preformal：70/70 gates、90/90 mutations；
- fresh 八模块全集：175 passed；
- runtime-binding 修复后 runner/verifier/preformal：110 passed；
- verifier 专项：34 passed；
- post-outcome audit：40/40 gates、69/69 mutations、14 tests；
- post-outcome 与任务板治理联合回归：24 passed（仓库内隔离 `basetemp`）；
- attempt chain 全事件 self-hash/previous-hash/index 独立重算通过；
- CSV 流式计数、row_id 唯一性、exception 空值通过；
- ZIP CRC、manifest、member/source SHA/size、mapping binding 通过；
- 528,384 行 semantic recomputation 误差约 `1e-14`；
- density quantization trace bound 约 `1.5e-7—2.0e-7/row`，不是失败原因。

## 反简化审计

本任务没有通过下列方式“救援”结果：

- 不放宽 margin；
- 不删除 failed cell/state/scenario；
- 不用旧 formal outcome 选择阈值；
- 不事后增加同一 formal 的样本量；
- 不把 raw log-evidence 或 mixed-unit max 当主指标；
- 不用 1,562 个通过门补偿 27 个失败门；
- 不以 diagnostic delete-one trace-norm jackknife 翻转已封印 verdict；
- 不把完整物理 density 未收敛降格成“logical survival 已收敛”；
- 不释放任何 downstream task 或论文 performance claim。

审计发现两个后续治理缺口：

1. 当前 density SE 是数学有效但未经 coverage calibration 的高维保守界，功效很低；
2. fault scalar 先跨六态聚合再取绝对值/norm，可发生正负抵消；
3. verifier/design-power/readout-power/lineage 的 module CLI 忽略 `sys.argv`，`--help` 会误入写路径；不改变当前 NO-GO，但必须在下一 seal 前修复；
4. production 没有真实走过“部分 chunk 中断后 resume”，该性质目前由代码、mutation 和 finalized 重入支持，不可写成 production-resume 实测。

## 产物路径

- `physics/phase9_iq_likelihood_reference.py`
- `cnn_fpga/benchmark/phase9_fresh_twin_lineage.py`
- `cnn_fpga/benchmark/phase9_iq_semantics_diagnostic.py`
- `cnn_fpga/benchmark/phase9_fresh_twin_readout_power.py`
- `cnn_fpga/benchmark/phase9_fresh_twin_design_power.py`
- `cnn_fpga/benchmark/phase9_fresh_twin_qualification.py`
- `cnn_fpga/benchmark/phase9_fresh_twin_verifier.py`
- `cnn_fpga/benchmark/phase9_fresh_twin_preformal_audit.py`
- `cnn_fpga/benchmark/phase9_fresh_twin_post_outcome_audit.py`
- `configs/phase9/t_risk_20260726_01_design_power.json`
- `configs/phase9/t_risk_20260726_01_fresh_twin_qualification.json`
- `configs/phase9/t_risk_20260726_01_fresh_release_pin.json`
- `docs/t_risk_20260726_01_*`
- `tests/test_phase9_iq_likelihood_reference.py`
- `tests/test_phase9_fresh_twin_*.py`

## Claim 影响

本任务只证明 fresh formal 事务完整，并给出科学 NO-GO。它不证明 twin qualified、frontend performance、LER、lifetime、physical break-even、真实板 latency/resource/power、official Puviani、Puviani surpass、external SOTA 或 rank。15 个字段继续为 literal `null`。

## 风险复核

- `R-N182`：从 Open 改为 Mitigated/Monitor；旧 IQ/likelihood 语义与伪重复已修复，但 fresh twin 仍未通过。
- 新增 `R-N184`（Critical / Immediate）：高 cutoff 物理尾未收敛，且六态聚合会隐藏 state-conditioned 差异。
- 新增 `R-N185`（High / Immediate）：trace-norm UCB 未做 coverage calibration，当前界功效过低；不得事后替换当前 formal estimator。
- 新增 `R-N186`（Medium / Soon）：四个 module CLI 忽略参数并可能在 `--help` 时写 artifact。

## 是否需要插入新 task

需要。插入 `T-RISK-20260727-01`，永久保留本次 NO-GO，只把当前数据作为 design-only pilot：

- 先修复 module CLI 与 finalized mapping 快速路径；
- 逐六态、逐阶段定位 backend density/energy/logical-block 差异；
- cutoff 扩到 `16/20/24/28`，必要时 `32`；
- fault density/energy/level/cutoff 全部按 `scenario × logical state × backend` 做 IUT/max；
- 预先校准 paired-cluster bootstrap/multiplier-bootstrap trace-norm UCB 的 coverage 与 power；
- 使用完全不重叠的新 seed、新 seal、新 release。

只有新 task 的独立 formal PASS 才可重新考虑六个 blocked downstream；本任务 verdict 永久不可修改。

## 任务板同步

`docs/new_task_board.md` 将本任务标为 Done（NO-GO），保持 T9.2.5/T9.2.7/T9.3.1/T9.3.4/T9.6.2/T9.6.5 为 Blocked，并将 `T-RISK-20260727-01` 设为下一顺序 In Progress。
