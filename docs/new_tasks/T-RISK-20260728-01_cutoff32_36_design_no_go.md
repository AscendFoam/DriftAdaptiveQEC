# T-RISK-20260728-01：cutoff32/36 design extension 科学 NO-GO

- **日期**：2026-07-28
- **状态**：Blocked（design 阶段完成并得到独立确认的科学 NO-GO；powered formal 未释放）
- **来源风险**：R-N184、R-N187、R-N188
- **后续阻塞解除任务**：T-RISK-20260728-03

## 输入材料

- cutoff28 sealed reference：`docs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3_manifest.json`
- immutable cutoff32/36 parent、released child 与 release receipt；
- density UQ analysis `dc2ab86f...`；
- scalar UQ report `de82db4d...`、independent verification `9dfe4506...`、factor `1.0`；
- 两个 byte-pinned backend、Mehler logical bridge 与 cutoff36 runtime validation-cap adapter；
- 同 seed 的六态、四 scenario、A/B、12 rounds、72 trajectories/cell design 合同。

## 实际完成内容

1. 资源预检先于科学 chunk 完成并通过：
   - 22 个 resource benchmark identity；
   - estimated wall `1962 s`；
   - estimated RSS `878,723,072 bytes`；
   - estimated artifact `259,877,621 bytes`。
2. fresh2 raw design 事务完整完成：
   - 22 cells、44 chunks、14,256 unique rows；
   - 1,584 retained densities；
   - 0 exception、0 conservation failure；
   - cutoff28 reference 合并后诊断分母为 30 receipts、21,168 rows、2,160 densities。
3. V1 diagnostic 正确 fail closed：
   - 先暴露 preregistered `analysis_sha256` binding 与 V1 reader schema 不兼容；
   - V1 source、launch meta、stdout/stderr 与 traceback 全部只读保留；
   - 未改写 V1 或 raw transaction。
4. V2 只修复 live reader，不改变 1,454 gates、margin、seed 或样本设计：
   - 校验 byte hash、自哈希与 semantic hash；
   - 消费完整八字段 production NPZ，并逐行核对 CSV/NPZ/IQ/density index；
   - 对 shared RESET 空白 `logical_survival` 使用各 backend 原生 logical isometry 从 archived density 推导；
   - 在使用该路径前，对 1,728 个 fault 终态逐一对拍 double-precision CSV，最大差 `1.3575e-08`，小于最大允许 `1.2728e-06`；
   - 对 logical/tail density-derived observable 传播 trace-distance quantization certificate。
5. V2 verified bootstrap 仅允许 `diagnostic`：
   - 不能重跑 design；
   - 不能启动 powered formal；
   - repair child 明确 `gates_or_margins_changed=false`、`design_outcomes_used_to_change_contract=false`。
6. 原子发布 diagnostic report、Source Data 与 completion，科学结果为：
   - `1,393 / 1,454` pass；
   - `61 / 1,454` fail；
   - maximum margin ratio `12.6189370592`；
   - verdict=`NO_GO_HIGH_CUTOFF_DESIGN`；
   - authorization=`POWERED_FORMAL_REMAINS_UNRELEASED`。
7. 独立 verifier 不导入 V2 判决实现，重新核对：
   - 全部 30 receipt/CSV/NPZ triples；
   - 21,168 raw rows、2,160 density IDs、1,728 fault terminals、432 shared terminals；
   - 全部 1,454 conservative point、pass/fail、family ledger、failed IDs 与 claim firewall；
   - verdict=`VERIFIED_NO_GO_HIGH_CUTOFF_DESIGN`。
8. NO-GO 诊断把失败分为三类：
   - 56/61 只位于 `28->32`；`32->36` 的 596 个对应门全部通过；
   - cutoff36 的 step/A、`+` 与 `+i` 两个 commutator-tail 门仍失败；
   - shared RESET A/B level L1 在 28/32/36 均为 `0.138888...`，来自 59/72 与 54/72 的独立 Bernoulli branch；Rao–Blackwell success mean 差仅约 `0.00229`。

## 产物路径

- `cnn_fpga/benchmark/phase9_cutoff32_36_design_bootstrap_v2.py`
- `cnn_fpga/benchmark/phase9_cutoff32_36_design_diagnostic_v2.py`
- `cnn_fpga/benchmark/phase9_cutoff32_36_design_diagnostic_v2_verify.py`
- `cnn_fpga/benchmark/phase9_cutoff32_36_no_go_diagnosis.py`
- `configs/phase9/t_risk_20260728_01_cutoff32_36_design_diagnostic_v2_released.json`
- `docs/t_risk_20260728_01_cutoff32_36_design_extension_fresh2_manifest.json`
- `docs/t_risk_20260728_01_cutoff32_36_design_diagnostic_fresh2.json`
- `docs/t_risk_20260728_01_cutoff32_36_design_diagnostic_fresh2_source_data.csv`
- `docs/t_risk_20260728_01_cutoff32_36_design_diagnostic_fresh2_completion.json`
- `docs/t_risk_20260728_01_cutoff32_36_design_diagnostic_fresh2_verification.json`
- `docs/t_risk_20260728_01_cutoff32_36_no_go_diagnosis.json`

## 验证方式和结果

- focused V2/legacy design regression：`38 passed`；
- V2 + independent verifier + mutation + NO-GO diagnosis：`10 passed`；
- independent verifier analysis：`1f444046a0acef4335a3e0e88fdb91cdfa42eaef336d5ee56154c207206bea98`；
- NO-GO diagnosis analysis：`567696e14f869301c189a472b66e9e65f1ce390ddf2c8b7a0609150c1b4e4bcd`；
- V2 publication owner lock 已清除，无 partial artifact；
- git pre-launch commit `4194045` 已先同步远端，再按精确 bootstrap SHA 运行；
- 所有 LER、lifetime、physical、hardware measured、official Puviani、Puviani surpass、external SOTA、rank 与 twin qualification 字段保持 `null`。

## 反简化审计

- 没有把 V1 traceback 当成科学失败，也没有删除失败证据；
- 没有只检查 JSON summary：独立 verifier 实际读取全部 30 个 raw triples；
- 没有用 logical survival 稳定掩盖 density/energy/tail；
- 没有增 seed、放宽 margin、删掉 `28->32` 或两个 cutoff36 tail failure；
- 没有把 596 个 `32->36` passing gates 与旧通过项拼成 qualification；
- 没有把 stochastic RESET sampled branch 当成 backend 期望通道；
- powered formal 未启动。

## 风险复核与插入任务

- R-N184 继续为 Critical / Immediate：cutoff36 仍有两个 state-conditioned boundary-tail failure。
- R-N187 被本次 NO-GO 实证触发；不得在原 fresh2 run 追加 cell 或无限升 cutoff。
- 新增 R-N190：stochastic RESET branch 污染 qualification estimand。
- 插入 `T-RISK-20260728-03`：一次性 fresh `36/40/44`、两连续增量、cutoff44 absolute-tail、Rao–Blackwellized expected RESET density/levels、先资源预检、cutoff44 失败即 terminal NO-GO；禁止自动扩展到 48+。

## 对任务板的同步

- `T-RISK-20260728-01`：`In Progress -> Blocked (scientific NO-GO; waiting for T-RISK-20260728-03)`；
- 新增并启动 `T-RISK-20260728-03`；
- 当前推荐任务改为 `T-RISK-20260728-03`；
- T9.2.5、T9.2.7、T9.3.1、T9.3.4、T9.6.2、T9.6.5 继续 Blocked。
