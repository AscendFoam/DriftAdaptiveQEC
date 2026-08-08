# T-RISK-20260728-03：cutoff36/40/44 有界物理收敛修复

- **日期**：2026-07-28
- **状态**：Done（design-repair PASS；只允许另行预注册 powered formal）
- **来源风险**：R-N184、R-N187、R-N190
- **后续任务**：T-RISK-20260728-04

## 输入材料

- T-RISK-20260728-01 的 cutoff32/36 科学 NO-GO、raw manifest、V2 diagnostic、独立 verifier 与失败根因诊断；
- T-RISK-20260728-02 的 scalar UQ report `de82db4d...`、independent verification `9dfe4506...` 与冻结 factor `1.0`；
- 两个 byte-pinned physics backend、Mehler logical bridge、cutoff44 runtime-cap adapter；
- 同 seed 六态、四 scenario、A/B、12 rounds、72 trajectories/cell 的 design-only 合同；
- 预注册的 `36→40`、`40→44` 连续增量、cutoff44 absolute-tail、旧 raw 不投票和禁止自动扩展到 48+ 的终止规则。

## 执行方案

1. 在访问任何 cutoff40/44 科学结果前，先运行资源、能力、runtime-cap 与 RESET Rao–Blackwell sidecar 预检。
2. 用全新 immutable run ID 同时生成 36/40/44，不向旧 run 追加 cell，也不复用旧通过门。
3. fault 主层保留实际 sampled trajectory；shared RESET 主 estimand 改为同一 pre-reset trajectory 上 success/failure 两个原生分支按原生概率混合的 expected post-reset density/levels。
4. sampled RESET branch 仅保存在 sidecar 中作非投票压力审计；逐样本核对 mixture、forced branch、trace、Hermiticity 与 branch separation。
5. diagnostic 只消费完整 manifest-addressed raw；独立 verifier 不导入 writer/scientific evaluator，逐行重算全部 gate 和 raw observable。
6. cutoff44 任一门失败即 terminal NO-GO；全部通过也只释放“另行预注册 powered formal”，不直接恢复任何性能下游。

## 实际完成内容

1. resource/capability preflight 完成并通过，随后 production transaction 完整结束：
   - 30/30 cells，其中 24 fault cells、6 shared RESET cells；
   - 66 primary chunk files、30 receipts；
   - 21,168 raw rows、2,160 primary retained densities；
   - 6 个 RESET sidecar、1,296 conditional branch densities；
   - 0 exception、0 conservation failure，owner lock 正常清除。
2. RESET estimand 修复不是概率手填：
   - 每条 primary expected density 都由归档的 success/failure 原生分支与原生概率混合；
   - 432 个 sidecar row 的 sampled action 与所选 forced branch bit-for-bit 一致；
   - primary archive 不含 sampled outcome/success/failure/probability 数组，避免 verifier 误用非投票 branch；
   - branch-swap/weight/selected-branch、trace-preserving与非退化 branch-distance测试均通过。
3. diagnostic 对 1,454 个唯一门全部通过：
   - fault absolute-tail `240/240`；
   - fault density `96/96`；
   - fault scalar `1,080/1,080`；
   - shared absolute-tail `10/10`；
   - shared density `7/7`；
   - shared scalar `21/21`；
   - maximum margin ratio `0.4721119736`；
   - verdict=`DESIGN_REPAIR_PASS_MAY_PREREGISTER_SEPARATE_POWERED_FORMAL`。
4. 独立 verifier 重读全部 raw/receipt：
   - 21,168 rows、2,160 densities；
   - 1,728 个 fault terminal cross-check；
   - 432 个 shared expected-density derivation；
   - gate recomputation 最大绝对差 `0`；
   - fault double-precision CSV 对 archived density 最大绝对差 `1.3766e-08`；
   - verdict=`VERIFIED_DESIGN_REPAIR_PASS_MAY_PREREGISTER_SEPARATE_POWERED_FORMAL`。
5. 证据边界保持：
   - `old_raw_or_gate_composition=false`；
   - `automatic_cutoff_extension_beyond_44=false`；
   - `powered_formal_released=false`；
   - `qualified_claim=null`；
   - LER、lifetime、physical break-even、hardware measured、official Puviani exact、Puviani NMF surpass、external SOTA 与 twin qualification 全部保持 `null`。

## 产物路径

- `physics/phase9_cutoff44_runtime_adapter.py`
- `physics/phase9_reset_rao_blackwell.py`
- `cnn_fpga/benchmark/phase9_cutoff36_44_repair_raw.py`
- `cnn_fpga/benchmark/phase9_cutoff36_44_repair_diagnostic.py`
- `cnn_fpga/benchmark/phase9_cutoff36_44_repair_verify.py`
- `configs/phase9/t_risk_20260728_03_cutoff36_44_repair.json`
- `runs/t_risk_20260728_03_cutoff36_44_repair_fresh1/`
- `docs/t_risk_20260728_03_cutoff36_44_repair_fresh1_manifest.json`
- `docs/t_risk_20260728_03_cutoff36_44_repair_fresh1_diagnostic.json`
- `docs/t_risk_20260728_03_cutoff36_44_repair_fresh1_source_data.csv`
- `docs/t_risk_20260728_03_cutoff36_44_repair_fresh1_verification.json`

## 验证方式和结果

- focused repair + RESET Rao–Blackwell：`14 passed`；
- cutoff44 runtime adapter：`4 passed`；
- cutoff32/36 parent regression：`19 passed`；
- V1/V2 parent diagnostic regression（隔离 basetemp）：`12 passed`、`10 passed`；
- scalar UQ preflight：`15 passed`；
- density UQ preflight：`22 passed`；
- high-cutoff pilot/diagnostic：`32 passed`、`15 passed`；
- bootstrap V2：`7 passed`；
- 合计 `150 passed`。

一次把 V1/V2 pytest 放在共享嵌套 `--basetemp` 下时，两个 test module 互删其父临时目录，出现 `20 passed, 2 setup errors`。这不是科学或实现失败；随后使用两个独立顶层 basetemp 分进程重跑，分别 `12 passed` 与 `10 passed`。该基础设施交互没有被隐藏，也未通过放宽测试解决。

## 反简化审计

- 不是只检查 summary：逐一校验 30 个 receipt 的 CSV/NPZ hash/bytes、6 个 sidecar hash/bytes，并独立重算全部 raw gate。
- 不是把 sampled RESET branch 重新包装成 expected channel：432 行 primary mixture 最大 Frobenius 误差 `1.0294e-08`，sampled-vs-selected forced branch 最大误差 `0`，conditional branches 的 trace distance 约为 `1`，证明两分支非退化。
- 没有复用 `28→32`、`32→36` 旧 gate/raw，没有让旧 passing item 投票，也没有扩展到 cutoff48+。
- 没有只看 logical survival；density、energy、commutator、photon/tail 与 state-conditioned observable 全部保留。
- 没有通过增 seed、改 margin、换 factor、删 state/scenario 或结果驱动停止取得 PASS。
- 全部 206,543,751 bytes 的 production run 连同 manifest-addressed raw 纳入仓库；NPZ 走 Git LFS，CSV/JSON/receipt/log 保持逐字节归档。

## 风险复核与插入任务

- R-N184、R-N187、R-N190 在冻结的 design domain 内降为 `Mitigated / Monitor`；这不等于 powered twin qualification。
- 新增 R-N191：若把 72-trajectory/cell 的无功效 design point gates 写成正式 twin PASS，会形成新的错误论文 claim。
- 插入 `T-RISK-20260728-04`：建立独立 fresh powered formal，原样消费 factor `1.0`、384 paired clusters/state、六态/四 scenario/A-B full denominator 与 cutoff36/40/44 修复域；不得复用本任务的 72 trajectories 或结果来改门。

## 对任务板的同步

- `T-RISK-20260728-03`：`In Progress -> Done (Design Repair PASS)`；
- 新增并启动 `T-RISK-20260728-04`；
- 当前推荐任务改为 `T-RISK-20260728-04`；
- T9.2.5、T9.2.7、T9.3.1、T9.3.4、T9.6.2、T9.6.5 继续 `Blocked`，只有 T-RISK-20260728-04 fresh powered PASS 才能解除；
- 所有 official/Puviani/SOTA/physical/hardware/LER/lifetime claim 字段继续保持 `null`。
