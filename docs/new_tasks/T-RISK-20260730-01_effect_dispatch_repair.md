# T-RISK-20260730-01：T06 注册 effect 分派与 verifier 输入修复

- **日期**：2026-07-30
- **状态**：Done（analysis-only repair + full independent verification closure）
- **来源风险**：R-N195、R-N196、R-N197
- **父任务**：T-RISK-20260728-06
- **后续任务**：T-RISK-20260728-06 终态验收；通过后才可恢复 T-RISK-20260728-04

## 输入材料

- T06 V1 writer report、8,404-row Source Data 与 selected 3,043-gate blueprint；
- 8 个 selection chunks、24 个 confirmation chunks，共 7,168 个 paired-density trial；
- T05 的完整 3,043-gate union、joint-maxT 与功效合同；
- T06 冻结的有限 count grid、互斥 selection/confirmation seed namespaces、factor=`1.0`、`B=199`、higher quantile 和最小通过规则；
- V1 根失败归档、首个多线程重分析失败归档、第二个 verifier-input 失败归档。

## 执行方案

1. 永久保存 V1 失败事务，不把 implementation failure 改写为科学 NO-GO。
2. 不生成新随机数、不改 raw chunk、不改 count、seed、factor、margin、gate family、correlation、threshold 或候选选择。
3. 由完整 canonical `cell_id` 恢复注册 effect 类别；derived `true_distance` 继续保留原物理重算值，并在 `1e-12` 内验证与注册类别一致。
4. 对四个注册 effect `{0, 0.05, 0.1, 0.12}` 使用显式分支；未注册类别、混合类别、后缀伪造或笛卡尔积缺失均 fail-closed。
5. repair runner 在调用 writer 前逐文件核对 V1、32 个 raw chunks、7,168 raw rows、8,404-row CSV、3,043-gate blueprint、source/runtime seals，并强制 `reuse_only`。
6. Source Data 的最小差异白名单只允许六个 canonical `effect_0.050` confirmation summary 的 `power_gate_pass` 与 `gate_pass` 从 `False` 变为 `True`。
7. 独立 verifier 仍从自身冻结 config 重建 canonical `spec.effect` 作为 trial generator 输入；source-derived truth 只作为独立观测校验，不进入离散 design dispatch。
8. repair 只可给出 analysis-only verdict；最终 T04 preregistration release 只能由完整独立 verifier 给出。

## 实际完成内容

### 1. V1 根失败保存

- V1 已完成全部 selection/confirmation 数据并选择 scale=`2.0`；
- writer 仅 G10 失败，gate summary=`14/15`；
- 六个注册 effect=`0.05` cell 的 derived truth 分别出现
  `0.04999999999999993`、`0.04999999999999999`、
  `0.05000000000000002` 等 ULP 差异；
- 六个 cell 均为 `256/256` equivalence，simultaneous Wilson LCB=`0.9643106148`；
- 原实现以 derived float 的 exact equality 做类别分派，错误使用 outside-effect UCB 规则；
- tracked 根归档：
  `docs/archive/t_risk_20260728_06_v1_fp_effect_dispatch/`；
- local full archive：
  `runs/archive/t_risk_20260728_06_v1_fp_effect_dispatch_20260730/`，
  共38文件、7,548,008 bytes；
- 38/38 文件逐字节有效，canonical manifest SHA256=
  `b92a8fe4b3e0d589e3642d4dee315edad341c3b09267dbc11db5c6c20cf09b57`。

### 2. 首个 repair attempt fail-closed

- 首个已封印 repair 在 raw/chunk/contract 核对后执行 deterministic 重分析；
- 最小 diff 门发现707个 maxT replicate、1个 selection critical、2个 meta point 出现最多
  `3.5527e-15` 的字符串级变化，另有六行预期 effect 布尔变化；
- 该变化来自外部 BLAS/OMP 多线程归约，不是统计合同变化；
- repair report 尚未写出即 fail-closed，live V1 原样恢复；
- 失败事务保存在
  `docs/archive/t_risk_20260730_01_v1_multithread_reanalysis_failure/`；
- 单线程只读复算全部1,204项 maxT 与 V1 字符串逐项相同；
- runner/config 现强制在 Python/NumPy 启动前将
  `OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`OMP_NUM_THREADS`、
  `NUMEXPR_NUM_THREADS` 全部设为字符串 `"1"`，不以数值容差放行。

### 3. 第二个 repair 与 verifier-input attempt fail-closed

- 单线程 analysis-only repair 对同一 raw 通过，writer=`15/15`；
- 原独立 verifier 在写出任何 verification artifact、启动长重算前，
  以 `density trial design drift` fail-closed；
- 根因为 source truth 已按原 `1e-15` 通过 canonical 核对，但旧 worker 又把该
  source-derived ULP 值作为 exact categorical generator 输入；
- 六文件失败事务保存在
  `docs/archive/t_risk_20260730_01_v1_verifier_effect_input_failure/`；
- archive manifest self-hash=
  `75ce175508f997d798d5163d03cd63a2873d06bd3bac29f07213dba4d055d77c`；
- 失败时没有 verification artifact、没有长重算、没有新增 trial 或 scientific vote；
- live V1 再次原样恢复。

### 4. 最小 verifier 输入修复与最终 analysis-only repair

- selection/confirmation payload 显式携带由 verifier 自身 config 重建的 canonical
  `spec.effect`；
- `_density_worker` 只使用该 canonical effect 生成 trial；
- `_validate_density_rows` 继续用原 `1e-15` 校验 source truth；
- 没有导入 writer、physics、既有 verifier 或 paired-UQ helper；
- 最终 repair verdict=
  `PASS_ANALYSIS_ONLY_EFFECT_DISPATCH_REPAIR`；
- 32/32 live raw chunks 与 V1 逐字节一致；
- Source Data 保持8,404行、header不变，只有六个完整 canonical cell 的两个布尔字段，
  共12个 field diff；
- selected blueprint 与 V1 逐字节一致：
  1,290,746 bytes，SHA256=
  `9bd9ed23fb30737390bacc69f2730755add80257a3814475e12b7f13c7d1409b`；
- repair report self-hash=
  `d8f965fbf0eb8f38e519f40dfd84808cf03e52100bdece7fd61e7d3310c58ca0`；
- repair report 仍保持
  `t04_preregistration_final_release=false`、
  `t04_scientific_execution_released=false`；
- 完整独立 verifier 已对 selection 和 confirmation 全量重算并给出
  `PASS_INDEPENDENT_COUNT_SELECTION_AND_CONFIRMATION`；
- verification=`21/21`，analysis SHA256=
  `5a49967fddc86283a42ff75091ce099eed9d3ae799763a3d1e4cca40c7a19c2e`，
  file SHA256=
  `ffcb66d4ca204e8fdbe7c9e10a0e46d6676a0e7fc8eb9a72c0af53bf14b4513c`；
- 独立重算最大数值差=`1.942890293094024e-16`；
- repair report 自身仍保持 final release=false；T04 preregistration release
  只由独立 verifier 签发，责任边界没有被事后改写。

## 产物路径

- `cnn_fpga/benchmark/phase9_count_selection_confirmation.py`
- `cnn_fpga/benchmark/phase9_count_effect_dispatch_repair.py`
- `cnn_fpga/benchmark/phase9_count_selection_confirmation_verify.py`
- `configs/phase9/t_risk_20260728_06_count_selection_confirmation.json`
- `tests/test_phase9_count_selection_confirmation.py`
- `tests/test_phase9_count_effect_dispatch_repair.py`
- `tests/test_phase9_count_selection_confirmation_verify.py`
- `docs/t_risk_20260730_01_effect_dispatch_repair.json`
- `docs/archive/t_risk_20260728_06_v1_fp_effect_dispatch/`
- `docs/archive/t_risk_20260730_01_v1_multithread_reanalysis_failure/`
- `docs/archive/t_risk_20260730_01_v1_verifier_effect_input_failure/`

## 验证方式和当前结果

- 三个 archive manifest 自哈希均通过；
- V1 local full archive：38/38 files、7,548,008 bytes、canonical digest通过；
- live raw chunks：32/32 与 V1 bytes/SHA 一致；
- V1→V2 Source Data：8,404/8,404 rows，只有6行×2布尔字段变化；
- 当前 repair report self-hash有效，analysis-only verdict与claim边界一致；
- focused writer/repair/verifier tests：终态单线程重跑
  `78 passed in 19.24s`；
- py_compile：writer、repair、verifier 及三份测试源码全部通过；
- 独立 live audit：144项自哈希、binding、分母、CSV diff、raw archive、
  blueprint、claim-null 和失败 lineage 检查全部通过；
- stdout JSON 与 atomic verification report 语义完全相同，stderr=`0 bytes`；
- verifier 终态前后没有重复 supervisor/child，进程正常退出。

## 反简化审计

- 没有把 exact-float 缺陷通过扩大统计容差掩盖；
- 没有把 derived truth 归一化覆盖原始物理值；
- 没有只检查 summary：repair runner 绑定 raw chunks、trial address、CSV、blueprint、source seal；
- 没有重抽样、追加 count、重选候选、删门或改变 maxT 相关结构；
- 没有将第一次多线程产生的 `<=3.55e-15` maxT 漂移加入白名单；
- verifier worker 输入与 source truth 分离，并有 ULP 接受、越界拒绝、另一注册 effect
  拒绝、配置 mutation、未注册 effect、worker independence 等负测试；
- repair PASS 不等于 T06 science PASS，更不等于 twin/LER/lifetime/hardware/Puviani/SOTA 证据。

## 风险复核与插入任务

- R-N195：最小 categorical repair、六行白名单与全量独立重算已闭合，降为
  Mitigated / Monitor；
- R-N196：单线程 runtime seal、maxT逐字符串0差与终态 focused regression
  已闭合，降为 Mitigated / Monitor；
- R-N197：verifier generator 输入与 source observation 分离，完整7,168行重算
  已通过，降为 Mitigated / Monitor；
- 当前不再插入新 task；若 verifier fail-closed，必须保存新失败事务、诊断后 fresh
  analysis-only 修复，不得静默覆盖或生成新 scientific raw。

## 对任务板的同步

- `T-RISK-20260730-01 = Done`；
- `T-RISK-20260728-06 = Done (Independent PASS)`；
- `T-RISK-20260728-04 = In Progress`，只释放其 preregistration/implementation；
- 所有 twin/LER/lifetime/physical/hardware/official/Puviani/SOTA/rank 字段继续保持 `null`。
