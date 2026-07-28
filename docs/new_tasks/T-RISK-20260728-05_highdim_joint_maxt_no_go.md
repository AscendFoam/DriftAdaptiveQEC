# T-RISK-20260728-05：高维 density UQ 与完整 gate-union joint-maxT 预检

- **日期**：2026-07-28
- **状态**：Done（independently verified statistical NO-GO）
- **来源风险**：R-N185、R-N188、R-N192、R-N193
- **后续任务**：T-RISK-20260728-06

## 输入材料

- T-RISK-20260726-01 的旧七族 design-power 合同；
- T-RISK-20260728-03 的 cutoff36/40/44 bounded-repair 合同、1,454 个修复 estimand 与资源实测；
- 已冻结的 paired-density multiplier UCB：factor=`1.0`、`B=199`、higher quantile；
- T04 候选完整矩阵：210 shared、252 logical、24 fault、32 probe，共518 chunks；
- 旧方案样本量：round cell=`768`，fault=`384/state × 6 states`。

## 执行方案

1. 在不存在且不访问任何 T04 formal outcome 的条件下封存 config、源码和测试；
2. 从两个父合同逐项展开 gate，而不是手写估算门数；
3. 在 d120/d132、384 paired clusters/state 上运行三种物理 PSD synthetic family 的 null/local/boundary/outside split；
4. 每个 cell 运行256个 trial，每个 trial 真实构造384对复密度矩阵并执行199次 Rademacher multiplier trace-norm UCB；
5. 对完整随机 gate union 使用同 cluster namespace 的相关、studentized joint maxT；确定性 mapping 单列 exact gate；
6. 运行4,000个全局 IUT 功效 pseudoexperiment，分别检查 null、单门0.5-margin local、boundary和1.25-margin outside；
7. 用与 writer 不共享项目实现的 verifier，从 seed 重建全部密度矩阵、UCB、blueprint、maxT和power；
8. 任一 coverage、local/boundary power、资源或独立验证门失败，即保持 T04 blocked。

## 实际完成内容

### 1. 完整 blueprint

- 精确 gate 数：`3,043`，不是审计阶段的手工近似数；
- 旧七族：`1,589`；
- T03 state-conditioned repair：`1,454`；
- 随机门：`3,037`；
- 确定性 mapping 门：`6`；
- 所有 gate ID 唯一，完整保留六态、stage、scenario、backend 和 cutoff scope；
- cluster count 只允许 `{384, 768, 2304}`，确定性门为0。

### 2. 高维 paired-density UQ

- `24 cells × 256 trials = 6,144` raw rows；
- dimensions=`{120,132}`，每个 trial=`384` paired clusters；
- trial seed 与 multiplier seed 都使用无取模单射地址，6,144/6,144 唯一；
- 所有24个 cell 的 coverage=`1.0`，simultaneous Wilson LCB=`0.964311`，说明 factor=1.0 没有欠覆盖；
- null equivalence 最低 Wilson LCB=`0.957120`；
- boundary/outside 最大 equivalence UCB=`0.035689`，方向正确；
- 但 `heteroskedastic_coherent` 在 d120/d132、true distance=0.05 时均为 `0/256` equivalence pass，local-power gate 明确失败。

这说明问题不是 coverage 失真，而是高维非光滑 trace norm 的 bias/radius 在384 clusters下没有足够 local power。

### 3. 完整 union joint maxT

- `B=199`、higher quantile、factor=`1.0`；
- closed stochastic family=`3,037` gates；
- joint critical=`4.1147516185`，显著高于 pointwise `1.6448536269`，因此没有用逐门 z 值冒充 joint maxT；
- null 全局等价：`1,732/4,000 = 0.433`，Wilson LCB=`0.417716 < 0.90`；
- 13个 family 的0.5-margin local 最低 point=`0.08875`、最低 LCB=`0.080327`，均未满足预注册0.60；
- boundary 最大 false-equivalence UCB=`0.001415`；
- outside 最大 false-equivalence UCB=`0.000959`；
- 因而检测非等价的方向正确，但旧样本量无法在完整门并集上以所需功效证明等价。

### 4. 资源预测

- 旧样本量下完整 T04：518 chunks、1,042,944 primary rows；
- 4 workers × 1 BLAS thread 的保守 wall 预测：`230,440.10 s`（约2.67天）；
- artifact safety estimate：`58,513,386,935 bytes`；
- total RSS estimate：`8,564,244,480 bytes`；
- 当前磁盘余量满足本预检上限；
- 该预测只允许设计判断，T04 scientific chunk 前仍必须用 fresh shared/logical/probe/fault/c44-B/RB-compression seeds 做真实资源门。

## 独立验证

独立 verifier：

- 不导入 writer、physics backend、paired-cluster UQ helper 或既有 evaluator；
- 重建 `6,144/6,144` density rows；
- 重建 `3,043/3,043` gates；
- 重建 `199/199` maxT replicates；
- 重建 `40/40` maxT power cases；
- 最大 raw 数值差=`2.4980e-16`；
- 验证门=`20/20`；
- verdict=`PASS_INDEPENDENT_T04_STATISTICAL_NO_GO_VERIFICATION`；
- `t04_preregistration_released=false`；
- verifier source SHA256=`3cf76166dedcabec054c540cbe0cd91fe6adc0ef5f2420920f0b9c4504dfd427`；
- verification analysis SHA256=`2141b9cddaf89fe946edc777e9ee8371a03acfb08a68ba5bf3eba270ea189f69`。

专项与父回归：

- T05 contract/blueprint/density/maxT/mutation：`28 passed`；
- paired-cluster UQ + fresh design power：`25 passed`；
- cutoff36/44 repair：`8 passed`；
- 合计：`61 passed`。

一次四文件组合回归在120秒上限后以51个进度点、0 failure输出被超时终止；随后按模块用独立300秒预算重跑，得到上述61项全绿，不把超时记为测试通过。

## 产物路径

- `configs/phase9/t_risk_20260728_05_highdim_joint_maxt_preflight.json`
- `cnn_fpga/benchmark/phase9_highdim_joint_maxt_preflight.py`
- `cnn_fpga/benchmark/phase9_highdim_joint_maxt_preflight_verify.py`
- `tests/test_phase9_highdim_joint_maxt_preflight.py`
- `docs/t_risk_20260728_05_full_gate_blueprint.json`
- `docs/t_risk_20260728_05_highdim_joint_maxt_source_data.csv`
- `docs/t_risk_20260728_05_highdim_joint_maxt_preflight.json`
- `docs/t_risk_20260728_05_highdim_joint_maxt_verification.json`
- `runs/t_risk_20260728_05_highdim_joint_maxt_preflight/`

关键文件 SHA256：

- config：`d0a6e575fddbc94fc53adaafa8bc54f17496ae4b8861acf6b886d8c9723332b5`
- writer report：`00dd9953d8b89897cbdc1303223b1f867e82c2c51076cacc2636c2239891abd0`
- Source Data：`f67bf42c54a8759d3b31fa0bfb8575266fa6f7185430c9ac8aa34b31c7971f3f`
- blueprint：`c9047035bbaa65ba7b99fe5ba5ee23a803af562efde9f8c46090e428a4f2d0ae`
- independent verification：`7f0dc94dff4e258c36e90b9650a3a5fb18e30bb78f0782843f39c149dc18d14a`

## 反简化审计

- 没有把完整门数写成约数：两个实现独立得到3,043；
- 没有只检查 writer summary：verifier从seed重建全部6,144个密度 trial；
- 没有把高维 density 降成标量 Gaussian demo：每行都执行 PSD复密度构造、特征值 trace norm和199次 multiplier；
- 没有把 pointwise TOST 写成 simultaneous：联合临界值4.11475由完整3,037门相关 Rademacher 最大统计量产生；
- 没有删掉失败的 heteroskedastic family、state-conditioned repair gate 或困难 tail family；
- 没有用 aggregate 平均、跨态抵消、margin放宽、factor增加、B减少、quantile替换或结果驱动 seed 来救援；
- 没有把“boundary/outside检测能力强”偷换成“null/local等价功效足够”；
- 所有 official/Puviani/SOTA/LER/lifetime/physical/hardware/twin 字段保持 `null`。

## 风险复核与插入任务

- R-N185：d120/d132 coverage已建立，但384/state的heteroskedastic local power失败，保持Open；
- R-N192：完整joint-maxT算法与3,043门blueprint已建立，但旧样本量全局功效失败，保持Open；
- R-N193：旧样本量资源预测可行，但任何增加样本的方案必须重新做artifact/wall/RSS门；
- 新增R-N194：若在看到NO-GO后直接把样本翻倍、只保留已知通过计数或复用同一seed确认，会形成新的design-selection偏差。

插入 `T-RISK-20260728-06`：把本任务仅作为design pilot，先冻结有限样本候选与单射seed，再用独立 selection split 选择最小可行计数，并用 untouched confirmation split 对完整高维density和3,043门joint-maxT重做资格。候选耗尽即terminal NO-GO，不形成无限加样本阶梯。

## 对任务板的同步

- `T-RISK-20260728-05`：`In Progress -> Done (Verified Statistical NO-GO)`；
- `T-RISK-20260728-04`：继续Blocked，不得按1,042,944-row旧方案启动；
- 新增并启动 `T-RISK-20260728-06`；
- 六个downstream继续Blocked；
- 全部 performance/physical/hardware/official/Puviani/SOTA/rank claim保持null。
